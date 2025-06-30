#!/usr/bin/env python3
# fast_make_paths_parallel.py  –  parish-aware, resumable path builder
# --------------------------------------------------------------------------
#  • Keeps the original cache layout (paths_npz/*.npz + connection_index.csv)
#  • Iterates CASA origins parish-by-parish instead of one giant list
#  • Still fan-outs ThreadPoolExecutor calls per origin
#  • Safe to ^C and restart; already-done origins are skipped
# --------------------------------------------------------------------------

import os, time, multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib                       import Path

import numpy as np
import pandas as pd
import geopandas as gpd
from tqdm import tqdm

from src_cpp.path_planner import path_from_poi          # C++ A*
from sample_pois_old     import pois_within_radius      # <- if you still need it

# ------------------------------- config -----------------------------------
GRID_PATH     = "cpp_data/grid.npy"
POI_PATH      = "pois/catastaci_adjusted.geojson"
OUTPUT_DIR    = "paths_npz"                # .npz path files + index

RADIUS_M      = 160
TOP_N         = 2                          # best N per meta-category
MAX_DIST      = -1                         # no cut-off inside A*
PREF_PARISH   = False                      # sort candidates by same parish first
INDEX_FILE    = Path(OUTPUT_DIR) / "connection_index.csv"

# --------------------------- helpers: cache -------------------------------
def load_index() -> pd.DataFrame:
    if INDEX_FILE.exists():
        return pd.read_csv(INDEX_FILE)
    return pd.DataFrame(columns=[
        "origin_uid", "origin_type",
        "target_uid", "target_type",
        "path_len",   "path_file",
    ])

def add_rows_to_index(df_index: pd.DataFrame, new_rows: list[dict]) -> pd.DataFrame:
    df_index = pd.concat([df_index, pd.DataFrame(new_rows)], ignore_index=True)
    INDEX_FILE.parent.mkdir(parents=True, exist_ok=True)
    df_index.to_csv(INDEX_FILE, index=False)
    return df_index

# --------------------------- helpers: A* wrapper --------------------------
def compute_path(sr, sc, trg_r, trg_c):
    found = path_from_poi(
        GRID, sr, sc, trg_r, trg_c,
        max_distance=MAX_DIST, diagonals=True
    )
    if found is None:
        return None
    rows, cols = found
    return len(rows), rows, cols

# --------------------------------------------------------------------------
def main() -> None:
    global GRID   # shared read-only inside thread workers
    Path(OUTPUT_DIR).mkdir(exist_ok=True)

    GRID    = np.load(GRID_PATH)                # uint8 grid
    poi_gdf = gpd.read_file(POI_PATH)

    # ensure metres CRS
    if not poi_gdf.crs or not poi_gdf.crs.axis_info[0].unit_name.lower().startswith("metre"):
        poi_gdf = poi_gdf.to_crs(poi_gdf.estimate_utm_crs())

    # load cache
    df_index      = load_index()
    done_uids     = set(df_index["origin_uid"].unique())

    # CASA origins grouped by parish (None→'UNKNOWN')
    is_casa = poi_gdf["PP_Function_TOP"] == "CASA"
    poi_gdf["parish_std"].fillna("UNKNOWN", inplace=True)

    parishes = sorted(poi_gdf.loc[is_casa, "parish_std"].unique())
    total_todo = poi_gdf[is_casa & ~poi_gdf["uid"].isin(done_uids)].shape[0]

    sindex   = poi_gdf.sindex                 # spatial index for fast buffer query
    n_threads = mp.cpu_count()
    executor  = ThreadPoolExecutor(max_workers=n_threads)

    overall_pbar = tqdm(total=total_todo, desc="CASA (all parishes)")

    start = time.time()

    for parish in parishes:
        parish_mask = (is_casa &
                       (poi_gdf["parish_std"] == parish) &
                       ~poi_gdf["uid"].isin(done_uids))

        parish_idx = poi_gdf.index[parish_mask].to_numpy()
        if len(parish_idx) == 0:
            continue

            # >>>>>>>>>> NEW LINE — banner <<<<<<<<<<
        print(f"\n→ Working on parish: {parish or 'UNKNOWN'} "
          f"({len(parish_idx):,} CASA origins)")
        # ---------------------------------------

        parish_pbar = tqdm(total=len(parish_idx),
                           desc=f"{parish or 'UNKNOWN'} CASA")

        # iterate origin-by-origin in this parish --------------------------
        for idx in parish_idx:
            origin       = poi_gdf.loc[idx]
            origin_uid   = origin["uid"]
            sr, sc       = int(origin["row_adj"]), int(origin["col_adj"])
            centre       = origin.geometry
            centre_par   = origin["parish_std"]

            # candidate POIs within radius
            cand_df = poi_gdf.iloc[
                list(sindex.intersection(centre.buffer(RADIUS_M).bounds))
            ].copy()
            cand_df = cand_df[
                (cand_df["PP_Bottega_METACATEGORY"].notna()) &
                (cand_df.index != idx)
            ]
            if cand_df.empty:
                parish_pbar.update(); overall_pbar.update();  continue

            cand_df["distance_m"] = cand_df.geometry.distance(centre)
            cand_df = cand_df[cand_df["distance_m"] <= RADIUS_M]
            if cand_df.empty:
                parish_pbar.update(); overall_pbar.update();  continue

            if PREF_PARISH:
                cand_df["same_parish"] = cand_df["parish_std"] == centre_par
                cand_df = cand_df.sort_values(
                    ["same_parish", "distance_m"], ascending=[False, True])
            else:
                cand_df = cand_df.sort_values("distance_m")

            cand_df = (
                cand_df
                .groupby("PP_Bottega_METACATEGORY", group_keys=False)
                .head(TOP_N)
            )

            # parallel A* for each candidate -------------------------------
            futures = {
                executor.submit(compute_path, sr, sc,
                                int(tgt["row_adj"]), int(tgt["col_adj"])
                               ): (tgt["PP_Bottega_METACATEGORY"], tgt["uid"])
                for _, tgt in cand_df.iterrows()
            }

            best_by_cat = {}
            for fut in as_completed(futures):
                res = fut.result()
                if res is None:       # no path
                    continue
                L, rows, cols = res
                cat, tgt_uid  = futures[fut]
                cur = best_by_cat.get(cat)
                if cur is None or L < cur["len"]:
                    best_by_cat[cat] = dict(len=L, uid=tgt_uid,
                                            rows=rows, cols=cols)

            # persist ------------------------------------------------------
            if best_by_cat:
                new_rows = []
                for cat, d in best_by_cat.items():
                    fname = f"{origin_uid}_{d['uid']}.npz"
                    fpath = Path(OUTPUT_DIR) / fname
                    np.savez_compressed(fpath, rows=d["rows"], cols=d["cols"])

                    new_rows.append(dict(
                        origin_uid  = origin_uid,
                        origin_type = "CASA",
                        target_uid  = d["uid"],
                        target_type = cat,
                        path_len    = d["len"],
                        path_file   = str(fpath)
                    ))

                df_index = add_rows_to_index(df_index, new_rows)

            done_uids.add(origin_uid)
            parish_pbar.update()
            overall_pbar.update()

        parish_pbar.close()

    executor.shutdown(wait=True)
    overall_pbar.close()

    rate = (time.time() - start)
    print(f"\n✓ Finished {len(done_uids):,} CASA in {rate/60:,.1f} min "
          f"→ {len(df_index):,} total connections")

if __name__ == "__main__":
    main()
