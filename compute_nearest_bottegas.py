#!/usr/bin/env python3
# fast_make_paths_parallel.py ----------------------------------------------
"""
*   Resumable (reads/writes paths_npz/connection_index.csv)
*   Thread-level parallelism for every path_from_poi() call
"""
import os
import time
import numpy as np
import pandas as pd
import geopandas as gpd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing as mp

from src_cpp.path_planner import path_from_poi
from src.sample_pois    import pois_within_radius

GRID_PATH   = "cpp_data/grid.npy"
POI_PATH    = "pois/catastaci_adjusted.geojson"
OUTPUT_DIR  = "paths_npz"

# RADIUS_M    = 320
RADIUS_M    = 160
TOP_N       = 2
MAX_DIST    = -1
PREF_PARISH = False
INDEX_FILE  = os.path.join(OUTPUT_DIR, "connection_index.csv")

# --------------------------------------------------------------------------
def load_index() -> pd.DataFrame:
    if os.path.exists(INDEX_FILE):
        return pd.read_csv(INDEX_FILE)
    return pd.DataFrame(columns=[
        "origin_uid", "origin_type",
        "target_uid", "target_type",
        "path_len",   "path_file",
    ])

def save_index(df: pd.DataFrame) -> None:
    df.to_csv(INDEX_FILE, index=False)

# --------------------------------------------------------------------------
def compute_path(sr, sc, trg_r, trg_c):
    """
    Thin wrapper so we can hand it straight to the executor.
    Returns (found, length, rows, cols)
    """
    res = path_from_poi(
        GRID, sr, sc, trg_r, trg_c,
        max_distance=MAX_DIST, diagonals=True,
    )
    if res is None:
        return False, None, None, None
    rows, cols = res
    return True, len(rows), rows, cols

# --------------------------------------------------------------------------
def main() -> None:
    global GRID                    # make grid available to worker threads
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    GRID    = np.load(GRID_PATH)
    poi_gdf = gpd.read_file(POI_PATH)

    if not poi_gdf.crs or not poi_gdf.crs.axis_info[0].unit_name.lower().startswith("metre"):
        poi_gdf = poi_gdf.to_crs(poi_gdf.estimate_utm_crs())

    df_index        = load_index()
    done_uids       = set(df_index["origin_uid"].unique())

    casa_idx = poi_gdf.index[poi_gdf["PP_Function_TOP"] == "CASA"].to_numpy()
    todo_idx = [idx for idx in casa_idx if poi_gdf.loc[idx, "uid"] not in done_uids]

    print(f"{len(done_uids):,} CASA done – {len(todo_idx):,} left.")

    sindex     = poi_gdf.sindex
    n_threads  = mp.cpu_count()
    executor   = ThreadPoolExecutor(max_workers=n_threads)

    pbar    = tqdm(total=len(todo_idx), desc="CASA POIs")
    start   = time.time()

    for processed, idx in enumerate(todo_idx, 1):
        origin      = poi_gdf.loc[idx]
        origin_uid  = origin["uid"]
        sr, sc      = int(origin["row_adj"]), int(origin["col_adj"])
        centre      = origin.geometry
        centre_par  = origin["parish_std"]

        # ---- spatial query (single buffer) --------------------------------
        cand_df = poi_gdf.iloc[
            list(sindex.intersection(centre.buffer(RADIUS_M).bounds))
        ].copy()

        cand_df = cand_df[
            (cand_df["PP_Bottega_METACATEGORY"].notna()) &
            (cand_df.index != idx)
        ]
        if cand_df.empty:
            pbar.update();  continue

        cand_df["distance_m"] = cand_df.geometry.distance(centre)
        cand_df = cand_df[cand_df["distance_m"] <= RADIUS_M]
        if cand_df.empty:
            pbar.update();  continue

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

        # ---- parallelise all A* calls for this CASA -----------------------
        futures = {}
        for _, tgt in cand_df.iterrows():
            fut = executor.submit(compute_path, sr, sc,
                                  int(tgt["row_adj"]), int(tgt["col_adj"]))
            futures[fut] = (tgt["PP_Bottega_METACATEGORY"], tgt["uid"])

        best_by_cat = {}
        for fut in as_completed(futures):
            found, L, rows, cols = fut.result()
            if not found:
                continue
            cat, tgt_uid = futures[fut]
            cur = best_by_cat.get(cat)
            if cur is None or L < cur["len"]:
                best_by_cat[cat] = dict(len=L, uid=tgt_uid,
                                        rows=rows, cols=cols)

        # ---- persist -------------------------------------------------------
        if not best_by_cat:
            pbar.update();  continue

        new_rows = []
        for cat, d in best_by_cat.items():
            fname = f"{origin_uid}_{d['uid']}.npz"
            fpath = os.path.join(OUTPUT_DIR, fname)
            np.savez_compressed(fpath, rows=d["rows"], cols=d["cols"])

            new_rows.append(dict(
                origin_uid  = origin_uid,
                origin_type = "CASA",
                target_uid  = d["uid"],
                target_type = cat,
                path_len    = d["len"],
                path_file   = fpath
            ))

        df_index = pd.concat([df_index, pd.DataFrame(new_rows)],
                             ignore_index=True)
        save_index(df_index)

        # ---- progress ------------------------------------------------------
        pbar.update()
        if processed % 100 == 0:
            rate = processed / (time.time() - start)
            pbar.set_postfix(rate=f"{rate:5.2f} CASA/s",
                             done=len(df_index["origin_uid"].unique()))

    pbar.close()
    executor.shutdown(wait=True)
    print(f"All done – {len(df_index):,} connections "
          f"for {len(df_index['origin_uid'].unique()):,} CASA POIs.")

if __name__ == "__main__":
    main()
