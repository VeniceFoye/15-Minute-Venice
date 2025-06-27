#!/usr/bin/env python3
"""
build_paths_needs.py — Compute shortest A* paths from parish houses (CASA) to the
workshop/service POIs that satisfy their social‑class needs.

The script is a linearised, parameterised version of the exploratory Jupyter
notebook you shared.  You can now switch parish subsets or change the output
folder simply with CLI flags — no manual cell editing required 🥳

Example
-------
python build_paths_needs.py \
    --needs-file ../data/needs_adjusted.feather \
    --parish "Santa Croce" \
    --grid-path  ../cpp_data/grid.npy \
    --output-dir ../paths_npz_needs_santa_croce \
    --top-n 2 --radius 1260

Main options
------------
--needs-file   Path to the POI GeoDataFrame (Feather/GeoJSON/…)
--parish       Name of the parish whose CASA points will act as origins
--output-dir   Folder that will receive the *.npz path files + index CSV
--grid-path    Numpy uint8 walkability grid produced by the rasteriser
--radius       Search radius around each CASA (metres, default 1260)
--top-n        How many nearest POIs to keep per needed service (default 2)
--max-dist     Max cells to explore inside A* (‑1 = unlimited)
--threads      Number of worker threads (default = CPU‑count)
--quiet        Reduce logging verbosity

A connection_index.csv is (re)‑packed inside the output folder to let the script
skip already processed houses on subsequent runs.

Requirements
------------
* geopandas, pandas, numpy, shapely, tqdm, argparse, logging
* src_cpp.path_planner.path_from_poi must be importable (C++ A*)
"""

from __future__ import annotations

import argparse
import logging
import multiprocessing as mp
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Helper: thin wrapper around the compiled C++ A*
# ---------------------------------------------------------------------------
try:
    sys.path.append(str(Path.cwd().parent))  # allow "src_cpp" next to repo root
    from src_cpp.path_planner import path_from_poi  # type: ignore
except (ModuleNotFoundError, ImportError) as exc:
    raise SystemExit(
        "❌  Could not import src_cpp.path_planner.  Make sure the C++ module is "
        "compiled and PYTHONPATH includes its parent directory."
    ) from exc


# pylint: disable=too-many-locals,too-many-branches,too-many-statements

def compute_path(grid: np.ndarray, sr: int, sc: int, trg_r: int, trg_c: int, *,
                 max_dist: int) -> tuple[int, np.ndarray, np.ndarray] | None:
    """Return (length, rows, cols) for the shortest path or *None* if blocked."""
    found = path_from_poi(grid, sr, sc, trg_r, trg_c,
                          max_distance=max_dist, diagonals=True)
    if found is None:
        return None
    rows, cols = found
    return len(rows), rows, cols


# ---------------------------------------------------------------------------
# Main routine
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:  # noqa: C901
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--needs-file", required=True,
                    help="Feather / GeoJSON file containing POIs + CASA rows")
    ap.add_argument("--parish", required=True,
                    help="Parish name used to pick CASA origins (column parish_std)")
    ap.add_argument("--output-dir", required=True,
                    help="Folder for *.npz paths and connection_index.csv")
    ap.add_argument("--grid-path", required=True,
                    help="NumPy uint8 walkability grid from the rasteriser")
    ap.add_argument("--radius", type=float, default=1260,
                    help="Search radius around each CASA (metres)")
    ap.add_argument("--top-n", type=int, default=2,
                    help="Keep the N closest POIs per needed service")
    ap.add_argument("--max-dist", type=int, default=-1,
                    help="Maximum cells explored inside A* (‑1 = unlimited)")
    ap.add_argument("--threads", type=int, default=mp.cpu_count(),
                    help="Thread pool size (defaults to CPU‑count)")
    ap.add_argument("--quiet", action="store_true",
                    help="Less verbose logging")

    args = ap.parse_args(argv)

    # ---------------- logging ----------------
    log_level = logging.INFO if args.quiet else logging.DEBUG
    logging.basicConfig(format="[%(levelname)s] %(message)s", level=log_level)
    log = logging.getLogger(__name__)

    # ---------------- load needs GDF ----------------
    log.info("Loading needs file → %s", args.needs_file)
    suf = Path(args.needs_file).suffix.lower()
    if suf in {".feather", ".ft"}:
        needs: gpd.GeoDataFrame = gpd.read_feather(args.needs_file)  # type: ignore[arg-type]
    else:
        needs = gpd.read_file(args.needs_file)  # type: ignore[arg-type]

    needs = needs.drop_duplicates()

    # ---------------- bottega_std_groups ----------------
    bool_cols = ["is_nobility_need", "is_middle_class_need", "is_lower_class_need"]
    bottega_std_groups = (
        needs.groupby("PP_Bottega_STD")[bool_cols]
             .agg(lambda x: x.mode().iloc[0] if not x.mode().empty else None)
             .reset_index()
    )
    bottega_std_groups = bottega_std_groups.dropna()
    bottega_std_groups = bottega_std_groups[bottega_std_groups.PP_Bottega_STD != "?"]
    bottega_std_groups[bool_cols] = (
        bottega_std_groups[bool_cols]
            .astype(str)
            .apply(lambda s: s.str.strip().str.lower() == "true")
    )

    # ---------------- CASA subset ----------------
    casas = needs[(needs.PP_Function_TOP == "CASA") &
                  (needs.parish_std == args.parish)].copy()
    if casas.empty:
        log.error("No CASA rows found for parish '%s' — aborting.", args.parish)
        sys.exit(2)
    casas = casas.drop_duplicates()

    log.info("%s CASA origins found for parish '%s'", len(casas), args.parish)

    # ---------------- ensure CRS in metres ----------------
    for name, gdf in (("casas", casas), ("needs", needs)):
        if not gdf.crs or not gdf.crs.axis_info[0].unit_name.lower().startswith("metre"):
            log.debug("Converting CRS for %s to metre‑based UTM …", name)
            gdf.to_crs(gdf.estimate_utm_crs(), inplace=True)

    # ---------------- spatial index on needs ----------------
    needs_sindex = needs.sindex

    # ---------------- load walkability grid ----------------
    grid = np.load(args.grid_path)
    log.info("Grid loaded – shape %s", grid.shape)

    # ---------------- bookkeeping ----------------
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    index_file = out_dir / "connection_index.csv"

    if index_file.exists():
        df_index = pd.read_csv(index_file)
        done_uid = set(df_index["origin_uid"].unique())
        log.info("Loaded %s cached houses from %s", len(done_uid), index_file)
    else:
        df_index = pd.DataFrame(columns=[
            "origin_uid", "origin_type",
            "target_uid", "target_type",
            "path_len", "path_file",
        ])
        done_uid = set()

    casas_todo = casas.loc[~casas["uid"].isin(done_uid)].drop_duplicates("uid")
    tot_houses = len(casas_todo)
    if tot_houses == 0:
        log.warning("Nothing to do – all houses already cached.")
        return

    # ---------------- thread pool ----------------
    executor = ThreadPoolExecutor(max_workers=args.threads)

    t0 = time.time()
    paths_written = houses_skipped = 0
    outer = tqdm(total=tot_houses, desc="Houses", disable=args.quiet)

    for _, casa in casas_todo.iterrows():
        uid = casa["uid"]
        outer.set_description_str(f"House {uid}")

        sr, sc = int(casa["row_adj"]), int(casa["col_adj"])

        # ----- which services does this tenant need? -----
        needs_codes: list[str] = []
        if casa.get("is_nobility_ten"):
            needs_codes += bottega_std_groups.loc[
                bottega_std_groups["is_nobility_need"], "PP_Bottega_STD"].tolist()
        if casa.get("is_middle_ten"):
            needs_codes += bottega_std_groups.loc[
                bottega_std_groups["is_middle_class_need"], "PP_Bottega_STD"].tolist()
        if casa.get("is_lower_class_ten"):
            needs_codes += bottega_std_groups.loc[
                bottega_std_groups["is_lower_class_need"], "PP_Bottega_STD"].tolist()
        if not needs_codes:
            houses_skipped += 1
            outer.update(); continue

        # ----- candidate POIs within radius -----
        buffer = casa.geometry.buffer(args.radius)
        cand_idx = list(needs_sindex.intersection(buffer.bounds))
        cand = needs.iloc[cand_idx].copy()
        cand = cand[cand["PP_Bottega_STD"].isin(needs_codes)]
        cand["distance_m"] = cand.geometry.distance(casa.geometry)
        cand = cand[cand["distance_m"] <= args.radius]
        if cand.empty:
            houses_skipped += 1
            outer.update(); continue

        cand = (cand.sort_values("distance_m")
                    .groupby("PP_Bottega_STD", group_keys=False)
                    .head(args.top_n))

        # ----- launch A* workers -----
        fut_map = {
            executor.submit(compute_path, grid, sr, sc,
                            int(row.row_adj), int(row.col_adj),
                            max_dist=args.max_dist): (row.PP_Bottega_STD, row.uid)
            for _, row in cand.iterrows()
        }
        inner = tqdm(total=len(fut_map), desc="services", position=1, leave=False, disable=args.quiet)

        best: dict[str, dict] = {}
        for fut in as_completed(fut_map):
            inner.update()
            res = fut.result()
            if res is None:
                continue
            length, rows, cols = res
            code, tgt_uid = fut_map[fut]
            cur = best.get(code)
            if cur is None or length < cur["len"]:
                best[code] = dict(len=length, uid=tgt_uid, rows=rows, cols=cols)
        inner.close()

        # ----- persist paths + index rows -----
        if best:
            new_rows: list[dict] = []
            for code, d in best.items():
                fname = f"{uid}_{d['uid']}.npz"
                fpath = out_dir / fname
                np.savez_compressed(fpath, rows=d["rows"], cols=d["cols"])
                new_rows.append(dict(
                    origin_uid=uid, origin_type="CASA",
                    target_uid=d["uid"], target_type=code,
                    path_len=d["len"], path_file=fname,
                ))
            df_index = pd.concat([df_index, pd.DataFrame(new_rows)], ignore_index=True)
            df_index.to_csv(index_file, index=False)
            paths_written += len(new_rows)
        else:
            houses_skipped += 1

        outer.update()

    executor.shutdown(wait=True)
    outer.close()

    dt = time.time() - t0
    log.info(
        "✓ Finished %s houses in %.1f min — %s paths saved, %s houses skipped.",
        tot_houses, dt / 60, paths_written, houses_skipped,
    )


if __name__ == "__main__":
    main()
