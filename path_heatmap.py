#!/usr/bin/env python3
# ---------------------------------------------------------------------------
#  path_heatmap.py   –   draw a heat-overlay of saved paths on a Venice grid
#
#  USAGE
#  -----
#     python path_heatmap.py \
#        --grid   cpp_data/grid.npy                # .npy  *or*  .npz (key "grid")
#        --index  paths_npz/connection_index.csv   # has column  path_file
#        --out    heatmap_overlay.png              # "-" → show in Jupyter
#        --scale  4                                # pixels per grid-cell
#        --alpha  0.6                              # heat transparency 0-1
#
#  DEPENDENCIES
#  ------------
#     numpy, pandas, pillow, matplotlib, tqdm
# ---------------------------------------------------------------------------

from __future__ import annotations
import argparse, time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.cm as cm
from tqdm import tqdm

# ─────────────────────────────── CLI ──────────────────────────────────────
ap = argparse.ArgumentParser(description="Draw a heat-overlay of path usage")
ap.add_argument("--grid",  required=True,  help=".npy or .npz grid file")
ap.add_argument("--index", required=True,  help="CSV with 'path_file' column")
ap.add_argument("--out",   default="heatmap_overlay.png",
                help="PNG to write (use '-' to display inline)")
ap.add_argument("--scale", type=int, default=1,
                help="pixels per grid cell (default 1)")
ap.add_argument("--alpha", type=float, default=0.6,
                help="heatmap opacity 0-1 (default 0.6)")
args = ap.parse_args()

t0 = time.time()

# ─────────────────────────────── 1. GRID ─────────────────────────────────
arr = np.load(args.grid, allow_pickle=False)
grid = arr["grid"] if isinstance(arr, np.lib.npyio.NpzFile) else arr
H, W = grid.shape

# palette (same as previous helpers)
PALETTE = {
    0: (173, 216, 230),   # ocean
    1: (190, 190, 190),   # street
    2: (178,  34,  34),   # building
    3: ( 64, 224, 208),   # canal
    4: (152, 251, 152),   # courtyard
}

# base RGB image (1-pixel per cell, later up-scaled)
base = np.zeros((H, W, 3), np.uint8)
for code, rgb in PALETTE.items():
    base[grid == code] = rgb

# ─────────────────────────────── 2. HEAT MAP ─────────────────────────────
heat = np.zeros_like(grid, dtype=np.uint32)

df = pd.read_csv(args.index)
paths = df["path_file"].tolist()
print(f"Accumulating {len(paths):,} paths …")

for f in tqdm(paths, unit="path"):
    data = np.load(f)
    heat[data["rows"], data["cols"]] += 1

print(f"max hits in one cell: {heat.max():,}")

# log-scale normalisation (avoids dominance by a few extremal cells)
norm = np.log1p(heat).astype(float)
norm /= norm.max() if norm.max() else 1.0        # 0-1

# map to RGBA via matplotlib's “hot” colormap
rgba = (cm.get_cmap("hot")(norm) * 255).astype(np.uint8)
rgba[..., 3] = (rgba[..., 3] * args.alpha).astype(np.uint8)  # global alpha

# ─────────────────────────────── 3. RENDER  ──────────────────────────────
scale = max(1, args.scale)
base_img = Image.fromarray(base, "RGB").resize((W*scale, H*scale),
                                               resample=Image.NEAREST)
heat_img = Image.fromarray(rgba, "RGBA").resize((W*scale, H*scale),
                                                resample=Image.NEAREST)

out_img = base_img.convert("RGBA")
out_img.alpha_composite(heat_img)

# ─────────────────────────────── 4. SAVE / SHOW ─────────────────────────
if args.out == "-":
    try:
        from IPython.display import display
        display(out_img)
    except ImportError:
        out_img.show()
else:
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out_img.save(args.out)
    print(f"✓ wrote {args.out}")

print(f"done in {time.time()-t0:,.1f}s")
