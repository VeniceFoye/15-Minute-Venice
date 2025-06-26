# viz_grid.py
from __future__ import annotations
from typing import Dict, Tuple
import numpy as np
from PIL import Image, ImageDraw
from affine import Affine
import geopandas as gpd
import matplotlib.pyplot as plt   # only for color cycle




# ----------------------------------------------------------------------
# Main API
# ----------------------------------------------------------------------

def grid_to_image(
    grid: np.ndarray,
    scale: int = 1,
    palette: Dict[int, Tuple[int, int, int]] | None = None,
) -> Image.Image:
    """
    Convert a uint8 grid of states into a PIL Image.

    Parameters
    ----------
    grid : ndarray[int]   shape (rows, cols)
        Cell codes: 0=ocean, 1=street, 2=building (extend as you wish).
    scale : int, default 1
        How many output pixels per cell (must be ≥1). 2 doubles width/height.
    palette : {state: (R, G, B)}, optional
        Custom colors.  Unspecified states default to white.

    Returns
    -------
    PIL.Image.Image  (mode "RGB")

    Notes
    -----
    * Uses pure NumPy → very fast, even for multi-million-cell grids.
    * Nearest-neighbour up-scaling keeps the crisp blocky CA look.
    """
    if scale < 1 or not isinstance(scale, int):
        raise ValueError("scale must be a positive integer")

    # ---------- default color map ------------------------------------
    default_palette = {
        0: (173, 216, 230),   # ocean – light blue
        1: (190, 190, 190),   # street – light grey
        2: (178,  34,  34),   # building – dark red
        3: ( 64, 224, 208),   # canal – turquoise
        4: (152, 251, 152),   # courtyard – pale green
    }
    if palette is not None:
        default_palette.update(palette)

    rows, cols = grid.shape
    rgb = np.zeros((rows, cols, 3), dtype=np.uint8)

    for state, color in default_palette.items():
        rgb[grid == state] = color

    img = Image.fromarray(rgb, mode="RGB")

    if scale > 1:
        img = img.resize((cols * scale, rows * scale), resample=Image.Resampling.NEAREST)

    return img


def save_grid_image(
    grid: np.ndarray,
    path: str,
    scale: int = 1,
    palette: Dict[int, Tuple[int, int, int]] | None = None,
) -> None:
    """Shortcut: build and save PNG/TIFF/etc.  Format inferred from file suffix."""
    img = grid_to_image(grid, scale=scale, palette=palette)
    img.save(path)



def grid_with_pois_image(
    grid: np.ndarray,
    transform: Affine,
    poi_gdf: gpd.GeoDataFrame,
    *,
    scale: int = 1,
    palette: dict[int, tuple[int, int, int]] | None = None,
    function_col: str = "PP_Function_TOP",
    color_map: dict[str, tuple[int, int, int]] | None = None,
    default_color: tuple[int, int, int] = (0, 0, 0),
    poi_radius: int | None = None,
) -> Image.Image:
    """
    Render the grid and overlay POIs colored by `function_col`.

    Parameters
    ----------
    function_col : str
        Column in poi_gdf to drive the color mapping.
    color_map : dict {category: (R,G,B)}
        Pass your own mapping.  If None, a palette is generated.
    default_color : RGB
        Used for categories that appear after the mapping is built (rare).
    """
    img = grid_to_image(grid, scale=scale, palette=palette)
    draw = ImageDraw.Draw(img)
    R = poi_radius if poi_radius is not None else max(1, scale // 2)

    # ------------------------------------------------------------------
    # build or validate category → color dict
    # ------------------------------------------------------------------
    cats = poi_gdf[function_col].fillna("UNKNOWN").astype(str)
    if color_map is None:
        base_cycle = plt.cm.get_cmap("tab20").colors   # 20 distinct colors
        color_map = {cat: tuple(int(255*c) for c in base_cycle[i % 20])
                      for i, cat in enumerate(sorted(cats.unique()))}
    else:
        # ensure RGB ints 0-255
        color_map = {k: tuple(int(x) for x in v) for k, v in color_map.items()}

    # ------------------------------------------------------------------
    # draw dots
    # ------------------------------------------------------------------
    for r, c, cat in zip(poi_gdf["row"], poi_gdf["col"], cats):
        color = color_map.get(cat, default_color)
        x = c * scale + scale // 2
        y = r * scale + scale // 2
        draw.ellipse((x - R, y - R, x + R, y + R), fill=color)

    return img