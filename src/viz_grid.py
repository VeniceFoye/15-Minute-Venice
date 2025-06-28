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


def path_on_grid_image(
    grid: np.ndarray,
    transform,                       # affine from venice_grid / gdfs_to_grid
    path_r: np.ndarray,
    path_c: np.ndarray,
    *,
    scale: int = 5,
    palette: dict[int, tuple[int, int, int]] | None = None,
    poi_start: tuple[int, int] | None = None,   # (row_adj, col_adj)
    poi_end: tuple[int, int]   | None = None,   # (row_adj, col_adj)
    poi_colour: tuple[int, int, int] = (0, 0, 0),
    path_colour: tuple[int, int, int] = (255, 215, 0),     # gold
    path_width: int | None = None,
) -> Image.Image:
    """
    Render grid + (optional) POIs + path.  Returns a PIL Image.

    Parameters
    ----------
    path_r, path_c : int arrays
        Output from `path_from_poi` (exclusive of origin).
    poi_start / poi_end : (row,col) tuples in grid coords
        If given, small dots are drawn at those cells.
    path_width : int
        Width of the polyline in *pixels*. Defaults to max(1, scale//2).

    Returns
    -------
    PIL.Image.Image  (ready to save or display)
    """
    img = grid_to_image(grid, scale=scale, palette=palette)
    draw = ImageDraw.Draw(img)
    w = path_width if path_width is not None else max(1, scale // 2)

    # ------------------------------------------------------------------ #
    # 1. draw path as a polyline in pixel space
    # ------------------------------------------------------------------ #
    pts = [
        (c * scale + scale // 2, r * scale + scale // 2)
        for r, c in zip(path_r, path_c)
    ]
    if pts:
        draw.line(pts, fill=path_colour, width=w, joint="curve")

    # ------------------------------------------------------------------ #
    # 2. optional start / end POI dots
    # ------------------------------------------------------------------ #
    R = max(2, w)   # radius
    if poi_start is not None:
        x, y = poi_start[1] * scale + scale // 2, poi_start[0] * scale + scale // 2
        draw.ellipse((x - R, y - R, x + R, y + R), fill=poi_colour)
    if poi_end is not None:
        x, y = poi_end[1] * scale + scale // 2, poi_end[0] * scale + scale // 2
        draw.ellipse((x - R, y - R, x + R, y + R), fill=poi_colour)

    return img

# ---------------------------------------------------------------------
#  combined – grid + many POIs + a path  (all in one PIL.Image)
# ---------------------------------------------------------------------
from PIL import ImageDraw
import matplotlib.pyplot as plt         # only for tab20 colours

def grid_pois_and_path_image(
    grid: np.ndarray,
    transform,                                  # unused – keeps API symmetric
    poi_gdf: gpd.GeoDataFrame,
    path_r: np.ndarray,
    path_c: np.ndarray,
    *,
    # grid
    scale: int = 5,
    palette: dict[int, tuple[int, int, int]] | None = None,
    bg_alpha: float = 1.0,          # ⬅️  1 = full, 0.5 = 50 % faded, …
    # POIs
    function_col: str = "PP_Function_TOP",
    color_map: dict[str, tuple[int, int, int]] | None = None,
    default_color: tuple[int, int, int] = (0, 0, 0),
    poi_radius: int | None = None,
    # path
    path_colour: tuple[int, int, int] = (255, 215, 0),
    path_width: int | None = None,
    # emphasis dots
    poi_start: tuple[int, int] | None = None,
    poi_end:   tuple[int, int] | None = None,
    poi_dot_colour: tuple[int, int, int] = (0, 0, 0),
) -> Image.Image:
    """
    Render land-use grid, POI dots, and a path polyline.

    Parameters
    ----------
    bg_alpha : float
        1 → original grid colours,  
        0 → completely white; values in between fade the background.

    Returns
    -------
    PIL.Image.Image
    """
    # ------------------------------------------------------------------
    # 0 | base land-use raster (RGB)
    # ------------------------------------------------------------------
    base = grid_to_image(grid, scale=scale, palette=palette)

    if not (0 < bg_alpha <= 1):
        raise ValueError("bg_alpha must be in (0, 1].")

    if bg_alpha < 1:                     # dim → blend with white
        white = Image.new("RGB", base.size, (255, 255, 255))
        base  = Image.blend(base, white, 1 - bg_alpha)

    draw = ImageDraw.Draw(base)

    # ------------------------------------------------------------------
    # 1 | path polyline (draw *before* POI dots so dots sit on top)
    # ------------------------------------------------------------------
    w = path_width if path_width is not None else max(1, scale // 2)
    if len(path_r):
        pts = [(c*scale + scale//2, r*scale + scale//2)
               for r, c in zip(path_r, path_c)]
        draw.line(pts, fill=path_colour, width=w, joint="curve")

    # optional emphasised start/end dots
    Rdot = max(2, w + 1)
    for cell in (poi_start, poi_end):
        if cell is not None:
            x = cell[1]*scale + scale//2
            y = cell[0]*scale + scale//2
            draw.ellipse((x-Rdot, y-Rdot, x+Rdot, y+Rdot),
                         fill=poi_dot_colour)

    # ------------------------------------------------------------------
    # 2 | category → colour map (same logic as before)
    # ------------------------------------------------------------------
    cats = poi_gdf[function_col].fillna("UNKNOWN").astype(str)
    if color_map is None:
        base_cycle = plt.cm.get_cmap("tab20").colors
        color_map = {cat: tuple(int(255*c) for c in base_cycle[i % 20])
                     for i, cat in enumerate(sorted(cats.unique()))}
    else:
        color_map = {k: tuple(int(x) for x in v) for k, v in color_map.items()}

    # ------------------------------------------------------------------
    # 3 | draw POI dots on top
    # ------------------------------------------------------------------
    R = poi_radius if poi_radius is not None else max(1, scale // 2)
    for r, c, cat in zip(poi_gdf["row"], poi_gdf["col"], cats):
        colour = color_map.get(cat, default_color)
        x = c*scale + scale//2
        y = r*scale + scale//2
        draw.ellipse((x-R, y-R, x+R, y+R), fill=colour)

    return base
