"""
gridify.py – rasterise Venice-style layers into a NumPy grid
State codes:
    0 = ocean   (background)
    1 = street
    2 = building      
    3 = canal         
    4 = courtyard
"""

from __future__ import annotations
import math
from typing import Tuple, Dict, Iterable

from scipy import ndimage as ndi
import numpy as np
import geopandas as gpd
from affine import Affine
from shapely.geometry import box
from rasterio import features
from scipy.ndimage import binary_propagation, distance_transform_edt
from pandas import DataFrame



# ----------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------


def gdfs_to_grid(
    buildings: gpd.GeoDataFrame,
    streets: gpd.GeoDataFrame,
    canals: gpd.GeoDataFrame | None = None,
    *,
    cell_size: float,
) -> Tuple[np.ndarray, Affine, Dict[int, str]]:
    """
    Rasterise buildings, streets, (optionally) canals to a numpy grid.

    Override order (lowest → highest priority)
        0  ocean   (implicit background)
        1  street
        3  canal
        2  building

    Parameters
    ----------
    buildings, streets, canals : GeoDataFrames
        All must share the **same CRS**.
        `canals` may be None – then it’s skipped.
    cell_size : float
        Square cell width/height in CRS units.

    Returns
    -------
    grid, transform, legend     (same semantics as before, but legend has 3)
    """
    # ---------- sanity checks ------------------------------------------------
    layers = [g for g in (buildings, streets, canals) if g is not None]
    crs_set = {g.crs for g in layers}
    if len(crs_set) != 1:
        raise ValueError("All GeoDataFrames must share the same CRS")

    # ---------- grid extent --------------------------------------------------
    minx, miny, maxx, maxy = _union_bounds(*layers)

    # snap extent to whole cells
    minx = math.floor(minx / cell_size) * cell_size
    miny = math.floor(miny / cell_size) * cell_size
    maxx = math.ceil(maxx / cell_size) * cell_size
    maxy = math.ceil(maxy / cell_size) * cell_size

    cols = int((maxx - minx) / cell_size)
    rows = int((maxy - miny) / cell_size)

    transform = Affine(cell_size, 0, minx,
                       0, -cell_size, maxy)

    grid = np.zeros((rows, cols), dtype=np.uint8)        # 0 = ocean


    if canals is not None:
        _burn_geoms(canals.geometry, 3, grid, transform)

    _burn_geoms(buildings.geometry, 2, grid, transform)

    # burn order: streets win
    _burn_geoms(streets.geometry, 1, grid, transform)
    

    legend = {0: "ocean", 1: "street", 2: "building", 3: "canal"}
    return grid, transform, legend


def point_to_cell(x: float, y: float, transform: Affine) -> Tuple[int, int]:
    """
    Convert a CRS-aligned point to (row, col).
    Raises ValueError if the point is outside the grid.
    """
    col_f, row_f = ~transform * (x, y)          # inverse affine
    row, col = int(row_f), int(col_f)
    if row < 0 or col < 0:
        raise ValueError("Point outside grid (negative index)")
    return row, col


def cell_to_polygon(row: int, col: int, transform: Affine):
    """
    Return the Shapely polygon representing a single grid cell.
    Useful for plotting/debugging.
    """
    x_min, y_max = transform * (col, row)
    x_max, y_min = transform * (col + 1, row + 1)
    return box(x_min, y_min, x_max, y_max)

def add_courtyards_fast(grid: np.ndarray,
                        *,
                        empty_code: int = 0,
                        courtyard_code: int = 4,
                        structure: np.ndarray | None = None):
    """
    Mark every empty cell that is *not* 8-connected to the border as courtyard.
    Works in-place and is ~10× faster than the label-and-loop version.
    """
    if structure is None:
        structure = np.ones((3, 3), dtype=bool)      # 8-connectivity

    mask = (grid == empty_code)

    # ---- seed: empty cells on the border -----------------------------------
    seed = np.zeros_like(mask, dtype=bool)
    seed[0, :]  = mask[0, :]
    seed[-1, :] = mask[-1, :]
    seed[:, 0]  = mask[:, 0]
    seed[:, -1] = mask[:, -1]

    # ---- propagate the seed through the empty mask -------------------------
    connected = binary_propagation(seed, mask=mask, structure=structure)

    # ---- anything empty *and* not connected → courtyard --------------------
    grid[mask & ~connected] = courtyard_code
    legend = {0: "ocean", 1: "street", 2: "building", 3: "canal", 4: "courtyard"}
    return grid, legend

def pois_to_grid_coords(
    poi_gdf: gpd.GeoDataFrame,
    transform: Affine,
    grid_shape: tuple[int, int],
    *,
    keep_outside: bool = False,
) -> gpd.GeoDataFrame:
    """
    Vectorised, guaranteed-correct conversion of POI coordinates
    → grid row/col using the affine coefficients directly.

    Parameters
    ----------
    transform     – affine returned by venice_grid / gdfs_to_grid
    grid_shape    – (rows, cols) of the uint8 grid
    keep_outside  – if False, rows outside the grid are dropped

    Returns
    -------
    GeoDataFrame identical to input + 'row', 'col' integer columns.
    """
    rows, cols = grid_shape       # grid.shape
    a, b, c, d, e, f, *_ = transform   # Affine tuple

    # Affine is [ x ]   [ a  b  c ] [ col ]
    #             y   = [ d  e  f ] [ row ]
    # For our transform: a = +cell, e = –cell, b=d=0, c=minx, f=maxy
    # Solve:
    #   col = (x - c) / a
    #   row = (f - y) / cell_size
    x = poi_gdf.geometry.x.values
    y = poi_gdf.geometry.y.values

    col_idx = ((x - c) / a).astype(int)
    row_idx = ((f - y) / (-e)).astype(int)   # note: e = –cell_size

    inside = (row_idx >= 0) & (row_idx < rows) & (col_idx >= 0) & (col_idx < cols)

    if keep_outside:
        poi_gdf = poi_gdf.copy()
    else:
        poi_gdf = poi_gdf.loc[inside].copy()   # keep only valid points
        row_idx = row_idx[inside]
        col_idx = col_idx[inside]

    poi_gdf["row"] = row_idx
    poi_gdf["col"] = col_idx
    return poi_gdf


def pois_to_grid_coords_adjusted(
    poi_gdf: gpd.GeoDataFrame,
    transform: Affine,
    grid: np.ndarray,
    *,
    keep_outside: bool = False,
    street_code: int = 1,
    building_code: int = 2,
) -> gpd.GeoDataFrame:
    """
    Map POI coordinates to grid indices AND snap them to the nearest
    'door cell' (street cell adjacent to a building).

    Parameters
    ----------
    poi_gdf : GeoDataFrame       – must be in the same CRS as *transform*
    transform : Affine           – from gdfs_to_grid / venice_grid
    grid : ndarray[uint8]        – the uint8 grid with state codes
    keep_outside : bool          – keep/drop POIs that fall outside grid
    street_code, building_code : int – codes used in *grid*

    Returns
    -------
    GeoDataFrame  – original columns +:
        row, col       integer indices of the original cell
        row_adj, col_adj  integer indices of the snapped door cell
    """
    rows, cols = grid.shape
    a, b, c, d, e, f, *_ = transform  # affine coeffs (b=d=0)

    # ------------------------------------------------------------------ #
    # 1. raw (row, col) from coordinates (vectorised)                    #
    # ------------------------------------------------------------------ #
    x = poi_gdf.geometry.x.values
    y = poi_gdf.geometry.y.values
    col_idx = ((x - c) / a).astype(int)
    row_idx = ((f - y) / (-e)).astype(int)

    inside = (
        (row_idx >= 0) & (row_idx < rows) &
        (col_idx >= 0) & (col_idx < cols)
    )
    if not keep_outside:
        poi_gdf = poi_gdf.loc[inside].copy()
        row_idx = row_idx[inside]
        col_idx = col_idx[inside]
    else:
        poi_gdf = poi_gdf.copy()

    poi_gdf["row"] = row_idx
    poi_gdf["col"] = col_idx

    if len(poi_gdf) == 0:
        # nothing to snap
        poi_gdf["row_adj"] = np.array([], dtype=int)
        poi_gdf["col_adj"] = np.array([], dtype=int)
        return poi_gdf

    # ------------------------------------------------------------------ #
    # 2. build 'door mask' = street cells touching a building            #
    # ------------------------------------------------------------------ #
    street = (grid == street_code)
    building = (grid == building_code)

    # 4-neighbour shifts of building mask
    adj = (
        np.roll(building,  1, axis=0) |  # north
        np.roll(building, -1, axis=0) |  # south
        np.roll(building,  1, axis=1) |  # west
        np.roll(building, -1, axis=1)    # east
    )
    door_mask = street & adj                 # True where street touches building

    if not door_mask.any():
        # no door cells found – fall back: keep original indices
        poi_gdf["row_adj"] = row_idx
        poi_gdf["col_adj"] = col_idx
        return poi_gdf

    # ------------------------------------------------------------------ #
    # 3. snap every cell to nearest door cell using a distance transform #
    # ------------------------------------------------------------------ #
    # EDT on inverse mask; return_indices=True gives row/col of nearest True
    ind_row, ind_col = distance_transform_edt(
        ~door_mask, return_distances=False, return_indices=True
    )


    # ------------------------------------------------------------------ #
    # 4. attach snapped coordinates                                       #
    # ------------------------------------------------------------------ #
    poi_gdf["row_adj"] = ind_row[row_idx, col_idx]
    poi_gdf["col_adj"] = ind_col[row_idx, col_idx]

    return poi_gdf
# ----------------------------------------------------------------------
# Internal helpers
# ----------------------------------------------------------------------

def _union_bounds(*gdfs: gpd.GeoDataFrame) -> Tuple[float, float, float, float]:
    """Return combined bounding box of several GeoDataFrames."""
    minx, miny, maxx, maxy = gdfs[0].total_bounds
    for gdf in gdfs[1:]:
        x0, y0, x1, y1 = gdf.total_bounds
        minx, miny = min(minx, x0), min(miny, y0)
        maxx, maxy = max(maxx, x1), max(maxy, y1)
    return minx, miny, maxx, maxy


def _burn_geoms(
    geoms: Iterable,
    value: int,
    out: np.ndarray,
    transform: Affine,
):
    """Rasterise geometries into an existing array (in-place)."""
    shapes = ((geom, value) for geom in geoms if not geom.is_empty)
    features.rasterize(
        shapes=shapes,
        out=out,                       # burn into existing array
        transform=transform,
        all_touched=True,              # mark a cell if geom touches it at all
        default_value=value,
    )
