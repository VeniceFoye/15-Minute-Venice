from affine import Affine

import geopandas as gpd
import numpy as np
from scipy.ndimage import distance_transform_edt

def pois_to_grid_coords(
    poi_gdf: gpd.GeoDataFrame,
    transform: Affine,
    grid: np.ndarray,
    *,
    do_adjusted: bool = True,
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
    
    # Quit if not do_adjusted
    if not do_adjusted:
        return poi_gdf

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