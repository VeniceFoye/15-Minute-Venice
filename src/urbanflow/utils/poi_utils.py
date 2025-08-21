from affine import Affine

import geopandas as gpd
import numpy as np
from typing import Tuple
from scipy.ndimage import distance_transform_edt

from pyproj import CRS
from shapely.geometry import Point

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


def pois_within_radius(poi_gdf: gpd.GeoDataFrame,
                       poi_idx: int,
                       meta_val: str,
                       radius_m: float,
                       distance_col: str = "distance_m",
                       prefer_same_parish: bool = False) -> gpd.GeoDataFrame:
    """
    Parameters
    ----------
    poi_gdf : GeoDataFrame
        Full POI table (must have 'geometry', 'PP_Bottega_METACATEGORY',
        and optionally 'parish_std').
    poi_idx : int
        Row position of the central POI in `poi_gdf`.
    meta_val : str
        Desired PP_Bottega_METACATEGORY value.
    radius_m : float
        Search radius in metres.
    distance_col : str, optional
        Column name that will store computed centre-to-POI distances.
    prefer_same_parish : bool, optional
        If True, results from the same parish (`parish_std`) as the centre
        POI are listed first (default False).

    Returns
    -------
    GeoDataFrame
        Matching POIs within radius, sorted by parish preference (if enabled)
        and then by straight-line distance.
    """
    if poi_idx < 0 or poi_idx >= len(poi_gdf):
        raise IndexError("poi_idx out of range")

    # ------------------------------------------------------------------ 1
    # Ensure we are in a metric CRS (metre units)
    # ------------------------------------------------------------------
    gdf = poi_gdf
    if gdf.crs is None:
        raise ValueError("GeoDataFrame has no CRS – cannot compute metric distances")

    crs_unit = CRS(gdf.crs).axis_info[0].unit_name.lower()
    if not crs_unit.startswith("metre"):
        centre_lon = gdf.iloc[poi_idx].geometry.x
        utm_zone   = int((centre_lon + 180) // 6 + 1)
        utm_crs    = CRS.from_epsg(32600 + utm_zone)  # EPSG 326xx = WGS-84 UTM North
        gdf = gdf.to_crs(utm_crs)

    # ------------------------------------------------------------------ 2
    # Filter by metacategory & compute distances
    # ------------------------------------------------------------------
    centre      = gdf.iloc[poi_idx].geometry
    centre_par  = gdf.iloc[poi_idx]["parish_std"] if "parish_std" in gdf.columns else None

    subset = gdf[gdf["PP_Bottega_METACATEGORY"] == meta_val].copy()
    subset[distance_col] = subset.geometry.distance(centre)
    subset = subset[subset[distance_col] <= radius_m]

    # ------------------------------------------------------------------ 3
    # Optional parish preference
    # ------------------------------------------------------------------
    if prefer_same_parish and "parish_std" in subset.columns and centre_par is not None:
        subset["same_parish"] = subset["parish_std"] == centre_par
        subset = subset.sort_values(
            by=["same_parish", distance_col],
            ascending=[False, True]          # same_parish=True first, then nearest
        )
        subset = subset.drop(columns="same_parish")
    else:
        subset = subset.sort_values(distance_col)

    return subset.reset_index(drop=True)
