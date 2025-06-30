import geopandas as gpd
from pyproj import CRS
from shapely.geometry import Point

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
