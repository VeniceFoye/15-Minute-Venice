"""
sommarioni_utils.py

Util functions to process 1808 Sommarioni Cadaster
"""
import geopandas as gpd


def load_sommarioni_layers(
    path: str,
    target_crs: int | str = "EPSG:32633",
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame,
           gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """
    Read the 1808 GeoJSON and return four GeoDataFrames:
        buildings_gdf, streets_gdf, canals_gdf, courtyards_gdf
    All re-projected to *target_crs* (EPSG:32633 metres by default).
    """
    if path.split(".")[-1] == "feather":
        gdf = gpd.read_feather(path)
    else:
        print(path)
        gdf = gpd.read_file(path)


    buildings = gdf[gdf["geometry_type"] == 'building'].copy()
    streets   = gdf[gdf["geometry_type"].isin(["street", "sottoportico"])].copy()
    canals    = gdf[gdf["geometry_type"] == "water"].copy()
    courtyards = gdf[gdf["geometry_type"] == "courtyard"].copy()

    # occasionally sottoportici are MultiPolygons: explode for cleaner raster
    buildings = buildings.explode(index_parts=False)

    return buildings, streets, canals, courtyards