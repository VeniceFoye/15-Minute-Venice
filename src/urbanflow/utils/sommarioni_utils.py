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
    Load and split the 1808 Sommarioni cadaster dataset into thematic layers.

    Reads a GeoJSON or Feather file of cadaster geometries, filters them by
    `geometry_type`, and returns four separate GeoDataFrames:
    buildings, streets, canals, and courtyards. All layers are re-projected
    to the target CRS.

    Parameters
    ----------
    path : str
        Path to the cadaster file. May be GeoJSON (default) or Feather.
    target_crs : str or int, optional
        Coordinate reference system to reproject the data into.
        Defaults to ``"EPSG:32633"`` (UTM zone 33N, meters).

    Returns
    -------
    buildings : geopandas.GeoDataFrame
        Polygons labeled as ``"building"`` in the input data.
    streets : geopandas.GeoDataFrame
        Polygons labeled as ``"street"`` or ``"sottoportico"``.
    canals : geopandas.GeoDataFrame
        Polygons labeled as ``"water"``.
    courtyards : geopandas.GeoDataFrame
        Polygons labeled as ``"courtyard"``.

    Notes
    -----
    - If input file is Feather, `geopandas.read_feather` is used.
      Otherwise `geopandas.read_file` is used.
    - Buildings that are stored as MultiPolygons are exploded into single
      polygons for cleaner rasterization.
    - The function prints the path if it is not a Feather file (for debugging).

    Examples
    --------
    >>> from urbanflow.utils.sommarioni_utils import load_sommarioni_layers
    >>> buildings, streets, canals, courtyards = load_sommarioni_layers("sommarioni.geojson")
    >>> buildings.crs
    <Derived CRS: EPSG:32633>
    Name: WGS 84 / UTM zone 33N
    Axis Info [cartesian]:
    - E[east]: Easting (metre)
    - N[north]: Northing (metre)
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