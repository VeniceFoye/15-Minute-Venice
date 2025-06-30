# sommarioni_layers.py
"""
End-to-end utilities for 1808 Venice land‐register data.

Dependencies
------------
geopandas, rasterio, shapely, numpy
plus the earlier grid helpers:
    • gdfs_to_grid (gridify.py)
    • add_courtyards_fast (optional, gridify.py)

State codes used in the resulting grid
--------------------------------------
0 ocean/background
1 street
2 building (incl. sottoportico)
3 canal / lagoon
4 courtyard (manual polygons first, algorithmic fill-ins optional)
"""

from __future__ import annotations
import geopandas as gpd
import numpy as np
from affine import Affine
from rasterio import features

from gridify_old import gdfs_to_grid, add_courtyards_fast


def load_sommarioni_layers(
    json_path: str,
    target_crs: int | str = 32633,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame,
           gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """
    Read the 1808 GeoJSON and return four GeoDataFrames:
        buildings_gdf, streets_gdf, canals_gdf, courtyards_gdf
    All re-projected to *target_crs* (EPSG:3857 metres by default).
    """
    gdf = gpd.read_file(json_path)
    if target_crs:
        gdf = gdf.to_crs(target_crs)

    buildings = gdf[gdf["geometry_type"] == 'building'].copy()
    streets   = gdf[gdf["geometry_type"].isin(["street", "sottoportico"])].copy()
    canals    = gdf[gdf["geometry_type"] == "water"].copy()
    courtyards = gdf[gdf["geometry_type"] == "courtyard"].copy()

    # occasionally sottoportici are MultiPolygons: explode for cleaner raster
    buildings = buildings.explode(index_parts=False)

    return buildings, streets, canals, courtyards


def venice_grid(
    json_path: str,
    cell_size: float = 1.0,
    target_crs: int | str = 32633,
    *,
    include_auto_courtyards: bool = True,
):
    """
    Build a uint8 grid from the 1808 land-register GeoJSON.

    Returns
    -------
    grid : np.ndarray  (rows, cols)
    transform : affine.Affine
    legend : dict[int, str]
    layers : dict[str, gpd.GeoDataFrame]
        {"buildings":…, "streets":…, "canals":…, "courtyards":…}
    """
    buildings, streets, canals, courtyards = load_sommarioni_layers(
        json_path, target_crs
    )

    grid, transform, legend = gdfs_to_grid(
        buildings, streets, canals=canals, cell_size=cell_size
    )

    # --- manual courtyards (parcel_type == 'courtyard') ---------------------
    _burn_polygons_to_grid(
        courtyards.geometry, grid, transform, value=4
    )
    legend[4] = "courtyard"

    # --- optional algorithmic flood-fill of *remaining* interior empties ----
    if include_auto_courtyards:
        add_courtyards_fast(grid)        # only touches empty pockets

    layers = {
        "buildings": buildings,
        "streets": streets,
        "canals": canals,
        "courtyards": courtyards,
    }
    return grid, transform, legend, layers


# ----- internal utility -----------------------------------------------------

def _burn_polygons_to_grid(
    geoms,
    grid: np.ndarray,
    transform: Affine,
    *,
    value: int,
):
    """
    Rasterise a set of polygons INTO an existing uint8 grid (in-place).
    """
    shapes = ((geom, value) for geom in geoms if not geom.is_empty)
    features.rasterize(
        shapes=shapes,
        out=grid,
        transform=transform,
        all_touched=True,
        default_value=value,
    )
