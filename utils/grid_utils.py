"""
grid_utils.py

Utilities for combining and reading GeoPandas files, or applying operations on the grid
"""

import geopandas as gpd
import numpy as np
from affine import Affine
from typing import Tuple, Iterable
from rasterio import features
import math


def instantiate_grid_and_transform(
    cell_size: float, gdfs: gpd.GeoDataFrame, default_grid_value : int = 0
) -> Tuple[np.typing.NDArray[np.uint8], Affine]:
    # Find union bounds
    min_x, max_x, min_y, max_y = _union_bounds(*gdfs)

    # Snap extent to whole cells
    min_x, max_x, min_y, max_y = _snap_extent(cell_size, min_x, max_x, min_y, max_y)

    # Find number of cols and rows
    cols = int((max_x - min_x) / cell_size)
    rows = int((max_y - min_y) / cell_size)

    # Calculate Affine transform
    transform = Affine(cell_size, 0, min_x, 0, -cell_size, max_y)

    # Instantiate Grid
    grid = np.full((rows, cols), dtype=np.uint8, fill_value=default_grid_value)

    return grid, transform


def _union_bounds(*gdfs: gpd.GeoDataFrame) -> Tuple[float, float, float, float]:
    """Return combined bounding box of several GeoDataFrames."""
    min_x, min_y, max_x, max_y = gdfs[0].total_bounds
    for gdf in gdfs[1:]:
        x0, y0, x1, y1 = gdf.total_bounds
        min_x, min_y = min(min_x, x0), min(min_y, y0)
        max_x, max_y = max(max_x, x1), max(max_y, y1)
    return min_x, max_x, min_y, max_y


def _snap_extent(
    cell_size: float, min_x: float, max_x: float, min_y: float, max_y: float
):
    """Snap the extent of the boundingbox of the grid to the cell size"""
    min_x = math.floor(min_x / cell_size) * cell_size
    min_y = math.floor(min_y / cell_size) * cell_size
    max_x = math.ceil(max_x / cell_size) * cell_size
    max_y = math.ceil(max_y / cell_size) * cell_size

    return min_x, max_x, min_y, max_y


def rasterize_geoms(
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


def cell_to_polygon(row: int, col: int, transform: Affine):
    """
    Return the Shapely polygon representing a single grid cell.
    Useful for plotting/debugging.
    """
    x_min, y_max = transform * (col, row)
    x_max, y_min = transform * (col + 1, row + 1)
    return box(x_min, y_min, x_max, y_max)