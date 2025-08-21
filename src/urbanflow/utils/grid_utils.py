"""
grid_utils.py

Utilities for combining and reading GeoPandas files, or applying operations on the grid
"""

import geopandas as gpd
import numpy as np
from shapely.geometry import box
from affine import Affine
from typing import Tuple, Iterable
from rasterio import features
import math


def instantiate_grid_and_transform(
    cell_size: float, gdfs: gpd.GeoDataFrame, default_grid_value : int = 0
) -> Tuple[np.typing.NDArray[np.uint8], Affine]:
    """
    Instantiate a raster grid and affine transform covering the extent of given geometries.

    Parameters
    ----------
    cell_size : float
        Resolution of grid cells in coordinate units (e.g., meters).
    gdfs : geopandas.GeoDataFrame or sequence of GeoDataFrame
        One or more GeoDataFrames whose geometries define the spatial extent of the grid.
    default_grid_value : int, optional
        Initial value used to fill all cells, by default ``0``.

    Returns
    -------
    grid : ndarray of uint8, shape (rows, cols)
        Raster grid array initialized with ``default_grid_value``.
    transform : affine.Affine
        Affine transform mapping raster indices (row, col) to spatial coordinates.

    Notes
    -----
    - The grid extent is snapped to whole multiples of ``cell_size`` so that the
      grid completely covers the union of input bounds.
    - The origin of the transform is the upper-left corner.
    """
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
    """
    Snap bounding box coordinates to a multiple of the grid cell size.

    Parameters
    ----------
    cell_size : float
        Size of a grid cell in coordinate units.
    min_x, max_x, min_y, max_y : float
        Original bounding box coordinates.

    Returns
    -------
    min_x, min_y, max_x, max_y : tuple of float
        Adjusted bounding box coordinates aligned to the grid cell size.

    Notes
    -----
    - The minimum bounds are floored to the nearest multiple of ``cell_size``.
    - The maximum bounds are ceiled to the nearest multiple of ``cell_size``.
    """
    min_x, min_y, max_x, max_y = gdfs[0].total_bounds
    for gdf in gdfs[1:]:
        x0, y0, x1, y1 = gdf.total_bounds
        min_x, min_y = min(min_x, x0), min(min_y, y0)
        max_x, max_y = max(max_x, x1), max(max_y, y1)
    return min_x, min_y, max_x, max_y


def _snap_extent(
    cell_size: float, min_x: float, max_x: float, min_y: float, max_y: float
):
    """
    Snap bounding box coordinates to a multiple of the grid cell size.

    Parameters
    ----------
    cell_size : float
        Size of a grid cell in coordinate units.
    min_x, max_x, min_y, max_y : float
        Original bounding box coordinates.

    Returns
    -------
    min_x, max_x, min_y, max_y : tuple of float
        Adjusted bounding box coordinates aligned to the grid cell size.

    Notes
    -----
    - The minimum bounds are floored to the nearest multiple of ``cell_size``.
    - The maximum bounds are ceiled to the nearest multiple of ``cell_size``.
    """
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
    """
    Rasterize geometries into an existing array (in-place).

    Parameters
    ----------
    geoms : iterable of shapely.geometry.BaseGeometry
        Geometries to rasterize.
    value : int
        Integer value to burn into cells touched by the geometries.
    out : ndarray
        Target raster array that will be modified in place.
    transform : affine.Affine
        Affine transform mapping raster indices (row, col) to spatial coordinates.

    Notes
    -----
    - Uses ``rasterio.features.rasterize`` under the hood.
    - A cell is marked if the geometry touches it at all (``all_touched=True``).
    """
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
    Convert a raster cell index to its polygon footprint in spatial coordinates.

    Parameters
    ----------
    row : int
        Row index of the cell.
    col : int
        Column index of the cell.
    transform : affine.Affine
        Affine transform mapping raster indices to spatial coordinates.

    Returns
    -------
    polygon : shapely.geometry.Polygon
        Polygon representing the spatial footprint of the cell.

    Examples
    --------
    >>> transform = Affine(1, 0, 0, 0, -1, 2)  # 1x1 cells
    >>> cell_to_polygon(0, 0, transform).bounds
    (0.0, 1.0, 1.0, 2.0)
    """
    x_min, y_max = transform * (col, row)
    x_max, y_min = transform * (col + 1, row + 1)
    return box(x_min, y_min, x_max, y_max)