import numpy as np
import geopandas as gpd
from shapely.geometry import Polygon, Point
from affine import Affine

import pytest

from urbanflow.utils.courtyard_utils import add_auto_courtyards
from urbanflow.utils.grid_utils import (
    instantiate_grid_and_transform,
    _union_bounds,
    _snap_extent,
    rasterize_geoms,
    cell_to_polygon,
)
from urbanflow.utils.poi_utils import (
    pois_to_grid_coords,
    point_to_cell,
    pois_within_radius,
)

def test_add_auto_courtyards():
    """
    Test automatic courtyard filling in a raster grid.

    - Construct a grid with buildings (2) enclosing a single empty cell (0).
    - Call add_auto_courtyards, which should convert enclosed empty cells
      into courtyards (value 4).
    - Assert that the central cell (2,2) becomes a courtyard.
    - Assert that border cells remain unchanged.
    """
    grid = np.array([
        [0,0,0,0,0],
        [0,2,2,2,0],
        [0,2,0,2,0],
        [0,2,2,2,0],
        [0,0,0,0,0],
    ], dtype=np.uint8)
    res = add_auto_courtyards(grid.copy())
    assert res[2,2] == 4
    # border cells remain unchanged
    assert res[0,0] == 0

def test_grid_utils_functions():
    """
    Test core utility functions for grid instantiation, snapping, rasterization,
    and cell-to-polygon conversion.

    - Create two adjacent polygons and wrap them in GeoDataFrames.
    - instantiate_grid_and_transform should produce a 2x2 grid and affine transform.
    - _union_bounds should return the combined bounding box of both geoms.
    - _snap_extent should align floating bounds to integer grid boundaries.
    - rasterize_geoms should burn a polygon into the grid with a given value.
    - cell_to_polygon should reconstruct a polygon from a row/col index and transform.
    """
    poly1 = Polygon([(0,0),(1,0),(1,1),(0,1)])
    poly2 = Polygon([(1,1),(2,1),(2,2),(1,2)])
    gdf1 = gpd.GeoDataFrame({'geometry':[poly1]}, crs='EPSG:32633')
    gdf2 = gpd.GeoDataFrame({'geometry':[poly2]}, crs='EPSG:32633')

    grid, transform = instantiate_grid_and_transform(1, [gdf1, gdf2])
    assert grid.shape == (2,2)
    assert isinstance(transform, Affine)

    ub = _union_bounds(gdf1, gdf2)
    assert ub == (0.0, 0.0, 2.0, 2.0)

    snap = _snap_extent(1, 0.2,1.7,0.3,1.9)
    assert snap == (0,2,0,2)

    rasterize_geoms([poly1], 3, grid, transform)
    assert grid[1,0] == 3

    poly = cell_to_polygon(0,0,transform)
    assert poly.bounds == (0.0,1.0,1.0,2.0)

def make_basic_grid():
    grid = np.array([
        [1,1,1],
        [1,2,1],
        [1,1,1],
    ], dtype=np.uint8)
    transform = Affine(1,0,0,0,-1,3)
    return grid, transform

def test_pois_to_grid_coords_and_point_to_cell():
    """
    Test conversion of POIs to grid coordinates and single-point cell lookup.

    - Construct a 3x3 grid with a single building cell at the center.
    - pois_to_grid_coords should add row/col and adjusted row/col to POI GeoDataFrame.
    - point_to_cell should map coordinates (1.5,1.5) to the central cell (1,1).
    - Adjusted coordinates should map to a neighboring cell since the center is a building.
    - In a grid with only buildings (no doors/streets), adjusted coords should equal raw.
    """
    grid, transform = make_basic_grid()
    gdf = gpd.GeoDataFrame({
        'uid':['A'],
        'geometry':[Point(1.5,1.5)]
    }, crs='EPSG:32633')

    res = pois_to_grid_coords(gdf, transform, grid)
    assert {'row','col','row_adj','col_adj'} <= set(res.columns)
    r,c = point_to_cell(1.5,1.5, transform)
    assert (r,c) == (1,1)
    assert (res.loc[0,'row'], res.loc[0,'col']) == (1,1)
    assert (res.loc[0,'row_adj'], res.loc[0,'col_adj']) in {(1,0),(0,1),(1,2),(2,1)}

    # no door cells -> adjusted equals raw
    grid_full = np.full((2,2), 2, np.uint8)
    gdf2 = gpd.GeoDataFrame({'geometry':[Point(0.5,1.5)]}, crs='EPSG:32633')
    res2 = pois_to_grid_coords(gdf2, Affine(1,0,0,0,-1,2), grid_full)
    assert res2.loc[0,'row_adj'] == res2.loc[0,'row']

def test_pois_to_grid_coords_options():
    """
    Test pois_to_grid_coords behavior with options.

    - Pass a POI located outside the grid bounds.
    - With keep_outside=True and do_adjusted=False:
        * Function should return row/col indices outside the grid shape.
        * No row_adj/col_adj columns should be created.
    """
    grid, transform = make_basic_grid()
    gdf = gpd.GeoDataFrame({'geometry':[Point(-1, -1)]}, crs='EPSG:32633')
    res = pois_to_grid_coords(gdf, transform, grid, keep_outside=True, do_adjusted=False)
    assert 'row_adj' not in res.columns
    assert res.loc[0,"row"] >= grid.shape[0]


def test_point_to_cell_outside():
    """
    Test error handling when a point lies outside the raster extent.

    - Given a 3x3 grid and its transform, attempt to convert (0,4) into a cell.
    - Expect a ValueError to be raised since y=4 lies outside the affine mapping.
    """
    grid, transform = make_basic_grid()
    with pytest.raises(ValueError):
        point_to_cell(0, 4, transform)

def test_pois_within_radius(tmp_path):
    """
    Test nearest-POI search within a radius.

    - Build a GeoDataFrame with three POIs across two parishes, with categories X and Y.
    - Query pois_within_radius for POIs of category X around index 0 (POI A):
        * With prefer_same_parish=True, POI A and C are returned with A first
          (same parish preferred over distance).
    - If CRS is missing, function should raise ValueError.
    - If CRS is geographic (EPSG:4326), function should still work by reprojecting,
      and return both candidate POIs within the distance threshold.
    """
    gdf = gpd.GeoDataFrame({
        'uid':['A','B','C'],
        'PP_Bottega_METACATEGORY':['X','Y','X'],
        'parish_std':['p1','p1','p2'],
        'geometry':[Point(0,0),Point(1,0),Point(2,0)]
    }, crs='EPSG:32633')

    res = pois_within_radius(gdf, 0, 'X', 1500, prefer_same_parish=True)
    assert list(res['uid']) == ['A','C']  # same parish first then nearest

    gdf_nocrs = gdf.copy()
    gdf_nocrs.crs = None
    with pytest.raises(ValueError):
        pois_within_radius(gdf_nocrs, 0, 'X', 10)

    # non metric crs
    gdf_wgs = gdf.to_crs('EPSG:4326')
    res2 = pois_within_radius(gdf_wgs, 0, 'X', 1500)
    assert len(res2) == 2
