import pytest
import numpy as np
import geopandas as gpd
from shapely.geometry import Polygon
from affine import Affine
from PIL import Image

from urbanflow.RasterGrid import RasterGrid


def make_simple_rastergrid():
    grid = np.array([[0,1],[2,3]], dtype=np.uint8)
    rg = RasterGrid()
    rg.grid = grid
    rg.transform = Affine(1,0,0,0,-1,2)
    rg.legend = {"ocean":0,"street":1,"building":2,"canal":3,"courtyard":4}
    return rg

def test_save_and_load(tmp_path):
    """
    Test that a RasterGrid can be saved to disk and reloaded without data loss.

    - Create a simple 2x2 raster grid with legend and affine transform.
    - Save it to a temporary .npz file and reload using RasterGrid.load.
    - Assert that:
        * grid values are identical,
        * affine transform is preserved,
        * legend is preserved,
        * cell size and CRS are preserved.
    """
    rg = make_simple_rastergrid()
    path = tmp_path / "grid.npz"
    rg.save(path)
    rg2 = RasterGrid.load(path)
    assert np.array_equal(rg.grid, rg2.grid)
    assert rg.transform == rg2.transform
    assert rg.legend == rg2.legend
    assert rg.cell_size == rg2.cell_size
    assert rg.coordinate_reference_system == rg2.coordinate_reference_system

def test_to_image_errors_and_scaling():
    """
    Test image rendering from a RasterGrid.

    - Calling to_image with scale=0 should raise ValueError.
    - Rendering with scale=2 and a minimal palette should return a PIL.Image.
    - Assert that the image size matches the scaled grid dimensions (4x4 pixels).
    """
    rg = make_simple_rastergrid()
    with pytest.raises(ValueError):
        rg.to_image(scale=0)
    img = rg.to_image(scale=2, palette={0:(0,0,0)})
    assert isinstance(img, Image.Image)
    assert img.size == (4,4)

def test_from_geojson_dataframes():
    """
    Test building a RasterGrid from GeoDataFrames of different feature types.

    - Construct small polygons representing buildings, streets, canals, and a courtyard.
    - Call RasterGrid.from_geojson_dataframes with auto_courtyards disabled.
    - Assert that the resulting grid has the correct shape.
    - Assert that all raster values are contained within the legend values.
    """
    buildings = gpd.GeoDataFrame({'geometry':[Polygon([(0,0),(1,0),(1,1),(0,1)])]}, crs='EPSG:32633')
    streets = gpd.GeoDataFrame({'geometry':[Polygon([(1,0),(2,0),(2,1),(1,1)])]}, crs='EPSG:32633')
    canals = gpd.GeoDataFrame({'geometry':[Polygon([(0,1),(1,1),(1,2),(0,2)])]}, crs='EPSG:32633')
    courtyards = gpd.GeoDataFrame({'geometry':[Polygon([(0.2,0.2),(0.8,0.2),(0.8,0.8),(0.2,0.8)])]}, crs='EPSG:32633')
    rg = RasterGrid.from_geojson_dataframes(buildings, streets, canals, courtyards=courtyards, auto_courtyards=False)
    assert rg.grid.shape == (2,2)
    assert set(np.unique(rg.grid)) <= set(rg.legend.values())


def test_rastergrid_crs_conversion():
    """Test RasterGrid CRS conversion warning and code path."""
    # Create test data in different CRS
    buildings = gpd.GeoDataFrame(
        {'geometry': [Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])]}, 
        crs='EPSG:4326'  # WGS84, different from target
    )
    streets = gpd.GeoDataFrame(
        {'geometry': [Polygon([(1, 0), (2, 0), (2, 1), (1, 1)])]}, 
        crs='EPSG:4326'
    )
    canals = gpd.GeoDataFrame(
        {'geometry': [Polygon([(0, 1), (1, 1), (1, 2), (0, 2)])]}, 
        crs='EPSG:4326'
    )
    
    # This should trigger the CRS conversion warning and code path (lines 103-104)
    rg = RasterGrid.from_geojson_dataframes(
        buildings, streets, canals, 
        coordinate_reference_system='EPSG:32633',  # UTM zone 33N
        cell_size=100,  # larger cell size for lat/lon to UTM conversion
        auto_courtyards=False
    )
    
    assert rg.grid.shape[0] > 0
    assert rg.coordinate_reference_system == 'EPSG:32633'


def test_rastergrid_no_courtyards_message():
    """Test RasterGrid logging when no courtyards are provided."""
    buildings = gpd.GeoDataFrame(
        {'geometry': [Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])]}, 
        crs='EPSG:32633'
    )
    streets = gpd.GeoDataFrame(
        {'geometry': [Polygon([(1, 0), (2, 0), (2, 1), (1, 1)])]}, 
        crs='EPSG:32633'
    )
    canals = gpd.GeoDataFrame(
        {'geometry': [Polygon([(0, 1), (1, 1), (1, 2), (0, 2)])]}, 
        crs='EPSG:32633'
    )
    
    # This should trigger line 121: "No manual courtyards found."
    rg = RasterGrid.from_geojson_dataframes(
        buildings, streets, canals, 
        courtyards=None,  # Explicitly no courtyards
        auto_courtyards=False,
        cell_size=1
    )
    
    assert rg.grid.shape == (2, 2)


def test_rastergrid_auto_courtyards():
    """Test RasterGrid with auto_courtyards enabled."""
    buildings = gpd.GeoDataFrame(
        {'geometry': [Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])]}, 
        crs='EPSG:32633'
    )
    streets = gpd.GeoDataFrame(
        {'geometry': [Polygon([(1, 0), (2, 0), (2, 1), (1, 1)])]}, 
        crs='EPSG:32633'
    )
    canals = gpd.GeoDataFrame(
        {'geometry': [Polygon([(0, 1), (1, 1), (1, 2), (0, 2)])]}, 
        crs='EPSG:32633'
    )
    
    # This should trigger lines 124-125: auto courtyard logging and processing
    rg = RasterGrid.from_geojson_dataframes(
        buildings, streets, canals, 
        courtyards=None,
        auto_courtyards=True,  # Enable auto courtyards
        cell_size=1
    )
    
    assert rg.grid.shape == (2, 2)

