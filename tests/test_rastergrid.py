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
    rg = make_simple_rastergrid()
    with pytest.raises(ValueError):
        rg.to_image(scale=0)
    img = rg.to_image(scale=2, palette={0:(0,0,0)})
    assert isinstance(img, Image.Image)
    assert img.size == (4,4)

def test_from_geojson_dataframes():
    buildings = gpd.GeoDataFrame({'geometry':[Polygon([(0,0),(1,0),(1,1),(0,1)])]}, crs='EPSG:32633')
    streets = gpd.GeoDataFrame({'geometry':[Polygon([(1,0),(2,0),(2,1),(1,1)])]}, crs='EPSG:32633')
    canals = gpd.GeoDataFrame({'geometry':[Polygon([(0,1),(1,1),(1,2),(0,2)])]}, crs='EPSG:32633')
    courtyards = gpd.GeoDataFrame({'geometry':[Polygon([(0.2,0.2),(0.8,0.2),(0.8,0.8),(0.2,0.8)])]}, crs='EPSG:32633')
    rg = RasterGrid.from_geojson_dataframes(buildings, streets, canals, courtyards=courtyards, auto_courtyards=False)
    assert rg.grid.shape == (2,2)
    assert set(np.unique(rg.grid)) <= set(rg.legend.values())

