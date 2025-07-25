import numpy as np
import geopandas as gpd
from shapely.geometry import Point
from affine import Affine
from PIL import Image
import pytest

from urbanflow.RasterGrid import RasterGrid
from urbanflow.RasterGridWithPOIs import RasterGridWithPOIs


def make_rg_with_pois():
    grid = np.array([[1,1,1],[1,1,1],[1,1,1]], dtype=np.uint8)
    rg = RasterGrid()
    rg.grid = grid
    rg.transform = Affine(1,0,0,0,-1,3)
    rg.legend = {"ocean":0,"street":1,"building":2,"canal":3,"courtyard":4}

    gdf = gpd.GeoDataFrame({
        'uid':['A','B'],
        'PP_Function_TOP':['X','Y'],
        'geometry':[Point(0.5,0.5), Point(2.5,2.5)]
    }, crs='EPSG:32633')

    return RasterGridWithPOIs.from_RasterGrid_and_POIs(rg, gdf)
def test_save_load(tmp_path, monkeypatch):
    rgp = make_rg_with_pois()
    monkeypatch.setattr(gpd.GeoDataFrame, "to_parquet", lambda self,*a,**k: None)
    monkeypatch.setattr(gpd, "read_parquet", lambda f: rgp.POI_gdf.copy())
    path = tmp_path / "rgp"
    rgp.save(path)
    loaded = RasterGridWithPOIs.load(path)
    assert list(loaded.POI_gdf.index) == ['A','B']
    assert np.array_equal(loaded.grid, rgp.grid)


def test_compute_path_between_POIs(monkeypatch):
    rgp = make_rg_with_pois()
    def fake_path(grid, sr, sc, tr, tc):
        return np.array([sr,tr]), np.array([sc,tc])
    monkeypatch.setattr("urbanflow.RasterGridWithPOIs.path_between_pois", fake_path, raising=False)
    r, c = rgp.compute_path_between_POIs("A","B")
    assert r.tolist() == [1,0,0,0]
    assert c.tolist() == [0,0,1,2]
def test_to_image_methods():
    rgp = make_rg_with_pois()
    img1 = rgp.to_image_with_POIs(scale=2)
    assert isinstance(img1, Image.Image)
    path_r = np.array([0,1,2])
    path_c = np.array([0,1,2])
    img2 = rgp.to_image_with_path(path_r, path_c, scale=2)
    assert isinstance(img2, Image.Image)

def test_align_poi_gdf_to_grid():
    grid = np.ones((3,3), dtype=np.uint8)
    rg = RasterGrid()
    rg.grid = grid
    rg.transform = Affine(1,0,0,0,-1,3)
    rg.legend = {"ocean":0,"street":1,"building":2,"canal":3,"courtyard":4}

    gdf = gpd.GeoDataFrame({'uid':['X'],'PP_Function_TOP':['Z'],'geometry':[Point(1.5,1.5)]}, crs='EPSG:32633')
    rgp = RasterGridWithPOIs.from_RasterGrid_and_POIs(rg, gdf, do_adjusted=False)
    rgp.POI_gdf = gdf.copy()
    rgp._align_POI_gdf_to_grid()
    assert 'row' in rgp.POI_gdf.columns
    assert 'col' in rgp.POI_gdf.columns
