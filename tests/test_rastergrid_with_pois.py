import numpy as np
import geopandas as gpd
from shapely.geometry import Point
from affine import Affine
from PIL import Image
import pytest

from urbanflow.RasterGrid import RasterGrid
from urbanflow.RasterGridWithPOIs import RasterGridWithPOIs


def make_rg_with_pois():
    """
    Helper function to construct a small RasterGridWithPOIs.

    - Creates a 3x3 grid filled with "street" cells (value 1).
    - Defines an affine transform mapping pixel indices to coordinates.
    - Attaches two POIs:
        A at (0.5, 0.5)
        B at (2.5, 2.5)
    - Returns the RasterGridWithPOIs instance.
    """
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
    """
    Test that RasterGridWithPOIs.save and .load correctly persist both the raster grid
    and the POI GeoDataFrame.

    - Monkeypatch parquet I/O to avoid actual file operations.
    - Save the object to a temporary path and reload it.
    - Assert that:
        * POI index values are preserved.
        * Raster grid array is identical after reload.
    """
    rgp = make_rg_with_pois()
    monkeypatch.setattr(gpd.GeoDataFrame, "to_parquet", lambda self,*a,**k: None)
    monkeypatch.setattr(gpd, "read_parquet", lambda f: rgp.POI_gdf.copy())
    path = tmp_path / "rgp"
    rgp.save(path)
    loaded = RasterGridWithPOIs.load(path)
    assert list(loaded.POI_gdf.index) == ['A','B']
    assert np.array_equal(loaded.grid, rgp.grid)


def test_compute_path_between_POIs_wrapper(monkeypatch):
    """
    Test that compute_path_between_POIs correctly wraps the underlying pathfinding function.

    - Monkeypatch path_between_pois with a fake implementation returning just start/end coords.
    - Call compute_path_between_POIs between POI 'A' and 'B'.
    - Assert that the wrapper returns exactly the fake path output.
    """
    rgp = make_rg_with_pois()

    def fake_path(grid, sr, sc, tr, tc):
        return np.array([sr,tr]), np.array([sc,tc])
    monkeypatch.setattr("urbanflow.RasterGridWithPOIs.path_between_pois", fake_path, raising=False)
    
    r, c = rgp.compute_path_between_POIs("A","B")
    assert r.tolist() == [1,0]
    assert c.tolist() == [1,2]
    
def test_to_image_with_POIs():
    """
    Test that the grid can be rendered to a PIL image including POIs.

    - Calls to_image_with_POIs with scale factor.
    - Asserts the result is a valid PIL.Image instance.
    """
    rgp = make_rg_with_pois()
    img1 = rgp.to_image_with_POIs(scale=2)
    assert isinstance(img1, Image.Image)

def test_to_image_with_path():
    """
    Test that a path overlay can be rendered onto a PIL image.

    - Constructs a simple diagonal path (0,0) → (2,2).
    - Calls to_image_with_path with scale factor.
    - Asserts the result is a valid PIL.Image instance.

    TODO: Validate the pixels of the image
    """
    rgp = make_rg_with_pois()
    path_r = np.array([0,1,2])
    path_c = np.array([0,1,2])
    img2 = rgp.to_image_with_path(path_r, path_c, scale=2)
    assert isinstance(img2, Image.Image)

def test_to_image_with_POIs_pixels():
    """
    Render a grid with POIs and assert that the pixels at each POI's center
    are exactly the colors we requested via `color_map`.

    Strategy:
    - Provide a known `palette` for the background and a `color_map` for POIs.
    - Use the POI_gdf's (row, col) to compute pixel centers with the same
      formula as the implementation: (c*scale + scale//2, r*scale + scale//2).
    - Assert those center pixels equal the expected RGB colors.
    """
    rgp = make_rg_with_pois()

    # Deterministic background + POI colors
    palette = {0: (0, 0, 0), 1: (0, 0, 0), 2: (0, 0, 0), 3: (0, 0, 0), 4: (0, 0, 0)}
    color_map = {"X": (10, 20, 30), "Y": (40, 50, 60)}  # uid A->X, B->Y in our helper

    scale = 4  # slightly bigger so the POI dot clearly covers the center
    img = rgp.to_image_with_POIs(scale=scale, palette=palette, color_map=color_map)

    px = img.load()
    for uid, row in rgp.POI_gdf.iterrows():
        r, c = int(row["row"]), int(row["col"])
        x = c * scale + scale // 2
        y = r * scale + scale // 2
        expected = color_map[str(row["PP_Function_TOP"])]
        assert px[x, y] == expected, f"Center pixel at ({x},{y}) should be {expected} for POI {uid}"

def test_to_image_with_path_pixels():
    """
    Render a grid with an overlaid path and assert that center pixels along the
    path are the requested `path_colour`.

    Strategy:
    - Provide a known black background palette.
    - Draw a simple path along cell centers.
    - Check those center pixels equal `path_colour` and that some background
      pixel remains unchanged.
    """
    rgp = make_rg_with_pois()

    # Black background for clear contrast
    palette = {0: (0, 0, 0), 1: (0, 0, 0), 2: (0, 0, 0), 3: (0, 0, 0), 4: (0, 0, 0)}

    # Path: diagonal through three cells
    path_r = np.array([0, 1, 2])
    path_c = np.array([0, 1, 2])

    scale = 4
    path_colour = (250, 0, 0)
    img = rgp.to_image_with_path(
        path_r,
        path_c,
        scale=scale,
        palette=palette,
        path_colour=path_colour,
        path_width=1,  # thin line centered on the cell centers
    )

    px = img.load()

    # Assert the center of each path cell is path-coloured
    for r, c in zip(path_r, path_c):
        x = int(c) * scale + scale // 2
        y = int(r) * scale + scale // 2
        assert px[x, y] == path_colour, f"Path center pixel at ({x},{y}) should be {path_colour}"

    # Also sanity-check a background pixel is still background-colored
    bg_expected = (0, 0, 0)
    # pick a pixel not on the path: cell (0,2) center if it's not part of the path
    x_bg = 2 * scale + scale // 2
    y_bg = 0 * scale + scale // 2
    if (0, 2) not in set(zip(path_r.tolist(), path_c.tolist())):
        assert px[x_bg, y_bg] == bg_expected, "Background pixel should remain unchanged"



def test_align_poi_gdf_to_grid():
    """
    Test that POIs are correctly aligned to the raster grid.

    - Creates a grid and a GeoDataFrame with one POI at (1.5, 1.5).
    - Builds a RasterGridWithPOIs without adjusted alignment.
    - Calls _align_POI_gdf_to_grid to compute row/col indices.
    - Asserts that:
        * 'row' and 'col' columns are added.
        * 'row_adj' and 'col_adj' columns are also present.
        * row == row_adj and col == col_adj for each POI (since no adjustment needed).
    """
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

    assert 'row_adj' in rgp.POI_gdf.columns
    assert 'col_adj' in rgp.POI_gdf.columns

    # Assert that for each row, row == row_adj and col == col_adj
    for _, row in rgp.POI_gdf.iterrows():
        assert row['row'] == row['row_adj']
        assert row['col'] == row['col_adj']
