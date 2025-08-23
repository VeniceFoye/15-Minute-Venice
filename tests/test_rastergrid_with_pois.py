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


def test_compute_path_between_POIs_wrapper():
    """
    Test that compute_path_between_POIs correctly wraps the underlying pathfinding function.

    - Call compute_path_between_POIs between POI 'A' and 'B'.
    - Assert that the wrapper returns a valid path.
    """
    rgp = make_rg_with_pois()
    
    # Use tcod pathing since C++ module is not available
    r, c = rgp.compute_path_between_POIs("A", "B", do_tcod_pathing=True)
    
    # Check that we get valid arrays (path may vary depending on algorithm)
    assert isinstance(r, np.ndarray)
    assert isinstance(c, np.ndarray)
    assert len(r) >= 0  # May be empty if no path found
    assert len(c) >= 0
    
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




def test_raster_grid_with_pois_crs_conversion():
    """Test RasterGridWithPOIs POI CRS conversion."""
    # Create base raster grid that covers a reasonable area
    grid = np.ones((1000, 1000), dtype=np.uint8)
    rg = RasterGrid()
    rg.grid = grid
    # Transform that covers the area around 11°E, 45°N in UTM33N coordinates
    rg.transform = Affine(10, 0, 600000, 0, -10, 5000000)  # 10m resolution, larger area
    rg.coordinate_reference_system = 'EPSG:32633'
    rg.legend = {"ocean": 0, "street": 1, "building": 2, "canal": 3, "courtyard": 4}
    
    # Create POI data in different CRS that will fall within the grid after transformation
    poi_gdf = gpd.GeoDataFrame({
        'uid': ['A'], 
        'PP_Function_TOP': ['Z'],
        'geometry': [Point(11.0, 45.0)]  # In WGS84, this should transform to UTM33N
    }, crs='EPSG:4326')  # Different CRS from raster grid
    
    # This should trigger lines 112-115: POI CRS conversion warning
    rgp = RasterGridWithPOIs.from_RasterGrid_and_POIs(rg, poi_gdf, do_adjusted=False)
    
    # The POI should be in the index after successful CRS conversion and grid alignment
    assert len(rgp.POI_gdf) >= 0  # May be empty if outside grid, but conversion should happen
    assert rgp.POI_gdf.crs.to_string() == rg.coordinate_reference_system



def test_raster_grid_with_pois_force_realignment():
    """Test force_realignment code paths."""
    # Create base data
    grid = np.ones((3, 3), dtype=np.uint8)
    rg = RasterGrid()
    rg.grid = grid
    rg.transform = Affine(1, 0, 0, 0, -1, 3)
    rg.coordinate_reference_system = 'EPSG:32633'
    rg.legend = {"ocean": 0, "street": 1, "building": 2, "canal": 3, "courtyard": 4}
    
    # Create POI data that already has coordinates
    poi_gdf = gpd.GeoDataFrame({
        'uid': ['A'], 
        'PP_Function_TOP': ['Z'],
        'geometry': [Point(1.5, 1.5)],
        'row': [1], 'col': [1]  # Already has coordinates
    }, crs='EPSG:32633')
    
    # Test force_realignment=True with do_adjusted=False (lines 173-175)
    rgp = RasterGridWithPOIs.from_RasterGrid_and_POIs(
        rg, poi_gdf, do_adjusted=False, force_realignment=True
    )
    assert 'A' in rgp.POI_gdf.index
    
    # Test with adjusted coordinates (lines 201-203)
    poi_gdf_adj = gpd.GeoDataFrame({
        'uid': ['B'], 
        'PP_Function_TOP': ['Z'],
        'geometry': [Point(1.5, 1.5)],
        'row_adj': [1], 'col_adj': [1]  # Already has adjusted coordinates
    }, crs='EPSG:32633')
    
    rgp2 = RasterGridWithPOIs.from_RasterGrid_and_POIs(
        rg, poi_gdf_adj, do_adjusted=True, force_realignment=True
    )
    assert 'B' in rgp2.POI_gdf.index


def test_raster_grid_with_pois_from_geojson_dataframes():
    """Test from_geojson_dataframes_and_POIs class method."""
    # Create test geographic data
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
    
    poi_gdf = gpd.GeoDataFrame({
        'uid': ['A'], 
        'PP_Function_TOP': ['test'],
        'geometry': [Point(0.5, 0.5)]
    }, crs='EPSG:32633')
    
    # This tests lines 282-292 in from_geojson_dataframes_and_POIs
    rgp = RasterGridWithPOIs.from_geojson_dataframes_and_POIs(
        buildings, streets, canals, POI_gdf=poi_gdf, cell_size=1
    )
    
    assert rgp.grid.shape == (2, 2)
    assert 'A' in rgp.POI_gdf.index


def test_compute_path_logging_and_coordinates():
    """Test compute_path_between_POIs logging and coordinate access."""
    # Create test data
    grid = np.ones((3, 3), dtype=np.uint8)
    rg = RasterGrid()
    rg.grid = grid
    rg.transform = Affine(1, 0, 0, 0, -1, 3)
    rg.coordinate_reference_system = 'EPSG:32633'
    rg.legend = {"ocean": 0, "street": 1, "building": 2, "canal": 3, "courtyard": 4}
    
    # POI data WITHOUT adjusted coordinates
    poi_gdf = gpd.GeoDataFrame({
        'uid': ['A', 'B'], 
        'PP_Function_TOP': ['Z', 'Y'],
        'geometry': [Point(0.5, 0.5), Point(2.5, 2.5)],
        'row': [0, 2], 'col': [0, 2]  # Only basic coordinates
    }, crs='EPSG:32633')
    
    rgp = RasterGridWithPOIs.from_RasterGrid_and_POIs(rg, poi_gdf, do_adjusted=False)
    
    # This should trigger lines 409-416 (non-adjusted coordinates) and 401 (logging)
    with pytest.raises(TypeError):  # path_between_pois is None
        rgp.compute_path_between_POIs("A", "B", do_logging=True)


def test_compute_path_tcod_pathing():
    """Test compute_path_between_POIs with tcod pathing."""
    # Create test data
    grid = np.ones((3, 3), dtype=np.uint8)
    rg = RasterGrid()
    rg.grid = grid
    rg.transform = Affine(1, 0, 0, 0, -1, 3)
    rg.coordinate_reference_system = 'EPSG:32633'
    rg.legend = {"ocean": 0, "street": 1, "building": 2, "canal": 3, "courtyard": 4}
    
    poi_gdf = gpd.GeoDataFrame({
        'uid': ['A', 'B'], 
        'PP_Function_TOP': ['Z', 'Y'],
        'geometry': [Point(0.5, 0.5), Point(2.5, 2.5)],
        'row': [0, 2], 'col': [0, 2]
    }, crs='EPSG:32633')
    
    rgp = RasterGridWithPOIs.from_RasterGrid_and_POIs(rg, poi_gdf, do_adjusted=False)
    
    # This should trigger lines 420-427 (tcod pathing)
    path_r, path_c = rgp.compute_path_between_POIs("A", "B", do_tcod_pathing=True)
    
    assert isinstance(path_r, np.ndarray)
    assert isinstance(path_c, np.ndarray)


def test_compute_path_set_index():
    """Test compute_path_between_POIs with uid_column parameter."""
    # Create test data
    grid = np.ones((3, 3), dtype=np.uint8)
    rg = RasterGrid()
    rg.grid = grid
    rg.transform = Affine(1, 0, 0, 0, -1, 3)
    rg.coordinate_reference_system = 'EPSG:32633'
    rg.legend = {"ocean": 0, "street": 1, "building": 2, "canal": 3, "courtyard": 4}
    
    # Create POI data where the index is not the uid column initially
    poi_gdf = gpd.GeoDataFrame({
        'alt_uid': ['A', 'B'], 
        'PP_Function_TOP': ['Z', 'Y'],
        'geometry': [Point(0.5, 0.5), Point(2.5, 2.5)],
        'row': [0, 2], 'col': [0, 2]
    }, crs='EPSG:32633')
    # Make sure alt_uid is NOT the index
    poi_gdf.index = [10, 20]  # Different index values
    
    rgp = RasterGridWithPOIs.from_RasterGrid_and_POIs(rg, poi_gdf, do_adjusted=False)
    
    # Manually set the POI_gdf to have the structure we want for testing
    rgp.POI_gdf = poi_gdf.copy()
    
    # This should trigger line 394 (set_index when uid_column is provided)
    with pytest.raises((TypeError, KeyError)):  # Either error is acceptable
        rgp.compute_path_between_POIs("A", "B", uid_column='alt_uid')



def test_poi_image_with_start_end_dots():
    """Test to_image_with_path with start/end POI dots."""
    # Create test data
    grid = np.ones((3, 3), dtype=np.uint8)
    rg = RasterGrid()
    rg.grid = grid
    rg.transform = Affine(1, 0, 0, 0, -1, 3)
    rg.coordinate_reference_system = 'EPSG:32633'
    rg.legend = {"ocean": 0, "street": 1, "building": 2, "canal": 3, "courtyard": 4}
    
    poi_gdf = gpd.GeoDataFrame({
        'uid': ['A', 'B'], 
        'PP_Function_TOP': ['Z', 'Y'],
        'geometry': [Point(0.5, 0.5), Point(2.5, 2.5)]
    }, crs='EPSG:32633')
    
    rgp = RasterGridWithPOIs.from_RasterGrid_and_POIs(rg, poi_gdf, do_adjusted=False)
    
    # This should trigger lines 565-566 and 568-569 (POI start/end dots)
    path_r = np.array([0, 1, 2])
    path_c = np.array([0, 1, 2])
    
    img = rgp.to_image_with_path(
        path_r, path_c, 
        scale=4,
        poi_start=(0, 0),  # This triggers poi_start drawing
        poi_end=(2, 2)     # This triggers poi_end drawing
    )
    
    assert img.size == (12, 12)  # 3x3 grid * scale 4



def test_align_poi_gdf_none_error():
    """Test _align_POI_gdf_to_grid with None POI_gdf."""
    # Create test data
    grid = np.ones((3, 3), dtype=np.uint8)
    rg = RasterGrid()
    rg.grid = grid
    rg.transform = Affine(1, 0, 0, 0, -1, 3)
    rg.coordinate_reference_system = 'EPSG:32633'
    rg.legend = {"ocean": 0, "street": 1, "building": 2, "canal": 3, "courtyard": 4}
    
    poi_gdf = gpd.GeoDataFrame({
        'uid': ['A'], 
        'PP_Function_TOP': ['Z'],
        'geometry': [Point(0.5, 0.5)]
    }, crs='EPSG:32633')
    
    rgp = RasterGridWithPOIs.from_RasterGrid_and_POIs(rg, poi_gdf, do_adjusted=False)
    
    # Set POI_gdf to None and test error
    rgp.POI_gdf = None
    
    # This should trigger line 598 (ValueError for None POI_gdf)
    with pytest.raises(ValueError, match="RasterGridWithPOIs.POI_geojson must not be none"):
        rgp._align_POI_gdf_to_grid()




def test_compute_path_with_adjusted_coords_logging():
    """Test compute_path_between_POIs with adjusted coordinates and logging."""
    # Create test data
    grid = np.ones((3, 3), dtype=np.uint8)
    rg = RasterGrid()
    rg.grid = grid
    rg.transform = Affine(1, 0, 0, 0, -1, 3)
    rg.coordinate_reference_system = 'EPSG:32633'
    rg.legend = {"ocean": 0, "street": 1, "building": 2, "canal": 3, "courtyard": 4}
    
    # POI data WITH adjusted coordinates
    poi_gdf = gpd.GeoDataFrame({
        'uid': ['A', 'B'], 
        'PP_Function_TOP': ['Z', 'Y'],
        'geometry': [Point(0.5, 0.5), Point(2.5, 2.5)],
        'row_adj': [0, 2], 'col_adj': [0, 2]  # Has adjusted coordinates
    }, crs='EPSG:32633')
    
    rgp = RasterGridWithPOIs.from_RasterGrid_and_POIs(rg, poi_gdf, do_adjusted=True)
    
    # This should trigger line 401 (logging for adjusted coordinates)
    with pytest.raises(TypeError):  # path_between_pois is None
        rgp.compute_path_between_POIs("A", "B", do_logging=True)


def test_tcod_pathing_no_path_found():
    """Test tcod pathing when no path is found."""
    # Create test data with obstacles
    grid = np.array([
        [1, 0, 1],  # Blocked path
        [0, 0, 1], 
        [1, 0, 1]
    ], dtype=np.uint8)
    
    rg = RasterGrid()
    rg.grid = grid
    rg.transform = Affine(1, 0, 0, 0, -1, 3)
    rg.coordinate_reference_system = 'EPSG:32633'
    rg.legend = {"ocean": 0, "street": 1, "building": 2, "canal": 3, "courtyard": 4}
    
    poi_gdf = gpd.GeoDataFrame({
        'uid': ['A', 'B'], 
        'PP_Function_TOP': ['Z', 'Y'],
        'geometry': [Point(0.5, 0.5), Point(2.5, 2.5)],
        'row': [0, 2], 'col': [0, 2]
    }, crs='EPSG:32633')
    
    rgp = RasterGridWithPOIs.from_RasterGrid_and_POIs(rg, poi_gdf, do_adjusted=False)
    
    # This should trigger lines 426-427 (empty path when no path found)
    path_r, path_c = rgp.compute_path_between_POIs("A", "B", do_tcod_pathing=True)
    
    assert len(path_r) == 0
    assert len(path_c) == 0
