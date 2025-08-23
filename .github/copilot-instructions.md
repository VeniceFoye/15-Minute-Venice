# Urbanflow Development Instructions

Urbanflow is a Python package for investigating urban morphology using multi-agent simulations. It processes and rasterizes GeoDataFrames, provides visualization capabilities, and includes a C++ extension for pathfinding between points-of-interest (POIs).

**Always reference these instructions first and fallback to search or bash commands only when you encounter unexpected information that does not match the info here.**

## Working Effectively

### Bootstrap and Setup

**CRITICAL**: Network connectivity issues are common in development environments. Use these exact steps:

1. **Install dependencies via conda (preferred method):**
   ```bash
   conda install -c conda-forge numpy pandas matplotlib geopandas shapely affine pillow scipy rasterio pytest pytest-cov ruff mypy ipython -y
   ```
   
2. **If conda fails, install core dependencies via pip:**
   ```bash
   pip install affine shapely matplotlib pandas geopandas pillow scipy rasterio tcod pyarrow --timeout 300
   ```

3. **Install documentation dependencies:**
   ```bash
   pip install sphinx furo myst-parser --timeout 300
   ```

### Building the Package

**Without C++ Extension (Python-only mode):**
```bash
# Set PYTHONPATH to include source directory
export PYTHONPATH=/path/to/Urbanflow/src:$PYTHONPATH

# Test basic import
python -c "import urbanflow; print('Import successful')"
```

**With C++ Extension (Full functionality):**
```bash
# Install build dependencies first
pip install "scikit-build-core[pyproject]>=0.10" "pybind11>=2.12" cmake ninja --timeout 300

# Set build environment variables
export CMAKE_BUILD_PARALLEL_LEVEL=2
export SKBUILD_CMAKE_ARGS="-DCMAKE_BUILD_TYPE=Release"

# Build and install in development mode
# NEVER CANCEL: Build takes 5-10 minutes. Set timeout to 15+ minutes.
pip install -e .[dev] --timeout 900
```

**Important**: The C++ extension may fail to build due to network issues downloading dependencies. The package works in Python-only mode with reduced functionality (no pathfinding between POIs).

### Testing

**Run basic functionality tests:**
```bash
cd /path/to/Urbanflow
PYTHONPATH=/path/to/Urbanflow/src python -c "
import urbanflow
import numpy as np
from shapely.geometry import Polygon
import geopandas as gpd

# Create test data
buildings = gpd.GeoDataFrame({'geometry':[Polygon([(0,0),(1,0),(1,1),(0,1)])]}, crs='EPSG:32633')
streets = gpd.GeoDataFrame({'geometry':[Polygon([(1,0),(2,0),(2,1),(1,1)])]}, crs='EPSG:32633')
canals = gpd.GeoDataFrame({'geometry':[Polygon([(0,1),(1,1),(1,2),(0,2)])]}, crs='EPSG:32633')

# Test RasterGrid creation
rg = urbanflow.RasterGrid.from_geojson_dataframes(
    buildings=buildings, 
    streets=streets,
    canals=canals,
    cell_size=1,
    coordinate_reference_system='EPSG:32633'
)
print('RasterGrid created successfully')
print('Grid shape:', rg.grid.shape)

# Test image generation
img = rg.to_image()
print('Image generated successfully, size:', img.size)
"
```

**Run official tests (if pytest is available):**
```bash
pytest -v tests/
```

**Note**: Tests require pytest which may fail to install due to network issues. The manual test above validates core functionality.

### Documentation

**Build documentation:**
```bash
cd docs
# NEVER CANCEL: Doc build takes 3-5 minutes. Set timeout to 10+ minutes.
make html --timeout 600
```

**Expected timing**: Documentation build completes in ~2 seconds when dependencies are available.

## Validation

**Always manually validate any changes via these scenarios:**

### Scenario 1: Basic RasterGrid Workflow
```bash
PYTHONPATH=/path/to/Urbanflow/src python -c "
import urbanflow
import geopandas as gpd
from shapely.geometry import Polygon

# Create simple urban features
buildings = gpd.GeoDataFrame({'geometry':[Polygon([(0,0),(2,0),(2,2),(0,2)])]}, crs='EPSG:32633')
streets = gpd.GeoDataFrame({'geometry':[Polygon([(2,0),(4,0),(4,2),(2,2)])]}, crs='EPSG:32633')
canals = gpd.GeoDataFrame({'geometry':[Polygon([(0,2),(2,2),(2,4),(0,4)])]}, crs='EPSG:32633')

# Create and test RasterGrid
rg = urbanflow.RasterGrid.from_geojson_dataframes(buildings=buildings, streets=streets, canals=canals, cell_size=1.0)
assert rg.grid.shape[0] > 0, 'Grid should not be empty'

# Test save/load
import tempfile, os
with tempfile.TemporaryDirectory() as tmpdir:
    path = os.path.join(tmpdir, 'test.npz')
    rg.save(path)
    rg_loaded = urbanflow.RasterGrid.load(path)
    assert rg_loaded.grid.shape == rg.grid.shape, 'Loaded grid should match original'

# Test image generation
img = rg.to_image()
assert img.size[0] > 0 and img.size[1] > 0, 'Image should have valid dimensions'
print('Basic RasterGrid workflow: PASSED')
"
```

### Scenario 2: POI Integration (if C++ extension available)
```bash
PYTHONPATH=/path/to/Urbanflow/src python -c "
import urbanflow
try:
    # Test POI functionality
    from shapely.geometry import Point
    import geopandas as gpd
    
    # Create test data
    rg = urbanflow.RasterGrid.from_geojson_dataframes(
        buildings=gpd.GeoDataFrame({'geometry':[Polygon([(0,0),(1,0),(1,1),(0,1)])]}, crs='EPSG:32633'),
        streets=gpd.GeoDataFrame({'geometry':[Polygon([(1,0),(2,0),(2,1),(1,1)])]}, crs='EPSG:32633'),
        canals=gpd.GeoDataFrame({'geometry':[Polygon([(0,1),(1,1),(1,2),(0,2)])]}, crs='EPSG:32633'),
        cell_size=1.0
    )
    
    pois = gpd.GeoDataFrame({'geometry': [Point(0.5, 0.5), Point(1.5, 0.5)]}, crs='EPSG:32633')
    rg_with_pois = urbanflow.RasterGridWithPOIs.from_RasterGrid_and_POIs(rg, pois)
    print('POI integration: PASSED')
except Exception as e:
    print(f'POI integration: SKIPPED ({e})')
"
```

## Common Tasks

### Building with Different Configurations

**Debug build:**
```bash
export SKBUILD_CMAKE_ARGS='-DCMAKE_BUILD_TYPE=Debug'
pip install -e .[dev]
```

**Release build:**
```bash
export SKBUILD_CMAKE_ARGS='-DCMAKE_BUILD_TYPE=Release'
pip install -e .[dev]
```

### Linting and Formatting

**If ruff is available:**
```bash
ruff check src/ tests/
ruff format src/ tests/
```

**If mypy is available:**
```bash
mypy src/urbanflow/
```

**Note**: Linting tools may fail to install due to network connectivity issues.

### Docker Development

**Using provided Docker setup:**
```bash
docker-compose up dev
docker exec -it urbanflow-dev bash
```

**Environment variables for container builds:**
- `CMAKE_BUILD_PARALLEL_LEVEL=4`
- `SKBUILD_CMAKE_ARGS="-DCMAKE_BUILD_TYPE=Debug"`

## Key Project Structure

```
Urbanflow/
├── src/urbanflow/              # Main Python package
│   ├── RasterGrid.py          # Core rasterization functionality
│   ├── RasterGridWithPOIs.py  # POI integration
│   ├── cpp/                   # C++ extension for pathfinding
│   │   ├── path_planner.cpp   # C++ pathfinding implementation
│   │   └── CMakeLists.txt     # C++ build configuration
│   └── utils/                 # Utility modules
│       ├── grid_utils.py      # Grid manipulation utilities
│       ├── poi_utils.py       # POI processing utilities
│       └── courtyard_utils.py # Courtyard detection utilities
├── tests/                     # Test suite
├── docs/                      # Sphinx documentation
├── pyproject.toml            # Project configuration
├── CMakeLists.txt            # Root CMake configuration
└── docker-compose.yml       # Docker development setup
```

## Expected Build Times and Timeouts

- **Dependency installation**: 1-3 minutes (conda), 2-5 minutes (pip with network issues)
- **C++ extension build**: 5-10 minutes (NEVER CANCEL - set timeout to 15+ minutes)
- **Documentation build**: 2-3 seconds (set timeout to 10+ minutes for safety)  
- **Test execution**: 10-30 seconds for basic tests
- **Manual validation scenarios**: 5-10 seconds each

**CRITICAL**: Always wait for builds to complete. Network timeouts and slow dependency resolution are normal.

## Known Issues and Workarounds

1. **Network connectivity**: PyPI timeouts are common. Use conda when possible, or retry pip commands with longer timeouts.

2. **C++ extension build failures**: The package works in Python-only mode. Import will show "No path planner found." message - this is expected.

3. **POI image generation**: May fail without certain columns in POI data. Use basic `to_image()` method instead of `to_image_with_POIs()`.

4. **Courtyard utilities**: Some edge cases in automatic courtyard detection may fail with pandas indexing errors.

## API Documentation

Available at https://venicefoye.github.io/Urbanflow/index.html

## Troubleshooting

**If imports fail:**
- Ensure `PYTHONPATH` includes `src/` directory
- Check that all required dependencies are installed
- Verify GeoPandas and related spatial libraries are working

**If C++ extension fails:**
- The package works without it - functionality is reduced but core features work
- Check that cmake, ninja, and pybind11 are properly installed
- Verify compiler toolchain is available

**If tests fail:**
- Run the manual validation scenarios above
- Check that test data can be created successfully
- Verify that basic RasterGrid operations work

**Always test the basic functionality validation scenario after making any changes to ensure the package remains operational.**