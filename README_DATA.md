# 15-Minute-Venice Data Readme

There are multiple types of data we have collected due to this project, and this folder should contain all of them. This will serve as a documentation of the data files: what they are, and how to read/use them.

#### A quick note on File Types

- `.feather` files: a fast version of a GeoPandas or Pandas dataframe. You need to have `pyarrow` (`pip install pyarrow`) in order to read them. GeoPandas has a function to read these files:
```
needs = gpd.read_feather("needs_no_duplicates.feather")
```
- `.npz` files: these serve as compressed numpy dictionaries: to use the data, load them with `np.load(filepath)`, and then access their stored numpy arrays as you would a python dictionary.


## The Data Files

This section describes the saved data formats for Urbanflow's main data types: `RasterGrid` and `RasterGridWithPOIs`.

### RasterGrid Saved Format

`RasterGrid` objects are saved as compressed `.npz` files using `numpy.savez_compressed()`. The file contains the following keys:

- **`grid`**: 2D numpy array (uint8) containing the rasterized urban morphology data
- **`transform`**: Affine transformation matrix for mapping grid indices (row, col) to spatial coordinates
- **`legend`**: Dictionary mapping feature types to integer codes, typically:
    + 0 = ocean (background)
    + 1 = street
    + 2 = building
    + 3 = canal
    + 4 = courtyard
- **`cell_size`**: Grid resolution in coordinate units (e.g., meters)
- **`coordinate_reference_system`**: Coordinate reference system string (e.g., "EPSG:32633")

**Loading a RasterGrid:**
```python
import urbanflow
raster_grid = urbanflow.RasterGrid.load("path/to/raster_grid.npz")
```

### RasterGridWithPOIs Saved Format

`RasterGridWithPOIs` objects are saved as a **directory** containing multiple files:

#### Directory Structure:
- **`poi_gdf.parquet.gzip`**: Compressed Parquet file containing the Points of Interest (POI) GeoDataFrame
- **`raster_grid_with_pois_filepath.npz`**: Compressed NPZ file containing all RasterGrid data plus POI file reference

#### NPZ File Contents:
- **`grid`**: 2D numpy array (uint8) with rasterized urban morphology
- **`transform`**: Affine transformation matrix
- **`legend`**: Feature type to integer code mapping
- **`cell_size`**: Grid resolution in coordinate units
- **`coordinate_reference_system`**: CRS string
- **`POI_gdf_filepath`**: Path to the POI Parquet file

#### POI GeoDataFrame Columns:
The POI data includes geometry and grid coordinate columns:
- **`geometry`**: Point geometries of POIs
- **`row`**, **`col`**: Grid cell coordinates for each POI
- **`row_adj`**, **`col_adj`**: Adjusted coordinates (snapped to nearest street-building intersection)
- Additional columns depend on the original POI data

**Loading a RasterGridWithPOIs:**
```python
import urbanflow
raster_grid_with_pois = urbanflow.RasterGridWithPOIs.load("path/to/directory")
``` 
