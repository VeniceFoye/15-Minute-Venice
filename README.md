# Urbanflow
A python package meant for investigating urban morphology using multi-agent simulations. Originally built for the [Venice Data Week project at EPFL](https://parcelsofvenice.epfl.ch/activities/venice-data-week-2025), this code was then adapted and refactored to apply to any urban morphology.

This package assumes the inputted morphological data is in the form of a GeoPandas GeoDataFrame: both the originating polygons and points-of-interest (POIs).

## Features

- Processing and Rasterization of GeoDataFrames
- Visualization of Rasterized Morphology
- Usage of points-of-interest (POIs) and shortest-distance pathing between them

### Planned Features
- Docker Deployment
- Interactive Visualization
- Multi-Agent Simulation (With hyperparameters)
- Data Collection and Processing from Simulations

## Installation

`pip install urbanflow`

## Quick Start / Usage

#### Creating a RasterGrid

```
import urbanflow
import geopandas as gpd

# Should be polygon geometries
geojson_buildings_df = gpd.read_geojson("/your/dir/buildings.geojson")
geojson_streets_df = gpd.read_geojson("/your/dir/streets.geojson")
geojson_canals_df = gpd.read_geojson("/your/dir/canals.geojson")
geojson_courtyards_df = gpd.read_geojson("/your/dir/courtyards.geojson")

raster_grid = urbanflow.RasterGrid.from_geojson_dataframes(
    buildings = geojson_buildings_df,
    streets = geojson_streets_df,
    canals = geojson_canals_df,
    courtyards = geojson_courtyards_df,

    auto_courtyards = True,
    coordinate_reference_system = "your CRS here",
    cell_size = 1
    legend = {"ocean" : 0, "street" : 1, "building" : 2, "canal" : 3, "courtyard" : 4}
)

img = raster_grid.to_image()
img.show()
```

#### Loading/Saving a RasterGrid

```
raster_grid.save("/your/dir/raster_grid.npz")

raster_grid = RasterGrid.load("/your/dir/raster_grid.npz")
```

#### Creating a RasterGridWithPOIs

```
geojson_POIs_gdf = gpd.read_geojson("/your/dir/POIs.geojson") # Should be point geometries

raster_grid_with_POIs = urbanflow.RasterGridWithPOIs.from_RasterGrid_and_POIs(
    raster_grid = raster_grid,
    POI_gdf = geojson_POIs_gdf
)

img = raster_grid_with_POIs.to_image_with_POIs()
img.show()
```

Loading, Saving, etc. works generally the same with `RasterGridWithPOIs` as it does for `RasterGrid` objects.

## API Documentation

Available at https://venicefoye.github.io/Urbanflow/index.html
