"""
Raster utils.

Functions to handle rasters, especially to prepare them for conversion into proper RasterGrid or RasterGridWithPOIs objects.

"""
import geopandas as gpd
import numpy as np
import rasterio.features

from PIL import Image
from shapely.geometry import shape

def raster_to_geo_data_frame(image_path : str, crs = "EPSG:3857") -> gpd.GeoDataFrame:
    """
    Convert a rastered image to a simple GeoDataFrame
    
    Parameters
    ----------
    image_path : str
        Path to a .png grayscale image (single channel)

    Returns
    -------
    gpd.GeoDataFrame : A GeoDataFrame of polygons from the region of the image.
    """

    # Load the image
    img = Image.open(image_path)

    # Convert to np grid
    grid = np.array(img)

    # Extract Polygons
    shapes = rasterio.features.shapes(grid, connectivity=8) # super useful: takes in a mask, connectivity value, and transform.

    # Create records for GeoDataFrame
    shapes_to_add = []
    for polygon, value in shapes:
        shapes_to_add.append({"geometry": shape(polygon), "value" : value})

    if not shapes_to_add:
        return gpd.GeoDataFrame(columns=["class_id","class_label","geometry"], geometry="geometry", crs=crs)

    gdf = gpd.GeoDataFrame(shapes_to_add, geometry="geometry", crs=crs)

    return gdf

    
