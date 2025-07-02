"""
RasterGridWithPOIs -> a version of RasterGrid that includes POIs.
"""

from src.RasterGrid import RasterGrid
from utils.poi_utils import pois_to_grid_coords
from typing import Dict

import geopandas as gpd

from copy import copy

class RasterGridWithPOIs(RasterGrid):
    """
    RasterGridWithPOIs

    A RasterGrid class that provides wrapping around a POI_gdf, which is a GeoPandas.GeoDataFrame.
    """

    POI_gdf : gpd.GeoDataFrame = None

    def __init__(self, POI_gdf : gpd.GeoDataFrame, coordinate_reference_system = "EPSG:32633", cell_size = 1,):
        super().__init__(coordinate_reference_system, cell_size)
        
        if POI_gdf.crs != self.coordinate_reference_system:
            print(f"Warning! POI GeoDataFrame not in CRS {self.coordinate_reference_system}. Changing to that CRS now.")
            POI_gdf = POI_gdf.to_crs(self.coordinate_reference_system)
        
        self.POI_gdf = POI_gdf

    ### Class Methods to Create

    @classmethod
    def from_RasterGrid_and_POIs(
        cls,
        raster_grid : RasterGrid,
        POI_gdf : gpd.GeoDataFrame,
        *,
        do_adjusted : bool = True
    ):
        """
        From RasterGrid and POIs:

        Pseudocode:
        - Load in RasterGrid and its transform
        - Add the 'row' and 'col' as well as 'row_adj' and 'col_adj' columns to POI_geojson
        """
        # Assign the RasterGrid Attributes
        new_RasterGridWithPOIs = cls(POI_gdf, raster_grid.coordinate_reference_system, raster_grid.cell_size)

        new_RasterGridWithPOIs.grid = copy(raster_grid.grid)
        new_RasterGridWithPOIs.transform = copy(raster_grid.transform)
        new_RasterGridWithPOIs.legend = copy(raster_grid.legend)

        print("Aligning POI_gdf to Grid . . .")
        new_RasterGridWithPOIs._align_POI_gdf_to_grid(do_adjusted=do_adjusted)

        return new_RasterGridWithPOIs


    @classmethod
    def from_geojson_dataframes_and_POIs(cls, buildings, streets, canals, POI_gdf, *, courtyards = None, auto_courtyards = True, coordinate_reference_system = "EPSG:32633", cell_size = 1, legend : Dict[str, int] = {"ocean" : 0, "street" : 1, "building" : 2, "canal" : 3, "courtyard" : 4}, do_adjusted=True):
        raster_grid =  RasterGrid.from_geojson_dataframes(buildings, streets, canals, courtyards=courtyards, auto_courtyards=auto_courtyards, coordinate_reference_system=coordinate_reference_system, cell_size=cell_size, legend=legend)
        return cls.from_RasterGrid_and_POIs(raster_grid, POI_gdf, do_adjusted=do_adjusted)
    
    ## Private Methods

    def _align_POI_gdf_to_grid(self, do_adjusted=True):
        """
        _align_POIs_to_grid: Align the POI_geojson to the grid
        """
        if self.POI_gdf is None:
            return ValueError("ValueError: RasterGridWithPOIs.POI_geojson must not be none!")

        self.POI_gdf = pois_to_grid_coords(
            poi_gdf=self.POI_gdf,
            transform=self.transform,
            grid=self.grid,
            do_adjusted=do_adjusted,
            street_code=self.legend['street'],
            building_code=self.legend['building']
        )