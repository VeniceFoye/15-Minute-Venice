import pytest
import geopandas as gpd

from src.RasterGrid import RasterGrid
from utils.sommarioni_utils import load_sommarioni_layers # this is probably bad practice, but thats ok.

class TestRasterGrid:
    buildings_path = "test_data/buildings.feather"
    streets_path = "test_data/streets.feather"
    canals_path = "test_data/canals.feather"

    sommarioni_path = "test_data/sommarioni_geometries.feather"

    def test_load_gdfs(self):
        """Testing loading the subselected Data"""
        buildings = gpd.read_feather(self.buildings_path)
        streets = gpd.read_feather(self.streets_path)
        canals = gpd.read_feather(self.canals_path)

    def test_create_raster_grid_no_courtyards(self):
        buildings = gpd.read_feather(self.buildings_path)
        streets = gpd.read_feather(self.streets_path)
        canals = gpd.read_feather(self.canals_path)

        raster_grid = RasterGrid.from_geojson_dataframes(buildings, streets, canals, auto_courtyards=False)

    def test_create_raster_grid_auto_courtyards(self):
        buildings = gpd.read_feather(self.buildings_path)
        streets = gpd.read_feather(self.streets_path)
        canals = gpd.read_feather(self.canals_path)

        raster_grid = RasterGrid.from_geojson_dataframes(buildings, streets, canals, auto_courtyards=True)

    def test_create_sommarioni_raster_grid_manual_courtyards(self):
        buildings, streets, canals, courtyards = load_sommarioni_layers(self.sommarioni_path)

        raster_grid = RasterGrid.from_geojson_dataframes(buildings, streets, canals, courtyards=courtyards, auto_courtyards=False)
    
    def test_create_sommarioni_raster_grid_all_courtyards(self):
        buildings, streets, canals, courtyards = load_sommarioni_layers(self.sommarioni_path)

        raster_grid = RasterGrid.from_geojson_dataframes(buildings, streets, canals, courtyards=courtyards, auto_courtyards=True)
    
    def test_create_sommarioni_raster_grid_auto_courtyards(self):
        buildings, streets, canals, courtyards = load_sommarioni_layers(self.sommarioni_path)

        raster_grid = RasterGrid.from_geojson_dataframes(buildings, streets, canals, auto_courtyards=True)
    

    def test_legend(self):
        pass

    def test_transforms(self):
        pass


