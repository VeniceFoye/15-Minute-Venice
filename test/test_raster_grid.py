import pytest
import geopandas as gpd

from src.RasterGrid import RasterGrid


class TestRasterGrid:
    buildings_path = "test_data/buildings.feather"
    streets_path = "test_data/streets.feather"
    canals_path = "test_data/canals.feather"

    def test_load_gpds(self):
        """Testing loading the subselected Data"""
        buildings = gpd.read_feather(self.buildings_path)
        streets = gpd.read_feather(self.streets_path)
        canals = gpd.read_feather(self.canals_path)

    def test_create_raster_grid(self):
        buildings = gpd.read_feather(self.buildings_path)
        streets = gpd.read_feather(self.streets_path)
        canals = gpd.read_feather(self.canals_path)

        raster_grid = RasterGrid.from_geojson_dataframes(buildings, streets, canals)
