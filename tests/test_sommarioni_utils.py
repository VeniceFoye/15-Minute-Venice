import pytest
import geopandas

from utils.sommarioni_utils import load_sommarioni_layers


class TestSommarioniUtils:
    gemoetries_path = "test_data/sommarioni_geometries.feather"

    def test_load_sommarioni_layers(self):
        buildings, streets, canals, courtyards = load_sommarioni_layers(self.gemoetries_path)