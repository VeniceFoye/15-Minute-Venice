"""
RasterGridWithPOIs -> a version of RasterGrid that includes POIs.
"""

from src.RasterGrid import RasterGrid
from utils.poi_utils import pois_to_grid_coords
from typing import Dict
import numpy as np

import os
import sys

import pickle

from affine import Affine

import geopandas as gpd

from copy import copy


class RasterGridWithPOIs(RasterGrid):
    """
    RasterGridWithPOIs

    A RasterGrid class that provides wrapping around a POI_gdf, which is a GeoPandas.GeoDataFrame.
    """

    POI_gdf: gpd.GeoDataFrame = None

    def __init__(
        self,
        POI_gdf: gpd.GeoDataFrame,
        coordinate_reference_system="EPSG:32633",
        cell_size=1,
    ):
        super().__init__(coordinate_reference_system, cell_size)

        if POI_gdf.crs != self.coordinate_reference_system:
            print(
                f"Warning! POI GeoDataFrame not in CRS {self.coordinate_reference_system}. Changing to that CRS now."
            )
            POI_gdf = POI_gdf.to_crs(self.coordinate_reference_system)

        self.POI_gdf = POI_gdf

    ### Class Methods to Create

    @classmethod
    def from_RasterGrid_and_POIs(
        cls,
        raster_grid: RasterGrid,
        POI_gdf: gpd.GeoDataFrame,
        *,
        do_adjusted: bool = True,
        force_realignment = False
    ):
        """
        From RasterGrid and POIs:

        Pseudocode:
        - Load in RasterGrid and its transform
        - Add the 'row' and 'col' as well as 'row_adj' and 'col_adj' columns to POI_geojson
        """
        # Assign the RasterGrid Attributes
        new_RasterGridWithPOIs = cls(
            POI_gdf, raster_grid.coordinate_reference_system, raster_grid.cell_size
        )

        new_RasterGridWithPOIs.grid = copy(raster_grid.grid)
        new_RasterGridWithPOIs.transform = copy(raster_grid.transform)
        new_RasterGridWithPOIs.legend = copy(raster_grid.legend)

        print("Aligning POI_gdf to Grid . . .")
        if "row" in new_RasterGridWithPOIs.POI_gdf.columns and "col" in new_RasterGridWithPOIs.POI_gdf.columns and not do_adjusted:
            print("POI_gdf already contains 'row' and 'col' columns. Skipping coordinate assignment.")
            if force_realignment and not do_adjusted:
                print("force_realignment is True. Aligning anyway.")
                new_RasterGridWithPOIs.POI_gdf = pois_to_grid_coords(
                    poi_gdf=new_RasterGridWithPOIs.POI_gdf,
                    transform=new_RasterGridWithPOIs.transform,
                    grid=new_RasterGridWithPOIs.grid,
                    do_adjusted=do_adjusted,
                    street_code=new_RasterGridWithPOIs.legend["street"],
                    building_code=new_RasterGridWithPOIs.legend["building"],
                )
        else:
            new_RasterGridWithPOIs.POI_gdf = pois_to_grid_coords(
                poi_gdf=new_RasterGridWithPOIs.POI_gdf,
                transform=new_RasterGridWithPOIs.transform,
                grid=new_RasterGridWithPOIs.grid,
                do_adjusted=do_adjusted,
                street_code=new_RasterGridWithPOIs.legend["street"],
                building_code=new_RasterGridWithPOIs.legend["building"],
            )

        if "row_adj" in new_RasterGridWithPOIs.POI_gdf.columns and "col_adj" in new_RasterGridWithPOIs.POI_gdf.columns and do_adjusted:
            print("POI_gdf already contains 'row_adj' and 'col_adj' columns. Skipping coordinate assignment.")
            if force_realignment and do_adjusted:
                print("force_realignment is True. Aligning anyway.")
                new_RasterGridWithPOIs.POI_gdf = pois_to_grid_coords(
                    poi_gdf=new_RasterGridWithPOIs.POI_gdf,
                    transform=new_RasterGridWithPOIs.transform,
                    grid=new_RasterGridWithPOIs.grid,
                    do_adjusted=do_adjusted,
                    street_code=new_RasterGridWithPOIs.legend["street"],
                    building_code=new_RasterGridWithPOIs.legend["building"],
                )
        else:
            new_RasterGridWithPOIs.POI_gdf = pois_to_grid_coords(
                poi_gdf=new_RasterGridWithPOIs.POI_gdf,
                transform=new_RasterGridWithPOIs.transform,
                grid=new_RasterGridWithPOIs.grid,
                do_adjusted=do_adjusted,
                street_code=new_RasterGridWithPOIs.legend["street"],
                building_code=new_RasterGridWithPOIs.legend["building"],
            )

        
        new_RasterGridWithPOIs._align_POI_gdf_to_grid(do_adjusted=do_adjusted)

        return new_RasterGridWithPOIs

    @classmethod
    def from_geojson_dataframes_and_POIs(
        cls,
        buildings,
        streets,
        canals,
        POI_gdf,
        *,
        courtyards=None,
        auto_courtyards=True,
        coordinate_reference_system="EPSG:32633",
        cell_size=1,
        legend: Dict[str, int] = {
            "ocean": 0,
            "street": 1,
            "building": 2,
            "canal": 3,
            "courtyard": 4,
        },
        do_adjusted=True,
        force_realignment=False
    ):
        raster_grid = RasterGrid.from_geojson_dataframes(
            buildings,
            streets,
            canals,
            courtyards=courtyards,
            auto_courtyards=auto_courtyards,
            coordinate_reference_system=coordinate_reference_system,
            cell_size=cell_size,
            legend=legend
        )
        return cls.from_RasterGrid_and_POIs(
            raster_grid, POI_gdf, do_adjusted=do_adjusted, force_realignment=force_realignment
        )

    ####### Saving / Loading

    def save(self, filepath: str):
        """
        Save a RasterGridWithPOIs to a *Directory*.
        """
        if not os.path.isdir(filepath):
            os.makedirs(filepath, exist_ok=True)

        parquet_filepath = os.path.join(filepath, "poi_gdf.parquet.gzip")
        self.POI_gdf.to_parquet(parquet_filepath, compression="gzip")

        np.savez_compressed(
            file=os.path.join(filepath, "raster_grid_with_pois_filepath.npz"),
            grid=self.grid,
            transform=self.transform,
            legend=self.legend,
            POI_gdf_filepath=parquet_filepath,
            cell_size=self.cell_size,
            coordinate_reference_system=self.coordinate_reference_system,
        )


    @classmethod
    def load(cls, filepath):
        """
        Create a RasterGridWithPOIs object from a directory.
        """
        npz_path = os.path.join(filepath, "raster_grid_with_pois_filepath.npz")
        npz_grid = np.load(npz_path, allow_pickle=True)
        parquet_filepath = npz_grid['POI_gdf_filepath'].item()

        POI_gdf = gpd.read_parquet(parquet_filepath)

        raster_grid = RasterGrid.load(npz_path)

        return cls.from_RasterGrid_and_POIs(raster_grid=raster_grid, POI_gdf=POI_gdf)

    ####### Private Methods

    def _align_POI_gdf_to_grid(self, do_adjusted=True):
        """
        _align_POIs_to_grid: Align the POI_geojson to the grid
        """
        if self.POI_gdf is None:
            return ValueError(
                "ValueError: RasterGridWithPOIs.POI_geojson must not be none!"
            )

        self.POI_gdf = pois_to_grid_coords(
            poi_gdf=self.POI_gdf,
            transform=self.transform,
            grid=self.grid,
            do_adjusted=do_adjusted,
            street_code=self.legend["street"],
            building_code=self.legend["building"],
        )
