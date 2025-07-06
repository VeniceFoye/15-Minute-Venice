"""
RasterGridWithPOIs -> a version of RasterGrid that includes POIs.
"""

from .RasterGrid import RasterGrid
from .utils.poi_utils import pois_to_grid_coords
from .cpp.path_planner import path_between_pois
from typing import Dict, Tuple
import numpy as np

import matplotlib.pyplot as plt

import tcod

from PIL import ImageDraw, Image

import os

import geopandas as gpd

from copy import copy


class RasterGridWithPOIs(RasterGrid):
    """
    RasterGridWithPOIs

    A RasterGrid class that provides wrapping around a POI_gdf, which is a GeoPandas.GeoDataFrame.
    """

    POI_gdf: gpd.GeoDataFrame = None
    tcod_cost_grid : np.ndarray = None

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
        
        if 'uid' in POI_gdf.columns:
            print(f"Setting column `uid` to be the index column for the POI_gdf.")
            POI_gdf = POI_gdf.set_index("uid")

        self.POI_gdf = POI_gdf

        # Create tcod path.


        # self.path2d = tcod.path.path2d(tcod_grid)
    ### Class Methods to Create

    @classmethod
    def from_RasterGrid_and_POIs(
        cls,
        raster_grid: RasterGrid,
        POI_gdf: gpd.GeoDataFrame,
        *,
        do_adjusted: bool = True,
        force_realignment=False,
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
        if (
            "row" in new_RasterGridWithPOIs.POI_gdf.columns
            and "col" in new_RasterGridWithPOIs.POI_gdf.columns
            and not do_adjusted
        ):
            print(
                "POI_gdf already contains 'row' and 'col' columns. Skipping coordinate assignment."
            )
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

        if (
            "row_adj" in new_RasterGridWithPOIs.POI_gdf.columns
            and "col_adj" in new_RasterGridWithPOIs.POI_gdf.columns
            and do_adjusted
        ):
            print(
                "POI_gdf already contains 'row_adj' and 'col_adj' columns. Skipping coordinate assignment."
            )
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

            # TODO: Add configuration for walkability, costs
        tcod_grid = new_RasterGridWithPOIs.grid.copy()
        tcod_grid[tcod_grid == new_RasterGridWithPOIs.legend['canal']] = 0
        tcod_grid[tcod_grid == new_RasterGridWithPOIs.legend['building']] = 0
        new_RasterGridWithPOIs.tcod_cost_grid = tcod_grid

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
        force_realignment=False,
    ):
        raster_grid = RasterGrid.from_geojson_dataframes(
            buildings,
            streets,
            canals,
            courtyards=courtyards,
            auto_courtyards=auto_courtyards,
            coordinate_reference_system=coordinate_reference_system,
            cell_size=cell_size,
            legend=legend,
        )
        return cls.from_RasterGrid_and_POIs(
            raster_grid,
            POI_gdf,
            do_adjusted=do_adjusted,
            force_realignment=force_realignment,
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
        parquet_filepath = npz_grid["POI_gdf_filepath"].item()

        POI_gdf = gpd.read_parquet(parquet_filepath)

        raster_grid = RasterGrid.load(npz_path)

        return cls.from_RasterGrid_and_POIs(raster_grid=raster_grid, POI_gdf=POI_gdf)
    

    ####### POI Pathing using cpp/path_planner.cpp/path_between_pois PyBind11 Wrapper

    def compute_path_between_POIs(self, start_poi_uid : str, end_poi_uid : str, *, do_tcod_pathing : bool = False, uid_column : str = None, do_logging : bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """
        compute_path_between_POIs

        Compute the path across the RasterGrid between two POIs in the POI_gdf, according to their "uid."
        Note that the path from X -> Y should be the same as Y -> X

        Parameters
        ----------
        start_poi_uid : str
            The uid of the originaiting POI, e.g. "CNC-0001"
        end_poi_uid : str
            The uid of the target POI, e.g. "APO-0419"

            
        Returns
        -------
        row : np.ndarray
            The row values of the path
        col : np.ndarray
            The column values of the path
        """
        poi_gdf = self.POI_gdf

        # TODO: add check for non uid index
        if uid_column is not None:
            poi_gdf.set_index(uid_column)
        
        source_poi = poi_gdf.loc[start_poi_uid]
        target_poi = poi_gdf.loc[end_poi_uid]

        if 'row_adj' in poi_gdf.columns:
            if do_logging:
                print("Using adjusted row and col values")

            source_poi_r = source_poi['row_adj']
            source_poi_c = source_poi['col_adj']

            target_poi_r = target_poi['row_adj']
            target_poi_c = target_poi['col_adj']
        else:
            if do_logging:
                print("Warning! Using NON adjusted row and col values")

            source_poi_r = source_poi['row']
            source_poi_c = source_poi['col']

            target_poi_r = target_poi['row']
            target_poi_c = target_poi['col']
        
        # TODO: add other param options
        if do_tcod_pathing:
            path = tcod.path.path2d(self.tcod_cost_grid, start_points=[(source_poi_r, source_poi_c)], end_points=[(target_poi_r, target_poi_c)], cardinal=10, diagonal=14)
            if path is not None and len(path) > 0:
                path = np.array(path)
                path_r = path[:, 0]
                path_c = path[:, 1]
            else:
                path_r = np.array([])
                path_c = np.array([])
        else:
            path_r, path_c = path_between_pois(self.grid, source_poi_r, source_poi_c, target_poi_r, target_poi_c)

        return path_r, path_c



    ####### Visualization Methods
    def to_image_with_POIs(
        self,
        scale: int = 1,
        palette: dict[int, tuple[int, int, int]] | None = None,
        function_col: str = "PP_Function_TOP",
        color_map: dict[str, tuple[int, int, int]] | None = None,
        default_color: tuple[int, int, int] = (0, 0, 0),
        poi_radius: int | None = None,
    ):
        # Get the image from the grid
        img = super().to_image(scale=scale, palette=palette)

        draw = ImageDraw.Draw(img)
        poi_radius = poi_radius if poi_radius is not None else max(1, scale // 2)

        # ------------------------------------------------------------------
        # build or validate category → color dict
        # ------------------------------------------------------------------
        cats = self.POI_gdf[function_col].fillna("UNKNOWN").astype(str)
        if color_map is None:
            base_cycle = plt.cm.get_cmap("tab20").colors  # 20 distinct colors
            color_map = {
                cat: tuple(int(255 * c) for c in base_cycle[i % 20])
                for i, cat in enumerate(sorted(cats.unique()))
            }
        else:
            # ensure RGB ints 0-255
            color_map = {k: tuple(int(x) for x in v) for k, v in color_map.items()}

        # ------------------------------------------------------------------
        # draw dots
        # ------------------------------------------------------------------
        for r, c, cat in zip(self.POI_gdf["row"], self.POI_gdf["col"], cats):
            color = color_map.get(cat, default_color)
            x = c * scale + scale // 2
            y = r * scale + scale // 2
            draw.ellipse(
                (x - poi_radius, y - poi_radius, x + poi_radius, y + poi_radius),
                fill=color,
            )

        return img

    
    def to_image_with_path(
        self,
        path_r: np.ndarray,
        path_c: np.ndarray,
        *,
        scale: int = 5,
        palette: dict[int, tuple[int, int, int]] | None = None,
        poi_start: tuple[int, int] | None = None,   # (row_adj, col_adj)
        poi_end: tuple[int, int]   | None = None,   # (row_adj, col_adj)
        poi_colour: tuple[int, int, int] = (0, 0, 0),
        path_colour: tuple[int, int, int] = (255, 215, 0),     # gold
        path_width: int | None = None,
    ) -> Image.Image:
        """
        Render grid + (optional) POIs + path.  Returns a PIL Image.

        Parameters
        ----------
        path_r, path_c : int arrays
            Output from `path_from_poi` (exclusive of origin).
        poi_start / poi_end : (row,col) tuples in grid coords
            If given, small dots are drawn at those cells.
        path_width : int
            Width of the polyline in *pixels*. Defaults to max(1, scale//2).

        Returns
        -------
        PIL.Image.Image  (ready to save or display)
        """
        img = self.to_image(scale=scale, palette=palette)
        draw = ImageDraw.Draw(img)

        w = path_width if path_width is not None else max(1, scale // 2)

        # ------------------------------------------------------------------ #
        # 1. draw path as a polyline in pixel space
        # ------------------------------------------------------------------ #
        pts = [
            (c * scale + scale // 2, r * scale + scale // 2)
            for r, c in zip(path_r, path_c)
        ]
        if pts:
            draw.line(pts, fill=path_colour, width=w, joint="curve")

        # ------------------------------------------------------------------ #
        # 2. optional start / end POI dots
        # ------------------------------------------------------------------ #
        R = max(2, w)   # radius
        if poi_start is not None:
            x, y = poi_start[1] * scale + scale // 2, poi_start[0] * scale + scale // 2
            draw.ellipse((x - R, y - R, x + R, y + R), fill=poi_colour)
        if poi_end is not None:
            x, y = poi_end[1] * scale + scale // 2, poi_end[0] * scale + scale // 2
            draw.ellipse((x - R, y - R, x + R, y + R), fill=poi_colour)

        return img



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
