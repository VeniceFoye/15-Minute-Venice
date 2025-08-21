"""
RasterGridWithPOIs
==================

Extension of :class:`urbanflow.RasterGrid` that integrates points of interest
(POIs) with the raster grid. This module provides tools to:

- Align POIs stored in a :class:`geopandas.GeoDataFrame` to raster grid cells.
- Store both raw (row, col) and adjusted (row_adj, col_adj) POI coordinates,
  where adjusted coordinates are snapped to the nearest accessible street cell.
- Compute shortest paths between POIs using either:
  * A fast C++ path planner (via PyBind11).
  * An optional `tcod`-based pathfinding backend.
- Visualize the raster grid with:
  * POIs drawn as colored dots by category.
  * Paths drawn as polylines with optional start/end markers.
- Save and load complete grid+POI objects to/from disk.

Classes
-------
RasterGridWithPOIs
    A subclass of :class:`RasterGrid` that adds POI handling, alignment,
    pathfinding, and visualization utilities.

Typical usage
-------------
>>> import geopandas as gpd
>>> from urbanflow.RasterGrid import RasterGrid
>>> from urbanflow.RasterGridWithPOIs import RasterGridWithPOIs
>>> # Load base raster and POIs
>>> rg = RasterGrid.from_geojson_dataframes(buildings, streets, canals)
>>> pois = gpd.read_file("pois.geojson")
>>> # Create combined object
>>> rgp = RasterGridWithPOIs.from_RasterGrid_and_POIs(rg, pois)
>>> # Compute path between POIs
>>> rows, cols = rgp.compute_path_between_POIs("POI_A", "POI_B")
>>> # Render visualization
>>> img = rgp.to_image_with_path(rows, cols)
>>> img.show()
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
    Raster grid with Points of Interest (POIs).

    Extends :class:`urbanflow.RasterGrid` by attaching a
    :class:`geopandas.GeoDataFrame` of POIs to the raster grid,
    aligning them to grid cells, and enabling pathfinding between POIs.
    Includes methods for saving/loading, path computation, and visualization.

    Attributes
    ----------
    POI_gdf : geopandas.GeoDataFrame
        GeoDataFrame of points of interest, with columns for grid coordinates
        (``row``, ``col``) and optionally adjusted coordinates
        (``row_adj``, ``col_adj``).
    tcod_cost_grid : ndarray of int
        Grid suitable for `tcod` pathfinding, with impassable cells set to 0.
    """

    POI_gdf: gpd.GeoDataFrame = None
    tcod_cost_grid : np.ndarray = None

    def __init__(
        self,
        POI_gdf: gpd.GeoDataFrame,
        coordinate_reference_system="EPSG:32633",
        cell_size=1,
    ):
        """
        Initialize a RasterGridWithPOIs.

        Parameters
        ----------
        POI_gdf : geopandas.GeoDataFrame
            GeoDataFrame of POIs. Must contain a geometry column. If a
            ``uid`` column exists, it is set as the index.
        coordinate_reference_system : str, optional
            CRS for the raster grid and POIs. Defaults to EPSG:32633 (UTM 33N).
        cell_size : int, optional
            Resolution of raster grid cells. Defaults to 1.
        """
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
        Construct a RasterGridWithPOIs from an existing RasterGrid and POIs.

        Parameters
        ----------
        raster_grid : RasterGrid
            The base raster grid object.
        POI_gdf : geopandas.GeoDataFrame
            GeoDataFrame containing POIs.
        do_adjusted : bool, optional
            Whether to compute adjusted POI coordinates (snapped to nearest
            street cell). Defaults to True.
        force_realignment : bool, optional
            If True, realign POIs to the grid even if row/col columns exist.
            Defaults to False.

        Returns
        -------
        RasterGridWithPOIs
            A new instance with POIs aligned to the raster grid.
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
        """
        Construct a RasterGridWithPOIs directly from GeoDataFrames of features
        and a POI GeoDataFrame.

        Parameters
        ----------
        buildings, streets, canals, courtyards : geopandas.GeoDataFrame
            Feature layers used to generate the base raster grid.
        POI_gdf : geopandas.GeoDataFrame
            GeoDataFrame of POIs.
        auto_courtyards : bool, optional
            Whether to automatically generate courtyard cells. Defaults to True.
        coordinate_reference_system : str, optional
            Target CRS. Defaults to EPSG:32633.
        cell_size : int, optional
            Grid resolution in coordinate units. Defaults to 1.
        legend : dict, optional
            Mapping of feature types to integer codes. Defaults to
            {ocean:0, street:1, building:2, canal:3, courtyard:4}.
        do_adjusted : bool, optional
            Compute adjusted POI coordinates. Defaults to True.
        force_realignment : bool, optional
            If True, force POI realignment. Defaults to False.

        Returns
        -------
        RasterGridWithPOIs
            A raster grid with POIs aligned to grid cells.
        """
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
        Save a RasterGridWithPOIs to a directory.

        Parameters
        ----------
        filepath : str
            Path to the directory where the object will be saved.
            Creates the directory if it does not exist.

        Notes
        -----
        - POIs are stored as a Parquet file.
        - Grid, transform, legend, and metadata are stored in a compressed NPZ.
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
        Load a RasterGridWithPOIs from a saved directory.

        Parameters
        ----------
        filepath : str
            Directory containing saved raster grid and POI files.

        Returns
        -------
        RasterGridWithPOIs
            Loaded raster grid with POIs.
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
        Compute the shortest path between two POIs on the raster grid.

        Parameters
        ----------
        start_poi_uid : str
            UID of the source POI (index in POI_gdf).
        end_poi_uid : str
            UID of the target POI.
        do_tcod_pathing : bool, optional
            If True, use `tcod` pathfinding instead of the C++ planner.
            Defaults to False.
        uid_column : str, optional
            Column name to set as index if POI_gdf is not indexed by UID.
        do_logging : bool, optional
            If True, print diagnostic information. Defaults to False.

        Returns
        -------
        path_r : ndarray of int
            Row indices of the computed path.
        path_c : ndarray of int
            Column indices of the computed path.

        Notes
        -----
        - If ``row_adj``/``col_adj`` exist, adjusted coordinates are used.
        - If no path is found, empty arrays are returned.
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
        """
        Render the raster grid with POIs drawn as colored dots.

        Parameters
        ----------
        scale : int, optional
            Scale factor for rendering. Each cell becomes a square of
            size ``scale``. Defaults to 1.
        palette : dict of int -> (r,g,b), optional
            Mapping from raster values to RGB colors.
        function_col : str, optional
            Column in POI_gdf used for categorical coloring. Defaults to
            ``"PP_Function_TOP"``.
        color_map : dict of str -> (r,g,b), optional
            Mapping from categories to RGB colors. If None, a default
            matplotlib ``tab20`` palette is used.
        default_color : tuple of int, optional
            Fallback color (RGB). Defaults to black.
        poi_radius : int, optional
            Radius of POI dots in pixels. Defaults to ``max(1, scale//2)``.

        Returns
        -------
        PIL.Image.Image
            Image of the raster grid with POIs overlayed.
        """
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
        Render the raster grid with an overlayed path and optional POI markers.

        Parameters
        ----------
        path_r, path_c : ndarray of int
            Row and column indices of the path.
        scale : int, optional
            Scale factor for rendering. Defaults to 5.
        palette : dict of int -> (r,g,b), optional
            Mapping from raster values to RGB colors.
        poi_start, poi_end : tuple of (row, col), optional
            Coordinates of start and end POIs to highlight.
        poi_colour : tuple of int, optional
            Color of start/end POI dots. Defaults to black.
        path_colour : tuple of int, optional
            Color of the path polyline. Defaults to gold.
        path_width : int, optional
            Width of the path polyline in pixels. Defaults to
            ``max(1, scale//2)``.

        Returns
        -------
        PIL.Image.Image
            Image of the raster grid with path overlayed.
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
        Align POIs in the GeoDataFrame to raster grid coordinates.

        Parameters
        ----------
        do_adjusted : bool, optional
            If True, compute adjusted POI coordinates snapped to nearest
            accessible cells (e.g., streets). Defaults to True.

        Raises
        ------
        ValueError
            If POI_gdf is None.

        Notes
        -----
        Updates ``POI_gdf`` in place by adding or overwriting
        ``row``, ``col``, and (optionally) ``row_adj``/``col_adj``.
        """
        if self.POI_gdf is None:
            raise ValueError(
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
