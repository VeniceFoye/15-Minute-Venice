import numpy as np
from affine import Affine
import geopandas as gpd
from typing import Dict, Tuple
from PIL import Image

from utils.grid_utils import instantiate_grid_and_transform, rasterize_geoms
from utils.courtyard_utils import add_auto_courtyards

# DEFAULT_VENICE_LEGEND = {"ocean" : 0, "street" : 1, "building" : 2, "canal" : 3, "courtyard" : 4}

class RasterGrid:
    """
    RasterGrid class. Stores the following information:
    1. `grid`: an np.ndarray[np.uint8] with unsigned integer values.
    2. `transform` an Affine transform object to transform points onto the grid
    3. `legend`: a python dict describing the different grid values and their meaning.

    Attributes
    ----------
    cell_size : float
        Square cell width/height in CRS units

    Interface:

    __init__: constructor

    .save(filepath) : saving the grid, as small as possible

    .laod(filepath) : load the grid, as quickly as possible
    """

    ## Constructors
    def __init__(self, coordinate_reference_system: str = "EPSG:32633", cell_size: float = 1):
        """
        Constructor given a grid, transform, and legend

        Parameters
        ----------
        cell_size : float
            Square cell width/height in CRS units. Defaults to 1
        coordinate_reference_system : str
            Name of the CRS of this grid. Defaults to EPSG:32633.

        """
        self.cell_size = cell_size
        self.coordinate_reference_system = coordinate_reference_system

        # Create grid, transform, legend attributes
        self.grid = None
        self.transform = None
        self.legend = None

    # Constructors from geodataframes
    # https://stackoverflow.com/questions/2164258/is-it-not-possible-to-define-multiple-constructors-in-python
    @classmethod
    def from_geojson_dataframes(
        cls,
        buildings: gpd.GeoDataFrame,
        streets: gpd.GeoDataFrame,
        canals: gpd.GeoDataFrame,
        *,
        courtyards : gpd.GeoDataFrame = None,
        auto_courtyards : bool = True,
        coordinate_reference_system: str = "EPSG:32633",
        cell_size: float = 1,
        legend : Dict[str, int] = {"ocean" : 0, "street" : 1, "building" : 2, "canal" : 3, "courtyard" : 4}
    ):
        """
        Rasterize buildings, streets, and canals of Venice into a RasterGrid.

        Parameters
        ----------

        buildings, streets, canals : GeoDataFrames
            All must share the **same CRS**.
        cell_size : float
            Square cell width/height in CRS units.
        legend : Dict[str, int]
            A python dict that contains what cell values contian what. The dict must define:
            "ocean" : (default 0)
            "street" : (default 1)
            "building" : default 2)
            "canal" : (default 3) 

        Returns
        -------
        RasterGrid object with grid, transform, and legend.
        """

        # Create new grid
        new_raster_grid = cls(coordinate_reference_system = coordinate_reference_system, cell_size=cell_size)
        # Assign the legend
        new_raster_grid.legend = legend

        # Check that all the GeoDataFrames are in the same, and correct CRS
        layers = [g for g in (buildings, streets, canals, courtyards) if g is not None]
        LAYER_INDEXES_TO_NAMES = {0 : "Buildings", 1: "Streets", 2: "Canals", 3: "Courtyards"}

        for i, current_layer in enumerate(layers):
            if current_layer.crs != coordinate_reference_system:
                print(f"WARNING! {LAYER_INDEXES_TO_NAMES[i]} layer not in {coordinate_reference_system}. Converting now.")
                layers[i] = current_layer.to_crs(coordinate_reference_system)
        # Create the grid
        grid, transform = instantiate_grid_and_transform(cell_size, layers, default_grid_value = legend["ocean"])

        # Immediately assign the transform
        new_raster_grid.transform = transform

        # Rasterize the Canals, Buildings, and then Streets
        rasterize_geoms(layers[2].geometry, legend['canal'], grid, transform)
        rasterize_geoms(layers[0].geometry, legend['building'], grid, transform)
        rasterize_geoms(layers[1].geometry, legend['street'], grid, transform)

        # See if courtyards exist
        if courtyards is not None:
            # Add couryards
            rasterize_geoms(layers[3].geometry, legend['courtyard'], grid, transform)
        else:
            print("No manual courtyards found.")

        if auto_courtyards:
            print("Adding automatic courtyards. . .")
            grid = add_auto_courtyards(grid, empty_code=legend['ocean'], courtyard_code=legend['courtyard'])
            
        # Assign the Filled-In Grid
        new_raster_grid.grid = grid

        return new_raster_grid

    ## Saving and Loading

    def save(self, filepath : str):
        """
        Save a RasterGrid to a .npz file.
        """
        np.savez_compressed(file=filepath, grid=self.grid, transform=self.transform, legend=self.legend, cell_size=self.cell_size, coordinate_reference_system=self.coordinate_reference_system)

    @classmethod
    def load(cls, filepath):
        """
        Create a RasterGrid object from a .npz compressed file.
        """
        npz_grid = np.load(filepath, allow_pickle=True)

        new_raster_grid = RasterGrid(coordinate_reference_system=npz_grid['coordinate_reference_system'], cell_size=npz_grid['cell_size'])

        new_raster_grid.grid = npz_grid['grid']
        new_raster_grid.transform = npz_grid['transform']
        new_raster_grid.legend = npz_grid['legend']

        return new_raster_grid


    ## Visualization Functions
    def to_image(
        self,
        scale: int = 1,
        palette: Dict[int, Tuple[int, int, int]] | None = None,
    ) -> Image.Image:
        """
        Convert a uint8 grid of states into a PIL Image.

        Parameters
        ----------
        grid : ndarray[int]   shape (rows, cols)
            Cell codes: 0=ocean, 1=street, 2=building (extend as you wish).
        scale : int, default 1
            How many output pixels per cell (must be ≥1). 2 doubles width/height.
        palette : {state: (R, G, B)}, optional
            Custom colors.  Unspecified states default to white.

        Returns
        -------
        PIL.Image.Image  (mode "RGB")

        Notes
        -----
        * Uses pure NumPy → very fast, even for multi-million-cell grids.
        * Nearest-neighbour up-scaling keeps the crisp blocky CA look.
        """
        if scale < 1 or not isinstance(scale, int):
            raise ValueError("scale must be a positive integer")

        # ---------- default color map ------------------------------------
        default_palette = {
            0: (173, 216, 230),   # ocean – light blue
            1: (190, 190, 190),   # street – light grey
            2: (178,  34,  34),   # building – dark red
            3: ( 64, 224, 208),   # canal – turquoise
            4: (152, 251, 152),   # courtyard – pale green
        }
        if palette is not None:
            default_palette.update(palette)

        rows, cols = self.grid.shape
        rgb = np.zeros((rows, cols, 3), dtype=np.uint8)

        for state, color in default_palette.items():
            rgb[self.grid == state] = color

        img = Image.fromarray(rgb, mode="RGB")

        if scale > 1:
            img = img.resize((cols * scale, rows * scale), resample=Image.Resampling.NEAREST)

        return img