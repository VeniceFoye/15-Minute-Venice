from importlib import import_module

# Pure-Python re-exports
from .RasterGrid import RasterGrid
from .RasterGridWithPOIs import RasterGridWithPOIs

# Set up logging
from .logging_config import logger

# Try loading the compiled extension lazily
try:
    path_planner = import_module("urbanflow.cpp.path_planner")
    path_between_pois = path_planner.path_between_pois
except ModuleNotFoundError:   # editable-install before build?
    logger.warning("No path planner found.")
    path_between_pois = None

__all__ = [
    "RasterGrid",
    "RasterGridWithPOIs",
    "path_between_pois",
    "utils"
]
