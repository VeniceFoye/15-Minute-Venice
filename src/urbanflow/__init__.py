from importlib import import_module

# Pure-Python re-exports
from .RasterGrid import RasterGrid
from .RasterGridWithPOIs import RasterGridWithPOIs

# Try loading the compiled extension lazily
try:
    path_planner = import_module("urbanflow.cpp.path_planner")
    path_between_pois = path_planner.path_between_pois
except ModuleNotFoundError:   # editable-install before build?
    path_between_pois = None

__all__ = [
    "RasterGrid",
    "RasterGridWithPOIs",
    "path_between_pois",
]
