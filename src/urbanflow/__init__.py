from importlib import import_module

# Pure-Python re-exports
from .RasterGrid import RasterGrid
from .RasterGridWithPOIs import RasterGridWithPOIs

# Try loading the compiled extension lazily
try:
    path_planner = import_module("urbanflow.cpp.path_planner")
    path_from_poi = path_planner.path_from_poi
except ModuleNotFoundError:   # editable-install before build?
    path_from_poi = None

__all__ = [
    "RasterGrid",
    "RasterGridWithPOIs",
    "path_from_poi",
]
