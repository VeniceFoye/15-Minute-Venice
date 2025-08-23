# tests/test_cpp_binding.py

import sys
import types
import pytest
import importlib
from unittest.mock import patch, MagicMock
import numpy as np

# --- helpers ---------------------------------------------------------------

def _reload_urbanflow():
    """Remove 'urbanflow' from sys.modules to force a clean re-import."""
    for key in [k for k in list(sys.modules) if k == "urbanflow" or k.startswith("urbanflow.")]:
        del sys.modules[key]
    return importlib.import_module("urbanflow")


# --- tests -----------------------------------------------------------------

def test_init_cpp_import_other_exception():
    """
    If the import of the C++ module raises a *different* ImportError (not ModuleNotFoundError),
    __init__ should *re-raise* it (not swallow).
    """
    real_import = importlib.import_module

    def side_effect(name, *args, **kwargs):
        if name == "urbanflow.cpp.path_planner":
            # Not a ModuleNotFoundError => should propagate
            raise ImportError("Some other import error")
        return real_import(name, *args, **kwargs)

    with patch("importlib.import_module", side_effect=side_effect):
        with pytest.raises(ImportError, match="Some other import error"):
            _reload_urbanflow()


def test_logging_config_existing_handlers():
    """
    Given a logger that already has handlers, setup_logger should early-return
    without adding duplicates.
    """
    # Import after ensuring package is importable
    uf = _reload_urbanflow()
    # setup_logger is defined in the package (adjust if you export it differently)
    setup_logger = uf.setup_logger

    import logging
    logger_name = "test_urbanflow_unique"
    test_logger = logging.getLogger(logger_name)

    # Ensure clean slate
    for h in list(test_logger.handlers):
        test_logger.removeHandler(h)

    # Add an existing handler
    existing = logging.StreamHandler()
    test_logger.addHandler(existing)

    # Call setup_logger — should not add more handlers
    result_logger = setup_logger(logger_name)

    try:
        assert result_logger is test_logger
        assert len(result_logger.handlers) == 1  # no duplicates
        assert result_logger.handlers[0] is existing
    finally:
        # cleanup – avoid bleeding handlers into other tests
        for h in list(test_logger.handlers):
            test_logger.removeHandler(h)


def test_init_cpp_module_import():
    """
    When the C++ module isn't available (ModuleNotFoundError), package should
    set `path_between_pois` to None instead of crashing. We enforce that
    deterministically by making only that import raise ModuleNotFoundError.
    """
    real_import = importlib.import_module

    def side_effect(name, *args, **kwargs):
        if name == "urbanflow.cpp.path_planner":
            raise ModuleNotFoundError("simulated missing extension")
        return real_import(name, *args, **kwargs)

    with patch("importlib.import_module", side_effect=side_effect):
        uf = _reload_urbanflow()
        assert getattr(uf, "path_between_pois", None) is None

def test_init_cpp_import_success():
    """
    Simulate a successful import of the C++ module by injecting a fake module
    in sys.modules (avoids breaking other imports like pandas/geopandas).
    If the real extension is present, this test should still pass by validating
    the interface (callable + tuple of same-length ndarrays).
    """
    # Prepare fake package + module objects
    fake_cpp_pkg = types.ModuleType("urbanflow.cpp")
    fake_mod = types.ModuleType("urbanflow.cpp.path_planner")
    # minimal API surface used by your package
    fake_mod.path_between_pois = MagicMock(
        return_value=(np.array([0, 1], dtype=np.uint32),
                      np.array([0, 1], dtype=np.uint32))
    )

    # Register in sys.modules so import resolves cleanly *if* the real one isn't there
    sys.modules["urbanflow.cpp"] = fake_cpp_pkg
    sys.modules["urbanflow.cpp.path_planner"] = fake_mod

    try:
        uf = _reload_urbanflow()

        # Must expose the symbol and be callable
        assert hasattr(uf, "path_between_pois")
        fn = uf.path_between_pois
        assert callable(fn)

        # Call it with a tiny valid grid and ensure it returns two ndarrays of equal length
        grid = np.ones((2, 2), dtype=np.uint8)  # all "street" cells
        r, c = fn(grid, 0, 0, 1, 1)

        # Accept both real extension and fake: types and shape must match
        assert isinstance(r, np.ndarray) and isinstance(c, np.ndarray)
        assert r.dtype.kind in ("i", "u") and c.dtype.kind in ("i", "u")
        assert r.shape == c.shape
        # Optional: ensure values are within grid bounds
        assert np.all((r >= 0) & (r < grid.shape[0]))
        assert np.all((c >= 0) & (c < grid.shape[1]))
    finally:
        # cleanup fakes
        sys.modules.pop("urbanflow.cpp.path_planner", None)
        sys.modules.pop("urbanflow.cpp", None)
