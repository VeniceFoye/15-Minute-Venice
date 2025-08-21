# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import os
import sys
from datetime import datetime

# -- Path setup: add project root so autodoc can import urbanflow ----------
ROOT = os.path.abspath(os.path.join(__file__, "..", "..", ".."))
sys.path.insert(0, ROOT)

project = '15-Minute-Venice'
copyright = '2025, Clay Foye'
author = 'Clay Foye'
release = '0.1.0'
copyright = f"{datetime.now():%Y}, {author}"


# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",   # <-- required for intersphinx_mapping
    "sphinx.ext.viewcode",
    "sphinx_autodoc_typehints",
    "myst_parser",
    "sphinx_copybutton",
]

autosummary_generate = True          # create stubs automatically
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "inherited-members": True,
    "show-inheritance": True,
}
autodoc_typehints = "description"    # put type hints into the description
napoleon_google_docstring = False    # we use NumPy style
napoleon_numpy_docstring = True
napoleon_attr_annotations = True

templates_path = ["_templates"]
exclude_patterns = []

# -- HTML --------------------------------------------------------------------
html_theme = "furo"    # or "sphinx_rtd_theme"
html_static_path = ["_static"]

# -- Intersphinx: link out to common libs -----------------------------------
intersphinx_mapping = {
    "python":   ("https://docs.python.org/3", None),
    "numpy":    ("https://numpy.org/doc/stable/", None),
    "scipy":    ("https://docs.scipy.org/doc/scipy/", None),
    "pandas":   ("https://pandas.pydata.org/pandas-docs/stable/", None),
    "geopandas":("https://geopandas.org/en/stable/", None),
    "shapely":  ("https://shapely.readthedocs.io/en/stable/", None),
}
