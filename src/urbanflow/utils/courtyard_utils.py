"""
courtyard_utils.py

Utilities for loading and generating courtyard files
"""
from scipy.ndimage import binary_propagation #https://docs.scipy.org/doc/scipy/reference/ndimage.html#morphology
import numpy as np


def add_auto_courtyards(grid: np.ndarray,
                        *,
                        empty_code: int = 0,
                        courtyard_code: int = 4,
                        structure: np.ndarray | None = None):
    """
    Automatically label enclosed empty cells as courtyards.

    Any cell in ``grid`` with value ``empty_code`` that is *not* 8-connected
    to the raster border is reclassified as ``courtyard_code``. The operation
    is performed in-place using fast morphological propagation.

    Parameters
    ----------
    grid : ndarray of int
        2D raster array representing land-use classes or categories.
        Modified in-place.
    empty_code : int, optional
        Value used to indicate empty cells, by default ``0``.
    courtyard_code : int, optional
        Value used to indicate courtyard cells, by default ``4``.
    structure : ndarray of bool, optional
        Structuring element defining connectivity for propagation.
        Defaults to a 3×3 matrix of ones (8-connectivity).

    Returns
    -------
    grid : ndarray of int
        Same array as input, with enclosed empty cells relabeled as courtyards.

    Notes
    -----
    - Connectivity is determined using ``scipy.ndimage.binary_propagation``.
    - Empty cells connected to the border remain unchanged.
    - Empty cells fully enclosed by structures are converted into courtyards.

    Examples
    --------
    >>> import numpy as np
    >>> from urbanflow.utils.courtyard_utils import add_auto_courtyards
    >>> grid = np.array([
    ...     [0,0,0,0],
    ...     [0,2,2,0],
    ...     [0,2,0,0],
    ...     [0,0,0,0],
    ... ], dtype=np.uint8)
    >>> add_auto_courtyards(grid, empty_code=0, courtyard_code=4)
    array([[0, 0, 0, 0],
           [0, 2, 2, 0],
           [0, 2, 4, 0],
           [0, 0, 0, 0]], dtype=uint8)
    """
    if structure is None:
        structure = np.ones((3, 3), dtype=bool)      # 8-connectivity / diagonals

    mask = (grid == empty_code)

    # ---- seed: empty cells on the border -----------------------------------
    seed = np.zeros_like(mask, dtype=bool)
    seed[0, :]  = mask[0, :]
    seed[-1, :] = mask[-1, :]
    seed[:, 0]  = mask[:, 0]
    seed[:, -1] = mask[:, -1]

    # ---- propagate the seed through the empty mask -------------------------
    connected = binary_propagation(seed, mask=mask, structure=structure)

    # ---- anything empty *and* not connected → courtyard --------------------
    grid[mask & ~connected] = courtyard_code
    return grid
