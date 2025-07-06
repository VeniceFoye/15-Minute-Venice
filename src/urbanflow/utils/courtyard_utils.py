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
    Mark every empty cell that is *not* 8-connected to the border as courtyard.
    Works in-place and is ~10× faster than the label-and-loop version.
    """
    if structure is None:
        structure = np.ones((3, 3), dtype=bool)      # 8-connectivity

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
