"""
RasterGrid class. Stores the following information:
1. `grid`: an np.array with integer values.
2. `transform` an Affine transform object to transform points onto the grid
3. `legend`: a python dict describing the different grid values and their meaning.

Interface:

__init__: constructor

.save(filepath) : saving the grid, as small as possible

.laod(filepath) : load the grid, as quickly as possible
"""

class RasterGrid:

