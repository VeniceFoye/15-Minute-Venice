# 15-Minute-Venice Data Readme

There are multiple types of data we have collected due to this project, and this folder should contain all of them. This will serve as a documentation of the data files: what they are, and how to read/use them.

#### A quick note on File Types

- `.feather` files: a fast version of a GeoPandas or Pandas dataframe. You need to have `pyarrow` (`pip install pyarrow`) in order to read them. GeoPandas has a function to read these files:
```
needs = gpd.read_feather("needs_no_duplicates.feather")
```
- `.npz` files: these serve as compressed numpy dictionaries: to use the data, load them with `np.load(filepath)`, and then access their stored numpy arrays as you would a python dictionary.


## The Data Files
- `grid_with_legend.npz`: an npz file that has two keys: `grid` and `legend`. This is a rasterization of the 1808 Cadaster map into grid squares of 1x1 meter. It uses the following legend: 
    + 0 = ocean   (background)
    + 1 = street
    + 2 = building      
    + 3 = canal         
    + 4 = courtyard

- `catastici_adjusted_feather.feather`: The Catastici dataset, with the `row_adj` and `col_adj` columns, which are the rows and columns on the `grid` of the point, when snapped to the nearest building-street intersection. Deprecated by `needs_no_duplicates.feather`
-`needs_no_duplicates.feather`: The Catastici dataset, except with information about the different classes' needs, tenancy, and ownership.
- **The three "paths" zip files**: These directories contain the paths of a single parish of all the houses to their respective tenants' needs.  The different paths take the form of `.npz` files with `row` and `column` keys to indicate the steps of a single path.  These directories each have a `connection_index.csv`, which has information about each path:
    + The origin CASA `uid`
    + The origin type (I believe only CASA right now)
    + The target `uid`
    + The target type (this is the mid-level, standardized bottega function)
- `bottega_needs_by_class.feather`: This has a row for every Bottega type, and three columns for every class, indicating whether or not this class needs this specific bottega type. 
