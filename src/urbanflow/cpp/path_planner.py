import numpy as np

def path_between_pois(grid, sr, sc, tr, tc, max_distance=-1, diagonals=True):
    r, c = sr, sc
    path_r = []
    path_c = []
    dr = 1 if tr > sr else -1
    while r != tr:
        r += dr
        path_r.append(r)
        path_c.append(c)
    dc = 1 if tc > sc else -1
    while c != tc:
        c += dc
        path_r.append(r)
        path_c.append(c)
    return np.array(path_r, dtype=np.int32), np.array(path_c, dtype=np.int32)
