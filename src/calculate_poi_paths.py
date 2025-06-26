import heapq
import numpy as np

# ----------------------------------------------------------------------
# PARAMETERS (tweak freely)
# ----------------------------------------------------------------------
STREET_CODE     = 1
COURTYARD_CODE  = 4
COST_STREET     = 1          # weight per step on street
COST_COURTYARD  = 3          # weight per step on courtyard
DIAGONALS       = True      # change to True if you allow 8-neighbour moves

# ----------------------------------------------------------------------
# A* (or Dijkstra when h=0) path planner
# ----------------------------------------------------------------------
def path_from_poi(
    grid: np.ndarray,
    sr: int, sc: int,     # start cell (row, col) – e.g. POI A row_adj/col_adj
    tr: int, tc: int,     # target cell            – e.g. POI B row_adj/col_adj
):
    """
    Return (path_r, path_c) – arrays of rows/cols from first step to dest.
    If no path exists → return None.
    """
    H, W = grid.shape

    # neighbourhood
    NBH = [(-1, 0, COST_STREET), (1, 0, COST_STREET),
           ( 0,-1, COST_STREET), (0, 1, COST_STREET)]
    if DIAGONALS:
        diag_cost = np.hypot(1, 1) * COST_STREET
        NBH += [(-1,-1, diag_cost), (-1,1, diag_cost),
                ( 1,-1, diag_cost), ( 1,1, diag_cost)]

    # admissible A* heuristic: Manhattan × min(costs)
    min_cost = min(COST_STREET, COST_COURTYARD)
    def h(r, c):
        return (abs(r - tr) + abs(c - tc)) * min_cost

    # open list: (f, g, r, c)
    pq = [(h(sr, sc), 0, sr, sc)]
    came = {}
    g_score = { (sr, sc): 0 }

    while pq:
        f, g_cur, r, c = heapq.heappop(pq)
        if (r, c) == (tr, tc):
            break

        for dr, dc, base_cost in NBH:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < H and 0 <= nc < W):
                continue

            cell = grid[nr, nc]
            if cell == STREET_CODE:
                step_cost = base_cost
            elif cell == COURTYARD_CODE:
                step_cost = base_cost * COST_COURTYARD / COST_STREET
            else:
                continue  # blocked (building, canal, ocean…)

            ng = g_cur + step_cost
            if ng < g_score.get((nr, nc), 1e30):
                g_score[(nr, nc)] = ng
                f_new = ng + h(nr, nc)
                heapq.heappush(pq, (f_new, ng, nr, nc))
                came[(nr, nc)] = (r, c)

    else:
        # exhausted queue → no path
        return None

    # reconstruct (excluding origin)
    rev_r, rev_c = [], []
    v = (tr, tc)
    while v != (sr, sc):
        rev_r.append(v[0]); rev_c.append(v[1])
        v = came[v]
    rev_r.reverse(); rev_c.reverse()

    return np.array(rev_r, dtype=np.int32), np.array(rev_c, dtype=np.int32)


