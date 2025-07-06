import heapq
import numpy as np
import itertools 
import math


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

    return np.array(rev_r, dtype=np.uint32), np.array(rev_c, dtype=np.uint32)



# ----------------------------------------------------------------------
# NEW:  multi-POI travelling-salesman wrapper
# ----------------------------------------------------------------------
def path_through_pois(
    grid: np.ndarray,
    start_rc: tuple[int,int],
    poi_rcs: list[tuple[int,int]],
):
    """
    Parameters
    ----------
    start_rc : (row, col) of the house
    poi_rcs  : list of (row, col) you want to visit

    Returns
    -------
    full_rows, full_cols : ndarray
        Concatenated path: house → POIs → house.
        Empty arrays if no complete tour is possible.
    order_idx : list[int]
        The visiting order *indices* into `poi_rcs`.
        Example with 3 POIs → [2, 0, 1] means
        house → poi[2] → poi[0] → poi[1] → house
    """
    nodes = [start_rc] + poi_rcs
    N     = len(nodes)

    # --------------------------------------------------
    # 1) pairwise shortest paths & costs
    # --------------------------------------------------
    paths    = {}    # (i,j) → (rows, cols)
    dist     = np.full((N, N), np.inf)
    for i, (ri, ci) in enumerate(nodes):
        for j, (rj, cj) in enumerate(nodes):
            if i == j:
                dist[i, j] = 0
                continue
            seg = path_from_poi(grid, ri, ci, rj, cj)
            if seg is None:          # graph is disconnected
                continue
            rows, cols = seg
            paths[(i, j)] = (rows, cols)
            dist[i, j]   = len(rows)   # path length = number of steps

    # if any POI is unreachable from start or another POI → abort
    if np.isinf(dist).any():
        return np.array([], dtype=int), np.array([], dtype=int), []

    # --------------------------------------------------
    # 2) solve TSP  (Held-Karp up to 11 POIs, else heuristic)
    # --------------------------------------------------
    m = len(poi_rcs)
    if m <= 11:                     # (m+1)! explodes quickly
        # bitmask DP – subsets are enumerated only over POIs
        FULL = 1 << m
        DP   = { (1<<k, k): (dist[0, k+1], 0) for k in range(m) }
        for subset_size in range(2, m+1):
            for subset in itertools.combinations(range(m), subset_size):
                mask = sum(1<<k for k in subset)
                for k in subset:
                    prev_mask = mask ^ (1<<k)
                    best = (math.inf, None)
                    for j in subset:
                        if j == k: continue
                        cand_cost = DP[(prev_mask, j)][0] + dist[j+1, k+1]
                        if cand_cost < best[0]:
                            best = (cand_cost, j)
                    DP[(mask, k)] = best

        # close the loop (back to start)
        best_cost, last = min(
            (DP[(FULL-1, k)][0] + dist[k+1, 0], k) for k in range(m)
        )

        # reconstruct visiting order (indices into poi_rcs)
        order = []
        mask  = FULL-1
        while last is not None:
            order.append(last)
            next_last = DP[(mask, last)][1]
            mask ^= 1<<last
            last = next_last
        order.reverse()

    else:
        # --------------------------------------------------
        # heuristic: nearest-neighbour + 2-opt improvement
        # --------------------------------------------------
        unused = set(range(m))
        cur    = 0
        order  = []
        while unused:
            # pick closest next POI
            nxt  = min(unused, key=lambda k: dist[cur+1, k+1])
            order.append(nxt)
            unused.remove(nxt)
            cur = nxt

        # simple 2-opt to polish
        improved = True
        while improved:
            improved = False
            for i in range(len(order)-1):
                for j in range(i+1, len(order)):
                    a, b = order[i-1] if i>0 else None, order[i]
                    c, d = order[j], order[(j+1)%len(order)] if j+1<len(order) else None
                    before = (dist[(a or 0)+1, b+1] if a is not None else dist[0, b+1]) \
                           + (dist[c+1, (d or 0)+1] if d is not None else dist[c+1, 0])
                    after  = (dist[(a or 0)+1, c+1] if a is not None else dist[0, c+1]) \
                           + (dist[b+1, (d or 0)+1] if d is not None else dist[b+1, 0])
                    if after < before:
                        order[i:j+1] = reversed(order[i:j+1])
                        improved = True

    # --------------------------------------------------
    # 3) stitch the segment paths together
    # --------------------------------------------------
    full_r, full_c = [], []
    cur_idx = 0   # start node
    for poi_idx in order:
        seg_r, seg_c = paths[(cur_idx, poi_idx+1)]
        full_r.extend(seg_r); full_c.extend(seg_c)
        cur_idx = poi_idx+1
    # return to start
    seg_r, seg_c = paths[(cur_idx, 0)]
    full_r.extend(seg_r); full_c.extend(seg_c)

    return np.array(full_r, dtype=np.int32), np.array(full_c, dtype=np.int32), order