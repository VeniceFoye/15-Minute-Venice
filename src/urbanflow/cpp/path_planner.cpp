// path_planner.cpp ------------------------------------------------------------
// A* path-finder for street-/courtyard-grids with an optional max_distance
// cut-off and a clean PyBind11 interface so that you can call it directly
// from Python.
//
// Build (Linux/Mac):
//   c++ -O3 -std=c++17 -fPIC \                                    
//       $(python3 -m pybind11 --includes) path_planner.cpp         \
//       -shared -o path_planner$(python3-config --extension-suffix)
//
// Or with CMake, add pybind11 and this file to your target.
// ---------------------------------------------------------------------------
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <vector>
#include <queue>
#include <cmath>
#include <limits>
#include <cstdint>

namespace py = pybind11;

// ---------------------------------------------------------------------------
// Tunables – keep in sync with your Python constants if you tweak them there.
// ---------------------------------------------------------------------------
static constexpr uint8_t STREET_CODE    = 1;
static constexpr uint8_t COURTYARD_CODE = 4;
static constexpr double  COST_STREET    = 1.0;
static constexpr double  COST_COURTYARD = 3.0;  // multiplier relative to COST_STREET

struct PQNode {
    double f, g;  // f = g + h
    int r, c;
    bool operator<(const PQNode &o) const { return f > o.f; } // min-heap
};

// ---------------------------------------------------------------------------
// Core C++ implementation: returns true + fills row/col vectors on success.
// If the destination is unreachable within max_distance it returns false.
// ---------------------------------------------------------------------------
static bool path_between_pois_cpp(const uint8_t *grid, int H, int W,
                              int sr, int sc, int tr, int tc,
                              int max_distance, bool diagonals,
                              std::vector<int> &out_rows,
                              std::vector<int> &out_cols) {

    if (sr == tr && sc == tc) return true; // already there – empty path

    // --- neighbourhood ------------------------------------------------------
    struct Step { int dr, dc; double cost; };
    std::vector<Step> NBH = {
        {-1, 0,  COST_STREET}, {1, 0,  COST_STREET},
        { 0,-1,  COST_STREET}, {0, 1,  COST_STREET}
    };
    if (diagonals) {
        double diag_cost = std::sqrt(2.0) * COST_STREET;
        NBH.insert(NBH.end(), {
            {-1,-1, diag_cost}, {-1, 1, diag_cost},
            { 1,-1, diag_cost}, { 1, 1, diag_cost}
        });
    }

    // --- heuristic: Manhattan * cheapest step cost --------------------------
    auto h = [&](int r,int c){ return (std::abs(r-tr)+std::abs(c-tc)) * COST_STREET; };

    const int N = H * W;
    const double INF = std::numeric_limits<double>::infinity();
    std::vector<double> g_score(N, INF);
    std::vector<int>    came(N, -1);          // predecessor idx
    std::vector<int>    steps(N,  INT_MAX);   // #steps from start (for max_distance)

    auto idx  = [&](int r,int c){ return r*W + c; };
    auto row  = [&](int i){ return i / W; };
    auto col  = [&](int i){ return i % W; };

    int sidx = idx(sr,sc);
    g_score[sidx] = 0.0;
    steps[sidx]   = 0;

    std::priority_queue<PQNode> pq;
    pq.push({h(sr,sc),0.0,sr,sc});

    const int cut = (max_distance < 0) ? std::numeric_limits<int>::max() : max_distance;

    while (!pq.empty()) {
        auto [f_cur,g_cur,r,c] = pq.top(); pq.pop();
        int u = idx(r,c);
        if (r==tr && c==tc) break; // found

        if (g_cur > g_score[u] + 1e-9) continue; // skip stale entry

        if (steps[u] >= cut) continue; // depth limit

        for (const auto &st: NBH) {
            int nr = r + st.dr;
            int nc = c + st.dc;
            if (nr < 0 || nr >= H || nc < 0 || nc >= W) continue;

            uint8_t cell = grid[idx(nr,nc)];
            double step_cost;
            if (cell == STREET_CODE)        step_cost = st.cost;
            else if (cell == COURTYARD_CODE) step_cost = st.cost * (COST_COURTYARD / COST_STREET);
            else                             continue; // blocked

            int v = idx(nr,nc);
            double ng = g_cur + step_cost;
            int    ns = steps[u] + 1;

            if (ng + 1e-9 < g_score[v]) {
                g_score[v] = ng;
                steps[v]   = ns;
                came[v]    = u;
                double f = ng + h(nr,nc);
                pq.push({f,ng,nr,nc});
            }
        }
    }

    int tidx = idx(tr,tc);
    if (came[tidx] == -1) return false; // unreachable or beyond max_distance

    // --- reconstruct path (exclude the origin cell) -------------------------
    std::vector<int> rev;
    for (int v = tidx; v != sidx; v = came[v]) rev.push_back(v);

    out_rows.resize(rev.size());
    out_cols.resize(rev.size());
    for (std::size_t i = 0, j = rev.size(); i < rev.size(); ++i, --j) {
        int v = rev[j-1];
        out_rows[i] = row(v);
        out_cols[i] = col(v);
    }
    return true;
}

// ---------------------------------------------------------------------------
// PYBIND11 WRAPPER -----------------------------------------------------------
// ---------------------------------------------------------------------------
py::object path_between_pois_py(py::array_t<uint8_t, py::array::c_style | py::array::forcecast> grid,
                            int sr, int sc, int tr, int tc,
                            int max_distance = -1, bool diagonals = true) {
    // request buffer info & sanity checks
    py::buffer_info info = grid.request();
    if (info.ndim != 2) throw std::runtime_error("grid must be 2-D");
    int H = static_cast<int>(info.shape[0]);
    int W = static_cast<int>(info.shape[1]);

    const uint8_t *gptr = static_cast<const uint8_t*>(info.ptr);

    std::vector<int> rows, cols;
    bool ok = path_between_pois_cpp(gptr, H, W, sr, sc, tr, tc, max_distance,
                                diagonals, rows, cols);
    if (!ok) return py::none();

    // convert to NumPy arrays (uint32)
    py::array_t<uint32_t> rarr(rows.size());
    py::array_t<uint32_t> carr(cols.size());
    std::memcpy(rarr.mutable_data(), rows.data(), rows.size()*sizeof(uint32_t));
    std::memcpy(carr.mutable_data(), cols.data(), cols.size()*sizeof(uint32_t));

    return py::make_tuple(std::move(rarr), std::move(carr));
}

PYBIND11_MODULE(path_planner, m) {
    m.doc() = "Fast A* path-finder for street/courtyard grids with max_distance cut-off";
    m.def("path_between_pois", &path_between_pois_py,
          py::arg("grid"), py::arg("sr"), py::arg("sc"),
          py::arg("tr"), py::arg("tc"),
          py::arg("max_distance") = -1,
          py::arg("diagonals")    = true,
          R"pbdoc(
Compute shortest walkable path between two cells using A*.

Parameters
----------
grid : numpy.ndarray[uint8]
    2-D occupancy grid. 1=street (cheap), 4=courtyard (expensive), everything else = blocked.
sr, sc : int
    Start cell (row, col).
tr, tc : int
    Target cell (row, col).
max_distance : int, optional
    Abort search once more than this many steps from start have been expanded.
    Negative → unlimited (default).
diagonals : bool, optional
    Allow 8-neighbour moves (default True).

Returns
-------
(path_r, path_c) : tuple[np.ndarray[int32], np.ndarray[int32]]
    Row/col indices **excluding** the origin cell.  None if no path was found.)pbdoc");
}

