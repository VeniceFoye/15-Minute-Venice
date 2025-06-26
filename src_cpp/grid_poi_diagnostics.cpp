// grid_poi_diagnostics_fast.cpp ------------------------------------------------
// Optimised diagnostics tool: exploits duplicate POI coordinates so that only
// one BFS is run per *unique* (row_adj,col_adj) origin.
// Builds a heat-map of edge usage for 22 000 random POI pairs.
//
// Build:
//   g++ -O2 -std=c++17 -fopenmp grid_poi_diagnostics_fast.cpp -o diag -lcnpy -lz
// -----------------------------------------------------------------------------
#include <cnpy.h>

#include <array>
#include <chrono>
#include <cctype>
#include <deque>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <algorithm>
#include <ctime>
#include <omp.h>

// ---------- constants ---------------------------------------------------------
constexpr uint8_t STREET_CODE = 1;    // passable road
constexpr uint8_t COURT_CODE  = 4;    // passable court
constexpr int     LOG_EVERY   = 10; // log every N processed pairs
constexpr int     QUERY_CT    = 22000;// number of random query pairs

struct POI { int row{}, col{}; std::string func; };
struct Coord { int r, c; };

// ---------- misc helpers -------------------------------------------------------
static std::string timestamp() {
    std::time_t t = std::time(nullptr);
    char buf[32]; std::strftime(buf, sizeof(buf), "%F %T", std::localtime(&t));
    return buf;
}

// pack two 32-bit signed ints into a single 64-bit key (row<<32 | col)
static inline uint64_t coord_key(int r, int c) {
    return (uint64_t)(uint32_t)r << 32 | (uint32_t)c;
}

// CSV helpers (compact; unchanged logic) ----------------------------------------
static std::vector<std::string> split_n(const std::string &line, std::size_t n) {
    std::vector<std::string> out; out.reserve(n);
    std::size_t start = 0, cur = 0, commas = 0;
    while (cur < line.size() && commas < n - 1) {
        if (line[cur] == ',') { out.emplace_back(line.substr(start, cur - start)); start = cur + 1; ++commas; }
        ++cur;
    }
    out.emplace_back(line.substr(start));
    while (out.size() < n) out.emplace_back("");
    return out;
}

static bool is_int(const std::string &s) {
    std::size_t i = 0; while (i < s.size() && std::isspace((unsigned char)s[i])) ++i;
    if (i == s.size()) return false;
    if (s[i] == '+' || s[i] == '-') ++i;
    bool ok = false;
    for (; i < s.size(); ++i) {
        if (!std::isdigit((unsigned char)s[i])) return false;
        ok = true;
    }
    return ok;
}

static std::vector<POI> load_pois(const std::string &csv) {
    std::ifstream in(csv);
    if (!in) throw std::runtime_error("Cannot open " + csv);

    std::string header; std::getline(in, header);
    std::vector<std::string> cols; { std::stringstream ss(header); std::string c; while (std::getline(ss, c, ',')) cols.push_back(c); }
    int ir = -1, ic = -1, ifn = -1;
    for (std::size_t i = 0; i < cols.size(); ++i) {
        if (cols[i] == "row_adj") ir = (int)i;
        else if (cols[i] == "col_adj") ic = (int)i;
        else if (cols[i] == "PP_Function_TOP") ifn = (int)i;
    }
    if (ir < 0 || ic < 0 || ifn < 0) throw std::runtime_error("Required columns missing");

    std::vector<POI> v; v.reserve(10000);
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        auto f = split_n(line, cols.size());
        if (!is_int(f[ir]) || !is_int(f[ic])) continue;
        v.push_back({ std::stoi(f[ir]), std::stoi(f[ic]), f[ifn] });
    }
    return v;
}

// ---------- grid / geometry helpers -------------------------------------------
static inline bool inside(int r, int c, int H, int W) {
    return r >= 0 && r < H && c >= 0 && c < W;
}

static void print_grid_stats(const cnpy::NpyArray &arr) {
    std::size_t H = arr.shape[0], W = arr.shape[1];
    std::cout << "Grid: " << H << " × " << W << " (" << H*W << " cells)\n";
    std::array<std::size_t, 256> hist{};
    const uint8_t *d = arr.data<uint8_t>();
    for (std::size_t i = 0; i < H * W; ++i) ++hist[d[i]];
    for (int c = 0; c < 256; ++c) if (hist[c]) std::cout << "  " << std::setw(3) << c << ": " << hist[c] << '\n';
}

// -----------------------------------------------------------------------------
// MAIN ------------------------------------------------------------------------
// -----------------------------------------------------------------------------
int main(int argc, char *argv[]) {
    if (argc != 2) { std::cerr << "Usage: " << argv[0] << " <export_dir>\n"; return 1; }
    const std::string dir = argv[1];

    try {
        const auto total_start = std::chrono::steady_clock::now();

        // ---- load data -------------------------------------------------------
        auto grid_npy = cnpy::npy_load(dir + "/grid.npy");
        if (grid_npy.word_size != 1 || grid_npy.shape.size() != 2) throw std::runtime_error("grid.npy bad format");
        const int H = (int)grid_npy.shape[0];
        const int W = (int)grid_npy.shape[1];
        const uint8_t *grid = grid_npy.data<uint8_t>();
        print_grid_stats(grid_npy);

        auto pois = load_pois(dir + "/pois.csv");
        if (pois.size() < 2) throw std::runtime_error("Need ≥2 POIs");
        std::size_t casa = 0; for (const auto &p: pois) if (p.func == "CASA") ++casa;
        std::cout << "POIs: " << pois.size() << ", CASA: " << casa << " (" << std::fixed << std::setprecision(2)
                  << 100.0 * casa / pois.size() << "%)\n";

        // ---- collapse duplicate coordinates ---------------------------------
        std::vector<Coord> coords; coords.reserve(pois.size());
        std::unordered_map<uint64_t,int> key2cid; key2cid.reserve(pois.size()*2);
        std::vector<int> poi2cid(pois.size());

        for (std::size_t i = 0; i < pois.size(); ++i) {
            uint64_t k = coord_key(pois[i].row, pois[i].col);
            auto it = key2cid.find(k);
            if (it == key2cid.end()) {
                int cid = (int)coords.size();
                key2cid.emplace(k, cid);
                coords.push_back({ pois[i].row, pois[i].col });
                poi2cid[i] = cid;
            } else {
                poi2cid[i] = it->second;
            }
        }
        std::cout << "Unique coordinates: " << coords.size() << '\n';

        // ---- generate QUERY_CT distinct random POI pairs ---------------------
        std::vector<std::pair<int,int>> pairs; pairs.reserve(QUERY_CT);
        std::unordered_set<uint64_t> seen; seen.reserve(QUERY_CT*2);
        std::mt19937 rng(12345);
        std::uniform_int_distribution<int> uni(0, (int)pois.size() - 1);
        auto pack = [&](int a, int b) { return (uint64_t)a << 32 | (uint32_t)b; };
        while (pairs.size() < QUERY_CT) {
            int a = uni(rng), b = uni(rng); if (a == b) continue; if (a > b) std::swap(a,b);
            if (seen.emplace(pack(a,b)).second) pairs.emplace_back(a,b);
        }

        // ---- bucket queries by origin coordinate ----------------------------
        std::vector<std::vector<int>> by_src(coords.size());
        for (int i = 0; i < (int)pairs.size(); ++i) by_src[ poi2cid[ pairs[i].first ] ].push_back(i);

        // ---- shared accumulators --------------------------------------------
        std::vector<uint32_t> heat((std::size_t)H*W, 0u);
        long long steps_sum = 0;
        long long found = 0;

        const auto t_start = std::chrono::steady_clock::now();

        // ---------------------------------------------------------------------
        // PARALLEL LOOP: one BFS per unique origin -----------------------------
        // ---------------------------------------------------------------------
#pragma omp parallel for schedule(dynamic,1) reduction(+:steps_sum,found)
        for (int cid = 0; cid < (int)coords.size(); ++cid) {
            const auto &bucket = by_src[cid];
            if (bucket.empty()) continue; // no query starts here

            if (cid % LOG_EVERY == 0 && omp_get_thread_num() == 0) {
                auto now = std::chrono::steady_clock::now();
                double sec = std::chrono::duration<double>(now - t_start).count();
                std::cout << '[' << timestamp() << "] " << cid << " / " << coords.size() << " origins done ("
                          << std::setprecision(2) << cid / sec << " BFS/s)\n";
            }

            const int sr = coords[cid].r, sc = coords[cid].c;
            const int sidx = sr * W + sc;
            std::vector<int> prev((std::size_t)H*W, -1);
            prev[sidx] = -2;

            // ---- BFS from this origin --------------------------------------
            std::deque<int> dq; dq.push_back(sidx);
            const int dr[4] = { -1, 1, 0, 0 };
            const int dc[4] = {  0, 0,-1, 1 };
            while (!dq.empty()) {
                int u = dq.front(); dq.pop_front();
                int r = u / W, c = u % W;
                for (int k = 0; k < 4; ++k) {
                    int nr = r + dr[k], nc = c + dc[k];
                    if (!inside(nr, nc, H, W)) continue;
                    uint8_t cell = grid[nr * W + nc];
                    if (cell != STREET_CODE && cell != COURT_CODE) continue;
                    int vidx = nr * W + nc;
                    if (prev[vidx] == -1) { prev[vidx] = u; dq.push_back(vidx); }
                }
            }

            // ---- answer every query starting at this origin -----------------
            for (int qi : bucket) {
                int pid_a = pairs[qi].first;
                int pid_b = pairs[qi].second;
                int tr = pois[pid_b].row, tc = pois[pid_b].col;
                int tidx = tr * W + tc;
                if (prev[tidx] == -1) continue; // unreachable

                ++found;
                // reconstruct path (tidx → sidx)
                std::vector<int> path;
                for (int v = tidx; v != -2; v = prev[v]) path.push_back(v);
                std::reverse(path.begin(), path.end());
                steps_sum += (int)path.size() - 1; // edges

                // update heatmap, skip first node (origin)
                for (std::size_t k = 1; k < path.size(); ++k) {
#pragma omp atomic
                    ++heat[path[k]];
                }
            }
        }

        std::cout << "Paths found: " << found << ", avg length: "
                  << (found ? steps_sum / double(found) : 0.0) << "\n";

        cnpy::npy_save(dir + "/path_heat.npy", heat.data(), { (size_t)H, (size_t)W }, "w");
        std::cout << "Heatmap saved to " << dir << "/path_heat.npy\n";

        const auto total_end = std::chrono::steady_clock::now();
        double total_sec = std::chrono::duration<double>(total_end - total_start).count();
        std::cout << "Total elapsed time: " << std::fixed << std::setprecision(2)
                  << total_sec << " seconds\n";

    } catch (const std::exception &ex) {
        std::cerr << "Error: " << ex.what() << '\n';
        return 2;
    }
    return 0;
}
