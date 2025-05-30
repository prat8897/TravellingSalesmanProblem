#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <algorithm>
#include <limits>
#include <bitset>
#include <map>
#include <fstream>
#include <string>
#include <sstream>
#include <set>
#include <cmath>
#include <utility>
#include <omp.h>  // OpenMP for multithreading
#include <unordered_map>  // For memoization
#include <filesystem>  // For creating directories
#include <iomanip>     // For formatted output

struct Point {
    double x, y;
};

// Calculate distance between two points directly
inline double calculate_distance(const Point& p1, const Point& p2) {
    double dx = p1.x - p2.x;
    double dy = p1.y - p2.y;
    return std::sqrt(dx*dx + dy*dy);
}

// Hash function for vectors to use in unordered_map
struct VectorHash {
    size_t operator()(const std::vector<int>& v) const {
        std::hash<int> hasher;
        size_t seed = 0;
        for (int i : v) {
            seed ^= hasher(i) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        }
        return seed;
    }
};

// Generate random points in [0,1] x [0,1]
std::vector<Point> generate_random_instance(int n) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);
    
    std::vector<Point> points(n);
    for (int i = 0; i < n; i++) {
        points[i].x = dis(gen);
        points[i].y = dis(gen);
    }
    return points;
}

// Function to calculate total distance of a cycle
double total_cycle_distance(const std::vector<int>& cycle, const std::vector<Point>& points) {
    double distance = 0.0;
    int n = cycle.size();
    for (int i = 0; i < n; ++i) {
        int from = cycle[i];
        int to = cycle[(i + 1) % n];
        distance += calculate_distance(points[from], points[to]);
    }
    return distance;
}

// Generate SVG representation of a TSP cycle
std::string generate_svg(const std::vector<int>& cycle, const std::vector<Point>& points, 
                        double distance, const std::string& title = "TSP Solution") {
    // Calculate bounds for scaling
    double min_x = 1.0, min_y = 1.0, max_x = 0.0, max_y = 0.0;
    for (const auto& p : points) {
        min_x = std::min(min_x, p.x);
        min_y = std::min(min_y, p.y);
        max_x = std::max(max_x, p.x);
        max_y = std::max(max_y, p.y);
    }
    
    // Add some padding
    min_x -= 0.05; min_y -= 0.05;
    max_x += 0.05; max_y += 0.05;
    
    const int SVG_WIDTH = 800;
    const int SVG_HEIGHT = 600;
    const int POINT_RADIUS = 4;
    const int TEXT_SIZE = 12;
    
    // Scale factors
    double scale_x = SVG_WIDTH / (max_x - min_x);
    double scale_y = SVG_HEIGHT / (max_y - min_y);
    
    // Transform a point from problem space to SVG space
    auto transform = [&](const Point& p) -> std::pair<int, int> {
        int x = static_cast<int>((p.x - min_x) * scale_x);
        int y = static_cast<int>((p.y - min_y) * scale_y);
        return {x, y};
    };
    
    std::stringstream svg;
    
    // SVG header
    svg << "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"no\"?>\n";
    svg << "<svg width=\"" << SVG_WIDTH << "\" height=\"" << SVG_HEIGHT 
        << "\" xmlns=\"http://www.w3.org/2000/svg\">\n";
    
    // Background
    svg << "  <rect width=\"100%\" height=\"100%\" fill=\"white\"/>\n";
    
    // Title and distance
    svg << "  <text x=\"10\" y=\"20\" font-family=\"Arial\" font-size=\"16\">" 
        << title << "</text>\n";
    svg << "  <text x=\"10\" y=\"40\" font-family=\"Arial\" font-size=\"14\">"
        << "Distance: " << std::fixed << std::setprecision(6) << distance << "</text>\n";
    
    // Draw edges
    for (size_t i = 0; i < cycle.size(); i++) {
        int from = cycle[i];
        int to = cycle[(i + 1) % cycle.size()];
        
        auto [x1, y1] = transform(points[from]);
        auto [x2, y2] = transform(points[to]);
        
        svg << "  <line x1=\"" << x1 << "\" y1=\"" << y1 
            << "\" x2=\"" << x2 << "\" y2=\"" << y2 
            << "\" stroke=\"blue\" stroke-width=\"2\"/>\n";
    }
    
    // Draw points
    for (size_t i = 0; i < points.size(); i++) {
        auto [x, y] = transform(points[i]);
        
        // Circle for the point
        svg << "  <circle cx=\"" << x << "\" cy=\"" << y 
            << "\" r=\"" << POINT_RADIUS << "\" fill=\"red\"/>\n";
        
        // Label with index
        svg << "  <text x=\"" << x + POINT_RADIUS + 2 << "\" y=\"" << y - POINT_RADIUS - 2
            << "\" font-family=\"Arial\" font-size=\"" << TEXT_SIZE << "\">" 
            << i << "</text>\n";
    }
    
    // SVG footer
    svg << "</svg>\n";
    
    return svg.str();
}

// Save SVG to file
void save_svg(const std::string& svg_content, const std::string& filename) {
    std::ofstream file(filename);
    if (file.is_open()) {
        file << svg_content;
        file.close();
        std::cout << "Saved SVG to " << filename << std::endl;
    } else {
        std::cerr << "Failed to open file for writing: " << filename << std::endl;
    }
}

// For comparison - dynamic programming approach with on-the-fly distance calculation
double held_karp(const std::vector<Point>& points) {
    int n = points.size();
    int num_subsets = 1 << n;
    
    // dp[S][i] = minimum cost path visiting all vertices in subset S, ending at vertex i
    std::vector<std::vector<double>> dp(num_subsets, std::vector<double>(n, std::numeric_limits<double>::infinity()));
    
    // Base cases: paths from vertex 0 to each other vertex
    #pragma omp parallel for
    for (int j = 0; j < n; j++) {
        dp[1 << j][j] = j == 0 ? 0 : calculate_distance(points[0], points[j]);
    }
    
    // Iterate through subsets by size
    for (int size = 2; size <= n; size++) {
        // Get all subsets of size 'size' containing vertex 0
        std::vector<int> subsets_of_size;
        for (int mask = 0; mask < num_subsets; mask++) {
            if (__builtin_popcount(mask) == size && (mask & 1)) {  // Check if size matches and contains vertex 0
                subsets_of_size.push_back(mask);
            }
        }
        
        // Process subsets of current size in parallel
        #pragma omp parallel for schedule(dynamic)
        for (int subset_idx = 0; subset_idx < subsets_of_size.size(); subset_idx++) {
            int mask = subsets_of_size[subset_idx];
            std::vector<std::vector<double>> thread_local_updates(n, std::vector<double>(n, std::numeric_limits<double>::infinity()));
            
            // Try all possible end vertices
            for (int end = 0; end < n; end++) {
                if (!(mask & (1 << end))) continue;
                
                // Consider all possible previous vertices
                for (int prev = 0; prev < n; prev++) {
                    if (prev == end || !(mask & (1 << prev))) continue;
                    
                    int prev_mask = mask ^ (1 << end);  // Remove end vertex to get previous subset
                    double candidate = dp[prev_mask][prev] + calculate_distance(points[prev], points[end]);
                    thread_local_updates[end][prev] = std::min(thread_local_updates[end][prev], candidate);
                }
            }
            
            // Update global dp table with thread-local results
            #pragma omp critical
            {
                for (int end = 0; end < n; end++) {
                    for (int prev = 0; prev < n; prev++) {
                        if (thread_local_updates[end][prev] < dp[mask][end]) {
                            dp[mask][end] = thread_local_updates[end][prev];
                        }
                    }
                }
            }
        }
    }
    
    // Find the minimum cost cycle by connecting back to vertex 0
    double min_cost = std::numeric_limits<double>::infinity();
    int full_set = num_subsets - 1;
    
    #pragma omp parallel
    {
        double local_min = std::numeric_limits<double>::infinity();
        
        #pragma omp for nowait
        for (int last = 1; last < n; last++) {
            if (dp[full_set][last] < std::numeric_limits<double>::infinity()) {
                local_min = std::min(local_min, dp[full_set][last] + calculate_distance(points[last], points[0]));
            }
        }
        
        #pragma omp critical
        {
            if (local_min < min_cost) {
                min_cost = local_min;
            }
        }
    }
    
    return min_cost;
}

// Function to sort and canonicalize a partial cycle to use as a key
std::vector<int> canonicalize_partial_cycle(const std::vector<int>& cycle) {
    if (cycle.size() <= 2) return cycle;
    
    // Find the minimum element
    auto min_it = std::min_element(cycle.begin(), cycle.end());
    
    // Rotate to start with the minimum element
    std::vector<int> canonical(cycle.size());
    std::rotate_copy(cycle.begin(), min_it, cycle.end(), canonical.begin());
    
    // Check if the reverse order would be lexicographically smaller
    std::vector<int> reversed = canonical;
    std::reverse(reversed.begin() + 1, reversed.end());
    
    // Use the lexicographically smaller one
    if (reversed < canonical) {
        return reversed;
    }
    return canonical;
}

// Create a key for a partial cycle and remaining points
std::string create_memoization_key(
    const std::vector<int>& partial_cycle, 
    const std::set<int>& remaining_points) {
    
    // Canonicalize the partial cycle
    std::vector<int> canonical_cycle = canonicalize_partial_cycle(partial_cycle);
    
    // Create a string key
    std::string key;
    for (int p : canonical_cycle) {
        key += std::to_string(p) + "_";
    }
    key += "R";
    for (int p : remaining_points) {
        key += "_" + std::to_string(p);
    }
    
    return key;
}

// Helper function: Incremental Insertion without Lookahead
std::vector<int> build_cycle_incremental(
    const std::vector<int>& current_cycle, 
    std::set<int> remaining_points, 
    const std::vector<Point>& points) {
    
    std::vector<int> cycle = current_cycle;

    while (!remaining_points.empty()) {
        int best_r = -1;
        int best_insertion_position = -1;
        double best_delta_distance = std::numeric_limits<double>::infinity();

        // Iterate over each remaining point r
        for (int r : remaining_points) {
            // Find the best insertion position for r based on delta distance
            for (int i = 0; i < cycle.size(); ++i) {
                int p = cycle[i];
                int q = cycle[(i + 1) % cycle.size()];

                // Calculate the incremental distance
                double existing_edge = calculate_distance(points[p], points[q]);
                double new_edges = calculate_distance(points[p], points[r]) + calculate_distance(points[r], points[q]);
                double delta = new_edges - existing_edge;

                // Update if this is the best (smallest) delta found so far
                if (delta < best_delta_distance) {
                    best_delta_distance = delta;
                    best_r = r;
                    best_insertion_position = i + 1;
                }
            }
        }

        if (best_r != -1) {
            // Insert the best r into the cycle at the best position
            cycle.insert(cycle.begin() + best_insertion_position, best_r);
            remaining_points.erase(best_r);
        } else {
            // If no suitable point found, break to avoid infinite loop
            break;
        }
    }

    return cycle;
}

// Modified Function with Lookahead and Memoization
std::vector<int> build_cycle_least_distance_updated(
    const std::vector<int>& start_edge, 
    std::set<int> remaining_points, 
    const std::vector<Point>& points,
    std::unordered_map<std::string, std::pair<double, std::vector<int>>>& memo) {
    
    std::vector<int> cycle = start_edge;

    while (!remaining_points.empty()) {
        int best_r = -1;
        int best_insertion_position = -1;
        double best_total_distance = std::numeric_limits<double>::infinity();

        // Iterate over each remaining point r
        for (int r : remaining_points) {
            // Iterate over each possible insertion position for r
            for (int i = 0; i < cycle.size(); ++i) {
                int p = cycle[i];
                int q = cycle[(i + 1) % cycle.size()];

                // Simulate inserting r between p and q
                std::vector<int> temp_cycle = cycle;
                temp_cycle.insert(temp_cycle.begin() + i + 1, r);
                std::set<int> temp_remaining = remaining_points;
                temp_remaining.erase(r);

                // Create a memoization key
                std::string key_str = create_memoization_key(temp_cycle, temp_remaining);

                double simulated_distance;
                std::vector<int> simulated_cycle;

                // Check if we've already computed this partial cycle
                if (memo.find(key_str) != memo.end()) {
                    simulated_distance = memo[key_str].first;
                    simulated_cycle = memo[key_str].second;
                } else {
                    // Continue inserting the rest of the points using incremental heuristic
                    simulated_cycle = build_cycle_incremental(temp_cycle, temp_remaining, points);
                    
                    // Calculate total distance of the simulated cycle
                    simulated_distance = total_cycle_distance(simulated_cycle, points);
                    
                    // Store in memo
                    memo[key_str] = {simulated_distance, simulated_cycle};
                }

                // Update if this is the best (smallest) total distance found so far
                if (simulated_distance < best_total_distance) {
                    best_total_distance = simulated_distance;
                    best_r = r;
                    best_insertion_position = i + 1;
                }
            }
        }

        if (best_r != -1) {
            // Insert the best r into the cycle at the best position
            cycle.insert(cycle.begin() + best_insertion_position, best_r);
            remaining_points.erase(best_r);
        } else {
            // If no suitable point found, break to avoid infinite loop
            break;
        }
    }

    return cycle;
}

// Structure to hold the result of processing an edge
struct CycleResult {
    double distance;
    std::vector<int> cycle;
};

// Function to process a single edge with memoization
CycleResult process_edge(
    const std::pair<int, int>& edge, 
    int n, 
    const std::vector<Point>& points,
    std::unordered_map<std::string, std::pair<double, std::vector<int>>>& thread_memo) {
    
    std::set<int> remaining;
    for (int i = 0; i < n; ++i) {
        remaining.insert(i);
    }
    remaining.erase(edge.first);
    remaining.erase(edge.second);

    std::vector<int> start_edge = { edge.first, edge.second };
    std::vector<int> cycle = build_cycle_least_distance_updated(start_edge, remaining, points, thread_memo);

    // Ensure the cycle includes all points
    if (cycle.size() == n) {
        double distance = total_cycle_distance(cycle, points);
        CycleResult result;
        result.distance = distance;
        result.cycle = cycle;
        return result;
    } else {
        // No complete cycle found
        return CycleResult{ -1.0, {} };
    }
}

// Function to generate all possible edges (pairs of distinct points)
std::vector<std::pair<int, int>> generate_edges(int n) {
    std::vector<std::pair<int, int>> edges;
    edges.reserve(n * (n-1) / 2); // Pre-allocate memory
    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {
            edges.emplace_back(i, j);
        }
    }
    return edges;
}

// Dynamic lookahead insertion algorithm, now returns the best cycle as well
std::pair<double, std::vector<int>> dynamic_lookahead_insertion(const std::vector<Point>& points) {
    int n = points.size();
    
    // Generate all possible edges
    std::vector<std::pair<int, int>> edges = generate_edges(n);

    // Initialize variables to store the best cycle found
    double global_best_distance = std::numeric_limits<double>::infinity();
    std::vector<int> global_best_cycle;

    // Thread-local memoization
    using MemoType = std::unordered_map<std::string, std::pair<double, std::vector<int>>>;
    std::vector<MemoType> thread_memos(omp_get_max_threads());
    
    // Use a thread-safe counter for load balancing
    std::atomic<int> counter(0);
    
    #pragma omp parallel
    {
        int thread_id = omp_get_thread_num();
        MemoType& thread_memo = thread_memos[thread_id];
        
        // Each thread maintains its own best distance and cycle
        double local_best_distance = std::numeric_limits<double>::infinity();
        std::vector<int> local_best_cycle;

        int local_counter;
        while ((local_counter = counter++) < edges.size()) {
            const auto& edge = edges[local_counter];
            
            CycleResult result = process_edge(edge, n, points, thread_memo);
            
            if (result.distance >= 0.0) {
                if (result.distance < local_best_distance) {
                    local_best_distance = result.distance;
                    local_best_cycle = result.cycle;
                }
            }
            
            // Every few iterations, share the best results with other threads
            if (local_counter % 10 == 0) {
                #pragma omp critical
                {
                    if (local_best_distance < global_best_distance) {
                        global_best_distance = local_best_distance;
                        global_best_cycle = local_best_cycle;
                    }
                }
            }
        }

        // Final update of global best
        #pragma omp critical
        {
            if (local_best_distance < global_best_distance) {
                global_best_distance = local_best_distance;
                global_best_cycle = local_best_cycle;
            }
        }
    }

    return {global_best_distance, global_best_cycle};
}

void print_usage() {
    std::cout << "Usage: ./tsp [options]\n";
    std::cout << "Options:\n";
    std::cout << "  -n SIZE            Set the size of the problem (number of points) [default: 14]\n";
    std::cout << "  -i INSTANCES       Number of random instances to test [default: 1]\n";
    std::cout << "  -p, --plot         Generate SVG plots of the solutions\n";
    std::cout << "  -d, --plot-dir DIR Set the directory for plot outputs [default: plots]\n";
    std::cout << "  -h, --help         Show this help message\n";
}

int main(int argc, char* argv[]) {
    int NUM_INSTANCES = 1; // Default
    int N = 14; // Default
    bool generate_plots = false;
    std::string plot_dir = "plots";
    
    // Parse command line arguments
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "-n" && i + 1 < argc) {
            N = std::stoi(argv[++i]);
        } else if (arg == "-i" && i + 1 < argc) {
            NUM_INSTANCES = std::stoi(argv[++i]);
        } else if (arg == "-p" || arg == "--plot") {
            generate_plots = true;
        } else if ((arg == "-d" || arg == "--plot-dir") && i + 1 < argc) {
            plot_dir = argv[++i];
        } else if (arg == "-h" || arg == "--help") {
            print_usage();
            return 0;
        }
    }
    
    std::cout << "Testing " << NUM_INSTANCES << " random instances of size " << N << std::endl;
    
    // Set the number of threads to use
    int num_threads = omp_get_max_threads();
    std::cout << "Using " << num_threads << " threads" << std::endl;
    
    // Create plots directory if needed
    if (generate_plots) {
        namespace fs = std::filesystem;
        if (!fs::exists(plot_dir)) {
            std::cout << "Creating directory: " << plot_dir << std::endl;
            fs::create_directories(plot_dir);
        }
    }
    
    // Add timing code
    auto start_time = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < NUM_INSTANCES; i++) {
        std::cout << "\nProcessing instance " << i << std::endl;
        auto points = generate_random_instance(N);
        
        std::vector<int> optimal_cycle;
        std::vector<int> heuristic_cycle;
        
        // For small instances, compute optimal solution
        bool compute_optimal = (N <= 20); // Only compute optimal for N <= 20
        double optimal = 0.0;
        
        if (compute_optimal) {
            auto hk_start = std::chrono::high_resolution_clock::now();
            optimal = held_karp(points);
            auto hk_end = std::chrono::high_resolution_clock::now();
            auto hk_time = std::chrono::duration_cast<std::chrono::milliseconds>(hk_end - hk_start).count();
            std::cout << "Held-Karp time: " << hk_time << " ms, optimal distance: " << optimal << std::endl;
            
            // We don't get the optimal cycle from Held-Karp, we'd need to reconstruct it
            // which is beyond the scope of this implementation
        }
        
        auto dl_start = std::chrono::high_resolution_clock::now();
        auto [dynamic_lookahead, dl_cycle] = dynamic_lookahead_insertion(points);
        auto dl_end = std::chrono::high_resolution_clock::now();
        auto dl_time = std::chrono::duration_cast<std::chrono::milliseconds>(dl_end - dl_start).count();
        std::cout << "Dynamic Lookahead time: " << dl_time << " ms, distance: " << dynamic_lookahead << std::endl;
        
        heuristic_cycle = dl_cycle;
        
        if (compute_optimal) {
            if (dynamic_lookahead < optimal - 1e-10) {  // Allow for small numerical errors
                std::cout << "Found suboptimal solution on instance " << i << std::endl;
                std::cout << "Dynamic Lookahead solution: " << dynamic_lookahead << std::endl;
                std::cout << "Optimal solution: " << optimal << std::endl;
                break;
            } 
            else {
                std::cout << "Solution quality: " << (dynamic_lookahead/optimal) << std::endl;
            }
        }
        
        // Generate plots if requested
        if (generate_plots) {
            std::string heuristic_svg = generate_svg(heuristic_cycle, points, dynamic_lookahead, 
                                                    "Dynamic Lookahead Solution");
            
            std::string heuristic_filename = plot_dir + "/instance_" + std::to_string(i) + 
                                            "_n" + std::to_string(N) + "_lookahead.svg";
            save_svg(heuristic_svg, heuristic_filename);
            
            // Also save point coordinates for reference
            std::string points_filename = plot_dir + "/instance_" + std::to_string(i) + 
                                        "_n" + std::to_string(N) + "_points.txt";
            std::ofstream points_file(points_filename);
            if (points_file.is_open()) {
                points_file << "# Point coordinates for instance " << i << ", n=" << N << std::endl;
                for (int j = 0; j < N; j++) {
                    points_file << j << " " << std::fixed << std::setprecision(10) 
                               << points[j].x << " " << points[j].y << std::endl;
                }
                points_file.close();
                std::cout << "Saved points to " << points_filename << std::endl;
            }
        }
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto total_time = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time).count();
    std::cout << "\nTesting complete. Total time: " << total_time << " seconds" << std::endl;
    
    return 0;
}