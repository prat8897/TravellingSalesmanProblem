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

struct Point {
    double x, y;
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

std::vector<std::vector<double>> compute_distance_matrix(const std::vector<Point>& points) {
    int n = points.size();
    std::vector<std::vector<double>> dist(n, std::vector<double>(n));
    for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < n; j++) {
            double dx = points[i].x - points[j].x;
            double dy = points[i].y - points[j].y;
            dist[i][j] = dist[j][i] = std::sqrt(dx*dx + dy*dy);
        }
    }
    return dist;
}

// For comparison
double held_karp(const std::vector<std::vector<double>>& dist) {
    int n = dist.size();
    int num_subsets = 1 << n;
    
    // dp[S][i] = minimum cost path visiting all vertices in subset S, ending at vertex i
    std::vector<std::vector<double>> dp(num_subsets, std::vector<double>(n, std::numeric_limits<double>::infinity()));
    
    // Base cases: paths from vertex 0 to each other vertex
    #pragma omp parallel for
    for (int j = 0; j < n; j++) {
        dp[1 << j][j] = j == 0 ? 0 : dist[0][j];
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
                    double candidate = dp[prev_mask][prev] + dist[prev][end];
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
                local_min = std::min(local_min, dp[full_set][last] + dist[last][0]);
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

// Function to calculate total distance of a cycle
double total_cycle_distance(const std::vector<int>& cycle, const std::vector<std::vector<double>>& distance_matrix) {
    double distance = 0.0;
    int n = cycle.size();
    for (int i = 0; i < n; ++i) {
        int from = cycle[i];
        int to = cycle[(i + 1) % n];
        distance += distance_matrix[from][to];
    }
    return distance;
}

// Helper function: Incremental Insertion without Lookahead
std::vector<int> build_cycle_incremental(const std::vector<int>& current_cycle, std::set<int> remaining_points, const std::vector<std::vector<double>>& distance_matrix) {
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
                double delta = distance_matrix[p][r] + distance_matrix[r][q];

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

// Modified Function with Lookahead
std::vector<int> build_cycle_least_distance_updated(const std::vector<int>& start_edge, std::set<int> remaining_points, const std::vector<std::vector<double>>& distance_matrix) {
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

                // Continue inserting the rest of the points using incremental heuristic
                std::vector<int> simulated_cycle = build_cycle_incremental(temp_cycle, temp_remaining, distance_matrix);

                // Calculate total distance of the simulated cycle
                double simulated_distance = total_cycle_distance(simulated_cycle, distance_matrix);

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

// Function to process a single edge
CycleResult process_edge(const std::pair<int, int>& edge, int n, const std::vector<std::vector<double>>& distance_matrix) {
    std::set<int> remaining;
    for (int i = 0; i < n; ++i) {
        remaining.insert(i);
    }
    remaining.erase(edge.first);
    remaining.erase(edge.second);

    std::vector<int> start_edge = { edge.first, edge.second };
    std::vector<int> cycle = build_cycle_least_distance_updated(start_edge, remaining, distance_matrix);

    // Ensure the cycle includes all points
    if (cycle.size() == n) {
        double distance = total_cycle_distance(cycle, distance_matrix);
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
    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {
            edges.emplace_back(i, j);
        }
    }
    return edges;
}

double dynamic_lookahead_insertion(const std::vector<std::vector<double>>& dist) {
    int n = dist.size();
    // Generate all possible edges
    std::vector<std::pair<int, int>> edges = generate_edges(n);

    // Initialize variables to store the best cycle found
    double global_best_distance = std::numeric_limits<double>::infinity();
    std::vector<int> global_best_cycle;

    // Determine the number of threads to use
    int num_threads = omp_get_max_threads();

    // Parallel processing of edges using OpenMP
    #pragma omp parallel
    {
        // Each thread maintains its own best distance and cycle
        double local_best_distance = std::numeric_limits<double>::infinity();
        std::vector<int> local_best_cycle;

        #pragma omp for schedule(dynamic)
        for (int i = 0; i < edges.size(); ++i) {
            const auto& edge = edges[i];
            CycleResult result = process_edge(edge, n, dist);
            if (result.distance >= 0.0) {
                if (result.distance < local_best_distance) {
                    local_best_distance = result.distance;
                    local_best_cycle = result.cycle;
                }
            }
        }

        // After processing, update the global best if necessary
        #pragma omp critical
        {
            if (local_best_distance < global_best_distance) {
                global_best_distance = local_best_distance;
                global_best_cycle = local_best_cycle;
            }
        }
    }

    return global_best_distance;
}

int main() {
    const int NUM_INSTANCES = 100000;
    const int N = 16;  // Size of instances
    
    int suboptimal_count = 0;
    std::cout << "Testing " << NUM_INSTANCES << " random instances of size " << N << std::endl;
    
    for (int i = 0; i < NUM_INSTANCES; i++) {
        auto points = generate_random_instance(N);
        auto dist_matrix = compute_distance_matrix(points);
        
        double optimal = held_karp(dist_matrix);
        double dynamic_lookahead = dynamic_lookahead_insertion(dist_matrix);
        
        if (dynamic_lookahead < optimal - 1e-10) {  // Allow for small numerical errors
            std::cout << "Found suboptimal solution on instance " << i << std::endl;
            std::cout << "Dynamic Lookahead solution: " << dynamic_lookahead << std::endl;
            std::cout << "Optimal solution: " << optimal << std::endl;
            break;
        }
        
        if (i % 10 == 0) {
            std::cout << "Processed " << i << " instances..." << std::endl;
        }
    }
    
    std::cout << "Testing complete" << std::endl;
    return 0;
}