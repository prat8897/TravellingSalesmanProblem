#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <set>
#include <cmath>
#include <limits>
#include <algorithm>
#include <utility>
#include <chrono> // For timing
#include <omp.h>  // OpenMP for multithreading

// Structure to hold point data
struct Point {
    int id; // Node ID from the .tsp file
    double x;
    double y;
};

// Function to read a .tsp file and return a vector of Points
std::vector<Point> read_tsp_file(const std::string& filename) {
    std::vector<Point> points;
    std::ifstream infile(filename);
    if (!infile) {
        std::cerr << "Error opening file " << filename << std::endl;
        return points;
    }

    std::string line;
    bool in_node_section = false;
    while (std::getline(infile, line)) {
        if (line.find("NODE_COORD_SECTION") != std::string::npos) {
            in_node_section = true;
            continue;
        } else if (line.find("EOF") != std::string::npos) {
            break;
        }
        if (in_node_section) {
            std::istringstream iss(line);
            int id;
            double x, y;
            if (!(iss >> id >> x >> y)) {
                // Handle possible end of node section or malformed lines
                continue;
            }
            Point p;
            p.id = id - 1; // Adjusting ID to start from 0
            p.x = x;
            p.y = y;
            points.push_back(p);
        }
    }
    return points;
}

// Function to compute the Euclidean distance matrix
std::vector<std::vector<double>> compute_distance_matrix(const std::vector<Point>& points) {
    int n = points.size();
    std::vector<std::vector<double>> distance_matrix(n, std::vector<double>(n, 0.0));
    for (int i = 0; i < n; ++i) {
        const Point& p1 = points[i];
        for (int j = i + 1; j < n; ++j) {
            const Point& p2 = points[j];
            double dx = p1.x - p2.x;
            double dy = p1.y - p2.y;
            double dist = std::sqrt(dx * dx + dy * dy);
            distance_matrix[i][j] = dist;
            distance_matrix[j][i] = dist;
        }
    }
    return distance_matrix;
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

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: tsp_solver filename.tsp" << std::endl;
        return 1;
    }

    std::string filename = argv[1];
    std::vector<Point> points = read_tsp_file(filename);
    int n = points.size();
    if (n == 0) {
        std::cerr << "No points read from file." << std::endl;
        return 1;
    }

    // Start timing
    auto start_time = std::chrono::high_resolution_clock::now();

    // Compute distance matrix
    std::vector<std::vector<double>> distance_matrix = compute_distance_matrix(points);

    // Generate all possible edges
    std::vector<std::pair<int, int>> edges = generate_edges(n);
    std::cout << "Total number of edges to process: " << edges.size() << std::endl;

    // Initialize variables to store the best cycle found
    double global_best_distance = std::numeric_limits<double>::infinity();
    std::vector<int> global_best_cycle;

    // Determine the number of threads to use
    int num_threads = omp_get_max_threads();
    std::cout << "Using " << num_threads << " threads for processing." << std::endl;

    // Parallel processing of edges using OpenMP
    #pragma omp parallel
    {
        // Each thread maintains its own best distance and cycle
        double local_best_distance = std::numeric_limits<double>::infinity();
        std::vector<int> local_best_cycle;

        #pragma omp for schedule(dynamic)
        for (int i = 0; i < edges.size(); ++i) {
            const auto& edge = edges[i];
            CycleResult result = process_edge(edge, n, distance_matrix);
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

    // End timing
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> computation_time = end_time - start_time;

    // Output the results
    if (!global_best_cycle.empty()) {
        std::cout << "\nBest Cycle (Modified Algorithm with Lookahead and Skipping):" << std::endl;
        for (int node : global_best_cycle) {
            std::cout << (node + 1) << " "; // Adjusting node ID back to original
        }
        std::cout << std::endl;
        std::cout << "Total Distance: " << global_best_distance << std::endl;
    } else {
        std::cout << "\nNo complete cycle was found." << std::endl;
    }

    std::cout << "\nTotal computation time: " << computation_time.count() << " seconds" << std::endl;

    return 0;
}