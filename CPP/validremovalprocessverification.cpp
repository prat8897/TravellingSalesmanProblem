#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <algorithm>
#include <limits>
#include <bitset>
#include <map>
#include <string>
#include <set>
#include <cmath>
#include <utility>
#include <omp.h>

// Generate random points in [0,1] x [0,1]
std::vector<std::pair<double,double>> generate_random_instance(int n) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);
    
    std::vector<std::pair<double,double>> points(n);
    for (int i = 0; i < n; i++) {
        points[i].first = dis(gen);
        points[i].second = dis(gen);
    }
    return points;
}

// Compute distance matrix
std::vector<std::vector<double>> compute_distances(const std::vector<std::pair<double,double>>& points) {
    int n = points.size();
    std::vector<std::vector<double>> dist(n, std::vector<double>(n));
    for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < n; j++) {
            double dx = points[i].first - points[j].first;
            double dy = points[i].second - points[j].second;
            dist[i][j] = dist[j][i] = std::sqrt(dx*dx + dy*dy);
        }
    }
    return dist;
}

// Calculate tour length
double tour_length(const std::vector<int>& tour, const std::vector<std::vector<double>>& dist) {
    double length = 0;
    for (int i = 0; i < tour.size(); i++) {
        length += dist[tour[i]][tour[(i + 1) % tour.size()]];
    }
    return length;
}

// Helper function: Incremental Insertion without Lookahead
std::vector<int> build_cycle_incremental(const std::vector<int>& current_cycle, std::set<int> remaining_points, 
                                       const std::vector<std::vector<double>>& distance_matrix) {
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
                double delta = distance_matrix[p][r] + distance_matrix[r][q] - distance_matrix[p][q];

                // Update if this is the best (smallest) delta found so far
                if (delta < best_delta_distance) {
                    best_delta_distance = delta;
                    best_r = r;
                    best_insertion_position = i + 1;
                }
            }
        }

        if (best_r != -1) {
            cycle.insert(cycle.begin() + best_insertion_position, best_r);
            remaining_points.erase(best_r);
        } else {
            break;
        }
    }

    return cycle;
}

// Modified Insertion with Lookahead
std::vector<int> build_cycle_least_distance_updated(const std::vector<int>& start_edge, std::set<int> remaining_points,
                                                  const std::vector<std::vector<double>>& distance_matrix,
                                                  std::vector<int>& insertion_sequence) {
    std::vector<int> cycle = start_edge;

    while (!remaining_points.empty()) {
        int best_r = -1;
        int best_insertion_position = -1;
        double best_total_distance = std::numeric_limits<double>::infinity();

        // Try each remaining point r
        for (int r : remaining_points) {
            // Try each possible insertion position for r
            for (int i = 0; i < cycle.size(); ++i) {
                int p = cycle[i];
                int q = cycle[(i + 1) % cycle.size()];

                // Simulate inserting r between p and q
                std::vector<int> temp_cycle = cycle;
                temp_cycle.insert(temp_cycle.begin() + i + 1, r);
                std::set<int> temp_remaining = remaining_points;
                temp_remaining.erase(r);

                // Continue inserting the rest using incremental
                std::vector<int> simulated_cycle = build_cycle_incremental(temp_cycle, temp_remaining, distance_matrix);

                // Calculate total distance of the simulated cycle
                double simulated_distance = tour_length(simulated_cycle, distance_matrix);

                // Update if this is better (or equal with earlier position)
                if (simulated_distance < best_total_distance || 
                   (simulated_distance == best_total_distance && i + 1 < best_insertion_position)) {
                    best_total_distance = simulated_distance;
                    best_r = r;
                    best_insertion_position = i + 1;
                }
            }
        }

        if (best_r != -1) {
            cycle.insert(cycle.begin() + best_insertion_position, best_r);
            remaining_points.erase(best_r);
            insertion_sequence.push_back(best_r);
        } else {
            break;
        }
    }

    return cycle;
}

// Try building from each edge and take best result
std::vector<int> find_best_tour(const std::vector<std::vector<double>>& dist) {
    int n = dist.size();
    double best_cost = std::numeric_limits<double>::infinity();
    std::vector<int> best_tour;
    
    // Try each possible starting edge
    for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < n; j++) {
            std::vector<int> edge = {i, j};
            std::set<int> remaining;
            for (int k = 0; k < n; k++) {
                if (k != i && k != j) remaining.insert(k);
            }
            
            auto tour = build_cycle_least_distance_updated(edge, remaining, dist);
            double cost = tour_length(tour, dist);
            
            if (cost < best_cost) {
                best_cost = cost;
                best_tour = tour;
            }
        }
    }
    
    return best_tour;
}

// Try removing points and reconstructing
void find_removal_sequence(const std::vector<std::vector<double>>& dist) {
    // First get best tour using lookahead building
    std::vector<int> tour = find_best_tour(dist);
    std::vector<std::pair<int,int>> removal_sequence;  // (point, position) pairs
    std::vector<int> current_tour = tour;
    
    std::cout << "Starting with tour: ";
    for (int pt : current_tour) std::cout << pt << " ";
    std::cout << std::endl;
    
    // Remove points until only 2 remain
    while (current_tour.size() > 2) {
        std::vector<std::pair<double,int>> removal_costs;  // (cost, position)
        
        // Try removing each point
        for (int pos = 0; pos < current_tour.size(); pos++) {
            // Remove point temporarily
            auto next_tour = current_tour;
            int pt = next_tour[pos];
            next_tour.erase(next_tour.begin() + pos);
            
            // Calculate reinsertion cost
            std::set<int> points_to_reinsert = {pt};
            auto simulated_tour = build_cycle_incremental(next_tour, points_to_reinsert, dist);
            double cost = tour_length(simulated_tour, dist);
            
            removal_costs.emplace_back(cost, pos);
        }
        
        // Sort by cost, breaking ties by position
        std::sort(removal_costs.begin(), removal_costs.end());
        
        // Take best removal
        auto [best_cost, best_pos] = removal_costs[0];
        int removed_point = current_tour[best_pos];
        current_tour.erase(current_tour.begin() + best_pos);
        removal_sequence.emplace_back(removed_point, best_pos);
        
        std::cout << "Step " << removal_sequence.size() << ": ";
        std::cout << "Removed " << removed_point << " from position " << best_pos;
        std::cout << " (cost: " << best_cost << ")" << std::endl;
        
        std::cout << "Current tour: ";
        for (int pt : current_tour) std::cout << pt << " ";
        std::cout << std::endl << std::endl;
    }
    
    // Now try reconstructing
    std::cout << "\nReconstructing from edge [" << current_tour[0] << "," << current_tour[1] << "]" << std::endl;
    
    // Get set of points not in edge
    std::set<int> points_to_insert;
    for (int i = 0; i < dist.size(); i++) {
        if (i != current_tour[0] && i != current_tour[1]) {
            points_to_insert.insert(i);
        }
    }
    
    // Get actual insertion sequence when building from this edge
    std::vector<int> observed_insertions;
    auto track_insertions = [&observed_insertions](int inserted_pt) {
        observed_insertions.push_back(inserted_pt);
    };
    
    // Build tour and track insertions
    std::vector<int> expected_insertions;
    for (auto it = removal_sequence.rbegin(); it != removal_sequence.rend(); ++it) {
        expected_insertions.push_back(it->first);
    }
    
    std::cout << "Expected insertion sequence: ";
    for (int pt : expected_insertions) std::cout << pt << " ";
    std::cout << std::endl;
    
    std::cout << "Observed insertion sequence: ";
    for (int pt : observed_insertions) std::cout << pt << " ";
    std::cout << std::endl;

    
    std::cout << "Original tour:      ";
    for (int pt : tour) std::cout << pt << " ";
    std::cout << std::endl;
    
    std::cout << "Reconstructed tour: ";
    for (int pt : reconstructed) std::cout << pt << " ";
    std::cout << std::endl;
}

int main() {
    int n = 8;  // Size of instance
    auto points = generate_random_instance(n);
    auto dist = compute_distances(points);
    
    find_removal_sequence(dist);
    
    return 0;
}