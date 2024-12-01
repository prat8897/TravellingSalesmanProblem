#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <set>
#include <cmath>
#include <limits>
#include <algorithm>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/reduce.h>
#include <thrust/execution_policy.h>
#include <chrono>

// Error checking macro
#define cudaCheckError() { \
    cudaError_t err = cudaGetLastError(); \
    if (err != cudaSuccess) { \
        printf("CUDA error: %s, line %d\n", cudaGetErrorString(err), __LINE__); \
        exit(1); \
    } \
}

// CUDA kernel for computing distance matrix
__global__ void computeDistanceMatrixKernel(const double* points_x, const double* points_y, 
                                          double* distance_matrix, int n) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (row < n && col < n) {
        if (row != col) {
            double dx = points_x[row] - points_x[col];
            double dy = points_y[row] - points_y[col];
            distance_matrix[row * n + col] = sqrt(dx * dx + dy * dy);
        } else {
            distance_matrix[row * n + col] = 0.0;
        }
    }
}

struct Point {
    int id;
    double x;
    double y;
};

// Global kernel constants
__constant__ int d_n;  // Problem size
__constant__ double d_inf = INFINITY;

// Device helper functions
__device__ double calculate_cycle_distance(const int* cycle, const double* distance_matrix) {
    double total_distance = 0.0;
    for (int i = 0; i < d_n; i++) {
        total_distance += distance_matrix[cycle[i] * d_n + cycle[(i + 1) % d_n]];
    }
    return total_distance;
}

__device__ void insert_point(int* cycle, int* cycle_size, int point, int position) {
    for (int i = *cycle_size; i > position; i--) {
        cycle[i] = cycle[i - 1];
    }
    cycle[position] = point;
    (*cycle_size)++;
}

__device__ void build_cycle_incremental_gpu(int* cycle, int cycle_size, 
                                          bool* remaining_points,
                                          const double* distance_matrix,
                                          int* final_size) {
    while (cycle_size < d_n) {
        int best_r = -1;
        int best_insertion_position = -1;
        double best_delta_distance = d_inf;

        for (int r = 0; r < d_n; r++) {
            if (!remaining_points[r]) continue;

            for (int i = 0; i < cycle_size; i++) {
                int p = cycle[i];
                int q = cycle[(i + 1) % cycle_size];

                double delta = distance_matrix[p * d_n + r] + 
                             distance_matrix[r * d_n + q] - 
                             distance_matrix[p * d_n + q];

                if (delta < best_delta_distance) {
                    best_delta_distance = delta;
                    best_r = r;
                    best_insertion_position = i + 1;
                }
            }
        }

        if (best_r == -1) break;

        insert_point(cycle, &cycle_size, best_r, best_insertion_position);
        remaining_points[best_r] = false;
    }

    *final_size = cycle_size;
}

__global__ void processEdgesKernel(const double* distance_matrix,
                                 const int* edges,
                                 int num_edges,
                                 double* best_distances,
                                 int* best_cycles) {
    int edge_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (edge_idx >= num_edges) return;

    int temp_cycle[1024];
    bool remaining[1024];
    int sim_cycle[1024];
    bool sim_remaining[1024];
    
    int first = edges[edge_idx * 2];
    int second = edges[edge_idx * 2 + 1];
    temp_cycle[0] = first;
    temp_cycle[1] = second;
    int cycle_size = 2;

    for (int i = 0; i < d_n; i++) {
        remaining[i] = true;
    }
    remaining[first] = false;
    remaining[second] = false;

    while (cycle_size < d_n) {
        int best_r = -1;
        int best_insertion_position = -1;
        double best_total_distance = d_inf;

        for (int r = 0; r < d_n; r++) {
            if (!remaining[r]) continue;

            for (int i = 0; i <= cycle_size; i++) {
                for (int j = 0; j < cycle_size; j++) {
                    sim_cycle[j] = temp_cycle[j];
                }
                
                // Insert r at position i
                int sim_size = cycle_size;
                insert_point(sim_cycle, &sim_size, r, i);
                
                for (int j = 0; j < d_n; j++) {
                    sim_remaining[j] = remaining[j];
                }
                sim_remaining[r] = false;

                build_cycle_incremental_gpu(sim_cycle, sim_size, sim_remaining, 
                                         distance_matrix, &sim_size);

                if (sim_size == d_n) {
                    double total_distance = calculate_cycle_distance(sim_cycle, distance_matrix);
                    if (total_distance < best_total_distance) {
                        best_total_distance = total_distance;
                        best_r = r;
                        best_insertion_position = i;
                    }
                }
            }
        }

        if (best_r == -1) break;

        insert_point(temp_cycle, &cycle_size, best_r, best_insertion_position);
        remaining[best_r] = false;
    }

    if (cycle_size == d_n) {
        double final_distance = calculate_cycle_distance(temp_cycle, distance_matrix);
        best_distances[edge_idx] = final_distance;
        for (int i = 0; i < d_n; i++) {
            best_cycles[edge_idx * d_n + i] = temp_cycle[i];
        }
    } else {
        best_distances[edge_idx] = d_inf;
    }
}

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
            if (!(iss >> id >> x >> y)) continue;
            points.push_back({id - 1, x, y});
        }
    }
    return points;
}

thrust::device_vector<double> compute_distance_matrix_cuda(const std::vector<Point>& points) {
    int n = points.size();
    
    thrust::device_vector<double> d_points_x(n);
    thrust::device_vector<double> d_points_y(n);
    for (int i = 0; i < n; i++) {
        d_points_x[i] = points[i].x;
        d_points_y[i] = points[i].y;
    }
    
    thrust::device_vector<double> d_distance_matrix(n * n);
    
    dim3 block_size(16, 16);
    dim3 grid_size((n + block_size.x - 1) / block_size.x,
                   (n + block_size.y - 1) / block_size.y);
    
    computeDistanceMatrixKernel<<<grid_size, block_size>>>(
        thrust::raw_pointer_cast(d_points_x.data()),
        thrust::raw_pointer_cast(d_points_y.data()),
        thrust::raw_pointer_cast(d_distance_matrix.data()),
        n
    );
    cudaCheckError();
    
    return d_distance_matrix;
}

void process_edges_cuda(const thrust::device_vector<double>& d_distance_matrix,
                       const std::vector<std::pair<int, int>>& edges,
                       int n,
                       double& best_distance,
                       std::vector<int>& best_cycle) {
    int num_edges = edges.size();
    
    cudaMemcpyToSymbol(d_n, &n, sizeof(int));
    
    thrust::device_vector<int> d_edges(num_edges * 2);
    for (int i = 0; i < num_edges; i++) {
        d_edges[i * 2] = edges[i].first;
        d_edges[i * 2 + 1] = edges[i].second;
    }

    thrust::device_vector<double> d_best_distances(num_edges);
    thrust::device_vector<int> d_best_cycles(num_edges * n);

    int threads_per_block = 256;
    int blocks = (num_edges + threads_per_block - 1) / threads_per_block;

    processEdgesKernel<<<blocks, threads_per_block>>>(
        thrust::raw_pointer_cast(d_distance_matrix.data()),
        thrust::raw_pointer_cast(d_edges.data()),
        num_edges,
        thrust::raw_pointer_cast(d_best_distances.data()),
        thrust::raw_pointer_cast(d_best_cycles.data())
    );
    cudaCheckError();

    thrust::host_vector<double> h_best_distances = d_best_distances;
    thrust::host_vector<int> h_best_cycles = d_best_cycles;

    best_distance = std::numeric_limits<double>::infinity();
    for (int i = 0; i < num_edges; i++) {
        if (h_best_distances[i] < best_distance) {
            best_distance = h_best_distances[i];
            best_cycle.resize(n);
            for (int j = 0; j < n; j++) {
                best_cycle[j] = h_best_cycles[i * n + j];
            }
        }
    }
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " filename.tsp" << std::endl;
        return 1;
    }

    try {
        std::vector<Point> points = read_tsp_file(argv[1]);
        int n = points.size();
        if (n == 0) {
            std::cerr << "No points read from file." << std::endl;
            return 1;
        }

        if (n > 1024) {
            std::cerr << "Problem size too large. Maximum supported size is 1024 points." << std::endl;
            return 1;
        }

        cudaFree(0);
        
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        std::cout << "Using GPU: " << prop.name << std::endl;
        std::cout << "Max threads per block: " << prop.maxThreadsPerBlock << std::endl;

        auto start_time = std::chrono::high_resolution_clock::now();

        thrust::device_vector<double> d_distance_matrix = compute_distance_matrix_cuda(points);

        std::vector<std::pair<int, int>> edges;
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                edges.emplace_back(i, j);
            }
        }
        std::cout << "Total edges to process: " << edges.size() << std::endl;

        double best_distance = std::numeric_limits<double>::infinity();
        std::vector<int> best_cycle;
        process_edges_cuda(d_distance_matrix, edges, n, best_distance, best_cycle);

        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> computation_time = end_time - start_time;

        if (!best_cycle.empty()) {
            std::cout << "\nBest Cycle:" << std::endl;
            for (int node : best_cycle) {
                std::cout << (node + 1) << " ";
            }
            std::cout << "\nTotal Distance: " << best_distance << std::endl;
        } else {
            std::cout << "\nNo complete cycle was found." << std::endl;
        }

        std::cout << "Computation time: " << computation_time.count() << " seconds" << std::endl;

        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}