# TravellingSalesmanProblem


### An Exact Polynomial-Time Algorithm for the Euclidean Traveling Salesman Problem

This repository contains the implementation and research paper for an exact algorithm solving the Euclidean Traveling Salesman Problem (TSP) with a time complexity of  O(n^7) . The algorithm utilizes combinatorial optimization techniques and efficient insertion strategies to systematically explore all possible tours while ensuring optimality.

### Overview

The Traveling Salesman Problem is a classic optimization problem that seeks the shortest possible route visiting each city exactly once and returning to the origin city. Despite its simple formulation, TSP is an NP-hard problem, making it computationally challenging for large instances.

This project introduces an exact polynomial-time algorithm specifically designed for the Euclidean TSP, where cities are points in a plane and distances are calculated using the Euclidean metric. The algorithm is efficient for small to moderately sized datasets and consistently produces optimal solutions.


### Getting Started

#### Prerequisites

	•	Python 3.6 or higher
	•	Required Python packages:
	•	numpy
	•	matplotlib
	•	ortools
	•	pulp (for comparison with PuLP)
	•	tqdm (for progress bars during experiments)

Install the required packages using pip:

```
pip install numpy matplotlib ortools pulp tqdm
```


To solve .tsp instances using C++, compile `tsp_cycle.cpp` using:

```
g++ -std=c++17 -fopenmp -O2 -o tsp_solver tsp_cycle.cpp
```

To solve .tsp instances using CUDA gpu, compile `tsp_gpu.cu` using:
```
nvcc -G -g tsp_gpu.cu -o tsp_solver
```

and then run using:
```
./tsp_solver Tnm52.tsp
```
or any other .tsp file.