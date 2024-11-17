# TravellingSalesmanProblem


### An Exact Polynomial-Time Algorithm for the Euclidean Traveling Salesman Problem

This repository contains the implementation and research paper for an exact algorithm solving the Euclidean Traveling Salesman Problem (TSP) with a time complexity of  O(n^4) . The algorithm utilizes combinatorial optimization techniques and efficient insertion strategies to systematically explore all possible tours while ensuring optimality.

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