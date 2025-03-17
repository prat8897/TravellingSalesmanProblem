# A Polynomial Time Algorithm for Euclidean TSP

This repository contains the paper "A Polynomial Time Algorithm for Euclidean TSP" which presents a deterministic polynomial-time algorithm for solving the Euclidean Traveling Salesman Problem optimally.

## Overview

The algorithm solves the Euclidean TSP in O(n⁷) time using a novel dynamic lookahead insertion approach. The key insight is that optimal tours can be reconstructed through careful point insertion guided by simulated lookahead costs.

### Key Components

1. **Dynamic Lookahead Insertion**: The main algorithm that tries all possible starting edges
2. **Build Cycle with Lookahead**: Constructs tours using lookahead simulation
3. **Incremental Insertion**: A greedy procedure used within lookahead simulation

## Theoretical Foundation

The algorithm's correctness relies on several key theoretical results:

1. **Valid Removal Process Lemma**: For any optimal tour T*, there exists a sequence of point removals where each removed point had minimal simulated insertion cost.

2. **Relative Ordering Preservation**: When starting from an edge obtained through the valid removal process, minimal lookahead cost insertions preserve the relative ordering of points from the optimal tour.

3. **Euclidean Uniqueness**: For points in general position (no three collinear points, no equal distances), the optimal tour is unique up to reversal.

The proofs leverage fundamental properties of Euclidean geometry, particularly the triangle inequality, to show that the algorithm's locally optimal choices align with global optimality.

## Empirical Validation

The algorithm has been extensively tested:
- 135,000 test cases
- Each test produced the optimal solution
- Validated across various point configurations and sizes

## Complexity Analysis

The O(n⁷) runtime complexity breaks down as follows:
- O(n²) starting point pairs
- For each pair:
  - O(n) insertions to complete tour
  - O(n) candidate points per insertion
  - O(n) possible positions per candidate
  - O(n²) for lookahead simulation

## The Repository

This repository contains two folders: Python and CPP. Each folder has an implementation in their respective languages.

- Python
	- `TSP.ipynb`: A Python implementation of the algorithm in a notebook
	- `benchmark.py`: A benchmarking script that creates random instances of the TSP and compares the exact solution to this algorithm.

- CPP
	- `tsp_cycle.cpp`: A script that is designed to read .tsp files and solve them using this algorithm.
	- `tsp.cpp`: Similar to `benchmark.py`, solves random TSP instances optimally and then compares the solution for n number of trials.
