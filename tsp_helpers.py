# tsp_helpers.py

import numpy as np

# Function to calculate Euclidean distance
def euclidean_distance(p1, p2):
    return np.linalg.norm(p1 - p2)

# Function to calculate total distance of a cycle using precomputed distances
def total_cycle_distance(cycle, distance_matrix):
    distance = 0.0
    for i in range(len(cycle)):
        distance += distance_matrix[cycle[i]][cycle[(i + 1) % len(cycle)]]
    return distance

# Function to normalize a cycle for consistent comparison
def normalize_cycle(cycle):
    """
    Normalize the cycle by choosing the lexicographically smallest representation
    considering all rotations and reflections.
    """
    n = len(cycle)
    cyclic_shifts = [tuple(cycle[i:] + cycle[:i]) for i in range(n)]
    reversed_cycle = list(reversed(cycle))
    cyclic_shifts += [tuple(reversed_cycle[i:] + reversed_cycle[:i]) for i in range(n)]
    return min(cyclic_shifts)

# *** Modified Function with Lookahead ***
def build_cycle_least_distance_updated(start_edge, remaining_points, distance_matrix):
    """
    Build a cycle starting from a given edge, inserting remaining points to minimize total distance.
    No pruning based on partial cycles is applied.
    """
    cycle = list(start_edge)
    steps = [cycle.copy()]  # Record initial edge

    while remaining_points:
        best_r = None
        best_insertion_position = None
        best_total_distance = float('inf')

        # Iterate over each remaining point r
        for r in remaining_points:
            # Iterate over each possible insertion position for r
            for i in range(len(cycle)):
                p = cycle[i]
                q = cycle[(i + 1) % len(cycle)]

                # Simulate inserting r between p and q
                temp_cycle = cycle.copy()
                temp_cycle.insert(i + 1, r)
                temp_remaining = remaining_points.copy()
                temp_remaining.remove(r)

                # Continue inserting the rest of the points using incremental heuristic
                simulated_cycle, _ = build_cycle_incremental(temp_cycle, temp_remaining.copy(), distance_matrix)

                # Calculate total distance of the simulated cycle
                simulated_distance = total_cycle_distance(simulated_cycle, distance_matrix)

                # Update if this is the best (smallest) total distance found so far
                if simulated_distance < best_total_distance:
                    best_total_distance = simulated_distance
                    best_r = r
                    best_insertion_position = i + 1

        if best_r is not None:
            # Insert the best r into the cycle at the best position
            cycle.insert(best_insertion_position, best_r)
            remaining_points.remove(best_r)
            steps.append(cycle.copy())  # Record the new state of the cycle
        else:
            # If no suitable point found, break to avoid infinite loop
            break

    return cycle, steps

# Helper Function: Incremental Insertion without Lookahead
def build_cycle_incremental(current_cycle, remaining_points, distance_matrix):
    """
    Incrementally build a cycle by inserting remaining points to minimize incremental distance.
    No pruning based on partial cycles is applied.
    """
    cycle = list(current_cycle)
    steps = [cycle.copy()]

    while remaining_points:
        best_r = None
        best_insertion_position = None
        best_delta_distance = float('inf')

        # Iterate over each remaining point r
        for r in remaining_points:
            # Find the best insertion position for r based on delta distance
            for i in range(len(cycle)):
                p = cycle[i]
                q = cycle[(i + 1) % len(cycle)]

                # Calculate the incremental distance
                delta = (distance_matrix[p][r] + distance_matrix[r][q] - distance_matrix[p][q])

                # Update if this is the best (smallest) delta found so far
                if delta < best_delta_distance:
                    best_delta_distance = delta
                    best_r = r
                    best_insertion_position = i + 1

        if best_r is not None:
            # Insert the best r into the cycle at the best position
            cycle.insert(best_insertion_position, best_r)
            remaining_points.remove(best_r)
            steps.append(cycle.copy())  # Record the new state of the cycle
        else:
            # If no suitable point found, break to avoid infinite loop
            break

    return cycle, steps

# Function to process a single edge
def process_edge(args):
    edge, n, distance_matrix = args
    remaining = set(range(n)) - set(edge)
    cycle, steps = build_cycle_least_distance_updated(edge, remaining.copy(), distance_matrix)

    # Ensure the cycle includes all points
    if len(cycle) == n:
        distance = total_cycle_distance(cycle, distance_matrix)
        return (distance, cycle, steps)
    else:
        return None  # No complete cycle found