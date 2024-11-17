import numpy as np
import itertools
import pandas as pd
import matplotlib.pyplot as plt
import pulp
import time
from tqdm import tqdm  # For progress bars
from multiprocessing import Pool, cpu_count
import os

# Parameters
n = 25      # Number of points per instance
m = 8*2      # Number of instances (adjust as needed)
base_seed = 14201  # Base seed for reproducibility

# Control variable for plotting
plot_comparisons = True  # Set to True to enable plotting

# Directories for saving plots
plots_directory = 'tsp_comparison_plots'
if plot_comparisons:
    os.makedirs(plots_directory, exist_ok=True)

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
def build_cycle_least_distance_updated(start_triangle, remaining_points, distance_matrix, seen_partial_cycles):
    """
    Build a cycle starting from a given triangle, inserting remaining points to minimize total distance.
    No pruning based on partial cycles is applied.
    """
    cycle = list(start_triangle)
    steps = [cycle.copy()]  # Record initial triangle

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
                simulated_cycle, _ = build_cycle_incremental(temp_cycle, temp_remaining.copy(), distance_matrix, seen_partial_cycles)

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
def build_cycle_incremental(current_cycle, remaining_points, distance_matrix, seen_partial_cycles):
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

# Function to solve TSP using PuLP with MTZ formulation
def solve_tsp_pulp(distance_matrix, fixed_start=0):
    n = len(distance_matrix)
    # Create the problem variable to contain the problem data
    prob = pulp.LpProblem("TSP", pulp.LpMinimize)

    # Create a list of all possible edges excluding self-loops
    edges = [(i, j) for i in range(n) for j in range(n) if i != j]

    # Create a binary variable for each edge
    edge_vars = pulp.LpVariable.dicts("Edge", edges, cat='Binary')

    # Objective function: minimize the total distance
    prob += pulp.lpSum([distance_matrix[i][j] * edge_vars[i, j] for (i, j) in edges]), "Total Distance"

    # Constraint: Each city must be entered exactly once
    for j in range(n):
        prob += pulp.lpSum([edge_vars[i, j] for i in range(n) if i != j]) == 1, f"Enter_{j}"

    # Constraint: Each city must be left exactly once
    for i in range(n):
        prob += pulp.lpSum([edge_vars[i, j] for j in range(n) if j != i]) == 1, f"Leave_{i}"

    # Subtour elimination using MTZ formulation
    u = pulp.LpVariable.dicts("u", range(n), lowBound=0, upBound=n-1, cat='Integer')

    for i in range(1, n):
        for j in range(1, n):
            if i != j:
                prob += u[i] - u[j] + (n-1)*edge_vars[i, j] <= n-2, f"MTZ_{i}_{j}"

    # Solve the problem
    prob.solve(pulp.PULP_CBC_CMD(msg=0))  # Suppress solver output

    # Check if an optimal solution was found
    if pulp.LpStatus[prob.status] != 'Optimal':
        return None, None  # No solution found

    # Extract the route from the solution
    pulp_route = []
    current = fixed_start
    pulp_route.append(current)
    while True:
        next_city = None
        for j in range(n):
            if j != current and pulp.value(edge_vars[current, j]) == 1:
                next_city = j
                break
        if next_city is None or next_city == fixed_start:
            break
        pulp_route.append(next_city)
        current = next_city

    # Ensure the cycle is complete
    if len(pulp_route) != n:
        return None, None  # Incomplete cycle

    # Calculate total distance
    pulp_distance = total_cycle_distance(pulp_route, distance_matrix)

    return pulp_route, pulp_distance

# Function to plot and save comparison of custom and PuLP routes
def plot_routes(seed, points, custom_cycle, pulp_cycle, save_dir):
    """
    Plots the custom and PuLP routes on the same graph and saves the plot.

    Parameters:
    - seed: The seed used for the instance (used for naming the plot file).
    - points: Numpy array of shape (n, 2) representing the coordinates of the nodes.
    - custom_cycle: List representing the custom algorithm's cycle.
    - pulp_cycle: List representing PuLP's cycle.
    - save_dir: Directory where the plot will be saved.
    """
    plt.figure(figsize=(8, 8))
    # Scatter plot of all points
    plt.scatter(points[:, 0], points[:, 1], color='blue', zorder=5)

    # Annotate points
    for idx, (x, y) in enumerate(points):
        plt.text(x + 0.01, y + 0.01, str(idx), fontsize=9, ha='right', va='bottom')

    # Function to plot a cycle
    def plot_cycle(cycle, color, label, linestyle='-'):
        cycle_points = points[list(cycle) + [cycle[0]]]  # Complete the cycle by returning to the start
        plt.plot(cycle_points[:, 0], cycle_points[:, 1], color=color, label=label, linestyle=linestyle, linewidth=1)

    # Plot Custom Cycle
    if custom_cycle is not None and len(custom_cycle) == n:
        plot_cycle(custom_cycle, 'orange', 'Custom Algorithm')

    # Plot PuLP Cycle
    if pulp_cycle is not None and len(pulp_cycle) == n:
        plot_cycle(pulp_cycle, 'green', 'PuLP', linestyle='--')

    plt.title(f"TSP Instance Seed: {seed}")
    plt.xlabel('X-coordinate')
    plt.ylabel('Y-coordinate')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Save the plot
    plot_filename = os.path.join(save_dir, f"seed_{seed}.png")
    plt.savefig(plot_filename)
    plt.close()

# Function to process a single TSP instance
def process_instance(args):
    instance_id, seed = args
    np.random.seed(seed)

    # Generate random 2D points
    points = np.random.rand(n, 2)

    # Precompute distance matrix
    distance_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                distance_matrix[i][j] = euclidean_distance(points[i], points[j])
            else:
                distance_matrix[i][j] = 0.0

    # ----- Solve with Custom Algorithm -----
    custom_start_time = time.time()

    # Generate all possible triangles (combinations of 3 distinct points)
    triangles = list(itertools.combinations(range(n), 3))
    total_triangles = len(triangles)

    best_cycle = None
    best_distance = float('inf')

    seen_partial_cycles = set()

    # Uncomment the next line if you want to see progress per instance
    # print(f"Instance {instance_id + 1}: Processing {total_triangles} triangles...")

    for idx, triangle in enumerate(triangles):
        normalized_triangle = normalize_cycle(triangle)
        if normalized_triangle in seen_partial_cycles:
            continue  # Skip processing this triangle as its normalized form has been processed

        remaining = set(range(n)) - set(triangle)
        cycle, _ = build_cycle_least_distance_updated(triangle, remaining.copy(), distance_matrix, seen_partial_cycles)

        # Normalize the current cycle
        normalized = normalize_cycle(cycle)
        if normalized in seen_partial_cycles:
            continue  # Skip processing this cycle further

        # Add the normalized cycle to seen_partial_cycles
        seen_partial_cycles.add(normalized)

        # Ensure the cycle includes all points
        if cycle and len(cycle) == n:
            distance = total_cycle_distance(cycle, distance_matrix)
            if distance < best_distance:
                best_distance = distance
                best_cycle = cycle
                # Uncomment the next line if you want to see improvements during processing
                # print(f"Instance {instance_id + 1}: Found better cycle with distance {best_distance:.6f}")

    custom_end_time = time.time()
    custom_elapsed_time = custom_end_time - custom_start_time

    if best_cycle is not None:
        custom_distance = best_distance
        # Uncomment the next line if you want detailed instance results
        # print(f"Instance {instance_id + 1}: Custom algorithm found a cycle with distance {custom_distance:.6f} in {custom_elapsed_time:.4f} seconds.")
    else:
        custom_distance = None
        # Uncomment the next line if you want detailed instance results
        # print(f"Instance {instance_id + 1}: Custom algorithm failed to find a complete cycle in {custom_elapsed_time:.4f} seconds.")

    # ----- Solve with PuLP -----
    pulp_start_time = time.time()

    pulp_route, pulp_distance = solve_tsp_pulp(distance_matrix, fixed_start=0)

    pulp_end_time = time.time()
    pulp_elapsed_time = pulp_end_time - pulp_start_time

    if pulp_route is not None:
        # Uncomment the next line if you want detailed instance results
        # print(f"Instance {instance_id + 1}: PuLP found a cycle with distance {pulp_distance:.6f} in {pulp_elapsed_time:.4f} seconds.")
        pass
    else:
        # Uncomment the next line if you want detailed instance results
        # print(f"Instance {instance_id + 1}: PuLP failed to find a complete cycle in {pulp_elapsed_time:.4f} seconds.")
        pass

    # ----- Compare Cycles -----
    if best_cycle is not None and pulp_route is not None:
        normalized_custom = normalize_cycle(best_cycle)
        normalized_pulp = normalize_cycle(pulp_route)
        same_cycle = 'Yes' if normalized_custom == normalized_pulp else 'No'
    else:
        same_cycle = 'No'

    # ----- Calculate Distance Difference -----
    if custom_distance is not None and pulp_distance is not None:
        distance_difference = custom_distance - pulp_distance
    else:
        distance_difference = None

    # ----- Calculate Same Distance Using np.allclose -----
    if custom_distance is not None and pulp_distance is not None:
        same_distance = 'Yes' if np.allclose(pulp_distance, custom_distance, atol=1e-8) else 'No'
    else:
        same_distance = 'No'

    # ----- Prepare the Result -----
    result = {
        'Instance_ID': instance_id + 1,
        'Seed': seed,
        'Custom_Distance': custom_distance,
        'PuLP_Distance': pulp_distance,
        'Distance_Difference': distance_difference,
        'Same_Distance': same_distance,
        'Same_Cycle': same_cycle,
        'Custom_Time_sec': round(custom_elapsed_time, 4),
        'PuLP_Time_sec': round(pulp_elapsed_time, 4),
        'PuLP_Cycle': ','.join(map(str, pulp_route)) if pulp_route is not None else '',
        'Custom_Cycle': ','.join(map(str, best_cycle)) if best_cycle is not None else '',
        'Points': ';'.join([f"{x},{y}" for x, y in points])
    }

    # ----- Generate and Save Plot if Enabled -----
    if plot_comparisons and best_cycle is not None and pulp_route is not None:
        plot_routes(
            seed=seed,
            points=points,
            custom_cycle=best_cycle,
            pulp_cycle=pulp_route,
            save_dir=plots_directory
        )

    return result

# Main execution block
def main():
    # Initialize a list to store all records
    data_records = []

    # Initialize a list to store all seeds for reproducibility
    seeds = range(base_seed, base_seed + m)

    # Prepare arguments as list of tuples (instance_id, seed)
    args_list = list(zip(range(m), seeds))

    # Determine the number of worker processes (use all available CPU cores)
    num_workers = cpu_count()
    print(f"Starting processing with {num_workers} parallel workers...")

    # Create a multiprocessing Pool
    with Pool(processes=num_workers) as pool:
        # Use imap_unordered for better performance and progress bar integration
        for result in tqdm(pool.imap_unordered(process_instance, args_list), total=m, desc="Processing Instances"):
            data_records.append(result)

    # Convert the list of dictionaries to a DataFrame
    results = pd.DataFrame(data_records)

    # Save results to CSV
    results.to_csv('tsp_comparison_results.csv', index=False)
    print("\nComparison results saved to 'tsp_comparison_results.csv'.")

    # ----- Generate Summary Statistics -----
    # Drop instances where either solver failed
    valid_results = results.dropna(subset=['Custom_Distance', 'PuLP_Distance'])

    if not valid_results.empty:
        avg_custom_distance = valid_results['Custom_Distance'].mean()
        avg_pulp_distance = valid_results['PuLP_Distance'].mean()
        avg_distance_difference = valid_results['Distance_Difference'].mean()
        num_same_cycles = valid_results['Same_Cycle'].value_counts().get('Yes', 0)
        num_same_distance = valid_results['Same_Distance'].value_counts().get('Yes', 0)

        print("\nSummary Statistics:")
        print(f"Average Custom Algorithm Distance: {avg_custom_distance:.4f}")
        print(f"Average PuLP Distance: {avg_pulp_distance:.4f}")
        print(f"Average Distance Difference (Custom - PuLP): {avg_distance_difference:.4f}")
        print(f"Number of Same Cycles: {num_same_cycles}/{len(valid_results)}")
        print(f"Number of Same Distances: {num_same_distance}/{len(valid_results)}")
    else:
        print("\nNo valid results to summarize.")

    # ----- Inform About Comparison Plots -----
    if plot_comparisons:
        print(f"Comparison plots have been saved in the '{plots_directory}' directory, named by their seed values.")

if __name__ == '__main__':
    main()