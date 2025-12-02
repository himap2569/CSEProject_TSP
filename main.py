#!/usr/bin/env python3

"""
TSP Solver using BF, Approx, LS Algorithm Implementations

This script implements three approaches to solve the Traveling Salesman Problem (TSP):

Main Functions:
- parse_tsp_file(path): Parses .tsp file format (id x y coordinates) and builds distance matrix
- compute_tour_cost(dist, tour): Calculates total cost of a tour including return to start

Algorithm Implementations:
- tsp_bruteforce(dist, cutoff_seconds): Simple brute-force with time cutoff
- tsp_bruteforce_optimized(dist, cutoff_seconds): Enhanced brute-force with branch-and-bound pruning,
    symmetry reduction, and nearest-neighbor ordering
- tsp_approx(ids, dist, seed): 2-approximation algorithm using MST (Prim's) followed by DFS preorder traversal
- tsp_local_search(dist, cutoff_seconds, seed): Simulated annealing with path reversal for finding neighbors,
    exponential cooling schedule (T *= 0.995 every 1000 steps)

Helper Functions:
- swap_node_pair_at_random(path): Random pairwise node swap (deprecated in favor of reversal)
- reverse_path_part(path): Reverses random subpath for neighborhood generation
- plot_cost_history(cost_history, step_history): Visualization for hyperparameter tuning

Output:
- write_solution(instance, method, cutoff, seed, cost, tour, ids): Writes .sol files with format
    <city>_<algorithm>_<params>.sol containing tour cost and vertex sequence

Command-line Interface:
- BF: ./exec -inst <file> -alg BF -time <cutoff>
- Approx: ./exec -inst <file> -alg Approx -seed <seed>
- LS: ./exec -inst <file> -alg LS -time <cutoff> -seed <seed>
"""

import sys
import os
import math
import time
import itertools
import random
import matplotlib.pyplot as plt
import argparse

# ---------- Parsing the .tsp file ----------

def parse_tsp_file(path):
    """
    Parse a .tsp-like file with lines:
        id x y
    Skips header lines that don't start with an integer.

    Returns:
        ids  : list of vertex IDs as in file (e.g., [1,2,3,...])
        dist : NxN distance matrix with rounded Euclidean distances
    """
    coords = []
    ids = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()

            # Skip non-data lines (like NAME, TYPE, etc.)
            if not parts[0].isdigit():
                continue
            if len(parts) < 3:
                continue

            vid = int(parts[0])
            x = float(parts[1])
            y = float(parts[2])

            ids.append(vid)
            coords.append((x, y))

    n = len(coords)
    if n == 0:
        raise ValueError("No coordinates found in file: " + path)

    # Build full distance matrix
    dist = [[0] * n for _ in range(n)]
    for i in range(n):
        x1, y1 = coords[i]
        for j in range(i + 1, n):
            x2, y2 = coords[j]
            d = math.hypot(x1 - x2, y1 - y2)
            w = int(round(d))   # rounded Euclidean distance
            dist[i][j] = w
            dist[j][i] = w

    return ids, dist


# ---------- Utility to compute cost of a tour ----------

def compute_tour_cost(dist, tour):
    """
    tour: sequence of vertex indices (0..n-1)
    Returns cost of the cycle:
        tour[0] -> tour[1] -> ... -> tour[-1] -> tour[0]
    """
    n = len(tour)
    cost = 0
    for i in range(n):
        u = tour[i]
        v = tour[(i + 1) % n]  # wrap around to form a cycle
        cost += dist[u][v]
    return cost

# ---------- Simple Brute-force TSP with cutoff ----------

def tsp_bruteforce(dist, cutoff_seconds):
    """
    Simple brute-force TSP:
      - Fix start node at 0
      - Enumerate all permutations of remaining nodes
      - Time cutoff (seconds): stop exploring when cutoff is reached

    Returns:
        best_cost : cost of best tour found
        best_tour : tuple of vertex indices (0..n-1), starting at 0
    """
    n = len(dist)
    if n == 0:
        return 0, ()
    if n == 1:
        return 0, (0,)

    start_time = time.time()

    best_cost = float("inf")
    best_tour = None

    # Nodes other than 0
    nodes = list(range(1, n))

    for perm in itertools.permutations(nodes):
        # Enforce cutoff
        if time.time() - start_time > cutoff_seconds:
            break

        tour = (0,) + perm  # full tour starting at 0
        cost = compute_tour_cost(dist, tour)

        if cost < best_cost:
            best_cost = cost
            best_tour = tour

    # If we never improved (e.g., cutoff was extremely tiny), fall back
    if best_tour is None:
        best_tour = tuple(range(n))
        best_cost = compute_tour_cost(dist, best_tour)

    return best_cost, best_tour
    
# ---------- Optimized Brute-force TSP with cutoff ----------

def tsp_bruteforce_optimized(dist, cutoff_seconds):
    """
    Brute-force TSP with:
      - fixed start node 0
      - branch-and-bound pruning
      - symmetry reduction (avoid reverse duplicates)
      - neighbors visited in nearest-first order
      - time cutoff (seconds)

    Returns:
        best_cost : cost of best tour found
        best_tour : tuple of vertex indices (0..n-1), starting at 0
    """
    n = len(dist)
    if n == 0:
        return 0, ()
    if n == 1:
        return 0, (0,)

    start_time = time.time()

    # Precompute nearest-neighbor ordering for each node
    neighbors = []
    for i in range(n):
        # sort all nodes by distance from i (including itself, but we'll skip visited)
        order = sorted(range(n), key=lambda j: dist[i][j])
        neighbors.append(order)

    best_cost = float("inf")
    best_tour = None

    visited = [False] * n
    tour = [0] * n

    start_node = 0
    visited[start_node] = True
    tour[0] = start_node

    def dfs(pos, current_cost):
        nonlocal best_cost, best_tour

        # Time cutoff
        if time.time() - start_time > cutoff_seconds:
            return

        # If we placed all nodes, close the tour
        if pos == n:
            # Symmetry reduction:
            # only accept tours where second city < last city
            if tour[1] > tour[-1]:
                return

            total_cost = current_cost + dist[tour[-1]][tour[0]]
            if total_cost < best_cost:
                best_cost = total_cost
                best_tour = tuple(tour)
            return

        last = tour[pos - 1]

        for nxt in neighbors[last]:
            if visited[nxt]:
                continue

            # cost so far including edge (last -> nxt)
            new_cost = current_cost + dist[last][nxt]
            if new_cost >= best_cost:
                # branch-and-bound: no need to go deeper
                continue

            tour[pos] = nxt
            visited[nxt] = True
            dfs(pos + 1, new_cost)
            visited[nxt] = False

    # Start DFS from position 1 (position 0 is fixed at start_node)
    dfs(1, 0)

    # If we never improved best_tour (e.g., cutoff super tiny), fall back
    if best_tour is None:
        best_tour = tuple(range(n))
        best_cost = compute_tour_cost(dist, best_tour)

    return best_cost, best_tour


# ---------- Approx ----------

def tsp_approx(ids, dist, seed):
    """
    2-approximation MST-based TSP algorithm should be implemented here.
    """
    n = len(dist)

    # Base cases
    if n == 0:
        return 0, ()
    if n == 1:
        return 0, (0,)

    random.seed(seed)

    # ----- Prim's algorithm for MST -----
    start = seed % n
    in_mst = [False] * n
    parent = [-1] * n
    key = [float('inf')] * n
    key[start] = 0

    for _ in range(n):
        # Pick the next node with smallest key value
        u = -1
        min_val = float('inf')
        for v in range(n):
            if not in_mst[v] and key[v] < min_val:
                min_val = key[v]
                u = v

        if u == -1:
            break

        in_mst[u] = True

        # Update keys for neighbors
        for v in range(n):
            w = dist[u][v]
            if not in_mst[v] and w < key[v]:
                key[v] = w
                parent[v] = u

    # ----- Build adjacency list for MST -----
    adj = [[] for _ in range(n)]
    for v in range(n):
        p = parent[v]
        if p != -1:
            adj[p].append(v)
            adj[v].append(p)

    # ----- DFS preorder traversal -----
    visited = [False] * n
    tour = []

    def dfs(u):
        visited[u] = True
        tour.append(u)

        # Randomized child order
        children = adj[u][:]
        random.shuffle(children)

        for v in children:
            if not visited[v]:
                dfs(v)


    dfs(start)

    # Safety check: add any missed nodes (should not happen)
    if len(tour) < n:
        for v in range(n):
            if not visited[v]:
                dfs(v)

    # Compute tour cost (cycle)
    cost = compute_tour_cost(dist, tour)

    return cost, tuple(tour)


# ---------- Local Search ----------

def tsp_local_search(dist, cutoff_seconds, seed):
    """
    Local search using Simulated Annealing:
    - Start with random path
    - [Dropped due to better results with path reversal] Generate a neighbour by swapping a pair of nodes in path
    - Generate a neighbour path by reversing part of the path
    - Choose neighbour path 
        - Deterministically, if lower cost
        - With a probability, if higher cost
    - Update temperature (and thus probability of choosing higher cost path) after each M iterations

    Returns:
        best_cost: cost of best tour found
        best_tour: tuples of vertex indices (0..n-1)
    """

    n = len(dist)

    # Base cases
    if n == 0:
        return 0, ()
    if n == 1:
        return 0, (0,)

    # Setting the seed
    random.seed(seed);

    # Measuring the algorithm start time
    start_time = time.time()

    ### Constants/ Hyperparameters chosen ###
    nSteps = 10000000 # Total steps our algorithm runs for, unless stopped basis cutoff time
    coolingFraction = 0.995 # Fraction by which temperature T is reduced after M steps
    M = 1000
    T = 500.0
    k = 1.0

    ### Main Algorithm ###
    # Random Start
    S = list(range(0,n))
    random.shuffle(S)
    currentCost = compute_tour_cost(dist, S)
    bestS = tuple(S)
    bestCost = currentCost

    # We store cost after every 1000 iters, and use it for plotting. 
    # This helped looking at the plots and tuning hyperparameters
    cost_history = []
    step_history = []

    # Function to swap a pair of nodes in a path
    def swap_node_pair_at_random(path):

        # Initializing the new path
        new_path = list(path)
        n = len(new_path)

        # base case
        if n<=3 :
            return new_path

        # indices to swap
        i1, i2 = random.sample(range(n), 2)

        # Updating the new path
        temp = new_path[i1]
        new_path[i1] = new_path[i2]
        new_path[i2] = temp

        return new_path

    # Function to reverse a part of the path
    def reverse_path_part(path):

        # Initializing the new path
        new_path = list(path)
        n = len(new_path)

        # base case
        if n<=2 :
            return new_path

        # indices between which we reverse
        i = random.randint(0, n-2)
        j = random.randint(i+1, n-1)

        # Iteratively swapping
        while i<j:
            temp = new_path[j]
            new_path[j] = new_path[i]
            new_path[i] = temp
            i += 1
            j -= 1

        return new_path

    for t in range(1, nSteps+1):
        # Time curoff
        if time.time() - start_time > cutoff_seconds:
            break

        # if t%5000 == 0:
        #     print(f"Currently on iteration {t}/{nSteps}")

        ### Option 1: Swap node pair - Less efficient ###
        # S_new = swap_node_pair_at_random(S)

        ### Option 2: Reverse a part of the graph - Better Results ###
        S_new = reverse_path_part(S)

        cost_new = compute_tour_cost(dist, S_new)

        if cost_new <= currentCost: # if the new path has a lower cost, we directly choose it
            S = S_new
            currentCost = cost_new

            if currentCost < bestCost:
                bestCost = currentCost
                bestS = tuple(S)
        else: # else, we choose the new path with a probability
            deltaE = cost_new - currentCost
            p = math.exp(-deltaE/(k*T))
            r = random.random()
            if r<p:
                S = S_new
                currentCost = cost_new

        if t%M==0: # Updating temperate after M steps each time
            T = T*coolingFraction
            cost_history.append(currentCost)
            step_history.append(t)

    # Plotting the graph - used this for hyperparameter tuning
    # plot_cost_history(cost_history, step_history)

    return bestCost, bestS

def plot_cost_history(cost_history, step_history):
    """
    Plots the best cost found till current step, against the iteration step.
    """

    if not cost_history or len(cost_history)<2:
        return

    plt.figure(figsize=(10, 6))

    # Plot the cost history
    plt.plot(step_history, cost_history, marker='o', markersize=3, linestyle='-', color='indigo')

    # Add labels and title
    plt.title('Simulated Annealing Cost Convergence Trend')
    plt.xlabel('Iteration') 
    plt.ylabel('Best Tour Cost Found So Far')

    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.show()

# ---------- Writing .sol file ----------

def write_solution(instance, method, cutoff, seed, cost, tour, ids):
    """
    Writes the solution file with correct name and format.

    File name:
        <instance> BF <cutoff>.sol            for BF
        <instance> Approx <seed>.sol          for Approx
        <instance> LS <cutoff> <seed>.sol     for LS

    Contents:
        line 1: total cost
        line 2: comma-separated vertex IDs
    """
    base = os.path.splitext(os.path.basename(instance))[0].lower()

    if method == "BF":
        filename = f"{base}_BF_{cutoff}.sol"
    elif method == "Approx":
        filename = f"{base}_Approx_{seed}.sol"
    elif method == "LS":
        filename = f"{base}_LS_{cutoff}_{seed}.sol"
    else:
        raise ValueError("Unknown method")

    # Map internal indices (0..n-1) back to original vertex IDs from the file
    tour_ids = [str(ids[i]) for i in tour]

    with open(filename, "w") as f:
        f.write(str(cost) + "\n")
        f.write(",".join(tour_ids) + "\n")

    print(f"Wrote solution to {filename}")



def main():
    """
    Usage expected by the project:

      BF:
        ./exec -inst <instance> -alg BF -time <cutoff>

      Approx:
        ./exec -inst <instance> -alg Approx -seed <seed>

      LS:
        ./exec -inst <instance> -alg LS -time <cutoff> -seed <seed>
    """
    

    parser = argparse.ArgumentParser(
        description='TSP Solve',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument('-inst', required=True)
    parser.add_argument('-alg', required=True,
                        choices=['BF', 'Approx', 'LS'])
    parser.add_argument('-time', type=int)
    parser.add_argument('-seed', type=int)

    args = parser.parse_args()
    instance = args.inst
    method = args.alg
    cutoff = args.time
    seed = args.seed

    # Update filepath, if someone did not add folder as prefix
    if not os.path.exists(instance):
        instance = os.path.join('Data', os.path.basename(instance))
        if not os.path.exists(instance):
            print(f"Error: File not found: {args.inst}")
            sys.exit(1)

    # Load instance
    ids, dist = parse_tsp_file(instance)

    if method == "BF":
        if cutoff is None:
            print("BF requires: ./exec -inst <instance> -alg BF -time <cutoff>")
            sys.exit(1)

        print(f"[BF] instance={instance}, cutoff={cutoff}")
        cost, tour = tsp_bruteforce(dist, cutoff)
        write_solution(instance, method, cutoff, seed, cost, tour, ids)

    elif method == "Approx":
        if seed is None:
            print("Approx requires: ./exec -inst <instance> -alg Approx -seed <seed>")
            sys.exit(1)

        print(f"[Approx] instance={instance}, seed={seed}")
        cost, tour = tsp_approx(ids, dist, seed)
        write_solution(instance, method, cutoff, seed, cost, tour, ids)

    elif method == "LS":
        if seed is None or cutoff is None:
            print("LS requires: ./exec -inst <instance> -alg LS -time <cutoff> -seed <seed>")
            sys.exit(1)

        print(f"[LS] instance={instance}, cutoff={cutoff}, seed={seed}")
        cost, tour = tsp_local_search(dist, cutoff, seed)
        write_solution(instance, method, cutoff, seed, cost, tour, ids)

    else:
        print(f"Unknown method: {method}")
        sys.exit(1)


if __name__ == "__main__":
    main()
