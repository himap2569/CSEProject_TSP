#!/usr/bin/env python3
import sys
import os
import math
import time
import itertools
import random

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


# ---------- Optimized Brute-force TSP with cutoff ----------

def tsp_bruteforce(dist, cutoff_seconds):
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


# ---------- Approx (TO BE IMPLEMENTED) ----------

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
    start = 0
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
        for v in sorted(adj[u], key=lambda x: dist[u][x]):
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


# ---------- Local Search (TO BE IMPLEMENTED) ----------

def tsp_local_search(ids, dist, cutoff_seconds, seed):
    """
    Local search TSP algorithm (e.g., 2-opt, simulated annealing) goes here.
    """
    raise NotImplementedError("Local search algorithm not implemented yet.")


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
        filename = f"{base} BF {cutoff}.sol"
    elif method == "Approx":
        filename = f"{base} Approx {seed}.sol"
    elif method == "LS":
        filename = f"{base} LS {cutoff} {seed}.sol"
    else:
        raise ValueError("Unknown method")

    # Map internal indices (0..n-1) back to original vertex IDs from the file
    tour_ids = [str(ids[i]) for i in tour]

    with open(filename, "w") as f:
        f.write(str(cost) + "\n")
        f.write(",".join(tour_ids) + "\n")

    print(f"Wrote solution to {filename}")



def main(argv):
    """
    Usage expected by the project:

      BF:
        ./exec <instance> BF <cutoff>

      Approx:
        ./exec <instance> Approx <seed>

      LS:
        ./exec <instance> LS <cutoff> <seed>
    """
    if len(argv) < 4:
        print("Usage:")
        print("  ./exec <instance> BF <cutoff>")
        print("  ./exec <instance> Approx <seed>")
        print("  ./exec <instance> LS <cutoff> <seed>")
        sys.exit(1)

    instance = argv[1]
    method = argv[2]

    # Load instance
    ids, dist = parse_tsp_file(instance)

    if method == "BF":
        cutoff = int(argv[3])
        seed = None
        print(f"[BF] instance={instance}, cutoff={cutoff}")
        cost, tour = tsp_bruteforce(dist, cutoff)
        write_solution(instance, method, cutoff, seed, cost, tour, ids)

    elif method == "Approx":
        seed = int(argv[3])
        cutoff = None
        print(f"[Approx] instance={instance}, seed={seed}")
        cost, tour = tsp_approx(ids, dist, seed)
        write_solution(instance, method, cutoff, seed, cost, tour, ids)

    elif method == "LS":
        if len(argv) < 5:
            print("LS requires: ./exec <instance> LS <cutoff> <seed>")
            sys.exit(1)
        cutoff = int(argv[3])
        seed = int(argv[4])
        print(f"[LS] instance={instance}, cutoff={cutoff}, seed={seed}")
        cost, tour = tsp_local_search(ids, dist, cutoff, seed)
        write_solution(instance, method, cutoff, seed, cost, tour, ids)

    else:
        print(f"Unknown method: {method}")
        sys.exit(1)


if __name__ == "__main__":
    main(sys.argv)