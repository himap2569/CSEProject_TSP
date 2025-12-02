#!/usr/bin/env python3

"""
Benchmarking script for TSP algorithms (Brute Force, Approx, Local Search).
Runs each algorithm on all .tsp instances in Data/, measures runtime and solution quality,
and outputs results to results.csv.
"""

import os
import time
import glob
import csv
import math
import subprocess

DATA_DIR = "Data"           # folder with .tsp files
EXEC_NAME = "exec"          # the provided exec wrapper

BF_CUTOFF = 300             # seconds for brute force
LS_CUTOFF = 300             # seconds for local search
LS_SEEDS = list(range(10))  # at least 10 seeds as required

RESULTS_CSV = "results.csv"

def run_cmd(cmd):
    """Run a shell command and measure wall-clock time (in seconds)."""
    start = time.time()
    subprocess.run(cmd, check=True)
    end = time.time()
    return end - start


def parse_sol_file(path):
    """
    Reads a .sol file:
      line 1: cost
      line 2: comma-separated vertex IDs
    Returns cost (int).
    """
    with open(path, "r") as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]
    if not lines:
        raise ValueError(f"Empty solution file: {path}")
    cost = int(lines[0])
    return cost


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    exec_path = os.path.join(script_dir, EXEC_NAME)

    # get all .tsp instances
    tsp_paths = sorted(glob.glob(os.path.join(script_dir, DATA_DIR, "*.tsp")))
    if not tsp_paths:
        print("No .tsp files found in Data/. Check DATA_DIR.")
        return

    rows = []

    for inst_path in tsp_paths:
        inst_name = os.path.basename(inst_path)
        base = os.path.splitext(inst_name)[0].lower()
        print(f"\n=== Instance: {inst_name} ===")

        # Brute Force
        print("Running BF...")
        bf_time = run_cmd([
            exec_path,
            "-inst", inst_path,
            "-alg", "BF",
            "-time", str(BF_CUTOFF),
        ])

        bf_sol = os.path.join(script_dir, f"{base}_BF_{BF_CUTOFF}.sol")
        bf_cost = parse_sol_file(bf_sol)

        # Heuristic: if BF finishes well before cutoff, we assume full tour explored
        bf_full = bf_time < 0.95 * BF_CUTOFF

        # Approx
        print("Running Approx...")
        approx_seed = 0
        approx_time = run_cmd([
            exec_path,
            "-inst", inst_path,
            "-alg", "Approx",
            "-seed", str(approx_seed),
        ])

        approx_sol = os.path.join(script_dir, f"{base}_Approx_{approx_seed}.sol")
        approx_cost = parse_sol_file(approx_sol)
        approx_full = True  # MST-based approx always returns a full tour

        # Local Search (multiple seeds) 
        print("Running LS with multiple seeds...")
        ls_times = []
        ls_costs = []

        for seed in LS_SEEDS:
            print(f"  LS seed={seed}")
            t = run_cmd([
                exec_path,
                "-inst", inst_path,
                "-alg", "LS",
                "-time", str(LS_CUTOFF),
                "-seed", str(seed),
            ])
            ls_times.append(t)

            ls_sol = os.path.join(script_dir, f"{base}_LS_{LS_CUTOFF}_{seed}.sol")
            ls_cost = parse_sol_file(ls_sol)
            ls_costs.append(ls_cost)

        ls_avg_time = sum(ls_times) / len(ls_times)
        ls_avg_cost = sum(ls_costs) / len(ls_costs)
        ls_best_cost = min(ls_costs)
        ls_full = True   # LS always maintains a full tour

        # Relative Errors (vs best LS)
        # RelError = (algo_cost - best_LS_cost) / best_LS_cost
        def rel_error(cost):
            return (cost - ls_best_cost) / ls_best_cost if ls_best_cost > 0 else 0.0

        bf_rel = rel_error(bf_cost)
        approx_rel = rel_error(approx_cost)
        ls_rel = rel_error(ls_avg_cost)

        # Add row
        rows.append({
            "instance": inst_name,
            "bf_time": bf_time,
            "bf_cost": bf_cost,
            "bf_full_tour": int(bf_full),
            "bf_rel_error": bf_rel,
            "approx_time": approx_time,
            "approx_cost": approx_cost,
            "approx_full_tour": 1,     # always full
            "approx_rel_error": approx_rel,
            "ls_avg_time": ls_avg_time,
            "ls_avg_cost": ls_avg_cost,
            "ls_full_tour": 1,         # always full
            "ls_rel_error": ls_rel,
        })

    # Write CSV
    fieldnames = [
        "instance",
        "bf_time", "bf_cost", "bf_full_tour", "bf_rel_error",
        "approx_time", "approx_cost", "approx_full_tour", "approx_rel_error",
        "ls_avg_time", "ls_avg_cost", "ls_full_tour", "ls_rel_error",
    ]

    with open(os.path.join(script_dir, RESULTS_CSV), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"\nWrote results to {RESULTS_CSV}")


if __name__ == "__main__":
    main()