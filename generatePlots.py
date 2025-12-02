#!/usr/bin/env python3

"""
TSP Single-City Algorithm Comparison Tool

This script runs all three TSP algorithms (Brute Force, Approximation, and Local Search)
on a single specified city and generates a comparative performance plot.

Main Functions:
- run_cmd(cmd): Executes subprocess command and measures wall-clock execution time
- parse_sol_file(path): Extracts tour cost from .sol output files

Algorithm Execution:
- Brute Force: Tests with cutoff times [1, 5, 10, 30, 50, 100, 200, 300] seconds
    - Early termination if solution found before cutoff
- Approximation: Single run with seed=0 (deterministic MST-based algorithm)
- Local Search: Tests same cutoffs as BF, averaging over 10 random seeds (0-9)

Visualization:
- Single normalized comparison plot showing all three algorithms
- All algorithms normalized to the best solution found across all methods
- Y-axis: "Solution Quality Gap (%)" - percentage above optimal solution
- X-axis: Logarithmic time scale in seconds
- Red horizontal line at y=0 marks the best solution
- Text annotation displays the best solution cost found

Output:
- plots/<city>_combined_normalized.png: Single plot comparing all algorithms

Command-line Usage:
- python3 generatePlots.py -city Atlanta
- Requires: exec binary, Data/ directory with .tsp files

Configuration:
- CUTOFF_TIMES: List of time limits to test [1, 5, 10, 30, 50, 100, 200, 300]
- LS_SEEDS: Range of random seeds for LS averaging [0-9]
- OUTPUT_DIR: Directory for plot output (default: "plots")
"""

import os
import sys
import time
import subprocess
import matplotlib.pyplot as plt
import numpy as np
import argparse

DATA_DIR = "Data"
EXEC_NAME = "exec"

# Testing different cutoff times
CUTOFF_TIMES = [1, 5, 10, 30, 50, 100, 200, 300]

LS_SEEDS = list(range(10))  # average over 10 seeds for LS

OUTPUT_DIR = "plots"

def run_cmd(cmd):
    """Run a shell command and measure wall-clock time (in seconds)."""
    start = time.time()
    subprocess.run(cmd, check=True, capture_output=True)
    end = time.time()
    return end - start

def parse_sol_file(path):
    """
    Reads a .sol file and returns cost (int).
    """
    with open(path, "r") as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]
    if not lines:
        return None
    cost = int(lines[0])
    return cost

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run TSP algorithm analysis on specified city')
    parser.add_argument('-city', '--city', type=str, required=True,
                       help='City name (e.g., Cincinnati, NYC, Roanoke)')
    args = parser.parse_args()
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    exec_path = os.path.join(script_dir, EXEC_NAME)
    
    # Create output directory
    os.makedirs(os.path.join(script_dir, OUTPUT_DIR), exist_ok=True)

    # Construct instance filename
    inst_name = f"{args.city}.tsp"
    inst_path = os.path.join(script_dir, DATA_DIR, inst_name)

    if not os.path.exists(inst_path):
        print(f"Error: Instance {inst_name} not found at {inst_path}")
        print(f"Please check that the file exists in the {DATA_DIR} directory")
        sys.exit(1)

    base = os.path.splitext(inst_name)[0].lower()
    print(f"\n=== Processing {inst_name} ===")

    # Storage for results
    bf_times = []
    bf_costs = []

    approx_time = None
    approx_cost = None

    ls_times = []
    ls_costs = []

    # Brute Force with varying cutoffs
    print("Testing BF with different cutoffs...")
    for cutoff in CUTOFF_TIMES:
        print(f"  BF cutoff={cutoff}s")
        elapsed = run_cmd([
            exec_path,
            "-inst", inst_path,
            "-alg", "BF",
            "-time", str(cutoff),
        ])
        
        if elapsed is None:
            continue
            
        bf_sol = os.path.join(script_dir, f"{base}_BF_{cutoff}.sol")
        cost = parse_sol_file(bf_sol)
        
        if cost is not None:
            bf_times.append(elapsed)
            bf_costs.append(cost)
        
        # If BF finished early, no point testing longer cutoffs
        if elapsed < 0.95 * cutoff:
            print(f"    BF completed in {elapsed:.2f}s, skipping longer cutoffs")
            break

    # Approx (single run, no cutoff variation)
    print("Running Approx...")
    elapsed = run_cmd([
        exec_path,
        "-inst", inst_path,
        "-alg", "Approx",
        "-seed", "0",
    ])
    
    if elapsed is not None:
        approx_sol = os.path.join(script_dir, f"{base}_Approx_0.sol")
        cost = parse_sol_file(approx_sol)
        if cost is not None:
            approx_time = elapsed
            approx_cost = cost

    # Local Search with varying cutoffs
    print("Testing LS with different cutoffs...")
    for cutoff in CUTOFF_TIMES:
        print(f"  LS cutoff={cutoff}s")
        costs_for_cutoff = []
        times_for_cutoff = []
        
        for seed in LS_SEEDS:
            elapsed = run_cmd([
                exec_path,
                "-inst", inst_path,
                "-alg", "LS",
                "-time", str(cutoff),
                "-seed", str(seed),
            ])
            
            if elapsed is None:
                continue
                
            ls_sol = os.path.join(script_dir, f"{base}_LS_{cutoff}_{seed}.sol")
            cost = parse_sol_file(ls_sol)
            
            if cost is not None:
                costs_for_cutoff.append(cost)
                times_for_cutoff.append(elapsed)
        
        if costs_for_cutoff:
            ls_times.append(np.mean(times_for_cutoff))
            ls_costs.append(np.mean(costs_for_cutoff))

    # Create Normalized Comparison Plot
    print(f"Creating normalized comparison plot for {inst_name}...")

    # Normalized comparison (quality relative to best known)
    all_costs = []
    if bf_costs:
        all_costs.extend(bf_costs)
    if approx_cost:
        all_costs.append(approx_cost)
    if ls_costs:
        all_costs.extend(ls_costs)
    
    if all_costs:
        best_cost = min(all_costs)
        
        plt.figure(figsize=(12, 7))
        
        if bf_times and bf_costs:
            normalized_bf = [(c / best_cost - 1) * 100 for c in bf_costs]
            plt.plot(bf_times, normalized_bf, 'o-', label='Brute Force', 
                    linewidth=2.5, markersize=10, color='#1f77b4', alpha=0.8)
        
        if approx_time and approx_cost:
            normalized_approx = (approx_cost / best_cost - 1) * 100
            plt.plot([approx_time], [normalized_approx], 's', label='Approx', 
                    markersize=15, color='#2ca02c', alpha=0.8, markeredgewidth=2, markeredgecolor='darkgreen')
        
        if ls_times and ls_costs:
            normalized_ls = [(c / best_cost - 1) * 100 for c in ls_costs]
            plt.plot(ls_times, normalized_ls, '^-', label='Local Search (avg)', 
                    linewidth=2.5, markersize=10, color='#ff7f0e', alpha=0.8)
        
        plt.xscale('log')
        plt.xlabel('Time (seconds, log scale)', fontsize=13, fontweight='bold')
        plt.ylabel('Solution Quality Gap (%)', fontsize=13, fontweight='bold')
        plt.title(f'Algorithm Comparison: % Above Best Solution\n{inst_name}', fontsize=15, fontweight='bold')
        plt.legend(fontsize=11, loc='best', framealpha=0.9)
        plt.grid(True, alpha=0.3, which='both', linestyle='--')
        plt.axhline(y=0, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Optimal (0%)')
        
        # Add text annotation for best solution
        plt.text(0.02, 0.98, f'Best solution: {best_cost}', 
                transform=plt.gca().transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        plot_path = os.path.join(script_dir, OUTPUT_DIR, f"{base}_combined_normalized.png")
        plt.savefig(plot_path, dpi=300)
        print(f"  Saved: {plot_path}")
        plt.close()
    else:
        print("  Warning: No valid cost data collected, skipping plot generation")

    print(f"\nPlot saved to {OUTPUT_DIR}/ directory")


if __name__ == "__main__":
    main()