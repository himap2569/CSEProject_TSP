#!/usr/bin/env python3

"""
TSP Multi-City Performance Analysis through Graphs

This script runs TSP algorithms (Brute Force and Local Search) across multiple cities
and generates comparative performance plots showing solution quality improvement over time.

Main Functions:
- run_cmd(cmd): Executes subprocess command and measures wall-clock execution time
- parse_sol_file(path): Extracts tour cost from .sol output files
- process_city(city_name, script_dir, exec_path): Runs BF and LS algorithms for a single city
    - Tests BF with cutoff times: [1, 5, 10, 30, 50, 100, 200, 300] seconds
    - Tests LS with same cutoffs, averaging over 10 random seeds
    - Early termination for BF if solution found before cutoff
    - Returns dict with times, costs for both algorithms

Visualization:
- Generates two comparison plots (one per algorithm) across all cities
- Each city's performance normalized to its own final solution (ends at 0%)
- Y-axis shows "% Above Final Solution" to visualize improvement trajectory
- X-axis uses linear time scale in seconds
- Red horizontal line at y=0 marks final solution baseline

Output:
- plots/bf_all_cities_comparison.png: BF performance across cities
- plots/ls_all_cities_comparison.png: LS performance across cities

Command-line Usage:
- python3 generateAllCityPlot.py -cities Cincinnati NYC Roanoke

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


def process_city(city_name, script_dir, exec_path):
    """Process a single city and return results."""
    inst_name = f"{city_name}.tsp"
    inst_path = os.path.join(script_dir, DATA_DIR, inst_name)
    
    if not os.path.exists(inst_path):
        print(f"Warning: Instance {inst_name} not found, skipping...")
        return None
        
    base = os.path.splitext(inst_name)[0].lower()
    print(f"\n=== Processing {inst_name} ===")

    # Storage for results
    bf_times = []
    bf_costs = []
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

    return {
        'city': city_name,
        'bf_times': bf_times,
        'bf_costs': bf_costs,
        'ls_times': ls_times,
        'ls_costs': ls_costs
    }


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run TSP algorithm analysis on multiple cities')
    parser.add_argument('-cities', '--cities', type=str, nargs='+', required=True,
                       help='City names (e.g., Cincinnati NYC Roanoke)')
    args = parser.parse_args()
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    exec_path = os.path.join(script_dir, EXEC_NAME)
    
    # Create output directory
    os.makedirs(os.path.join(script_dir, OUTPUT_DIR), exist_ok=True)

    # Process all cities
    all_results = []
    for city in args.cities:
        result = process_city(city, script_dir, exec_path)
        if result is not None:
            all_results.append(result)
    
    if not all_results:
        print("\nError: No valid results collected from any city")
        sys.exit(1)

    # Create Comparison Plots
    print("\n=== Creating comparison plots ===")
    
    # Define colors for different cities
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    
    # Plot 1: Brute Force - Normalized relative to final value
    plt.figure(figsize=(12, 7))
    
    for idx, result in enumerate(all_results):
        if result['bf_times'] and result['bf_costs']:
            # Normalize relative to the LAST (final/best) value
            final_cost = result['bf_costs'][-1]
            normalized_bf = [(c / final_cost - 1) * 100 for c in result['bf_costs']]
            
            plt.plot(result['bf_times'], normalized_bf, 'o-', 
                    label=result['city'], linewidth=2.5, markersize=8, 
                    color=colors[idx % len(colors)], alpha=0.8)
    
    plt.xlabel('Time (seconds)', fontsize=13, fontweight='bold')
    plt.ylabel('% Above Final Solution', fontsize=13, fontweight='bold')
    plt.title('Brute Force: Solution Quality Improvement Over Time', fontsize=15, fontweight='bold')
    plt.legend(fontsize=11, loc='best', framealpha=0.9)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.axhline(y=0, color='red', linestyle='--', linewidth=2, alpha=0.7)
    plt.tight_layout()
    
    plot_path = os.path.join(script_dir, OUTPUT_DIR, "bf_all_cities_comparison.png")
    plt.savefig(plot_path, dpi=300)
    print(f"Saved: {plot_path}")
    plt.close()
    
    # Plot 2: Local Search - Normalized relative to final value
    plt.figure(figsize=(12, 7))
    
    for idx, result in enumerate(all_results):
        if result['ls_times'] and result['ls_costs']:
            # Normalize relative to the LAST (final/best) value
            final_cost = result['ls_costs'][-1]
            normalized_ls = [(c / final_cost - 1) * 100 for c in result['ls_costs']]
            
            plt.plot(result['ls_times'], normalized_ls, '^-', 
                    label=result['city'], linewidth=2.5, markersize=8, 
                    color=colors[idx % len(colors)], alpha=0.8)
    
    plt.xlabel('Time (seconds)', fontsize=13, fontweight='bold')
    plt.ylabel('% Above Final Solution', fontsize=13, fontweight='bold')
    plt.title('Local Search: Solution Quality Improvement Over Time', fontsize=15, fontweight='bold')
    plt.legend(fontsize=11, loc='best', framealpha=0.9)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.axhline(y=0, color='red', linestyle='--', linewidth=2, alpha=0.7)
    plt.tight_layout()
    
    plot_path = os.path.join(script_dir, OUTPUT_DIR, "ls_all_cities_comparison.png")
    plt.savefig(plot_path, dpi=300)
    print(f"Saved: {plot_path}")
    plt.close()

    print(f"\nAll plots saved to {OUTPUT_DIR}/ directory")
    print("\nGenerated plots:")
    print("  - bf_all_cities_comparison.png")
    print("  - ls_all_cities_comparison.png")


if __name__ == "__main__":
    main()