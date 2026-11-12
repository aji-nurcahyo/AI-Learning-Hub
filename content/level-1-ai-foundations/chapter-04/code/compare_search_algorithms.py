"""
compare_search_algorithms.py
Skrip Analisis dan Evaluasi Komparatif 7 Algoritma Pencarian AI:
1. Breadth-First Search (BFS)
2. Depth-First Search (DFS)
3. Depth-Limited Search (DLS, limit=4)
4. Iterative Deepening Search (IDDFS)
5. Uniform-Cost Search (UCS)
6. Greedy Best-First Search
7. A* Search

Diuji pada dua skenario masalah:
- Skenario A: Peta Graf Berbobot Romania (Arad -> Bucharest)
- Skenario B: Navigasi Grid 2D 4x4 dengan Rintangan

Kurikulum AI Learning Hub - Institut Teknologi Sumatera
"""

import time
from typing import Dict, List, Any
from search_framework import GridNavigationProblem, RomaniaMapProblem
from bfs_search import breadth_first_search
from dfs_search import depth_first_search
from depth_limited_search import depth_limited_search
from iterative_deepening_search import iterative_deepening_search
from uniform_cost_search import uniform_cost_search
from greedy_best_first_search import greedy_best_first_search
from astar_search import astar_search


def benchmark_runner(name: str, search_fn, problem, **kwargs) -> Dict[str, Any]:
    """Menjalankan algoritma dan mencatat waktu eksekusi serta metrik performa."""
    start_time = time.perf_counter()
    # Jalankan algoritma
    res = search_fn(problem, **kwargs)
    elapsed_ms = (time.perf_counter() - start_time) * 1000.0

    path_len = len(res.solution_path) if res.solution_path is not None else 0
    return {
        "algorithm": name,
        "status": res.status,
        "path_length": path_len,
        "path_cost": res.path_cost,
        "expanded": res.nodes_expanded,
        "generated": res.nodes_generated,
        "max_frontier": res.max_frontier_size,
        "time_ms": elapsed_ms,
        "solution": res.solution_path
    }


def print_comparison_table(title: str, results: List[Dict[str, Any]]):
    """Mencetak tabel perbandingan performa dalam format yang rapi."""
    print(f"\n{'='*95}")
    print(f"BENCHMARK: {title.upper()}")
    print(f"{'='*95}")
    header = f"{'Algoritma':<25} | {'Status':<8} | {'Langkah':<7} | {'Biaya/Jarak':<12} | {'Expanded':<8} | {'Generated':<9} | {'Frontier':<8} | {'Waktu (ms)':<10}"
    print(header)
    print("-" * 95)
    for r in results:
        cost_str = f"{r['path_cost']:.1f}" if r['path_cost'] != float('inf') else "inf"
        print(f"{r['algorithm']:<25} | {r['status']:<8} | {r['path_length']:<7} | {cost_str:<12} | {r['expanded']:<8} | {r['generated']:<9} | {r['max_frontier']:<8} | {r['time_ms']:<10.3f}")
    print("=" * 95)


def run_all_benchmarks():
    # -------------------------------------------------------------
    # SKENARIO 1: PETA ROMANIA (Arad -> Bucharest)
    # -------------------------------------------------------------
    romania_prob = RomaniaMapProblem('Arad', 'Bucharest')
    romania_results = [
        benchmark_runner("Breadth-First Search", breadth_first_search, romania_prob),
        benchmark_runner("Depth-First Search", depth_first_search, romania_prob),
        benchmark_runner("Depth-Limited (L=4)", depth_limited_search, romania_prob, limit=4),
        benchmark_runner("Iterative Deepening", iterative_deepening_search, romania_prob, max_depth=8),
        benchmark_runner("Uniform-Cost Search", uniform_cost_search, romania_prob),
        benchmark_runner("Greedy Best-First", greedy_best_first_search, romania_prob),
        benchmark_runner("A* Search", astar_search, romania_prob),
    ]
    print_comparison_table("Peta Graf Berbobot Romania (Arad -> Bucharest)", romania_results)

    # -------------------------------------------------------------
    # SKENARIO 2: GRID NAVIGATION 4x4 DENGAN RINTANGAN
    # -------------------------------------------------------------
    grid_prob = GridNavigationProblem(
        initial_state=(0, 0), goal_state=(3, 3), rows=4, cols=4,
        obstacles={(1, 1), (1, 2), (2, 1)}
    )
    grid_results = [
        benchmark_runner("Breadth-First Search", breadth_first_search, grid_prob),
        benchmark_runner("Depth-First Search", depth_first_search, grid_prob),
        benchmark_runner("Depth-Limited (L=6)", depth_limited_search, grid_prob, limit=6),
        benchmark_runner("Iterative Deepening", iterative_deepening_search, grid_prob, max_depth=10),
        benchmark_runner("Uniform-Cost Search", uniform_cost_search, grid_prob),
        benchmark_runner("Greedy Best-First", greedy_best_first_search, grid_prob),
        benchmark_runner("A* Search (Manhattan)", astar_search, grid_prob),
    ]
    print_comparison_table("Navigasi Grid 2D 4x4 dengan Rintangan", grid_results)


if __name__ == "__main__":
    run_all_benchmarks()

