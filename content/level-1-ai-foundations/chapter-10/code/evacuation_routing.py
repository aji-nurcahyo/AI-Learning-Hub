"""Informed A* Search & Evacuation Path Routing.

Chapter 10: AI Foundations Capstone.
Mengimplementasikan algoritma A* Search dengan heuristik Manhattan admissible & consistent
untuk merutekan jalur evakuasi optimal terbebas dari ancaman titik api dan asap beracun.
Membandingkan efisiensi eksplorasi simpul A* vs Breadth-First Search (BFS).
"""

import heapq
from collections import deque
from typing import Dict, List, Optional, Set, Tuple
from problem_formulation import CampusSearchProblem
from knowledge_rules import CampusKnowledgeBase
import pandas as pd
from pathlib import Path


def manhattan_heuristic(state: Tuple[int, int], goals: Set[Tuple[int, int]]) -> float:
    """Heuristik jarak Manhattan minimum ke titik kumpul aman terdekat.
    
    Sifat Teoretis:
    - Admissible: h(n) <= h*(n) karena pergerakan hanya 4 arah dan biaya langkah >= 1.0.
    - Consistent: h(n) <= c(n, a, n') + h(n') mematuhi ketidaksamaan segitiga.
    """
    x, y = state
    return min(abs(x - gx) + abs(y - gy) for gx, gy in goals)


def a_star_search(
    problem: CampusSearchProblem
) -> Tuple[Optional[List[Tuple[int, int]]], float, int]:
    """Algoritma A* Graph Search dengan Frontier Priority Queue."""
    start = problem.initial_state
    if problem.is_goal(start):
        return [start], 0.0, 0

    # Entry: (f_score, g_score, state, path)
    counter = 0
    start_h = manhattan_heuristic(start, problem.goals)
    frontier = [(start_h, 0.0, counter, start, [start])]
    explored: Set[Tuple[int, int]] = set()
    best_g: Dict[Tuple[int, int], float] = {start: 0.0}
    nodes_expanded = 0

    while frontier:
        f, g, _, current_state, path = heapq.heappop(frontier)

        if problem.is_goal(current_state):
            return path, g, nodes_expanded

        if current_state in explored:
            continue
        explored.add(current_state)
        nodes_expanded += 1

        for action in problem.actions(current_state):
            next_state = problem.result(current_state, action)
            cost = problem.step_cost(current_state, action, next_state)
            new_g = g + cost

            if next_state not in best_g or new_g < best_g[next_state]:
                best_g[next_state] = new_g
                h = manhattan_heuristic(next_state, problem.goals)
                counter += 1
                heapq.heappush(frontier, (new_g + h, new_g, counter, next_state, path + [next_state]))

    return None, float('inf'), nodes_expanded


def bfs_search(
    problem: CampusSearchProblem
) -> Tuple[Optional[List[Tuple[int, int]]], float, int]:
    """Breadth-First Search (BFS) tanpa heuristik sebagai baseline pembanding."""
    start = problem.initial_state
    if problem.is_goal(start):
        return [start], 0.0, 0

    queue = deque([(start, [start], 0.0)])
    explored: Set[Tuple[int, int]] = {start}
    nodes_expanded = 0

    while queue:
        current_state, path, g = queue.popleft()
        nodes_expanded += 1

        for action in problem.actions(current_state):
            next_state = problem.result(current_state, action)
            cost = problem.step_cost(current_state, action, next_state)
            new_g = g + cost

            if problem.is_goal(next_state):
                return path + [next_state], new_g, nodes_expanded

            if next_state not in explored:
                explored.add(next_state)
                queue.append((next_state, path + [next_state], new_g))

    return None, float('inf'), nodes_expanded


def run_routing_experiment() -> None:
    data_dir = Path(__file__).resolve().parent / "data"
    clean_csv = data_dir / "campus_telemetry_clean.csv"
    if not clean_csv.exists():
        raise FileNotFoundError(f"{clean_csv} belum tersedia. Jalankan sensor_data_pipeline.py!")

    df = pd.read_csv(clean_csv)

    # 1. Bangun sensor readings dari dataset bersih
    readings = []
    for _, r in df.iterrows():
        readings.append({
            "coord": (int(r["grid_x"]), int(r["grid_y"])),
            "temp_c": float(r["temp_c"]),
            "smoke_ppm": float(r["smoke_ppm"]),
            "has_ramp": bool(r["has_ramp"])
        })

    # 2. Inferensikan zona bahaya via Knowledge Base
    kb = CampusKnowledgeBase()
    blocked, penalties = kb.infer_safety_status(readings, wheelchair_mode=False)

    # 3. Bentuk instance masalah pencarian
    start_pos = (1, 1)
    goals = {(9, 9), (0, 9)}
    problem = CampusSearchProblem(
        grid_size=(10, 10),
        initial_state=start_pos,
        goals=goals,
        blocked_zones=blocked,
        hazard_penalties=penalties
    )

    print("=== PERBANDINGAN ALGORITMA PENCARIAN RUTE EVAKUASI KAMPUS ===")
    print(f"Posisi Awal Evakuasi : {start_pos}")
    print(f"Titik Kumpul Target  : {goals}")
    print(f"Zona Api Terblokir   : {blocked}\n")

    # Eksekusi A* Search
    path_astar, cost_astar, nodes_astar = a_star_search(problem)

    # Eksekusi BFS Baseline
    path_bfs, cost_bfs, nodes_bfs = bfs_search(problem)

    print(f"{'Metrik Evaluasi':<28} | {'A* Search (Informed)':<22} | {'BFS (Uninformed)':<20}")
    print("-" * 76)
    print(f"{'Panjang Jalur (Langkah)':<28} | {len(path_astar):<22} | {len(path_bfs):<20}")
    print(f"{'Total Biaya Jalur (Path Cost)':<28} | {cost_astar:<22.2f} | {cost_bfs:<20.2f}")
    print(f"{'Simpul Dieksplorasi (Nodes)':<28} | {nodes_astar:<22} | {nodes_bfs:<20}")
    efficiency = ((nodes_bfs - nodes_astar) / nodes_bfs) * 100 if nodes_bfs > 0 else 0
    print(f"{'Efisiensi Pencarian (%)':<28} | {efficiency:<21.1f}% | {'Baseline':<20}")
    print("-" * 76)

    print("\nJALUR OPTIMAL A*:")
    print(" -> ".join(str(coord) for coord in path_astar))


def main() -> None:
    run_routing_experiment()


if __name__ == "__main__":
    main()

