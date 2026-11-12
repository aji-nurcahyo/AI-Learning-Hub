"""
greedy_best_first_search.py
Implementasi Greedy Best-First Search menggunakan fungsi evaluasi f(n) = h(n).
Karakteristik:
- Memilih node dengan estimasi jarak terdekat menuju tujuan (heuristic h(n) terkecil).
- Sangat cepat menemukan tujuan dalam banyak masalah praktis (ekspansi node sedikit).
- Keterbatasan: Tidak menjamin optimalitas (dapat terjebak rute memutar yang mahal) dan tidak lengkap pada graf tak hingga dengan siklus.

Kurikulum AI Learning Hub - Institut Teknologi Sumatera
"""

import heapq
from typing import Optional, Dict
from search_framework import SearchProblem, Node, SearchResult, GridNavigationProblem, RomaniaMapProblem


def greedy_best_first_search(problem: SearchProblem) -> SearchResult:
    """Menjalankan Greedy Best-First Search terurut berdasarkan heuristik h(n)."""
    initial_node = Node(state=problem.initial_state, path_cost=0.0, depth=0)
    
    # Priority Queue: (h(state), counter_id, node)
    counter = 0
    h_init = problem.heuristic(initial_node.state)
    frontier = []
    heapq.heappush(frontier, (h_init, counter, initial_node))

    explored = set()
    frontier_states = {initial_node.state}

    nodes_expanded = 0
    nodes_generated = 1
    max_frontier_size = 1

    while frontier:
        max_frontier_size = max(max_frontier_size, len(frontier))
        h_val, _, current_node = heapq.heappop(frontier)
        
        if current_node.state in frontier_states:
            frontier_states.remove(current_node.state)

        if current_node.state in explored:
            continue

        explored.add(current_node.state)
        nodes_expanded += 1

        # Goal test saat pop
        if problem.is_goal(current_node.state):
            return SearchResult(
                solution_path=current_node.solution(),
                path_cost=current_node.path_cost,
                nodes_expanded=nodes_expanded,
                nodes_generated=nodes_generated,
                max_frontier_size=max_frontier_size,
                status="SUCCESS"
            )

        for action in problem.actions(current_node.state):
            child_state = problem.result(current_node.state, action)
            step_cost = problem.step_cost(current_node.state, action, child_state)
            cost = current_node.path_cost + step_cost
            nodes_generated += 1

            if child_state not in explored and child_state not in frontier_states:
                child_node = Node(state=child_state, parent=current_node, action=action,
                                  path_cost=cost, depth=current_node.depth + 1)
                h_child = problem.heuristic(child_state)
                counter += 1
                heapq.heappush(frontier, (h_child, counter, child_node))
                frontier_states.add(child_state)

    return SearchResult(solution_path=None, path_cost=float('inf'),
                        nodes_expanded=nodes_expanded, nodes_generated=nodes_generated,
                        max_frontier_size=max_frontier_size, status="FAILURE")


def main():
    print("=== PENGUJIAN GREEDY BEST-FIRST SEARCH ===")
    
    # Uji Kasus: Romania Map (Arad -> Bucharest)
    romania_prob = RomaniaMapProblem('Arad', 'Bucharest')
    res = greedy_best_first_search(romania_prob)
    print("\n1. Peta Romania (Arad -> Bucharest) dengan Greedy Search:")
    print(f"   Status          : {res.status}")
    print(f"   Solusi Rute     : {res.solution_path}")
    print(f"   Total Jarak     : {res.path_cost} km")
    print(f"   Node Diekspansi : {res.nodes_expanded} node (Sangat sedikit/rakus)")
    print("   Catatan         : Greedy langsung memilih Sibiu -> Fagaras -> Bucharest (450 km) karena h(Fagaras)=176 < h(Rimnicu Vilcea)=193, meskipun total jarak aslinya lebih boros dibanding via Rimnicu Vilcea (418 km).")


if __name__ == "__main__":
    main()

