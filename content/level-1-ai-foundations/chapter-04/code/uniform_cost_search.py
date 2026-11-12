"""
uniform_cost_search.py
Implementasi Uniform-Cost Search (UCS / Dijkstra's Algorithm pada Search Graph).
Karakteristik:
- Menjamin jalur dengan biaya kumulatif minimum g(n) pada sembarang graf berbobot non-negatif.
- Menggunakan Priority Queue (heapq) dengan urutan prioritas terkecil g(n).
- Goal test wajib dieksekusi saat node di-pop dari frontier, bukan saat generasi!
- Menangani perbaruan jalur yang lebih baik (better path updates) dengan memelihara best_cost.

Kurikulum AI Learning Hub - Institut Teknologi Sumatera
"""

import heapq
from typing import Optional, Dict
from search_framework import SearchProblem, Node, SearchResult, GridNavigationProblem, RomaniaMapProblem


def uniform_cost_search(problem: SearchProblem) -> SearchResult:
    """Menjalankan Uniform-Cost Search (Graph Search) terurut berdasarkan path_cost g(n)."""
    initial_node = Node(state=problem.initial_state, path_cost=0.0, depth=0)
    
    # Priority Queue tuple: (path_cost, counter_id, node)
    # counter_id digunakan untuk tie-breaking deterministik FIFO saat cost bernilai sama persis
    counter = 0
    frontier = []
    heapq.heappush(frontier, (initial_node.path_cost, counter, initial_node))

    # Memelihara biaya terbaik yang pernah dicapai untuk setiap status
    best_cost: Dict[Any, float] = {initial_node.state: 0.0}
    explored = set()

    nodes_expanded = 0
    nodes_generated = 1
    max_frontier_size = 1

    while frontier:
        max_frontier_size = max(max_frontier_size, len(frontier))
        cost, _, current_node = heapq.heappop(frontier)

        # Abaikan stale entries jika sudah ada jalur yang lebih murah ditemukan sebelumnya
        if current_node.state in explored:
            continue

        explored.add(current_node.state)
        nodes_expanded += 1

        # Goal test saat node di-pop untuk menjamin optimalitas biaya minimum
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
            new_cost = current_node.path_cost + step_cost
            nodes_generated += 1

            # Jika state belum pernah dikunjungi atau ditemukan jalur dengan biaya lebih rendah
            if child_state not in best_cost or new_cost < best_cost[child_state]:
                best_cost[child_state] = new_cost
                child_node = Node(state=child_state, parent=current_node, action=action,
                                  path_cost=new_cost, depth=current_node.depth + 1)
                counter += 1
                heapq.heappush(frontier, (new_cost, counter, child_node))

    return SearchResult(solution_path=None, path_cost=float('inf'),
                        nodes_expanded=nodes_expanded, nodes_generated=nodes_generated,
                        max_frontier_size=max_frontier_size, status="FAILURE")


def main():
    print("=== PENGUJIAN UNIFORM-COST SEARCH (UCS) ===")
    
    # Uji Kasus: Romania Map (Arad -> Bucharest)
    romania_prob = RomaniaMapProblem('Arad', 'Bucharest')
    res = uniform_cost_search(romania_prob)
    print("\n1. Peta Romania (Arad -> Bucharest) dengan UCS:")
    print(f"   Status          : {res.status}")
    print(f"   Solusi Rute     : {res.solution_path}")
    print(f"   Total Jarak     : {res.path_cost} km (Optimal terpendek!)")
    print(f"   Node Diekspansi : {res.nodes_expanded}")
    print(f"   Node Dibuat     : {res.nodes_generated}")
    print(f"   Maks Frontier   : {res.max_frontier_size}")
    print("   Penjelasan      : Rute Sibiu -> Rimnicu Vilcea -> Pitesti -> Bucharest (418 km) terbukti lebih hemat daripada via Fagaras (450 km).")


if __name__ == "__main__":
    main()

