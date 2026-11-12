"""
astar_search.py
Implementasi A* Search (Informed Heuristic Graph Search).
Karakteristik:
- Fungsi evaluasi: f(n) = g(n) + h(n)
  * g(n): Biaya aktual yang telah ditempuh dari initial state ke node n.
  * h(n): Estimasi biaya terpendek dari node n menuju goal (fungsi heuristik).
- Menjamin kelengkapan (complete) dan optimalitas biaya (optimal) jika heuristik memenuhi syarat konsistensi (consistent / monotone) pada graph search.
- Jika h(n) = 0 untuk seluruh node, A* berperilaku setara dengan Uniform-Cost Search (UCS).

Kurikulum AI Learning Hub - Institut Teknologi Sumatera
"""

import heapq
from typing import Optional, Dict, Any
from search_framework import SearchProblem, Node, SearchResult, GridNavigationProblem, RomaniaMapProblem


def astar_search(problem: SearchProblem) -> SearchResult:
    """Menjalankan A* Search menggunakan Priority Queue terurut berdasarkan f(n) = g(n) + h(n)."""
    initial_node = Node(state=problem.initial_state, path_cost=0.0, depth=0)
    
    # Priority Queue: (f_score, counter_id, node)
    counter = 0
    f_init = initial_node.path_cost + problem.heuristic(initial_node.state)
    frontier = []
    heapq.heappush(frontier, (f_init, counter, initial_node))

    # Memelihara biaya terbaik g(s) yang pernah dicapai untuk setiap state
    best_g: Dict[Any, float] = {initial_node.state: 0.0}
    explored = set()

    nodes_expanded = 0
    nodes_generated = 1
    max_frontier_size = 1

    while frontier:
        max_frontier_size = max(max_frontier_size, len(frontier))
        f_val, _, current_node = heapq.heappop(frontier)

        # Abaikan stale entries jika state telah diproses dengan g(n) yang lebih baik
        if current_node.state in explored:
            continue

        explored.add(current_node.state)
        nodes_expanded += 1

        # Goal test saat node di-pop untuk menjamin optimalitas A*
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
            new_g = current_node.path_cost + step_cost
            nodes_generated += 1

            # Jika ditemukan jalur baru atau jalur yang lebih murah ke child_state
            if child_state not in best_g or new_g < best_g[child_state]:
                best_g[child_state] = new_g
                child_node = Node(state=child_state, parent=current_node, action=action,
                                  path_cost=new_g, depth=current_node.depth + 1)
                f_score = new_g + problem.heuristic(child_state)
                counter += 1
                heapq.heappush(frontier, (f_score, counter, child_node))

    return SearchResult(solution_path=None, path_cost=float('inf'),
                        nodes_expanded=nodes_expanded, nodes_generated=nodes_generated,
                        max_frontier_size=max_frontier_size, status="FAILURE")


def main():
    print("=== PENGUJIAN A* SEARCH ===")
    
    # 1. Peta Romania (Arad -> Bucharest)
    romania_prob = RomaniaMapProblem('Arad', 'Bucharest')
    res_romania = astar_search(romania_prob)
    print("\n1. Peta Romania (Arad -> Bucharest) dengan A*:")
    print(f"   Status          : {res_romania.status}")
    print(f"   Solusi Rute     : {res_romania.solution_path}")
    print(f"   Total Jarak     : {res_romania.path_cost} km (Optimal 418 km!)")
    print(f"   Node Diekspansi : {res_romania.nodes_expanded} node")
    print(f"   Node Dibuat     : {res_romania.nodes_generated}")
    print("   Analisis        : A* menghasilkan rute optimal persis seperti UCS (418 km), namun mengekspansi node lebih sedikit berkat panduan heuristik garis lurus.")

    # 2. Grid Navigation dengan Manhattan Distance
    grid_prob = GridNavigationProblem(
        initial_state=(0, 0), goal_state=(3, 3), rows=4, cols=4,
        obstacles={(1, 1), (1, 2), (2, 1)}
    )
    res_grid = astar_search(grid_prob)
    print("\n2. Grid Navigation (0,0) -> (3,3) dengan Manhattan Heuristic:")
    print(f"   Status          : {res_grid.status}")
    print(f"   Solusi (Actions): {res_grid.solution_path}")
    print(f"   Total Biaya     : {res_grid.path_cost}")
    print(f"   Node Diekspansi : {res_grid.nodes_expanded}")


if __name__ == "__main__":
    main()

