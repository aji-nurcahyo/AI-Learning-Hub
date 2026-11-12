"""
bfs_search.py
Implementasi Breadth-First Search (BFS) menggunakan FIFO Queue (collections.deque).
Karakteristik:
- Mengeksplorasi node lapis demi lapis (shallowest unexpanded nodes first).
- Optimal untuk unweighted graph (menjamin solusi dengan jumlah edge/langkah paling sedikit).
- Early goal test: memeriksa tujuan saat node digenerate (menghemat ekspansi satu level).

Kurikulum AI Learning Hub - Institut Teknologi Sumatera
"""

from collections import deque
from typing import Optional
from search_framework import SearchProblem, Node, SearchResult, GridNavigationProblem, RomaniaMapProblem


def breadth_first_search(problem: SearchProblem) -> SearchResult:
    """Menjalankan Breadth-First Search (Graph Search) pada SearchProblem."""
    # 1. Cek apakah start node sudah merupakan goal
    initial_node = Node(state=problem.initial_state, path_cost=0.0, depth=0)
    if problem.is_goal(initial_node.state):
        return SearchResult(solution_path=[], path_cost=0.0,
                            nodes_expanded=0, nodes_generated=1, max_frontier_size=1)

    # 2. Inisialisasi FIFO queue dan explored set
    frontier = deque([initial_node])
    frontier_states = {initial_node.state}
    explored = set()

    nodes_expanded = 0
    nodes_generated = 1
    max_frontier_size = 1

    while frontier:
        max_frontier_size = max(max_frontier_size, len(frontier))
        current_node = frontier.popleft()
        frontier_states.remove(current_node.state)
        explored.add(current_node.state)
        nodes_expanded += 1

        # Ekspansi tetangga (actions)
        for action in problem.actions(current_node.state):
            child_state = problem.result(current_node.state, action)
            cost = current_node.path_cost + problem.step_cost(current_node.state, action, child_state)
            child_node = Node(state=child_state, parent=current_node, action=action,
                              path_cost=cost, depth=current_node.depth + 1)
            nodes_generated += 1

            if child_state not in explored and child_state not in frontier_states:
                # Early Goal Test
                if problem.is_goal(child_state):
                    return SearchResult(
                        solution_path=child_node.solution(),
                        path_cost=child_node.path_cost,
                        nodes_expanded=nodes_expanded,
                        nodes_generated=nodes_generated,
                        max_frontier_size=max_frontier_size,
                        status="SUCCESS"
                    )
                frontier.append(child_node)
                frontier_states.add(child_state)

    # Jika frontier kosong dan goal belum tercapai
    return SearchResult(solution_path=None, path_cost=float('inf'),
                        nodes_expanded=nodes_expanded, nodes_generated=nodes_generated,
                        max_frontier_size=max_frontier_size, status="FAILURE")


def main():
    print("=== PENGUJIAN BREADTH-FIRST SEARCH (BFS) ===")
    
    # Uji Kasus 1: Grid Navigation 4x4
    grid_prob = GridNavigationProblem(
        initial_state=(0, 0), goal_state=(3, 3), rows=4, cols=4,
        obstacles={(1, 1), (1, 2), (2, 1)}
    )
    res_grid = breadth_first_search(grid_prob)
    print("\n1. Grid Navigation (0,0) -> (3,3):")
    print(f"   Status          : {res_grid.status}")
    print(f"   Solusi (Actions): {res_grid.solution_path}")
    print(f"   Total Biaya     : {res_grid.path_cost}")
    print(f"   Node Diekspansi : {res_grid.nodes_expanded}")
    print(f"   Node Dibuat     : {res_grid.nodes_generated}")
    print(f"   Maks Frontier   : {res_grid.max_frontier_size}")

    # Uji Kasus 2: Romania Map (Arad -> Bucharest)
    romania_prob = RomaniaMapProblem('Arad', 'Bucharest')
    res_romania = breadth_first_search(romania_prob)
    print("\n2. Peta Romania (Arad -> Bucharest):")
    print(f"   Status          : {res_romania.status}")
    print(f"   Solusi (Kota)   : {res_romania.solution_path}")
    print(f"   Total Jarak     : {res_romania.path_cost} km")
    print(f"   Node Diekspansi : {res_romania.nodes_expanded}")
    print(f"   Catatan         : BFS menjamin rute langkah kota paling sedikit (3 hop), bukan jarak terpendek.")


if __name__ == "__main__":
    main()

