"""
dfs_search.py
Implementasi Depth-First Search (DFS) menggunakan LIFO Stack.
Karakteristik:
- Mengeksplorasi cabang terdalam terlebih dahulu (deepest unexpanded nodes first).
- Hemat memori pada tree search O(b * m), namun pada graph search menyimpan explored set.
- Tidak menjamin kelengkapan (completeness) pada ruang keadaan tak berhingga dan tidak menjamin optimalitas biaya/langkah.

Kurikulum AI Learning Hub - Institut Teknologi Sumatera
"""

from typing import Optional
from search_framework import SearchProblem, Node, SearchResult, GridNavigationProblem, RomaniaMapProblem


def depth_first_search(problem: SearchProblem) -> SearchResult:
    """Menjalankan Depth-First Search (Graph Search) menggunakan LIFO Stack."""
    initial_node = Node(state=problem.initial_state, path_cost=0.0, depth=0)
    if problem.is_goal(initial_node.state):
        return SearchResult(solution_path=[], path_cost=0.0,
                            nodes_expanded=0, nodes_generated=1, max_frontier_size=1)

    # LIFO stack untuk frontier
    frontier = [initial_node]
    frontier_states = {initial_node.state}
    explored = set()

    nodes_expanded = 0
    nodes_generated = 1
    max_frontier_size = 1

    while frontier:
        max_frontier_size = max(max_frontier_size, len(frontier))
        current_node = frontier.pop()
        frontier_states.remove(current_node.state)
        explored.add(current_node.state)
        nodes_expanded += 1

        # Goal test saat node di-pop dari stack
        if problem.is_goal(current_node.state):
            return SearchResult(
                solution_path=current_node.solution(),
                path_cost=current_node.path_cost,
                nodes_expanded=nodes_expanded,
                nodes_generated=nodes_generated,
                max_frontier_size=max_frontier_size,
                status="SUCCESS"
            )

        # Ambil tindakan legal (dibalik agar aksi pertama dieksekusi terlebih dahulu di LIFO stack)
        available_actions = problem.actions(current_node.state)
        for action in reversed(available_actions):
            child_state = problem.result(current_node.state, action)
            cost = current_node.path_cost + problem.step_cost(current_node.state, action, child_state)
            child_node = Node(state=child_state, parent=current_node, action=action,
                              path_cost=cost, depth=current_node.depth + 1)
            nodes_generated += 1

            if child_state not in explored and child_state not in frontier_states:
                frontier.append(child_node)
                frontier_states.add(child_state)

    return SearchResult(solution_path=None, path_cost=float('inf'),
                        nodes_expanded=nodes_expanded, nodes_generated=nodes_generated,
                        max_frontier_size=max_frontier_size, status="FAILURE")


def main():
    print("=== PENGUJIAN DEPTH-FIRST SEARCH (DFS) ===")
    
    # Uji Kasus 1: Grid Navigation 4x4
    grid_prob = GridNavigationProblem(
        initial_state=(0, 0), goal_state=(3, 3), rows=4, cols=4,
        obstacles={(1, 1), (1, 2), (2, 1)}
    )
    res_grid = depth_first_search(grid_prob)
    print("\n1. Grid Navigation (0,0) -> (3,3):")
    print(f"   Status          : {res_grid.status}")
    print(f"   Solusi (Actions): {res_grid.solution_path}")
    print(f"   Total Biaya     : {res_grid.path_cost}")
    print(f"   Node Diekspansi : {res_grid.nodes_expanded}")
    print(f"   Catatan         : DFS dapat menghasilkan jalur yang lebih panjang/berkelok-kelok dibanding BFS.")

    # Uji Kasus 2: Romania Map (Arad -> Bucharest)
    romania_prob = RomaniaMapProblem('Arad', 'Bucharest')
    res_romania = depth_first_search(romania_prob)
    print("\n2. Peta Romania (Arad -> Bucharest):")
    print(f"   Status          : {res_romania.status}")
    print(f"   Solusi (Kota)   : {res_romania.solution_path}")
    print(f"   Total Jarak     : {res_romania.path_cost} km")
    print(f"   Node Diekspansi : {res_romania.nodes_expanded}")


if __name__ == "__main__":
    main()

