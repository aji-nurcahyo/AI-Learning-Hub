"""
depth_limited_search.py
Implementasi Depth-Limited Search (DLS).
Karakteristik:
- Membatasi kedalaman eksplorasi DFS hingga batas kedalaman tetap (limit L).
- Mengatasi kelemahan loop tak berhingga pada DFS murni.
- Wajib membedakan status CUTOFF (pencarian terpotong batas kedalaman) dan FAILURE (seluruh ruang keadaan habis tanpa solusi).

Kurikulum AI Learning Hub - Institut Teknologi Sumatera
"""

from typing import Tuple, Optional, List
from search_framework import SearchProblem, Node, SearchResult, GridNavigationProblem, RomaniaMapProblem


def depth_limited_search(problem: SearchProblem, limit: int) -> SearchResult:
    """Menjalankan Depth-Limited Search dengan batas kedalaman limit."""
    initial_node = Node(state=problem.initial_state, path_cost=0.0, depth=0)
    
    nodes_expanded = 0
    nodes_generated = 1
    max_frontier_size = 1

    # Rekursif DLS internal
    def recursive_dls(node: Node, current_limit: int, path_states: set) -> Tuple[Optional[Node], str]:
        nonlocal nodes_expanded, nodes_generated, max_frontier_size
        max_frontier_size = max(max_frontier_size, len(path_states))

        if problem.is_goal(node.state):
            return node, "SUCCESS"
        elif current_limit == 0:
            return None, "CUTOFF"
        else:
            cutoff_occurred = False
            nodes_expanded += 1
            for action in problem.actions(node.state):
                child_state = problem.result(node.state, action)
                if child_state in path_states:
                    continue  # Cegah siklus pada jalur aktif saat ini

                cost = node.path_cost + problem.step_cost(node.state, action, child_state)
                child_node = Node(state=child_state, parent=node, action=action,
                                  path_cost=cost, depth=node.depth + 1)
                nodes_generated += 1

                path_states.add(child_state)
                result_node, status = recursive_dls(child_node, current_limit - 1, path_states)
                path_states.remove(child_state)

                if status == "SUCCESS":
                    return result_node, "SUCCESS"
                elif status == "CUTOFF":
                    cutoff_occurred = True

            if cutoff_occurred:
                return None, "CUTOFF"
            else:
                return None, "FAILURE"

    res_node, final_status = recursive_dls(initial_node, limit, {initial_node.state})
    if final_status == "SUCCESS" and res_node:
        return SearchResult(
            solution_path=res_node.solution(),
            path_cost=res_node.path_cost,
            nodes_expanded=nodes_expanded,
            nodes_generated=nodes_generated,
            max_frontier_size=max_frontier_size,
            status="SUCCESS"
        )
    elif final_status == "CUTOFF":
        return SearchResult(solution_path=None, path_cost=float('inf'),
                            nodes_expanded=nodes_expanded, nodes_generated=nodes_generated,
                            max_frontier_size=max_frontier_size, status="CUTOFF")
    else:
        return SearchResult(solution_path=None, path_cost=float('inf'),
                            nodes_expanded=nodes_expanded, nodes_generated=nodes_generated,
                            max_frontier_size=max_frontier_size, status="FAILURE")


def main():
    print("=== PENGUJIAN DEPTH-LIMITED SEARCH (DLS) ===")
    romania_prob = RomaniaMapProblem('Arad', 'Bucharest')

    # Kasus A: Limit = 2 (Terlalu dangkal, Bucharest ada di depth >= 3)
    res_cutoff = depth_limited_search(romania_prob, limit=2)
    print("\n1. Peta Romania dengan Limit = 2 (Terlalu Dangkal):")
    print(f"   Status          : {res_cutoff.status} (Terpotong batas limit)")
    print(f"   Node Diekspansi : {res_cutoff.nodes_expanded}")

    # Kasus B: Limit = 4 (Cukup dalam)
    res_ok = depth_limited_search(romania_prob, limit=4)
    print("\n2. Peta Romania dengan Limit = 4 (Cukup Dalam):")
    print(f"   Status          : {res_ok.status}")
    print(f"   Solusi (Kota)   : {res_ok.solution_path}")
    print(f"   Total Jarak     : {res_ok.path_cost} km")
    print(f"   Node Diekspansi : {res_ok.nodes_expanded}")


if __name__ == "__main__":
    main()

