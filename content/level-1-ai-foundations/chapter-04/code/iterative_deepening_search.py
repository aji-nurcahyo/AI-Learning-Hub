"""
iterative_deepening_search.py
Implementasi Iterative Deepening Depth-First Search (IDDFS).
Karakteristik:
- Memadukan keunggulan efisiensi ruang linier DFS O(b * d) dengan kelengkapan dan optimalitas BFS (unweighted).
- Menjalankan DLS secara berulang dari kedalaman limit = 0 hingga solusi ditemukan atau kegagalan pasti.
- Meskipun node tingkat atas digenerate ulang, overhead komputasi hanya proporsional kecil karena sebagian besar node berada pada daun terbawah O(b^d).

Kurikulum AI Learning Hub - Institut Teknologi Sumatera
"""

from typing import Optional
from search_framework import SearchProblem, Node, SearchResult, GridNavigationProblem, RomaniaMapProblem
from depth_limited_search import depth_limited_search


def iterative_deepening_search(problem: SearchProblem, max_depth: int = 50) -> SearchResult:
    """Menjalankan IDDFS dengan menaikkan limit kedalaman dari 0 hingga max_depth."""
    total_expanded = 0
    total_generated = 0
    peak_frontier = 0

    for depth in range(max_depth + 1):
        res = depth_limited_search(problem, limit=depth)
        total_expanded += res.nodes_expanded
        total_generated += res.nodes_generated
        peak_frontier = max(peak_frontier, res.max_frontier_size)

        if res.status == "SUCCESS":
            return SearchResult(
                solution_path=res.solution_path,
                path_cost=res.path_cost,
                nodes_expanded=total_expanded,
                nodes_generated=total_generated,
                max_frontier_size=peak_frontier,
                status="SUCCESS"
            )
        elif res.status == "FAILURE":
            # Seluruh ruang keadaan telah habis dieksplorasi di dalam limit tanpa ada cutoff
            return SearchResult(solution_path=None, path_cost=float('inf'),
                                nodes_expanded=total_expanded, nodes_generated=total_generated,
                                max_frontier_size=peak_frontier, status="FAILURE")

    return SearchResult(solution_path=None, path_cost=float('inf'),
                        nodes_expanded=total_expanded, nodes_generated=total_generated,
                        max_frontier_size=peak_frontier, status="CUTOFF")


def main():
    print("=== PENGUJIAN ITERATIVE DEEPENING SEARCH (IDDFS) ===")
    
    # Uji Kasus: Romania Map (Arad -> Bucharest)
    romania_prob = RomaniaMapProblem('Arad', 'Bucharest')
    res = iterative_deepening_search(romania_prob, max_depth=10)
    print("\n1. Peta Romania (Arad -> Bucharest) dengan IDDFS:")
    print(f"   Status             : {res.status}")
    print(f"   Solusi (Kota)      : {res.solution_path}")
    print(f"   Total Jarak        : {res.path_cost} km")
    print(f"   Total Node Di-expand: {res.nodes_expanded}")
    print(f"   Total Node Dibuat  : {res.nodes_generated}")
    print(f"   Puncak Frontier Mem: {res.max_frontier_size} node")


if __name__ == "__main__":
    main()

