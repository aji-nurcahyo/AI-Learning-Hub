"""
search_framework.py
Kerangka Kerja Abstrak Algoritma Pencarian AI (Search Framework).
Menyediakan kelas dasar Node, SearchResult, SearchProblem, serta dua masalah benchmark:
1. GridNavigationProblem (Navigasi grid 2D berhalangan dengan Manhattan/Euclidean heuristic)
2. RomaniaMapProblem (Graf berbobot klasik peta Romania dengan Straight-Line Heuristic ke Bucharest)

Sesuai standar kurikulum AI Learning Hub - Institut Teknologi Sumatera
Berdasarkan formulasi formal Russell & Norvig (2020).
"""

from typing import Any, List, Tuple, Set, Dict, Optional
import math


class Node:
    """Representasi sebuah simpul pada pohon/graf pencarian (Search Node)."""

    def __init__(self, state: Any, parent: Optional['Node'] = None,
                 action: Optional[Any] = None, path_cost: float = 0.0, depth: int = 0):
        self.state = state
        self.parent = parent
        self.action = action
        self.path_cost = path_cost
        self.depth = depth

    def path(self) -> List['Node']:
        """Merekonstruksi daftar node dari root hingga node saat ini."""
        node = self
        result = []
        while node is not None:
            result.append(node)
            node = node.parent
        return list(reversed(result))

    def solution(self) -> List[Any]:
        """Mengembalikan urutan tindakan (actions) dari root ke goal."""
        return [node.action for node in self.path()[1:]]

    def __repr__(self) -> str:
        return f"<Node {self.state} cost={self.path_cost} depth={self.depth}>"

    # Operator pembanding untuk tie-breaking di heapq
    def __lt__(self, other: 'Node') -> bool:
        return self.path_cost < other.path_cost


class SearchResult:
    """Objek rangkuman hasil eksekusi algoritma pencarian."""

    def __init__(self, solution_path: Optional[List[Any]], path_cost: float,
                 nodes_expanded: int, nodes_generated: int, max_frontier_size: int,
                 status: str = "SUCCESS"):
        self.solution_path = solution_path  # Daftar actions
        self.path_cost = path_cost
        self.nodes_expanded = nodes_expanded
        self.nodes_generated = nodes_generated
        self.max_frontier_size = max_frontier_size
        self.status = status  # 'SUCCESS', 'FAILURE', atau 'CUTOFF'

    def __repr__(self) -> str:
        return (f"SearchResult(status={self.status}, cost={self.path_cost}, "
                f"expanded={self.nodes_expanded}, generated={self.nodes_generated}, "
                f"max_frontier={self.max_frontier_size})")


class SearchProblem:
    """Kelas abstrak formulasi masalah pencarian AI formal (reuse dari Chapter 03)."""

    def __init__(self, initial_state: Any, goal_state: Any = None):
        self.initial_state = initial_state
        self.goal_state = goal_state

    def actions(self, state: Any) -> List[Any]:
        """Mengembalikan daftar tindakan sah dari state ini."""
        raise NotImplementedError

    def result(self, state: Any, action: Any) -> Any:
        """Fungsi transisi deterministik: T(s, a) -> s'."""
        raise NotImplementedError

    def is_goal(self, state: Any) -> bool:
        """Menguji apakah state ini memenuhi kriteria tujuan."""
        return state == self.goal_state

    def step_cost(self, state: Any, action: Any, next_state: Any) -> float:
        """Biaya langkah perpindahan c(s, a, s'). Standar: 1.0."""
        return 1.0

    def heuristic(self, state: Any) -> float:
        """Fungsi heuristik h(n): estimasi biaya minimum menuju goal. Default: 0.0."""
        return 0.0


# =====================================================================
# BENCHMARK 1: GRID NAVIGATION PROBLEM
# =====================================================================

class GridNavigationProblem(SearchProblem):
    """Masalah navigasi grid 2D dengan koordinat (baris, kolom) dan rintangan."""

    def __init__(self, initial_state: Tuple[int, int], goal_state: Tuple[int, int],
                 rows: int, cols: int, obstacles: Optional[Set[Tuple[int, int]]] = None,
                 step_cost_value: float = 1.0):
        super().__init__(initial_state, goal_state)
        self.rows = rows
        self.cols = cols
        self.obstacles = obstacles or set()
        self.step_cost_value = step_cost_value
        
        # Urutan eksplorasi deterministik: UP, DOWN, LEFT, RIGHT
        self.action_vectors: Dict[str, Tuple[int, int]] = {
            "UP": (-1, 0),
            "DOWN": (1, 0),
            "LEFT": (0, -1),
            "RIGHT": (0, 1),
        }

    def actions(self, state: Tuple[int, int]) -> List[str]:
        r, c = state
        valid = []
        for act, (dr, dc) in self.action_vectors.items():
            nr, nc = r + dr, c + dc
            if 0 <= nr < self.rows and 0 <= nc < self.cols and (nr, nc) not in self.obstacles:
                valid.append(act)
        return valid

    def result(self, state: Tuple[int, int], action: str) -> Tuple[int, int]:
        if action not in self.action_vectors:
            return state
        dr, dc = self.action_vectors[action]
        nr, nc = state[0] + dr, state[1] + dc
        if 0 <= nr < self.rows and 0 <= nc < self.cols and (nr, nc) not in self.obstacles:
            return (nr, nc)
        return state

    def step_cost(self, state: Tuple[int, int], action: str, next_state: Tuple[int, int]) -> float:
        return self.step_cost_value

    def heuristic(self, state: Tuple[int, int]) -> float:
        """Heuristik Manhattan Distance untuk pergerakan 4 arah pada grid unweighted."""
        if self.goal_state is None:
            return 0.0
        return float(abs(state[0] - self.goal_state[0]) + abs(state[1] - self.goal_state[1]))

    def euclidean_heuristic(self, state: Tuple[int, int]) -> float:
        """Heuristik Euclidean Distance garis lurus (Straight-line)."""
        if self.goal_state is None:
            return 0.0
        return math.sqrt((state[0] - self.goal_state[0])**2 + (state[1] - self.goal_state[1])**2)


# =====================================================================
# BENCHMARK 2: ROMANIA MAP PROBLEM (RUSSELL & NORVIG KLASIK)
# =====================================================================

# Peta jalan dan jarak (km) antar kota di Romania
ROMANIA_MAP: Dict[str, Dict[str, float]] = {
    'Arad': {'Zerind': 75.0, 'Sibiu': 140.0, 'Timisoara': 118.0},
    'Zerind': {'Arad': 75.0, 'Oradea': 71.0},
    'Oradea': {'Zerind': 71.0, 'Sibiu': 151.0},
    'Sibiu': {'Arad': 140.0, 'Oradea': 151.0, 'Fagaras': 99.0, 'Rimnicu Vilcea': 80.0},
    'Fagaras': {'Sibiu': 99.0, 'Bucharest': 211.0},
    'Rimnicu Vilcea': {'Sibiu': 80.0, 'Craiova': 146.0, 'Pitesti': 97.0},
    'Pitesti': {'Rimnicu Vilcea': 97.0, 'Craiova': 138.0, 'Bucharest': 101.0},
    'Timisoara': {'Arad': 118.0, 'Lugoj': 111.0},
    'Lugoj': {'Timisoara': 111.0, 'Mehadia': 70.0},
    'Mehadia': {'Lugoj': 70.0, 'Dobreta': 75.0},
    'Dobreta': {'Mehadia': 75.0, 'Craiova': 120.0},
    'Craiova': {'Dobreta': 120.0, 'Rimnicu Vilcea': 146.0, 'Pitesti': 138.0},
    'Bucharest': {'Fagaras': 211.0, 'Pitesti': 101.0, 'Giurgiu': 90.0, 'Urziceni': 85.0},
    'Giurgiu': {'Bucharest': 90.0},
    'Urziceni': {'Bucharest': 85.0, 'Vaslui': 142.0, 'Hirsova': 98.0},
    'Hirsova': {'Urziceni': 98.0, 'Eforie': 86.0},
    'Eforie': {'Hirsova': 86.0},
    'Vaslui': {'Urziceni': 142.0, 'Iasi': 92.0},
    'Iasi': {'Vaslui': 92.0, 'Neamt': 87.0},
    'Neamt': {'Iasi': 87.0}
}

# Jarak garis lurus (Straight-line distance, SLD) ke Bucharest
ROMANIA_SLD_BUCHAREST: Dict[str, float] = {
    'Arad': 366.0,
    'Bucharest': 0.0,
    'Craiova': 160.0,
    'Dobreta': 242.0,
    'Eforie': 161.0,
    'Fagaras': 176.0,
    'Giurgiu': 77.0,
    'Hirsova': 151.0,
    'Iasi': 226.0,
    'Lugoj': 244.0,
    'Mehadia': 241.0,
    'Neamt': 234.0,
    'Oradea': 380.0,
    'Pitesti': 100.0,
    'Rimnicu Vilcea': 193.0,
    'Sibiu': 253.0,
    'Timisoara': 329.0,
    'Urziceni': 80.0,
    'Vaslui': 199.0,
    'Zerind': 374.0
}


class RomaniaMapProblem(SearchProblem):
    """Masalah navigasi graf berbobot antar kota di Romania menuju Bucharest."""

    def __init__(self, initial_state: str = 'Arad', goal_state: str = 'Bucharest'):
        super().__init__(initial_state, goal_state)
        self.map = ROMANIA_MAP
        self.sld = ROMANIA_SLD_BUCHAREST

    def actions(self, state: str) -> List[str]:
        """Tindakan sah adalah berpindah ke kota tetangga (terurut alfabetis untuk determinisme)."""
        if state not in self.map:
            return []
        return sorted(list(self.map[state].keys()))

    def result(self, state: str, action: str) -> str:
        """Hasil aksi adalah kota tujuan tetangga."""
        return action

    def step_cost(self, state: str, action: str, next_state: str) -> float:
        """Bobot jalan (km) antara kota state ke next_state."""
        if state in self.map and next_state in self.map[state]:
            return self.map[state][next_state]
        return float('inf')

    def heuristic(self, state: str) -> float:
        """Straight-line distance ke Bucharest (admissible & consistent)."""
        return self.sld.get(state, 0.0)

