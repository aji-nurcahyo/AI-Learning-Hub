"""
05_problem_formulation_search_prep.py
Formulasi Formal Masalah Pencarian AI (Problem Formulation).
Mengimplementasikan 5 komponen formal standar Russell & Norvig:
1. Initial State (Status Awal)
2. Actions (Himpunan Tindakan Legal)
3. Transition Model (Fungsi Hasil Transisi)
4. Goal Test (Uji Ketercapaian Tujuan)
5. Path Cost (Akumulasi Biaya Jalur & Step Cost)

Sebagai jembatan dari konsep Intelligent Agent menuju Algoritma Pencarian (Chapter 04).

Kurikulum AI Learning Hub - Institut Teknologi Sumatera
"""

from typing import Any, List, Tuple, Set


class SearchProblem:
    """Kelas abstrak dasar untuk mendefinisikan masalah pencarian formal."""

    def __init__(self, initial_state: Any, goal_state: Any = None):
        self.initial_state = initial_state
        self.goal_state = goal_state

    def actions(self, state: Any) -> List[Any]:
        """Mengembalikan daftar tindakan legal yang dapat diambil dari state ini."""
        raise NotImplementedError

    def result(self, state: Any, action: Any) -> Any:
        """Model Transisi: Mengembalikan state penerus setelah mengeksekusi action."""
        raise NotImplementedError

    def is_goal(self, state: Any) -> bool:
        """Goal Test: Menguji apakah state ini memenuhi syarat tujuan."""
        return state == self.goal_state

    def step_cost(self, state: Any, action: Any, next_state: Any) -> float:
        """Biaya satu langkah perpindahan c(s, a, s'). Standar: 1.0."""
        return 1.0

    def path_cost(self, path: List[Tuple[Any, Any]]) -> float:
        """Menghitung total biaya jalur dari daftar tuple (state, action)."""
        cost = 0.0
        for i in range(len(path) - 1):
            s_curr, act = path[i]
            s_next, _ = path[i + 1]
            cost += self.step_cost(s_curr, act, s_next)
        return cost


class GridNavigationProblem(SearchProblem):
    """Implementasi konkrit perumusan masalah navigasi grid robot."""

    def __init__(self, initial_state: Tuple[int, int], goal_state: Tuple[int, int],
                 rows: int = 3, cols: int = 3, obstacles: Set[Tuple[int, int]] = None):
        super().__init__(initial_state, goal_state)
        self.rows = rows
        self.cols = cols
        self.obstacles = obstacles or set()
        self.vectors = {
            "UP": (-1, 0),
            "DOWN": (1, 0),
            "LEFT": (0, -1),
            "RIGHT": (0, 1)
        }

    def actions(self, state: Tuple[int, int]) -> List[str]:
        r, c = state
        valid = []
        for act, (dr, dc) in self.vectors.items():
            nr, nc = r + dr, c + dc
            if 0 <= nr < self.rows and 0 <= nc < self.cols and (nr, nc) not in self.obstacles:
                valid.append(act)
        return valid

    def result(self, state: Tuple[int, int], action: str) -> Tuple[int, int]:
        if action not in self.actions(state):
            return state
        r, c = state
        dr, dc = self.vectors[action]
        return (r + dr, c + dc)

    def is_goal(self, state: Tuple[int, int]) -> bool:
        return state == self.goal_state

    def step_cost(self, state: Tuple[int, int], action: str, next_state: Tuple[int, int]) -> float:
        # Contoh variasi step cost: perpindahan horizontal/vertikal standar bernilai 1.0
        return 1.0


def demonstrate_problem_formulation():
    print("=== DEMONSTRASI PROBLEM FORMULATION LENGKAP ===")
    
    # Formulasi: Grid 3x3, mulai di (0,0), target di (2,2), rintangan di (0,1) dan (1,1)
    obstacles = {(0, 1), (1, 1)}
    problem = GridNavigationProblem(
        initial_state=(0, 0),
        goal_state=(2, 2),
        rows=3,
        cols=3,
        obstacles=obstacles
    )

    print("1. Initial State :", problem.initial_state)
    print("2. Goal State    :", problem.goal_state)
    print("3. Rintangan     :", problem.obstacles)
    print("4. Actions di Initial State:", problem.actions(problem.initial_state))

    # Simulasi sebuah rencana jalur (plan): (0,0) -DOWN-> (1,0) -DOWN-> (2,0) -RIGHT-> (2,1) -RIGHT-> (2,2)
    candidate_plan = ["DOWN", "DOWN", "RIGHT", "RIGHT"]
    print(f"\nUji Jalur Rencana: {candidate_plan}")

    current_state = problem.initial_state
    history = [(current_state, candidate_plan[0])]
    total_cost = 0.0

    for step_num, act in enumerate(candidate_plan, start=1):
        available_actions = problem.actions(current_state)
        if act not in available_actions:
            print(f"Error pada Step {step_num}: Tindakan '{act}' tidak legal dari {current_state}!")
            break
        
        next_state = problem.result(current_state, act)
        cost = problem.step_cost(current_state, act, next_state)
        total_cost += cost
        print(f"  Step {step_num}: Dari {current_state} ambil '{act}' -> Tiba di {next_state} (Biaya langkah: {cost})")
        
        current_state = next_state
        history.append((current_state, act if step_num < len(candidate_plan) else None))

    print(f"\nHasil Akhir Status: {current_state}")
    print(f"Goal Reached?     : {problem.is_goal(current_state)}")
    print(f"Total Path Cost   : {total_cost} unit biaya")


if __name__ == "__main__":
    demonstrate_problem_formulation()

