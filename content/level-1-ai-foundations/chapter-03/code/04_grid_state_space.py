"""
04_grid_state_space.py
Representasi Ruang Keadaan (State Space Representation) pada Grid Navigasi 3x3.
Mendemonstrasikan konsep dasar AI Problem Solving:
- States: Himpunan koordinat (x, y) yang dapat dicapai.
- Actions: Himpunan aksi diskret (UP, DOWN, LEFT, RIGHT).
- Transition Model: T(s, a) -> s' yang memvalidasi batas dinding dan rintangan.

Kurikulum AI Learning Hub - Institut Teknologi Sumatera
"""

from typing import Tuple, List, Dict, Set


# Tipe data untuk koordinat (baris, kolom)
State = Tuple[int, int]
Action = str


class GridStateSpace:
    """Representasi formal ruang keadaan grid 2D diskret."""

    def __init__(self, rows: int = 3, cols: int = 3, obstacles: Set[State] = None):
        self.rows = rows
        self.cols = cols
        self.obstacles = obstacles or set()
        
        # Seluruh kemungkinan state dalam semesta grid
        self.all_states: Set[State] = {
            (r, c)
            for r in range(rows)
            for c in range(cols)
            if (r, c) not in self.obstacles
        }

        # Vektor pergeseran aksi (dr, dc)
        self.action_vectors: Dict[Action, Tuple[int, int]] = {
            "UP": (-1, 0),
            "DOWN": (1, 0),
            "LEFT": (0, -1),
            "RIGHT": (0, 1),
        }

    def is_valid_state(self, state: State) -> bool:
        """Memeriksa apakah koordinat berada di dalam grid dan bukan rintangan."""
        r, c = state
        return 0 <= r < self.rows and 0 <= c < self.cols and (r, c) not in self.obstacles

    def get_legal_actions(self, state: State) -> List[Action]:
        """Menghasilkan himpunan aksi legal dari state s saat ini."""
        if not self.is_valid_state(state):
            return []
        
        legal = []
        r, c = state
        for action, (dr, dc) in self.action_vectors.items():
            next_state = (r + dr, c + dc)
            if self.is_valid_state(next_state):
                legal.append(action)
        return legal

    def transition(self, state: State, action: Action) -> State:
        """Model Transisi Determinis: T(s, a) -> s'."""
        if action not in self.get_legal_actions(state):
            # Jika aksi tidak valid (menabrak dinding/rintangan), status tidak berubah
            return state
        
        r, c = state
        dr, dc = self.action_vectors[action]
        return (r + dr, c + dc)

    def print_grid_map(self, current: State = None, goal: State = None):
        """Visualisasi peta grid ASCII."""
        print(f"Peta Grid ({self.rows}x{self.cols}):")
        for r in range(self.rows):
            row_str = []
            for c in range(self.cols):
                st = (r, c)
                if st in self.obstacles:
                    row_str.append("[ X ]")  # Rintangan
                elif st == current and st == goal:
                    row_str.append("[ * ]")  # Posisi saat ini = Goal
                elif st == current:
                    row_str.append("[ A ]")  # Posisi Agen
                elif st == goal:
                    row_str.append("[ G ]")  # Posisi Goal
                else:
                    row_str.append("[ . ]")  # Jalur kosong
            print("  " + " ".join(row_str))
        print("  Legenda: [A]=Agen, [G]=Tujuan, [X]=Rintangan, [.]=Lantai Kosong\n")


def main():
    print("=== REPRESENSTASI FORMAL RUANG KEADAAN GRID ===")
    
    # Rintangan pada koordinat (1, 1) tengah
    obstacles = {(1, 1)}
    grid = GridStateSpace(rows=3, cols=3, obstacles=obstacles)

    start_state = (0, 0)
    goal_state = (2, 2)

    print(f"Total State Legal yang Tersedia: {len(grid.all_states)} states")
    print(f"Himpunan State: {sorted(list(grid.all_states))}\n")

    grid.print_grid_map(current=start_state, goal=goal_state)

    # Uji coba transisi beruntun manual: (0,0) -> DOWN -> DOWN -> RIGHT -> RIGHT
    test_sequence = ["DOWN", "DOWN", "RIGHT", "RIGHT"]
    
    curr = start_state
    print(f"Mulai eksplorasi dari Initial State: {curr}")
    for step, action in enumerate(test_sequence, start=1):
        legal_actions = grid.get_legal_actions(curr)
        next_state = grid.transition(curr, action)
        print(f"Step {step}: Di {curr} -> Aksi '{action}' (Legal: {legal_actions}) -> Pindah ke {next_state}")
        curr = next_state

    print(f"\nPosisi Akhir: {curr}")
    is_goal = (curr == goal_state)
    print(f"Goal Test Tercapai: {'YA, SUKSES!' if is_goal else 'BELUM TERCAPAI'}")


if __name__ == "__main__":
    main()

