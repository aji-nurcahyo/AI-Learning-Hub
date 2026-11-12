"""AI Problem Formulation & PEAS Specification.

Chapter 10: AI Foundations Capstone.
Mendefinisikan spesifikasi formal lingkungan PEAS dan kelas abstrak
SearchProblem untuk sistem navigasi tanggap darurat kampus cerdas (ICER-Agent).
"""

from typing import List, Tuple, Dict, Set


class CampusSearchProblem:
    """Representasi formal masalah pencarian rute evakuasi kampus (Search Problem).
    
    Elemen Formal:
    1. Initial State : Posisi koordinat awal agen/warga kampus (x, y).
    2. Actions       : Himpunan aksi pergerakan valid {'UP', 'DOWN', 'LEFT', 'RIGHT'}.
    3. Transition    : Model transisi deterministik result(state, action).
    4. Goal Test     : Memeriksa apakah state berada di titik kumpul aman (Assembly Point).
    5. Path Cost     : Akumulasi bobot langkah (jarak dasar + penalti kedekatan bahaya).
    """

    def __init__(
        self,
        grid_size: Tuple[int, int] = (10, 10),
        initial_state: Tuple[int, int] = (1, 1),
        goals: Set[Tuple[int, int]] = None,
        blocked_zones: Set[Tuple[int, int]] = None,
        hazard_penalties: Dict[Tuple[int, int], float] = None
    ):
        self.width, self.height = grid_size
        self.initial_state = initial_state
        self.goals = goals if goals is not None else {(9, 9), (0, 9)}
        self.blocked_zones = blocked_zones if blocked_zones is not None else set()
        self.hazard_penalties = hazard_penalties if hazard_penalties is not None else {}

        self.action_vectors = {
            'UP': (0, 1),
            'DOWN': (0, -1),
            'LEFT': (-1, 0),
            'RIGHT': (1, 0)
        }

    def is_valid_coord(self, coord: Tuple[int, int]) -> bool:
        """Memeriksa batas grid dan zona terhalang (Blocked / Fire Hazard)."""
        x, y = coord
        in_bounds = (0 <= x < self.width) and (0 <= y < self.height)
        not_blocked = coord not in self.blocked_zones
        return in_bounds and not_blocked

    def actions(self, state: Tuple[int, int]) -> List[str]:
        """Mengembalikan daftar aksi legal yang tidak menabrak rintangan."""
        x, y = state
        valid_actions = []
        for act, (dx, dy) in self.action_vectors.items():
            next_coord = (x + dx, y + dy)
            if self.is_valid_coord(next_coord):
                valid_actions.push(act) if hasattr(valid_actions, 'push') else valid_actions.append(act)
        return valid_actions

    def result(self, state: Tuple[int, int], action: str) -> Tuple[int, int]:
        """Model transisi deterministik s' = T(s, a)."""
        dx, dy = self.action_vectors[action]
        return (state[0] + dx, state[1] + dy)

    def is_goal(self, state: Tuple[int, int]) -> bool:
        """Uji kondisi tercapainya titik kumpul aman (Goal Test)."""
        return state in self.goals

    def step_cost(self, state: Tuple[int, int], action: str, next_state: Tuple[int, int]) -> float:
        """Biaya langkah: Jarak fisik (1.0) ditambah penalti kedekatan asap/api."""
        base_cost = 1.0
        penalty = self.hazard_penalties.get(next_state, 0.0)
        return base_cost + penalty


def print_peas_specification() -> None:
    """Menampilkan ringkasan spesifikasi PEAS sistem cerdas ICER-Agent."""
    print("===========================================================================")
    print("SPESIFIKASI PEAS: INTELLIGENT CAMPUS EMERGENCY RESPONSE (ICER-AGENT)")
    print("===========================================================================")
    print("1. Performance Measure (Ukuran Keberhasilan):")
    print("   - Waktu evakuasi rata-rata minimum (detik/orang).")
    print("   - Nol korban jiwa (zero casualties) dan minimalisasi paparan gas beracun.")
    print("   - Kepatuhan 100% aksesibilitas bagi warga berkebutuhan khusus (disabilitas).")
    print("   - Total biaya jalur rute penyelamatan optimal.")
    print("\n2. Environment (Lingkungan Operasi):")
    print("   - Grid fasilitas kampus ITERA (10 x 10 node: gedung perkuliahan, koridor, lapangan).")
    print("   - Status dinamis: rintangan api, koridor roboh, zona terkunci, titik kumpul aman.")
    print("   - Karakteristik: Partially Observable, Stochastic, Dynamic, Multi-Agent.")
    print("\n3. Actuators (Aktuator Tindakan):")
    print("   - Notifikasi push rute evakuasi ke ponsel pintar mahasiswa & dosen.")
    print("   - Sistem pembuka kunci darurat otomatis pada pintu keluar darurat.")
    print("   - Lampu penunjuk arah pintar (smart corridor guidance lights).")
    print("   - Sirene peringatan dan integrasi tim SAR darurat kampus.")
    print("\n4. Sensors (Sensor Persepsi Lingkungan):")
    print("   - Sensor asap optik IoT (optical smoke density in ppm).")
    print("   - Sensor suhu termokopel (deg C).")
    print("   - Sensor kontak status pintu (Open/Closed).")
    print("   - Kamera penghitung kepadatan kerumunan koridor (occupancy counter).")
    print("===========================================================================")


def main() -> None:
    print_peas_specification()
    
    # Inisialisasi masalah pencarian demo
    blocked = {(3, 3), (3, 4), (3, 5), (4, 5), (5, 5)}
    penalties = {(2, 3): 3.0, (2, 4): 3.0, (4, 4): 5.0}  # Zona asap dekat api
    problem = CampusSearchProblem(
        grid_size=(10, 10),
        initial_state=(1, 1),
        goals={(9, 9)},
        blocked_zones=blocked,
        hazard_penalties=penalties
    )
    
    print(f"\nState Awal             : {problem.initial_state}")
    print(f"Titik Kumpul Aman (Goal): {problem.goals}")
    print(f"Zona Terblokir (Bahaya) : {len(problem.blocked_zones)} blok")
    print(f"Aksi Legal dari (1, 1)  : {problem.actions((1, 1))}")
    print("Status Formulasi: Valid & Siap untuk Modul Search Routing.")


if __name__ == "__main__":
    main()
