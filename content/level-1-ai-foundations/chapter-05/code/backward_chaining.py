"""
backward_chaining.py
Implementasi Mesin Inferensi Backward Chaining (Goal-Driven / Top-Down).
Karakteristik:
- Berangkat dari hipotesis/tujuan (Goal / Query) yang ingin dibuktikan.
- Mencari kaidah-kaidah yang menghasilkan kesimpulan sesuai tujuan.
- Menjadikan premis kaidah tersebut sebagai sub-tujuan (sub-goals) baru secara rekursif hingga mencapai fakta dasar.
- Mencegah loop siklik tak berhingga dengan memelihara 'active_goals_stack'.
- Menerapkan Closed-World Assumption (CWA) secara eksplisit jika suatu fakta tidak dapat dibuktikan.

Kurikulum AI Learning Hub - Institut Teknologi Sumatera
"""

import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from typing import Set, List, Dict, Optional, Tuple
from facts_and_rules import KnowledgeBase, Rule, create_network_troubleshooting_kb


class BackwardChainingEngine:
    """Mesin inferensi mundur berbasis pembuktian sub-tujuan rekursif."""

    def __init__(self, kb: KnowledgeBase):
        self.kb = kb
        # Menyimpan cache tujuan yang telah terbukti benar
        self.proven_goals: Set[str] = set(kb.facts)
        # Menyimpan tujuan yang gagal dibuktikan (untuk mencegah re-evaluasi)
        self.failed_goals: Set[str] = set()
        # Pohon bukti penalaran
        self.proof_tree: Dict[str, Tuple[Rule, List[str]]] = {}

    def prove(self, goal: str, depth: int = 0, active_stack: Optional[Set[str]] = None) -> bool:
        """
        Membuktikan apakah goal bernilai Benar berdasarkan fakta dan aturan di KB.
        Mengembalikan True jika terbukti, False jika gagal.
        """
        if active_stack is None:
            active_stack = set()

        indent = "  " * depth
        print(f"{indent}[?] Menguji Sub-Goal: '{goal}'")

        # 1. Kasus Basis: Goal sudah ada di fakta dasar atau telah terbukti sebelumnya
        if goal in self.proven_goals:
            print(f"{indent}  => SUKSES: '{goal}' adalah FAKTA DIKETAHUI.")
            return True

        if goal in self.failed_goals:
            print(f"{indent}  => GAGAL: '{goal}' sebelumnya telah terbukti tidak dapat dicapai.")
            return False

        # 2. Deteksi Siklus (Cycle Detection): Mencegah loop rekursif tak berhingga
        if goal in active_stack:
            print(f"{indent}  => PERINGATAN: Siklus terdeteksi pada '{goal}'! Memotong cabang rekursi.")
            return False

        active_stack.add(goal)

        # 3. Cari seluruh kaidah yang konklusinya adalah goal ini
        candidate_rules = [r for r in self.kb.rules if r.conclusion == goal]
        if not candidate_rules:
            print(f"{indent}  => GAGAL: Tidak ada kaidah di KB yang dapat menyimpulkan '{goal}'. (Closed-World Assumption: Dianggap False)")
            active_stack.remove(goal)
            self.failed_goals.add(goal)
            return False

        # 4. Uji setiap kaidah kandidat
        for rule in candidate_rules:
            print(f"{indent}  [!] Mencoba Kaidah {rule}:")
            all_premises_satisfied = True
            supporting_premises = []

            for premise in sorted(list(rule.premises)):
                # Buktikan premis secara rekursif
                premise_proven = self.prove(premise, depth + 1, active_stack)
                if not premise_proven:
                    all_premises_satisfied = False
                    break
                supporting_premises.append(premise)

            if all_premises_satisfied:
                print(f"{indent}  => SUKSES: Kaidah {rule} BERHASIL MEMBUKTIKAN '{goal}'!")
                self.proven_goals.add(goal)
                self.proof_tree[goal] = (rule, supporting_premises)
                active_stack.remove(goal)
                return True

        # Jika seluruh kaidah kandidat gagal memenuhi seluruh premisnya
        print(f"{indent}  => GAGAL: Seluruh kaidah untuk '{goal}' tidak dapat dipenuhi.")
        active_stack.remove(goal)
        self.failed_goals.add(goal)
        return False


def test_cyclic_dependency():
    """Pengujian penanganan ketergantungan siklik pada backward chaining."""
    print("\n--- UJI PENANGANAN KETERGANTUNGAN SIKLIK (CYCLIC DEPENDENCIES) ---")
    cyclic_kb = KnowledgeBase()
    # Aturan siklik A -> B dan B -> A
    cyclic_kb.tell_rule(["B"], "A", "R_A")
    cyclic_kb.tell_rule(["A"], "B", "R_B")
    
    engine = BackwardChainingEngine(cyclic_kb)
    result = engine.prove("A")
    print(f"Hasil Pembuktian Siklik: {'BERHASIL' if result else 'GAGAL DENGAN AMAN (BEBAS INFINITE LOOP)'}")
    assert result is False, "Siklus tanpa fakta basis harus gagal dengan aman!"


def main():
    print("=== PENGUJIAN BACKWARD CHAINING ENGINE ===")
    kb = create_network_troubleshooting_kb()
    
    engine = BackwardChainingEngine(kb)
    target_hypothesis = "internet_inaccessible"
    
    print(f"\nMembuktikan Hipotesis Utama: '{target_hypothesis}'")
    is_success = engine.prove(target_hypothesis)
    
    print(f"\nHasil Akhir Pembuktian: {'HIPOTESIS TERBUKTI (TRUE)' if is_success else 'HIPOTESIS TIDAK TERBUKTI (FALSE)'}")
    assert is_success is True, "Hipotesis internet_inaccessible harus terbukti dari fakta dasar jaringan!"

    # Uji siklik
    test_cyclic_dependency()


if __name__ == "__main__":
    main()
