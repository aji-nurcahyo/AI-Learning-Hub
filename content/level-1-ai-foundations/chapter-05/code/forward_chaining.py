"""
forward_chaining.py
Implementasi Mesin Inferensi Forward Chaining (Data-Driven / Bottom-Up).
Karakteristik:
- Berangkat dari himpunan fakta awal yang diketahui di Knowledge Base.
- Memindai kaidah IF-THEN yang seluruh premisnya telah terpenuhi (rule firing).
- Menambahkan kesimpulan ke basis fakta dan mengulang proses hingga mencapai titik tetap (fixpoint / jenuh) atau query terbukti.
- Menyediakan jejak inferensi (explanation proof trail) untuk akuntabilitas sistem pakar.

Kurikulum AI Learning Hub - Institut Teknologi Sumatera
"""

import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from typing import Set, List, Dict, Tuple, Optional
from facts_and_rules import KnowledgeBase, Rule, create_network_troubleshooting_kb


class ForwardChainingEngine:
    """Mesin inferensi maju berbasis titik tetap (fixpoint)."""

    def __init__(self, kb: KnowledgeBase):
        self.kb = kb
        self.inferred_facts: Set[str] = set(kb.facts)
        # Mencatat riwayat penurunan: fact -> (Rule, [premises])
        self.derivation_history: Dict[str, Tuple[Rule, Set[str]]] = {}

    def run(self, query: Optional[str] = None) -> Tuple[bool, List[str]]:
        """
        Mengeksekusi forward chaining sampai titik jenuh atau query terbukti.
        Mengembalikan (is_query_proven, list_of_new_facts_in_order).
        """
        new_facts_ordered: List[str] = []
        iteration = 1

        print("--- MEMULAI INFERENSI FORWARD CHAINING ---")
        while True:
            newly_derived_in_this_pass = []
            
            for rule in self.kb.rules:
                # Periksa apakah kaidah ini belum menarik kesimpulan dan semua premisnya ada
                if rule.conclusion not in self.inferred_facts:
                    if rule.premises.issubset(self.inferred_facts):
                        newly_derived_in_this_pass.append(rule)

            if not newly_derived_in_this_pass:
                print(f"[Iterasi {iteration}] Titik Tetap (Fixpoint) Tercapai. Tidak ada lagi kaidah baru yang terpicu.")
                break

            print(f"[Iterasi {iteration}] {len(newly_derived_in_this_pass)} kaidah terpicu:")
            for rule in newly_derived_in_this_pass:
                fact = rule.conclusion
                self.inferred_facts.add(fact)
                self.derivation_history[fact] = (rule, set(rule.premises))
                new_facts_ordered.append(fact)
                print(f"  -> Firing {rule} ==> Menyimpulkan: '{fact}'")

                # Early stop jika query telah terbukti
                if query and fact == query:
                    print(f"\n[SUKSES] Query '{query}' berhasil dibuktikan pada iterasi {iteration}!")
                    return True, new_facts_ordered

            iteration += 1

        is_proven = (query in self.inferred_facts) if query else True
        return is_proven, new_facts_ordered

    def explain(self, fact_name: str):
        """Menjelaskan jalur penalaran (How Facility) mengapa sebuah fakta dapat disimpulkan."""
        print(f"\n=== FASILITAS PENJELASAN (HOW) UNTUK: '{fact_name}' ===")
        if fact_name not in self.inferred_facts:
            print(f"Fakta '{fact_name}' TIDAK terbukti dalam basis pengetahuan.")
            return

        if fact_name not in self.derivation_history:
            print(f"Fakta '{fact_name}' adalah FAKTA DASAR (Aksioma Masukan Pengguna).")
            return

        rule, prems = self.derivation_history[fact_name]
        print(f"Fakta '{fact_name}' diturunkan melalui Kaidah {rule}:")
        print(f"  Premis yang mendukung: {list(prems)}")
        for p in prems:
            if p in self.derivation_history:
                sub_rule, _ = self.derivation_history[p]
                print(f"    * '{p}' diturunkan dari Kaidah {sub_rule}")
            else:
                print(f"    * '{p}' berasal dari fakta dasar masukan.")


def main():
    print("=== PENGUJIAN FORWARD CHAINING ENGINE ===")
    kb = create_network_troubleshooting_kb()
    
    engine = ForwardChainingEngine(kb)
    query = "internet_inaccessible"
    
    is_proven, new_facts = engine.run(query=query)
    print(f"\nHasil Pengujian Query '{query}': {'TERBUKTI (TRUE)' if is_proven else 'TIDAK TERBUKTI'}")
    print(f"Fakta-fakta Baru yang Ditarik: {new_facts}")

    # Minta sistem menjelaskan bagaimana ia menyimpulkan internet_inaccessible
    engine.explain("internet_inaccessible")


if __name__ == "__main__":
    main()
