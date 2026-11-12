"""
modus_ponens.py
Implementasi Aturan Inferensi Deduktif Modus Ponens dan Generalized Modus Ponens.
Bentuk Baku:
  Premis 1 : P
  Premis 2 : P -> Q
  -----------------
  Konklusi : Q

Bentuk Jamak (Generalized Modus Ponens untuk Aturan IF-THEN):
  Premis 1 : P1, P2, ..., Pk
  Premis 2 : (P1 AND P2 AND ... AND Pk) -> Q
  -------------------------------------------
  Konklusi : Q

Kurikulum AI Learning Hub - Institut Teknologi Sumatera
"""

import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from typing import Set, List, Optional, Tuple
from facts_and_rules import KnowledgeBase, Rule


class ModusPonensEngine:
    """Mesin inferensi satu langkah berbasis Modus Ponens."""

    @staticmethod
    def apply_single_step(kb: KnowledgeBase) -> List[Tuple[Rule, str]]:
        """
        Memindai seluruh kaidah dalam KB. Jika seluruh premis suatu kaidah
        telah terpenuhi dalam himpunan fakta dan kesimpulannya belum tercatat,
        tarik kesimpulan baru secara deduktif.
        Mengembalikan daftar pasangan (Rule yang terpicu, Fakta baru yang ditarik).
        """
        new_inferences = []
        for rule in kb.rules:
            # Periksa apakah semua premis terpenuhi dalam himpunan fakta
            if rule.premises.issubset(kb.facts):
                if rule.conclusion not in kb.facts:
                    new_inferences.append((rule, rule.conclusion))
        return new_inferences


def main():
    print("=== DEMONSTRASI ATURAN INFERENSI MODUS PONENS ===")
    
    kb = KnowledgeBase()
    # Kasus: Troubleshooting Antarmuka Jaringan
    # Fakta Awal: Kabel terpasang, Lampu indikator mati
    kb.tell_fact("cable_plugged")
    kb.tell_fact("link_light_off")

    # Kaidah Logika:
    # IF (cable_plugged AND link_light_off) THEN nic_hardware_fault
    # IF (link_light_off) THEN interface_down
    kb.tell_rule(["cable_plugged", "link_light_off"], "nic_hardware_fault", "R1")
    kb.tell_rule(["link_light_off"], "interface_down", "R2")
    kb.tell_rule(["interface_down", "dns_failing"], "gateway_unreachable", "R3")

    kb.display()

    print("\nMenjalankan Inferensi Modus Ponens Satu Langkah:")
    inferences = ModusPonensEngine.apply_single_step(kb)
    
    for rule, new_fact in inferences:
        print(f"  [FIRING] Kaidah {rule} TERPENUHI!")
        print(f"           -> Menyimpulkan fakta baru: '{new_fact}'")
        kb.tell_fact(new_fact)

    print("\nStatus Fakta Baru Setelah Modus Ponens:")
    for f in sorted(list(kb.facts)):
        print(f"  + {f}")

    print("\nCatatan: Kaidah R3 TIDAK terpicu karena premis 'dns_failing' belum terpenuhi (soundness terjaga).")


if __name__ == "__main__":
    main()
