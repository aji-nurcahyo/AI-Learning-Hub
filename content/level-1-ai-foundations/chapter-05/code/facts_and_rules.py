"""
facts_and_rules.py
Struktur Data Representasi Pengetahuan Logika Berbasis Kaidah (Rule-Based Representation).
Mendefinisikan komponen inti Knowledge Base:
- Fact: Proposisi fakta atomik (misal: "link_down", "dns_failing").
- Rule: Kaidah implikasi IF [Premises] THEN [Conclusion].
- KnowledgeBase: Wadah penyimpan fakta awal dan himpunan aturan inferensi.

Kurikulum AI Learning Hub - Institut Teknologi Sumatera
"""

import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from typing import Set, List, Dict, Tuple, Optional
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Fact:
    """Representasi fakta atomik tunggal."""
    name: str

    def __repr__(self) -> str:
        return self.name


@dataclass
class Rule:
    """Representasi aturan logika implikasi IF (premises) THEN (conclusion)."""
    premises: Set[str]
    conclusion: str
    rule_id: Optional[str] = None

    def __repr__(self) -> str:
        prem_str = " AND ".join(sorted(list(self.premises)))
        id_prefix = f"[{self.rule_id}] " if self.rule_id else ""
        return f"{id_prefix}IF ({prem_str}) THEN {self.conclusion}"


class KnowledgeBase:
    """Basis Pengetahuan (Knowledge Base) penyimpan fakta dan aturan inferensi."""

    def __init__(self):
        self.facts: Set[str] = set()
        self.rules: List[Rule] = []

    def tell_fact(self, fact_name: str):
        """Menambahkan sebuah fakta yang telah diverifikasi benar ke dalam KB."""
        self.facts.add(fact_name)

    def tell_rule(self, premises: List[str], conclusion: str, rule_id: Optional[str] = None):
        """Menambahkan sebuah kaidah IF-THEN ke dalam KB."""
        self.rules.append(Rule(premises=set(premises), conclusion=conclusion, rule_id=rule_id))

    def ask_fact(self, fact_name: str) -> bool:
        """Memeriksa apakah sebuah fakta berada dalam basis pengetahuan saat ini."""
        return fact_name in self.facts

    def display(self):
        """Menampilkan isi basis pengetahuan."""
        print("=== KNOWLEDGE BASE STATUS ===")
        print(f"Himpunan Fakta Diketahui ({len(self.facts)}):")
        for f in sorted(list(self.facts)):
            print(f"  * {f}")
        print(f"\nHimpunan Aturan Inferensi ({len(self.rules)}):")
        for r in self.rules:
            print(f"  * {r}")
        print("=" * 45)


def create_network_troubleshooting_kb() -> KnowledgeBase:
    """Membangun contoh Knowledge Base untuk studi kasus Network Troubleshooting."""
    kb = KnowledgeBase()

    # 1. Fakta Observasi Gejala Jaringan
    kb.tell_fact("cable_plugged")
    kb.tell_fact("link_light_off")
    kb.tell_fact("ping_gateway_fails")

    # 2. Himpunan Kaidah Inferensi Jaringan
    kb.tell_rule(["cable_plugged", "link_light_off"], "nic_hardware_fault", "R1")
    kb.tell_rule(["link_light_off"], "interface_down", "R2")
    kb.tell_rule(["interface_down", "ping_gateway_fails"], "local_network_disconnected", "R3")
    kb.tell_rule(["local_network_disconnected"], "internet_inaccessible", "R4")
    kb.tell_rule(["ping_gateway_fails", "dns_query_fails"], "gateway_or_dns_outage", "R5")

    return kb


def main():
    print("=== PENGUJIAN REPRESENTASI FACTS & RULES ===")
    kb = create_network_troubleshooting_kb()
    kb.display()


if __name__ == "__main__":
    main()
