"""
truth_table.py
Pembangkit Tabel Kebenaran (Truth Table Generator) & Entailment Verifier.
Mendukung operator logika proposisional:
- NOT: Negasi (~P)
- AND: Konjungsi (P & Q)
- OR: Disjungsi (P | Q)
- IMPLIES: Implikasi material (P -> Q = ~P | Q)
- IFF: Ekuivalensi bikondisional (P <-> Q)

Menghitung seluruh 2^n model interpretasi dan memvalidasi Logical Entailment (KB |= query).
Implementasi aman tanpa menggunakan fungsi eval() bawaan Python.

Kurikulum AI Learning Hub - Institut Teknologi Sumatera
"""

import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import itertools
from typing import List, Dict, Callable, Any


class PropositionalLogic:
    """Fungsi-fungsi evaluasi operator logika dasar."""

    @staticmethod
    def NOT(p: bool) -> bool:
        return not p

    @staticmethod
    def AND(p: bool, q: bool) -> bool:
        return p and q

    @staticmethod
    def OR(p: bool, q: bool) -> bool:
        return p or q

    @staticmethod
    def IMPLIES(p: bool, q: bool) -> bool:
        # Implikasi material: Salah HANYA JIKA premis Benar tetapi kesimpulan Salah
        return (not p) or q

    @staticmethod
    def IFF(p: bool, q: bool) -> bool:
        return p == q


def generate_truth_table(variables: List[str], formula_fn: Callable[[Dict[str, bool]], bool], formula_name: str):
    """Mencetak tabel kebenaran lengkap untuk fungsi formula_fn."""
    print(f"\nTABEL KEBENARAN: {formula_name}")
    header = " | ".join([f"{v:^5}" for v in variables]) + f" || {formula_name}"
    print("-" * len(header))
    print(header)
    print("-" * len(header))

    n = len(variables)
    is_tautology = True
    is_satisfiable = False

    for values in itertools.product([True, False], repeat=n):
        model = dict(zip(variables, values))
        res = formula_fn(model)

        val_str = " | ".join([f"{str(v):^5}" for v in values])
        print(f"{val_str} || {str(res):^10}")

        if res:
            is_satisfiable = True
        else:
            is_tautology = False

    print("-" * len(header))
    print(f"Sifat Formula: Tautologi = {is_tautology}, Satisfiable = {is_satisfiable}\n")


def check_entailment(variables: List[str], kb_fn: Callable[[Dict[str, bool]], bool],
                     query_fn: Callable[[Dict[str, bool]], bool]) -> bool:
    """
    Model Checking: Memeriksa apakah KB |= Query.
    Definisi: KB |= Query bernilai Benar jika dan hanya jika
    dalam setiap model di mana KB Benar, Query juga Benar (M(KB) subset of M(Query)).
    """
    for values in itertools.product([True, False], repeat=len(variables)):
        model = dict(zip(variables, values))
        if kb_fn(model):
            # Dalam model di mana KB bernilai True, periksa apakah Query juga True
            if not query_fn(model):
                return False  # Ditemukan counterexample (KB=True, Query=False)
    return True


def main():
    print("=== PENGUJIAN TABEL KEBENARAN DAN ENTAILMENT ===")
    
    # 1. Tabel Kebenaran untuk Implikasi Material: P -> Q
    vars_pq = ["P", "Q"]
    formula_implies = lambda m: PropositionalLogic.IMPLIES(m["P"], m["Q"])
    generate_truth_table(vars_pq, formula_implies, "P -> Q")

    # 2. Tabel Kebenaran untuk Hukum De Morgan: NOT(P AND Q) <-> (NOT P OR NOT Q)
    formula_demorgan = lambda m: PropositionalLogic.IFF(
        PropositionalLogic.NOT(PropositionalLogic.AND(m["P"], m["Q"])),
        PropositionalLogic.OR(PropositionalLogic.NOT(m["P"]), PropositionalLogic.NOT(m["Q"]))
    )
    generate_truth_table(vars_pq, formula_demorgan, "NOT(P AND Q) <-> (NOT P OR NOT Q)")

    # 3. Model Checking Entailment:
    # KB: P (Benar), P -> Q (Benar)
    # Query: Q
    # Apakah KB |= Q ? (Prinsip Modus Ponens)
    kb = lambda m: m["P"] and PropositionalLogic.IMPLIES(m["P"], m["Q"])
    query = lambda m: m["Q"]
    entails = check_entailment(vars_pq, kb, query)
    print("Uji Model Checking: KB { P, P -> Q } |= Q (Modus Ponens):", entails)
    assert entails is True, "Modus Ponens harus valid secara semantik!"


if __name__ == "__main__":
    main()
