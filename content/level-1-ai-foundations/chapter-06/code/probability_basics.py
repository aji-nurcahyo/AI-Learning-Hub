"""Probability Foundations for Artificial Intelligence.

Chapter 06: Probability, Statistics & Linear Algebra for AI
Artificial Intelligence Learning Hub - ITERA

Skrip ini mendemonstrasikan konsep probabilitas dasar dari awal menggunakan Python Standard Library:
1. Ruang Sampel (Sample Space) dan Kejadian (Events)
2. Aturan Komplemen, Gabungan (Union), dan Irisan (Intersection)
3. Probabilitas Bersyarat (Conditional Probability): P(A|B) = P(A ∩ B) / P(B)
4. Uji Independensi Stokastik (Independence Test)
5. Variabel Acak (Random Variable) dan Nilai Harapan (Expected Value)
6. Distribusi Probabilitas Diskrit: Bernoulli dan Binomial
"""

import math
import sys
from typing import Dict, List, Set

sys.stdout.reconfigure(encoding="utf-8")


def probability(event: Set, sample_space: Set) -> float:
    """Menghitung P(E) = |E| / |S| untuk ruang sampel seragam."""
    if not sample_space:
        raise ValueError("Ruang sampel tidak boleh kosong.")
    return len(event) / len(sample_space)


def conditional_probability(event_a: Set, event_b: Set, sample_space: Set) -> float:
    """Menghitung P(A|B) = P(A ∩ B) / P(B) = |A ∩ B| / |B|."""
    intersection = event_a.intersection(event_b)
    p_b = probability(event_b, sample_space)
    if p_b == 0.0:
        raise ZeroDivisionError("P(B) bernilai 0, probabilitas bersyarat tidak terdefinisi.")
    p_intersection = probability(intersection, sample_space)
    return p_intersection / p_b


def is_independent(event_a: Set, event_b: Set, sample_space: Set, tolerance: float = 1e-6) -> bool:
    """Menguji independensi: Apakah P(A ∩ B) == P(A) * P(B)?"""
    p_a = probability(event_a, sample_space)
    p_b = probability(event_b, sample_space)
    p_intersection = probability(event_a.intersection(event_b), sample_space)
    return math.isclose(p_intersection, p_a * p_b, abs_tol=tolerance)


def expected_value(distribution: Dict[float, float]) -> float:
    """Menghitung E[X] = sum(x_i * P(X = x_i))."""
    # Validasi total probabilitas = 1.0
    total_p = sum(distribution.values())
    if not math.isclose(total_p, 1.0, abs_tol=1e-5):
        raise ValueError(f"Total probabilitas harus 1.0, diperoleh {total_p}")
    return sum(x * p for x, p in distribution.items())


def binomial_pmf(k: int, n: int, p: float) -> float:
    """Menghitung probabilitas binomial: P(X=k) = C(n, k) * p^k * (1-p)^(n-k)."""
    comb = math.comb(n, k)
    return comb * (p ** k) * ((1.0 - p) ** (n - k))


def run_probability_demo():
    print("=" * 70)
    print("EKSPERIMEN 2: FONDASI PROBABILITAS UNTUK AI")
    print("=" * 70)

    # 1. Ruang Sampel Eksperimen Transmisi Paket Jaringan AI
    # Representasi 100 paket uji dengan tuple: (id, is_corrupted, is_high_priority)
    # 10 paket korup (10%), 30 paket prioritas tinggi (30%), 3 paket korup & prioritas tinggi (3%)
    sample_space = {f"pkt_{i:03d}" for i in range(100)}
    corrupted_pkts = {f"pkt_{i:03d}" for i in range(10)}  # 10 paket
    high_priority_pkts = {f"pkt_{i:03d}" for i in range(7, 37)}  # 30 paket (irisan pkt 7,8,9 = 3 paket)

    p_corrupt = probability(corrupted_pkts, sample_space)
    p_priority = probability(high_priority_pkts, sample_space)
    intersection = corrupted_pkts.intersection(high_priority_pkts)
    p_intersect = probability(intersection, sample_space)
    union = corrupted_pkts.union(high_priority_pkts)
    p_union = probability(union, sample_space)

    print("--- 1. Aturan Dasar Probabilitas ---")
    print(f"Total Ruang Sampel |S|            : {len(sample_space)} paket")
    print(f"P(Corrupted) [A]                  : {p_corrupt:.4f} (10%)")
    print(f"P(High Priority) [B]              : {p_priority:.4f} (30%)")
    print(f"P(Corrupted ∩ High Priority) [A∩B]: {p_intersect:.4f} (3%)")
    print(f"P(Corrupted ∪ High Priority) [A∪B]: {p_union:.4f} (P(A) + P(B) - P(A∩B) = {p_corrupt + p_priority - p_intersect:.2f})")
    print(f"P(Komplemen / Tidak Korup) [A']   : {1.0 - p_corrupt:.4f} (90%)\n")

    print("--- 2. Probabilitas Bersyarat & Uji Independensi ---")
    p_corrupt_given_priority = conditional_probability(corrupted_pkts, high_priority_pkts, sample_space)
    p_priority_given_corrupt = conditional_probability(high_priority_pkts, corrupted_pkts, sample_space)
    indep = is_independent(corrupted_pkts, high_priority_pkts, sample_space)

    print(f"P(Corrupted | High Priority)      : {p_corrupt_given_priority:.4f} (3 / 30 = 10%)")
    print(f"P(High Priority | Corrupted)      : {p_priority_given_corrupt:.4f} (3 / 10 = 30%)")
    print(f"Apakah Kejadian Independen?       : {indep}")
    print(f"P(A ∩ B) = {p_intersect:.4f} vs P(A)*P(B) = {p_corrupt * p_priority:.4f} -> Tepat Sama!")
    print("Interpretasi AI: Mengetahui paket prioritas tinggi TIDAK mengubah peluang paket tersebut korup.\n")

    print("--- 3. Variabel Acak & Nilai Harapan (Expected Value) ---")
    # Biaya komputasi pemrosesan paket (latency dalam milidetik)
    latency_distribution = {
        5.0: 0.70,   # 70% paket diproses cepat (5 ms)
        20.0: 0.20,  # 20% paket mengalami antrean (20 ms)
        100.0: 0.10  # 10% paket mengalami retransmisi (100 ms)
    }
    exp_lat = expected_value(latency_distribution)
    print("Distribusi Latensi: 5ms (70%), 20ms (20%), 100ms (10%)")
    print(f"Expected Latency E[X]             : {exp_lat:.2f} ms")
    print("Perhitungan: (5*0.7) + (20*0.2) + (100*0.1) = 3.5 + 4.0 + 10.0 = 17.5 ms\n")

    print("--- 4. Distribusi Binomial (Model Kegagalan n Percobaan) ---")
    # Misalkan agen mengirim n=10 paket dengan probabilitas error per paket p=0.05
    n_trials = 10
    p_error = 0.05
    print(f"Percobaan Pengiriman n={n_trials} paket dengan p(error)={p_error}:")
    for k in range(4):
        p_k = binomial_pmf(k, n_trials, p_error)
        print(f"  P(Tepat {k} paket error)       : {p_k:.6f} ({p_k * 100:.3f}%)")
    p_at_least_one = 1.0 - binomial_pmf(0, n_trials, p_error)
    print(f"  P(Minimal 1 paket error)        : {p_at_least_one:.6f} ({p_at_least_one * 100:.3f}%)")
    print("=" * 70)


if __name__ == "__main__":
    run_probability_demo()

