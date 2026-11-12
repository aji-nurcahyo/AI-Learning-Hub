"""Bayes' Theorem Implementation & Base Rate Analysis.

Chapter 06: Probability, Statistics & Linear Algebra for AI
Artificial Intelligence Learning Hub - ITERA

Skrip ini mendemonstrasikan penerapan Teorema Bayes:
P(H | E) = (P(E | H) * P(H)) / P(E)

Di mana:
- P(H)      : Probabilitas Prior (Keyakinan awal terhadap hipotesis H)
- P(E | H)  : Likelihood (Peluang muncul bukti E jika hipotesis H benar)
- P(E)      : Evidensi Total = P(E | H)*P(H) + P(E | ¬H)*P(¬H)
- P(H | E)  : Probabilitas Posterior (Pembaruan keyakinan setelah bukti E diamati)
"""

import sys
from typing import Dict, List

sys.stdout.reconfigure(encoding="utf-8")


def bayes_theorem(
    prior: float,
    true_positive_rate: float,
    false_positive_rate: float
) -> Dict[str, float]:
    """Menghitung probabilitas posterior menggunakan Teorema Bayes.
    
    Args:
        prior: P(H) - Peluang awal kejadian (base rate).
        true_positive_rate: P(E | H) - Sensitivitas (peluang bukti muncul jika H benar).
        false_positive_rate: P(E | ¬H) - Peluang alarm palsu jika H salah.
        
    Returns:
        Dictionary berisi rincian probabilitas prior, likelihood, evidensi total, dan posterior.
    """
    if not (0.0 <= prior <= 1.0):
        raise ValueError("Prior harus berada dalam rentang [0, 1].")
    if not (0.0 <= true_positive_rate <= 1.0):
        raise ValueError("True positive rate harus berada dalam rentang [0, 1].")
    if not (0.0 <= false_positive_rate <= 1.0):
        raise ValueError("False positive rate harus berada dalam rentang [0, 1].")

    prior_neg = 1.0 - prior
    # Hukum Probabilitas Total untuk Penyebut P(E)
    evidence_positive = (true_positive_rate * prior) + (false_positive_rate * prior_neg)

    if evidence_positive == 0.0:
        raise ZeroDivisionError("Evidensi bernilai nol.")

    posterior = (true_positive_rate * prior) / evidence_positive

    return {
        "prior": prior,
        "prior_neg": prior_neg,
        "tpr": true_positive_rate,
        "fpr": false_positive_rate,
        "evidence": evidence_positive,
        "posterior": posterior,
    }


def run_bayes_experiments():
    print("=" * 70)
    print("EKSPERIMEN 3: TEOREMA BAYES & FENOMENA BASE RATE FALLACY")
    print("=" * 70)

    # Studi Kasus: Sensor Deteksi Kebakaran Gedung Pintar (Smart Building)
    # Hipotesis H = Ada Kebakaran (Fire)
    # Bukti E = Sensor Mendeteksi Asap dan Mengaktifkan Alarm (Alarm)
    prior_fire = 0.01          # Base rate kebakaran langka: 1% (0.01)
    sensitivity = 0.95         # Likelihood P(Alarm|Fire) = 95%
    false_alarm = 0.05         # False positive P(Alarm|No Fire) = 5% (debu/uap)

    result = bayes_theorem(prior_fire, sensitivity, false_alarm)

    print("--- 1. Parameter Skenario Sensor Kebakaran ---")
    print(f"Prior P(Fire)                       : {result['prior']:.4f} (1.0%)")
    print(f"Likelihood P(Alarm | Fire)          : {result['tpr']:.4f} (95.0% - Sensitivitas)")
    print(f"False Positive P(Alarm | No Fire)   : {result['fpr']:.4f} (5.0% - Alarm Palsu)")
    print(f"P(No Fire)                          : {result['prior_neg']:.4f} (99.0%)\n")

    print("--- 2. Perhitungan Langkah demi Langkah ---")
    joint_fire = result['tpr'] * result['prior']
    joint_no_fire = result['fpr'] * result['prior_neg']
    print(f"P(Alarm ∩ Fire)    = 0.95 * 0.01    = {joint_fire:.6f}")
    print(f"P(Alarm ∩ No Fire) = 0.05 * 0.99    = {joint_no_fire:.6f}")
    print(f"Evidensi P(Alarm)  = {joint_fire:.6f} + {joint_no_fire:.6f} = {result['evidence']:.6f} (5.9%)\n")

    print(f"Posterior P(Fire | Alarm)           : {result['posterior']:.4f} ({result['posterior'] * 100:.2f}%)")
    print("\n[INSIGHT PENTING AI]:")
    print("Meskipun akurasi sensor 95%, ketika alarm berbunyi peluang kebakaran")
    print("sebenarnya HANYA 16.10%! Mengapa? Karena kasus kebakaran sangat jarang (1%),")
    print("sehingga jumlah alarm palsu (4.95%) jauh lebih banyak daripada alarm sejati (0.95%).")
    print("Ini adalah demonstrasi klasik Base Rate Fallacy dalam evaluasi sistem AI!\n")

    print("--- 3. Analisis Sensitivitas: Pengaruh Nilai Prior terhadap Posterior ---")
    print(f"{'Prior P(Fire)':<15} | {'P(Alarm)':<15} | {'Posterior P(Fire|Alarm)':<25}")
    print("-" * 60)
    for p in [0.001, 0.005, 0.01, 0.05, 0.10, 0.30, 0.50]:
        res = bayes_theorem(p, sensitivity, false_alarm)
        print(f"{p:<15.4f} | {res['evidence']:<15.4f} | {res['posterior'] * 100:>6.2f}%")
    print("=" * 70)


if __name__ == "__main__":
    run_bayes_experiments()

