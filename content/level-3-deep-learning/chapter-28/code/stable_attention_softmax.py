"""
Chapter 28 - Attention & Transformer Architecture
Script 3: Numerical Stability in Attention Softmax

Menganalisis dan mengimplementasikan trik pergeseran nilai maksimum (Shift-Max Trick)
untuk mencegah masalah luapan numerik (Floating-Point Overflow / NaN) pada Softmax:
    softmax(z_i) = exp(z_i - max(z)) / sum(exp(z_j - max(z)))
Membuktikan secara aljabar dan komputasional bahwa pergeseran konstanta tidak mengubah
nilai probabilitas akhir, serta membandingkan perilakunya pada logit bernilai ekstrim.
"""

import numpy as np


def naive_softmax(z: np.ndarray, axis: int = -1) -> np.ndarray:
    """Implementasi naif yang rentan mengalami overflow jika z > 88.72 pada float32."""
    exp_z = np.exp(z)
    return exp_z / np.sum(exp_z, axis=axis, keepdims=True)


def numerically_stable_softmax(z: np.ndarray, axis: int = -1) -> np.ndarray:
    """Implementasi stabil dengan mengurangkan nilai maksimum lokal per baris."""
    z_max = np.max(z, axis=axis, keepdims=True)
    exp_shifted = np.exp(z - z_max)
    sum_exp = np.sum(exp_shifted, axis=axis, keepdims=True)
    return exp_shifted / sum_exp


def main():
    print("=== Chapter 28: Demo Numerically Stable Softmax ===")

    # 1. Kasus Logit Normal
    logits_normal = np.array([2.0, 1.0, 0.1], dtype=np.float32)
    p_naive = naive_softmax(logits_normal)
    p_stable = numerically_stable_softmax(logits_normal)

    print("1. Kasus Logit Normal [2.0, 1.0, 0.1]:")
    print(f"  Naive Softmax  : {np.round(p_naive, 4)}")
    print(f"  Stable Softmax : {np.round(p_stable, 4)}")
    assert np.allclose(p_naive, p_stable), "Pada nilai normal, kedua metode harus menghasilkan output identik"

    # 2. Kasus Logit Besar / Ekstrim (Rentang Skor Atensi Tanpa Scaling)
    # Pada float32, batas maksimum representasi adalah exp(88.72) ~ 3.4e38
    logits_large = np.array([1000.0, 1005.0, 995.0], dtype=np.float32)

    print("\n2. Kasus Logit Ekstrim [1000.0, 1005.0, 995.0] (Overflow Test):")
    
    # Nonaktifkan runtime warning numpy sesaat untuk demonstrasi overflow terukur
    with np.errstate(over="ignore", invalid="ignore"):
        p_large_naive = naive_softmax(logits_large)

    p_large_stable = numerically_stable_softmax(logits_large)

    print(f"  Naive Softmax (Mengalami Overflow) : {p_large_naive} -> Menghasilkan NaN / Inf!")
    print(f"  Stable Softmax (Presisi Terjaga)   : {np.round(p_large_stable, 6)}")

    assert np.isnan(p_large_naive).any(), "Naive softmax harus mengalami NaN akibat overflow"
    assert not np.isnan(p_large_stable).any(), "Stable softmax harus bebas NaN"
    assert np.isclose(np.sum(p_large_stable), 1.0), "Total probabilitas stable softmax harus 1.0"

    print("\n[Pembuktian Aljabar]]:")
    print("  exp(z_i - c) / sum(exp(z_j - c)) = (exp(z_i) * exp(-c)) / (sum(exp(z_j)) * exp(-c))")
    print("                                   = exp(z_i) / sum(exp(z_j))")
    print("  -> Konstanta c = max(z) tereliminasi secara matematis tanpa mendistorsi rasio probabilitas!")

    print("\nStatus: stable_attention_softmax.py berhasil dieksekusi 100%!")


if __name__ == "__main__":
    main()

