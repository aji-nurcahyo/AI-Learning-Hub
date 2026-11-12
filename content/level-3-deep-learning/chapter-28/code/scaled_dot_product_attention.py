"""
Chapter 28 - Attention & Transformer Architecture
Script 2: Scaled Dot-Product Attention & Variance Normalization

Mengimplementasikan formula utama Scaled Dot-Product Attention:
    Attention(Q, K, V) = softmax( (Q @ K^T) / sqrt(d_k) ) @ V
Membuktikan secara teoritis dan empiris mengapa pembagian dengan sqrt(d_k)
diperlukan untuk menormalkan variansi nilai skor agar gradien Softmax tidak lenyap (vanish).
"""

import math
import numpy as np
from typing import Tuple, Optional


def scaled_dot_product_attention(
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
    mask: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Q: (..., Seq_Q, d_k)
    K: (..., Seq_K, d_k)
    V: (..., Seq_K, d_v)
    mask: Matriks biner atau boolean dengan bentuk kompatibel broadcast (..., Seq_Q, Seq_K).
          Posisi dengan True / 1 dipertahankan, posisi dengan False / 0 diisi -1e9.
    """
    d_k = Q.shape[-1]
    scaling_factor = math.sqrt(d_k)

    # 1. Matriks Skor Produk Titik Berskala: S = (Q @ K^T) / sqrt(d_k)
    # Gunakan np.matmul untuk mendukung batch dimensions
    scores = np.matmul(Q, np.swapaxes(K, -2, -1)) / scaling_factor

    # 2. Aplikasi Masking jika tersedia
    if mask is not None:
        # Ganti posisi yang dimask (False/0) dengan bilangan negatif sangat besar (-1e9)
        # sehingga setelah softmax nilainya menjadi 0.0
        scores = np.where(mask, scores, -1e9)

    # 3. Softmax Numerik Stabil pada Dimensi Terakhir (Key Sequence)
    max_scores = np.max(scores, axis=-1, keepdims=True)
    exp_scores = np.exp(scores - max_scores)
    # Tangani baris yang seluruhnya dimask agar tidak terjadi pembagian 0/0
    sum_exp = np.sum(exp_scores, axis=-1, keepdims=True)
    sum_exp = np.where(sum_exp == 0.0, 1.0, sum_exp)
    attention_weights = exp_scores / sum_exp

    # 4. Agregasi Vektor Value Terbobot
    output = np.matmul(attention_weights, V)
    return output, attention_weights


def demonstrate_variance_scaling():
    """Mendemonstrasikan secara empiris mengapa variansi membesar seiring d_k."""
    rng = np.random.default_rng(42)
    d_k = 64
    num_samples = 10000

    # Dua vektor acak standar N(0, 1)
    q = rng.standard_normal((num_samples, d_k))
    k = rng.standard_normal((num_samples, d_k))

    dot_unscaled = np.sum(q * k, axis=-1)
    dot_scaled = dot_unscaled / math.sqrt(d_k)

    var_unscaled = np.var(dot_unscaled)
    var_scaled = np.var(dot_scaled)

    print("--- Analisis Variansi Skor Produk Titik (d_k = 64) ---")
    print(f"  Variansi Unscaled (Q . K)          : {var_unscaled:6.2f} (Teori: d_k = {d_k})")
    print(f"  Variansi Scaled   (Q . K / sqrt(d)): {var_scaled:6.2f} (Teori: 1.00)")
    assert abs(var_scaled - 1.0) < 0.1, "Variansi terstandarisasi harus mendekati 1.0"
    print("  -> Pembagian dengan sqrt(d_k) terbukti menjaga variansi skor pada skala stabil O(1)!")


def main():
    print("=== Chapter 28: Demo Scaled Dot-Product Attention ===")

    demonstrate_variance_scaling()

    # Worked Example: 2 Query tokens, 3 Key/Value tokens, d_k = 4, d_v = 4
    Q = np.array([
        [1.0, 0.0, 1.0, 0.0],
        [0.0, 2.0, 0.0, 1.0]
    ], dtype=np.float32)

    K = np.array([
        [1.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 1.0],
        [1.0, 1.0, 0.0, 0.0]
    ], dtype=np.float32)

    V = np.array([
        [1.0, 2.0, 3.0, 4.0],
        [5.0, 6.0, 7.0, 8.0],
        [9.0, 10.0, 11.0, 12.0]
    ], dtype=np.float32)

    output, weights = scaled_dot_product_attention(Q, K, V)

    print("\nWorked Example Matriks Input:")
    print(f"  Bentuk Q: {Q.shape}, Bentuk K: {K.shape}, Bentuk V: {V.shape}")
    print("\nBobot Atensi Tervalidasi (Softmax Weights):")
    print(np.round(weights, 4))
    print("\nLuaran Konteks Atensi (Attention Output):")
    print(np.round(output, 4))

    assert output.shape == (2, 4), "Bentuk output harus (Seq_Q, d_v)"
    assert np.allclose(np.sum(weights, axis=-1), 1.0), "Jumlah bobot per baris harus tepat 1.0"

    print("\nStatus: scaled_dot_product_attention.py berhasil dieksekusi 100%!")


if __name__ == "__main__":
    main()

