"""
Chapter 28 - Attention & Transformer Architecture
Script 1: Dot-Product Attention Concept & Query-Key-Value Roles

Mendemonstrasikan konsep dasar mekanisme atensi berbasis produk titik (Dot-Product Attention):
- Membedah peran Query (Q), Key (K), dan Value (V).
- Menghitung matriks skor kesesuaian mentah: S = Q @ K^T.
- Mengagregasikan nilai Value terbobot untuk menghasilkan representasi konteks.
"""

import numpy as np


def naive_dot_product_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray) -> np.ndarray:
    """
    Menghitung unscaled dot-product attention sederhana:
    1. Skor kesesuaian: S = Q @ K^T (Seq_Q x Seq_K)
    2. Bobot probabilitas: W = softmax(S)
    3. Agregasi nilai: Out = W @ V (Seq_Q x Dim_V)
    """
    # 1. Matriks Skor Dot Product
    scores = np.matmul(Q, K.T)

    # 2. Softmax stabil per baris
    exp_scores = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
    weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)

    # 3. Output atensi terbobot
    output = np.matmul(weights, V)
    return output, scores, weights


def main():
    print("=== Chapter 28: Demo Dot-Product Attention & Q-K-V Roles ===")

    # Contoh terhitung sederhana: Sekuens 3 token, dimensi fitur d = 2
    # Token 1: 'bank' (lembaga keuangan), Token 2: 'sungai', Token 3: 'uang'
    tokens = ["bank", "sungai", "uang"]

    Q = np.array([
        [1.0, 0.0],  # Query 'bank'
        [0.0, 1.0],  # Query 'sungai'
        [0.8, 0.2],  # Query 'uang'
    ], dtype=np.float32)

    K = np.array([
        [0.9, 0.1],  # Key 'bank'
        [0.1, 0.9],  # Key 'sungai'
        [0.8, 0.1],  # Key 'uang'
    ], dtype=np.float32)

    V = np.array([
        [10.0, 1.0],  # Value 'bank'
        [1.0, 10.0],  # Value 'sungai'
        [8.0, 2.0],   # Value 'uang'
    ], dtype=np.float32)

    output, scores, weights = naive_dot_product_attention(Q, K, V)

    print("1. Matriks Skor Kesesuaian (Q @ K^T):")
    print(np.round(scores, 3))

    print("\n2. Matriks Bobot Atensi (Softmax Weights):")
    for i, tok in enumerate(tokens):
        w_row = [f"{w:.4f}" for w in weights[i]]
        print(f"  Token '{tok:6s}' memberikan atensi ke {tokens}: {w_row}")

    print("\n3. Vektor Representasi Konteks Luaran Atensi (Out = W @ V):")
    print(np.round(output, 3))

    # Verifikasi assert
    assert np.allclose(np.sum(weights, axis=-1), 1.0), "Total bobot atensi per baris harus tepat 1.0"
    print("\nStatus: dot_product_attention.py berhasil dieksekusi 100%!")


if __name__ == "__main__":
    main()

