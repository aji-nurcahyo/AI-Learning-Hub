"""
Chapter 28 - Attention & Transformer Architecture
Script 7: Multi-Head Attention (MHA) from Scratch in NumPy

Mengimplementasikan modul Multi-Head Attention lengkap dari dasar berbasis NumPy:
- Pemecahan dimensi model menjadi h kepala atensi independen: d_k = d_model / h.
- Komputasi atensi paralel melintasi seluruh head via tensor 4D: (Batch, Heads, SeqLen, d_k).
- Penggabungan (concatenation) dan proyeksi akhir via matriks W_O.
- Menghitung jumlah parameter bobot: 4 * d_model^2 (+ bias opsional).
"""

import math
import numpy as np
from typing import Tuple, Optional


class MultiHeadAttentionNumPy:
    """Implementasi Multi-Head Attention murni dari dasar berbasis manipulasi tensor NumPy."""

    def __init__(self, d_model: int, num_heads: int, seed: int = 42):
        assert d_model % num_heads == 0, f"d_model ({d_model}) harus habis dibagi num_heads ({num_heads})"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        rng = np.random.default_rng(seed)
        # Matriks proyeksi linier: W_q, W_k, W_v, W_o masing-masing berukuran (d_model, d_model)
        scale = 1.0 / math.sqrt(d_model)
        self.W_q = rng.standard_normal((d_model, d_model), dtype=np.float32) * scale
        self.W_k = rng.standard_normal((d_model, d_model), dtype=np.float32) * scale
        self.W_v = rng.standard_normal((d_model, d_model), dtype=np.float32) * scale
        self.W_o = rng.standard_normal((d_model, d_model), dtype=np.float32) * scale

    def _split_heads(self, x: np.ndarray) -> np.ndarray:
        """
        Mengubah bentuk dari (batch, seq_len, d_model)
        menjadi (batch, num_heads, seq_len, d_k) untuk komputasi atensi paralel.
        """
        batch_size, seq_len, _ = x.shape
        # 1. Reshape ke (batch, seq_len, num_heads, d_k)
        x_reshaped = x.reshape(batch_size, seq_len, self.num_heads, self.d_k)
        # 2. Swapaxes ke (batch, num_heads, seq_len, d_k)
        return np.swapaxes(x_reshaped, 1, 2)

    def _combine_heads(self, x: np.ndarray) -> np.ndarray:
        """
        Mengembalikan bentuk dari (batch, num_heads, seq_len, d_k)
        ke bentuk gabungan (batch, seq_len, d_model).
        """
        batch_size, num_heads, seq_len, d_k = x.shape
        # 1. Swapaxes ke (batch, seq_len, num_heads, d_k)
        x_swapped = np.swapaxes(x, 1, 2)
        # 2. Reshape ke (batch, seq_len, d_model)
        return x_swapped.reshape(batch_size, seq_len, self.d_model)

    def forward(
        self,
        Q: np.ndarray,
        K: np.ndarray,
        V: np.ndarray,
        mask: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Q: (batch_size, seq_len_q, d_model)
        K: (batch_size, seq_len_k, d_model)
        V: (batch_size, seq_len_k, d_model)
        mask: (batch_size, 1, seq_len_q, seq_len_k) atau bentuk yang kompatibel
        """
        batch_size = Q.shape[0]

        # 1. Proyeksi Linier Global: (batch, seq, d_model)
        q_proj = np.matmul(Q, self.W_q)
        k_proj = np.matmul(K, self.W_k)
        v_proj = np.matmul(V, self.W_v)

        # 2. Pemisahan Multi-Head: (batch, heads, seq, d_k)
        q_heads = self._split_heads(q_proj)
        k_heads = self._split_heads(k_proj)
        v_heads = self._split_heads(v_proj)

        # 3. Scaled Dot-Product Attention Paralel pada seluruh head:
        # scores: (batch, heads, seq_q, seq_k)
        scores = np.matmul(q_heads, np.swapaxes(k_heads, -2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            scores = np.where(mask, scores, -1e9)

        # Softmax stabil
        scores_max = np.max(scores, axis=-1, keepdims=True)
        exp_s = np.exp(scores - scores_max)
        attn_weights = exp_s / np.sum(exp_s, axis=-1, keepdims=True)

        # Context per head: (batch, heads, seq_q, d_k)
        context_heads = np.matmul(attn_weights, v_heads)

        # 4. Penggabungan seluruh head (Concatenation): (batch, seq_q, d_model)
        concat_context = self._combine_heads(context_heads)

        # 5. Proyeksi Luaran Akhir via W_O
        output = np.matmul(concat_context, self.W_o)
        return output, attn_weights


def main():
    print("=== Chapter 28: Demo Multi-Head Attention (NumPy Scratch) ===")

    batch_size = 2
    seq_len = 4
    d_model = 8
    num_heads = 2
    d_k = d_model // num_heads

    print(f"Konfigurasi Multi-Head Attention:")
    print(f"  d_model = {d_model}, num_heads = {num_heads}, d_k per head = {d_k}")
    print(f"  Jumlah Parameter Bobot (4 x d_model^2) = {4 * (d_model ** 2)} parameter")

    rng = np.random.default_rng(42)
    X = rng.standard_normal((batch_size, seq_len, d_model), dtype=np.float32)

    mha = MultiHeadAttentionNumPy(d_model=d_model, num_heads=num_heads, seed=42)
    out, weights = mha.forward(Q=X, K=X, V=X)

    print(f"\nBentuk Tensor Input X         : {X.shape} (Batch x SeqLen x d_model)")
    print(f"Bentuk Matriks Atensi Weights : {weights.shape} (Batch x Heads x SeqLen x SeqLen)")
    print(f"Bentuk Tensor Output Akhir    : {out.shape} (Batch x SeqLen x d_model)")

    # Periksa bobot atensi per head
    print("\nVisualisasi Matriks Atensi Head 0 (Batch 0):")
    print(np.round(weights[0, 0], 4))
    print("\nVisualisasi Matriks Atensi Head 1 (Batch 0):")
    print(np.round(weights[0, 1], 4))

    # Buktikan bahwa setiap head memiliki pola atensi yang berbeda (spesialisasi representasi)
    diff_heads = np.max(np.abs(weights[0, 0] - weights[0, 1]))
    print(f"\nPerbedaan Maksimum Antar-Head: {diff_heads:.4f}")
    assert diff_heads > 0.05, "Masing-masing head harus mempelajari representasi atensi yang berbeda!"
    assert out.shape == X.shape, "Dimensi luaran MHA harus sesuai dengan input sekuens"

    print("\nStatus: multi_head_attention_numpy.py berhasil dieksekusi 100%!")


if __name__ == "__main__":
    main()

