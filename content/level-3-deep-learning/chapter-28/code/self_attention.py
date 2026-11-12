"""
Chapter 28 - Attention & Transformer Architecture
Script 4: Self-Attention Mechanism with Learned Projections

Mengimplementasikan modul Self-Attention mandiri dari dasar:
- Memproyeksikan matriks input X ke ruang Query, Key, dan Value menggunakan bobot terpelajari:
    Q = X @ W_Q,  K = X @ W_K,  V = X @ W_V
- Menghitung interaksi antar-token (token-to-token contextual relationships).
- Memproyeksikan luaran kembali ke dimensi model via matriks proyeksi W_O.
"""

import math
import numpy as np


class SelfAttentionScratch:
    """Modul Self-Attention tunggal berbasis NumPy dengan proyeksi linier."""

    def __init__(self, d_model: int, d_k: int, d_v: int, seed: int = 42):
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v

        rng = np.random.default_rng(seed)
        # Matriks proyeksi linier terpelajari
        self.W_q = rng.standard_normal((d_model, d_k), dtype=np.float32) / math.sqrt(d_model)
        self.W_k = rng.standard_normal((d_model, d_k), dtype=np.float32) / math.sqrt(d_model)
        self.W_v = rng.standard_normal((d_model, d_v), dtype=np.float32) / math.sqrt(d_model)
        self.W_o = rng.standard_normal((d_v, d_model), dtype=np.float32) / math.sqrt(d_v)

    def forward(self, X: np.ndarray, mask: np.ndarray = None):
        """
        X: Tensor representasi sekuens masukan (batch_size, seq_len, d_model)
        Mengembalikan:
          - output: Representasi kontekstual akhir (batch_size, seq_len, d_model)
          - attention_weights: Matriks bobot atensi (batch_size, seq_len, seq_len)
        """
        # 1. Proyeksi Linier ke ruang Query, Key, Value
        Q = np.matmul(X, self.W_q)  # (batch_size, seq_len, d_k)
        K = np.matmul(X, self.W_k)  # (batch_size, seq_len, d_k)
        V = np.matmul(X, self.W_v)  # (batch_size, seq_len, d_v)

        # 2. Scaled Dot-Product
        scores = np.matmul(Q, np.swapaxes(K, -2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            scores = np.where(mask, scores, -1e9)

        # 3. Softmax
        scores_max = np.max(scores, axis=-1, keepdims=True)
        exp_s = np.exp(scores - scores_max)
        attention_weights = exp_s / np.sum(exp_s, axis=-1, keepdims=True)

        # 4. Agregasi Value
        context = np.matmul(attention_weights, V)  # (batch_size, seq_len, d_v)

        # 5. Proyeksi Akhir kembali ke d_model
        output = np.matmul(context, self.W_o)  # (batch_size, seq_len, d_model)
        return output, attention_weights


def main():
    print("=== Chapter 28: Demo Self-Attention with Learned Linear Projections ===")

    batch_size = 2
    seq_len = 3  # Misal 3 token: ['model', 'bahasa', 'cerdas']
    d_model = 4  # Dimensi embedding kata
    d_k = 4      # Dimensi proyeksi key/query
    d_v = 4      # Dimensi proyeksi value

    # Sekuens input representasi kata
    rng = np.random.default_rng(101)
    X = rng.standard_normal((batch_size, seq_len, d_model), dtype=np.float32)

    self_attn = SelfAttentionScratch(d_model=d_model, d_k=d_k, d_v=d_v, seed=42)
    out, attn_weights = self_attn.forward(X)

    print(f"Dimensi Input Tensor X  : {X.shape} (Batch x SeqLen x d_model)")
    print(f"Dimensi Output Tensor   : {out.shape} (Preservasi bentuk identik input)")
    print(f"Dimensi Matriks Atensi  : {attn_weights.shape} (Batch x SeqLen x SeqLen)")

    print("\nContoh Matriks Atensi Interaksi Antar-Token (Batch 0):")
    tokens = ["model", "bahasa", "cerdas"]
    for i, t_q in enumerate(tokens):
        row_str = " | ".join(f"{attn_weights[0, i, j]:.4f}" for j in range(len(tokens)))
        print(f"  Token '{t_q:6s}' -> [{row_str}]")

    assert out.shape == X.shape, "Dimensi luaran Self-Attention harus sesuai dimensi input d_model"
    assert np.allclose(np.sum(attn_weights, axis=-1), 1.0), "Bobot atensi per baris harus berjumlah 1.0"

    print("\nStatus: self_attention.py berhasil dieksekusi 100%!")


if __name__ == "__main__":
    main()

