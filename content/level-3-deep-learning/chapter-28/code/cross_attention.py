"""
Chapter 28 - Attention & Transformer Architecture
Script 5: Cross-Attention Mechanism (Encoder-Decoder Interaction)

Mengimplementasikan modul Cross-Attention di mana Query berasal dari sekuens Decoder,
sedangkan Key dan Value diproyeksikan dari sekuens luaran Encoder.
Membuktikan secara komputasional kemampuan menangani panjang sekuens asimetris:
    T_dec != T_enc (misal kalimat terjemahan sumber vs target).
"""

import math
import numpy as np


class CrossAttentionScratch:
    """Modul Cross-Attention berbasis NumPy untuk interaksi Decoder-Encoder."""

    def __init__(self, d_model: int, d_k: int, d_v: int, seed: int = 42):
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v

        rng = np.random.default_rng(seed)
        # Proyeksi Query dari representasi decoder
        self.W_q = rng.standard_normal((d_model, d_k), dtype=np.float32) / math.sqrt(d_model)
        # Proyeksi Key dan Value dari representasi encoder
        self.W_k = rng.standard_normal((d_model, d_k), dtype=np.float32) / math.sqrt(d_model)
        self.W_v = rng.standard_normal((d_model, d_v), dtype=np.float32) / math.sqrt(d_model)
        self.W_o = rng.standard_normal((d_v, d_model), dtype=np.float32) / math.sqrt(d_v)

    def forward(self, x_dec: np.ndarray, x_enc: np.ndarray, mask: np.ndarray = None):
        """
        x_dec: Sekuens masukan dari decoder (batch_size, seq_len_dec, d_model)
        x_enc: Sekuens luaran dari encoder (batch_size, seq_len_enc, d_model)
        Mengembalikan:
          - output: Representasi konteks decoder (batch_size, seq_len_dec, d_model)
          - attention_weights: Matriks atensi (batch_size, seq_len_dec, seq_len_enc)
        """
        # 1. Proyeksi Linier Asimetris
        Q = np.matmul(x_dec, self.W_q)  # (batch_size, seq_len_dec, d_k)
        K = np.matmul(x_enc, self.W_k)  # (batch_size, seq_len_enc, d_k)
        V = np.matmul(x_enc, self.W_v)  # (batch_size, seq_len_enc, d_v)

        # 2. Skor Produk Titik Berskala: (batch_size, seq_len_dec, seq_len_enc)
        scores = np.matmul(Q, np.swapaxes(K, -2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            scores = np.where(mask, scores, -1e9)

        # 3. Softmax pada dimensi panjang sekuens encoder (sumbu terakhir)
        scores_max = np.max(scores, axis=-1, keepdims=True)
        exp_s = np.exp(scores - scores_max)
        attention_weights = exp_s / np.sum(exp_s, axis=-1, keepdims=True)

        # 4. Agregasi Vektor Value Encoder
        context = np.matmul(attention_weights, V)  # (batch_size, seq_len_dec, d_v)

        # 5. Proyeksi Luaran Akhir
        output = np.matmul(context, self.W_o)  # (batch_size, seq_len_dec, d_model)
        return output, attention_weights


def main():
    print("=== Chapter 28: Demo Cross-Attention (Decoder Queries Encoder) ===")

    batch_size = 2
    seq_len_dec = 3  # Misal 3 token target bahasa Indonesia: ['selamat', 'pagi', 'kawan']
    seq_len_enc = 4  # Misal 4 token sumber bahasa Inggris  : ['good', 'morning', 'my', 'friend']
    d_model = 8
    d_k = 8
    d_v = 8

    rng = np.random.default_rng(202)
    x_dec = rng.standard_normal((batch_size, seq_len_dec, d_model), dtype=np.float32)
    x_enc = rng.standard_normal((batch_size, seq_len_enc, d_model), dtype=np.float32)

    cross_attn = CrossAttentionScratch(d_model=d_model, d_k=d_k, d_v=d_v, seed=42)
    output, weights = cross_attn.forward(x_dec, x_enc)

    print(f"Bentuk Input Decoder x_dec : {x_dec.shape} (Batch x Seq_Dec x d_model)")
    print(f"Bentuk Input Encoder x_enc : {x_enc.shape} (Batch x Seq_Enc x d_model)")
    print(f"Bentuk Matriks Atensi W    : {weights.shape} (Batch x Seq_Dec x Seq_Enc)")
    print(f"Bentuk Output Terkonteks   : {output.shape} (Batch x Seq_Dec x d_model)")

    print("\nVisualisasi Distribusi Atensi Tiap Token Target ke Token Sumber (Batch 0):")
    tokens_target = ["selamat", "pagi", "kawan"]
    tokens_sumber = ["good", "morning", "my", "friend"]
    
    print(f"{'Target':<10} | {'Atensi ke Sumber ' + str(tokens_sumber)}")
    print("-" * 55)
    for i, t_tgt in enumerate(tokens_target):
        row_str = " | ".join(f"{weights[0, i, j]:.4f}" for j in range(len(tokens_sumber)))
        print(f"{t_tgt:<10} | [{row_str}]")

    # Verifikasi assert
    assert output.shape == (batch_size, seq_len_dec, d_model), "Bentuk output harus mengikuti dimensi decoder"
    assert weights.shape == (batch_size, seq_len_dec, seq_len_enc), "Bentuk matriks atensi harus (Seq_Dec, Seq_Enc)"
    assert np.allclose(np.sum(weights, axis=-1), 1.0), "Total bobot atensi per token target harus 1.0"

    print("\nStatus: cross_attention.py berhasil dieksekusi 100%!")


if __name__ == "__main__":
    main()

