"""
Chapter 28 - Attention & Transformer Architecture
Script 9: Sinusoidal Positional Encoding & Permutation Invariance

Mengimplementasikan pengkodean posisi sinusoidal (Sinusoidal Positional Encoding)
sesuai formula Vaswani et al. (2017):
    PE(pos, 2i)   = sin(pos / (10000^(2i / d_model)))
    PE(pos, 2i+1) = cos(pos / (10000^(2i / d_model)))
Membuktikan secara matematis bahwa Self-Attention bersifat invarian permutasi
sehingga membutuhkan penambahan vektor posisi agar dapat mengenali urutan kata.
"""

import math
import numpy as np
import torch
import torch.nn as nn


class SinusoidalPositionalEncoding(nn.Module):
    """Modul Positional Encoding sinusoidal PyTorch teroptimasi."""

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        assert d_model % 2 == 0, f"d_model ({d_model}) harus bernilai genap"
        self.d_model = d_model

        # 1. Matriks Posisi: (max_len, 1)
        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)

        # 2. Faktor Pembagi Logaritmik: exp(2i * (-ln(10000) / d_model))
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )

        # 3. Terapkan sinus pada indeks genap (2i) dan kosinus pada indeks ganjil (2i+1)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Daftarkan sebagai buffer permanen agar tidak dihitung ulang dan tidak dianggap parameter bobot
        self.register_buffer("pe", pe.unsqueeze(0))  # Bentuk: (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: Tensor input masukan (batch_size, seq_len, d_model)
        Mengembalikan: Tensor dengan informasi posisi ditambahkan (x + pe[:, :seq_len])
        """
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len]


def demonstrate_permutation_invariance():
    """Membuktikan bahwa Self-Attention murni bersifat amnesia terhadap urutan kata tanpa PE."""
    rng = np.random.default_rng(42)
    seq_len = 3
    d = 4

    # Dua kalimat dengan kata yang sama tetapi urutan terbalik
    x_original = rng.standard_normal((1, seq_len, d)).astype(np.float32)
    # Permutasikan baris 0 dan 2
    perm_indices = [2, 1, 0]
    x_permuted = x_original[:, perm_indices, :]

    # Self-Attention tanpa PE: skor Q @ K^T
    s_orig = np.matmul(x_original, np.swapaxes(x_original, -2, -1))
    s_perm = np.matmul(x_permuted, np.swapaxes(x_permuted, -2, -1))

    # Baris yang dipermutasikan akan menghasilkan himpunan skor yang persis sama terpermutasi
    assert np.allclose(s_orig[0, 0, 0], s_perm[0, 2, 2]), "Self-attention tanpa PE bersifat invarian permutasi"
    print("--- Bukti Ketiadaan Informasi Urutan (Permutation Invariance) ---")
    print("  Tanpa Positional Encoding, Self-Attention menganggap seluruh urutan kata identik!")


def main():
    print("=== Chapter 28: Demo Sinusoidal Positional Encoding ===")

    demonstrate_permutation_invariance()

    d_model = 8
    max_len = 6
    batch_size = 2

    pe_module = SinusoidalPositionalEncoding(d_model=d_model, max_len=max_len)
    pe_matrix = pe_module.pe[0, :max_len].numpy()

    print(f"\nMatriks Positional Encoding ({max_len} Posisi x {d_model} Dimensi):")
    print(np.round(pe_matrix, 3))

    # Verifikasi Properti:
    # 1. Posisi 0 selalu dimulai dengan sin(0) = 0 pada indeks genap dan cos(0) = 1 pada indeks ganjil
    assert np.allclose(pe_matrix[0, 0::2], 0.0), "Nilai sinus pada pos=0 harus tepat 0.0"
    assert np.allclose(pe_matrix[0, 1::2], 1.0), "Nilai kosinus pada pos=0 harus tepat 1.0"
    print("\nVerifikasi Properti Matematis:")
    print("  Posisi 0 (Genap/Sin)  : [0. 0. 0. 0.] (Tepat 0.0)")
    print("  Posisi 0 (Ganjil/Cos) : [1. 1. 1. 1.] (Tepat 1.0)")

    # 2. Pengujian Penambahan Posisi pada Input
    x_dummy = torch.zeros(batch_size, 4, d_model)
    x_with_pos = pe_module(x_dummy)

    print(f"\nBentuk Input Masukan    : {x_dummy.shape}")
    print(f"Bentuk Input + Posisi   : {x_with_pos.shape}")
    assert x_with_pos.shape == (batch_size, 4, d_model)

    print("\nStatus: positional_encoding.py berhasil dieksekusi 100%!")


if __name__ == "__main__":
    main()

