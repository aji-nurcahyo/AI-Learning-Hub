"""
Chapter 27 - Natural Language Processing Fundamentals
Script 7: Word Embeddings & Dense Vector Lookup

Membandingkan representasi renggang (Sparse One-Hot Encoding) dengan representasi
padat terdistribusi (Dense Word Embeddings), mengimplementasikan Embedding Lookup
dari dasar menggunakan NumPy, serta memvalidasi kesetaraan terhadap PyTorch nn.Embedding.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import List, Dict


class EmbeddingLookupNumPy:
    """Implementasi tabel lookup vektor embedding padat dari dasar berbasis NumPy."""

    def __init__(self, vocab_size: int, embedding_dim: int, padding_idx: int = 0, seed: int = 42):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        
        # Inisialisasi bobot embedding terdistribusi normal standar
        rng = np.random.default_rng(seed)
        self.weights = rng.standard_normal((vocab_size, embedding_dim), dtype=np.float32)
        
        # Nol-kan vektor pada padding index jika didefinisikan
        if padding_idx is not None and 0 <= padding_idx < vocab_size:
            self.weights[padding_idx] = 0.0

    def forward(self, indices: np.ndarray) -> np.ndarray:
        """Operasi lookup: memilih baris matriks sesuai indeks integer token masukan."""
        return self.weights[indices]


def one_hot_encode(indices: List[int], vocab_size: int) -> np.ndarray:
    """Mengonversi daftar indeks integer menjadi matriks sparse one-hot encoding."""
    one_hot = np.zeros((len(indices), vocab_size), dtype=np.float32)
    for i, idx in enumerate(indices):
        if 0 <= idx < vocab_size:
            one_hot[i, idx] = 1.0
    return one_hot


def main():
    print("=== Chapter 27: Demo Word Embeddings & Lookup Operation ===")

    vocab = ["<pad>", "<unk>", "kucing", "anjing", "berlari", "tidur"]
    vocab_size = len(vocab)
    embedding_dim = 4  # Dimensi representasi vektor padat

    print(f"Kosakata Uji ({vocab_size} kata): {vocab}")
    print(f"Dimensi Representasi: One-Hot = {vocab_size} dimensi | Word Embedding = {embedding_dim} dimensi")

    # 1. Representasi One-Hot
    sample_indices = [2, 4, 3, 0]  # "kucing berlari anjing <pad>"
    one_hot_rep = one_hot_encode(sample_indices, vocab_size)
    print("\n1. Representasi Sparse One-Hot (Sangat renggang, banyak nol):")
    print(one_hot_rep)

    # 2. Embedding Lookup Berbasis NumPy Scratch
    emb_numpy = EmbeddingLookupNumPy(vocab_size=vocab_size, embedding_dim=embedding_dim, padding_idx=0, seed=42)
    embedded_vectors_numpy = emb_numpy.forward(np.array(sample_indices))

    print(f"\n2. Matriks Bobot Embedding E ({vocab_size} x {embedding_dim}):")
    print(np.round(emb_numpy.weights, 3))

    print("\n3. Hasil Lookup NumPy untuk Sekuens Input [2, 4, 3, 0]:")
    print(np.round(embedded_vectors_numpy, 3))

    # 3. Verifikasi terhadap PyTorch nn.Embedding
    emb_torch = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim, padding_idx=0)
    # Salin bobot identik untuk pengujian deterministik
    with torch.no_grad():
        emb_torch.weight.copy_(torch.from_numpy(emb_numpy.weights))

    torch_input = torch.tensor(sample_indices, dtype=torch.long)
    embedded_vectors_torch = emb_torch(torch_input).detach().numpy()

    # Periksa deviasi numerik
    max_dev = np.max(np.abs(embedded_vectors_numpy - embedded_vectors_torch))
    print(f"\nDeviasi Numerik Maksimum NumPy vs PyTorch: {max_dev:.2e}")
    assert max_dev < 1e-6, "Operasi embedding lookup NumPy harus identik sempurna dengan PyTorch"
    print("Status: Kesetaraan fungsional Embedding Lookup terbukti 100%!")

    # Verifikasi vektor padding selalu bernilai nol
    pad_vector = emb_numpy.forward(np.array([0]))[0]
    assert np.all(pad_vector == 0.0), "Vektor padding_idx=0 harus bernilai 0.0 pada seluruh dimensi"
    print("Status: Padding vector (indeks 0) terbukti nol di seluruh dimensi.")

    print("\nStatus: word_embeddings.py berhasil dieksekusi 100%!")


if __name__ == "__main__":
    main()

