"""
Chapter 27 - Natural Language Processing Fundamentals
Script 8: Cosine Similarity & Vector Semantic Space

Mengimplementasikan kalkulasi geometris Cosine Similarity dan Cosine Distance dari dasar:
    cos(u, v) = (u . v) / (||u||_2 * ||v||_2)
Membuktikan sifat ortogonalitas dan kolinearitas pada ruang vektor, melakukan
pencarian kata terdekat (Nearest Neighbor Query), serta menganalisis batas interpretasinya.
"""

import numpy as np
from typing import Dict, List, Tuple


def cosine_similarity(u: np.ndarray, v: np.ndarray) -> float:
    """Menghitung derajat kesamaan kosinus antara dua vektor u dan v."""
    dot_product = np.dot(u, v)
    norm_u = np.linalg.norm(u)
    norm_v = np.linalg.norm(v)
    if norm_u == 0.0 or norm_v == 0.0:
        return 0.0
    return float(dot_product / (norm_u * norm_v))


def cosine_distance(u: np.ndarray, v: np.ndarray) -> float:
    """Menghitung jarak kosinus: d_cos(u, v) = 1 - cos(u, v)."""
    return 1.0 - cosine_similarity(u, v)


def find_nearest_neighbors(
    query_vector: np.ndarray, word_vectors: Dict[str, np.ndarray], top_k: int = 3
) -> List[Tuple[str, float]]:
    """Mencari k kata dengan kesamaan kosinus tertinggi terhadap vektor query."""
    scores = []
    for word, vec in word_vectors.items():
        sim = cosine_similarity(query_vector, vec)
        scores.append((word, sim))
    # Urutkan menurun berdasarkan skor kesamaan
    scores.sort(key=lambda item: -item[1])
    return scores[:top_k]


def main():
    print("=== Chapter 27: Demo Cosine Similarity & Semantic Space ===")

    # 1. Pembuktian Sifat Matematis Geometri
    v_base = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    v_parallel = np.array([2.0, 4.0, 6.0], dtype=np.float64)  # Searah, beda magnitudo
    v_orthogonal = np.array([-2.0, 1.0, 0.0], dtype=np.float64)  # Tegak lurus: dot product = 0
    v_opposite = np.array([-1.0, -2.0, -3.0], dtype=np.float64)  # Berlawanan arah

    sim_parallel = cosine_similarity(v_base, v_parallel)
    sim_ortho = cosine_similarity(v_base, v_orthogonal)
    sim_opposite = cosine_similarity(v_base, v_opposite)

    print("1. Pembuktian Teorema Geometri Sudut:")
    print(f"  Vektor Sejajar Searah   : Cosine = {sim_parallel:.4f} (Ekspektasi: +1.0000)")
    print(f"  Vektor Tegak Lurus      : Cosine = {sim_ortho:.4f} (Ekspektasi:  0.0000)")
    print(f"  Vektor Berlawanan Arah  : Cosine = {sim_opposite:.4f} (Ekspektasi: -1.0000)")

    assert abs(sim_parallel - 1.0) < 1e-6, "Vektor sejajar harus bernilai 1.0"
    assert abs(sim_ortho - 0.0) < 1e-6, "Vektor tegak lurus harus bernilai 0.0"
    assert abs(sim_opposite - (-1.0)) < 1e-6, "Vektor berlawanan harus bernilai -1.0"

    # 2. Ruang Vektor Semantik Kata Sederhana (3 Dimensi Simbolik)
    # Dimensi 1: [Hewan Peliharaan], Dimensi 2: [Elektronik], Dimensi 3: [Aktivitas]
    embeddings: Dict[str, np.ndarray] = {
        "kucing": np.array([0.92, 0.05, 0.30]),
        "anjing": np.array([0.88, 0.08, 0.35]),
        "kelinci": np.array([0.80, 0.02, 0.20]),
        "laptop": np.array([0.05, 0.95, 0.15]),
        "komputer": np.array([0.08, 0.90, 0.20]),
        "ponsel": np.array([0.03, 0.92, 0.18]),
        "berlari": np.array([0.30, 0.10, 0.90]),
        "melompat": np.array([0.25, 0.05, 0.92]),
    }

    print("\n2. Analisis Ruang Semantik Terdistribusi:")
    sim_kucing_anjing = cosine_similarity(embeddings["kucing"], embeddings["anjing"])
    sim_kucing_laptop = cosine_similarity(embeddings["kucing"], embeddings["laptop"])
    print(f"  Kesamaan ('kucing', 'anjing')   : {sim_kucing_anjing:.4f} (Kelompok Hewan)")
    print(f"  Kesamaan ('kucing', 'laptop')   : {sim_kucing_laptop:.4f} (Lain Domain)")

    # 3. Kueri Tetangga Terdekat (Nearest Neighbors Query)
    query_word = "anjing"
    top_neighbors = find_nearest_neighbors(embeddings[query_word], embeddings, top_k=3)
    print(f"\n3. Kata Paling Mirip terhadap '{query_word}':")
    for rank, (word, score) in enumerate(top_neighbors, 1):
        print(f"  Peringkat {rank}: '{word:10s}' dengan Cosine Similarity = {score:.4f}")

    # Catatan Metodologi Ilmiah
    print("\n[Peringatan Metodologis Responsible NLP]:")
    print("  Cosine similarity mengukur sudut geometris dalam ruang koordinat terdefinisi,")
    print("  tetapi tidak boleh diklaim secara sepihak membuktikan 'kesamaan makna' mutlak")
    print("  karena representasi statis tidak memperhitungkan polisemi dan konteks kalimat.")

    print("\nStatus: cosine_similarity.py berhasil dieksekusi 100%!")


if __name__ == "__main__":
    main()

