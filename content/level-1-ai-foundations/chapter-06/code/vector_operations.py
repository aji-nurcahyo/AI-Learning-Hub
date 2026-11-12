"""Vector Operations and Linear Algebra Fundamentals for AI.

Chapter 06: Probability, Statistics & Linear Algebra for AI
Artificial Intelligence Learning Hub - ITERA

Skrip ini mendemonstrasikan operasi vektor murni dari awal menggunakan Python Standard Library:
1. Penjumlahan dan Pengurangan Vektor
2. Perkalian Skalar (Scalar Multiplication)
3. Hasil Kali Titik (Dot Product)
4. Norma Vektor (L1, L2 / Euclidean, dan Linf)
5. Vektor Satuan (Unit Vector / Normalization)
6. Kemiripan Kosinus (Cosine Similarity) dan Ortogonalitas
"""

import math
import sys
from typing import List, Tuple

sys.stdout.reconfigure(encoding="utf-8")

Vector = List[float]


def vector_add(u: Vector, v: Vector) -> Vector:
    """Menghitung u + v element-wise."""
    if len(u) != len(v):
        raise ValueError(f"Dimensi tidak cocok: {len(u)} != {len(v)}")
    return [ui + vi for ui, vi in zip(u, v)]


def vector_sub(u: Vector, v: Vector) -> Vector:
    """Menghitung u - v element-wise."""
    if len(u) != len(v):
        raise ValueError(f"Dimensi tidak cocok: {len(u)} != {len(v)}")
    return [ui - vi for ui, vi in zip(u, v)]


def scalar_multiply(c: float, v: Vector) -> Vector:
    """Menghitung c * v."""
    return [c * vi for vi in v]


def dot_product(u: Vector, v: Vector) -> float:
    """Menghitung u . v = sum(u_i * v_i)."""
    if len(u) != len(v):
        raise ValueError(f"Dimensi tidak cocok: {len(u)} != {len(v)}")
    return sum(ui * vi for ui, vi in zip(u, v))


def norm_l1(v: Vector) -> float:
    """Menghitung norma L1 (Manhattan): sum(|v_i|)."""
    return sum(abs(vi) for vi in v)


def norm_l2(v: Vector) -> float:
    """Menghitung norma L2 (Euclidean): sqrt(sum(v_i^2))."""
    return math.sqrt(sum(vi ** 2 for vi in v))


def norm_linf(v: Vector) -> float:
    """Menghitung norma Linf (Chebyshev / Maximum): max(|v_i|)."""
    return max(abs(vi) for vi in v)


def normalize_vector(v: Vector) -> Vector:
    """Menghasilkan unit vector v / ||v||_2."""
    magnitude = norm_l2(v)
    if magnitude == 0.0:
        raise ValueError("Vektor nol tidak dapat dinormalisasi.")
    return [vi / magnitude for vi in v]


def cosine_similarity(u: Vector, v: Vector) -> float:
    """Menghitung cosine similarity: (u . v) / (||u|| * ||v||)."""
    mag_u = norm_l2(u)
    mag_v = norm_l2(v)
    if mag_u == 0.0 or mag_v == 0.0:
        raise ValueError("Vektor nol tidak memiliki arah kemiripan sudut.")
    return dot_product(u, v) / (mag_u * mag_v)


def run_vector_experiments():
    print("=" * 70)
    print("EKSPERIMEN 4: OPERASI VEKTOR & GEOMETRI DASAR UNTUK AI")
    print("=" * 70)

    # Representasi fitur embedding dari 3 dokumen teks:
    # Dimensi [frekuensi_ai, frekuensi_robotik, frekuensi_ekonomi]
    doc_a = [3.0, 4.0, 0.0]    # Fokus AI dan Robotika
    doc_b = [6.0, 8.0, 0.0]    # Konten serupa dengan doc_a (skala 2x lipat)
    doc_c = [0.0, 0.0, 5.0]    # Fokus Ekonomi murni (ortogonal terhadap doc_a & doc_b)
    doc_d = [1.0, 2.0, 1.0]    # Campuran

    print(f"Vektor Dokumen A: {doc_a}")
    print(f"Vektor Dokumen B: {doc_b} (Skalar 2 * A)")
    print(f"Vektor Dokumen C: {doc_c} (Ekonomi)")
    print(f"Vektor Dokumen D: {doc_d} (Campuran)\n")

    print("--- 1. Operasi Aljabar Vektor ---")
    sum_ab = vector_add(doc_a, doc_b)
    diff_ad = vector_sub(doc_a, doc_d)
    scaled_a = scalar_multiply(0.5, doc_a)
    print(f"A + B               : {sum_ab}")
    print(f"A - D               : {diff_ad}")
    print(f"0.5 * A             : {scaled_a}\n")

    print("--- 2. Norma Vektor (Panjang / Magnitudo) ---")
    print(f"Norma L1 (Manhattan) A : {norm_l1(doc_a):.4f} (|3| + |4| + |0| = 7)")
    print(f"Norma L2 (Euclidean) A : {norm_l2(doc_a):.4f} (sqrt(3^2 + 4^2 + 0^2) = 5.0)")
    print(f"Norma Linf (Maksimum) A: {norm_linf(doc_a):.4f} (max(3, 4, 0) = 4)")
    unit_a = normalize_vector(doc_a)
    print(f"Unit Vector A (A / ||A||2): {unit_a}")
    print(f"Verifikasi ||Unit A||2   : {norm_l2(unit_a):.6f}\n")

    print("--- 3. Dot Product & Cosine Similarity (Pondasi Semantic Search) ---")
    dot_ab = dot_product(doc_a, doc_b)
    dot_ac = dot_product(doc_a, doc_c)
    cos_ab = cosine_similarity(doc_a, doc_b)
    cos_ac = cosine_similarity(doc_a, doc_c)
    cos_ad = cosine_similarity(doc_a, doc_d)

    print(f"Dot Product A . B   : {dot_ab:.2f} (3*6 + 4*8 + 0*0 = 18 + 32 = 50)")
    print(f"Cosine Similarity A & B: {cos_ab:.4f} -> Sudut 0° (Korelasi Arah Sempurna = 1.0)")
    print(f"Dot Product A . C   : {dot_ac:.2f} -> Hasil nol!")
    print(f"Cosine Similarity A & C: {cos_ac:.4f} -> Sudut 90° (Ortogonal / Topik Independen)")
    print(f"Cosine Similarity A & D: {cos_ad:.4f} -> Sebagian relevan\n")

    print("[INSIGHT AI]: Cosine similarity mengukur kesamaan arah vektor terlepas dari")
    print("panjang dokumen. Inilah mengapa embedding AI (Word2Vec, BERT, LLM embeddings)")
    print("menggunakan cosine similarity untuk mengukur kedekatan makna semantik!")
    print("=" * 70)


if __name__ == "__main__":
    run_vector_experiments()

