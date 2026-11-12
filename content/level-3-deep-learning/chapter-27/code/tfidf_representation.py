"""
Chapter 27 - Natural Language Processing Fundamentals
Script 4: TF-IDF (Term Frequency - Inverse Document Frequency) Representation

Menurunkan dan mengimplementasikan perhitungan matematis TF-IDF secara analitis langkah-demi-langkah,
mengaplikasikan smoothing IDF standar Scikit-learn:
    idf(t) = ln((1 + N) / (1 + df(t))) + 1
diikuti normalisasi vektor Euclidean (L2-norm), serta membuktikan kesetaraan numerik
terhadap Scikit-learn TfidfVectorizer dengan toleransi komputasi < 1e-6.
"""

import math
import numpy as np
from typing import List, Dict, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer


class TFIDFFromScratch:
    """Implementasi perhitungan matematis analitis TF-IDF sesuai formula Scikit-learn."""

    def __init__(self, norm: str = "l2", smooth_idf: bool = True):
        self.norm = norm
        self.smooth_idf = smooth_idf
        self.vocab: Dict[str, int] = {}
        self.feature_names: List[str] = []
        self.idf_values: np.ndarray = np.array([])

    def fit(self, documents: List[str]):
        """Membangun kosakata dan menghitung nilai Inverse Document Frequency (IDF)."""
        N = len(documents)
        doc_tokens = []
        unique_words = set()

        for doc in documents:
            tokens = [w for w in doc.lower().split() if w.isalnum()]
            doc_tokens.append(set(tokens))
            unique_words.update(tokens)

        self.feature_names = sorted(list(unique_words))
        self.vocab = {w: i for i, w in enumerate(self.feature_names)}
        V = len(self.feature_names)

        # Hitung Document Frequency: jumlah dokumen yang memuat kata t
        df = np.zeros(V, dtype=np.float64)
        for d_set in doc_tokens:
            for w in d_set:
                if w in self.vocab:
                    df[self.vocab[w]] += 1.0

        # Hitung IDF dengan formula smoothed Scikit-learn
        if self.smooth_idf:
            # idf(t) = ln((1 + N) / (1 + df(t))) + 1
            self.idf_values = np.log((1.0 + N) / (1.0 + df)) + 1.0
        else:
            # idf(t) = ln(N / df(t)) + 1
            self.idf_values = np.log(N / df) + 1.0

        return self

    def transform(self, documents: List[str]) -> np.ndarray:
        """Menghitung bobot TF x IDF dan menerapkan normalisasi L2 pada setiap dokumen."""
        N = len(documents)
        V = len(self.vocab)
        tf_matrix = np.zeros((N, V), dtype=np.float64)

        for doc_idx, doc in enumerate(documents):
            tokens = [w for w in doc.lower().split() if w.isalnum()]
            for w in tokens:
                if w in self.vocab:
                    tf_matrix[doc_idx, self.vocab[w]] += 1.0

        # Bobot TF-IDF mentah = TF x IDF
        tfidf_raw = tf_matrix * self.idf_values

        # Normalisasi L2 per baris dokumen: v_norm = v / ||v||_2
        if self.norm == "l2":
            norms = np.linalg.norm(tfidf_raw, axis=1, keepdims=True)
            # Hindari pembagian dengan nol
            norms[norms == 0.0] = 1.0
            tfidf_norm = tfidf_raw / norms
            return tfidf_norm

        return tfidf_raw

    def fit_transform(self, documents: List[str]) -> np.ndarray:
        return self.fit(documents).transform(documents)


def main():
    print("=== Chapter 27: Demo TF-IDF Step-by-Step Calculation ===")

    corpus = [
        "data sains mempelajari data",
        "mesin kecerdasan buatan",
        "data mesin buatan sains",
    ]

    print("Korpus Latihan (3 Dokumen):")
    for i, d in enumerate(corpus):
        print(f"  Dokumen {i+1}: '{d}'")

    # 1. Eksekusi Scratch
    tfidf_manual = TFIDFFromScratch(norm="l2", smooth_idf=True)
    matrix_manual = tfidf_manual.fit_transform(corpus)

    print(f"\nKosakata Unik ({len(tfidf_manual.feature_names)} kata):")
    for word, idf_val in zip(tfidf_manual.feature_names, tfidf_manual.idf_values):
        df_count = sum(1 for d in corpus if word in d.split())
        print(f"  Kata '{word:12s}': DF={df_count}, IDF={idf_val:.4f}")

    print("\nMatriks TF-IDF Manual (Ternormalisasi L2):")
    print(np.round(matrix_manual, 4))

    # 2. Verifikasi terhadap Scikit-learn TfidfVectorizer
    sklearn_tfidf = TfidfVectorizer(norm="l2", smooth_idf=True)
    matrix_sklearn = sklearn_tfidf.fit_transform(corpus).toarray()

    # Periksa deviasi numerik absolut maksimum
    max_deviation = np.max(np.abs(matrix_manual - matrix_sklearn))
    print(f"\nDeviasi Numerik Maksimum terhadap Scikit-learn: {max_deviation:.2e}")
    assert max_deviation < 1e-6, "Perhitungan TF-IDF manual harus identik dengan Scikit-learn"
    print("Status: Kesetaraan matematis TF-IDF terbukti 100% IDENTIK!")

    # 3. Analisis Nilai Penting Kata
    # Kata 'data' muncul di 2 dokumen, sementara 'kecerdasan' hanya di 1 dokumen
    idx_data = tfidf_manual.vocab["data"]
    idx_kecerdasan = tfidf_manual.vocab["kecerdasan"]
    print("\n[Interpretasi Pembobotan IDF]:")
    print(f"  IDF 'data'       (DF=2): {tfidf_manual.idf_values[idx_data]:.4f} (Bobot lebih rendah karena lebih umum)")
    print(f"  IDF 'kecerdasan' (DF=1): {tfidf_manual.idf_values[idx_kecerdasan]:.4f} (Bobot lebih tinggi karena kata pembeda spesifik)")

    print("\nStatus: tfidf_representation.py berhasil dieksekusi 100%!")


if __name__ == "__main__":
    main()

