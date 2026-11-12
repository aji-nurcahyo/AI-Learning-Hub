"""
Chapter 27 - Natural Language Processing Fundamentals
Script 3: Bag-of-Words (BoW) & Term Frequency Matrix

Mengimplementasikan model representasi Bag-of-Words dari dasar (scratch NumPy)
serta membandingkan hasilnya terhadap CountVectorizer Scikit-learn, menganalisis
kepadatan/kelangkaan matriks (matrix sparsity), dan batasan ketiadaan urutan kata.
"""

import numpy as np
from typing import List, Dict
from sklearn.feature_extraction.text import CountVectorizer


class BagOfWordsFromScratch:
    """Implementasi vektorisasi Bag-of-Words murni berbasis kamus dan NumPy."""

    def __init__(self):
        self.vocab: Dict[str, int] = {}
        self.feature_names: List[str] = []

    def fit(self, documents: List[str]):
        """Membangun kosakata unik terurut secara alfabetis dari kumpulan dokumen."""
        unique_words = set()
        for doc in documents:
            words = doc.lower().split()
            for w in words:
                clean_w = "".join(c for c in w if c.isalnum())
                if clean_w:
                    unique_words.add(clean_w)

        self.feature_names = sorted(list(unique_words))
        self.vocab = {word: idx for idx, word in enumerate(self.feature_names)}
        return self

    def transform(self, documents: List[str]) -> np.ndarray:
        """Mengonversi dokumen menjadi matriks frekuensi istilah (N dokumen x V kosakata)."""
        matrix = np.zeros((len(documents), len(self.vocab)), dtype=np.int32)
        for doc_idx, doc in enumerate(documents):
            words = doc.lower().split()
            for w in words:
                clean_w = "".join(c for c in w if c.isalnum())
                if clean_w in self.vocab:
                    vocab_idx = self.vocab[clean_w]
                    matrix[doc_idx, vocab_idx] += 1
        return matrix

    def fit_transform(self, documents: List[str]) -> np.ndarray:
        return self.fit(documents).transform(documents)


def calculate_matrix_sparsity(matrix: np.ndarray) -> float:
    """Menghitung persentase elemen bernilai nol dalam matriks representasi."""
    zero_count = np.sum(matrix == 0)
    total_elements = matrix.size
    return (zero_count / total_elements) * 100.0


def main():
    print("=== Chapter 27: Demo Bag-of-Words Representation ===")

    corpus = [
        "Kucing mengejar tikus kecil",
        "Tikus kecil memakan keju",
        "Kucing tidur di atas kasur",
        "Keju gurih dimakan tikus",
    ]

    # 1. Eksekusi model BoW Scratch
    bow_scratch = BagOfWordsFromScratch()
    matrix_scratch = bow_scratch.fit_transform(corpus)

    print(f"Jumlah Dokumen: {len(corpus)}")
    print(f"Ukuran Kosakata (V): {len(bow_scratch.feature_names)} kata unik")
    print("Daftar Fitur Kosakata:")
    print(" ", bow_scratch.feature_names)

    print("\nMatriks Frekuensi Istilah (Scratch):")
    print(matrix_scratch)

    sparsity_scratch = calculate_matrix_sparsity(matrix_scratch)
    print(f"Tingkat Kelangkaan Matriks (Sparsity): {sparsity_scratch:.2f}% elemen nol")

    # 2. Verifikasi terhadap Scikit-learn CountVectorizer
    sklearn_vec = CountVectorizer()
    matrix_sklearn = sklearn_vec.fit_transform(corpus).toarray()
    sklearn_features = sklearn_vec.get_feature_names_out().tolist()

    # Periksa kesesuaian dimensi
    assert matrix_scratch.shape == matrix_sklearn.shape, "Dimensi matriks BoW harus identik"
    print("\n[Verifikasi Scikit-learn]: Dimensi matriks cocok sempurna:", matrix_scratch.shape)

    # 3. Demonstrasi Batasan Ketiadaan Urutan Kata (Order Invariance)
    s1 = "Saya tidak benci kamu"
    s2 = "Kamu tidak benci saya"
    s3 = "Tidak kamu benci saya"

    demo_vocab = BagOfWordsFromScratch().fit([s1, s2, s3])
    m_s1 = demo_vocab.transform([s1])
    m_s2 = demo_vocab.transform([s2])
    m_s3 = demo_vocab.transform([s3])

    print("\n[Batasan BoW: Hilangnya Urutan Kata]:")
    print(f"  Vektor s1 ('{s1}') : {m_s1.tolist()[0]}")
    print(f"  Vektor s2 ('{s2}') : {m_s2.tolist()[0]}")
    print(f"  Vektor s3 ('{s3}') : {m_s3.tolist()[0]}")
    assert np.array_equal(m_s1, m_s2) and np.array_equal(m_s2, m_s3), "BoW mengabaikan permutasi susunan kata"
    print("  -> Ketiga kalimat dengan makna atau struktur gramatikal berbeda menghasilkan vektor identik!")

    print("\nStatus: bag_of_words.py berhasil dieksekusi 100%!")


if __name__ == "__main__":
    main()

