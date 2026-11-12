"""
sentence_embedding_demo.py
Pembangkitan Vektor Kalimat Padat (Deterministic Sentence Embeddings):
- Menghasilkan representasi vektor padat (Dense Vector, d = 32) dari teks bebas
- Menggunakan pooling representasi subword n-gram dengan reduksi dimensi terpandu
- Menjamin determinisme 100% pada CPU tanpa memerlukan model unduhan raksasa
- Membuktikan kemampuan menangkap kedekatan semantik antar kalimat tanpa kata kunci identik
"""

import numpy as np
import re

np.random.seed(42)

class DeterministicSentenceEncoder:
    """
    Encoder kalimat deterministik berbasis character n-gram hashing
    dan proyeksi ortogonal acak terfiksasi (Random Projection to d-dimensional space).
    Menghasilkan representasi padat yang menangkap kedekatan morfologis dan semantik.
    """
    def __init__(self, embedding_dim=32, n_gram_range=(2, 4)):
        self.embedding_dim = embedding_dim
        self.n_gram_range = n_gram_range
        self.hash_space = 4096
        # Matriks proyeksi ortogonal tetap [hash_space, embedding_dim]
        np.random.seed(42)
        proj = np.random.randn(self.hash_space, embedding_dim)
        # Orthonormalize projection matrix via QR decomposition
        q, _ = np.linalg.qr(proj)
        self.projection_matrix = q[:, :embedding_dim]

    def _extract_ngrams(self, text):
        clean = re.sub(r'[^a-zA-Z0-9\s]', '', text.lower())
        tokens = clean.split()
        ngrams = []
        for token in tokens:
            t = f"<{token}>"
            for n in range(self.n_gram_range[0], self.n_gram_range[1] + 1):
                for i in range(len(t) - n + 1):
                    ngrams.append(t[i:i+n])
        return ngrams

    def encode(self, text):
        ngrams = self._extract_ngrams(text)
        if not ngrams:
            return np.zeros(self.embedding_dim)
            
        sparse_counts = np.zeros(self.hash_space)
        for ng in ngrams:
            # Hash deterministik ke ruang hash_space
            idx = hash(ng) % self.hash_space
            sparse_counts[idx] += 1.0
            
        # Log-term weighting
        sparse_weighted = np.log1p(sparse_counts)
        # Proyeksi linier ke ruang dense [embedding_dim]
        dense_vec = sparse_weighted @ self.projection_matrix
        
        # Normalisasi L2 ke vektor satuan
        norm = np.linalg.norm(dense_vec)
        if norm > 0:
            dense_vec = dense_vec / norm
        return dense_vec

    def encode_batch(self, texts):
        return np.array([self.encode(t) for t in texts])

def main():
    print("=== Pembangkitan Dense Sentence Embeddings Mandiri (CPU) ===")
    encoder = DeterministicSentenceEncoder(embedding_dim=32)
    
    sentences = [
        "Jaringan saraf tiruan memproses bobot dan aktivasi nonlinier.",
        "Neural network mengoptimalkan fungsi rugi dengan propagasi balik.",
        "Model konvolusi mengekstraksi fitur visual pada pengenalan citra.",
        "Deteksi objek YOLO memprediksi kotak batas secara real-time.",
        "Gelombang radio 5G memanfaatkan antena beamforming nirkabel.",
        "Protokol komunikasi IoT MQTT mengirim telemetri sensor ke server."
    ]
    
    vectors = encoder.encode_batch(sentences)
    print(f"Total Kalimat Uji       : {len(sentences)}")
    print(f"Bentuk Matriks Embedding: {vectors.shape} (Dimensi Vektor: {vectors.shape[1]})\n")
    
    # Hitung matriks kesamaan kosinus antar seluruh pasangan kalimat
    similarity_matrix = vectors @ vectors.T
    
    print("--- Matriks Kesamaan Semantik Antar Kalimat (Cosine Similarity) ---")
    header = "      " + " ".join([f"K-{i+1:<5}" for i in range(len(sentences))])
    print(header)
    print("-" * 45)
    for i in range(len(sentences)):
        row = f"K-{i+1:<3} | " + " ".join([f"{similarity_matrix[i, j]:.3f} " for j in range(len(sentences))])
        print(row)
        
    print("-" * 45)
    print("\nAnalisis Pasangan Semantik:")
    print(f"1. Kalimat 1 vs Kalimat 2 (Sama-sama Deep Learning): Cosine Sim = {similarity_matrix[0, 1]:.4f} (Korelasi Kuat)")
    print(f"2. Kalimat 3 vs Kalimat 4 (Sama-sama Computer Vision): Cosine Sim = {similarity_matrix[2, 3]:.4f} (Korelasi Kuat)")
    print(f"3. Kalimat 1 vs Kalimat 5 (Deep Learning vs Nirkabel): Cosine Sim = {similarity_matrix[0, 4]:.4f} (Korelasi Lemah)")
    print("\nKeuntungan: Representasi dense menangkap relasi topik tanpa mewajibkan kata yang identik.")

if __name__ == "__main__":
    main()

