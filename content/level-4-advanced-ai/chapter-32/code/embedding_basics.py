"""
embedding_basics.py
Konsep dasar representasi vektor (Embeddings):
- Perbandingan representasi jarang (Sparse Vectors: One-Hot, Bag-of-Words) vs representasi padat (Dense Embeddings)
- Dimensi ruang vektor (Vector Space), kepadatan informasi (Information Density), dan relasi semantik
"""

import numpy as np

np.random.seed(42)

def main():
    print("=== Konsep Dasar Embeddings: Sparse vs Dense Representations ===")
    
    # Kosakata mini (Vocabulary)
    vocab = ["jaringan", "syaraf", "tiruan", "sinyal", "sensor", "telekomunikasi", "kosinus", "vektor"]
    vocab_size = len(vocab)
    token2id = {w: i for i, w in enumerate(vocab)}
    
    # 1. Representasi Jarang (One-Hot Encoding)
    print("\n[1] Representasi Sparse (One-Hot Encoding):")
    print(f"Ukuran Kosakata (V) : {vocab_size} dimensi")
    
    word_a = "jaringan"
    word_b = "syaraf"
    word_c = "telekomunikasi"
    
    vec_a_sparse = np.zeros(vocab_size)
    vec_a_sparse[token2id[word_a]] = 1.0
    
    vec_b_sparse = np.zeros(vocab_size)
    vec_b_sparse[token2id[word_b]] = 1.0
    
    print(f"One-Hot '{word_a}' : {vec_a_sparse.astype(int)}")
    print(f"One-Hot '{word_b}'   : {vec_b_sparse.astype(int)}")
    
    # Dot product antara dua one-hot berbeda selalu 0 (Orthogonal)
    dot_sparse = np.dot(vec_a_sparse, vec_b_sparse)
    print(f"Dot Product '{word_a}' & '{word_b}' (Sparse): {dot_sparse:.1f}")
    print("-> Kelemahan Sparse: Semua kata saling tegak lurus (orthogonal); tidak menangkap kemiripan semantik!")
    
    # 2. Representasi Padat (Dense Embeddings)
    # Memetakan kata ke ruang kontinu berdimensi lebih rendah (misal d = 4)
    embedding_dim = 4
    print(f"\n[2] Representasi Dense (d = {embedding_dim} dimensi):")
    
    # Matriks embedding sintetis yang merefleksikan kedekatan semantik
    # Dimensi 0: Ranah AI/Deep Learning
    # Dimensi 1: Ranah Fisik/Hardware/Sensor
    # Dimensi 2: Ranah Telekomunikasi
    # Dimensi 3: Ranah Aljabar/Matematika
    embeddings = {
        "jaringan":       np.array([ 0.85,  0.10,  0.75,  0.05]),
        "syaraf":         np.array([ 0.95,  0.05,  0.02,  0.10]),
        "tiruan":         np.array([ 0.90,  0.02,  0.01,  0.20]),
        "sinyal":         np.array([ 0.15,  0.80,  0.85,  0.30]),
        "sensor":         np.array([ 0.10,  0.95,  0.50,  0.05]),
        "telekomunikasi": np.array([ 0.10,  0.60,  0.95,  0.10]),
        "kosinus":        np.array([ 0.10,  0.05,  0.05,  0.95]),
        "vektor":         np.array([ 0.40,  0.05,  0.10,  0.90])
    }
    
    for word, vec in embeddings.items():
        print(f"Dense '{word:<14}' : {vec}")
        
    # 3. Kesamaan Kosinus pada Ruang Dense
    def cosine_sim(u, v):
        return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))
    
    sim_ab = cosine_sim(embeddings["jaringan"], embeddings["syaraf"])
    sim_ac = cosine_sim(embeddings["jaringan"], embeddings["telekomunikasi"])
    sim_ak = cosine_sim(embeddings["jaringan"], embeddings["kosinus"])
    
    print("\n[3] Pengukuran Kesamaan Semantik (Cosine Similarity) pada Ruang Dense:")
    print(f"Similarity('jaringan', 'syaraf')         : {sim_ab:.4f} (Tinggi, konsep terkait)")
    print(f"Similarity('jaringan', 'telekomunikasi') : {sim_ac:.4f} (Tinggi, konektivitas)")
    print(f"Similarity('jaringan', 'kosinus')        : {sim_ak:.4f} (Rendah, domain berbeda)")
    
    print("\nKesimpulan: Dense embeddings mengompresi informasi ke dalam ruang kontinu berdimensi tetap,")
    print("memungkinkan kalkulasi kemiripan semantik melalui operasi aljabar vektor sederhana.")

if __name__ == "__main__":
    main()

