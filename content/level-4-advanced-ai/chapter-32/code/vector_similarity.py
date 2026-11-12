"""
vector_similarity.py
Operasi Aljabar Vektor dan Metrik Kesamaan:
- Dot Product (Inner Product)
- Euclidean Distance (Jarak L2)
- Cosine Similarity dan Cosine Distance
- Perilaku metrik pada vektor kolinear, ortogonal, berlawanan arah, dan magnitudo berbeda
"""

import numpy as np

def dot_product(u, v):
    return np.dot(u, v)

def euclidean_distance(u, v):
    return np.linalg.norm(u - v)

def cosine_similarity(u, v):
    norm_u = np.linalg.norm(u)
    norm_v = np.linalg.norm(v)
    if norm_u == 0 or norm_v == 0:
        return 0.0
    return np.dot(u, v) / (norm_u * norm_v)

def cosine_distance(u, v):
    return 1.0 - cosine_similarity(u, v)

def main():
    print("=== Eksplorasi Metrik Kesamaan & Jarak Vektor ===")
    
    # 4 Vektor Kasus Uji
    u = np.array([2.0, 3.0])            # Vektor referensi
    v_collinear = np.array([4.0, 6.0])  # Kolinear (arah sama persis, magnitudo 2x lipat)
    v_orthogonal = np.array([-3.0, 2.0]) # Ortogonal (tegak lurus, sudut 90 derajat)
    v_opposite = np.array([-2.0, -3.0]) # Berlawanan arah (sudut 180 derajat)
    
    test_cases = [
        ("Kolinear (Arah Sama)", u, v_collinear),
        ("Ortogonal (Tegak Lurus)", u, v_orthogonal),
        ("Berlawanan Arah (Invers)", u, v_opposite)
    ]
    
    print(f"Vektor Referensi u : {u} (Norm L2: {np.linalg.norm(u):.4f})\n")
    print(f"{'Kasus Hubungan':<26} {'Vektor v':<15} {'Dot Product':<14} {'Euclidean Dist':<16} {'Cosine Sim':<12} {'Cosine Dist'}")
    print("-" * 95)
    
    for label, vec_u, vec_v in test_cases:
        dp = dot_product(vec_u, vec_v)
        euc = euclidean_distance(vec_u, vec_v)
        cos_sim = cosine_similarity(vec_u, vec_v)
        cos_dist = cosine_distance(vec_u, vec_v)
        print(f"{label:<26} {str(vec_v.tolist()):<15} {dp:<14.2f} {euc:<16.4f} {cos_sim:<12.4f} {cos_dist:.4f}")
        
    print("-" * 95)
    print("\nTemuan Kritis:")
    print("1. Pada vektor kolinear, Cosine Similarity bernilai tepat 1.0000 meskipun Euclidean Distance > 0 (3.6056).")
    print("   -> Cosine Similarity HANYA mengukur orientasi/sudut, kebal terhadap skala magnitudo!")
    print("2. Dot Product sangat sensitif terhadap magnitudo: 26.00 pada kolinear panjang, meskipun orientasi sama.")
    print("3. Pada vektor ortogonal, Dot Product = 0.00 dan Cosine Similarity = 0.00.")
    print("4. Pada vektor berlawanan arah, Cosine Similarity bernilai minimum -1.0000.")

if __name__ == "__main__":
    main()

