"""
cosine_similarity.py
Analisis Mendalam dan Stabilitas Numerik Cosine Similarity:
- Penanganan eksplisit vektor ber-norma nol (Zero Vector) tanpa ZeroDivisionError
- Penjepitan numerik (Clamping) pada interval [-1.0, 1.0] untuk mencegah kesalahan floating-point
- Perbedaan arah perankingan: Cosine Similarity (Makin besar makin mirip) vs Cosine Distance (Makin kecil makin mirip)
"""

import numpy as np

def safe_cosine_similarity(u, v, eps=1e-12):
    """
    Menghitung Cosine Similarity dengan proteksi numerik ketat:
    - Menangani vektor nol secara eksplisit
    - Menjepit hasil ke interval [-1.0, 1.0]
    """
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)
    
    norm_u = np.linalg.norm(u)
    norm_v = np.linalg.norm(v)
    
    # Deteksi Vektor Nol
    if norm_u < eps or norm_v < eps:
        # Vektor nol tidak memiliki arah yang terdefinisi
        return 0.0
        
    dot = np.dot(u, v)
    cos_sim = dot / (norm_u * norm_v)
    
    # Penjepitan batas floating-point (clamping)
    return float(np.clip(cos_sim, -1.0, 1.0))

def main():
    print("=== Stabilitas Numerik & Penanganan Edge Cases Cosine Similarity ===")
    
    # Uji Kasus Ekstrem
    cases = [
        ("Vektor Identik (Paritas Penuh)", [1.0, 2.0, 3.0], [1.0, 2.0, 3.0]),
        ("Vektor Sangat Kecil / Dekat Nol", [1e-15, 1e-15], [2.0, 4.0]),
        ("Vektor Nol Mutlak [0, 0, 0]", [0.0, 0.0, 0.0], [1.0, 5.0, 2.0]),
        ("Kedua Vektor Nol Mutlak", [0.0, 0.0], [0.0, 0.0]),
        ("Potensi Overflow Floating-Point", [1.0, 1.0000000000000002], [1.0, 1.0]),
        ("Berlawanan Arah Sempurna", [3.0, -4.0], [-3.0, 4.0])
    ]
    
    print(f"{'Deskripsi Kasus Uji':<35} {'Cosine Sim':<15} {'Cosine Distance':<18} {'Status Numerik'}")
    print("-" * 80)
    
    for desc, u, v in cases:
        sim = safe_cosine_similarity(u, v)
        dist = 1.0 - sim
        status = "Normal"
        if sim == 0.0 and (np.linalg.norm(u) == 0 or np.linalg.norm(v) == 0):
            status = "Zero Vector Handled"
        elif sim == 1.0:
            status = "Max Identity"
        elif sim == -1.0:
            status = "Opposite Direction"
            
        print(f"{desc:<35} {sim:<15.6f} {dist:<18.6f} {status}")
        
    print("-" * 80)
    print("\nPedoman Rekayasa Sistem Temu Kembali (Information Retrieval):")
    print("1. Saat menyusun peringkat (ranking):")
    print("   - Gunakan `Cosine Similarity` dengan pengurutan menurun (DESCENDING: nilai tertinggi di posisi atas).")
    print("   - Gunakan `Cosine Distance` dengan pengurutan menaik (ASCENDING: nilai terendah di posisi atas).")
    print("2. Dokumen atau kueri kosong wajib ditangani dengan skor netral (0.0) agar tidak merusak kalkulasi matriks.")

if __name__ == "__main__":
    main()

