"""
03_exact_matching_vs_variation.py
Contoh 3: Exact Matching vs Variasi Input Alami Manusia.

Tujuan:
Menganalisis kerapuhan (*brittleness*) aturan deterministik saat berhadapan
dengan variasi bahasa manusia: salah ketik (*typo*), sinonim, dan perubahan struktur kalimat.
"""

from typing import Dict, Optional

# Basis aturan FAQ deterministik (Exact Pattern Mapping)
FAQ_DATABASE: Dict[str, str] = {
    "kapan jadwal uas": "UAS dijadwalkan pada minggu ke-16 semester berjalan.",
    "bagaimana cara reset password": "Kunjungi portal akademik dan klik 'Lupa Password'.",
    "berapa sks batas maksimal": "Batas maksimal pengambilan adalah 24 SKS per semester.",
}

def answer_exact_match(query: str) -> Optional[str]:
    """Mencari jawaban hanya dengan pencocokan string persis."""
    cleaned = query.strip().lower()
    return FAQ_DATABASE.get(cleaned, None)

if __name__ == "__main__":
    print("=== PERBANDINGAN: EXACT MATCHING VS VARIASI INPUT ===")
    
    variations = [
        ("kapan jadwal uas", "Input persis sesuai kamus"),
        ("Kapan jadwal UAS?", "Variasi tanda baca tanda tanya dan huruf kapital"),
        ("kapan jadual uas", "Variasi salah ketik / typo ('jadual')"),
        ("kapan ujian akhir semester dimulai", "Variasi sinonim / frasa formal"),
        ("kapan uas diadakan", "Variasi struktur kalimat"),
    ]

    print("Kueri acuan pada database: 'kapan jadwal uas'\n")
    print(f"{'Input Pengguna':<40} | {'Hasil Pencocokan':<25} | {'Status'}")
    print("-" * 80)

    for query, description in variations:
        ans = answer_exact_match(query)
        if ans:
            hasil = "Jawaban Ditemukan"
            status = "SUKSES"
        else:
            hasil = "Tidak Diketahui (None)"
            status = "GAGAL"
        print(f"{query:<40} | {hasil:<25} | {status} ({description})")

    print("\n[Kesimpulan Penting]")
    print("Exact matching gagal pada 4 dari 5 variasi input!")
    print("Sistem NLP / Machine Learning modern menyelesaikan masalah ini")
    print("dengan memetakan teks ke representasi numerik makna (vector embedding),")
    print("sehingga 'jadwal uas' dan 'ujian akhir semester' memiliki kemiripan semantik tinggi.")

