"""
01_rule_based_recommender.py
Contoh 1: Sistem Rekomendasi Berbasis Aturan (Rule-Based Recommender).

Tujuan:
Mendemonstrasikan pendekatan deterministik berbasis aturan logika eksplisit
(if-else / aturan kondisi). Program ini BUKAN model Machine Learning karena
aturan ditentukan secara manual oleh manusia, bukan dipelajari dari data.
"""

from typing import Dict, Any, List

def recommend_study_path(profile: Dict[str, Any]) -> List[str]:
    """
    Menghasilkan rekomendasi materi berdasarkan aturan logika kondisional.
    
    Parameter:
    profile: dictionary berisi minat (interest) dan pengalaman (experience_level).
    
    Output:
    Daftar topik pembelajaran yang direkomendasikan.
    """
    interest = profile.get("interest", "").lower()
    exp = profile.get("experience_level", "").lower()
    recommendations = []

    # Aturan 1: Fondasi pemrograman
    if exp in ["pemula", "beginner"]:
        recommendations.append("Fondasi Python & Struktur Data Dasar")

    # Aturan 2: Pemetaan minat spesifik
    if interest == "computer vision":
        if exp in ["menengah", "advanced"]:
            recommendations.append("Convolutional Neural Networks (CNN) & Pengolahan Citra")
        else:
            recommendations.append("Dasar Representasi Matriks Citra Digital")
    elif interest == "natural language processing":
        if exp in ["menengah", "advanced"]:
            recommendations.append("Large Language Models & Fine-Tuning")
        else:
            recommendations.append("Pengantar Tokenisasi & Klasifikasi Teks")
    elif interest == "robotics":
        recommendations.append("Integrasi Sensor, Kinematika & Reinforcement Learning Dasar")
    else:
        recommendations.append("Dasar-dasar Algoritma AI & Representasi Pengetahuan")

    # Aturan 3: Etika AI berlaku untuk semua tingkat
    recommendations.append("Prinsip Responsible AI & Mitigasi Bias")

    return recommendations

if __name__ == "__main__":
    print("=== DEMO: SISTEM REKOMENDASI BERBASIS ATURAN ===")
    
    # Pengujian beberapa profil pengguna
    test_profiles = [
        {"user": "Mahasiswa A", "interest": "computer vision", "experience_level": "pemula"},
        {"user": "Mahasiswa B", "interest": "natural language processing", "experience_level": "menengah"},
        {"user": "Mahasiswa C", "interest": "finance", "experience_level": "pemula"},
    ]

    for p in test_profiles:
        print(f"\nProfil: {p['user']} (Minat: {p['interest']}, Level: {p['experience_level']})")
        recs = recommend_study_path(p)
        for i, rec in enumerate(recs, start=1):
            print(f"  {i}. {rec}")

    print("\n[Catatan Analisis]")
    print("Kelebihan: Transparan, 100% dapat dilacak (auditable), tidak butuh dataset training.")
    print("Keterbatasan: Kaku (*brittle*). Jika ada input tak terduga (misal 'AI bahasa'), sistem jatuh ke fallback.")

