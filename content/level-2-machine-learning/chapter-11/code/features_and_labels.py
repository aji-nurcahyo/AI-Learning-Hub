"""
Chapter 11: Introduction to Machine Learning
Script 2: features_and_labels.py

Tujuan:
Mengenalkan struktur fundamental dataset Machine Learning:
- Matriks Fitur (Feature Matrix X): representasi matematis 2D (N x D)
- Vektor Target / Label (Target Vector y): besaran 1D (N,) yang ingin diprediksi
- Perbedaan semantic roles antara input dan output.
"""

import pandas as pd
import numpy as np


def create_sample_student_dataset() -> pd.DataFrame:
    """Membangkitkan dataset akademik sintetis berukuran 10 sampel untuk ilustrasi."""
    data = {
        "student_id": [f"MHS-{i:02d}" for i in range(1, 11)],
        "study_hours_weekly": [12.5, 4.0, 18.0, 6.5, 14.0, 3.0, 16.5, 9.0, 11.0, 5.0],
        "attendance_pct": [92.0, 65.0, 98.0, 70.0, 88.0, 60.0, 95.0, 80.0, 85.0, 68.0],
        "previous_gpa": [3.45, 2.30, 3.80, 2.75, 3.20, 2.10, 3.65, 2.95, 3.10, 2.50],
        "final_exam_score": [82.5, 52.0, 94.0, 61.0, 80.0, 48.0, 89.0, 70.0, 76.0, 55.0], # Target Regresi
        "passed": [1, 0, 1, 0, 1, 0, 1, 1, 1, 0]                                             # Target Klasifikasi
    }
    return pd.DataFrame(data)


def main():
    print("=" * 65)
    print("DEMO: ANATOMI DATASET MACHINE LEARNING (FEATURES & LABELS)")
    print("=" * 65)

    df = create_sample_student_dataset()
    print("\n1. Data Tabel Asli (DataFrame):")
    print(df.to_string(index=False))

    # 1. Pemisahan Feature Matrix X
    # Kolom identitas (student_id) tidak memiliki daya generalisasi, sehingga dibuang
    feature_cols = ["study_hours_weekly", "attendance_pct", "previous_gpa"]
    X = df[feature_cols].to_numpy()

    # 2. Pemisahan Target y
    # Kasus A: Regresi (Prediksi Nilai Ujian Kontinu)
    y_reg = df["final_exam_score"].to_numpy()
    # Kasus B: Klasifikasi (Prediksi Kelulusan Diskrit Biner: 1=Lulus, 0=Tidak)
    y_clf = df["passed"].to_numpy()

    print("\n2. Matriks Fitur (Feature Matrix X):")
    print(f"   Dimensi X (N samples x D features): {X.shape}")
    print(f"   Nama Fitur: {feature_cols}")
    print(f"   Tipe Data: {X.dtype}")
    print("   3 Baris Pertama X:")
    print(X[:3])

    print("\n3. Vektor Target Regresi (y_reg - Kontinu):")
    print(f"   Dimensi y: {y_reg.shape}")
    print(f"   3 Sampel Pertama: {y_reg[:3]}")

    print("\n4. Vektor Target Klasifikasi (y_clf - Diskrit Biner):")
    print(f"   Dimensi y: {y_clf.shape}")
    print(f"   Distribusi Kelas: Lulus (1)={np.sum(y_clf == 1)}, Tidak Lulus (0)={np.sum(y_clf == 0)}")
    print(f"   3 Sampel Pertama: {y_clf[:3]}")

    print("\n--- Kesimpulan Penting ---")
    print("Fitur adalah variabel independen (deskriptor yang teramati).")
    print("Label/Target adalah variabel dependen yang ingin dipelajari asosiasinya.")
    print("=" * 65)


if __name__ == "__main__":
    main()

