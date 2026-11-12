"""Reproducible Project Pipeline & Environment Validation.

Chapter 07: Python Ecosystem for AI
Artificial Intelligence Learning Hub - ITERA

Skrip ini mendemonstrasikan prinsip-prinsip rekayasa perangkat lunak untuk riset AI yang dapat direproduksi (reproducible AI):
1. Pengaturan Random Seed Global (Python & NumPy)
2. Manajemen Jalur Berkas Relatif yang Kebal Lingkungan (pathlib)
3. Validasi Integritas Skema Data Masukan
4. Pipeline Komputasi Deterministik (Prapemrosesan -> Analisis -> Verifikasi)
5. Uji Kesamaan Deterministik (Dua Eksekusi Menghasilkan Nilai Tepat Identik)
6. Dokumentasi Keterbatasan Reproduksibilitas (Perbedaan Arsitektur CPU/GPU & Library Stacks)
"""

import os
import random
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")


def set_reproducible_seeds(seed: int = 42):
    """Menetapkan seed acak untuk seluruh mesin bilangan pseudo-random."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def run_pipeline(seed: int = 42) -> np.ndarray:
    """Menjalankan alur komputasi numerik secara deterministik."""
    set_reproducible_seeds(seed)

    base_dir = Path(__file__).resolve().parent
    csv_file = base_dir / "data" / "sensor_readings.csv"

    if not csv_file.exists():
        raise FileNotFoundError(f"Dataset tidak ditemukan: {csv_file}")

    df = pd.read_csv(csv_file)
    # Penanganan missing values deterministik
    df["temperature_c"] = df["temperature_c"].fillna(df["temperature_c"].mean())
    df["vibration_mms"] = df["vibration_mms"].fillna(df["vibration_mms"].mean())

    # Fitur numerik
    X = df[["temperature_c", "vibration_mms"]].values

    # Inisialisasi bobot acak deterministik
    random_weights = np.random.normal(loc=0.0, scale=1.0, size=(2, 2))

    # Transformasi matriks
    transformed_features = X @ random_weights
    return transformed_features


def run_reproducible_demo():
    print("=" * 70)
    print("DEMO 7: REPRODUCIBLE PYTHON DATA ANALYSIS WORKFLOW")
    print("=" * 70)

    print("--- 1. Prinsip Reproduksibilitas dalam AI ---")
    print("- Random Seed Terkendali : Menghilangkan variasi acak semu pada eksperimen.")
    print("- Pathlib Relatif        : Mencegah hardcoded absolute path (D:\\... vs /home/...).")
    print("- Requirements Terkunci  : Menghindari ketidakcocokan versi pustaka.\n")

    # Uji determinisme: Eksekusi 1 vs Eksekusi 2
    print("--- 2. Uji Determinisme Dua Siklus Eksekusi Berturut-turut ---")
    run_1_output = run_pipeline(seed=42)
    run_2_output = run_pipeline(seed=42)

    is_exact_match = np.array_equal(run_1_output, run_2_output)
    max_diff = np.max(np.abs(run_1_output - run_2_output))

    print(f"Bentuk Matriks Luaran Pipeline : {run_1_output.shape}")
    print(f"Apakah Hasil Eksekusi 1 == 2? : {is_exact_match}")
    print(f"Selisih Maksimum Antar Siklus : {max_diff:.1e} (Nol Mutlak!)\n")

    print("5 Sampel Baris Hasil Transformasi Deterministik (Run 1):")
    for idx, row in enumerate(run_1_output[:5]):
        print(f"  Sampel {idx+1:02d} : [{row[0]:>8.4f}, {row[1]:>8.4f}]")

    print("\n--- 3. Catatan Keterbatasan Reproduksibilitas di Dunia Nyata ---")
    print("1. Seed menjamin hasil identik pada mesin & versi library yang sama.")
    print("2. Perbedaan arsitektur prosesor (x86 vs ARM64) atau level kompilasi BLAS/LAPACK")
    print("   dapat memicu perbedaan floating-point pada digit ke-15 (presisi mesin).")
    print("3. Pelatihan GPU paralel skala besar (CUDA non-deterministic operations) memerlukan")
    print("   flag deterministik khusus (torch.use_deterministic_algorithms).")
    print("=" * 70)


if __name__ == "__main__":
    run_reproducible_demo()

