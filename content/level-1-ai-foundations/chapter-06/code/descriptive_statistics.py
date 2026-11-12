"""Descriptive Statistics for Sensor Data.

Chapter 06: Probability, Statistics & Linear Algebra for AI
Artificial Intelligence Learning Hub - ITERA

Skrip ini mendemonstrasikan perhitungan statistik deskriptif dari awal (from scratch)
menggunakan Python Standard Library, mencakup:
- Mean (Rata-rata hitung)
- Median (Nilai tengah)
- Mode (Modus / nilai tersering)
- Range (Jangkauan)
- Varians Populasi (sigma^2) vs Varians Sampel (s^2)
- Standar Deviasi Populasi (sigma) vs Standar Deviasi Sampel (s)
- Kuartil (Q1, Q2, Q3) dan Interquartile Range (IQR)
- Deteksi Outlier berbasis batas Tukey (1.5 x IQR)
"""

import math
import sys
from collections import Counter
from typing import Dict, List, Tuple

sys.stdout.reconfigure(encoding="utf-8")


def calculate_mean(data: List[float]) -> float:
    """Menghitung rata-rata aritmetika: sum(x) / N."""
    if not data:
        raise ValueError("Data tidak boleh kosong.")
    return sum(data) / len(data)


def calculate_median(data: List[float]) -> float:
    """Menghitung nilai tengah setelah pengurutan."""
    if not data:
        raise ValueError("Data tidak boleh kosong.")
    sorted_data = sorted(data)
    n = len(sorted_data)
    mid = n // 2
    if n % 2 == 1:
        return sorted_data[mid]
    return (sorted_data[mid - 1] + sorted_data[mid]) / 2.0


def calculate_mode(data: List[float]) -> List[float]:
    """Menemukan modus (nilai dengan frekuensi tertinggi)."""
    if not data:
        raise ValueError("Data tidak boleh kosong.")
    counts = Counter(data)
    max_freq = max(counts.values())
    if max_freq == 1 and len(data) > 1:
        return []  # Tidak ada modus jika semua nilai unik
    return [val for val, freq in counts.items() if freq == max_freq]


def calculate_variance(data: List[float], is_sample: bool = True) -> float:
    """Menghitung varians.
    
    Jika is_sample=True, pembagi adalah (N - 1) - Koreksi Bessel.
    Jika is_sample=False, pembagi adalah N (populasi).
    """
    n = len(data)
    if is_sample and n < 2:
        raise ValueError("Sampel memerlukan minimal 2 data poin.")
    mean = calculate_mean(data)
    squared_diffs = [(x - mean) ** 2 for x in data]
    divisor = (n - 1) if is_sample else n
    return sum(squared_diffs) / divisor


def calculate_std_dev(data: List[float], is_sample: bool = True) -> float:
    """Menghitung standar deviasi sebagai akar kuadrat dari varians."""
    return math.sqrt(calculate_variance(data, is_sample=is_sample))


def calculate_quartiles(data: List[float]) -> Tuple[float, float, float]:
    """Menghitung Q1 (25%), Q2 (Median 50%), dan Q3 (75%)."""
    sorted_data = sorted(data)
    q2 = calculate_median(sorted_data)
    n = len(sorted_data)
    mid = n // 2
    if n % 2 == 0:
        lower_half = sorted_data[:mid]
        upper_half = sorted_data[mid:]
    else:
        lower_half = sorted_data[:mid]
        upper_half = sorted_data[mid + 1:]
    q1 = calculate_median(lower_half)
    q3 = calculate_median(upper_half)
    return q1, q2, q3


def detect_outliers(data: List[float]) -> Tuple[List[float], float, float]:
    """Mendeteksi pencilan menggunakan kriteria Tukey 1.5 x IQR."""
    q1, _, q3 = calculate_quartiles(data)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outliers = [x for x in data if x < lower_bound or x > upper_bound]
    return outliers, lower_bound, upper_bound


def run_descriptive_summary():
    print("=" * 70)
    print("EKSPERIMEN 1: STATISTIK DESKRIPTIF DATA SENSOR IOT (SUHU)")
    print("=" * 70)

    # Dataset pengamatan suhu (derajat Celsius) dari sensor industri IoT
    # 12 sampel pengukuran reguler ditambah 1 anomali lonjakan panas
    sensor_readings = [
        22.4, 23.1, 22.8, 24.5, 23.0, 22.8,
        25.2, 23.4, 22.8, 24.0, 23.6, 22.9,
        35.0  # Anomali lonjakan ekstrem (outlier)
    ]

    print(f"Data Mentah Pengamatan (N={len(sensor_readings)}):")
    print(f"  {sensor_readings}\n")

    mean_val = calculate_mean(sensor_readings)
    median_val = calculate_median(sensor_readings)
    modes = calculate_mode(sensor_readings)
    val_range = max(sensor_readings) - min(sensor_readings)
    pop_var = calculate_variance(sensor_readings, is_sample=False)
    sample_var = calculate_variance(sensor_readings, is_sample=True)
    pop_std = calculate_std_dev(sensor_readings, is_sample=False)
    sample_std = calculate_std_dev(sensor_readings, is_sample=True)
    q1, q2, q3 = calculate_quartiles(sensor_readings)
    iqr = q3 - q1
    outliers, low_b, up_b = detect_outliers(sensor_readings)

    print("--- 1. Ukuran Pemusatan Data (Central Tendency) ---")
    print(f"Mean (Rata-rata)        : {mean_val:.4f} °C")
    print(f"Median (Nilai Tengah)   : {median_val:.4f} °C")
    print(f"Modus (Nilai Tersering) : {modes} (Frekuensi = 3)")
    print("Catatan: Mean terangkat naik oleh nilai 35.0 °C, sedangkan Median resisten terhadap outlier!\n")

    print("--- 2. Ukuran Penyebaran Data (Dispersion) ---")
    print(f"Rentang (Range)         : {val_range:.4f} °C (Min={min(sensor_readings)}, Max={max(sensor_readings)})")
    print(f"Varians Populasi (σ²)   : {pop_var:.4f}")
    print(f"Varians Sampel (s²)     : {sample_var:.4f} (dengan Koreksi Bessel N-1)")
    print(f"Standar Deviasi Pop (σ) : {pop_std:.4f} °C")
    print(f"Standar Deviasi Samp (s): {sample_std:.4f} °C\n")

    print("--- 3. Analisis Kuartil & Deteksi Pencilan (Outliers) ---")
    print(f"Kuartil 1 (Q1 - 25%)    : {q1:.4f} °C")
    print(f"Kuartil 2 (Q2 - Median) : {q2:.4f} °C")
    print(f"Kuartil 3 (Q3 - 75%)    : {q3:.4f} °C")
    print(f"Interquartile Range(IQR): {iqr:.4f} °C")
    print(f"Batas Bawah Wajar       : {low_b:.4f} °C (Q1 - 1.5*IQR)")
    print(f"Batas Atas Wajar        : {up_b:.4f} °C (Q3 + 1.5*IQR)")
    print(f"Pencilan Terdeteksi     : {outliers} -> Data 35.0 °C berada di luar ambang batas!")
    print("=" * 70)


if __name__ == "__main__":
    run_descriptive_summary()

