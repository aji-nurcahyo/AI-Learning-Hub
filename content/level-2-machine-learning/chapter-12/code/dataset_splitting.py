"""
Script 01: Dataset Splitting (Train, Validation, Test 3-Way Split)
Chapter 12: ML Workflow, Dataset & Feature Engineering

Mendemonstrasikan pembagian dataset menjadi tiga partisi:
1. Training Set (60%): Untuk melatih parameter internal model.
2. Validation Set (20%): Untuk tuning hyperparameter dan seleksi model.
3. Test Set (20%): Evaluasi final generalisasi model (unseen data).
"""

from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split


def load_telemetry_data() -> pd.DataFrame:
    data_path = Path(__file__).resolve().parent / "data" / "equipment_telemetry.csv"
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset tidak ditemukan di {data_path}. Jalankan generate_dataset.py terlebih dahulu.")
    return pd.read_csv(data_path)


def three_way_split(df: pd.DataFrame, train_ratio: float = 0.60, val_ratio: float = 0.20, test_ratio: float = 0.20, random_state: int = 42):
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Total rasio split harus tepat 1.0"
    
    # Tahap 1: Pisahkan test set terlebih dahulu
    # Test set rasio terhadap total = test_ratio
    train_val_df, test_df = train_test_split(
        df,
        test_size=test_ratio,
        random_state=random_state
    )
    
    # Tahap 2: Pisahkan train dan validation dari sisanya
    # Rasio val di dalam train_val = val_ratio / (train_ratio + val_ratio)
    val_rel_ratio = val_ratio / (train_ratio + val_ratio)
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_rel_ratio,
        random_state=random_state
    )
    
    return train_df, val_df, test_df


def main():
    print("=" * 70)
    print("DEMONSTRASI DATASET SPLITTING 3-WAY (TRAIN / VAL / TEST)")
    print("=" * 70)
    
    df = load_telemetry_data()
    n_total = len(df)
    print(f"Total Sampel Dataset Asli : {n_total} baris\n")
    
    train_df, val_df, test_df = three_way_split(df, 0.60, 0.20, 0.20, random_state=42)
    
    print("Hasil Pembagian Partisi:")
    print(f"  * Training Set   : {len(train_df):>3} sampel ({len(train_df)/n_total*100:>5.1f}%)")
    print(f"  * Validation Set : {len(val_df):>3} sampel ({len(val_df)/n_total*100:>5.1f}%)")
    print(f"  * Test Set       : {len(test_df):>3} sampel ({len(test_df)/n_total*100:>5.1f}%)")
    print(f"  * Total Cek      : {len(train_df) + len(val_df) + len(test_df)} sampel")
    
    # Verifikasi integritas indeks (tidak ada overlap)
    idx_train = set(train_df.index)
    idx_val = set(val_df.index)
    idx_test = set(test_df.index)
    
    overlap = (idx_train & idx_val) | (idx_train & idx_test) | (idx_val & idx_test)
    print(f"\nIntegritas Data:")
    print(f"  * Jumlah Data Tumpang Tindih (Overlap) : {len(overlap)} (Harus 0)")
    assert len(overlap) == 0, "Ditemukan tumpang tindih data antar partisi!"
    
    # Periksa distribusi target pada tiap split
    print("\nDistribusi Target (failure=1):")
    print(f"  * Dataset Penuh  : {df['failure'].sum():>2}/{n_total} ({df['failure'].mean()*100:>5.2f}%)")
    print(f"  * Training Set   : {train_df['failure'].sum():>2}/{len(train_df)} ({train_df['failure'].mean()*100:>5.2f}%)")
    print(f"  * Validation Set : {val_df['failure'].sum():>2}/{len(val_df)} ({val_df['failure'].mean()*100:>5.2f}%)")
    print(f"  * Test Set       : {test_df['failure'].sum():>2}/{len(test_df)} ({test_df['failure'].mean()*100:>5.2f}%)")
    print("=" * 70)


if __name__ == "__main__":
    main()

