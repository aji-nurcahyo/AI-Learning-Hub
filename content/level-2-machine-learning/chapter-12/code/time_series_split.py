"""
Script 04: Time-Series Splitting (Preventing Forward-Looking Leakage)
Chapter 12: ML Workflow, Dataset & Feature Engineering

Mendemonstrasikan pemisahan data berbasis urutan waktu (chronological split).
Pada data time-series/sensor sequential, pengacakan acak (random shuffle)
menyebabkan Look-Ahead Bias (data masa depan bocor untuk memprediksi masa lalu).
TimeSeriesSplit memastikan setiap fold hanya melatih model pada data masa lalu
dan mengevaluasinya pada masa depan (expanding window).
"""

from pathlib import Path
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit, train_test_split


def load_telemetry_data() -> pd.DataFrame:
    data_path = Path(__file__).resolve().parent / "data" / "equipment_telemetry.csv"
    df = pd.read_csv(data_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def demo_lookahead_leakage(df: pd.DataFrame):
    print("--- 1. RISIKO LOOK-AHEAD BIAS PADA RANDOM SHUFFLE SPLIT ---")
    
    # Random split
    train_rand, test_rand = train_test_split(df, test_size=0.20, shuffle=True, random_state=42)
    
    min_train = train_rand["timestamp"].min()
    max_train = train_rand["timestamp"].max()
    min_test = test_rand["timestamp"].min()
    max_test = test_rand["timestamp"].max()
    
    print("A. Random Split (Shuffle=True):")
    print(f"   Rentang Waktu Training : {min_train} s/d {max_train}")
    print(f"   Rentang Waktu Testing  : {min_test} s/d {max_test}")
    
    # Cek berapa banyak data test yang terjadi SEBELUM titik akhir data training
    future_leak = test_rand[test_rand["timestamp"] < max_train]
    print(f"   Jumlah Sampel Test Sebelum Max Train : {len(future_leak)}/{len(test_rand)} ({len(future_leak)/len(test_rand)*100:.1f}%)")
    print("   -> BENCANA METODOLOGIS: Model dilatih dengan informasi yang terjadi SETELAH data uji!")
    
    # Chronological Split
    split_idx = int(len(df) * 0.80)
    train_chrono = df.iloc[:split_idx]
    test_chrono = df.iloc[split_idx:]
    
    print("\nB. Chronological Split (Cut-off berbasis waktu):")
    print(f"   Rentang Waktu Training : {train_chrono['timestamp'].min()} s/d {train_chrono['timestamp'].max()}")
    print(f"   Rentang Waktu Testing  : {test_chrono['timestamp'].min()} s/d {test_chrono['timestamp'].max()}")
    print(f"   Max Train < Min Test   : {train_chrono['timestamp'].max() < test_chrono['timestamp'].min()} (Bebas Leakage Temporal!)\n")


def demo_time_series_cv(df: pd.DataFrame, n_splits: int = 5):
    print(f"--- 2. {n_splits}-FOLD TIME SERIES SPLIT (EXPANDING WINDOW) ---")
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    print(f"{'Fold':<6} | {'Train Range':<44} | {'Train N':<8} | {'Test Range':<44} | {'Test N':<8}")
    print("-" * 115)
    
    for fold, (train_idx, test_idx) in enumerate(tscv.split(df)):
        t_train_start = df.iloc[train_idx[0]]["timestamp"]
        t_train_end = df.iloc[train_idx[-1]]["timestamp"]
        t_test_start = df.iloc[test_idx[0]]["timestamp"]
        t_test_end = df.iloc[test_idx[-1]]["timestamp"]
        
        # Validasi temporal
        assert t_train_end <= t_test_start, "Lookahead violation terdeteksi!"
        
        train_str = f"{t_train_start.strftime('%m-%d %H:%M')} -> {t_train_end.strftime('%m-%d %H:%M')}"
        test_str = f"{t_test_start.strftime('%m-%d %H:%M')} -> {t_test_end.strftime('%m-%d %H:%M')}"
        
        print(f"Fold {fold+1:<2} | {train_str:<44} | {len(train_idx):<8} | {test_str:<44} | {len(test_idx):<8}")
        
    print("-" * 115)
    print("Prinsip: Model selalu dievaluasi terhadap masa depan yang belum terjadi pada data latih.")


def main():
    print("=" * 75)
    print("DEMONSTRASI TIME-SERIES SPLITTING & LOOK-AHEAD PREVENTION")
    print("=" * 75)
    df = load_telemetry_data()
    demo_lookahead_leakage(df)
    demo_time_series_cv(df)
    print("=" * 75)


if __name__ == "__main__":
    main()

