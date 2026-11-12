"""
Script 05: Numeric Preprocessing (Imputation & Scaling Without Leakage)
Chapter 12: ML Workflow, Dataset & Feature Engineering

Mendemonstrasikan pra-pemrosesan data numerik yang benar:
1. Imputasi nilai hilang (SimpleImputer dengan median).
2. Penskalaan fitur (StandardScaler, MinMaxScaler, RobustScaler).
3. Aturan Emas: 'fit' HANYA pada data latih (Train Set),
   kemudian 'transform' pada Train, Val, dan Test Set.
"""

from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler


def load_telemetry_data() -> pd.DataFrame:
    data_path = Path(__file__).resolve().parent / "data" / "equipment_telemetry.csv"
    return pd.read_csv(data_path)


def main():
    print("=" * 75)
    print("DEMONSTRASI PRA-PEMROSESAN FITUR NUMERIK (IMPUTASI & SKALASI)")
    print("=" * 75)
    
    df = load_telemetry_data()
    num_cols = ["operating_hours", "vibration_rms", "temperature_c", "pressure_kpa"]
    
    # 1. Split train dan test terlebih dahulu
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        df[num_cols], df["failure"], test_size=0.20, random_state=42, stratify=df["failure"]
    )
    
    print("1. KONDISI AWAL MISSING VALUES:")
    print(f"   Train missing:\n{X_train_raw.isna().sum().to_dict()}")
    print(f"   Test missing :\n{X_test_raw.isna().sum().to_dict()}\n")
    
    # 2. Imputasi Nilai Hilang (Fit pada train saja)
    imputer = SimpleImputer(strategy="median")
    imputer.fit(X_train_raw)
    
    print("2. PARAMETER YANG DIPELAJARI DARI TRAIN (MEDIAN):")
    for col, med in zip(num_cols, imputer.statistics_):
        print(f"   - {col:<16}: median = {med:.2f}")
        
    X_train_imp = pd.DataFrame(imputer.transform(X_train_raw), columns=num_cols, index=X_train_raw.index)
    X_test_imp = pd.DataFrame(imputer.transform(X_test_raw), columns=num_cols, index=X_test_raw.index)
    
    print("\n   Verifikasi Missing Setelah Imputasi:")
    print(f"   Train missing: {X_train_imp.isna().sum().sum()} | Test missing: {X_test_imp.isna().sum().sum()}\n")
    
    # 3. Penskalaan Fitur: Perbandingan StandardScaler, MinMaxScaler, RobustScaler
    scaler_std = StandardScaler()
    scaler_std.fit(X_train_imp)
    
    scaler_rob = RobustScaler()
    scaler_rob.fit(X_train_imp)
    
    scaler_minmax = MinMaxScaler()
    scaler_minmax.fit(X_train_imp)
    
    X_train_scaled = scaler_std.transform(X_train_imp)
    X_test_scaled = scaler_std.transform(X_test_imp)
    
    print("3. STATISTIK STANDARD SCALER SETELAH TRANSFORMASI:")
    print(f"{'Fitur':<16} | {'Train Mean':<12} | {'Train Std':<12} | {'Test Mean':<12} | {'Test Std':<12}")
    print("-" * 75)
    for i, col in enumerate(num_cols):
        tr_mean = X_train_scaled[:, i].mean()
        tr_std = X_train_scaled[:, i].std()
        ts_mean = X_test_scaled[:, i].mean()
        ts_std = X_test_scaled[:, i].std()
        print(f"{col:<16} | {tr_mean:<12.4f} | {tr_std:<12.4f} | {ts_mean:<12.4f} | {ts_std:<12.4f}")
        
    print("-" * 75)
    print("Catatan: Train mean tepat 0.0000 dan std tepat 1.0000.")
    print("Test set memiliki mean/std mendekati train, merefleksikan variasi distribusi nyata!")
    print("=" * 75)


if __name__ == "__main__":
    main()

