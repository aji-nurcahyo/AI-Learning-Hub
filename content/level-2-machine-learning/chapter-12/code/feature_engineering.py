"""
Script 07: Feature Engineering (Domain, Datetime, & Interaction Terms)
Chapter 12: ML Workflow, Dataset & Feature Engineering

Mendemonstrasikan rekayasa fitur (feature engineering) untuk meningkatkan sinyal prediktif:
1. Fitur Domain: Rasio vibrasi terhadap tekanan, laju kenaikan suhu per jam kerja.
2. Fitur Temporal / Datetime: Jam (hour), hari dalam minggu (dayofweek), indikator akhir pekan (is_weekend).
3. Suku Interaksi (Interaction Terms): Perkalian non-linear antara getaran dan temperatur.
"""

from pathlib import Path
import pandas as pd
import numpy as np


def load_telemetry_data() -> pd.DataFrame:
    data_path = Path(__file__).resolve().parent / "data" / "equipment_telemetry.csv"
    df = pd.read_csv(data_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df_out = df.copy()
    
    # 1. Fitur Temporal / Kalender
    df_out["hour"] = df_out["timestamp"].dt.hour
    df_out["dayofweek"] = df_out["timestamp"].dt.dayofweek
    df_out["is_weekend"] = (df_out["dayofweek"] >= 5).astype(int)
    
    # 2. Fitur Domain Engineering
    # Seringkali kegagalan mesin bukan sekadar suhu mutlak, melainkan beban suhu relatif terhadap jam operasional
    df_out["temp_per_1000h"] = df_out["temperature_c"] / (df_out["operating_hours"] / 1000.0 + 1e-5)
    
    # Rasio stres mekanik: vibrasi per satuan tekanan
    df_out["vibration_pressure_ratio"] = df_out["vibration_rms"] / (df_out["pressure_kpa"] + 1e-5)
    
    # 3. Interaksi Non-Linear Fitur Kritis
    # Kerusakan termal dan mekanis terjadi bersamaan
    df_out["thermal_mechanical_index"] = df_out["vibration_rms"] * df_out["temperature_c"]
    
    # Indikator Ambang Batas Operasional Kritis (Domain Flag)
    df_out["critical_heat_warning"] = ((df_out["temperature_c"] > 75.0) & (df_out["warning_flag"] == "HIGH")).astype(int)
    
    return df_out


def main():
    print("=" * 75)
    print("DEMONSTRASI FEATURE ENGINEERING: DOMAIN, TEMPORAL & INTERACTION")
    print("=" * 75)
    
    df = load_telemetry_data()
    print(f"Dimensi Awal Dataset: {df.shape[0]} baris x {df.shape[1]} kolom")
    
    df_eng = engineer_features(df)
    print(f"Dimensi Setelah Rekayasa: {df_eng.shape[0]} baris x {df_eng.shape[1]} kolom\n")
    
    engineered_cols = [
        "hour", "is_weekend", "temp_per_1000h", 
        "vibration_pressure_ratio", "thermal_mechanical_index", "critical_heat_warning"
    ]
    
    print("Contoh 5 Baris Fitur Baru yang Direkayasa:")
    print(df_eng[engineered_cols].head())
    
    # Analisis Korelasi terhadap Target Failure
    print("\nKorelasi Pearson Fitur terhadap Target 'failure':")
    corrs = df_eng.corr(numeric_only=True)["failure"].sort_values(ascending=False)
    
    print(f"{'Nama Fitur':<30} | {'Korelasi dengan Failure':<25}")
    print("-" * 58)
    for feat, corr_val in corrs.items():
        if feat != "failure":
            flag = " (FITUR BARU)" if feat in engineered_cols else " (ORIGINAL)"
            print(f"{feat + flag:<30} | {corr_val:<25.4f}")
            
    print("-" * 58)
    print("Catatan: Fitur interaksi 'thermal_mechanical_index' dan 'critical_heat_warning'")
    print("menghasilkan korelasi kuat yang memperjelas sinyal bagi model klasifikasi.")
    print("=" * 75)


if __name__ == "__main__":
    main()

