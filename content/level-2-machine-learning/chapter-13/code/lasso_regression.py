"""
Script 05: Lasso Regression (Regularisasi L1 & Sparse Feature Selection)
Chapter 13: Supervised Learning — Regression Algorithms

Mendemonstrasikan Regularisasi L1 (Lasso) dan sifat sparsity:
1. Fungsi Tujuan: Loss = (1 / 2n) * ||y - Xw||_2^2 + alpha * ||w||_1
2. Geometri belah ketupat L1 yang memaksa koefisien fitur yang kurang informatif menjadi tepat 0
3. Seleksi fitur otomatis terintegrasi (embedded feature selection)
4. Perbandingan jumlah fitur aktif lintas nilai alpha
"""

from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score


def load_energy_data() -> pd.DataFrame:
    data_path = Path(__file__).resolve().parent / "data" / "equipment_power_demand.csv"
    return pd.read_csv(data_path)


def main():
    print("=" * 75)
    print("DEMONSTRASI LASSO REGRESSION (REGULARISASI L1 & SPARSITY)")
    print("=" * 75)
    
    df = load_energy_data()
    feature_cols = [
        "ambient_temp_c", "operating_load_pct", "machine_speed_rpm",
        "inlet_pressure_bar", "vibration_severity_mms", "equipment_age_months"
    ]
    X = df[feature_cols]
    y = df["power_consumption_kw"]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )
    
    alphas = [0.05, 0.5, 2.0, 8.0]
    
    print(f"Data: {len(X_train)} Train, {len(X_test)} Test ({len(feature_cols)} Fitur)\n")
    print(f"{'Alpha':<10} | {'Fitur Aktif (w!=0)':<20} | {'||w||_1 (Norm L1)':<18} | {'Test RMSE (kW)':<16} | {'Test R^2':<10}")
    print("-" * 80)
    
    coef_history = {}
    
    for a in alphas:
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("lasso", Lasso(alpha=a, random_state=42, max_iter=10000))
        ])
        pipe.fit(X_train, y_train)
        
        coefs = pipe.named_steps["lasso"].coef_
        active_count = np.sum(coefs != 0)
        l1_norm = np.sum(np.abs(coefs))
        
        test_pred = pipe.predict(X_test)
        test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
        test_r2 = r2_score(y_test, test_pred)
        
        coef_history[a] = coefs
        print(f"{a:<10.3f} | {f'{active_count} dari {len(feature_cols)}':<20} | {l1_norm:<18.4f} | {test_rmse:<16.2f} | {test_r2:<10.4f}")
        
    print("-" * 80)
    print("\nDETAIL KOEFISIEN FITUR (PERHATIKAN NILAI 0.0000 PADA LASSO):")
    coef_df = pd.DataFrame(coef_history, index=feature_cols)
    coef_df.columns = [f"alpha={a}" for a in alphas]
    print(coef_df.round(4).to_string())
    
    print("\nAnalisis Kunci:")
    print("1. Pada alpha moderat (0.5), fitur dengan kontribusi kecil dimatikan menjadi 0.0000.")
    print("2. Lasso bertindak sebagai mekanisme seleksi fitur otomatis (embedded feature selection).")
    print("3. Berbeda dengan Ridge yang hanya mengecilkan bobot, Lasso mengeliminasi fitur secara absolut.")
    print("=" * 75)


if __name__ == "__main__":
    main()

