"""
Script 04: Ridge Regression (Regularisasi L2 & Coefficient Shrinkage)
Chapter 13: Supervised Learning — Regression Algorithms

Mendemonstrasikan pengendalian kompleksitas model menggunakan Regularisasi L2 (Ridge):
1. Fungsi Tujuan: Loss = MSE + alpha * ||w||_2^2
2. Pentingnya penskalaan fitur (StandardScaler) sebelum regularisasi
3. Pengaruh nilai hiperparameter alpha terhadap penyusutan koefisien (shrinkage)
4. Pengurangan varians model tanpa memusnahkan fitur (koefisien mengecil tapi tidak tepat 0)
"""

from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score


def load_energy_data() -> pd.DataFrame:
    data_path = Path(__file__).resolve().parent / "data" / "equipment_power_demand.csv"
    return pd.read_csv(data_path)


def main():
    print("=" * 75)
    print("DEMONSTRASI RIDGE REGRESSION (REGULARISASI L2)")
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
    
    alphas = [0.001, 1.0, 50.0, 500.0]
    
    print(f"Data: {len(X_train)} Train, {len(X_test)} Test ({len(feature_cols)} Fitur)")
    print("StandardScaler diterapkan di dalam Pipeline agar bebas kebocoran statistik.\n")
    
    print(f"{'Alpha':<10} | {'||w||_2 (Norm L2)':<18} | {'Test RMSE (kW)':<16} | {'Test R^2':<10} | {'Koefisien Tertinggi':<20}")
    print("-" * 80)
    
    coef_history = {}
    
    for a in alphas:
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=a, random_state=42))
        ])
        pipe.fit(X_train, y_train)
        
        coefs = pipe.named_steps["ridge"].coef_
        l2_norm = np.linalg.norm(coefs)
        test_pred = pipe.predict(X_test)
        test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
        test_r2 = r2_score(y_test, test_pred)
        
        max_idx = np.argmax(np.abs(coefs))
        max_feat_str = f"{feature_cols[max_idx]} ({coefs[max_idx]:.2f})"
        coef_history[a] = coefs
        
        print(f"{a:<10.3f} | {l2_norm:<18.4f} | {test_rmse:<16.2f} | {test_r2:<10.4f} | {max_feat_str:<20}")
        
    print("-" * 80)
    print("\nDETAIL PENYUSUTAN KOEFISIEN (COEFFICIENT SHRINKAGE):")
    coef_df = pd.DataFrame(coef_history, index=feature_cols)
    coef_df.columns = [f"alpha={a}" for a in alphas]
    print(coef_df.round(4).to_string())
    print("\nKesimpulan:")
    print("1. Semakin besar alpha, norma L2 (||w||_2) semakin menyusut drastis menuju nol.")
    print("2. Semua koefisien mengecil secara halus, tetapi TIDAK ADA yang menjadi tepat nol.")
    print("=" * 75)


if __name__ == "__main__":
    main()

