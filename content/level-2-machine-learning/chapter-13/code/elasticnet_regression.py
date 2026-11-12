"""
Script 06: ElasticNet Regression (Kombinasi L1 dan L2 Regularisasi)
Chapter 13: Supervised Learning — Regression Algorithms

Mendemonstrasikan ElasticNet sebagai kombinasi konveks penalti L1 (Lasso) dan L2 (Ridge):
1. Formula Penalti: alpha * l1_ratio * ||w||_1 + 0.5 * alpha * (1 - l1_ratio) * ||w||_2^2
2. Mengatasi kelemahan Lasso saat terdapat fitur-fitur berkorelasi tinggi (collinear features)
3. Evaluasi pengaruh parameter l1_ratio (0.1 mendekati Ridge, 0.9 mendekati Lasso)
"""

from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score


def load_energy_data() -> pd.DataFrame:
    data_path = Path(__file__).resolve().parent / "data" / "equipment_power_demand.csv"
    return pd.read_csv(data_path)


def main():
    print("=" * 75)
    print("DEMONSTRASI ELASTICNET REGRESSION (L1 + L2 PENALTY)")
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
    
    # Memvariasikan l1_ratio pada nilai alpha tetap
    alpha_fixed = 0.5
    l1_ratios = [0.1, 0.5, 0.8, 0.95]
    
    print(f"Data: {len(X_train)} Train, {len(X_test)} Test ({len(feature_cols)} Fitur)")
    print(f"Konfigurasi: Alpha tetap = {alpha_fixed}\n")
    print(f"{'l1_ratio':<10} | {'Fitur Aktif':<14} | {'Test RMSE (kW)':<16} | {'Test R^2':<10} | {'Karakteristik Dominan':<22}")
    print("-" * 80)
    
    coef_history = {}
    
    for r in l1_ratios:
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("elastic", ElasticNet(alpha=alpha_fixed, l1_ratio=r, random_state=42, max_iter=10000))
        ])
        pipe.fit(X_train, y_train)
        
        coefs = pipe.named_steps["elastic"].coef_
        active_count = np.sum(coefs != 0)
        
        test_pred = pipe.predict(X_test)
        test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
        test_r2 = r2_score(y_test, test_pred)
        
        coef_history[f"ratio={r}"] = coefs
        dominance = "Dominan Ridge (L2)" if r < 0.3 else ("Seimbang L1/L2" if r <= 0.7 else "Dominan Lasso (L1)")
        
        print(f"{r:<10.2f} | {f'{active_count}/{len(feature_cols)}':<14} | {test_rmse:<16.2f} | {test_r2:<10.4f} | {dominance:<22}")
        
    print("-" * 80)
    print("\nPERBANDINGAN KOEFISIEN BERDASARKAN l1_ratio:")
    coef_df = pd.DataFrame(coef_history, index=feature_cols)
    print(coef_df.round(4).to_string())
    
    print("\nKeunggulan Praktis ElasticNet:")
    print("1. Pada l1_ratio rendah, model mempertahankan fitur berkorelasi secara stabil (keuntungan Ridge).")
    print("2. Pada l1_ratio tinggi, model meniadakan fitur tidak relevan menjadi 0 (keuntungan Lasso).")
    print("3. ElasticNet adalah jalan tengah terbaik saat menghadapi multikolinearitas moderat.")
    print("=" * 75)


if __name__ == "__main__":
    main()

