"""
Script 02: Multiple Linear Regression (Regresi Linier Berganda)
Chapter 13: Supervised Learning — Regression Algorithms

Mendemonstrasikan regresi linier multivariat dengan 6 fitur operasional:
1. Formulasi: y_hat = w_1*x_1 + w_2*x_2 + ... + w_d*x_d + b
2. Interpretasi bobot koefisien parsial (partial slope coefficients)
3. Evaluasi terhadap Baseline Mean Dummy Regressor
4. Pengukuran kenaikan R^2 dan penurunan RMSE
"""

from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def load_energy_data() -> pd.DataFrame:
    data_path = Path(__file__).resolve().parent / "data" / "equipment_power_demand.csv"
    return pd.read_csv(data_path)


def main():
    print("=" * 75)
    print("DEMONSTRASI MULTIPLE LINEAR REGRESSION")
    print("=" * 75)
    
    df = load_energy_data()
    feature_cols = [
        "ambient_temp_c", "operating_load_pct", "machine_speed_rpm",
        "inlet_pressure_bar", "vibration_severity_mms", "equipment_age_months"
    ]
    target_col = "power_consumption_kw"
    
    X = df[feature_cols]
    y = df[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )
    
    # 1. Baseline Model (Prediksi Rata-rata Pelatihan)
    baseline = DummyRegressor(strategy="mean")
    baseline.fit(X_train, y_train)
    y_pred_base = baseline.predict(X_test)
    
    # 2. Multiple Linear Regression Model
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred_mlr = model.predict(X_test)
    
    print(f"Data: {len(X_train)} Train, {len(X_test)} Test ({len(feature_cols)} Fitur)\n")
    print("KOEFISIEN REGRESI BERGANDA TERESTIMASI:")
    print(f"{'Nama Fitur':<26} | {'Koefisien (w)':<16} | {'Satuan Efek Marginal':<25}")
    print("-" * 75)
    for col, coef in zip(feature_cols, model.coef_):
        print(f"{col:<26} | {coef:>14.4f}   | kW per satuan {col}")
    print(f"{'Intercept (b)':<26} | {model.intercept_:>14.4f}   | kW (baseline tanpa beban)")
    print("-" * 75)
    
    # 3. Komparasi Terhadap Baseline
    print("\nPERBANDINGAN TERHADAP DUMMY MEAN BASELINE:")
    base_mae = mean_absolute_error(y_test, y_pred_base)
    base_rmse = np.sqrt(mean_squared_error(y_test, y_pred_base))
    base_r2 = r2_score(y_test, y_pred_base)
    
    mlr_mae = mean_absolute_error(y_test, y_pred_mlr)
    mlr_rmse = np.sqrt(mean_squared_error(y_test, y_pred_mlr))
    mlr_r2 = r2_score(y_test, y_pred_mlr)
    
    print(f"{'Model':<24} | {'MAE (kW)':<10} | {'RMSE (kW)':<10} | {'R^2 Score':<12}")
    print("-" * 62)
    print(f"{'1. Dummy Mean Baseline':<24} | {base_mae:<10.2f} | {base_rmse:<10.2f} | {base_r2:<12.4f}")
    print(f"{'2. Multiple Linear Reg':<24} | {mlr_mae:<10.2f} | {mlr_rmse:<10.2f} | {mlr_r2:<12.4f}")
    print("-" * 62)
    print(f"Penurunan Error RMSE : {(base_rmse - mlr_rmse) / base_rmse * 100:.2f}%")
    print(f"Proporsi Varians     : {mlr_r2*100:.2f}% dijelaskan oleh model regresi berganda.")
    print("=" * 75)


if __name__ == "__main__":
    main()

