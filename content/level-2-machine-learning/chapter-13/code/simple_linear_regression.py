"""
Script 01: Simple Linear Regression (Regresi Linier Sederhana)
Chapter 13: Supervised Learning — Regression Algorithms

Mendemonstrasikan pemodelan hubungan linier antara satu variabel independen
(operating_load_pct) dengan target kontinu (power_consumption_kw):
1. Persamaan garis: y_hat = w * x + b
2. Estimasi parameter Ordinary Least Squares (OLS)
3. Ekstraksi slope coefficient (w) dan intercept (b)
4. Perhitungan galat residual secara manual dan via Scikit-learn
"""

from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


def load_energy_data() -> pd.DataFrame:
    data_path = Path(__file__).resolve().parent / "data" / "equipment_power_demand.csv"
    return pd.read_csv(data_path)


def main():
    print("=" * 75)
    print("DEMONSTRASI SIMPLE LINEAR REGRESSION (OLS)")
    print("=" * 75)
    
    df = load_energy_data()
    X = df[["operating_load_pct"]]
    y = df["power_consumption_kw"]
    
    # 1. Train / Test Split (80% Train, 20% Test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )
    
    print(f"Data Latih : {len(X_train)} sampel")
    print(f"Data Uji   : {len(X_test)} sampel\n")
    
    # 2. Inisialisasi dan Fitting Model OLS
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    slope = model.coef_[0]
    intercept = model.intercept_
    
    print("HASIL PEMODELAN MATEMATIKA:")
    print(f"  * Slope Coefficient (w) : {slope:.4f} kW / % beban")
    print(f"  * Intercept (b)         : {intercept:.4f} kW")
    print(f"  * Persamaan Garis       : Power = {slope:.4f} * Load + {intercept:.4f}\n")
    
    # 3. Prediksi dan Perhitungan Residual
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    residuals_test = y_test - y_pred_test
    
    # 4. Evaluasi Metrik
    train_mse = mean_squared_error(y_train, y_pred_train)
    test_mse = mean_squared_error(y_test, y_pred_test)
    test_r2 = r2_score(y_test, y_pred_test)
    
    print("EVALUASI KINERJA MODEL:")
    print(f"  * Train MSE             : {train_mse:.4f} kW^2 (RMSE = {np.sqrt(train_mse):.2f} kW)")
    print(f"  * Test MSE              : {test_mse:.4f} kW^2 (RMSE = {np.sqrt(test_mse):.2f} kW)")
    print(f"  * Test R^2 Score        : {test_r2:.4f} ({test_r2*100:.2f}% varians dijelaskan)")
    print(f"  * Rata-rata Residual    : {residuals_test.mean():.4f} kW (Mendekati nol)")
    print(f"  * Standar Dev Residual  : {residuals_test.std():.4f} kW\n")
    
    # Contoh inferensi operasional
    test_loads = [25.0, 50.0, 75.0, 90.0]
    preds = model.predict(pd.DataFrame({"operating_load_pct": test_loads}))
    print("CONTOH PREDIKSI OPERASIONAL:")
    for load, p in zip(test_loads, preds):
        print(f"  - Beban {load:>4.1f}% -> Prediksi Konsumsi Daya: {p:>6.2f} kW")
    print("=" * 75)


if __name__ == "__main__":
    main()

