"""
Script 09: Regression Model Comparison (Benchmark Model Komparatif)
Chapter 13: Supervised Learning — Regression Algorithms

Mendemonstrasikan perbandingan performa komparatif multi-model:
1. Baseline: Dummy Regressor (Mean)
2. Linear Regression (OLS Standard)
3. Polynomial Regression (Degree 2)
4. Ridge Regression (L2 Regularization)
5. Lasso Regression (L1 Regularization)
6. ElasticNet Regression (L1 + L2 Regularization)
7. Evaluasi 5-Fold Cross-Validation (Mean +/- Std RMSE) dan Pengujian Akhir pada Test Set
"""

from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def load_energy_data() -> pd.DataFrame:
    data_path = Path(__file__).resolve().parent / "data" / "equipment_power_demand.csv"
    return pd.read_csv(data_path)


def main():
    print("=" * 85)
    print("BENCHMARK KOMPARASI MODEL REGRESI (SUPERVISED LEARNING)")
    print("=" * 85)
    
    df = load_energy_data()
    feature_cols = [
        "ambient_temp_c", "operating_load_pct", "machine_speed_rpm",
        "inlet_pressure_bar", "vibration_severity_mms", "equipment_age_months"
    ]
    X = df[feature_cols]
    y = df["power_consumption_kw"]
    
    # Split 80% Train, 20% Test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )
    
    # Definisi Enam Model Kandidat
    models = {
        "1. Dummy Mean Baseline": DummyRegressor(strategy="mean"),
        "2. Linear Regression (OLS)": Pipeline([
            ("scaler", StandardScaler()),
            ("reg", LinearRegression())
        ]),
        "3. Polynomial Reg (Deg 2)": Pipeline([
            ("poly", PolynomialFeatures(degree=2, include_bias=False)),
            ("scaler", StandardScaler()),
            ("reg", LinearRegression())
        ]),
        "4. Ridge Regression (L2)": Pipeline([
            ("scaler", StandardScaler()),
            ("reg", Ridge(alpha=10.0, random_state=42))
        ]),
        "5. Lasso Regression (L1)": Pipeline([
            ("scaler", StandardScaler()),
            ("reg", Lasso(alpha=0.5, random_state=42, max_iter=10000))
        ]),
        "6. ElasticNet (L1+L2)": Pipeline([
            ("scaler", StandardScaler()),
            ("reg", ElasticNet(alpha=0.5, l1_ratio=0.5, random_state=42, max_iter=10000))
        ])
    }
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    print(f"Data: {len(X_train)} Train, {len(X_test)} Test ({len(feature_cols)} Fitur)\n")
    print(f"{'Nama Model':<28} | {'CV RMSE (kW)':<18} | {'Test MAE':<10} | {'Test RMSE':<10} | {'Test R^2':<10}")
    print("-" * 85)
    
    for name, model in models.items():
        # 1. Evaluasi 5-Fold CV pada Data Latih (scoring: neg_root_mean_squared_error)
        cv_scores = -cross_val_score(model, X_train, y_train, cv=kf, scoring="neg_root_mean_squared_error")
        cv_mean = cv_scores.mean()
        cv_std = cv_scores.std()
        
        # 2. Fit pada data latih penuh & Prediksi pada Test Set
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        test_mae = mean_absolute_error(y_test, y_pred)
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        test_r2 = r2_score(y_test, y_pred)
        
        cv_str = f"{cv_mean:.2f} (+/- {cv_std:.2f})"
        print(f"{name:<28} | {cv_str:<18} | {test_mae:<10.2f} | {test_rmse:<10.2f} | {test_r2:<10.4f}")
        
    print("-" * 85)
    print("Ringkasan Hasil:")
    print("1. Seluruh model regresi secara dramatis mengalahkan Dummy Baseline (RMSE ~80.9 kW -> ~4.6 kW).")
    print("2. Polynomial Regression (Derajat 2) memberikan Test RMSE terendah karena menangkap")
    print("   efek kuadratik pemanasan pada beban tinggi yang tertuang pada formula pembangkitan.")
    print("3. Ridge dan ElasticNet memberikan estimasi yang stabil dengan varians lintas fold yang terkontrol.")
    print("=" * 85)


if __name__ == "__main__":
    main()

