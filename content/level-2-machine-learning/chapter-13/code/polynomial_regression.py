"""
Script 03: Polynomial Regression (Regresi Polinomial & Non-Linearitas)
Chapter 13: Supervised Learning — Regression Algorithms

Mendemonstrasikan penanganan hubungan non-linear melalui PolynomialFeatures:
1. Menunjukkan bahwa Polynomial Regression tetap merupakan model linier terhadap parameter (w),
   meskipun non-linear terhadap fitur input (x).
2. Membandingkan Derajat 1 (Linier Sederhana), Derajat 2 (Kuadratik Nyata), dan Derajat 4.
3. Mengukur trade-off antara bias dan varians (Train vs Test RMSE).
"""

from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score


def load_energy_data() -> pd.DataFrame:
    data_path = Path(__file__).resolve().parent / "data" / "equipment_power_demand.csv"
    return pd.read_csv(data_path)


def main():
    print("=" * 75)
    print("DEMONSTRASI POLYNOMIAL REGRESSION (DERAJAT 1, 2, DAN 4)")
    print("=" * 75)
    
    df = load_energy_data()
    # Fokus pada fitur beban operasi dan kecepatan mesin yang memiliki interaksi kuadratik
    feature_cols = ["operating_load_pct", "machine_speed_rpm", "ambient_temp_c"]
    X = df[feature_cols]
    y = df["power_consumption_kw"]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )
    
    degrees = [1, 2, 4]
    results = []
    
    print(f"Data: {len(X_train)} Train, {len(X_test)} Test ({len(feature_cols)} Fitur Dasar)\n")
    print(f"{'Derajat':<8} | {'Jumlah Fitur Poly':<18} | {'Train RMSE (kW)':<16} | {'Test RMSE (kW)':<16} | {'Test R^2':<10}")
    print("-" * 75)
    
    for deg in degrees:
        # PENTING: Pipeline menjamin PolynomialFeatures & StandardScaler fit strictly pada data latih!
        pipe = Pipeline([
            ("poly", PolynomialFeatures(degree=deg, include_bias=False)),
            ("scaler", StandardScaler()),
            ("linear", LinearRegression())
        ])
        
        pipe.fit(X_train, y_train)
        
        n_features_poly = pipe.named_steps["poly"].n_output_features_
        train_pred = pipe.predict(X_train)
        test_pred = pipe.predict(X_test)
        
        tr_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
        ts_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
        ts_r2 = r2_score(y_test, test_pred)
        
        results.append((deg, n_features_poly, tr_rmse, ts_rmse, ts_r2))
        print(f"Derajat {deg:<2} | {n_features_poly:<18} | {tr_rmse:<16.2f} | {ts_rmse:<16.2f} | {ts_r2:<10.4f}")
        
    print("-" * 75)
    print("Analisis Matematis:")
    print("1. Derajat 1 mengasumsikan efisiensi beban linier murni (underfitting pada non-linearitas halus).")
    print("2. Derajat 2 menangkap suku kuadratik pemanasan termal, menurunkan Test RMSE secara signifikan.")
    print("3. Derajat 4 memperluas dimensi fitur secara masif (34 fitur); Train RMSE turun tetapi")
    print("   Test RMSE mulai rentan variansi tinggi jika tidak di-regularisasi.")
    print("=" * 75)


if __name__ == "__main__":
    main()

