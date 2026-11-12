"""Regresi Nilai Kontinu Menggunakan Decision Tree Regressor
Mendemonstrasikan pemisahan berbasis penurunan varians (MSE/Squared Error)
dan karakteristik prediksi piecewise constant.
"""

import pathlib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

def run_decision_tree_regressor():
    data_path = pathlib.Path(__file__).parent / "data" / "equipment_fault_sensor_data.csv"
    df = pd.read_csv(data_path)

    # Prediksi suhu peralatan kontinu berdasarkan beban, jam operasi, dan arus
    features = ["load_percentage", "operating_hours", "current_a", "vibration_mm_s"]
    X = df[features]
    y = df["temperature_c"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    print("=== DECISION TREE REGRESSOR ===")
    print(f"Target Kontinu: temperature_c (Rentang: {y.min():.2f} - {y.max():.2f} °C)")

    # Bandingkan kedalaman dangkal vs dalam
    for depth in [2, 4, 8, None]:
        reg = DecisionTreeRegressor(max_depth=depth, random_state=42)
        reg.fit(X_train, y_train)

        train_pred = reg.predict(X_train)
        test_pred = reg.predict(X_test)

        train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
        test_r2 = r2_score(y_test, test_pred)
        n_leaves = reg.get_n_leaves()

        depth_str = str(depth) if depth is not None else "Tanpa Batas (Deep)"
        print(f"\nKedalaman: {depth_str:<18} | Daun: {n_leaves:<3} | Train RMSE: {train_rmse:.2f} | Test RMSE: {test_rmse:.2f} | Test R²: {test_r2:.4f}")

    print("\nCatatan: DecisionTreeRegressor membagi ruang fitur menjadi wilayah ortogonal hiper-persegi;")
    print("setiap daun memprediksi nilai rata-rata sampel lokal (piecewise constant function).")

if __name__ == "__main__":
    run_decision_tree_regressor()

