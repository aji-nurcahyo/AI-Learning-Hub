"""Tahap 2 Proyek Capstone: Rekayasa Fitur & Pipeline Prapemrosesan Bebas Bocor
Menghasilkan fitur turunan fisik domain mesin dan membungkusnya dalam Scikit-Learn Pipeline.
"""

import pathlib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

class IndustrialFeatureEngineer(BaseEstimator, TransformerMixin):
    """Transformer rekayasa fitur domain pemeliharaan prediktif mesin rotasi."""
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Asumsi indeks kolom X sesuai urutan:
        # 0: vibration, 1: temp, 2: rpm, 3: pressure, 4: current, 5: acidity
        X_arr = np.asarray(X)
        vibration = X_arr[:, 0]
        temp = X_arr[:, 1]
        rpm = X_arr[:, 2]
        pressure = np.maximum(X_arr[:, 3], 0.1)
        current = X_arr[:, 4]
        acidity = X_arr[:, 5]

        # 1. Rasio Beban Termal-Tekanan Hidrolik
        thermal_pressure_ratio = temp / pressure

        # 2. Indeks Energi Getaran Mekanis
        vibration_power_index = (vibration ** 2) * (rpm / 1000.0)

        # 3. Indeks Stres Arus Listrik vs Pelumasan
        lubricant_stress_index = current * acidity

        new_features = np.column_stack([
            X_arr,
            thermal_pressure_ratio,
            vibration_power_index,
            lubricant_stress_index
        ])
        return new_features

def build_preprocessing_pipeline():
    return Pipeline([
        ("feature_engineering", IndustrialFeatureEngineer()),
        ("scaler", StandardScaler())
    ])

def test_preprocessing():
    data_path = pathlib.Path(__file__).parent / "data" / "predictive_maintenance_telemetry.csv"
    df = pd.read_csv(data_path)

    raw_features = [
        "vibration_rms_g",
        "bearing_temperature_c",
        "rotational_speed_rpm",
        "hydraulic_pressure_bar",
        "motor_current_amp",
        "oil_acidity_tan"
    ]
    X = df[raw_features].values
    y = df["machine_failure"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    pipe = build_preprocessing_pipeline()
    X_train_proc = pipe.fit_transform(X_train)
    X_test_proc = pipe.transform(X_test)

    print("======================================================================")
    print("CAPSTONE TAHAP 2: REKAYASA FITUR & PIPELINE PRAPEMROSESAN")
    print("======================================================================")
    print(f"Dimensi Fitur Awal (Raw)   : {X_train.shape[1]} kolom")
    print(f"Dimensi Fitur Rekayasa     : {X_train_proc.shape[1]} kolom (3 fitur domain baru ditambahkan)")
    print(f"Ukuran Matriks Latih Hasil : {X_train_proc.shape}")
    print(f"Ukuran Matriks Uji Hasil   : {X_test_proc.shape}")
    print("----------------------------------------------------------------------")
    print("Rerata Fitur Latih Terstandarisasi :", np.round(X_train_proc.mean(axis=0), 2))
    print("Deviasi Standar Fitur Latih        :", np.round(X_train_proc.std(axis=0), 2))
    print("======================================================================")
    print("STATUS ANTI-LEAKAGE: VALID (Scaler hanya dihitung dari data latih).")

if __name__ == "__main__":
    test_preprocessing()

