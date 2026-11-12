"""Analisis Kepentingan Fitur: Permutation Feature Importance vs MDI
Mengevaluasi penurunan performa model saat fitur diacak pada data uji.
"""

import pathlib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance

def run_permutation_importance():
    data_path = pathlib.Path(__file__).parent / "data" / "sensor_fleet_eval_telemetry.csv"
    df = pd.read_csv(data_path)

    features = [
        "vibration_amplitude_g",
        "operating_temp_c",
        "pressure_kpa",
        "power_draw_kw",
        "acoustic_noise_db",
        "operating_hours"
    ]
    X = df[features].values
    y = df["failure_status"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("rf", RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    pipe.fit(X_train, y_train)

    # 1. Feature Importance Bawaan Model (MDI - Mean Decrease in Impurity / Gini)
    rf_model = pipe.named_steps["rf"]
    mdi_importance = rf_model.feature_importances_

    # 2. Permutation Importance Dihitung pada DATA UJI (Test Set)
    perm_result = permutation_importance(
        estimator=pipe,
        X=X_test,
        y=y_test,
        scoring="f1",
        n_repeats=15,
        random_state=42,
        n_jobs=-1
    )

    perm_mean = perm_result.importances_mean
    perm_std = perm_result.importances_std

    summary_df = pd.DataFrame({
        "Fitur": features,
        "MDI Latih (Gini)": mdi_importance,
        "Permutation Mean (Test)": perm_mean,
        "Permutation Std (Test)": perm_std
    }).sort_values(by="Permutation Mean (Test)", ascending=False)

    print("======================================================================")
    print("ANALISIS EXPLAINABILITY: PERMUTATION IMPORTANCE VS MDI")
    print("======================================================================")
    print(f"{'Nama Fitur':<24} {'MDI Latih (Gini)':<20} {'Permutation Drop (F1)':<24} {'Std Error'}")
    print("----------------------------------------------------------------------")

    for _, row in summary_df.iterrows():
        print(
            f"{row['Fitur']:<24} "
            f"{row['MDI Latih (Gini)']*100:6.2f}%              "
            f"{row['Permutation Mean (Test)']*100:6.2f}%                  "
            f"{row['Permutation Std (Test)']*100:4.2f}%"
        )

    print("======================================================================")
    print("INSIGHT KRITIS EXPLAINABILITY:")
    print("1. MDI dihitung pada data latih dan cenderung bias melebih-lebihkan fitur numerik kontinu.")
    print("2. Permutation Importance dihitung pada data uji: mengukur penurunan riil metrik F1")
    print("   ketika korelasi fitur dengan target dihancurkan secara acak.")
    print("3. Peringatan Ilmiah: Feature Importance HANYA mengukur dependensi prediktif,")
    print("   BUKAN bukti hubungan sebab-akibat (causation) secara hukum fisika.")

if __name__ == "__main__":
    run_permutation_importance()

