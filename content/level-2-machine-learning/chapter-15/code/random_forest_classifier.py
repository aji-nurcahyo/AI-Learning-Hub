"""Random Forest Classifier
Mendemonstrasikan random feature subsampling (sqrt(p)), bootstrap sampling,
dan analisis Feature Importance (Mean Decrease in Impurity / MDI).
"""

import pathlib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def run_random_forest():
    data_path = pathlib.Path(__file__).parent / "data" / "equipment_fault_sensor_data.csv"
    df = pd.read_csv(data_path)

    features = ["temperature_c", "vibration_mm_s", "voltage_v", "current_a", "operating_hours", "load_percentage"]
    X = df[features]
    y = df["fault_category"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    print("=== RANDOM FOREST CLASSIFIER ===")
    print(f"Dimensi Fitur (p): {len(features)} | Sub-fitur per split (max_features='sqrt'): {int(len(features)**0.5)}")

    rf = RandomForestClassifier(
        n_estimators=100,
        max_features="sqrt",
        max_depth=6,
        random_state=42
    )
    rf.fit(X_train, y_train)

    train_acc = accuracy_score(y_train, rf.predict(X_train))
    test_acc = accuracy_score(y_test, rf.predict(X_test))

    print(f"\nAkurasi Latih: {train_acc * 100:.2f}%")
    print(f"Akurasi Uji  : {test_acc * 100:.2f}%")

    # Analisis Feature Importance (MDI)
    importances = rf.feature_importances_
    feat_imp = pd.Series(importances, index=features).sort_values(ascending=False)

    print("\nAnalisis Kepentingan Fitur (Feature Importance - MDI):")
    for feat, val in feat_imp.items():
        bar = "#" * int(val * 40)
        print(f"  {feat:<18}: {val:.4f} ({val*100:5.1f}%) | {bar}")

    print("\nCatatan Kritis Keterbatasan MDI:")
    print("MDI dapat memberikan bobot berlebih pada fitur numerik kontinu dengan kardinalitas tinggi")
    print("atau fitur berkorelasi. Untuk validasi mendalam, pertimbangkan Permutation Importance.")

    print("\nLaporan Klasifikasi Data Uji:")
    y_pred = rf.predict(X_test)
    print(classification_report(y_test, y_pred, digits=4))

if __name__ == "__main__":
    run_random_forest()

