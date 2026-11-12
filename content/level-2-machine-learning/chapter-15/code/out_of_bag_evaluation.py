"""Evaluasi Out-of-Bag (OOB) pada Random Forest
Mendemonstrasikan estimasi performa generalisasi tanpa validasi silang eksplisit
melalui sampel sisa bootstrap (~36.8%).
"""

import pathlib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def run_oob_evaluation():
    data_path = pathlib.Path(__file__).parent / "data" / "equipment_fault_sensor_data.csv"
    df = pd.read_csv(data_path)

    features = ["temperature_c", "vibration_mm_s", "voltage_v", "current_a", "operating_hours", "load_percentage"]
    X = df[features]
    y = df["fault_category"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    print("=== EVALUASI OUT-OF-BAG (OOB) RANDOM FOREST ===")

    # Penjelasan Probabilistik OOB
    n = len(X_train)
    p_not_selected = (1.0 - 1.0 / n) ** n
    print(f"Jumlah sampel latih (N): {n}")
    print(f"Probabilitas sampel TIDAK terpilih dalam bootstrap: (1 - 1/N)^N = {p_not_selected:.4f} (Teoretis e^(-1) = {1/np.e:.4f})")
    print(f"Setiap pohon rata-rata dievaluasi pada {p_not_selected * 100:.1f}% data latih yang tersisa (out-of-bag).\n")

    # Eksperimen variasi jumlah pohon untuk stabilitas OOB Score
    print("Estimator | Skor OOB      | Akurasi Uji (Hold-out)")
    print("-" * 50)

    for n_trees in [15, 30, 50, 100, 200]:
        rf = RandomForestClassifier(
            n_estimators=n_trees,
            max_features="sqrt",
            oob_score=True,
            random_state=42
        )
        rf.fit(X_train, y_train)

        oob = rf.oob_score_
        test_acc = accuracy_score(y_test, rf.predict(X_test))
        print(f"{n_trees:<9} | {oob*100:6.2f}%       | {test_acc*100:6.2f}%")

    # Komparasi OOB vs 5-Fold Stratified Cross-Validation
    rf_final = RandomForestClassifier(n_estimators=100, max_features="sqrt", oob_score=True, random_state=42)
    rf_final.fit(X_train, y_train)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(rf_final, X_train, y_train, cv=cv, scoring="accuracy")

    print("\nKomparasi Metode Estimasi Generalisasi:")
    print(f"1. Out-of-Bag (OOB) Score          : {rf_final.oob_score_ * 100:.2f}% (Dihitung otomatis saat pelatihan)")
    print(f"2. 5-Fold Stratified Cross-Val Score: {cv_scores.mean() * 100:.2f}% (+/- {cv_scores.std()*100:.2f}%)")
    print(f"3. Independent Test Set Score      : {accuracy_score(y_test, rf_final.predict(X_test)) * 100:.2f}%")
    print("\nKesimpulan: OOB score memberikan estimasi performa generalisasi yang sangat andal dan murah komputasi.")

if __name__ == "__main__":
    run_oob_evaluation()

