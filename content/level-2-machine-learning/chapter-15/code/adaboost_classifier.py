"""Adaptive Boosting (AdaBoost) Classifier
Mendemonstrasikan sekuensial agregasi weak learners (decision stumps),
pembaruan bobot sampel, dan tuning learning rate.
"""

import pathlib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

def run_adaboost():
    data_path = pathlib.Path(__file__).parent / "data" / "equipment_fault_sensor_data.csv"
    df = pd.read_csv(data_path)

    features = ["temperature_c", "vibration_mm_s", "voltage_v", "current_a", "operating_hours", "load_percentage"]
    X = df[features]
    y = df["fault_category"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    print("=== ADABOOST CLASSIFIER ===")

    # 1. Base Estimator: Single Decision Stump (max_depth=1)
    stump = DecisionTreeClassifier(max_depth=1, random_state=42)
    stump.fit(X_train, y_train)
    stump_acc = accuracy_score(y_test, stump.predict(X_test))
    print(f"1. Single Decision Stump (Weak Learner) Test Accuracy: {stump_acc * 100:.2f}%")

    # 2. Pengaruh Learning Rate dan Estimator
    print("\n2. Eksperimen Peningkatan Akurasi AdaBoost (50 Stumps):")
    for lr in [0.1, 0.5, 1.0, 1.5]:
        ada = AdaBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=1),
            n_estimators=50,
            learning_rate=lr,
            random_state=42
        )
        ada.fit(X_train, y_train)
        tr_acc = accuracy_score(y_train, ada.predict(X_train))
        te_acc = accuracy_score(y_test, ada.predict(X_test))
        print(f"   learning_rate = {lr:<4} -> Train Acc: {tr_acc*100:.2f}% | Test Acc: {te_acc*100:.2f}%")

    # 3. Model Final dengan 100 Estimator
    ada_final = AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=1),
        n_estimators=100,
        learning_rate=0.5,
        random_state=42
    )
    ada_final.fit(X_train, y_train)
    y_pred = ada_final.predict(X_test)

    print("\n3. Laporan Klasifikasi Final AdaBoost (100 Stumps, LR=0.5):")
    print(classification_report(y_test, y_pred, digits=4))

if __name__ == "__main__":
    run_adaboost()

