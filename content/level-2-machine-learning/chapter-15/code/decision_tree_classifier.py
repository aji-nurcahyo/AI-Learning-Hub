"""Klasifikasi Kondisi Peralatan Menggunakan Decision Tree Classifier
Mendemonstrasikan splitting criterion (Gini vs Entropy), kedalaman pohon, dan evaluasi.
"""

import pathlib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def run_decision_tree_classifier():
    data_path = pathlib.Path(__file__).parent / "data" / "equipment_fault_sensor_data.csv"
    df = pd.read_csv(data_path)

    features = ["temperature_c", "vibration_mm_s", "voltage_v", "current_a", "operating_hours", "load_percentage"]
    X = df[features]
    y = df["fault_category"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    print("=== DECISION TREE CLASSIFIER ===")
    print(f"Data latih: {len(X_train)} sampel, Data uji: {len(X_test)} sampel")

    # 1. Model Gini Impurity
    clf_gini = DecisionTreeClassifier(criterion="gini", max_depth=4, random_state=42)
    clf_gini.fit(X_train, y_train)
    y_pred_gini = clf_gini.predict(X_test)
    acc_gini = accuracy_score(y_test, y_pred_gini)

    # 2. Model Entropy / Information Gain
    clf_entropy = DecisionTreeClassifier(criterion="entropy", max_depth=4, random_state=42)
    clf_entropy.fit(X_train, y_train)
    y_pred_entropy = clf_entropy.predict(X_test)
    acc_entropy = accuracy_score(y_test, y_pred_entropy)

    print(f"\n1. Akurasi Gini Impurity (max_depth=4)    : {acc_gini * 100:.2f}%")
    print(f"2. Akurasi Entropy / Info Gain (max_depth=4): {acc_entropy * 100:.2f}%")

    print("\nLaporan Klasifikasi (Model Gini):")
    print(classification_report(y_test, y_pred_gini, digits=4))

    cm = confusion_matrix(y_test, y_pred_gini, labels=["NORMAL", "WARNING", "FAULT"])
    print("Confusion Matrix (NORMAL, WARNING, FAULT):")
    print(cm)

if __name__ == "__main__":
    run_decision_tree_classifier()

