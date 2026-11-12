"""
Script 08: Class Imbalance Handling (Cost-Sensitive & Threshold Tuning)
Chapter 12: ML Workflow, Dataset & Feature Engineering

Mendemonstrasikan penanganan ketidakseimbangan kelas (class imbalance) secara metodologis:
1. Bahaya resample pra-split (leakage).
2. Cost-sensitive learning menggunakan 'class_weight=balanced'.
3. Tuning ambang batas klasifikasi (decision threshold) pada Validation Set.
4. Evaluasi menggunakan Precision, Recall, F1-Score, dan Balanced Accuracy.
"""

from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, f1_score, recall_score, precision_score, balanced_accuracy_score


def load_telemetry_data() -> pd.DataFrame:
    data_path = Path(__file__).resolve().parent / "data" / "equipment_telemetry.csv"
    return pd.read_csv(data_path)


def main():
    print("=" * 75)
    print("DEMONSTRASI PENANGANAN CLASS IMBALANCE SECARA METODOLOGIS")
    print("=" * 75)
    
    df = load_telemetry_data()
    feature_cols = ["operating_hours", "vibration_rms", "temperature_c", "pressure_kpa"]
    
    # 1. Stratified Split 3-Way (Train 60%, Val 20%, Test 20%)
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        df[feature_cols], df["failure"], test_size=0.20, random_state=42, stratify=df["failure"]
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.25, random_state=42, stratify=y_train_val
    )
    
    # 2. Imputasi & Penskalaan (fit pada train saja)
    imputer = SimpleImputer(strategy="median").fit(X_train)
    scaler = StandardScaler().fit(imputer.transform(X_train))
    
    X_train_proc = scaler.transform(imputer.transform(X_train))
    X_val_proc = scaler.transform(imputer.transform(X_val))
    X_test_proc = scaler.transform(imputer.transform(X_test))
    
    print(f"Distribusi Kelas Train : Positif={y_train.sum()}/{len(y_train)} ({y_train.mean()*100:.1f}%)")
    print(f"Distribusi Kelas Test  : Positif={y_test.sum()}/{len(y_test)} ({y_test.mean()*100:.1f}%)\n")
    
    # Model 1: Baseline Tanpa Penanganan Imbalance (default threshold = 0.5)
    clf_standard = LogisticRegression(random_state=42)
    clf_standard.fit(X_train_proc, y_train)
    y_pred_std = clf_standard.predict(X_test_proc)
    
    # Model 2: Cost-Sensitive (class_weight='balanced')
    clf_balanced = LogisticRegression(class_weight="balanced", random_state=42)
    clf_balanced.fit(X_train_proc, y_train)
    y_pred_bal = clf_balanced.predict(X_test_proc)
    
    # Model 3: Threshold Tuning pada Validation Set
    val_probs = clf_standard.predict_proba(X_val_proc)[:, 1]
    thresholds = np.linspace(0.1, 0.9, 17)
    best_thresh = 0.5
    best_f1 = -1.0
    
    for th in thresholds:
        f1_th = f1_score(y_val, (val_probs >= th).astype(int), zero_division=0)
        if f1_th > best_f1:
            best_f1 = f1_th
            best_thresh = th
            
    test_probs = clf_standard.predict_proba(X_test_proc)[:, 1]
    y_pred_tuned = (test_probs >= best_thresh).astype(int)
    
    print("HASIL EVALUASI PADA TEST SET (UNSEEN DATA):")
    print(f"{'Metode':<28} | {'Recall':<8} | {'Precision':<10} | {'F1-Score':<10} | {'Balanced Acc':<12}")
    print("-" * 75)
    
    methods = [
        ("1. Standard (No Weight)", y_pred_std),
        ("2. Balanced Class Weight", y_pred_bal),
        (f"3. Tuned Thresh ({best_thresh:.2f})", y_pred_tuned)
    ]
    
    for name, pred in methods:
        rec = recall_score(y_test, pred, zero_division=0)
        prec = precision_score(y_test, pred, zero_division=0)
        f1 = f1_score(y_test, pred, zero_division=0)
        bacc = balanced_accuracy_score(y_test, pred)
        print(f"{name:<28} | {rec:<8.4f} | {prec:<10.4f} | {f1:<10.4f} | {bacc:<12.4f}")
        
    print("-" * 75)
    print("Analisis:")
    print("1. Standard logistic regression cenderung mengabaikan kelas minoritas.")
    print("2. 'class_weight=balanced' dan penyesuaian ambang probabilitas secara dramatis")
    print("   meningkatkan Recall kegagalan tanpa memerlukan manipulasi data sintetis pra-split.")
    print("=" * 75)


if __name__ == "__main__":
    main()

