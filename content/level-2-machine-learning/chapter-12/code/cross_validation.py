"""
Script 09: Cross-Validation Strategies & Performance Variance
Chapter 12: ML Workflow, Dataset & Feature Engineering

Mendemonstrasikan evaluasi model menggunakan K-Fold dan Stratified K-Fold
Cross-Validation dengan Scikit-learn:
1. Estimasi skor rata-rata (mean score).
2. Analisis varians/standar deviasi performa lintas lipatan (fold variance).
3. Pemilihan metrik yang relevan (ROC-AUC, F1, Balanced Accuracy).
"""

from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


def load_telemetry_data() -> pd.DataFrame:
    data_path = Path(__file__).resolve().parent / "data" / "equipment_telemetry.csv"
    return pd.read_csv(data_path)


def main():
    print("=" * 75)
    print("DEMONSTRASI STRATEGI CROSS-VALIDATION & ANALISIS VARIANS")
    print("=" * 75)
    
    df = load_telemetry_data()
    feature_cols = ["operating_hours", "vibration_rms", "temperature_c", "pressure_kpa"]
    X = df[feature_cols]
    y = df["failure"]
    
    # Membangun pipeline yang bersih untuk evaluasi CV
    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(class_weight="balanced", random_state=42))
    ])
    
    # 1. Bandingkan 5-Fold Standar vs 5-Fold Stratified
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    scoring_metrics = ["f1", "roc_auc", "balanced_accuracy"]
    
    print(f"Dataset: {len(X)} sampel, Positif: {y.sum()} ({y.mean()*100:.2f}%)\n")
    
    for metric in scoring_metrics:
        scores_kf = cross_val_score(pipeline, X, y, cv=kf, scoring=metric)
        scores_skf = cross_val_score(pipeline, X, y, cv=skf, scoring=metric)
        
        print(f"--- METRIK: {metric.upper()} ---")
        print(f"  * K-Fold Standar : Mean = {scores_kf.mean():.4f} | Std = {scores_kf.std():.4f}")
        print(f"    Skor per-fold  : {[round(s, 4) for s in scores_kf]}")
        print(f"  * Stratified KF  : Mean = {scores_skf.mean():.4f} | Std = {scores_skf.std():.4f}")
        print(f"    Skor per-fold  : {[round(s, 4) for s in scores_skf]}\n")
        
    print("Analisis Stabilitas:")
    print("Stratified K-Fold menghasilkan variasi skor antar-fold yang lebih terkontrol")
    print("karena setiap fold mempertahankan rasio kegagalan mesin yang representatif.")
    print("=" * 75)


if __name__ == "__main__":
    main()

