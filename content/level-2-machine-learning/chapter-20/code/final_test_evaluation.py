"""
Chapter 20 - Machine Learning Capstone
Script 5: Final Test Evaluation
Mengevaluasi performa Champion Model (Random Forest yang telah di-tuning)
pada untouched test set (20%, 120 data) dan membandingkannya dengan Baseline Model.
"""

import warnings
warnings.filterwarnings('ignore')
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from preprocessing_pipeline import build_preprocessing_pipeline

DATA_PATH = Path(__file__).parent / "data" / "predictive_maintenance_telemetry.csv"

def evaluate_test_set():
    print("=" * 80)
    print("CAPSTONE TAHAP 5: EVALUASI AKHIR PADA UNTOUCHED TEST SET (N = 120)")
    print("=" * 80)

    df = pd.read_csv(DATA_PATH)
    feature_cols = [
        "vibration_rms_g", "bearing_temperature_c", "rotational_speed_rpm",
        "hydraulic_pressure_bar", "motor_current_amp", "oil_acidity_tan"
    ]
    X = df[feature_cols].values
    y = df['machine_failure'].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # 1. Baseline Model (Logistic Regression)
    baseline_pipe = Pipeline([
        ('prep', build_preprocessing_pipeline()),
        ('clf', LogisticRegression(random_state=42))
    ])
    baseline_pipe.fit(X_train, y_train)
    y_pred_base = baseline_pipe.predict(X_test)
    y_prob_base = baseline_pipe.predict_proba(X_test)[:, 1]

    # 2. Champion Model (Tuned Random Forest)
    champion_pipe = Pipeline([
        ('prep', build_preprocessing_pipeline()),
        ('rf', RandomForestClassifier(
            n_estimators=100,
            max_depth=8,
            min_samples_split=2,
            random_state=42
        ))
    ])
    champion_pipe.fit(X_train, y_train)
    y_pred_champ = champion_pipe.predict(X_test)
    y_prob_champ = champion_pipe.predict_proba(X_test)[:, 1]

    # Metrik Evaluasi
    models = {
        'Baseline (Logistic Regression)': (y_pred_base, y_prob_base),
        'Champion (Tuned Random Forest)': (y_pred_champ, y_prob_champ)
    }

    print(f"{'Model':<32} {'Akurasi':<12} {'Presisi':<12} {'Recall':<12} {'F1-Score':<12} {'ROC-AUC':<10}")
    print("-" * 90)

    for name, (pred, prob) in models.items():
        acc = accuracy_score(y_test, pred)
        prec = precision_score(y_test, pred)
        rec = recall_score(y_test, pred)
        f1 = f1_score(y_test, pred)
        auc = roc_auc_score(y_test, prob)
        print(f"{name:<32} {acc*100:>7.2f}%     {prec*100:>7.2f}%     {rec*100:>7.2f}%     {f1:>7.4f}      {auc:>7.4f}")

    print("=" * 90)

    cm = confusion_matrix(y_test, y_pred_champ)
    tn, fp, fn, tp = cm.ravel()
    print("\nCONFUSION MATRIX CHAMPION MODEL:")
    print(f"  True Negatives  (Normal diprediksi Normal)     : {tn}")
    print(f"  False Positives (Normal salah diprediksi Rusak): {fp}")
    print(f"  False Negatives (Rusak lolos/tidak terdeteksi) : {fn}")
    print(f"  True Positives  (Rusak berhasil terdeteksi)   : {tp}")

    print("\nLAPORAN KLASIFIKASI LENGKAP:")
    print(classification_report(y_test, y_pred_champ, target_names=['Normal (0)', 'Failure (1)']))

    return champion_pipe, X_test, y_test, y_pred_champ, y_prob_champ

if __name__ == "__main__":
    evaluate_test_set()
