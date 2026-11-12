"""
Script 06: Decision Threshold Tuning (Optimasi Ambang Keputusan)
Chapter 14: Supervised Learning — Classification Algorithms

Mendemonstrasikan penyesuaian ambang batas klasifikasi (Decision Threshold):
1. Default threshold 0.5 bukanlah hukum mutlak
2. Analisis Trade-off: Menurunkan threshold menaikkan Recall (tetapi menurunkan Presisi)
3. Skenario Keselamatan Pabrik: Biaya kerusakan mendadak jauh lebih mahal daripada inspeksi teknisi
4. Penentuan ambang optimal pada Validation Set
"""

from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import precision_score, recall_score, f1_score


def load_fault_data() -> pd.DataFrame:
    data_path = Path(__file__).resolve().parent / "data" / "equipment_fault_records.csv"
    return pd.read_csv(data_path)


def main():
    print("=" * 80)
    print("DEMONSTRASI DECISION THRESHOLD SELECTION & TRADE-OFF")
    print("=" * 80)
    
    df = load_fault_data()
    feature_cols = [
        "temperature_c", "vibration_rms", "acoustic_db",
        "voltage_fluctuation_pct", "pressure_bar", "current_amp"
    ]
    target_col = "fault_detected"
    
    X = df[feature_cols]
    y = df[target_col]
    
    # 3-Way Split (60% Train, 20% Val, 20% Test)
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.25, random_state=42, stratify=y_train_val
    )
    
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(random_state=42))
    ])
    pipe.fit(X_train, y_train)
    
    # 1. Eksplorasi Ambang Probabilitas pada Validation Set
    val_probs = pipe.predict_proba(X_val)[:, 1]
    thresholds = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
    
    print(f"Data: {len(X_train)} Train, {len(X_val)} Validation, {len(X_test)} Test\n")
    print("SIMULASI VARIASI THRESHOLD PADA VALIDATION SET:")
    print(f"{'Threshold (T)':<14} | {'Precision':<12} | {'Recall':<12} | {'F1-Score':<12} | {'Tindakan Alarm Operasional':<25}")
    print("-" * 80)
    
    best_thresh_f1 = 0.5
    best_f1 = -1.0
    
    for t in thresholds:
        preds = (val_probs >= t).astype(int)
        p = precision_score(y_val, preds, zero_division=0)
        r = recall_score(y_val, preds, zero_division=0)
        f = f1_score(y_val, preds, zero_division=0)
        
        if f > best_f1:
            best_f1 = f
            best_thresh_f1 = t
            
        desc = "Sangat Sensitif (Banyak FP)" if t <= 0.2 else ("Konservatif (Banyak FN)" if t >= 0.8 else "Moderat Seimbang")
        print(f"P(y=1) >= {t:<6.2f} | {p:<12.4f} | {r:<12.4f} | {f:<12.4f} | {desc:<25}")
        
    print("-" * 80)
    print(f"Ambang Batas Optimal untuk F1 Tertinggi pada Validasi: T = {best_thresh_f1:.2f} (F1 = {best_f1:.4f})\n")
    
    # 2. Pengujian Ambang Optimal pada Test Set (Unseen Data)
    test_probs = pipe.predict_proba(X_test)[:, 1]
    
    print("EVALUASI AKHIR PADA TEST SET DENGAN THRESHOLD TERPILIH:")
    for t in [0.50, best_thresh_f1, 0.20]:
        preds_test = (test_probs >= t).astype(int)
        p_t = precision_score(y_test, preds_test, zero_division=0)
        r_t = recall_score(y_test, preds_test, zero_division=0)
        f_t = f1_score(y_test, preds_test, zero_division=0)
        print(f"  * Threshold {t:.2f} -> Precision: {p_t:.4f} | Recall: {r_t:.4f} | F1: {f_t:.4f}")
        
    print("\nPrinsip Desain Bisnis:")
    print("Jika biaya melewatkan kegagalan mesin (False Negative) mencapai $50.000,")
    print("sedangkan biaya inspeksi teknisi (False Positive) hanya $200,")
    print("maka memilih threshold rendah (misal T=0.20) adalah keputusan teknis yang rasional.")
    print("=" * 80)


if __name__ == "__main__":
    main()

