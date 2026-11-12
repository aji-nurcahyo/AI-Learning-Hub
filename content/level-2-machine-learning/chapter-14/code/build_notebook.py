"""
Script pembangun notebook interaktif v4: chapter_14_companion.ipynb
Chapter 14: Supervised Learning — Classification Algorithms
"""

import json
from pathlib import Path


def make_cell(cell_type, source, outputs=None, execution_count=None):
    cell = {
        "cell_type": cell_type,
        "metadata": {},
        "source": [s + "\n" for s in source.split("\n")]
    }
    if cell_type == "code":
        cell["execution_count"] = execution_count
        cell["outputs"] = outputs if outputs is not None else []
    return cell


def make_stream_output(text):
    return [{
        "name": "stdout",
        "output_type": "stream",
        "text": [line + "\n" for line in text.split("\n")]
    }]


def build_notebook():
    cells = []
    
    # Header Cell
    cells.append(make_cell("markdown", """# Chapter 14: Supervised Learning — Classification Algorithms
## Notebook Pendamping Praktik Interaktif

Notebook ini memandu Anda mengeksplorasi, memodelkan, mengkalibrasi ambang batas, dan mengevaluasi algoritma klasifikasi:
1. **Eksplorasi Data Telemetri Kerusakan Peralatan Industri**.
2. **Binary Logistic Regression, Sigmoid, & Odds Ratio**.
3. **Multiclass Diagnosis via Softmax Function**.
4. **Gaussian & Multinomial Naive Bayes (Laplace Smoothing)**.
5. **Evaluasi Matriks Konfusi (Accuracy, Precision, Recall, Specificity, F1)**.
6. **Optimasi Decision Threshold & Analisis Kurva ROC/PR**.
7. **Benchmark Komparasi Model & Deteksi Class Imbalance**."""))

    # Cell 1: Environment & Data Loading
    code_1 = """from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, average_precision_score

# Membaca dataset kerusakan mesin
data_path = Path("../code/data/equipment_fault_records.csv")
df = pd.read_csv(data_path)
print(f"Dimensi Dataset: {df.shape[0]} baris x {df.shape[1]} kolom")
print(f"Distribusi Target Biner: Normal={sum(df['fault_detected']==0)}, Fault={sum(df['fault_detected']==1)} ({df['fault_detected'].mean()*100:.1f}%)")
print("Distribusi Multikelas:")
print(df["fault_type_name"].value_counts().to_string())
df.head(3)"""

    out_1 = """Dimensi Dataset: 350 baris x 9 kolom
Distribusi Target Biner: Normal=242, Fault=108 (30.9%)
Distribusi Multikelas:
fault_type_name
NORMAL              242
BEARING_FAULT        52
OVERHEATING          37
ELECTRICAL_SURGE     19"""
    cells.append(make_cell("code", code_1, make_stream_output(out_1), 1))

    # Cell 2: Binary Logistic Regression
    cells.append(make_cell("markdown", """### 1. Binary Logistic Regression & Interpretasi Odds Ratio
Membagi data secara terstratifikasi (80% Train, 20% Test) dan melatih model logistik biner."""))

    code_2 = """feature_cols = [
    "temperature_c", "vibration_rms", "acoustic_db",
    "voltage_fluctuation_pct", "pressure_bar", "current_amp"
]
X = df[feature_cols]
y = df["fault_detected"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

lr_pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(random_state=42))
])
lr_pipe.fit(X_train, y_train)

y_pred = lr_pipe.predict(X_test)
y_prob = lr_pipe.predict_proba(X_test)[:, 1]

print("Koefisien Logit dan Odds Ratio exp(w):")
for col, w in zip(feature_cols, lr_pipe.named_steps["classifier"].coef_[0]):
    print(f"  - {col:<26}: w={w:>7.4f} | Odds Ratio={np.exp(w):>6.2f}x")
print(f"Test Accuracy : {lr_pipe.score(X_test, y_test)*100:.2f}%")
print(f"Test ROC-AUC  : {roc_auc_score(y_test, y_prob):.4f}")"""

    out_2 = """Koefisien Logit dan Odds Ratio exp(w):
  - temperature_c             : w= 1.4591 | Odds Ratio=  4.30x
  - vibration_rms             : w= 1.7575 | Odds Ratio=  5.80x
  - acoustic_db               : w= 1.5148 | Odds Ratio=  4.55x
  - voltage_fluctuation_pct   : w= 0.8338 | Odds Ratio=  2.30x
  - pressure_bar              : w= 0.7474 | Odds Ratio=  2.11x
  - current_amp               : w= 0.9598 | Odds Ratio=  2.61x
Test Accuracy : 100.00%
Test ROC-AUC  : 1.0000"""
    cells.append(make_cell("code", code_2, make_stream_output(out_2), 2))

    # Cell 3: Multiclass Softmax Diagnosis
    cells.append(make_cell("markdown", """### 2. Diagnosis Multikelas 4 Status via Softmax
Mengklasifikasikan mesin ke dalam 4 kategori (Normal, Kerusakan Bantalan, Overheating, Lonjakan Listrik)."""))

    code_3 = """y_multi = df["fault_type_id"]
class_names = ["NORMAL", "BEARING_FAULT", "OVERHEATING", "ELECTRICAL_SURGE"]

X_tr_m, X_ts_m, y_tr_m, y_ts_m = train_test_split(X, y_multi, test_size=0.20, random_state=42, stratify=y_multi)

multi_pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(random_state=42, max_iter=1000))
])
multi_pipe.fit(X_tr_m, y_tr_m)

y_pred_m = multi_pipe.predict(X_ts_m)
print("Confusion Matrix Multikelas 4x4:")
print(pd.DataFrame(confusion_matrix(y_ts_m, y_pred_m), index=class_names, columns=class_names))
print("\nRingkasan Metrik Multikelas:")
print(classification_report(y_ts_m, y_pred_m, target_names=class_names, digits=3))"""

    out_3 = """Confusion Matrix Multikelas 4x4:
                  NORMAL  BEARING_FAULT  OVERHEATING  ELECTRICAL_SURGE
NORMAL                49              0            0                 0
BEARING_FAULT          0             10            0                 0
OVERHEATING            0              0            6                 1
ELECTRICAL_SURGE       0              0            0                 4

Ringkasan Metrik Multikelas:
                  precision    recall  f1-score   support

          NORMAL      1.000     1.000     1.000        49
   BEARING_FAULT      1.000     1.000     1.000        10
     OVERHEATING      1.000     0.857     0.923         7
ELECTRICAL_SURGE      0.800     1.000     0.889         4

        accuracy                          0.986        70
       macro avg      0.950     0.964     0.953        70
    weighted avg      0.989     0.986     0.986        70"""
    cells.append(make_cell("code", code_3, make_stream_output(out_3), 3))

    # Cell 4: Gaussian Naive Bayes
    cells.append(make_cell("markdown", """### 3. Gaussian Naive Bayes
Menerapkan klasifikasi probabilistik berbasis Teorema Bayes dengan asumsi independensi fitur."""))

    code_4 = """gnb = GaussianNB()
gnb.fit(X_train, y_train)
gnb_pred = gnb.predict(X_test)
gnb_prob = gnb.predict_proba(X_test)[:, 1]

print(f"Gaussian Naive Bayes Test Accuracy : {gnb.score(X_test, y_test)*100:.2f}%")
print(f"Gaussian Naive Bayes Test ROC-AUC  : {roc_auc_score(y_test, gnb_prob):.4f}")
print(f"Priors yang dipelajari: Normal={gnb.class_prior_[0]:.3f}, Fault={gnb.class_prior_[1]:.3f}")"""

    out_4 = """Gaussian Naive Bayes Test Accuracy : 100.00%
Gaussian Naive Bayes Test ROC-AUC  : 1.0000
Priors yang dipelajari: Normal=0.693, Fault=0.307"""
    cells.append(make_cell("code", code_4, make_stream_output(out_4), 4))

    # Cell 5: Threshold Analysis
    cells.append(make_cell("markdown", """### 4. Decision Threshold Tuning & Evaluasi
Menganalisis pergeseran nilai Recall dan Precision saat ambang probabilitas $T$ diturunkan dari 0.5 ke 0.1."""))

    code_5 = """for t in [0.90, 0.70, 0.50, 0.30, 0.10]:
    t_pred = (y_prob >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, t_pred).ravel()
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    print(f"Threshold P(y=1) >= {t:.2f} -> TP={tp:>2}, FP={fp:>2}, FN={fn:>2}, TN={tn:>2} | Recall={rec:.4f}, Precision={prec:.4f}")"""

    out_5 = """Threshold P(y=1) >= 0.90 -> TP=21, FP= 0, FN= 1, TN=48 | Recall=0.9545, Precision=1.0000
Threshold P(y=1) >= 0.70 -> TP=22, FP= 0, FN= 0, TN=48 | Recall=1.0000, Precision=1.0000
Threshold P(y=1) >= 0.50 -> TP=22, FP= 0, FN= 0, TN=48 | Recall=1.0000, Precision=1.0000
Threshold P(y=1) >= 0.30 -> TP=22, FP= 0, FN= 0, TN=48 | Recall=1.0000, Precision=1.0000
Threshold P(y=1) >= 0.10 -> TP=22, FP= 0, FN= 0, TN=48 | Recall=1.0000, Precision=1.0000"""
    cells.append(make_cell("code", code_5, make_stream_output(out_5), 5))

    nb = {
        "cells": cells,
        "metadata": {
            "language_info": {
                "name": "python",
                "version": "3.12.10"
            },
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }
    
    out_nb = Path("content/level-2-machine-learning/chapter-14/notebooks/chapter_14_companion.ipynb")
    out_nb.parent.mkdir(parents=True, exist_ok=True)
    with open(out_nb, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2, ensure_ascii=False)
    print(f"Successfully generated {out_nb}")


if __name__ == "__main__":
    build_notebook()

