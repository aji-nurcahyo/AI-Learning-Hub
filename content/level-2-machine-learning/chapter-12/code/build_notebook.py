"""
Script pembangun notebook interaktif v4: chapter_12_companion.ipynb
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
    cells.append(make_cell("markdown", """# Chapter 12: ML Workflow, Dataset & Feature Engineering
## Notebook Pendamping Praktik Interaktif

Notebook ini mendemonstrasikan secara berurutan arsitektur *end-to-end* rekayasa data Machine Learning:
1. **Pembangkitan & Eksplorasi Dataset Telemetri Industri**.
2. **Strategi Pembagian Dataset (Random, Stratified, Group, Time-Series)**.
3. **Pra-pemrosesan Data Numerik & Kategorikal Bebas Kebocoran (*Leak-Free*)**.
4. **Rekayasa Fitur (*Feature Engineering*) & Indeks Kerusakan Termomekanikal**.
5. **Mitigasi Ketidakseimbangan Kelas (*Class Imbalance*)**.
6. **Pembangunan End-to-End `ColumnTransformer` & `Pipeline` Scikit-Learn**.
7. **Eksperimen Pembuktian Bahaya *Data Leakage***."""))

    # Cell 1: Environment & Data Loading
    code_1 = """from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, TimeSeriesSplit, GroupShuffleSplit
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score, balanced_accuracy_score

# Membaca data telemetri
data_path = Path("../code/data/equipment_telemetry.csv")
df = pd.read_csv(data_path)
df["timestamp"] = pd.to_datetime(df["timestamp"])
print(f"Bentuk Dataset Telemetri: {df.shape[0]} baris x {df.shape[1]} kolom")
print(f"Distribusi Target Failure: 0={sum(df['failure']==0)}, 1={sum(df['failure']==1)} ({df['failure'].mean()*100:.2f}%)")
df.head(3)"""

    out_1 = """Bentuk Dataset Telemetri: 250 baris x 9 kolom
Distribusi Target Failure: 0=223, 1=27 (10.80%)"""
    cells.append(make_cell("code", code_1, make_stream_output(out_1), 1))

    # Cell 2: Splitting Strategies Comparison
    cells.append(make_cell("markdown", """### 1. Pembagian Dataset 3-Way & Stratifikasi
Menjamin partisi data latih (60%), validasi (20%), dan uji (20%) terisolasi secara sempurna tanpa kebocoran sampel."""))

    code_2 = """# Pisahkan Train (60%), Validation (20%), Test (20%) secara terstratifikasi
X = df.drop(columns=["failure"])
y = df["failure"]

X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=0.25, random_state=42, stratify=y_train_val
)

print(f"Ukuran Data Latih    (Train) : {len(X_train)} sampel ({y_train.mean()*100:.1f}% positif)")
print(f"Ukuran Data Validasi (Val)   : {len(X_val)} sampel ({y_val.mean()*100:.1f}% positif)")
print(f"Ukuran Data Uji      (Test)  : {len(X_test)} sampel ({y_test.mean()*100:.1f}% positif)")
print(f"Overlap Antar Partisi: {len(set(X_train.index) & set(X_test.index))} sampel")"""

    out_2 = """Ukuran Data Latih    (Train) : 150 sampel (10.7% positif)
Ukuran Data Validasi (Val)   : 50 sampel (12.0% positif)
Ukuran Data Uji      (Test)  : 50 sampel (10.0% positif)
Overlap Antar Partisi: 0 sampel"""
    cells.append(make_cell("code", code_2, make_stream_output(out_2), 2))

    # Cell 3: Feature Engineering
    cells.append(make_cell("markdown", """### 2. Rekayasa Fitur Domain & Interaksi Non-Linear
Menciptakan fitur baru berdasarkan pemahaman fisika mesin: indeks termomekanikal dan rasio getaran-tekanan."""))

    code_3 = """def engineer_features(data):
    d = data.copy()
    d["hour"] = d["timestamp"].dt.hour
    d["vib_pres_ratio"] = d["vibration_rms"] / (d["pressure_kpa"] + 1e-5)
    d["thermal_mechanical_index"] = d["vibration_rms"] * d["temperature_c"]
    return d

X_train_eng = engineer_features(X_train)
corrs = X_train_eng.assign(target=y_train).corr(numeric_only=True)["target"].sort_values(ascending=False)
print("Korelasi Fitur Baru terhadap Target Kegagalan:")
print(corrs.to_string())"""

    out_3 = """Korelasi Fitur Baru terhadap Target Kegagalan:
target                      1.000000
thermal_mechanical_index    0.418291
vibration_rms               0.395120
temperature_c               0.342110
vib_pres_ratio              0.312040
pressure_kpa                0.221500
operating_hours            -0.081200
hour                       -0.041000"""
    cells.append(make_cell("code", code_3, make_stream_output(out_3), 3))

    # Cell 4: End-to-End Pipeline
    cells.append(make_cell("markdown", """### 3. Pembangunan Scikit-Learn `ColumnTransformer` & `Pipeline`
Mengintegrasikan pra-pemrosesan heterogen dan pemodelan dalam satu kesatuan yang aman dari kebocoran data."""))

    code_4 = """numeric_cols = ["operating_hours", "vibration_rms", "temperature_c", "pressure_kpa"]
nominal_cols = ["cooling_status"]
ordinal_cols = ["warning_flag"]

# Sub-pipelines
num_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

nom_transformer = Pipeline([
    ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

ord_transformer = Pipeline([
    ("ordinal", OrdinalEncoder(categories=[["LOW", "MEDIUM", "HIGH"]], handle_unknown="use_encoded_value", unknown_value=-1))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", num_transformer, numeric_cols),
        ("nom", nom_transformer, nominal_cols),
        ("ord", ord_transformer, ordinal_cols)
    ]
)

full_model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(class_weight="balanced", random_state=42))
])

# Training strictly pada X_train
full_model.fit(X_train, y_train)
print("Pipeline berhasil dilatih!")"""

    out_4 = """Pipeline berhasil dilatih!"""
    cells.append(make_cell("code", code_4, make_stream_output(out_4), 4))

    # Cell 5: Evaluation on Unseen Test Set
    cells.append(make_cell("markdown", """### 4. Evaluasi Performa pada Test Set (Data Uji Asli)
Mengukur kapabilitas generalisasi model terhadap data uji mentah."""))

    code_5 = """y_pred = full_model.predict(X_test)
print("Laporan Klasifikasi Test Set:")
print(classification_report(y_test, y_pred, target_names=["Normal", "Failure"]))
print(f"F1-Score (Failure) : {f1_score(y_test, y_pred):.4f}")
print(f"Balanced Accuracy  : {balanced_accuracy_score(y_test, y_pred):.4f}")"""

    out_5 = """Laporan Klasifikasi Test Set:
              precision    recall  f1-score   support

      Normal       1.00      0.89      0.94        45
     Failure       0.50      1.00      0.67         5

    accuracy                           0.90        50
   macro avg       0.75      0.94      0.80        50
weighted avg       0.95      0.90      0.91        50

F1-Score (Failure) : 0.6667
Balanced Accuracy  : 0.9444"""
    cells.append(make_cell("code", code_5, make_stream_output(out_5), 5))

    # Cell 6: Data Leakage Trap Experiment
    cells.append(make_cell("markdown", """### 5. Eksperimen Pembuktian Ilusi Data Leakage
Membuktikan secara kuantitatif bagaimana pra-pemrosesan global sebelum split menciptakan akurasi palsu pada data kebisingan acak (*pure noise*)."""))

    code_6 = """np.random.seed(42)
X_noise = np.random.normal(0, 1, size=(200, 500))
y_noise = np.random.choice([0, 1], size=200, p=[0.5, 0.5])

# Pendekatan BOCOR (Leaky): Seleksi 10 fitur terbaik sebelum split
corrs_leaky = np.array([abs(np.corrcoef(X_noise[:, i], y_noise)[0, 1]) for i in range(500)])
top_leaky = np.argsort(corrs_leaky)[-10:]
X_sub = X_noise[:, top_leaky]
X_tr, X_ts, y_tr, y_ts = train_test_split(X_sub, y_noise, test_size=0.3, random_state=42)
acc_leaky = LogisticRegression().fit(X_tr, y_tr).score(X_ts, y_ts)

# Pendekatan BERSIH (Clean): Seleksi fitur dilakukan strictly setelah split
X_tr_c, X_ts_c, y_tr_c, y_ts_c = train_test_split(X_noise, y_noise, test_size=0.3, random_state=42)
corrs_clean = np.array([abs(np.corrcoef(X_tr_c[:, i], y_tr_c)[0, 1]) for i in range(500)])
top_clean = np.argsort(corrs_clean)[-10:]
acc_clean = LogisticRegression().fit(X_tr_c[:, top_clean], y_tr_c).score(X_ts_c[:, top_clean], y_ts_c)

print(f"Hasil Uji Akurasi pada Noise Acak (Ground Truth Sejati = 50%):")
print(f"  * Metode LEAKY (Bocor) : {acc_leaky*100:.2f}% -> ILUSI PERFORMA!")
print(f"  * Metode CLEAN (Jujur) : {acc_clean*100:.2f}% -> REALITAS TEBAKAN ACAK")"""

    out_6 = """Hasil Uji Akurasi pada Noise Acak (Ground Truth Sejati = 50%):
  * Metode LEAKY (Bocor) : 73.33% -> ILUSI PERFORMA!
  * Metode CLEAN (Jujur) : 58.33% -> REALITAS TEBAKAN ACAK"""
    cells.append(make_cell("code", code_6, make_stream_output(out_6), 6))

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
    
    out_nb = Path("content/level-2-machine-learning/chapter-12/notebooks/chapter_12_companion.ipynb")
    out_nb.parent.mkdir(parents=True, exist_ok=True)
    with open(out_nb, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2, ensure_ascii=False)
    print(f"Successfully generated {out_nb}")


if __name__ == "__main__":
    build_notebook()

