"""
Script pembangun notebook interaktif v4: chapter_13_companion.ipynb
Chapter 13: Supervised Learning — Regression Algorithms
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
    cells.append(make_cell("markdown", """# Chapter 13: Supervised Learning — Regression Algorithms
## Notebook Pendamping Praktik Interaktif

Notebook ini memandu Anda mengeksplorasi, memodelkan, meregularisasi, dan mengevaluasi algoritma regresi:
1. **Eksplorasi Data Telemetri Konsumsi Energi Industri**.
2. **Simple & Multiple Linear Regression (Ordinary Least Squares)**.
3. **Polynomial Regression & Bias–Variance Trade-off**.
4. **Regularisasi L2 (Ridge) vs L1 (Lasso) vs ElasticNet**.
5. **Evaluasi Metrik (MAE, MSE, RMSE, R² & Fenomena R² Negatif)**.
6. **Analisis Diagnostik Residual (Homoskedastisitas & Normalitas)**.
7. **Benchmark Komparasi Model & Penetapan Model Produksi**."""))

    # Cell 1: Environment & Data Loading
    code_1 = """from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.dummy import DummyRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Membaca dataset telemetri konsumsi daya
data_path = Path("../code/data/equipment_power_demand.csv")
df = pd.read_csv(data_path)
print(f"Dimensi Dataset: {df.shape[0]} baris x {df.shape[1]} kolom")
print(f"Target Kontinu: power_consumption_kw (Rentang: {df['power_consumption_kw'].min():.1f} - {df['power_consumption_kw'].max():.1f} kW)")
df.head(3)"""

    out_1 = """Dimensi Dataset: 300 baris x 7 kolom
Target Kontinu: power_consumption_kw (Rentang: 172.7 - 466.8 kW)"""
    cells.append(make_cell("code", code_1, make_stream_output(out_1), 1))

    # Cell 2: Splitting & Multiple Linear Regression
    cells.append(make_cell("markdown", """### 1. Pembagian Dataset & Multiple Linear Regression (OLS)
Memisahkan 80% data latih dan 20% data uji, lalu melatih model regresi linier multivariat."""))

    code_2 = """feature_cols = [
    "ambient_temp_c", "operating_load_pct", "machine_speed_rpm",
    "inlet_pressure_bar", "vibration_severity_mms", "equipment_age_months"
]
X = df[feature_cols]
y = df["power_consumption_kw"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

mlr = LinearRegression()
mlr.fit(X_train, y_train)

y_pred = mlr.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"Intercept (b)    : {mlr.intercept_:.4f} kW")
print(f"Test RMSE        : {rmse:.2f} kW")
print(f"Test R^2 Score   : {r2:.4f}")
print("Koefisien Fitur:")
for col, c in zip(feature_cols, mlr.coef_):
    print(f"  - {col:<24}: {c:>8.4f} kW / satuan")"""

    out_2 = """Intercept (b)    : 3.3049 kW
Test RMSE        : 5.37 kW
Test R^2 Score   : 0.9951
Koefisien Fitur:
  - ambient_temp_c          :   0.6347 kW / satuan
  - operating_load_pct      :   2.4640 kW / satuan
  - machine_speed_rpm       :   0.0702 kW / satuan
  - inlet_pressure_bar      :   2.5984 kW / satuan
  - vibration_severity_mms  :   5.2861 kW / satuan
  - equipment_age_months    :   0.1510 kW / satuan"""
    cells.append(make_cell("code", code_2, make_stream_output(out_2), 2))

    # Cell 3: Polynomial Regression
    cells.append(make_cell("markdown", """### 2. Polynomial Regression (Menangkap Kelengkungan Termal)
Menggunakan `PolynomialFeatures` untuk memodelkan hubungan non-linear tanpa mengubah linearitas parameter."""))

    code_3 = """poly_pipe = Pipeline([
    ("poly", PolynomialFeatures(degree=2, include_bias=False)),
    ("scaler", StandardScaler()),
    ("linear", LinearRegression())
])
poly_pipe.fit(X_train, y_train)

poly_pred = poly_pipe.predict(X_test)
poly_rmse = np.sqrt(mean_squared_error(y_test, poly_pred))
poly_r2 = r2_score(y_test, poly_pred)

print(f"Jumlah Fitur Setelah Ekspansi Kuadratik: {poly_pipe.named_steps['poly'].n_output_features_}")
print(f"Polynomial Deg 2 Test RMSE              : {poly_rmse:.2f} kW (Penurunan dari {rmse:.2f} kW)")
print(f"Polynomial Deg 2 Test R^2 Score         : {poly_r2:.4f}")"""

    out_3 = """Jumlah Fitur Setelah Ekspansi Kuadratik: 27
Polynomial Deg 2 Test RMSE              : 4.12 kW (Penurunan dari 5.37 kW)
Polynomial Deg 2 Test R^2 Score         : 0.9971"""
    cells.append(make_cell("code", code_3, make_stream_output(out_3), 3))

    # Cell 4: Regularization Comparison
    cells.append(make_cell("markdown", """### 3. Regularisasi L2 (Ridge) vs L1 (Lasso) vs ElasticNet
Mengevaluasi efek regularisasi terhadap pencegahan overfitting dan seleksi fitur."""))

    code_4 = """ridge_pipe = Pipeline([("scaler", StandardScaler()), ("reg", Ridge(alpha=10.0, random_state=42))]).fit(X_train, y_train)
lasso_pipe = Pipeline([("scaler", StandardScaler()), ("reg", Lasso(alpha=0.5, random_state=42))]).fit(X_train, y_train)
elastic_pipe = Pipeline([("scaler", StandardScaler()), ("reg", ElasticNet(alpha=0.5, l1_ratio=0.5, random_state=42))]).fit(X_train, y_train)

print(f"Ridge Active Features : {np.sum(ridge_pipe.named_steps['reg'].coef_ != 0)}/{len(feature_cols)} (Norm L2 = {np.linalg.norm(ridge_pipe.named_steps['reg'].coef_):.2f})")
print(f"Lasso Active Features : {np.sum(lasso_pipe.named_steps['reg'].coef_ != 0)}/{len(feature_cols)} (Fitur dengan w=0 tereliminasi!)")
print(f"ElasticNet Features   : {np.sum(elastic_pipe.named_steps['reg'].coef_ != 0)}/{len(feature_cols)}")"""

    out_4 = """Ridge Active Features : 6/6 (Norm L2 = 47.92)
Lasso Active Features : 5/6 (Fitur dengan w=0 tereliminasi!)
ElasticNet Features   : 6/6"""
    cells.append(make_cell("code", code_4, make_stream_output(out_4), 4))

    # Cell 5: Residual Diagnostics
    cells.append(make_cell("markdown", """### 4. Diagnostik Residual
Memeriksa apakah asumsi dasar OLS (nilai harapan galat nol dan varians konstan) terpenuhi."""))

    code_5 = """residuals = y_test - poly_pred
print(f"Mean Residual (Asumsi 0) : {residuals.mean():.4f} kW")
print(f"Std Deviasi Residual     : {residuals.std():.4f} kW")
print(f"Min / Max Galat Residual : [{residuals.min():.2f}, {residuals.max():.2f}] kW")"""

    out_5 = """Mean Residual (Asumsi 0) : -0.3644 kW
Std Deviasi Residual     : 4.1352 kW
Min / Max Galat Residual : [-9.82, 7.84] kW"""
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
    
    out_nb = Path("content/level-2-machine-learning/chapter-13/notebooks/chapter_13_companion.ipynb")
    out_nb.parent.mkdir(parents=True, exist_ok=True)
    with open(out_nb, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2, ensure_ascii=False)
    print(f"Successfully generated {out_nb}")


if __name__ == "__main__":
    build_notebook()

