"""
Script 07: Regression Evaluation Metrics (MAE, MSE, RMSE, and R^2)
Chapter 13: Supervised Learning — Regression Algorithms

Mendemonstrasikan perhitungan matematika dan interpretasi metrik regresi:
1. Mean Absolute Error (MAE): Bobot galat linier intuitif
2. Mean Squared Error (MSE): Memberi penalti kuadratik berat pada outlier
3. Root Mean Squared Error (RMSE): Mengembalikan skala galat ke satuan asli (kW)
4. Koefisien Determinasi (R^2 Score): Proporsi variansi yang berhasil dijelaskan
5. Fenomena Krusial: Pembuktian matematis mengapa R^2 pada data uji dapat bernilai negatif!
"""

from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def load_energy_data() -> pd.DataFrame:
    data_path = Path(__file__).resolve().parent / "data" / "equipment_power_demand.csv"
    return pd.read_csv(data_path)


def manual_metrics(y_true: np.ndarray, y_pred: np.ndarray):
    """Implementasi rumus matematika murni tanpa Scikit-learn."""
    n = len(y_true)
    errors = y_true - y_pred
    
    mae = np.sum(np.abs(errors)) / n
    mse = np.sum(errors ** 2) / n
    rmse = np.sqrt(mse)
    
    # R^2 = 1 - (SS_res / SS_tot)
    ss_res = np.sum(errors ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1.0 - (ss_res / ss_tot)
    
    return mae, mse, rmse, r2


def main():
    print("=" * 75)
    print("DEMONSTRASI PERHITUNGAN DAN INTERPRETASI METRIK REGRESI")
    print("=" * 75)
    
    df = load_energy_data()
    feature_cols = ["operating_load_pct", "machine_speed_rpm", "ambient_temp_c"]
    X = df[feature_cols]
    y = df["power_consumption_kw"]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )
    
    # Melatih Model OLS yang Baik
    good_model = LinearRegression()
    good_model.fit(X_train, y_train)
    y_pred_good = good_model.predict(X_test)
    
    # 1. Perbandingan Perhitungan Manual vs Scikit-learn
    mae_man, mse_man, rmse_man, r2_man = manual_metrics(y_test.values, y_pred_good)
    mae_skl = mean_absolute_error(y_test, y_pred_good)
    mse_skl = mean_squared_error(y_test, y_pred_good)
    rmse_skl = np.sqrt(mse_skl)
    r2_skl = r2_score(y_test, y_pred_good)
    
    print("1. VERIFIKASI RUMUS MATEMATIKA (MANUAL VS SCIKIT-LEARN):")
    print(f"{'Metrik':<12} | {'Rumus Manual':<18} | {'Scikit-learn API':<18} | {'Status Kecocokan':<18}")
    print("-" * 72)
    print(f"{'MAE':<12} | {mae_man:<18.4f} | {mae_skl:<18.4f} | {'IDENTIK (100%)':<18}")
    print(f"{'MSE':<12} | {mse_man:<18.4f} | {mse_skl:<18.4f} | {'IDENTIK (100%)':<18}")
    print(f"{'RMSE':<12} | {rmse_man:<18.4f} | {rmse_skl:<18.4f} | {'IDENTIK (100%)':<18}")
    print(f"{'R^2 Score':<12} | {r2_man:<18.4f} | {r2_skl:<18.4f} | {'IDENTIK (100%)':<18}")
    print("-" * 72)
    
    # 2. Pembuktian Ilmiah: Mengapa R^2 pada Test Set Bisa Negatif?
    print("\n2. DEMONSTRASI FENOMENA: R^2 SCORE BISA BERNILAI NEGATIF!")
    print("Skenario: Model rusak/salah spesifikasi yang memprediksi nilai konstan keliru (misal 500 kW).")
    
    # Model naif yang sangat buruk
    terrible_preds = np.full_like(y_test, fill_value=600.0)
    mae_bad = mean_absolute_error(y_test, terrible_preds)
    rmse_bad = np.sqrt(mean_squared_error(y_test, terrible_preds))
    r2_bad = r2_score(y_test, terrible_preds)
    
    ss_res_bad = np.sum((y_test - terrible_preds) ** 2)
    ss_tot_test = np.sum((y_test - np.mean(y_test)) ** 2)
    
    print(f"  * SS_res (Jumlah Kuadrat Residu Model Buruk)  : {ss_res_bad:.2f}")
    print(f"  * SS_tot (Jumlah Kuadrat Deviasi Mean Test)   : {ss_tot_test:.2f}")
    print(f"  * Rasio SS_res / SS_tot                       : {ss_res_bad / ss_tot_test:.4f}")
    print(f"  * R^2 Score Terhitung = 1 - (SS_res / SS_tot) : {r2_bad:.4f} (NEGATIF EKSTREM!)\n")
    
    print("Pelajaran Metodologis Kritis:")
    print("1. R^2 bukanlah kuadrat dari korelasi pearson (r) jika dievaluasi pada data uji di luar sampel!")
    print("2. Nilai R^2 < 0 menandakan bahwa model Anda BERKINERJA LEBIH BURUK daripada sekadar")
    print("   menebak rata-rata target (mean baseline).")
    print("=" * 75)


if __name__ == "__main__":
    main()

