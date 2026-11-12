"""Demonstrasi Hard-Margin vs Soft-Margin SVM & Variabel Slack
Mendemonstrasikan keterbatasan hard-margin pada data ber-derau (infeasible)
dan peran kompensasi variabel slack (xi_i) pada soft-margin.
"""

import numpy as np
from sklearn.svm import SVC

def run_hard_soft_demo():
    print("=== HARD-MARGIN VS SOFT-MARGIN SVM ===")

    # 1. Kasus Data Terpisah Linier Sempurna (Linearly Separable)
    # 2 Fitur: Kekuatan Sinyal (terstandarisasi) dan Paket Loss (terstandarisasi)
    # Kelas 0: Normal, Kelas 1: Kritis
    X_clean = np.array([
        [-2.0, -1.8], [-1.5, -2.1], [-1.8, -1.2], [-1.0, -1.5], # Normal
        [ 1.5,  1.8], [ 2.0,  1.2], [ 1.8,  2.1], [ 1.2,  1.5]  # Kritis
    ])
    y_clean = np.array([0, 0, 0, 0, 1, 1, 1, 1])

    # Hard-Margin didekati dengan nilai C sangat besar (misal C=1e6)
    svm_hard = SVC(kernel="linear", C=1e6).fit(X_clean, y_clean)
    w_hard = svm_hard.coef_[0]
    margin_hard = 2.0 / np.linalg.norm(w_hard)

    print("1. Data Terpisah Linier Murni:")
    print(f"   Hard-Margin SVM (C=1e6) -> Status: KONVERGEN SEMPURNA")
    print(f"   Bobot w: {np.round(w_hard, 4)}, Intercept b: {svm_hard.intercept_[0]:.4f}")
    print(f"   Lebar Margin Geometris (2 / ||w||): {margin_hard:.4f}")
    print(f"   Jumlah Support Vectors: {len(svm_hard.support_)} ({svm_hard.support_})")

    # 2. Injeksi Outlier / Titik Data yang Tumpang Tindih
    # Satu sampel kritis menyusup ke wilayah normal
    X_noisy = np.vstack([X_clean, [-1.2, -1.0]])
    y_noisy = np.append(y_clean, 1) # Outlier Kritis di wilayah Normal

    print("\n2. Injeksi 1 Sampel Outlier / Tumpang Tindih:")

    # Evaluasi Soft-Margin dengan variasi C (Toleransi Slack)
    for c_val in [0.01, 1.0, 100.0, 10000.0]:
        svm_soft = SVC(kernel="linear", C=c_val).fit(X_noisy, y_noisy)
        w = svm_soft.coef_[0]
        margin = 2.0 / np.linalg.norm(w)
        n_sv = len(svm_soft.support_)

        # Hitung signed distance dan margin violations (xi_i = max(0, 1 - y_i * (w.x + b)))
        decisions = svm_soft.decision_function(X_noisy)
        y_binary = np.where(y_noisy == 1, 1, -1)
        slacks = np.maximum(0, 1.0 - y_binary * decisions)
        total_slack = np.sum(slacks)

        print(f"   C = {c_val:<7} | Margin: {margin:.4f} | Total SV: {n_sv:<2} | Total Slack (Pelanggaran): {total_slack:.4f}")

    print("\nKesimpulan Konseptual:")
    print("Hard-Margin SVM gagal/tidak fisibel jika data memiliki tumpang tindih.")
    print("Soft-Margin SVM memperkenalkan variabel slack (xi_i >= 0) yang dikontrol parameter C:")
    print("- C kecil : Mengizinkan margin lebih lebar dengan toleransi pelanggaran lebih banyak (Low Variance, High Bias).")
    print("- C besar : Memaksa margin sempit dengan sedikit pelanggaran batas (High Variance, Low Bias).")

if __name__ == "__main__":
    run_hard_soft_demo()

