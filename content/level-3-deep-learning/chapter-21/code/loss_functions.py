"""
Chapter 21 - Neural Network Fundamentals
Script 9: Fungsi Rugi (Loss Functions & Numerical Stability)
Mengimplementasikan fungsi rugi utama: Mean Squared Error (MSE),
Binary Cross-Entropy (BCE), dan Categorical Cross-Entropy (CCE),
dengan teknik stabilisasi numerik pemotongan epsilon (epsilon clipping).
"""

import numpy as np

def mean_squared_error(y_true, y_pred):
    """
    Menghitung MSE: L = (1 / 2N) * sum((y - y_hat)^2) atau (1 / N) * sum((y - y_hat)^2).
    Standar machine learning: (1 / N) * sum((y - y_hat)^2).
    """
    y_t = np.asarray(y_true, dtype=float)
    y_p = np.asarray(y_pred, dtype=float)
    return np.mean((y_t - y_p) ** 2)

def binary_cross_entropy(y_true, y_pred, eps=1e-15):
    """
    Menghitung BCE stabil numerik:
    L = - (1 / N) * sum(y * log(p) + (1 - y) * log(1 - p))
    Menggunakan np.clip untuk mencegah log(0) yang menghasilkan -inf atau NaN.
    """
    y_t = np.asarray(y_true, dtype=float)
    y_p = np.asarray(y_pred, dtype=float)
    
    # Potong probabilitas ke rentang aman [eps, 1 - eps]
    y_p_clipped = np.clip(y_p, eps, 1.0 - eps)
    loss = - np.mean(y_t * np.log(y_p_clipped) + (1.0 - y_t) * np.log(1.0 - y_p_clipped))
    return loss

def categorical_cross_entropy(y_true_onehot, y_pred_probs, eps=1e-15):
    """
    Menghitung CCE stabil numerik untuk klasifikasi multikelas:
    L = - (1 / N) * sum_i sum_k y_{i,k} * log(p_{i,k})
    """
    y_t = np.asarray(y_true_onehot, dtype=float)
    y_p = np.asarray(y_pred_probs, dtype=float)
    
    y_p_clipped = np.clip(y_p, eps, 1.0 - eps)
    loss = - np.mean(np.sum(y_t * np.log(y_p_clipped), axis=-1))
    return loss

def run_loss_demonstration():
    print("=" * 80)
    print("CHAPTER 21: EVALUASI FUNGSI RUGI (LOSS FUNCTIONS) & STABILITAS NUMERIK")
    print("=" * 80)
    
    # 1. Evaluasi Mean Squared Error (Kasus Regresi)
    y_true_reg = np.array([10.5, 20.0, 15.2, 8.4])
    y_pred_close = np.array([10.8, 19.5, 15.0, 8.9])  # Prediksi dekat
    y_pred_far = np.array([18.0, 12.0, 25.0, 2.0])    # Prediksi jauh
    
    mse_close = mean_squared_error(y_true_reg, y_pred_close)
    mse_far = mean_squared_error(y_true_reg, y_pred_far)
    
    print("1. Regresi - Mean Squared Error (MSE):")
    print(f"   Target Aktual                 : {y_true_reg}")
    print(f"   Prediksi Dekat                : {y_pred_close} -> MSE = {mse_close:.4f}")
    print(f"   Prediksi Jauh                 : {y_pred_far} -> MSE = {mse_far:.4f}")
    
    # 2. Evaluasi Binary Cross-Entropy (Kasus Klasifikasi Biner)
    # y=1 (Kerusakan), y=0 (Normal)
    y_true_bin = np.array([1.0, 0.0, 1.0, 0.0])
    y_pred_confident_correct = np.array([0.95, 0.05, 0.90, 0.10])
    y_pred_uncertain = np.array([0.55, 0.45, 0.60, 0.50])
    y_pred_confident_wrong = np.array([0.05, 0.95, 0.10, 0.90])  # Kesalahan fatal dengan keyakinan tinggi
    
    bce_correct = binary_cross_entropy(y_true_bin, y_pred_confident_correct)
    bce_uncertain = binary_cross_entropy(y_true_bin, y_pred_uncertain)
    bce_wrong = binary_cross_entropy(y_true_bin, y_pred_confident_wrong)
    
    print("\n2. Klasifikasi Biner - Binary Cross-Entropy (BCE):")
    print(f"   Target Biner Aktual           : {y_true_bin}")
    print(f"   Prediksi Yakin & Benar        : {y_pred_confident_correct} -> BCE = {bce_correct:.4f}")
    print(f"   Prediksi Ragu-Ragu            : {y_pred_uncertain} -> BCE = {bce_uncertain:.4f}")
    print(f"   Prediksi Yakin & SALAH FATAL  : {y_pred_confident_wrong} -> BCE = {bce_wrong:.4f}")
    
    # 3. Evaluasi Categorical Cross-Entropy (Kasus Multikelas)
    y_true_mc = np.array([
        [1, 0, 0],  # Kelas 0
        [0, 1, 0]   # Kelas 1
    ])
    y_pred_mc_good = np.array([
        [0.85, 0.10, 0.05],
        [0.05, 0.90, 0.05]
    ])
    y_pred_mc_bad = np.array([
        [0.10, 0.80, 0.10],
        [0.70, 0.20, 0.10]
    ])
    cce_good = categorical_cross_entropy(y_true_mc, y_pred_mc_good)
    cce_bad = categorical_cross_entropy(y_true_mc, y_pred_mc_bad)
    
    print("\n3. Klasifikasi Multikelas - Categorical Cross-Entropy (CCE):")
    print(f"   Prediksi Baik                 : CCE = {cce_good:.4f}")
    print(f"   Prediksi Buruk                : CCE = {cce_bad:.4f}")
    
    print("=" * 80)
    print("INSIGHT TEORETIS:")
    print("Cross-entropy menghukum prediksi salah yang percaya diri (confident wrong) secara logaritmik.")
    print("Jika p = 0 sementara y = 1, -log(0) -> inf. Teknik clipping mencegah program crash akibat floating-point NaN.")

if __name__ == "__main__":
    run_loss_demonstration()

