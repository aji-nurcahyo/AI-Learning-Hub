"""
Chapter 11: Introduction to Machine Learning
Script 7: baseline_comparison.py

Tujuan:
Membuktikan secara kuantitatif pentingnya Baseline Model dalam evaluasi:
- Pada dataset yang tidak seimbang (Class Imbalance, misal 95% Negatif, 5% Positif),
  sebuah model sepele yang selalu menebak 'Negatif' akan memperoleh akurasi 95%.
- Klaim akurasi 95% sama sekali tidak berguna tanpa pembanding baseline.
- Menggunakan DummyClassifier dari Scikit-learn untuk membandingkan model terhadap baseline.
"""

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, balanced_accuracy_score


def main():
    print("=" * 65)
    print("DEMO: PERAN KRUSIAL BASELINE DALAM EVALUASI MODEL")
    print("=" * 65)

    # 1. Dataset deteksi penipuan transaksi (Fraud Detection) sangat timpang
    # 95 transaksi sah (0), 5 transaksi curang/fraud (1)
    np.random.seed(42)
    n_samples = 200
    n_fraud = 10
    n_legit = n_samples - n_fraud

    X_legit = np.random.normal(loc=0.0, scale=1.0, size=(n_legit, 2))
    y_legit = np.zeros(n_legit, dtype=int)

    X_fraud = np.random.normal(loc=2.5, scale=0.8, size=(n_fraud, 2))
    y_fraud = np.ones(n_fraud, dtype=int)

    X = np.vstack([X_legit, X_fraud])
    y = np.concatenate([y_legit, y_fraud])

    print(f"Total Sampel: {n_samples}")
    print(f"Distribusi Kelas: Legit (0) = {n_legit} ({n_legit/n_samples*100:.1f}%), Fraud (1) = {n_fraud} ({n_fraud/n_samples*100:.1f}%)")

    # 2. Baseline Model 1: Tebak Mayoritas Selalu (Dummy 'most_frequent')
    dummy_majority = DummyClassifier(strategy="most_frequent")
    dummy_majority.fit(X, y)
    y_pred_dummy = dummy_majority.predict(X)

    acc_dummy = accuracy_score(y, y_pred_dummy)
    bal_acc_dummy = balanced_accuracy_score(y, y_pred_dummy)

    print("\n--- 1. Baseline Model (Tebak Mayoritas Trivial) ---")
    print(f"Akurasi Mentah (Raw Accuracy)   : {acc_dummy * 100:.2f}%  <-- Tampak sangat tinggi!")
    print(f"Balanced Accuracy               : {bal_acc_dummy * 100:.2f}%  <-- Menunjukkan kegagalan sejati (0% fraud terdeteksi)")

    # 3. Model Machine Learning: Logistic Regression
    lr_model = LogisticRegression(class_weight="balanced", random_state=42)
    lr_model.fit(X, y)
    y_pred_lr = lr_model.predict(X)

    acc_lr = accuracy_score(y, y_pred_lr)
    bal_acc_lr = balanced_accuracy_score(y, y_pred_lr)

    print("\n--- 2. Machine Learning Model (Logistic Regression + Class Weight) ---")
    print(f"Akurasi Mentah (Raw Accuracy)   : {acc_lr * 100:.2f}%")
    print(f"Balanced Accuracy               : {bal_acc_lr * 100:.2f}%  <-- Berhasil mendeteksi fraud!")

    print("\nLaporan Klasifikasi ML Model:")
    print(classification_report(y, y_pred_lr, target_names=["Legit (0)", "Fraud (1)"]))

    print("--- Kesimpulan Penting ---")
    print("Jangan pernah mengevaluasi akurasi model dalam ruang hampa.")
    print("Selalu bandingkan metrik model Anda dengan baseline yang paling sederhana.")
    print("Akurasi 95% tidak berarti apa-apa jika baseline mayoritas juga 95%!")
    print("=" * 65)


if __name__ == "__main__":
    main()

