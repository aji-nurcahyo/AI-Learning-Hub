"""
Chapter 11: Introduction to Machine Learning
Script 3: supervised_learning_demo.py

Tujuan:
Mengenalkan dua tugas pokok dalam Supervised Learning:
1. Regresi: Memprediksi kuantitas numerik kontinu (Continuous Output).
2. Klasifikasi: Memprediksi kategori diskrit (Discrete Class Label).

Menggunakan implementasi model linier sederhana Scikit-learn sebagai ilustrasi.
"""

import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, confusion_matrix


def run_regression_example():
    print("\n--- 1. TUGAS REGRESI (PREDIKSI KONTINU) ---")
    # Kasus: Memprediksi konsumsi daya listrik server (Watt) berdasarkan beban CPU (%) dan Memori (%)
    # X: [CPU Load %, RAM Usage %]
    X_train = np.array([
        [15.0, 25.0], [25.0, 35.0], [40.0, 45.0], [55.0, 60.0],
        [70.0, 75.0], [85.0, 80.0], [95.0, 90.0], [30.0, 40.0]
    ])
    # y: Konsumsi Daya (Watt) kontinu
    y_train = np.array([120.5, 145.0, 180.2, 215.0, 250.8, 285.4, 310.0, 155.0])

    model_reg = LinearRegression()
    model_reg.fit(X_train, y_train)

    # Uji pada sampel baru
    X_test = np.array([[50.0, 55.0], [80.0, 85.0]])
    y_pred = model_reg.predict(X_test)

    print(f"Koefisien Bobot Fitur (Weights w): {model_reg.coef_}")
    print(f"Intersep Bias (b): {model_reg.intercept_:.2f}")
    for i, (sample, pred) in enumerate(zip(X_test, y_pred)):
        print(f"  Sampel Uji #{i+1} [CPU: {sample[0]}%, RAM: {sample[1]}%] -> Prediksi Daya: {pred:.2f} Watt")

    train_pred = model_reg.predict(X_train)
    mse = mean_squared_error(y_train, train_pred)
    r2 = r2_score(y_train, train_pred)
    print(f"Metrik Evaluasi: Mean Squared Error (MSE) = {mse:.2f}, R-squared (R2) = {r2:.4f}")


def run_classification_example():
    print("\n--- 2. TUGAS KLASIFIKASI (PREDIKSI KATEGORI DISKRIT) ---")
    # Kasus: Deteksi Anomali Suhu Ruang Server (0: Normal, 1: Anomali/Overheat)
    # X: [Suhu degC, Kecepatan Kipas RPM]
    X_train = np.array([
        [24.0, 1500], [25.5, 1600], [26.0, 1550], [27.0, 1650], [25.0, 1580],  # Normal
        [38.0, 3000], [42.5, 3200], [40.0, 3100], [45.0, 3400], [39.5, 2900]   # Anomali
    ])
    y_train = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

    model_clf = LogisticRegression()
    model_clf.fit(X_train, y_train)

    # Uji pada sampel baru
    X_test = np.array([[26.5, 1620], [41.0, 3150]])
    test_labels = [0, 1]
    y_pred = model_clf.predict(X_test)
    y_prob = model_clf.predict_proba(X_test)

    for i, (sample, pred, prob) in enumerate(zip(X_test, y_pred, y_prob)):
        kategori = "ANOMALI" if pred == 1 else "NORMAL"
        print(f"  Sampel Uji #{i+1} [Suhu: {sample[0]} C, Kipas: {sample[1]} RPM]:")
        print(f"    -> Prediksi: Kelas {pred} ({kategori}) | Probabilitas [Normal: {prob[0]:.3f}, Anomali: {prob[1]:.3f}]")

    train_acc = accuracy_score(y_train, model_clf.predict(X_train))
    print(f"Akurasi Latih Model: {train_acc * 100:.1f}%")


def main():
    print("=" * 65)
    print("DEMO: SUPERVISED LEARNING (REGRESI VS KLASIFIKASI)")
    print("=" * 65)
    run_regression_example()
    run_classification_example()
    print("=" * 65)


if __name__ == "__main__":
    main()

