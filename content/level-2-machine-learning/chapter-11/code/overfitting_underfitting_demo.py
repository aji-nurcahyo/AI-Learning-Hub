"""
Chapter 11: Introduction to Machine Learning
Script 6: overfitting_underfitting_demo.py

Tujuan:
Mendemonstrasikan spektrum Underfitting vs Good Fit vs Overfitting:
- Fungsi Sejati (Ground Truth): y = sin(x) + derau acak (noise).
- Underfitting (High Bias): Model derajat 1 (garis lurus kaku) gagal menangkap kurva.
- Good Fit (Optimal): Model derajat 3 menangkap kelengkungan kurva secara anggun.
- Overfitting (High Variance): Model derajat 15 meliuk-liuk menghafal setiap titik derau acak.
"""

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


def generate_noisy_sine_data(n_samples: int = 30, seed: int = 42):
    """Membangkitkan data non-linier y = sin(1.5 * pi * x) + noise."""
    np.random.seed(seed)
    X = np.sort(np.random.rand(n_samples))
    y = np.sin(1.5 * np.pi * X) + np.random.normal(0, 0.15, n_samples)
    return X[:, np.newaxis], y


def main():
    print("=" * 65)
    print("DEMO: UNDERFITTING VS GOOD FIT VS OVERFITTING")
    print("=" * 65)

    X_train, y_train = generate_noisy_sine_data(n_samples=25, seed=42)
    X_test, y_test = generate_noisy_sine_data(n_samples=25, seed=100)

    degrees = [1, 3, 15]
    descriptions = [
        "Underfitting (High Bias - Terlalu Sederhana)",
        "Good Fit (Optimal Trade-Off - Menangkap Pola Sejati)",
        "Overfitting (High Variance - Menghafal Noise)"
    ]

    print(f"{'Derajat':<8} | {'Deskripsi Model':<42} | {'Train MSE':<10} | {'Test MSE':<10}")
    print("-" * 76)

    for deg, desc in zip(degrees, descriptions):
        pipeline = Pipeline([
            ("poly", PolynomialFeatures(degree=deg, include_bias=False)),
            ("linear", LinearRegression())
        ])
        pipeline.fit(X_train, y_train)

        train_mse = mean_squared_error(y_train, pipeline.predict(X_train))
        test_mse = mean_squared_error(y_test, pipeline.predict(X_test))

        print(f"Deg {deg:<4} | {desc:<42} | {train_mse:<10.4f} | {test_mse:<10.4f}")

    print("-" * 76)
    print("\n--- Analisis Teoretis ---")
    print("1. Derajat 1: Train MSE tinggi dan Test MSE tinggi -> Model tidak mampu belajar (Underfit).")
    print("2. Derajat 3: Train MSE rendah dan Test MSE rendah -> Model generalisasi optimal.")
    print("3. Derajat 15: Train MSE mendekati nol tetapi Test MSE meledak -> Model menghafal noise (Overfit).")
    print("=" * 65)


if __name__ == "__main__":
    main()

