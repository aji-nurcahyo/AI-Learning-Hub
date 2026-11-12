"""Calculus Foundations for AI: Derivatives, Partial Derivatives & Gradients.

Chapter 06: Probability, Statistics & Linear Algebra for AI
Artificial Intelligence Learning Hub - ITERA

Skrip ini mendemonstrasikan konsep kalkulus diferensial pengantar untuk optimasi AI:
1. Kemiringan Garis Singgung (Slope of Tangent Line)
2. Perbandingan Turunan Analitis vs Aproksimasi Numerik (Central Difference)
3. Pencarian Titik Stasioner / Titik Ekstrim (Minimum)
4. Turunan Parsial (Partial Derivatives) untuk Fungsi Multivariat
5. Vektor Gradien (Gradient Vector) dan Arah Penurunan Tercuram
6. Konsep Pengantar Aturan Rantai (Chain Rule)
"""

import math
import sys
from typing import Callable, Tuple

sys.stdout.reconfigure(encoding="utf-8")


def numerical_derivative(f: Callable[[float], float], x: float, h: float = 1e-6) -> float:
    """Menghitung aproksimasi turunan pertama menggunakan Central Difference Method:
    f'(x) ≈ (f(x + h) - f(x - h)) / (2 * h)
    """
    return (f(x + h) - f(x - h)) / (2.0 * h)


def partial_derivative_x(
    f: Callable[[float, float], float], x: float, y: float, h: float = 1e-6
) -> float:
    """Menghitung turunan parsial terhadap x: ∂f/∂x."""
    return (f(x + h, y) - f(x - h, y)) / (2.0 * h)


def partial_derivative_y(
    f: Callable[[float, float], float], x: float, y: float, h: float = 1e-6
) -> float:
    """Menghitung turunan parsial terhadap y: ∂f/∂y."""
    return (f(x, y + h) - f(x, y - h)) / (2.0 * h)


def gradient_2d(
    f: Callable[[float, float], float], x: float, y: float, h: float = 1e-6
) -> Tuple[float, float]:
    """Menghitung vektor gradien ∇f(x, y) = [∂f/∂x, ∂f/∂y]."""
    df_dx = partial_derivative_x(f, x, y, h)
    df_dy = partial_derivative_y(f, x, y, h)
    return df_dx, df_dy


def run_derivative_experiments():
    print("=" * 70)
    print("EKSPERIMEN 6: TURUNAN, TURUNAN PARSIAL & VEKTOR GRADIEN")
    print("=" * 70)

    # 1. Fungsi Univariat: f(x) = x^2 - 4x + 7
    # Turunan Analitis: f'(x) = 2x - 4
    # Titik stasioner f'(x) = 0 terjadi pada x = 2
    f_univariate = lambda x: (x ** 2) - (4.0 * x) + 7.0
    f_prime_analytical = lambda x: (2.0 * x) - 4.0

    print("--- 1. Turunan Fungsi Univariat f(x) = x² - 4x + 7 ---")
    test_points = [-1.0, 0.0, 1.0, 2.0, 3.0, 4.0]
    print(f"{'x':<8} | {'f(x)':<10} | {'f_analitis':<12} | {'f_numerik':<12} | {'Galat Absolut':<15}")
    print("-" * 65)
    for x in test_points:
        fx = f_univariate(x)
        ana = f_prime_analytical(x)
        num = numerical_derivative(f_univariate, x)
        err = abs(ana - num)
        print(f"{x:<8.1f} | {fx:<10.2f} | {ana:<12.4f} | {num:<12.4f} | {err:<15.2e}")

    print("\n[ANALISIS TITIK MINIMUM]:")
    print("Pada x = 2.0, nilai f'(x) = 0.0 (kemiringan datar).")
    print(f"Nilai minimum fungsi adalah f(2.0) = {f_univariate(2.0):.2f}.")
    print("Jika x < 2, kemiringan negatif (fungsi menurun).")
    print("Jika x > 2, kemiringan positif (fungsi menaik).\n")

    # 2. Fungsi Multivariat: g(x, y) = x^2 + 2y^2 - 4x - 8y + 15
    # Turunan Parsial Analitis:
    # ∂g/∂x = 2x - 4
    # ∂g/∂y = 4y - 8
    # Gradien nol (titik stasioner) pada (x=2, y=2)
    g_multivariate = lambda x, y: (x ** 2) + (2.0 * (y ** 2)) - (4.0 * x) - (8.0 * y) + 15.0

    print("--- 2. Turunan Parsial & Vektor Gradien g(x, y) = x² + 2y² - 4x - 8y + 15 ---")
    multi_test_points = [(0.0, 0.0), (1.0, 1.0), (2.0, 2.0), (3.0, 4.0)]
    print(f"{'(x, y)':<12} | {'g(x, y)':<10} | {'∇g_analitis':<20} | {'∇g_numerik':<20}")
    print("-" * 68)
    for x, y in multi_test_points:
        val = g_multivariate(x, y)
        grad_ana = ((2.0 * x) - 4.0, (4.0 * y) - 8.0)
        grad_num = gradient_2d(g_multivariate, x, y)
        pt_str = f"({x:.1f}, {y:.1f})"
        ana_str = f"[{grad_ana[0]:.2f}, {grad_ana[1]:.2f}]"
        num_str = f"[{grad_num[0]:.2f}, {grad_num[1]:.2f}]"
        print(f"{pt_str:<12} | {val:<10.2f} | {ana_str:<20} | {num_str:<20}")

    print("\n[INSIGHT VEKTOR GRADIEN DALAM AI]:")
    print("Vektor gradien ∇g selalu menunjuk ke arah kenaikan fungsi tercepat.")
    print("Oleh karena itu, algoritma Gradient Descent melangkah ke arah -∇g (negatif gradien)")
    print("untuk menemukan parameter model yang meminimalkan nilai Loss Function!")
    print("=" * 70)


if __name__ == "__main__":
    run_derivative_experiments()

