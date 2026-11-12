"""Gradient Descent Optimization & MSE Loss Function Illustration.

Chapter 06: Probability, Statistics & Linear Algebra for AI
Artificial Intelligence Learning Hub - ITERA

Skrip ini mendemonstrasikan algoritma optimasi Gradient Descent secara bertahap:
1. Optimasi Fungsi Kuadratik 1D: f(x) = (x - 3)^2 + 5
2. Pengaruh Nilai Learning Rate (alpha) terhadap Konvergensi (alpha=0.1 vs alpha=0.01 vs alpha=1.05)
3. Ilustrasi Matematis Fungsi Loss: Mean Squared Error (MSE)
   L(w) = (1 / N) * sum((w * x_i - y_i)^2)
   Turunan Analitis: dL/dw = (2 / N) * sum(x_i * (w * x_i - y_i))

Catatan Pedagogis:
Simulasi ini merupakan demonstrasi matematika penerapan turunan dan optimasi numerik,
bukan pembahasan menyeluruh materi algoritma regresi (yang dipelajari di Level 2).
"""

import sys
from typing import List, Tuple

sys.stdout.reconfigure(encoding="utf-8")


def quadratic_f(x: float) -> float:
    """Fungsi objektif: f(x) = (x - 3)^2 + 5, minimum global pada x=3.0, f(x)=5.0."""
    return ((x - 3.0) ** 2) + 5.0


def quadratic_df(x: float) -> float:
    """Turunan pertama: f'(x) = 2(x - 3)."""
    return 2.0 * (x - 3.0)


def gradient_descent_1d(
    start_x: float,
    learning_rate: float,
    iterations: int = 25
) -> List[Tuple[int, float, float, float]]:
    """Menjalankan iterasi Gradient Descent 1D."""
    history = []
    current_x = start_x
    for step in range(iterations + 1):
        loss = quadratic_f(current_x)
        grad = quadratic_df(current_x)
        history.append((step, current_x, grad, loss))
        current_x = current_x - (learning_rate * grad)
    return history


def calculate_mse_and_gradient(w: float, points: List[Tuple[float, float]]) -> Tuple[float, float]:
    """Menghitung nilai Mean Squared Error (MSE) dan gradien analitis dL/dw."""
    n = len(points)
    total_loss = 0.0
    total_grad = 0.0
    for xi, yi in points:
        pred = w * xi
        err = pred - yi
        total_loss += err ** 2
        total_grad += xi * err
    mse = total_loss / n
    grad_w = (2.0 / n) * total_grad
    return mse, grad_w


def run_gradient_descent_experiments():
    print("=" * 70)
    print("EKSPERIMEN 7: GRADIENT DESCENT & ILUSTRASI MSE LOSS FUNCTION")
    print("=" * 70)

    # 1. Optimasi Fungsi Kuadratik 1D
    start_x = -2.0  # Titik awal jauh di kiri minimum (x*=3.0)
    alpha = 0.1     # Learning rate moderat
    max_steps = 20

    print(f"--- 1. Pelacakan Iterasi Gradient Descent (Start x={start_x}, alpha={alpha}) ---")
    print(f"{'Step':<6} | {'x_t':<12} | {'f\'(x_t) Gradien':<18} | {'f(x_t) Loss':<12}")
    print("-" * 55)
    hist = gradient_descent_1d(start_x, alpha, max_steps)
    for step, x_val, grad, loss in hist:
        if step <= 10 or step == max_steps:
            print(f"{step:<6} | {x_val:<12.5f} | {grad:<18.5f} | {loss:<12.5f}")

    final_x = hist[-1][1]
    print(f"\nHasil Akhir setelah {max_steps} langkah: x = {final_x:.4f} (Target x* = 3.0000)")
    print(f"Nilai Minimum yang Dicapai: f(x) = {quadratic_f(final_x):.4f} (Target f* = 5.0000)\n")

    # 2. Pengaruh Nilai Learning Rate
    print("--- 2. Eksperimen Nilai Learning Rate (alpha) ---")
    rates = [0.01, 0.10, 0.90, 1.05]
    for lr in rates:
        trajectory = gradient_descent_1d(start_x, lr, iterations=15)
        res_x = trajectory[-1][1]
        res_loss = trajectory[-1][3]
        behavior = ""
        if lr == 0.01:
            behavior = "Lambat tapi stabil (belum konvergen)"
        elif lr == 0.10:
            behavior = "Ideal & konvergen mulus"
        elif lr == 0.90:
            behavior = "Osilasi kuat tapi tetap konvergen"
        elif lr > 1.0:
            behavior = "DIVERGEN (Overshooting tak terkendali!)"
        print(f"alpha = {lr:<5.2f} -> Step 15: x = {res_x:>10.4f}, Loss = {res_loss:>10.4f} [{behavior}]")

    # 3. Ilustrasi Matematis Fungsi Loss MSE
    # Data sintetis sederhana: y = 2 * x (hubungan ideal)
    sample_data = [(1.0, 2.0), (2.0, 4.0), (3.0, 6.0), (4.0, 8.0)]
    print("\n--- 3. Ilustrasi Optimasi Parameter w pada Mean Squared Error (MSE) ---")
    print(f"Dataset Pengamatan Sederhana: {sample_data}")
    print("Model Linear: y_hat = w * x (Mencari parameter bobot w terbaik)")

    current_w = 0.0  # Bobot awal
    lr_w = 0.05
    print(f"Start w = {current_w:.2f}, alpha = {lr_w}\n")
    print(f"{'Iterasi':<8} | {'Bobot w':<12} | {'dL/dw Gradien':<16} | {'Nilai MSE Loss':<15}")
    print("-" * 60)
    for it in range(16):
        loss_val, grad_val = calculate_mse_and_gradient(current_w, sample_data)
        if it <= 10 or it == 15:
            print(f"{it:<8} | {current_w:<12.5f} | {grad_val:<16.5f} | {loss_val:<15.5f}")
        current_w = current_w - (lr_w * grad_val)

    print(f"\n[KESIMPULAN OPTIMASI]:")
    print(f"Parameter w berhasil konvergen ke w = {current_w:.4f} (Target analitis w* = 2.0).")
    print("Nilai MSE Loss menurun drastis mendekati 0.0 seiring parameter diperbarui.")
    print("Inilah mekanisme matematis paling mendasar di balik pembelajaran mesin!")
    print("=" * 70)


if __name__ == "__main__":
    run_gradient_descent_experiments()

