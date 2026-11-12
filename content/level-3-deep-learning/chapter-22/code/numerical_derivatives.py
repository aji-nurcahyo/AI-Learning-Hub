"""
Chapter 22 - Backpropagation & Optimization
Script 3: Turunan Numerik & Analisis Galat Pemotongan (Numerical Derivatives)
Membandingkan metode beda hingga maju (Forward Difference), mundur (Backward Difference),
dan terpusat (Central Difference), serta menganalisis trade-off antara galat pemotongan
(truncation error) dan galat pembulatan mesin (roundoff error).
"""

import numpy as np

def target_func(x):
    # Fungsi uji non-linier: f(x) = exp(x) * sin(x)
    return np.exp(x) * np.sin(x)

def analytical_derivative(x):
    # f'(x) = exp(x)*sin(x) + exp(x)*cos(x) = exp(x)*(sin(x) + cos(x))
    return np.exp(x) * (np.sin(x) + np.cos(x))

def forward_difference(f, x, eps):
    return (f(x + eps) - f(x)) / eps

def backward_difference(f, x, eps):
    return (f(x) - f(x - eps)) / eps

def central_difference(f, x, eps):
    return (f(x + eps) - f(x - eps)) / (2.0 * eps)

def run_derivative_analysis():
    print("=" * 85)
    print("CHAPTER 22: ANALISIS METODE BEDA HINGGA NUMERIK & AKURASI ORDE")
    print("=" * 85)
    x0 = 1.0
    true_deriv = analytical_derivative(x0)
    print(f"Fungsi Uji: f(x) = exp(x) * sin(x) pada x = {x0}")
    print(f"Turunan Analitis Sejati f'({x0}) = {true_deriv:.10f}")
    print("-" * 85)
    
    epsilons = [1e-1, 1e-2, 1e-4, 1e-6, 1e-8, 1e-10, 1e-12, 1e-14, 1e-16]
    print(f"{'Epsilon (eps)':<15} {'Forward Diff':<22} {'Central Diff':<22} {'Galat Forward':<15} {'Galat Central':<15}")
    print("-" * 85)
    
    for eps in epsilons:
        fd = forward_difference(target_func, x0, eps)
        cd = central_difference(target_func, x0, eps)
        
        err_fd = abs(fd - true_deriv)
        err_cd = abs(cd - true_deriv)
        
        print(f"{eps:<15.1e} {fd:>18.10f}   {cd:>18.10f}   {err_fd:>12.2e}    {err_cd:>12.2e}")
        
    print("=" * 85)
    print("KESIMPULAN METODOLOGIS:")
    print("1. Orde Akurasi: Central Difference memiliki akurasi O(eps^2) jauh lebih presisi dari Forward O(eps).")
    print("2. Titik Optimal: Nilai eps ~ 1e-7 hingga 1e-8 memberikan galat terendah (~10^-11) pada presisi float64.")
    print("3. Roundoff Error: Jika eps terlalu kecil (eps <= 1e-14), pembatalan katastropik (catastrophic cancellation)")
    print("   pada pembilang menghasilkan galat numerik besar atau nol.")

if __name__ == "__main__":
    run_derivative_analysis()

