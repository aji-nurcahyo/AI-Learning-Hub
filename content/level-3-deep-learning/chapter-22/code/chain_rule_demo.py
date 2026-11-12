"""
Chapter 22 - Backpropagation & Optimization
Script 1: Aturan Rantai Kalkulus (Chain Rule Demonstration)
Mendemonstrasikan penerapan aturan rantai diferensiasi pada fungsi komposit:
y = f(u), u = g(v), v = h(x)
dan memverifikasi turunan analitis terhadap aproksimasi turunan numerik.
"""

import numpy as np

def composite_function(x):
    """
    Fungsi komposit 3 tahap:
    v = h(x) = 2*x + 3
    u = g(v) = v^2
    y = f(u) = sin(u)
    """
    v = 2.0 * x + 3.0
    u = v ** 2
    y = np.sin(u)
    return y, u, v

def analytical_derivative_chain_rule(x):
    """
    Menghitung dy/dx menggunakan aturan rantai:
    dy/dx = (dy/du) * (du/dv) * (dv/dx)
    di mana:
      dy/du = cos(u)
      du/dv = 2*v
      dv/dx = 2
    """
    y, u, v = composite_function(x)
    dy_du = np.cos(u)
    du_dv = 2.0 * v
    dv_dx = 2.0
    
    dy_dx = dy_du * du_dv * dv_dx
    return dy_dx, (dy_du, du_dv, dv_dx)

def numerical_derivative(f, x, h=1e-7):
    """Turunan numerik menggunakan central finite difference: (f(x+h) - f(x-h)) / (2h)."""
    y_plus, _, _ = f(x + h)
    y_minus, _, _ = f(x - h)
    return (y_plus - y_minus) / (2.0 * h)

def run_chain_rule_demo():
    print("=" * 80)
    print("CHAPTER 22: DEMONSTRASI ATURAN RANTAI KALKULUS (CHAIN RULE)")
    print("=" * 80)
    print("Komposisi Fungsi: y = sin(u), u = v^2, v = 2x + 3")
    print("Formula Aturan Rantai: dy/dx = (dy/du) * (du/dv) * (dv/dx)")
    print("-" * 80)
    
    test_points = [-1.5, -0.5, 0.0, 0.5, 1.0, 1.5]
    print(f"{'Titik x':<10} {'dy/du (Lokal)':<15} {'du/dv (Lokal)':<15} {'dv/dx (Lokal)':<15} {'dy/dx (Analitis)':<18} {'dy/dx (Numerik)':<18} {'Galat Relatif':<12}")
    print("-" * 105)
    
    for x in test_points:
        grad_analytic, (dy_du, du_dv, dv_dx) = analytical_derivative_chain_rule(x)
        grad_numeric = numerical_derivative(composite_function, x)
        rel_error = abs(grad_analytic - grad_numeric) / (abs(grad_analytic) + abs(grad_numeric) + 1e-15)
        
        print(f"{x:>8.2f}   {dy_du:>12.4f}   {du_dv:>12.4f}   {dv_dx:>12.4f}   {grad_analytic:>15.6f}   {grad_numeric:>15.6f}   {rel_error:>10.2e}")
        
    print("=" * 105)
    print("INSIGHT BACKPROPAGATION:")
    print("Dalam jaringan saraf tiruan, perambatan galat mundur (backward pass) adalah aplikasi langsung")
    print("dari aturan rantai: gradien global dihitung dari perkalian gradien-gradien lokal di setiap node.")

if __name__ == "__main__":
    run_chain_rule_demo()

