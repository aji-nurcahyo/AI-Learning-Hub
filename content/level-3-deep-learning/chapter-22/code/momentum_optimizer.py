"""
Chapter 22 - Backpropagation & Optimization
Script 9: Pengoptimal Momentum (Momentum Optimizer)
Mengimplementasikan akumulasi kecepatan gradien eksponensial (Exponentially Weighted Moving Average)
untuk meredam osilasi transversal dan mempercepat penelusuran lembah landai (Ravines).
"""

import numpy as np

def rosenbrock_2d(x1, x2):
    # Fungsi uji jurang anisotropic: f(x1, x2) = 0.1 * x1^2 + 2.0 * x2^2
    # Memiliki kelengkungan sangat curam di sumbu x2 dan landai di sumbu x1
    return 0.1 * (x1 ** 2) + 2.0 * (x2 ** 2)

def rosenbrock_grad(x1, x2):
    df_dx1 = 0.2 * x1
    df_dx2 = 4.0 * x2
    return np.array([df_dx1, df_dx2])

def optimize_standard_sgd(start_point, lr=0.4, steps=30):
    pos = np.array(start_point, dtype=float)
    trajectory = [pos.copy()]
    for _ in range(steps):
        grad = rosenbrock_grad(pos[0], pos[1])
        pos -= lr * grad
        trajectory.append(pos.copy())
    return np.array(trajectory)

def optimize_momentum_sgd(start_point, lr=0.4, beta=0.85, steps=30):
    pos = np.array(start_point, dtype=float)
    v = np.zeros_like(pos)
    trajectory = [pos.copy()]
    for _ in range(steps):
        grad = rosenbrock_grad(pos[0], pos[1])
        # Akumulasi kecepatan (Velocity): v = beta * v + (1 - beta) * grad
        v = beta * v + (1.0 - beta) * grad
        pos -= lr * v
        trajectory.append(pos.copy())
    return np.array(trajectory)

def run_momentum_comparison():
    print("=" * 85)
    print("CHAPTER 22: PENGOPTIMAL MOMENTUM VS STANDARD GRADIENT DESCENT PADA ANISOTROPIC SURFACE")
    print("=" * 85)
    
    start_pos = [-4.0, 3.0]
    steps = 25
    lr = 0.35
    beta = 0.8
    
    print(f"Fungsi Uji: f(x1, x2) = 0.1*x1^2 + 2.0*x2^2 (Lembah curam di x2, landai di x1)")
    print(f"Posisi Awal   : {start_pos}")
    print(f"Learning Rate : {lr} | Koefisien Momentum (beta): {beta}")
    print("-" * 85)
    
    traj_sgd = optimize_standard_sgd(start_pos, lr=lr, steps=steps)
    traj_mom = optimize_momentum_sgd(start_pos, lr=lr, beta=beta, steps=steps)
    
    print(f"{'Langkah':<10} {'Posisi SGD Standar':<25} {'Loss SGD':<15} {'Posisi Momentum':<25} {'Loss Momentum':<15}")
    print("-" * 85)
    
    for step in [1, 5, 10, 15, 20, 25]:
        p_sgd = traj_sgd[step]
        loss_sgd = rosenbrock_2d(p_sgd[0], p_sgd[1])
        
        p_mom = traj_mom[step]
        loss_mom = rosenbrock_2d(p_mom[0], p_mom[1])
        
        str_sgd = f"[{p_sgd[0]:>6.2f}, {p_sgd[1]:>6.2f}]"
        str_mom = f"[{p_mom[0]:>6.2f}, {p_mom[1]:>6.2f}]"
        print(f"Langkah {step:<3} {str_sgd:<25} {loss_sgd:>10.4f}      {str_mom:<25} {loss_mom:>10.4f}")
        
    print("=" * 85)
    print("ANALISIS MEKANIKA MOMENTUM:")
    print("1. Standar SGD: Mengalami osilasi liar bolak-balik pada sumbu curam x2 dan bergerak lambat di sumbu x1.")
    print("2. Momentum SGD: Komponen osilasi positif-negatif saling membatalkan di sumbu x2, sementara dorongan searah")
    print("   di sumbu x1 terakumulasi cepat menembus dasar lembah.")

if __name__ == "__main__":
    run_momentum_comparison()

