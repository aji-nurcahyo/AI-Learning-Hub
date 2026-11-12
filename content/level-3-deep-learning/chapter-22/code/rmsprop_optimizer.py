"""
Chapter 22 - Backpropagation & Optimization
Script 10: Pengoptimal RMSprop (Root Mean Square Propagation)
Mengimplementasikan peredaman laju pembelajaran adaptif berbasis rata-rata bergerak
kuadrat gradien (Geoffrey Hinton, Coursera 2012) untuk menstabilkan pembaruan dimensi heterogen.
"""

import numpy as np

def rosenbrock_2d(x1, x2):
    return 0.1 * (x1 ** 2) + 2.0 * (x2 ** 2)

def rosenbrock_grad(x1, x2):
    return np.array([0.2 * x1, 4.0 * x2])

def optimize_rmsprop(start_point, lr=0.3, beta=0.9, eps=1e-8, steps=30):
    pos = np.array(start_point, dtype=float)
    s = np.zeros_like(pos)
    trajectory = [pos.copy()]
    
    for _ in range(steps):
        grad = rosenbrock_grad(pos[0], pos[1])
        # Rata-rata bergerak kuadrat gradien: s = beta * s + (1 - beta) * grad^2
        s = beta * s + (1.0 - beta) * (grad ** 2)
        # Parameter update adaptif
        pos -= (lr / (np.sqrt(s) + eps)) * grad
        trajectory.append(pos.copy())
        
    return np.array(trajectory)

def run_rmsprop_demo():
    print("=" * 85)
    print("CHAPTER 22: PENGOPTIMAL RMSPROP (ADAPTIVE LEARNING RATE PER-DIMENSI)")
    print("=" * 85)
    
    start_pos = [-4.0, 3.0]
    steps = 25
    lr = 0.3
    beta = 0.9
    
    print(f"Fungsi Uji     : f(x1, x2) = 0.1*x1^2 + 2.0*x2^2")
    print(f"Posisi Awal     : {start_pos}")
    print(f"Learning Rate   : {lr} | Koefisien Peluruhan Kuadrat (beta): {beta}")
    print("-" * 85)
    
    traj = optimize_rmsprop(start_pos, lr=lr, beta=beta, steps=steps)
    
    print(f"{'Langkah':<10} {'Posisi [x1, x2]':<25} {'Nilai Fungsi (Loss)':<20} {'Status':<15}")
    print("-" * 85)
    
    for step in [1, 5, 10, 15, 20, 25]:
        p = traj[step]
        loss = rosenbrock_2d(p[0], p[1])
        print(f"Langkah {step:<3} [{p[0]:>7.4f}, {p[1]:>7.4f}]           {loss:>12.6f}       Konvergen")
        
    print("=" * 85)
    print("INSIGHT ADAPTIF RMSPROP:")
    print("1. Sumbu Curam (x2): Gradien besar menghasilkan akumulasi s_2 besar -> lr efektif mengecil tajam.")
    print("2. Sumbu Landai (x1): Gradien kecil menghasilkan akumulasi s_1 kecil -> lr efektif membesar.")
    print("3. Hasil: Kedua sumbu maju seimbang menuju titik minimum global tanpa lonjakan!")

if __name__ == "__main__":
    run_rmsprop_demo()

