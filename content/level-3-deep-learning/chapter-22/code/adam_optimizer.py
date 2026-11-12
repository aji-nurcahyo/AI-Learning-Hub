"""
Chapter 22 - Backpropagation & Optimization
Script 11: Pengoptimal Adam (Adaptive Moment Estimation - Kingma & Ba, 2014)
Menggabungkan keunggulan Momentum (First Moment) dan RMSprop (Second Moment),
dilengkapi koreksi bias analitis pada iterasi awal dan perlindungan stabilitas numerik.
"""

import numpy as np

def rosenbrock_2d(x1, x2):
    return 0.1 * (x1 ** 2) + 2.0 * (x2 ** 2)

def rosenbrock_grad(x1, x2):
    return np.array([0.2 * x1, 4.0 * x2])

def optimize_adam(start_point, lr=0.3, beta1=0.9, beta2=0.999, eps=1e-8, steps=30):
    pos = np.array(start_point, dtype=float)
    m = np.zeros_like(pos)  # First moment vector
    v = np.zeros_like(pos)  # Second moment vector
    trajectory = [pos.copy()]
    
    print(f"{'Iterasi t':<10} {'Posisi [x1, x2]':<25} {'m_hat (1st)':<20} {'v_hat (2nd)':<20} {'Loss':<12}")
    print("-" * 87)
    
    for t in range(1, steps + 1):
        grad = rosenbrock_grad(pos[0], pos[1])
        
        # 1. Update First Moment (Estimasi Nilai Rata-rata Gradien / Momentum)
        m = beta1 * m + (1.0 - beta1) * grad
        
        # 2. Update Second Moment (Estimasi Varians Gradien / RMSprop)
        v = beta2 * v + (1.0 - beta2) * (grad ** 2)
        
        # 3. Koreksi Bias (Menghilangkan bias inisialisasi nol pada t awal)
        m_hat = m / (1.0 - (beta1 ** t))
        v_hat = v / (1.0 - (beta2 ** t))
        
        # 4. Parameter Update
        pos -= (lr / (np.sqrt(v_hat) + eps)) * m_hat
        trajectory.append(pos.copy())
        
        if t in [1, 2, 5, 10, 20, 30]:
            loss = rosenbrock_2d(pos[0], pos[1])
            m_str = f"[{m_hat[0]:.2f}, {m_hat[1]:.2f}]"
            v_str = f"[{v_hat[0]:.2f}, {v_hat[1]:.2f}]"
            print(f"t = {t:<7} [{pos[0]:>7.4f}, {pos[1]:>7.4f}]           {m_str:<20} {v_str:<20} {loss:>8.4f}")
            
    return np.array(trajectory)

def run_adam_demo():
    print("=" * 87)
    print("CHAPTER 22: PENGOPTIMAL ADAM (ADAPTIVE MOMENT ESTIMATION)")
    print("=" * 87)
    
    start_pos = [-4.0, 3.0]
    lr = 0.4
    beta1 = 0.9
    beta2 = 0.999
    
    print(f"Posisi Awal   : {start_pos}")
    print(f"Hyperparameters: lr = {lr}, beta1 = {beta1}, beta2 = {beta2}, eps = 1e-8")
    print("-" * 87)
    
    traj = optimize_adam(start_pos, lr=lr, beta1=beta1, beta2=beta2, steps=30)
    
    print("=" * 87)
    print("PEMBUKTIAN KOREKSI BIAS (BIAS CORRECTION):")
    print("Karena m_0 = 0 dan v_0 = 0, pada t=1 m_1 = 0.1*g1, yang sangat bias ke nol.")
    print("Faktor pembagi (1 - beta1^t) = (1 - 0.9^1) = 0.1 menormalkan m_1 kembali ke skala sejati g1!")
    print(f"Posisi Akhir t=30: [{traj[-1][0]:.6f}, {traj[-1][1]:.6f}] -> Loss = {rosenbrock_2d(traj[-1][0], traj[-1][1]):.6e}")

if __name__ == "__main__":
    run_adam_demo()

