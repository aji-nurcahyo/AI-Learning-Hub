"""
Chapter 22 - Backpropagation & Optimization
Script 12: Eksperimen Komparasi Laju Pembelajaran (Learning Rate Comparison)
Mensimulasikan dinamika konvergensi optimasi pada berbagai nilai learning rate (eta):
terlalu kecil (under-stepping/slow), optimal (rapid convergence), dan terlalu besar (overshooting/divergence).
"""

import numpy as np

def quadratic_loss(theta):
    # Permukaan kuadratik 1D: J(theta) = 2.5 * theta^2
    return 2.5 * (theta ** 2)

def quadratic_grad(theta):
    return 5.0 * theta

def run_learning_rate_experiment():
    print("=" * 85)
    print("CHAPTER 22: PENGARUH SELEKSI LEARNING RATE TERHADAP DINAMIKA KONVERGENSI")
    print("=" * 85)
    print("Fungsi Uji: J(theta) = 2.5 * theta^2 (Titik Minimum Global: theta = 0, J = 0)")
    print("Posisi Awal theta_0 = 5.0")
    print("-" * 85)
    
    # 4 Skenario Learning Rate:
    # Syarat kestabilan konvergensi untuk kurva f(x) = (1/2)*L*x^2 adalah lr < 2/L.
    # Di sini L = 5.0, sehingga batas stabilitas kritis mutlak adalah lr < 2/5 = 0.40!
    lr_scenarios = [
        {"name": "Terlalu Kecil (lr = 0.01)", "lr": 0.01, "expected": "Lambat merayap (Slow)"},
        {"name": "Moderat (lr = 0.10)",       "lr": 0.10, "expected": "Konvergen mulus (Smooth)"},
        {"name": "Optimal Cepat (lr = 0.35)", "lr": 0.35, "expected": "Konvergen sangat cepat"},
        {"name": "Terlalu Besar (lr = 0.45)", "lr": 0.45, "expected": "Overshooting & Divergensi"}
    ]
    
    steps = 10
    print(f"{'Langkah':<8} {'lr = 0.01':<18} {'lr = 0.10':<18} {'lr = 0.35':<18} {'lr = 0.45 (Divergen)':<22}")
    print("-" * 85)
    
    positions = {s["name"]: 5.0 for s in lr_scenarios}
    
    for step in range(steps + 1):
        line = f"Step {step:<3}"
        for s in lr_scenarios:
            val = positions[s["name"]]
            line += f"  {val:>14.4f}  "
        print(line)
        
        # Update langkah berikutnya
        for s in lr_scenarios:
            grad = quadratic_grad(positions[s["name"]])
            positions[s["name"]] -= s["lr"] * grad
            
    print("=" * 85)
    print("DIAGNOSTIK TEORETIS BATAS KESTABILAN (LIPSCHITZ CONTINUOUS GRADIENTS):")
    print("1. Jika eta < 2/L (eta = 0.35): theta_{t+1} = theta_t - 0.35*(5*theta_t) = (1 - 1.75)*theta_t = -0.75*theta_t -> Konvergen!")
    print("2. Jika eta > 2/L (eta = 0.45): theta_{t+1} = (1 - 0.45*5)*theta_t = (1 - 2.25)*theta_t = -1.25*theta_t.")
    print("   Nilai dikalikan faktor |1.25| > 1 di setiap langkah -> Meledak divergen menuju tak hingga!")

if __name__ == "__main__":
    run_learning_rate_experiment()

