"""
Chapter 21 - Neural Network Fundamentals
Script 2: Gerbang Logika Biner Menggunakan Perceptron (Rosenblatt Perceptron)
Membuktikan bahwa Perceptron tunggal dapat merepresentasikan fungsi logika
yang dapat dipisahkan secara linier (Linearly Separable): AND, OR, NAND, dan NOR.
"""

import numpy as np

class PerceptronGate:
    def __init__(self, weights, bias, name="Gate"):
        self.w = np.array(weights, dtype=float)
        self.b = float(bias)
        self.name = name
        
    def forward(self, x):
        z = np.dot(x, self.w) + self.b
        return 1 if z >= 0 else 0

def run_logic_gates():
    print("=" * 75)
    print("CHAPTER 21: IMPLEMENTASI GERBANG LOGIKA DENGAN PERCEPTRON TUNGGAL")
    print("=" * 75)
    
    # Masukan tabel kebenaran 2-bit
    inputs = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ])
    
    # Konfigurasi bobot dan bias analitik untuk tiap gerbang
    # Formula decision boundary: w1*x1 + w2*x2 + b = 0
    gates = [
        PerceptronGate(weights=[1.0, 1.0], bias=-1.5, name="Gerbang AND (x1 AND x2)"),
        PerceptronGate(weights=[1.0, 1.0], bias=-0.5, name="Gerbang OR  (x1 OR x2)"),
        PerceptronGate(weights=[-1.0, -1.0], bias=1.5, name="Gerbang NAND (x1 NAND x2)"),
        PerceptronGate(weights=[-1.0, -1.0], bias=0.5, name="Gerbang NOR  (x1 NOR x2)")
    ]
    
    for gate in gates:
        print(f"\n--- {gate.name} ---")
        print(f"Bobot [w1, w2]: {gate.w.tolist()} | Bias: {gate.b}")
        print(f"Persamaan Batas Keputusan (Hyperplane): {gate.w[0]}*x1 + {gate.w[1]}*x2 + ({gate.b}) = 0")
        print(f"{'x1':<6} {'x2':<6} {'Weighted Sum (z)':<20} {'Prediksi (y)':<12}")
        print("-" * 50)
        
        for x in inputs:
            z = np.dot(x, gate.w) + gate.b
            y_pred = gate.forward(x)
            print(f"{x[0]:<6} {x[1]:<6} {z:>14.2f}       {y_pred:>6}")
            
    print("\n" + "=" * 75)
    print("KESIMPULAN:")
    print("1. Seluruh gerbang di atas memiliki pola kelas yang linearly separable.")
    print("2. Garis batas keputusan tunggal (garis lurus 2D) mampu memisahkan kelas 0 dan 1 dengan sempurna.")
    print("=" * 75)

if __name__ == "__main__":
    run_logic_gates()

