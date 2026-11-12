"""
Chapter 21 - Neural Network Fundamentals
Script 3: Keterbatasan Perceptron Tunggal pada Masalah Non-Linier (XOR Problem)
Membuktikan secara aljabar kontradiksi pertidaksamaan linier Perceptron pada gerbang XOR,
serta menunjukkan solusi arsitektur Multilayer Perceptron (MLP) 2-layer.
"""

import numpy as np

def prove_xor_linear_impossibility():
    print("=" * 80)
    print("CHAPTER 21: BUKTI KONTRADIKSI MATEMATIS KETERBATASAN PERCEPTRON PADA XOR")
    print("=" * 80)
    print("Tabel Kebenaran XOR:")
    print("  (0, 0) -> y = 0")
    print("  (0, 1) -> y = 1")
    print("  (1, 0) -> y = 1")
    print("  (1, 1) -> y = 0")
    print("\nJika Perceptron tunggal dapat memisahkan XOR, maka harus ada w1, w2, dan b sehingga:")
    print("  1) x=(0,0): w1*(0) + w2*(0) + b < 0   ==>  b < 0")
    print("  2) x=(0,1): w1*(0) + w2*(1) + b >= 0  ==>  w2 + b >= 0")
    print("  3) x=(1,0): w1*(1) + w2*(0) + b >= 0  ==>  w1 + b >= 0")
    print("  4) x=(1,1): w1*(1) + w2*(1) + b < 0   ==>  w1 + w2 + b < 0")
    print("\nPenjumlahan pertidaksamaan (2) dan (3):")
    print("  (w1 + b) + (w2 + b) >= 0 + 0")
    print("  w1 + w2 + 2b >= 0")
    print("  w1 + w2 + b >= -b")
    print("Karena dari pertidaksamaan (1) b < 0, maka -b > 0.")
    print("Sehingga diperoleh: w1 + w2 + b > 0.")
    print("Namun, pertidaksamaan (4) menuntut: w1 + w2 + b < 0.")
    print("\n--> TERJADI KONTRADIKSI MATEMATIS MUTLAK: (w1 + w2 + b > 0) dan (w1 + w2 + b < 0)!")
    print("Kesimpulan Formal (Minsky & Papert, 1969):")
    print("Tidak ada garis lurus (linear hyperplane) apa pun di R^2 yang dapat memisahkan gerbang XOR.")

def solve_xor_with_mlp():
    print("\n" + "=" * 80)
    print("SOLUSI MULTILAYER PERCEPTRON: DEKOMPOSISI XOR = (x1 OR x2) AND (x1 NAND x2)")
    print("=" * 80)
    
    # Arsitektur 2-Layer:
    # Hidden Layer: Neuron 1 (OR), Neuron 2 (NAND)
    # Output Layer: Neuron 3 (AND dari luaran Neuron 1 dan Neuron 2)
    
    W_hidden = np.array([
        [1.0, -1.0],  # Bobot dari x1 ke [h1_OR, h2_NAND]
        [1.0, -1.0]   # Bobot dari x2 ke [h1_OR, h2_NAND]
    ])
    b_hidden = np.array([-0.5, 1.5])  # Bias [h1_OR, h2_NAND]
    
    W_output = np.array([1.0, 1.0])   # Bobot dari [h1, h2] ke output AND
    b_output = -1.5                   # Bias output AND
    
    inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    
    print(f"{'x1':<5} {'x2':<5} {'h1 (OR)':<12} {'h2 (NAND)':<12} {'z_out':<10} {'y_pred (XOR)':<15} {'Status':<10}")
    print("-" * 75)
    
    for x in inputs:
        # 1. Hidden layer pass
        z_h = np.dot(x, W_hidden) + b_hidden
        a_h = (z_h >= 0).astype(int)
        
        # 2. Output layer pass
        z_out = np.dot(a_h, W_output) + b_output
        y_pred = int(z_out >= 0)
        
        y_actual = x[0] ^ x[1]
        status = "BENAR" if y_pred == y_actual else "SALAH"
        
        print(f"{x[0]:<5} {x[1]:<5} {a_h[0]:<12} {a_h[1]:<12} {z_out:>6.1f}     {y_pred:<15} {status}")
        
    print("=" * 75)
    print("INSIGHT REPRESENTATION LEARNING:")
    print("Hidden layer mentransformasikan ruang input non-linier ke representasi baru [h1, h2]")
    print("di mana data menjadi linearly separable bagi neuron output!")

if __name__ == "__main__":
    prove_xor_linear_impossibility()
    solve_xor_with_mlp()

