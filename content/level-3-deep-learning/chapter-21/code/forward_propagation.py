"""
Chapter 21 - Neural Network Fundamentals
Script 7: Propagasi Maju Lengkap (Forward Propagation Worked Example)
Melakukan kalkulasi langkah demi langkah propagasi maju pada arsitektur 2-3-1:
2 input features -> 1 hidden layer (3 neuron, ReLU) -> 1 output neuron (Sigmoid),
memverifikasi perhitungan aritmatika skalar manual terhadap operasi matriks tervektorisasi.
"""

import numpy as np

def forward_propagation_worked_example():
    print("=" * 80)
    print("CHAPTER 21: WORKED EXAMPLE PROPAGASI MAJU (ARSITEKTUR 2 -> 3 -> 1)")
    print("=" * 80)
    
    # 1. Definisi Parameter Model Eksplisit (Nilai Ditentukan untuk Pembuktian Numerik)
    # Layer 1 (Hidden Layer): W^[1] berukuran (2, 3), b^[1] berukuran (1, 3)
    W1 = np.array([
        [0.5, -0.2,  0.8],   # Bobot dari x1 ke [h1, h2, h3]
        [0.4,  0.7, -0.5]    # Bobot dari x2 ke [h1, h2, h3]
    ])
    b1 = np.array([[0.1, -0.3, 0.2]])  # Bias hidden layer
    
    # Layer 2 (Output Layer): W^[2] berukuran (3, 1), b^[2] berukuran (1, 1)
    W2 = np.array([
        [ 0.6],   # Bobot dari h1 ke output
        [-0.8],   # Bobot dari h2 ke output
        [ 0.3]    # Bobot dari h3 ke output
    ])
    b2 = np.array([[-0.1]])  # Bias output layer
    
    # Sampel Masukan Tunggal: x = [1.5, 2.0]
    x_single = np.array([[1.5, 2.0]])
    
    print("PARAMETER MODEL:")
    print(f"Input x       : {x_single}")
    print(f"W^[1] (2x3)   :\n{W1}")
    print(f"b^[1] (1x3)   : {b1}")
    print(f"W^[2] (3x1)   :\n{W2}")
    print(f"b^[2] (1x1)   : {b2}")
    print("-" * 80)
    
    # 2. Perhitungan Manual Skalar Layer 1 (Hidden Layer)
    # z1_1 = x1*w11 + x2*w21 + b1_1 = 1.5*(0.5) + 2.0*(0.4) + 0.1 = 0.75 + 0.80 + 0.10 = 1.65
    # z1_2 = x1*w12 + x2*w22 + b1_2 = 1.5*(-0.2) + 2.0*(0.7) - 0.3 = -0.30 + 1.40 - 0.30 = 0.80
    # z1_3 = x1*w13 + x2*w23 + b1_3 = 1.5*(0.8) + 2.0*(-0.5) + 0.2 = 1.20 - 1.00 + 0.20 = 0.40
    print("\n[LANGKAH 1: HIDDEN LAYER LINEAR COMBINATION (Z^[1] = X @ W^[1] + b^[1])]")
    Z1 = np.dot(x_single, W1) + b1
    print(f"Z^[1] Terhitung : {Z1}")
    print("Verifikasi Skalar:")
    print(f"  z1_1 = 1.5*(0.5) + 2.0*(0.4) + 0.1 = {1.5*0.5 + 2.0*0.4 + 0.1:.4f}")
    print(f"  z1_2 = 1.5*(-0.2) + 2.0*(0.7) - 0.3 = {1.5*-0.2 + 2.0*0.7 - 0.3:.4f}")
    print(f"  z1_3 = 1.5*(0.8) + 2.0*(-0.5) + 0.2 = {1.5*0.8 + 2.0*-0.5 + 0.2:.4f}")
    
    print("\n[LANGKAH 2: HIDDEN LAYER NON-LINEAR ACTIVATION (A^[1] = ReLU(Z^[1]))]")
    A1 = np.maximum(0.0, Z1)
    print(f"A^[1] Terhitung : {A1}")
    
    print("\n[LANGKAH 3: OUTPUT LAYER LINEAR COMBINATION (Z^[2] = A^[1] @ W^[2] + b^[2])]")
    # z2_1 = a1_1*w1 + a1_2*w2 + a1_3*w3 + b2_1
    #       = 1.65*(0.6) + 0.80*(-0.8) + 0.40*(0.3) - 0.1
    #       = 0.990 - 0.640 + 0.120 - 0.100 = 0.370
    Z2 = np.dot(A1, W2) + b2
    print(f"Z^[2] Terhitung : {Z2}")
    manual_z2 = 1.65*0.6 + 0.80*(-0.8) + 0.40*0.3 - 0.1
    print(f"Verifikasi Skalar: {1.65*0.6:.3f} + ({0.80*-0.8:.3f}) + {0.40*0.3:.3f} - 0.1 = {manual_z2:.4f}")
    
    print("\n[LANGKAH 4: OUTPUT LAYER ACTIVATION (A^[2] = Sigmoid(Z^[2]))]")
    # a2 = 1 / (1 + exp(-0.370)) = 1 / (1 + 0.6907) = 1 / 1.6907 = 0.59146
    A2 = 1.0 / (1.0 + np.exp(-Z2))
    print(f"A^[2] Prediksi Probabilitas Akhir: {A2[0, 0]:.6f}")
    print("=" * 80)

def forward_propagation_batch():
    print("\n" + "=" * 80)
    print("PROPAGASI MAJU PEMROSESAN BATCH (N = 4 SAMPEL SEKALIGUS)")
    print("=" * 80)
    
    W1 = np.array([[0.5, -0.2, 0.8], [0.4, 0.7, -0.5]])
    b1 = np.array([[0.1, -0.3, 0.2]])
    W2 = np.array([[0.6], [-0.8], [0.3]])
    b2 = np.array([[-0.1]])
    
    # Batch N = 4 sampel masukan
    X_batch = np.array([
        [1.5, 2.0],   # Sampel 1
        [0.0, 1.0],   # Sampel 2
        [-1.0, 0.5],  # Sampel 3
        [2.0, -1.5]   # Sampel 4
    ])
    
    # Forward Pass Tervektorisasi:
    # 1. Z1: (4, 2) @ (2, 3) + (1, 3) -> (4, 3)
    Z1 = np.dot(X_batch, W1) + b1
    A1 = np.maximum(0.0, Z1)
    
    # 2. Z2: (4, 3) @ (3, 1) + (1, 1) -> (4, 1)
    Z2 = np.dot(A1, W2) + b2
    A2 = 1.0 / (1.0 + np.exp(-Z2))
    
    print(f"{'Sampel':<10} {'Fitur x1, x2':<18} {'A^[1] Hidden Activations':<30} {'Peluang Akhir (A^[2])':<20} {'Prediksi Kelas':<15}")
    print("-" * 95)
    for i in range(len(X_batch)):
        prob = A2[i, 0]
        pred_cls = 1 if prob >= 0.5 else 0
        print(f"Sampel {i+1:<3} {str(X_batch[i]):<18} {str(np.round(A1[i], 3)):<30} {prob:>12.4f}             {pred_cls}")
    print("=" * 95)

if __name__ == "__main__":
    forward_propagation_worked_example()
    forward_propagation_batch()

