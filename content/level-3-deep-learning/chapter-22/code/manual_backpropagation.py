"""
Chapter 22 - Backpropagation & Optimization
Script 4: Propagasi Mundur Manual Lengkap (Manual Backpropagation Worked Example)
Melakukan kalkulasi langkah-demi-langkah perambatan gradien analitis pada jaringan 2-2-1:
2 inputs -> 1 hidden layer (2 neuron, Sigmoid) -> 1 output neuron (Sigmoid, MSE Loss),
memperbarui bobot dengan Gradient Descent, dan membuktikan penurunan nilai loss secara empiris.
"""

import numpy as np

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def sigmoid_derivative(z):
    s = sigmoid(z)
    return s * (1.0 - s)

def run_manual_backprop_worked_example():
    print("=" * 85)
    print("CHAPTER 22: WORKED NUMERICAL EXAMPLE MANUAL BACKPROPAGATION (JARINGAN 2 -> 2 -> 1)")
    print("=" * 85)
    
    # 1. Konfigurasi Parameter Awal Eksplisit
    # Masukan x: (1, 2) dan Target y_true: (1, 1)
    x = np.array([[0.5, 1.0]])
    y_true = np.array([[1.0]])
    learning_rate = 0.5
    
    # Layer 1 (Hidden): W1 (2, 2), b1 (1, 2)
    W1 = np.array([
        [0.15, 0.20],
        [0.25, 0.30]
    ])
    b1 = np.array([[0.35, 0.35]])
    
    # Layer 2 (Output): W2 (2, 1), b2 (1, 1)
    W2 = np.array([
        [0.40],
        [0.45]
    ])
    b2 = np.array([[0.60]])
    
    print("KONFIGURASI PARAMETER AWAL:")
    print(f"Input x       : {x}")
    print(f"Target y      : {y_true}")
    print(f"W^[1] (2x2)   :\n{W1}")
    print(f"b^[1] (1x2)   : {b1}")
    print(f"W^[2] (2x1)   :\n{W2}")
    print(f"b^[2] (1x1)   : {b2}")
    print(f"Learning Rate : {learning_rate}")
    print("-" * 85)
    
    # 2. Forward Pass Awal
    # Z1 = x @ W1 + b1
    # z1_1 = 0.5(0.15) + 1.0(0.25) + 0.35 = 0.075 + 0.250 + 0.350 = 0.675
    # z1_2 = 0.5(0.20) + 1.0(0.30) + 0.35 = 0.100 + 0.300 + 0.350 = 0.750
    Z1 = np.dot(x, W1) + b1
    A1 = sigmoid(Z1)
    
    # Z2 = A1 @ W2 + b2
    # z2 = a1_1(0.40) + a1_2(0.45) + 0.60
    Z2 = np.dot(A1, W2) + b2
    A2 = sigmoid(Z2)
    y_pred = A2
    
    # Fungsi Rugi MSE: L = 0.5 * (y_true - y_pred)^2
    initial_loss = 0.5 * (y_true - y_pred)**2
    print("\n[LANGKAH 1: FORWARD PASS AWAL]")
    print(f"Z^[1] Hidden Sum        : {Z1}")
    print(f"A^[1] Hidden Activation : {A1}")
    print(f"Z^[2] Output Sum        : {Z2}")
    print(f"A^[2] Prediksi y_hat    : {y_pred[0, 0]:.6f}")
    print(f"Nilai Loss Awal (MSE)   : {initial_loss[0, 0]:.6f}")
    
    # 3. Backward Pass (Aturan Rantai Turunan)
    # dL/dy_pred = (y_pred - y_true)
    # dy_pred/dZ2 = A2 * (1 - A2)
    # delta2 = dL/dZ2 = (y_pred - y_true) * sigmoid'(Z2)
    print("\n[LANGKAH 2: KOMPUTASI GRADIEN OUTPUT LAYER]")
    dL_dA2 = (y_pred - y_true)
    dA2_dZ2 = sigmoid_derivative(Z2)
    delta2 = dL_dA2 * dA2_dZ2
    print(f"dL/dA2 (Error Output)   : {dL_dA2[0, 0]:.6f}")
    print(f"dA2/dZ2 (Turunan Sigmoid): {dA2_dZ2[0, 0]:.6f}")
    print(f"delta^[2] = dL/dZ2      : {delta2[0, 0]:.6f}")
    
    # dL/dW2 = A1^T @ delta2
    # dL/db2 = delta2
    dW2 = np.dot(A1.T, delta2)
    db2 = delta2
    print(f"dL/dW^[2] (Gradien W2)  :\n{dW2}")
    print(f"dL/db^[2] (Gradien b2)  : {db2}")
    
    # 4. Propagasi Balik ke Hidden Layer
    # delta1 = (delta2 @ W2^T) * sigmoid'(Z1)
    print("\n[LANGKAH 3: KOMPUTASI GRADIEN HIDDEN LAYER]")
    dZ1_propagated = np.dot(delta2, W2.T)
    dA1_dZ1 = sigmoid_derivative(Z1)
    delta1 = dZ1_propagated * dA1_dZ1
    print(f"delta^[1] = dL/dZ1      : {delta1}")
    
    # dL/dW1 = x^T @ delta1
    # dL/db1 = delta1
    dW1 = np.dot(x.T, delta1)
    db1 = delta1
    print(f"dL/dW^[1] (Gradien W1)  :\n{dW1}")
    print(f"dL/db^[1] (Gradien b1)  : {db1}")
    
    # 5. Pembaruan Parameter (Gradient Descent Step: theta <- theta - lr * grad)
    print("\n[LANGKAH 4: PEMBARUAN PARAMETER (GRADIENT DESCENT)]")
    W1_updated = W1 - learning_rate * dW1
    b1_updated = b1 - learning_rate * db1
    W2_updated = W2 - learning_rate * dW2
    b2_updated = b2 - learning_rate * db2
    
    print(f"W^[1] Terbarui:\n{W1_updated}")
    print(f"b^[1] Terbarui: {b1_updated}")
    print(f"W^[2] Terbarui:\n{W2_updated}")
    print(f"b^[2] Terbarui: {b2_updated}")
    
    # 6. Verifikasi Forward Pass Ulang dengan Parameter Baru
    print("\n[LANGKAH 5: VERIFIKASI FORWARD PASS ULANG]")
    Z1_new = np.dot(x, W1_updated) + b1_updated
    A1_new = sigmoid(Z1_new)
    Z2_new = np.dot(A1_new, W2_updated) + b2_updated
    y_pred_new = sigmoid(Z2_new)
    new_loss = 0.5 * (y_true - y_pred_new)**2
    
    print(f"Prediksi Baru y_hat     : {y_pred_new[0, 0]:.6f} (Mendekati target y = 1.0)")
    print(f"Nilai Loss Baru (MSE)   : {new_loss[0, 0]:.6f}")
    print(f"Penurunan Nilai Loss    : {initial_loss[0, 0] - new_loss[0, 0]:.6f}")
    
    # Pembuktian mutlak penurunan loss
    assert new_loss < initial_loss, "Loss gagal turun setelah pembaruan gradien!"
    print("=" * 85)
    print("STATUS PEMBUKTIAN MATEMATIS: SUKSES (Loss terbukti turun secara deterministik).")

if __name__ == "__main__":
    run_manual_backprop_worked_example()

