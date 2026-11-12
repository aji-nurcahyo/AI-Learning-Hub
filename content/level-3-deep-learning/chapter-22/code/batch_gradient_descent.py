"""
Chapter 22 - Backpropagation & Optimization
Script 6: Batch Gradient Descent (BGD)
Melakukan optimasi deterministik pada seluruh himpunan data latih per langkah pembaruan,
menghasilkan lintasan kurva loss yang mulus dan monotonic konvergen.
"""

import numpy as np

def generate_synthetic_data(n_samples=200, random_state=42):
    np.random.seed(random_state)
    X = np.random.randn(n_samples, 2)
    # Target klasifikasi biner non-linier lingkaran: x1^2 + x2^2 < 1.0
    y = ((X[:, 0]**2 + X[:, 1]**2) < 1.0).astype(float).reshape(-1, 1)
    return X, y

def run_batch_gradient_descent():
    print("=" * 80)
    print("CHAPTER 22: OPTIMASI BATCH GRADIENT DESCENT (BGD)")
    print("=" * 80)
    
    X, y = generate_synthetic_data(n_samples=300)
    N, D_in = X.shape
    D_h = 4
    D_out = 1
    epochs = 100
    learning_rate = 0.5
    
    np.random.seed(42)
    W1 = np.random.randn(D_in, D_h) * np.sqrt(2.0 / D_in)
    b1 = np.zeros((1, D_h))
    W2 = np.random.randn(D_h, D_out) * np.sqrt(2.0 / D_h)
    b2 = np.zeros((1, D_out))
    
    print(f"Dataset Size (N)     : {N} sampel")
    print(f"Topologi Model       : [{D_in}, {D_h}, {D_out}] (MLP)")
    print(f"Jumlah Epoch         : {epochs}")
    print(f"Learning Rate (eta)  : {learning_rate}")
    print("-" * 80)
    print(f"{'Epoch':<10} {'BCE Loss':<15} {'Akurasi Latih':<15} {'Norma Gradien W1':<18}")
    print("-" * 80)
    
    loss_history = []
    
    for epoch in range(1, epochs + 1):
        # 1. Forward Pass pada SELURUH N sampel sekaligus
        Z1 = np.dot(X, W1) + b1
        A1 = np.maximum(0.0, Z1)  # ReLU
        Z2 = np.dot(A1, W2) + b2
        A2 = 1.0 / (1.0 + np.exp(-np.clip(Z2, -30, 30)))  # Sigmoid
        
        # Loss BCE
        eps = 1e-15
        loss = - np.mean(y * np.log(np.clip(A2, eps, 1 - eps)) + (1 - y) * np.log(np.clip(1 - A2, eps, 1 - eps)))
        loss_history.append(loss)
        
        # Akurasi
        preds = (A2 >= 0.5).astype(float)
        acc = np.mean(preds == y)
        
        # 2. Backward Pass pada SELURUH N sampel
        dZ2 = (A2 - y) / N
        dW2 = np.dot(A1.T, dZ2)
        db2 = np.sum(dZ2, axis=0, keepdims=True)
        
        dZ1 = np.dot(dZ2, W2.T) * (Z1 > 0)
        dW1 = np.dot(X.T, dZ1)
        db1 = np.sum(dZ1, axis=0, keepdims=True)
        
        grad_norm = np.linalg.norm(dW1)
        
        # 3. Parameter Update
        W1 -= learning_rate * dW1
        b1 -= learning_rate * db1
        W2 -= learning_rate * dW2
        b2 -= learning_rate * db2
        
        if epoch == 1 or epoch % 20 == 0 or epoch == epochs:
            print(f"Epoch {epoch:<5} {loss:>10.4f}     {acc*100:>10.2f}%       {grad_norm:>12.6f}")
            
    print("=" * 80)
    print("KARAKTERISTIK BATCH GRADIENT DESCENT:")
    print(f"Loss Awal: {loss_history[0]:.4f} -> Loss Akhir: {loss_history[-1]:.4f}")
    print("Lintasan optimasi stabil dan deterministik karena mengevaluasi gradien eksak dari seluruh data.")

if __name__ == "__main__":
    run_batch_gradient_descent()

