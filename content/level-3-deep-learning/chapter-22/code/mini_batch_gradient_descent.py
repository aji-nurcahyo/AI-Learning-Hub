"""
Chapter 22 - Backpropagation & Optimization
Script 8: Mini-Batch Gradient Descent (MBGD)
Standar emas pelatihan Deep Learning modern: menyeimbangkan stabilitas estimasi gradien
vektor paralel GPU/CPU dengan efisiensi memori menggunakan ukuran batch (misal B = 32).
"""

import numpy as np

def generate_synthetic_data(n_samples=320, random_state=42):
    np.random.seed(random_state)
    X = np.random.randn(n_samples, 2)
    y = ((X[:, 0]**2 + X[:, 1]**2) < 1.0).astype(float).reshape(-1, 1)
    return X, y

def run_mini_batch_gradient_descent():
    print("=" * 80)
    print("CHAPTER 22: OPTIMASI MINI-BATCH GRADIENT DESCENT (MBGD - BATCH SIZE = 32)")
    print("=" * 80)
    
    X, y = generate_synthetic_data(n_samples=320)
    N, D_in = X.shape
    D_h = 4
    D_out = 1
    batch_size = 32
    num_batches = int(np.ceil(N / batch_size))
    epochs = 40
    learning_rate = 0.2
    
    np.random.seed(42)
    W1 = np.random.randn(D_in, D_h) * np.sqrt(2.0 / D_in)
    b1 = np.zeros((1, D_h))
    W2 = np.random.randn(D_h, D_out) * np.sqrt(2.0 / D_h)
    b2 = np.zeros((1, D_out))
    
    print(f"Total Sampel Data (N)      : {N}")
    print(f"Ukuran Mini-Batch (B)      : {batch_size}")
    print(f"Jumlah Iterasi per Epoch   : {num_batches} mini-batches")
    print(f"Jumlah Epoch               : {epochs}")
    print(f"Total Iterasi Pembaruan    : {epochs * num_batches} iterasi")
    print("-" * 80)
    print(f"{'Epoch':<10} {'BCE Loss Rata-rata':<22} {'Akurasi Latih':<15} {'Total Pembaruan':<15}")
    print("-" * 80)
    
    for epoch in range(1, epochs + 1):
        # Pengacakan indeks di awal epoch
        perm = np.random.permutation(N)
        X_shuffled = X[perm]
        y_shuffled = y[perm]
        
        batch_losses = []
        for b in range(num_batches):
            start_idx = b * batch_size
            end_idx = min(start_idx + batch_size, N)
            
            X_b = X_shuffled[start_idx:end_idx]
            y_b = y_shuffled[start_idx:end_idx]
            B_cur = X_b.shape[0]
            
            # Forward Pass pada mini-batch
            Z1 = np.dot(X_b, W1) + b1
            A1 = np.maximum(0.0, Z1)
            Z2 = np.dot(A1, W2) + b2
            A2 = 1.0 / (1.0 + np.exp(-np.clip(Z2, -30, 30)))
            
            loss_b = - np.mean(y_b * np.log(np.clip(A2, 1e-15, 1-1e-15)) + (1 - y_b) * np.log(np.clip(1 - A2, 1e-15, 1-1e-15)))
            batch_losses.append(loss_b)
            
            # Backward Pass pada mini-batch
            dZ2 = (A2 - y_b) / B_cur
            dW2 = np.dot(A1.T, dZ2)
            db2 = np.sum(dZ2, axis=0, keepdims=True)
            
            dZ1 = np.dot(dZ2, W2.T) * (Z1 > 0)
            dW1 = np.dot(X_b.T, dZ1)
            db1 = np.sum(dZ1, axis=0, keepdims=True)
            
            # Parameter Update per mini-batch
            W1 -= learning_rate * dW1
            b1 -= learning_rate * db1
            W2 -= learning_rate * dW2
            b2 -= learning_rate * db2
            
        # Evaluasi Akurasi Keseluruhan
        Z1_all = np.dot(X, W1) + b1
        A1_all = np.maximum(0.0, Z1_all)
        Z2_all = np.dot(A1_all, W2) + b2
        A2_all = 1.0 / (1.0 + np.exp(-np.clip(Z2_all, -30, 30)))
        acc = np.mean((A2_all >= 0.5) == y)
        mean_loss = np.mean(batch_losses)
        
        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            print(f"Epoch {epoch:<5} {mean_loss:>15.4f}         {acc*100:>10.2f}%       {epoch * num_batches:>12}")
            
    print("=" * 80)
    print("PERBANDINGAN DEFINISI FUNDAMENTAL:")
    print("- Epoch   : Satu lintasan penuh di mana model telah melihat SELURUH sampel data latih.")
    print("- Iteration: Satu kali eksekusi forward-backward dan pembaruan bobot pada satu mini-batch.")

if __name__ == "__main__":
    run_mini_batch_gradient_descent()

