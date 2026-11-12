"""
Chapter 22 - Backpropagation & Optimization
Script 7: Stochastic Gradient Descent (SGD)
Melakukan pembaruan parameter per satu sampel masukan (Batch Size = 1),
menunjukkan fluktuasi gradien stokastik dan efisiensi waktu komputasi per langkah.
"""

import numpy as np

def generate_synthetic_data(n_samples=200, random_state=42):
    np.random.seed(random_state)
    X = np.random.randn(n_samples, 2)
    y = ((X[:, 0]**2 + X[:, 1]**2) < 1.0).astype(float).reshape(-1, 1)
    return X, y

def run_stochastic_gradient_descent():
    print("=" * 80)
    print("CHAPTER 22: OPTIMASI STOCHASTIC GRADIENT DESCENT (SGD - BATCH SIZE = 1)")
    print("=" * 80)
    
    X, y = generate_synthetic_data(n_samples=300)
    N, D_in = X.shape
    D_h = 4
    D_out = 1
    epochs = 20
    learning_rate = 0.05
    
    np.random.seed(42)
    W1 = np.random.randn(D_in, D_h) * np.sqrt(2.0 / D_in)
    b1 = np.zeros((1, D_h))
    W2 = np.random.randn(D_h, D_out) * np.sqrt(2.0 / D_h)
    b2 = np.zeros((1, D_out))
    
    print(f"Dataset Size (N)     : {N} sampel")
    print(f"Pembaruan per Epoch  : {N} kali update parameter")
    print(f"Total Pembaruan      : {epochs * N} iterasi")
    print(f"Learning Rate (eta)  : {learning_rate}")
    print("-" * 80)
    print(f"{'Epoch':<10} {'Rata-rata Loss':<18} {'Akurasi Latih':<15} {'Total Iterasi Update':<20}")
    print("-" * 80)
    
    for epoch in range(1, epochs + 1):
        # Acak urutan sampel setiap awal epoch
        indices = np.random.permutation(N)
        epoch_losses = []
        
        for i in indices:
            xi = X[i:i+1]  # Bentuk (1, 2)
            yi = y[i:i+1]  # Bentuk (1, 1)
            
            # Forward Pass 1 sampel
            z1 = np.dot(xi, W1) + b1
            a1 = np.maximum(0.0, z1)
            z2 = np.dot(a1, W2) + b2
            a2 = 1.0 / (1.0 + np.exp(-np.clip(z2, -30, 30)))
            
            loss_i = - (yi * np.log(np.clip(a2, 1e-15, 1-1e-15)) + (1 - yi) * np.log(np.clip(1 - a2, 1e-15, 1-1e-15)))
            epoch_losses.append(loss_i[0, 0])
            
            # Backward Pass 1 sampel
            dz2 = (a2 - yi)
            dW2 = np.dot(a1.T, dz2)
            db2 = dz2
            
            dz1 = np.dot(dz2, W2.T) * (z1 > 0)
            dW1 = np.dot(xi.T, dz1)
            db1 = dz1
            
            # Pembaruan instan
            W1 -= learning_rate * dW1
            b1 -= learning_rate * db1
            W2 -= learning_rate * dW2
            b2 -= learning_rate * db2
            
        # Evaluasi akurasi akhir epoch
        Z1_all = np.dot(X, W1) + b1
        A1_all = np.maximum(0.0, Z1_all)
        Z2_all = np.dot(A1_all, W2) + b2
        A2_all = 1.0 / (1.0 + np.exp(-np.clip(Z2_all, -30, 30)))
        acc = np.mean((A2_all >= 0.5) == y)
        mean_loss = np.mean(epoch_losses)
        
        if epoch == 1 or epoch % 5 == 0 or epoch == epochs:
            print(f"Epoch {epoch:<5} {mean_loss:>12.4f}       {acc*100:>10.2f}%       {epoch * N:>15}")
            
    print("=" * 80)
    print("INSIGHT STOCHASTIC GRADIENT DESCENT:")
    print("1. Kecepatan: SGD memperbarui parameter berkali-kali dalam satu epoch (300 update vs 1 update pada BGD).")
    print("2. Derau Gradien: Fluktuasi stokastik membantu melompati saddle points dan local minima dangkal.")
    print("3. Keterbatasan: Komputasi vektor kurang optimal dan kurva loss berosilasi di sekitar minimum.")

if __name__ == "__main__":
    run_stochastic_gradient_descent()

