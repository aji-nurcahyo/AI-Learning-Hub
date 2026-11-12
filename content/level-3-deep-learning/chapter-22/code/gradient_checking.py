"""
Chapter 22 - Backpropagation & Optimization
Script 5: Pemeriksaan Gradien Numerik (Gradient Checking)
Mengimplementasikan prosedur verifikasi formal turunan analitis Backpropagation terhadap
gradien numerik beda hingga terpusat (Central Finite Differences, eps = 1e-7).
"""

import numpy as np

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))

def forward_and_loss(X, y, W1, b1, W2, b2):
    """Menghitung forward pass dan loss BCE."""
    Z1 = np.dot(X, W1) + b1
    A1 = np.maximum(0.0, Z1)  # ReLU
    Z2 = np.dot(A1, W2) + b2
    A2 = sigmoid(Z2)
    
    eps = 1e-15
    p_clip = np.clip(A2, eps, 1.0 - eps)
    loss = - np.mean(y * np.log(p_clip) + (1.0 - y) * np.log(1.0 - p_clip))
    return loss, (X, Z1, A1, Z2, A2)

def backward_pass(y, cache, W1, b1, W2, b2):
    """Menghitung gradien analitis melalui Backpropagation."""
    X, Z1, A1, Z2, A2 = cache
    N = X.shape[0]
    
    # Turunan BCE + Sigmoid output: dL/dZ2 = (A2 - y) / N
    dZ2 = (A2 - y) / N
    dW2 = np.dot(A1.T, dZ2)
    db2 = np.sum(dZ2, axis=0, keepdims=True)
    
    # Hidden layer ReLU: dZ1 = (dZ2 @ W2^T) * (Z1 > 0)
    dA1 = np.dot(dZ2, W2.T)
    dZ1 = dA1 * (Z1 > 0)
    dW1 = np.dot(X.T, dZ1)
    db1 = np.sum(dZ1, axis=0, keepdims=True)
    
    return dW1, db1, dW2, db2

def dictionary_to_vector(W1, b1, W2, b2):
    """Meratakan seluruh parameter menjadi satu vektor 1D theta."""
    return np.concatenate([W1.ravel(), b1.ravel(), W2.ravel(), b2.ravel()])

def vector_to_dictionary(theta, shapes):
    """Merekonstruksi kembali matriks parameter dari vektor 1D theta."""
    s_W1, s_b1, s_W2, s_b2 = shapes
    idx = 0
    
    end_W1 = idx + np.prod(s_W1)
    W1 = theta[idx:end_W1].reshape(s_W1)
    idx = end_W1
    
    end_b1 = idx + np.prod(s_b1)
    b1 = theta[idx:end_b1].reshape(s_b1)
    idx = end_b1
    
    end_W2 = idx + np.prod(s_W2)
    W2 = theta[idx:end_W2].reshape(s_W2)
    idx = end_W2
    
    end_b2 = idx + np.prod(s_b2)
    b2 = theta[idx:end_b2].reshape(s_b2)
    
    return W1, b1, W2, b2

def run_gradient_check():
    print("=" * 85)
    print("CHAPTER 22: PEMERIKSAAN GRADIEN NUMERIK (GRADIENT CHECKING)")
    print("=" * 85)
    
    np.random.seed(42)
    # Batch N = 3 sampel, 2 input features, 3 hidden neurons, 1 output neuron
    X = np.random.randn(3, 2)
    y = np.array([[1.0], [0.0], [1.0]])
    
    W1 = np.random.randn(2, 3) * 0.5
    b1 = np.zeros((1, 3))
    W2 = np.random.randn(3, 1) * 0.5
    b2 = np.zeros((1, 1))
    
    shapes = (W1.shape, b1.shape, W2.shape, b2.shape)
    
    # 1. Gradien Analitis via Backpropagation
    loss, cache = forward_and_loss(X, y, W1, b1, W2, b2)
    dW1, db1, dW2, db2 = backward_pass(y, cache, W1, b1, W2, b2)
    grad_analytical = dictionary_to_vector(dW1, db1, dW2, db2)
    
    # 2. Gradien Numerik via Central Difference
    theta = dictionary_to_vector(W1, b1, W2, b2)
    grad_numerical = np.zeros_like(theta)
    epsilon = 1e-7
    
    print(f"Jumlah Parameter Dievaluasi : {len(theta)}")
    print(f"Nilai Epsilon Perturbasi   : {epsilon}")
    
    for i in range(len(theta)):
        theta_plus = np.copy(theta)
        theta_plus[i] += epsilon
        W1_p, b1_p, W2_p, b2_p = vector_to_dictionary(theta_plus, shapes)
        loss_plus, _ = forward_and_loss(X, y, W1_p, b1_p, W2_p, b2_p)
        
        theta_minus = np.copy(theta)
        theta_minus[i] -= epsilon
        W1_m, b1_m, W2_m, b2_m = vector_to_dictionary(theta_minus, shapes)
        loss_minus, _ = forward_and_loss(X, y, W1_m, b1_m, W2_m, b2_m)
        
        # Beda hingga terpusat
        grad_numerical[i] = (loss_plus - loss_minus) / (2.0 * epsilon)
        
    # 3. Komputasi Relative Error: ||g_analyt - g_num||_2 / (||g_analyt||_2 + ||g_num||_2)
    numerator = np.linalg.norm(grad_analytical - grad_numerical)
    denominator = np.linalg.norm(grad_analytical) + np.linalg.norm(grad_numerical)
    relative_error = numerator / denominator
    
    print("-" * 85)
    print(f"Norma Selisih (Numerator)   : {numerator:.6e}")
    print(f"Norma Total (Denominator)   : {denominator:.6e}")
    print(f"Galat Relatif Akhir         : {relative_error:.6e}")
    print("-" * 85)
    
    # Kriteria toleransi standar industri:
    # rel_error < 1e-7: Hebat, implementasi gradien benar 100%.
    # 1e-7 <= rel_error < 1e-4: Perlu diperiksa jika ada kink activation.
    # rel_error >= 1e-4: Bug pada backpropagation!
    if relative_error < 1e-7:
        print("HASIL AUDIT GRADIENT CHECKING: LOLOS SEMPURNA (Relative error < 1e-7).")
        print("Implementasi Backpropagation terbukti benar secara matematis.")
    else:
        print("HASIL AUDIT GRADIENT CHECKING: PERHATIAN - Periksa kembali turunan!")

if __name__ == "__main__":
    run_gradient_check()

