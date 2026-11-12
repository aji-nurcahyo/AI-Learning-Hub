"""
Chapter 22 - Backpropagation & Optimization
Script 15: Pelatihan Lengkap Jaringan Saraf Tiruan dari Dasar (NumPy Neural Network Training)
Mengintegrasikan seluruh siklus hidup pelatihan MLP: inisialisasi He, forward pass,
backpropagation analitis, pembaruan pengoptimal adaptif (Adam), validasi konvergensi,
dan evaluasi generalisasi pada dataset sintetis non-linier lingkaran konsentris.
"""

import numpy as np

def generate_circles_dataset(n_samples=400, noise=0.1, random_state=42):
    """Membangkitkan dataset klasifikasi biner non-linier 2 lingkaran konsentris."""
    np.random.seed(random_state)
    n_pts = n_samples // 2
    
    # Lingkaran dalam (Kelas 1)
    r_inner = np.random.uniform(0.0, 0.5, n_pts)
    theta_inner = np.random.uniform(0, 2 * np.pi, n_pts)
    X_inner = np.column_stack([r_inner * np.cos(theta_inner), r_inner * np.sin(theta_inner)])
    y_inner = np.ones((n_pts, 1))
    
    # Lingkaran luar (Kelas 0)
    r_outer = np.random.uniform(0.8, 1.2, n_pts)
    theta_outer = np.random.uniform(0, 2 * np.pi, n_pts)
    X_outer = np.column_stack([r_outer * np.cos(theta_outer), r_outer * np.sin(theta_outer)])
    y_outer = np.zeros((n_pts, 1))
    
    X = np.vstack([X_inner, X_outer]) + np.random.normal(0, noise, (n_samples, 2))
    y = np.vstack([y_inner, y_outer])
    
    perm = np.random.permutation(n_samples)
    return X[perm], y[perm]

class NeuralNetworkTrainer:
    def __init__(self, layer_dims=[2, 8, 4, 1], lr=0.05, optimizer='adam', random_state=42):
        self.layer_dims = layer_dims
        self.lr = lr
        self.optimizer = optimizer.lower()
        self.num_layers = len(layer_dims) - 1
        
        np.random.seed(random_state)
        self.params = {}
        for l in range(1, len(layer_dims)):
            n_in = layer_dims[l-1]
            n_out = layer_dims[l]
            self.params[f'W{l}'] = np.random.randn(n_in, n_out) * np.sqrt(2.0 / n_in)
            self.params[f'b{l}'] = np.zeros((1, n_out))
            
        # State optimizer
        self.m = {k: np.zeros_like(v) for k, v in self.params.items()}
        self.v = {k: np.zeros_like(v) for k, v in self.params.items()}
        self.t = 0
        
    def _sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
        
    def forward(self, X):
        cache = {'A0': X}
        A_prev = X
        
        for l in range(1, self.num_layers + 1):
            W = self.params[f'W{l}']
            b = self.params[f'b{l}']
            Z = np.dot(A_prev, W) + b
            A = self._sigmoid(Z) if l == self.num_layers else np.maximum(0.0, Z)
            
            cache[f'Z{l}'] = Z
            cache[f'A{l}'] = A
            A_prev = A
            
        return A_prev, cache
        
    def compute_loss(self, y_true, y_pred, eps=1e-15):
        p = np.clip(y_pred, eps, 1.0 - eps)
        return - np.mean(y_true * np.log(p) + (1.0 - y_true) * np.log(1.0 - p))
        
    def backward(self, y_true, cache):
        grads = {}
        N = y_true.shape[0]
        AL = cache[f'A{self.num_layers}']
        
        # dL/dZ_L = (A_L - y) / N
        dZ = (AL - y_true) / N
        for l in reversed(range(1, self.num_layers + 1)):
            A_prev = cache[f'A{l-1}']
            grads[f'dW{l}'] = np.dot(A_prev.T, dZ)
            grads[f'db{l}'] = np.sum(dZ, axis=0, keepdims=True)
            
            if l > 1:
                W = self.params[f'W{l}']
                Z_prev = cache[f'Z{l-1}']
                dA_prev = np.dot(dZ, W.T)
                dZ = dA_prev * (Z_prev > 0)  # Turunan ReLU
                
        return grads
        
    def update_parameters(self, grads, beta1=0.9, beta2=0.999, eps=1e-8):
        self.t += 1
        for k in self.params:
            grad = grads[f'd{k}']
            if self.optimizer == 'sgd':
                self.params[k] -= self.lr * grad
            elif self.optimizer == 'adam':
                self.m[k] = beta1 * self.m[k] + (1.0 - beta1) * grad
                self.v[k] = beta2 * self.v[k] + (1.0 - beta2) * (grad ** 2)
                m_hat = self.m[k] / (1.0 - (beta1 ** self.t))
                v_hat = self.v[k] / (1.0 - (beta2 ** self.t))
                self.params[k] -= (self.lr / (np.sqrt(v_hat) + eps)) * m_hat

def run_training_experiment():
    print("=" * 85)
    print("CHAPTER 22: PELATIHAN LENGKAP NUMPY MULTILAYER PERCEPTRON DARI DASAR")
    print("=" * 85)
    
    # 1. Dataset Sintetis Non-Linier
    X, y = generate_circles_dataset(n_samples=400, noise=0.08, random_state=42)
    # Pembagian 80:20
    split = int(0.8 * len(X))
    X_train, y_train = X[:split], y[:split]
    X_val, y_val = X[split:], y[split:]
    
    print(f"Data Latih: {len(X_train)} sampel | Data Validasi: {len(X_val)} sampel")
    print("Arsitektur: [2 masukan -> 8 ReLU -> 4 ReLU -> 1 Sigmoid]")
    print("Optimizer : Adam (lr = 0.03)")
    print("-" * 85)
    print(f"{'Epoch':<8} {'Train Loss':<15} {'Train Acc':<15} {'Val Loss':<15} {'Val Acc':<15}")
    print("-" * 85)
    
    model = NeuralNetworkTrainer(layer_dims=[2, 8, 4, 1], lr=0.03, optimizer='adam', random_state=42)
    epochs = 120
    
    for epoch in range(1, epochs + 1):
        # Forward & Loss
        y_pred_train, cache = model.forward(X_train)
        train_loss = model.compute_loss(y_train, y_pred_train)
        
        # Backward & Update
        grads = model.backward(y_train, cache)
        model.update_parameters(grads)
        
        # Validasi
        y_pred_val, _ = model.forward(X_val)
        val_loss = model.compute_loss(y_val, y_pred_val)
        
        train_acc = np.mean((y_pred_train >= 0.5) == y_train)
        val_acc = np.mean((y_pred_val >= 0.5) == y_val)
        
        if epoch == 1 or epoch % 20 == 0 or epoch == epochs:
            print(f"Epoch {epoch:<4} {train_loss:>10.4f}     {train_acc*100:>10.2f}%    {val_loss:>10.4f}     {val_acc*100:>10.2f}%")
            
    print("=" * 85)
    print("HASIL EVALUASI AKHIR PELATIHAN:")
    print(f"Akurasi Akhir Data Latih    : {train_acc*100:.2f}%")
    print(f"Akurasi Akhir Data Validasi : {val_acc*100:.2f}%")
    assert val_acc >= 0.90, "Model gagal mengonvergensi klasifikasi lingkaran non-linier!"
    print("STATUS KELAYAKAN: LOLOS (Model sukses mengonvergensi batas keputusan non-linier >90%).")

if __name__ == "__main__":
    run_training_experiment()

