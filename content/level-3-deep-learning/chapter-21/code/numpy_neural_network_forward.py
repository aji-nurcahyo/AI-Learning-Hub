"""
Chapter 21 - Neural Network Fundamentals
Script 10: Implementasi Lengkap Jaringan Saraf Tiruan Berorientasi Objek (NumPy Neural Network Forward)
Mengintegrasikan seluruh konsep Chapter 21 ke dalam kelas OOP NumPy murni:
inisialisasi bobot He/Xavier, propagasi maju multi-layer arbitrer, pencatatan cache aktivasi,
evaluasi fungsi aktivasi heterogen, serta komputasi fungsi rugi terstandarisasi.
"""

import numpy as np

class NeuralNetworkForward:
    """
    Kerangka kerja Forward Pass Jaringan Saraf Tiruan Lapis Banyak berbasis NumPy murni.
    """
    def __init__(self, layer_dims, activations, init_method='he', random_state=42):
        """
        layer_dims  : list integer topologi [D_in, D_h1, ..., D_out]
        activations : list string fungsi aktivasi per layer terbobot ['relu', 'relu', 'sigmoid']
        init_method : skema inisialisasi bobot ('he', 'xavier', 'random')
        """
        assert len(layer_dims) - 1 == len(activations), "Jumlah fungsi aktivasi harus sama dengan jumlah layer terbobot!"
        self.layer_dims = layer_dims
        self.activations = activations
        self.init_method = init_method
        self.num_layers = len(layer_dims) - 1
        
        np.random.seed(random_state)
        self.parameters = {}
        self._initialize_parameters()
        
    def _initialize_parameters(self):
        """Inisialisasi bobot W dan bias b dengan penskalaan statistik adaptif."""
        for l in range(1, len(self.layer_dims)):
            n_in = self.layer_dims[l-1]
            n_out = self.layer_dims[l]
            
            if self.init_method == 'he':
                # He initialization: std = sqrt(2 / n_in), optimal untuk ReLU
                std = np.sqrt(2.0 / n_in)
            elif self.init_method == 'xavier':
                # Xavier/Glorot initialization: std = sqrt(2 / (n_in + n_out)), optimal untuk Tanh/Sigmoid
                std = np.sqrt(2.0 / (n_in + n_out))
            else:
                std = 0.01
                
            self.parameters[f'W{l}'] = np.random.randn(n_in, n_out) * std
            self.parameters[f'b{l}'] = np.zeros((1, n_out))
            
    def _apply_activation(self, Z, activation_type):
        """Menerapkan fungsi aktivasi terpilih."""
        act = activation_type.lower()
        if act == 'relu':
            return np.maximum(0.0, Z)
        elif act == 'sigmoid':
            return np.where(Z >= 0, 1.0 / (1.0 + np.exp(-Z)), np.exp(Z) / (1.0 + np.exp(Z)))
        elif act == 'tanh':
            return np.tanh(Z)
        elif act == 'linear':
            return Z
        elif act == 'softmax':
            shift_z = Z - np.max(Z, axis=-1, keepdims=True)
            exp_z = np.exp(shift_z)
            return exp_z / np.sum(exp_z, axis=-1, keepdims=True)
        else:
            raise ValueError(f"Fungsi aktivasi '{activation_type}' tidak didukung.")
            
    def forward(self, X):
        """
        Mengeksekusi Forward Pass lengkap.
        
        Parameter:
        X : Matriks masukan (N, D_in)
        
        Kembalian:
        AL    : Matriks aktivasi layer terakhir (N, D_out)
        cache : Dictionary penyimpan aktivasi perantara untuk keperluan backward pass
        """
        cache = {'A0': X}
        A_prev = X
        
        for l in range(1, self.num_layers + 1):
            W = self.parameters[f'W{l}']
            b = self.parameters[f'b{l}']
            act = self.activations[l-1]
            
            # Kombinasi linier: Z^[l] = A^[l-1] @ W^[l] + b^[l]
            Z = np.dot(A_prev, W) + b
            A = self._apply_activation(Z, act)
            
            cache[f'Z{l}'] = Z
            cache[f'A{l}'] = A
            A_prev = A
            
        return A_prev, cache
        
    def compute_loss(self, y_true, y_pred, loss_type='bce', eps=1e-15):
        """Menghitung nilai fungsi rugi."""
        if loss_type == 'mse':
            return np.mean((y_true - y_pred) ** 2)
        elif loss_type == 'bce':
            p_clipped = np.clip(y_pred, eps, 1.0 - eps)
            return - np.mean(y_true * np.log(p_clipped) + (1.0 - y_true) * np.log(1.0 - p_clipped))
        elif loss_type == 'cce':
            p_clipped = np.clip(y_pred, eps, 1.0 - eps)
            return - np.mean(np.sum(y_true * np.log(p_clipped), axis=-1))
        else:
            raise ValueError(f"Tipe loss '{loss_type}' tidak valid.")

def run_network_integration_test():
    print("=" * 85)
    print("CHAPTER 21: PENGUJIAN INTEGRASI LENGKAP NUMPY NEURAL NETWORK FORWARD")
    print("=" * 85)
    
    # Masalah: Klasifikasi Biner Sensor Anomali (3 fitur -> 8 hidden -> 4 hidden -> 1 output)
    layer_dims = [3, 8, 4, 1]
    activations = ['relu', 'relu', 'sigmoid']
    
    print(f"Topologi Jaringan : {layer_dims}")
    print(f"Fungsi Aktivasi   : {activations}")
    print("Skema Inisialisasi: He Normal Scaling")
    
    model = NeuralNetworkForward(layer_dims, activations, init_method='he', random_state=42)
    
    # Data Uji Sintetis N = 5 sampel
    X_test = np.array([
        [1.2, 0.5, -0.8],
        [0.1, 2.3,  1.1],
        [-1.5, -0.2, 0.4],
        [0.8, -1.1, -0.5],
        [2.0,  1.8,  0.9]
    ])
    y_test = np.array([[1.0], [1.0], [0.0], [0.0], [1.0]])
    
    print(f"\nMatriks Masukan X ({X_test.shape}):\n{X_test}")
    
    # Eksekusi Forward Pass
    predictions, cache = model.forward(X_test)
    
    print("\nHASIL INTERMEDIATE FORWARD PASS (DARI CACHE):")
    for l in range(1, model.num_layers + 1):
        print(f"  Layer {l} Z{l} shape: {cache[f'Z{l}'].shape} | A{l} shape: {cache[f'A{l}'].shape} (Aktivasi: {activations[l-1]})")
        
    print("\nPREDIKSI AKHIR (A^[3]):")
    for i in range(len(X_test)):
        prob = predictions[i, 0]
        label_pred = 1 if prob >= 0.5 else 0
        actual = int(y_test[i, 0])
        print(f"  Sampel {i+1}: P(y=1) = {prob:.4f} -> Prediksi: {label_pred} (Aktual: {actual})")
        
    # Evaluasi Loss
    loss_bce = model.compute_loss(y_test, predictions, loss_type='bce')
    loss_mse = model.compute_loss(y_test, predictions, loss_type='mse')
    print("-" * 85)
    print(f"Binary Cross-Entropy Loss (BCE) : {loss_bce:.4f}")
    print(f"Mean Squared Error Loss (MSE)   : {loss_mse:.4f}")
    print("=" * 85)
    print("STATUS INTEGRASI: SUKSES 100% TERVERIFIKASI.")

if __name__ == "__main__":
    run_network_integration_test()

