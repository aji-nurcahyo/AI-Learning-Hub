"""
Chapter 21 - Neural Network Fundamentals
Script 4: Fungsi Aktivasi Non-Linier & Fenomena Dead ReLU
Mengimplementasikan fungsi aktivasi utama: Sigmoid, Tanh, ReLU, dan Leaky ReLU,
menghitung turunan matematisnya, dan mendemonstrasikan fenomena saturasi dan Dead ReLU.
"""

import numpy as np

def sigmoid(z):
    # Numerically stable sigmoid: mencegah overflow pada nilai z negatif besar
    return np.where(z >= 0, 1.0 / (1.0 + np.exp(-z)), np.exp(z) / (1.0 + np.exp(z)))

def sigmoid_derivative(z):
    s = sigmoid(z)
    return s * (1.0 - s)

def tanh(z):
    return np.tanh(z)

def tanh_derivative(z):
    return 1.0 - np.tanh(z) ** 2

def relu(z):
    return np.maximum(0.0, z)

def relu_derivative(z):
    return np.where(z > 0, 1.0, 0.0)

def leaky_relu(z, alpha=0.01):
    return np.where(z > 0, z, alpha * z)

def leaky_relu_derivative(z, alpha=0.01):
    return np.where(z > 0, 1.0, alpha)

def demonstrate_activations():
    print("=" * 80)
    print("CHAPTER 21: ANALISIS KARAKTERISTIK FUNGSI AKTIVASI NEURAL NETWORK")
    print("=" * 80)
    
    test_values = np.array([-10.0, -3.0, -1.0, -0.1, 0.0, 0.1, 1.0, 3.0, 10.0])
    
    print(f"{'Input (z)':<10} {'Sigmoid':<12} {'Tanh':<12} {'ReLU':<12} {'Leaky ReLU':<14} {'Sigmoid Derivative':<20}")
    print("-" * 80)
    
    for z in test_values:
        sig_val = sigmoid(z)
        tanh_val = tanh(z)
        relu_val = relu(z)
        lrelu_val = leaky_relu(z)
        sig_grad = sigmoid_derivative(z)
        print(f"{z:>8.1f}   {sig_val:>8.4f}     {tanh_val:>8.4f}     {relu_val:>8.4f}     {lrelu_val:>8.4f}         {sig_grad:>10.4f}")
        
    print("=" * 80)
    print("\nPERBANDINGAN SIFAT MATEMATIS:")
    print("1. Sigmoid   : Domain (-inf, inf) -> Rentang (0, 1). Mengalami saturasi saat |z| > 4 (gradien mendekati 0).")
    print("2. Tanh      : Domain (-inf, inf) -> Rentang (-1, 1). Zero-centered, gradien lebih kuat daripada sigmoid.")
    print("3. ReLU      : Domain (-inf, inf) -> Rentang [0, inf). Tidak jenuh pada z > 0, komputasi sangat cepat.")
    print("4. Leaky ReLU: Mengatasi Dead ReLU dengan memberikan gradien kecil (alpha=0.01) saat z < 0.")

def demonstrate_dead_relu():
    print("\n" + "=" * 80)
    print("DEMONSTRASI FENOMENA 'DEAD RELU':")
    print("=" * 80)
    negative_inputs = np.array([-5.0, -3.2, -1.8, -0.5])
    print(f"Masukan Negatif        : {negative_inputs}")
    print(f"Luaran ReLU (f(z))     : {relu(negative_inputs)}")
    print(f"Gradien ReLU (f'(z))   : {relu_derivative(negative_inputs)}")
    print(f"Luaran Leaky ReLU      : {leaky_relu(negative_inputs)}")
    print(f"Gradien Leaky ReLU     : {leaky_relu_derivative(negative_inputs)}")
    print("\nPenjelasan:")
    print("Ketika neuron ReLU selalu menghasilkan z <= 0, gradiennya menjadi nol permanen.")
    print("Neuron tersebut 'mati' dan tidak akan pernah memperbarui bobotnya selama sisa proses training.")

if __name__ == "__main__":
    demonstrate_activations()
    demonstrate_dead_relu()

