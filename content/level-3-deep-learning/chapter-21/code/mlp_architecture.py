"""
Chapter 21 - Neural Network Fundamentals
Script 6: Arsitektur Multilayer Perceptron (MLP Architecture & Parameter Counting)
Mendefinisikan topologi jaringan saraf tiruan lapis banyak, memetakan dimensi matriks
bobot dan vektor bias di setiap layer, serta menghitung total parameter yang dapat dilatih.
"""

import numpy as np

class MLPArchitecture:
    """Mendefinisikan dan memvalidasi spesifikasi topologi Multilayer Perceptron."""
    def __init__(self, layer_dims):
        """
        layer_dims: list bilangan bulat berisi jumlah neuron per layer,
                    contoh: [4, 8, 5, 3] berarti 4 input, 2 hidden layers (8 dan 5), 3 output.
        """
        self.layer_dims = layer_dims
        self.num_layers = len(layer_dims) - 1  # Tidak menghitung input layer sebagai layer terbobot
        self.weights_shapes = []
        self.biases_shapes = []
        self._calculate_shapes()
        
    def _calculate_shapes(self):
        for l in range(1, len(self.layer_dims)):
            # Bobot W^[l] menghubungkan layer l-1 ke layer l
            # Konvensi bentuk forward batch: X @ W + b
            # Matriks bobot: (n_{l-1}, n_l)
            w_shape = (self.layer_dims[l-1], self.layer_dims[l])
            b_shape = (1, self.layer_dims[l])
            self.weights_shapes.append(w_shape)
            self.biases_shapes.append(b_shape)
            
    def count_parameters(self):
        total_w = sum(w[0] * w[1] for w in self.weights_shapes)
        total_b = sum(b[1] for b in self.biases_shapes)
        return total_w, total_b, total_w + total_b
        
    def print_summary(self):
        print("=" * 80)
        print("RINGKASAN ARSITEKTUR MULTILAYER PERCEPTRON (MLP)")
        print("=" * 80)
        print(f"Topologi Dimensi Layer: {self.layer_dims}")
        print(f"Jumlah Lapisan Terbobot (Depth L): {self.num_layers}")
        print("-" * 80)
        print(f"{'Layer':<10} {'Tipe Layer':<15} {'Jumlah Neuron':<15} {'Bentuk Matriks W':<20} {'Bentuk Bias b':<15} {'Total Param':<10}")
        print("-" * 80)
        
        # Layer 0 (Input)
        print(f"{'Layer 0':<10} {'Input Layer':<15} {self.layer_dims[0]:<15} {'(None)':<20} {'(None)':<15} {0:<10}")
        
        total_params = 0
        for l in range(1, len(self.layer_dims)):
            layer_type = "Hidden Layer" if l < self.num_layers else "Output Layer"
            w_shape = self.weights_shapes[l-1]
            b_shape = self.biases_shapes[l-1]
            n_w = w_shape[0] * w_shape[1]
            n_b = b_shape[1]
            layer_params = n_w + n_b
            total_params += layer_params
            
            print(f"{f'Layer {l}':<10} {layer_type:<15} {self.layer_dims[l]:<15} {str(w_shape):<20} {str(b_shape):<15} {layer_params:<10}")
            
        total_w, total_b, total = self.count_parameters()
        print("=" * 80)
        print(f"TOTAL PARAMETER DAPAT DILATIH (TRAINABLE PARAMETERS): {total}")
        print(f"  - Total Bobot Sinaptik (Weights) : {total_w}")
        print(f"  - Total Parameter Bias (Biases)  : {total_b}")
        print("=" * 80)

def demonstrate_architectures():
    # Contoh 1: Jaringan Klasifikasi Iris (4 fitur -> 8 hidden -> 3 kelas)
    mlp_iris = MLPArchitecture([4, 8, 3])
    mlp_iris.print_summary()
    
    # Contoh 2: Jaringan Deep Vision Sederhana (784 pixel -> 128 hidden -> 64 hidden -> 10 digit)
    print("\n")
    mlp_vision = MLPArchitecture([784, 128, 64, 10])
    mlp_vision.print_summary()

if __name__ == "__main__":
    demonstrate_architectures()

