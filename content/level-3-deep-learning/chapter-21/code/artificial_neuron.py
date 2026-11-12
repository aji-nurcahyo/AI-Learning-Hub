"""
Chapter 21 - Neural Network Fundamentals
Script 1: Model Neuron Buatan Matematis (Artificial Neuron)
Membandingkan neuron biologis dengan model matematis McCulloch-Pitts & Perceptron:
menghitung kombinasi linier terbobot (weighted sum) z = sum(w_i * x_i) + b
dan transformasi fungsi aktivasi biner (Heaviside step function).
"""

import numpy as np

def artificial_neuron(inputs, weights, bias, activation='step'):
    """
    Menghitung luaran neuron buatan tunggal.
    
    Parameter:
    inputs     : array-like fitur masukan [x_1, x_2, ..., x_n]
    weights    : array-like bobot sinaptik [w_1, w_2, ..., w_n]
    bias       : skalar bias b
    activation : jenis aktivasi ('step', 'linear')
    
    Kembalian:
    z : nilai kombinasi linier (weighted sum)
    a : nilai aktivasi luaran
    """
    x = np.array(inputs, dtype=float)
    w = np.array(weights, dtype=float)
    
    # 1. Linear combination: z = w . x + b
    z = np.dot(w, x) + bias
    
    # 2. Activation function
    if activation == 'step':
        a = 1.0 if z >= 0.0 else 0.0
    elif activation == 'linear':
        a = z
    else:
        raise ValueError(f"Fungsi aktivasi '{activation}' tidak dikenali.")
        
    return z, a

def run_neuron_demonstration():
    print("=" * 75)
    print("CHAPTER 21: MODEL NEURON BUATAN MATEMATIS (ARTIFICIAL NEURON)")
    print("=" * 75)
    
    # Studi kasus: Keputusan Persetujuan Pinjaman Kredit
    # x1: Skor Riwayat Kredit (0: Buruk, 1: Baik)
    # x2: Rasio Pendapatan terhadap Hutang (0: Rendah, 1: Cukup)
    # x3: Jaminan Aset (0: Tidak Ada, 1: Ada)
    feature_names = ["Skor Kredit (x1)", "Rasio Pendapatan (x2)", "Jaminan Aset (x3)"]
    weights = [2.5, 1.8, 1.2]  # Bobot kepentingan
    bias = -3.0               # Batas ambang keputusan awal
    
    test_cases = [
        {"desc": "Profil Risiko Tinggi (Semua Nol)", "inputs": [0, 0, 0]},
        {"desc": "Hanya Punya Jaminan Aset",        "inputs": [0, 0, 1]},
        {"desc": "Skor Baik tapi Pendapatan Rendah", "inputs": [1, 0, 0]},
        {"desc": "Skor Baik dan Pendapatan Cukup",   "inputs": [1, 1, 0]},
        {"desc": "Profil Ideal (Semua Satu)",       "inputs": [1, 1, 1]}
    ]
    
    print(f"Bobot Sinaptik (W) : {weights}")
    print(f"Bias (b)           : {bias}")
    print("-" * 75)
    print(f"{'Skenario Masukan':<35} {'Weighted Sum (z)':<20} {'Aktivasi (a)':<12} {'Keputusan':<10}")
    print("-" * 75)
    
    for case in test_cases:
        inp = case["inputs"]
        z, a = artificial_neuron(inp, weights, bias, activation='step')
        decision = "DISETUJUI (1)" if a == 1.0 else "DITOLAK (0)"
        inp_str = f"{case['desc']} {inp}"
        print(f"{inp_str:<35} {z:>14.2f}       {a:>6.1f}       {decision}")
        
    print("=" * 75)
    print("INTERPRETASI MATEMATIS:")
    print("1. Nilai z merupakan persamaan hyperplane berdimensi 3: 2.5*x1 + 1.8*x2 + 1.2*x3 - 3.0 = 0.")
    print("2. Ketika z >= 0, neuron melepaskan potensial aksi (firing rate output = 1).")
    print("3. Bias menggeser hyperplane keputusan dari titik asal (origin), mengendalikan kemudahan neuron aktif.")

if __name__ == "__main__":
    run_neuron_demonstration()

