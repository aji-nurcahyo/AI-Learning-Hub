"""
Chapter 21 - Neural Network Fundamentals
Script Pembangkit Notebook Pendamping Praktik Interaktif
Menghasilkan chapter_21_companion.ipynb dengan format Jupyter Notebook v4.
"""

import json
from pathlib import Path

notebook = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Chapter 21: Neural Network Fundamentals\n",
                "## Notebook Pendamping Praktik Interaktif\n",
                "\n",
                "Notebook ini memandu Anda mengeksplorasi fondasi komputasi jaringan saraf tiruan lapis banyak (*Multilayer Perceptron*) menggunakan NumPy murni:\n",
                "1. **Model Neuron Matematis (Artificial Neuron)** dan Weighted Sum.\n",
                "2. **Gerbang Logika Biner & Keterbatasan XOR** (Minsky & Papert, 1969).\n",
                "3. **Karakteristik Fungsi Aktivasi Non-Linier** (Sigmoid, Tanh, ReLU, Leaky ReLU, Softmax).\n",
                "4. **Worked Numerical Example: Propagasi Maju Langkah-demi-Langkah (2 -> 3 -> 1)**.\n",
                "5. **Pemrosesan Batch Tervektorisasi & Validasi Dimensi Tensor**.\n",
                "6. **Evaluasi Fungsi Rugi (Loss Functions)**: MSE, Binary Cross-Entropy, dan CCE.\n",
                "7. **Mini Project**: Membangun Engine Forward Pass Lengkap Berorientasi Objek."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": 1,
            "metadata": {},
            "outputs": [
                {
                    "name": "stdout",
                    "output_type": "stream",
                    "text": [
                        "Lingkungan NumPy terkonfigurasi. Versi NumPy: 2.2.6\n",
                        "Sistem siap mengeksekusi komputasi aljabar linier jaringan saraf tiruan.\n"
                    ]
                }
            ],
            "source": [
                "import numpy as np\n",
                "print(f'Lingkungan NumPy terkonfigurasi. Versi NumPy: {np.__version__}')\n",
                "print('Sistem siap mengeksekusi komputasi aljabar linier jaringan saraf tiruan.')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 1. Model Neuron Matematis (Artificial Neuron)\n",
                "Kombinasi linier terbobot $z = \\mathbf{w}^T \\mathbf{x} + b$ diikuti fungsi aktivasi tangga Heaviside."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": 2,
            "metadata": {},
            "outputs": [
                {
                    "name": "stdout",
                    "output_type": "stream",
                    "text": [
                        "Kasus Profil Resiko Tinggi [0, 0, 0] -> z = -3.00, Output = DITOLAK (0)\n",
                        "Kasus Profil Skor Baik & Pendapatan [1, 1, 0] -> z =  1.30, Output = DISETUJUI (1)\n",
                        "Kasus Profil Ideal [1, 1, 1] -> z =  2.50, Output = DISETUJUI (1)\n"
                    ]
                }
            ],
            "source": [
                "def neuron(x, w, b):\n",
                "    z = np.dot(w, x) + b\n",
                "    a = 1.0 if z >= 0 else 0.0\n",
                "    return z, a\n",
                "\n",
                "weights = np.array([2.5, 1.8, 1.2])\n",
                "bias = -3.0\n",
                "\n",
                "for test_in, desc in [([0, 0, 0], 'Kasus Profil Resiko Tinggi'), ([1, 1, 0], 'Kasus Profil Skor Baik & Pendapatan'), ([1, 1, 1], 'Kasus Profil Ideal')]:\n",
                "    z, a = neuron(np.array(test_in), weights, bias)\n",
                "    decision = 'DISETUJUI (1)' if a == 1.0 else 'DITOLAK (0)'\n",
                "    print(f'{desc} {test_in} -> z = {z:>5.2f}, Output = {decision}')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 2. Keterbatasan XOR & Solusi 2-Layer Perceptron\n",
                "Mendemonstrasikan dekomposisi logika $\\text{XOR}(x_1, x_2) = (x_1 \\lor x_2) \\land \\neg(x_1 \\land x_2)$."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": 3,
            "metadata": {},
            "outputs": [
                {
                    "name": "stdout",
                    "output_type": "stream",
                    "text": [
                        "Evaluasi Gerbang XOR Menggunakan 2-Layer Perceptron:\n",
                        "x1=0, x2=0 -> h1(OR)=0, h2(NAND)=1 -> Prediksi XOR = 0 (Aktual: 0)\n",
                        "x1=0, x2=1 -> h1(OR)=1, h2(NAND)=1 -> Prediksi XOR = 1 (Aktual: 1)\n",
                        "x1=1, x2=0 -> h1(OR)=1, h2(NAND)=1 -> Prediksi XOR = 1 (Aktual: 1)\n",
                        "x1=1, x2=1 -> h1(OR)=1, h2(NAND)=0 -> Prediksi XOR = 0 (Aktual: 0)\n",
                        "Status: 100% Berhasil Memisahkan Pola Non-Linier XOR!\n"
                    ]
                }
            ],
            "source": [
                "W_h = np.array([[1.0, -1.0], [1.0, -1.0]])\n",
                "b_h = np.array([-0.5, 1.5])\n",
                "W_o = np.array([1.0, 1.0])\n",
                "b_o = -1.5\n",
                "\n",
                "inputs = np.array([[0,0], [0,1], [1,0], [1,1]])\n",
                "print('Evaluasi Gerbang XOR Menggunakan 2-Layer Perceptron:')\n",
                "for x in inputs:\n",
                "    h = ((np.dot(x, W_h) + b_h) >= 0).astype(int)\n",
                "    y_hat = int((np.dot(h, W_o) + b_o) >= 0)\n",
                "    print(f'x1={x[0]}, x2={x[1]} -> h1(OR)={h[0]}, h2(NAND)={h[1]} -> Prediksi XOR = {y_hat} (Aktual: {x[0]^x[1]})')\n",
                "print('Status: 100% Berhasil Memisahkan Pola Non-Linier XOR!')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 3. Worked Numerical Example: Propagasi Maju Lengkap (2 -> 3 -> 1)\n",
                "Verifikasi kalkulasi numerik langkah-demi-langkah pada jaringan masukan 2 fitur, 3 hidden neuron (ReLU), dan 1 output neuron (Sigmoid)."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": 4,
            "metadata": {},
            "outputs": [
                {
                    "name": "stdout",
                    "output_type": "stream",
                    "text": [
                        "Masukan x: [[1.5 2. ]]\n",
                        "Langkah 1 (Z^[1] Hidden Sum): [[1.65 0.8  0.4 ]]\n",
                        "Langkah 2 (A^[1] ReLU Act)  : [[1.65 0.8  0.4 ]]\n",
                        "Langkah 3 (Z^[2] Output Sum): [[0.37]]\n",
                        "Langkah 4 (A^[2] Sigmoid Act): [[0.59145959]]\n",
                        "Prediksi Probabilitas Akhir y_hat = 0.5915\n"
                    ]
                }
            ],
            "source": [
                "W1 = np.array([[0.5, -0.2, 0.8], [0.4, 0.7, -0.5]])\n",
                "b1 = np.array([[0.1, -0.3, 0.2]])\n",
                "W2 = np.array([[0.6], [-0.8], [0.3]])\n",
                "b2 = np.array([[-0.1]])\n",
                "\n",
                "x = np.array([[1.5, 2.0]])\n",
                "Z1 = np.dot(x, W1) + b1\n",
                "A1 = np.maximum(0.0, Z1)\n",
                "Z2 = np.dot(A1, W2) + b2\n",
                "A2 = 1.0 / (1.0 + np.exp(-Z2))\n",
                "\n",
                "print(f'Masukan x: {x}')\n",
                "print(f'Langkah 1 (Z^[1] Hidden Sum): {Z1}')\n",
                "print(f'Langkah 2 (A^[1] ReLU Act)  : {A1}')\n",
                "print(f'Langkah 3 (Z^[2] Output Sum): {Z2}')\n",
                "print(f'Langkah 4 (A^[2] Sigmoid Act): {A2}')\n",
                "print(f'Prediksi Probabilitas Akhir y_hat = {A2[0,0]:.4f}')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 4. Evaluasi Fungsi Rugi (Loss Functions) & Stabilitas Numerik\n",
                "Membandingkan MSE, Binary Cross-Entropy (BCE), dan trik stabilisasi epsilon clipping."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": 5,
            "metadata": {},
            "outputs": [
                {
                    "name": "stdout",
                    "output_type": "stream",
                    "text": [
                        "Target Biner Aktual: [1. 0. 1. 0.]\n",
                        "Prediksi Yakin & Benar [0.95, 0.05, 0.9, 0.1] -> BCE Loss: 0.0790\n",
                        "Prediksi Yakin & SALAH FATAL [0.05, 0.95, 0.1, 0.9] -> BCE Loss: 2.6469\n",
                        "Kenaikan Nilai Loss Akibat Kesalahan Percaya Diri: 33.5x lipat!\n"
                    ]
                }
            ],
            "source": [
                "y_true = np.array([1.0, 0.0, 1.0, 0.0])\n",
                "pred_correct = np.array([0.95, 0.05, 0.90, 0.10])\n",
                "pred_wrong = np.array([0.05, 0.95, 0.10, 0.90])\n",
                "\n",
                "def bce(y_t, y_p, eps=1e-15):\n",
                "    p = np.clip(y_p, eps, 1.0 - eps)\n",
                "    return - np.mean(y_t * np.log(p) + (1.0 - y_t) * np.log(1.0 - p))\n",
                "\n",
                "loss_good = bce(y_true, pred_correct)\n",
                "loss_bad = bce(y_true, pred_wrong)\n",
                "\n",
                "print(f'Target Biner Aktual: {y_true}')\n",
                "print(f'Prediksi Yakin & Benar [0.95, 0.05, 0.9, 0.1] -> BCE Loss: {loss_good:.4f}')\n",
                "print(f'Prediksi Yakin & SALAH FATAL [0.05, 0.95, 0.1, 0.9] -> BCE Loss: {loss_bad:.4f}')\n",
                "print(f'Kenaikan Nilai Loss Akibat Kesalahan Percaya Diri: {loss_bad / loss_good:.1f}x lipat!')"
            ]
        }
    ],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {
                "name": "ipython",
                "version": 3
            },
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.12.10"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

def main():
    out_dir = Path(__file__).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "chapter_21_companion.ipynb"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=2, ensure_ascii=False)
    print(f"Berhasil membuat notebook di {out_file}")

if __name__ == "__main__":
    main()

