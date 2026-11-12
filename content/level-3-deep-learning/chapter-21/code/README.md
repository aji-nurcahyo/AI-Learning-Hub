# Chapter 21: Neural Network Fundamentals — Code Catalog

Katalog skrip Python mandiri berbasis NumPy murni untuk **Chapter 21: Neural Network Fundamentals**. Seluruh contoh kode dirancang tanpa ketergantungan wajib pada framework berat (seperti PyTorch/TensorFlow) untuk menjamin pemahaman dasar mekanisme aljabar linier jaringan saraf tiruan.

---

## Daftar Skrip Python

| No | Berkas Skrip | Topik / Fokus Utama | Output Terverifikasi |
|---|---|---|---|
| 1 | `artificial_neuron.py` | Model matematis neuron buatan (McCulloch-Pitts & Perceptron), weighted sum $z = \mathbf{w}^T \mathbf{x} + b$, dan fungsi step | Simulasi keputusan kredit biner |
| 2 | `perceptron_logic_gates.py` | Implementasi gerbang logika biner linier: AND, OR, NAND, NOR | Tabel kebenaran & batas hyperplane |
| 3 | `xor_limitation.py` | Bukti aljabar kontradiksi matematis ketidakmampuan Perceptron pada XOR & solusi MLP 2-layer | Dekomposisi $(x_1 \lor x_2) \land \neg(x_1 \land x_2)$ |
| 4 | `activation_functions.py` | Karakteristik Sigmoid, Tanh, ReLU, Leaky ReLU, turunan, dan fenomena Dead ReLU | Evaluasi nilai & gradien pada domain $[-10, 10]$ |
| 5 | `stable_softmax.py` | Fungsi Softmax stabil numerik untuk klasifikasi multikelas (subtract-max trick) | Pencegahan overflow $\exp(1002)$ |
| 6 | `mlp_architecture.py` | Topologi MLP, pemetaan dimensi matriks bobot $W^{[l]}$ & bias $b^{[l]}$, serta penghitungan parameter | Ringkasan arsitektur 4-8-3 dan 784-128-64-10 |
| 7 | `forward_propagation.py` | Worked numerical example lengkap arsitektur 2-3-1 (kalkulasi manual vs matriks NumPy) | Sampel tunggal & batch $N=4$ |
| 8 | `matrix_shape_demo.py` | Validasi aturan dimensi perkalian matriks dan broadcasting bias pada batch size 1, 16, 64, 128 | Validasi dimensi tensor tanpa error |
| 9 | `loss_functions.py` | Evaluasi fungsi rugi Mean Squared Error (MSE), Binary Cross-Entropy (BCE), dan CCE dengan epsilon clipping | Penalti logaritmik kesalahan percaya diri |
| 10 | `numpy_neural_network_forward.py` | Kelas OOP `NeuralNetworkForward` modular lengkap dengan inisialisasi bobot He/Xavier | Integrasi forward pass & loss computation |

---

## Petunjuk Menjalankan

Seluruh skrip dapat dijalankan langsung menggunakan interpreter Python:

```bash
# Menjalankan demonstrasi model neuron matematis
python content/level-3-deep-learning/chapter-21/code/artificial_neuron.py

# Menjalankan bukti keterbatasan XOR dan solusi MLP
python content/level-3-deep-learning/chapter-21/code/xor_limitation.py

# Menjalankan integrasi forward pass lengkap
python content/level-3-deep-learning/chapter-21/code/numpy_neural_network_forward.py
```

