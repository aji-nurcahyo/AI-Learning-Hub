# Chapter 22: Backpropagation & Optimization — Code Catalog

Katalog skrip Python mandiri berbasis NumPy murni untuk **Chapter 22: Backpropagation & Optimization**. Seluruh skrip dirancang untuk membedah kalkulus diferensiasi otomatis (*Automatic Differentiation*), algoritma perambatan mundur (*Backpropagation*), verifikasi gradien analitis (*Gradient Checking*), serta dinamika pengoptimal tingkat lanjut (*Gradient Descent, Momentum, RMSprop, Adam*).

---

## Daftar Skrip Python

| No | Berkas Skrip | Topik / Fokus Utama | Output Terverifikasi |
|---|---|---|---|
| 1 | `chain_rule_demo.py` | Penerapan aturan rantai kalkulus pada fungsi komposit 3-tahap | Komparasi turunan analitis vs beda hingga terpusat |
| 2 | `computational_graph.py` | Mini-engine graf komputasi (Directed Acyclic Graph) dengan autograd topological sort | Evaluasi gradien dL/dw1, dL/dw2, dL/db |
| 3 | `numerical_derivatives.py` | Komparasi orde akurasi Forward, Backward, dan Central Finite Differences | Analisis batas truncation vs roundoff error |
| 4 | `manual_backpropagation.py` | Worked example backpropagation manual langkah-demi-langkah (arsitektur 2-2-1) | Pembuktian penurunan loss setelah update parameter |
| 5 | `gradient_checking.py` | Prosedur verifikasi formal gradien Backpropagation terhadap beda terpusat | Relatif error $< 10^{-7}$ (Lolos audit 100%) |
| 6 | `batch_gradient_descent.py` | Optimasi deterministik pada seluruh dataset per langkah pembaruan | Kurva konvergensi loss mulus |
| 7 | `stochastic_gradient_descent.py` | Pembaruan parameter per 1 sampel (Batch Size = 1) | Fluktuasi stokastik dan kecepatan update |
| 8 | `mini_batch_gradient_descent.py` | Standar industri Mini-Batch GD (B = 32), epoch vs iterasi | Keseimbangan stabilitas gradien dan paralelisasi |
| 9 | `momentum_optimizer.py` | Akumulasi kecepatan gerak (Velocity) untuk meredam osilasi pada lembah curam | Akselerasi penelusuran sumbu landai |
| 10 | `rmsprop_optimizer.py` | Laju pembelajaran adaptif per dimensi berbasis rata-rata bergerak gradien kuadrat | Stabilisasi dimensi bergradien heterogen |
| 11 | `adam_optimizer.py` | Pengoptimal Adam lengkap dengan koreksi bias pada first & second moments | Konvergensi cepat pada fungsi 2D anisotropic |
| 12 | `learning_rate_comparison.py` | Eksperimen batas stabilitas Lipschitz continuous gradients ($\eta = 0.01, 0.1, 0.35, 0.45$) | Bukti empiris overshooting dan divergensi meledak |
| 13 | `weight_initialization.py` | Investigasi perambatan varians pada 10 hidden layer (Zero, Normal, Xavier, He) | Bukti kestabilan He pada ReLU dan Xavier pada Tanh |
| 14 | `gradient_clipping.py` | Mitigasi exploding gradients: Clip by Value vs Clip by Global Norm | Pembuktian arah vektor 100% terjaga pada norm clipping |
| 15 | `numpy_neural_network_training.py` | Training loop lengkap Multilayer Perceptron dari dasar pada dataset lingkaran non-linier | Akurasi validasi konvergen $>95\%$ |

---

## Petunjuk Menjalankan

Seluruh berkas Python dapat dijalankan secara mandiri dari terminal:

```bash
# Menjalankan verifikasi Gradient Checking formal
python content/level-3-deep-learning/chapter-22/code/gradient_checking.py

# Menjalankan worked example perhitungan manual propagasi mundur
python content/level-3-deep-learning/chapter-22/code/manual_backpropagation.py

# Menjalankan pelatihan lengkap MLP menggunakan Adam Optimizer
python content/level-3-deep-learning/chapter-22/code/numpy_neural_network_training.py
```

