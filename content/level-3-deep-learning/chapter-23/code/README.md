# Katalog Skrip Praktik: Chapter 23 — Deep Learning with PyTorch

Direktori ini berisi implementasi kode Python modular dan runnable untuk **Bab 23: Deep Learning with PyTorch**. Seluruh skrip menggunakan framework PyTorch resmi pada jalur komputasi CPU dan kompatibel dengan versi Python 3.12.

---

## 1. Daftar Skrip dan Deskripsi Komponen

| No | Nama Berkas | Deskripsi Fungsional |
|---|---|---|
| 1 | [`tensor_fundamentals.py`](tensor_fundamentals.py) | Operasi dasar tensor, inspeksi shape, ndim, dtype, device, manipulasi bentuk (view, reshape, squeeze, unsqueeze, permute, flatten), aljabar broadcasting, serta jembatan zero-copy dua arah dengan NumPy. |
| 2 | [`autograd_mechanics.py`](autograd_mechanics.py) | Mekanisme kalkulus otomatis (autograd), verifikasi turunan analitis vs autograd, pembuktian bahaya akumulasi gradien bawaan dan perlunya `zero_grad()`, serta penghematan memori via `torch.no_grad()` dan `.detach()`. |
| 3 | [`dataset_dataloader.py`](dataset_dataloader.py) | Abstraksi custom `Dataset` dan `DataLoader`, pembangkitan dataset sintetis klasifikasi berlabel jelas, pemisahan partisi 3-way (Train 70%, Val 15%, Test 15%) tanpa tumpang tindih, dan normalisasi bebas kebocoran data (*zero data leakage*). |
| 4 | [`neural_network_module.py`](neural_network_module.py) | Arsitektur jaringan saraf modular berbasis `torch.nn.Module`, forward pass, inspeksi parameter bobot/bias, serta penjelasan mengapa luaran model berupa logits mentah tanpa Softmax saat menggunakan `nn.CrossEntropyLoss`. |
| 5 | [`training_validation_pipeline.py`](training_validation_pipeline.py) | Alur pelatihan dan validasi lengkap terstandar industri (`model.train()`, `model.eval()`, `torch.no_grad()`), perbandingan performa MLP vs Baseline linier, evaluasi matriks konfusi, dan penyimpanan log hasil terstruktur. |
| 6 | [`checkpoint_inference.py`](checkpoint_inference.py) | Praktik terbaik penyimpanan parameter (`state_dict`), pemuatan aman dengan parameter `weights_only=True`, pengujian konsistensi inferensi sebelum vs sesudah reload (`torch.allclose`), dan fungsi prediksi produksi. |

---

## 2. Cara Menjalankan Skrip

Pastikan virtual environment telah aktif:
```bash
# Windows PowerShell
.venv\Scripts\python.exe content/level-3-deep-learning/chapter-23/code/tensor_fundamentals.py
.venv\Scripts\python.exe content/level-3-deep-learning/chapter-23/code/autograd_mechanics.py
.venv\Scripts\python.exe content/level-3-deep-learning/chapter-23/code/dataset_dataloader.py
.venv\Scripts\python.exe content/level-3-deep-learning/chapter-23/code/neural_network_module.py
.venv\Scripts\python.exe content/level-3-deep-learning/chapter-23/code/training_validation_pipeline.py
.venv\Scripts\python.exe content/level-3-deep-learning/chapter-23/code/checkpoint_inference.py
```

---

## 3. Hasil Pengujian Aktual (Terverifikasi)
- **Environment**: Python 3.12.10, PyTorch 2.15.0+cpu, torchvision 0.29.2+cpu, NumPy 2.5.4.
- **Gradien Autograd**: Selisih analitis vs autograd $< 10^{-5}$ pada seluruh turunan parsial ($w_1, w_2, b$).
- **Partisi Data**: Train (700 sampel), Val (150 sampel), Test (150 sampel) dengan 0% tumpang tindih index.
- **Performa Model (Test Set Terisolasi N=150)**:
  - `TabularMLP` (Non-linier, 266 parameter): Akurasi **88.67%**, Loss **0.3234**.
  - `SimpleBaseline` (Linier, 14 parameter): Akurasi **83.33%**, Loss **0.4274**.
  - Keunggulan representasi non-linier terbukti terukur (**+5.33%**).
- **Inference Checkpoint**: Logits model reloaded terbukti identik hingga toleransi mesin ($0.00 \times 10^0$).

