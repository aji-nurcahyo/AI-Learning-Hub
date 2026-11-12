# Chapter 37: Edge AI & Model Optimization — Code Catalog

Koleksi 16 script Python mandiri, deterministik, dan berjalan pada CPU lokal (PyTorch `2.15.0+cpu`, NumPy, Matplotlib) untuk mempelajari kompresi dan optimasi model Edge AI secara kuantitatif.

## Daftar Script

1. `edge_ai_foundations.py`: Dataset telemetri IoT sintetis 4-kelas (16 fitur) bebas kebocoran data beserta arsitektur baseline `EdgeTeacherMLP` (~46K parameter).
2. `model_profiling.py`: Profiling menyeluruh mencakup jumlah parameter, ukuran artefak `.pt` di disk, latensi cold/warm (Mean, P50, P95 ms), serta alokasi heap puncak `tracemalloc`.
3. `fp32_int8_quantization_math.py`: Perhitungan manual kuantisasi simetris dan asimetris afine dari FP32 ke INT8, faktor skala, zero-point, dekuantisasi, serta kesalahan MSE.
4. `quantization_error.py`: Analisis *Signal-to-Quantization-Noise Ratio* (SQNR) pada distribusi seragam, Gaussian, dan keberadaan outlier serta perbandingan *Per-Tensor* vs *Per-Channel*.
5. `dynamic_quantization.py`: Kuantisasi dinamis PyTorch (`quantize_dynamic` backend `onednn`) membandingkan ukuran disk, akurasi, dan latensi CPU aktual vs FP32.
6. `static_quantization_demo.py`: *Post-Training Static Quantization* lengkap dengan `QuantStub`/`DeQuantStub`, kalibrasi pada 250 sampel acuan, serta konversi INT8 statis.
7. `quantization_evaluation.py`: Audit kesetiaan numerik keluaran (*logit agreement*, *MAE*, *cosine similarity*) antara baseline FP32, Dynamic INT8, dan Static INT8.
8. `magnitude_pruning.py`: *Unstructured L1 Magnitude Pruning* hingga rasio jarang 70%, pemulihan akurasi 5-epoch, serta pembuktian bahwa berkas tensor padat `.pt` tidak otomatis mengecil tanpa kompresi khusus.
9. `structured_pruning_demo.py`: Pemangkasan terstruktur (*structured neuron pruning*) menghapus 50% neuron L1 terendah sehingga jumlah parameter dan ukuran `.pt` menyusut secara fisik.
10. `pruning_sparsity_analysis.py`: Kurva sensitivitas tingkat kejarangan (*sparsity* 0%–95%) terhadap akurasi sebelum dan sesudah fine-tuning (`experiments/pruning_sparsity_curve.png`).
11. `knowledge_distillation.py`: Demonstrasi *Teacher–Student* PyTorch menggunakan *Temperature-Scaled Softmax* ($T=3.0$) dan $T^2$-scaled KL Divergence dengan pembuktian bobot guru membeku (*frozen*).
12. `teacher_student_comparison.py`: Komparasi metrik langsung antara *Teacher*, *Student (Scratch)*, dan *Distilled Student*.
13. `model_size_comparison.py`: Perbandingan rasio ukuran berkas `.pt` mentah versus arsip kompresi `gzip` di seluruh metode optimasi.
14. `inference_latency_benchmark.py`: Pengukuran latensi CPU berulang (*3 trials x 400 passes*, 50 warm-up) dengan laporan Mean $\pm$ Std, P50, P95, serta konteks perangkat keras.
15. `optimization_comparison.py`: Kartu skor optimasi terpadu beserta grafik trade-off multi-panel (`experiments/edge_optimization_tradeoffs.png`).
16. `edge_hardware_constraints.py`: Penilaian simulasi anggaran memori SRAM/Flash terhadap batas mikrokontroler teoretis (ARM Cortex-M7, ESP32-S3, Raspberry Pi 4).

## Cara Menjalankan

```bash
# Jalankan perbandingan optimasi terpadu
python content/level-4-advanced-ai/chapter-37/code/optimization_comparison.py

# Jalankan benchmark latensi berulang pada CPU
python content/level-4-advanced-ai/chapter-37/code/inference_latency_benchmark.py
```

