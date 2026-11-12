# Katalog Kode & Panduan Praktik Laboratorium: Bab 30
## Deep Learning Capstone Reference Implementation

Direktori ini memuat implementasi proyek acuan (*Reference Implementation*) **Level 3 Deep Learning Capstone**: **End-to-End Neural Network Classification Using PyTorch** yang dieksekusi secara efisien dan deterministik pada CPU komputer standar.

---

### Struktur Komponen Proyek Capstone

| No | Berkas Skrip | Peran dalam Alur Capstone (8 Tahapan Siklus Hidup) | Pustaka Utama |
|---|---|---|---|
| 1 | `dataset_loader.py` | **Tahap 1 & 2**: Pemuatan data, skema validasi, dan partisi 3-way (*zero data leakage*). | `pandas`, `numpy` |
| 2 | `baseline_model.py` | **Tahap 3**: Model acuan pembanding (Logistic Regression) penentu batas bawah performa. | `sklearn.linear_model` |
| 3 | `deep_learning_model.py` | **Tahap 3**: Arsitektur jaringan saraf mendalam (`DeepPredictiveMaintenanceNet`) dengan blok residu. | `torch.nn` |
| 4 | `training_engine.py` | **Tahap 4**: Siklus pelatihan profesional, isolasi validasi, gradient clipping, dan early stopping. | `torch.optim` |
| 5 | `evaluation_diagnostics.py` | **Tahap 5 & 6**: Evaluasi akhir data uji (Test Set), matriks konfusi, dan diseksi galat (*error analysis*). | `sklearn.metrics` |
| 6 | `checkpoint_manager.py` | **Tahap 7**: Penyimpanan dan pemulihan checkpoint aman (`.pt`), verifikasi reproduksibilitas. | `torch`, `pathlib` |
| 7 | `responsible_ai_audit.py` | **Tahap 6 & 7**: Audit etika keadilan subgrup mesin (*subgroup parity*) dan uji ketahanan derau sensor. | `numpy`, `torch` |
| 8 | `capstone_pipeline_runner.py` | **Tahap 1–8**: Orkestrator terpadu yang menjalankan seluruh pipeline ujung-ke-ujung dan mencatat log JSON. | `json`, `time` |

---

### Petunjuk Eksekusi Cepat

Jalankan orkestrator pipeline utama untuk mengeksekusi seluruh tahapan proyek capstone secara otomatis:

```bash
# Menjalankan seluruh alur proyek capstone dari data loading hingga pelaporan JSON
python content/level-3-deep-learning/chapter-30/code/capstone_pipeline_runner.py
```

Seluruh proses inferensi, pelatihan konvergen, evaluasi diagnostik, dan audit keadilan selesai dalam waktu **< 5 detik** pada CPU tanpa memerlukan akselerator GPU.

