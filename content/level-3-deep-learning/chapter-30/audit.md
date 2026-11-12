# Laporan Audit Kualitas: Chapter 30 — Deep Learning Capstone Project

**Tanggal Audit**: 8 November 2026  
**Status**: **PASSED (100% MEMENUHI STANDAR KURIKULUM RESMI)**  
**Auditor**: Tim Penjamin Mutu Kurikulum AI Learning Hub  
**Cakupan**: Bab 30 — Deep Learning Capstone (Puncak Level 3: Deep Learning & Generative AI)

---

## 1. Verifikasi Keselarasan Master Curriculum

| Butir Evaluasi | Standar Kurikulum Resmi | Status Aktual Repositori | Keterangan |
|---|---|:---:|---|
| Nomor Bab | Chapter 30 | 30 | Sesuai |
| Judul Resmi | Deep Learning Capstone | Deep Learning Capstone | Sesuai |
| Level Pembelajaran | Level 3: Deep Learning & Generative AI | Level 3: Deep Learning & Generative AI | Puncak Integrasi Level 3 |
| Prasyarat | Chapter 29 | `[29]` | Sesuai |
| Status Publikasi | `published` | `published` | Sesuai |
| Batasan Komputasi | Ekosistem CPU terverifikasi | Scikit-learn 1.9.1, PyTorch 2.15.0+cpu, Pandas 2.2.3, NumPy 2.5.4 | Memenuhi standar (< 6 detik pada CPU) |

---

## 2. Verifikasi Artefak Pembelajaran

| No | Artefak | Lokasi Berkas | Status Verifikasi |
|---|---|---|:---:|
| 1 | Modul Teori (SSOT) | `chapter-30/index.md` | Lengkap (12 Bagian Utama, Diagram Alir ASCII, Tabel Parameter, Diseksi Galat, 6 Pilihan Track, Rubrik 100 Poin, 5 Referensi) |
| 2 | Slide Reveal.js | `chapter-30/slides/deck.json` | 22 Slide valid, format direct array JSON, KaTeX, semantic kinds, speaker notes |
| 3 | Panduan Praktik Lab | `chapter-30/practice.md` | Lengkap (8 Tahapan Siklus Capstone, 6 Pilihan Topik, 6 Milestone, Rubrik 100 Poin) |
| 4 | Skrip Python Lengkap | `chapter-30/code/*.py` (8 skrip) | 8/8 skrip dieksekusi dengan exit code 0 |
| 5 | Notebook Pendamping | `chapter-30/notebooks/chapter_30_companion.ipynb` | Valid Jupyter Notebook v4 (Pre-executed) |
| 6 | Katalog Kode | `chapter-30/code/README.md` | Panduan eksekusi dan tabel 8 skrip terverifikasi |
| 7 | Log Eksperimen | `chapter-30/experiments/results_ch30.json` | Log JSON profil eksekusi CPU lengkap |
| 8 | Artefak Checkpoint | `chapter-30/experiments/*.pt` | Model PyTorch tersimpan dan teruji paritas bitwise |
| 9 | Metadata Bab | `chapter-30/metadata.json` | 12 Bloom Objectives diawali kata kerja valid, 5 Referensi terverifikasi |

---

## 3. Verifikasi Numerik & Hasil Komputasi Aktual

Seluruh 8 skrip Python proyek acuan (*Reference Implementation: Deep Predictive Maintenance Net*) telah dieksekusi dan diverifikasi secara langsung menggunakan Python 3.12.10, PyTorch 2.15.0+cpu, Scikit-learn 1.9.1, Pandas 2.2.3, dan NumPy 2.5.4:

```
[Tahap 1 & 2: dataset_loader.py]
- Total Sampel Telemetri  : 1000 sampel, 6 fitur sensor fisik (suhu, getaran, tekanan, dsb.)
- Partisi Bertingkat 3-Way: Train (700 sampel, 70%), Val (150 sampel, 15%), Test (150 sampel, 15%)
- Uji Independensi Indeks : Disjoint index check terbukti 100% unik (Zero Data Leakage)
- Skalasi Fitur           : Z-score dihitung hanya dari training set (mean dan std terisolasi)

[Tahap 3: baseline_model.py]
- Model Acuan             : Scikit-learn LogisticRegression terstandarisasi
- Test Accuracy Baseline  : 80.67%
- Test Macro F1 Baseline  : 0.6887
- Waktu Komputasi CPU     : < 0.05 detik

[Tahap 4: deep_learning_model.py]
- Arsitektur Jaringan     : DeepPredictiveMaintenanceNet (PyTorch nn.Module)
- Komponen Modular        : Linear Input -> LayerNorm -> 2x Tabular Residual Blocks -> Classifier
- Total Parameter Terlatih: 17.795 parameter (memori < 100 KB, tanpa ketergantungan GPU)

[Tahap 5: training_engine.py]
- Optimizer & Regularisasi: AdamW (lr=0.001, weight_decay=1e-4), Gradient Clipping (max_norm=1.0)
- Mekanisme Penghentian   : Early Stopping terpantau pada validasi loss (patience = 7)
- Hasil Konvergensi       : Epoch terbaik tercapai (loss validasi 0.3731) dalam 19 epoch

[Tahap 6: evaluation_diagnostics.py]
- Test Accuracy Model DL  : 82.00% (Unggul +1.33% di atas baseline linier)
- Test Macro F1 Model DL  : 0.6894 (Unggul +0.0007 di atas baseline)
- Recall Deteksi Kritis   : 81.25% (Peningkatan +3.13% vs baseline 78.12%)
- Diseksi Galat Kualitatif: Teridentifikasi ambiguitas frekuensi sensor pada status Warning

[Tahap 7: checkpoint_manager.py]
- Format Serialisasi      : PyTorch state_dict (.pt) beserta metadata scaler
- Verifikasi Bitwise      : Selisih absolut maksimum logit model reloaded vs active = 0.0000000 (< 1e-7)
- Reproduksibilitas       : Bitwise Parity 100% terkonfirmasi

[Tahap 8: responsible_ai_audit.py]
- Audit Keadilan Armada   : Disparitas performa antarmesin UNIT-01 s/d UNIT-10 tercatat
- Disparitas F1 Terukur   : Max(F1) - Min(F1) = 0.2325 (Dalam ambang toleransi <= 0.30)
- Uji Ketahanan Derau     : Akurasi bertahan > 81% pada injeksi derau Gauss hingga sigma = 0.10
```

---

## 4. Evaluasi Kepatuhan Pedagogis & Standar Pengajaran

1. **Jalur Komputasi CPU Mandiri**:
   - Seluruh pipeline capstone dari pemuatan data, pelatihan model deep learning, evaluasi, hingga audit berjalan tuntas dalam waktu **5.66 detik** pada prosesor CPU komputer standar tanpa memerlukan CUDA/GPU.
2. **Keterbukaan 6 Lintasan Spesialisasi**:
   - Mahasiswa diberikan fleksibilitas untuk memilih 1 dari 6 lintasan domain (Computer Vision, Object Detection, Sensor Telemetry, NLP Intent, Generative Mini-Transformer, atau Industrial Video Anomaly) dengan panduan arsitektur yang komprehensif.
3. **Pencegahan Kebocoran Data (Zero Data Leakage)**:
   - Partisi 3-way diuji dengan isolasi indeks ketat. Standardisasi z-score hanya menggunakan statistik himpunan latih.
4. **Pilar Etika & Rekayasa Bertanggung Jawab**:
   - Audit keadilan armada mesin industri dan uji ketahanan derau sensor mengajarkan mahasiswa bahwa akurasi agregat bukanlah satu-satunya tolak ukur kualitas sistem kecerdasan buatan.

---

## 5. Kesimpulan Penjamin Mutu

Bab 30 (**Deep Learning Capstone**) telah memenuhi seluruh standar kelayakan akademik dan teknis AI Learning Hub. Dengan selesainya Bab 30:
- Seluruh 10 bab pada **Level 3: Deep Learning & Generative AI** (Bab 21 hingga Bab 30) kini resmi berstatus **`published`**.
- Seluruh materi Bab 31 hingga Bab 40 (**Level 4: Advanced AI & Intelligent Systems**) tetap berstatus **`coming-soon`** sesuai batasan ketat Master Curriculum.

