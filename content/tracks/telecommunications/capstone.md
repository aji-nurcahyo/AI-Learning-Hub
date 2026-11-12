# Track D Capstone Specification: Intelligent Wireless Network Quality Monitoring

**Jalur:** Track D — AI for Telecommunications  
**Kebijakan Teknis Keras:** Gunakan definisi metrik nirkabel secara tepat (`RSRP`, `RSRQ`, `SINR`, `SNR`, `Latency`, `Throughput`, `Packet Loss`). **Dilarang** menyamakan `SNR` dan `SINR`, **dilarang** mengarang kolom pengukuran yang tidak ada di dataset, dan **dilarang** mengklaim kesiapan lapangan (*field deployment*) tanpa bukti *field validation*.

---

## 1. Deliverables Wajib
1. **Network Dataset & Feature Definitions:** Dataset sintetis atau publik berlisensi terbuka lengkap dengan kamus satuan setiap fitur radio.
2. **Baseline & Leakage-Free Pipeline:** Pemecahan *Train/Val/Test* yang tepat (`TimeSeriesSplit` atau berstrata) serta baseline sederhana pembanding.
3. **Prediction / Classification Model:** Model terlatih untuk memprediksi kualitas sel atau *throughput/latency* tautan.
4. **Evaluation:** Metrik komparatif (`Accuracy`, `Macro-F1`, `ROC-AUC`, atau `MAE/RMSE`).
5. **Explainability:** Analisis kontribusi fitur (`Permutation Importance`) guna membuktikan keputusan model konsisten dengan fisika radio (`SINR`, `Packet Loss`).
6. **Monitoring & Technical Interpretation:** Simulasi pemantauan pergeseran distribusi metrik jaringan + interpretasi teknis telekomunikasi.
7. **Responsible AI & Caveats:** Pernyataan tegas bahwa model tidak melakukan kendali otonom perangkat jaringan tanpa otorisasi operator.

## 2. Rubrik Penilaian Track D (Total 100 Poin)

| Komponen Penilaian | Bobot | Kriteria Keunggulan |
| :--- | :---: | :--- |
| **Network Dataset & Metric Rigor (SINR vs SNR)** | **20 poin** | Definisi fitur radio akurat, tiada kolom rekaan, serta pipeline bebas leakage. |
| **Baseline & Model Implementation** | **25 poin** | Perbandingan adil terhadap baseline dengan kode modular terverifikasi CPU. |
| **Evaluation & Technical Interpretation** | **25 poin** | Explainability selaras dengan konsep telekomunikasi dan error analysis terperinci. |
| **Monitoring & Responsible Operations** | **20 poin** | Simulasi deteksi drift + pengendalian keamanan & batas non-field claim jujur. |
| **Documentation & Reproducibility** | **10 poin** | Laporan ilmiah rapi dan mudah direproduksi dengan seed tetap. |

