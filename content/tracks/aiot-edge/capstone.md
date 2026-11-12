# Track C Capstone Specification: Intelligent Sensor Monitoring with Edge AI

**Jalur:** Track C — AIoT & Edge AI  
**Kebijakan Simulasi:** Seluruh tahap dieksekusi di CPU lokal menggunakan dataset sensor sintetis/berlisensi. **Dilarang** mengklaim validasi nyata pada mikrokontroler ESP32 atau ARM Cortex-M fisik kecuali Anda benar-benar mem-flash dan mengukurnya pada hardware fisik tersebut.

---

## 1. Deliverables Wajib
1. **Sensor Dataset & Dictionary:** Penjelasan kanal sensor, satuan, dan batas operasi wajar.
2. **Zero-Leakage Data Pipeline:** Pemecahan *Train/Val/Test* bebas kebocoran data.
3. **ML / DL Baseline & Model:** Perbandingan model linier vs MLP/ansambel ringkas.
4. **Model Compression:** Penerapan *Dynamic/Static Quantization*, *Structured Pruning*, atau *Knowledge Distillation*.
5. **Inference Pipeline:** Penjagaan payload `NaN`/di luar rentang + mekanisme *Human Escalation*.
6. **Latency & Memory Budget Assessment:** Ukuran `.pt`/`.pkl` di disk vs anggaran SRAM/Flash target simulasi serta latensi P50/P95 CPU.
7. **Drift Monitoring:** Uji simulasi *PSI/KS* terhadap pergeseran pembacaan sensor.
8. **Risk Register & Model Card:** Identifikasi risiko keselamatan perangkat dan batasan simulasi CPU.

## 2. Rubrik Penilaian Track C (Total 100 Poin)

| Komponen Penilaian | Bobot | Kriteria Keunggulan |
| :--- | :---: | :--- |
| **Sensor Data Pipeline & Leak Prevention** | **15 poin** | Data dictionary lengkap, penanganan missing/outlier aman, nol kebocoran data. |
| **Model Training & Compression Implementation** | **30 poin** | Kompresi berjalan benar pada CPU beserta perbandingan trade-off ukuran vs akurasi. |
| **Inference Guardrails, Latency & Memory Audit** | **25 poin** | Benchmark berulang sesudah warm-up + analisis jujur anggaran SRAM/Flash. |
| **Drift Monitoring & Risk Assessment** | **20 poin** | Uji PSI/KS drift berhasil dan Model Card memuat batas simulasi vs hardware nyata. |
| **Documentation & Reproducibility** | **10 poin** | Skrip terstruktur dengan seed tetap mudah direproduksi ulang. |

