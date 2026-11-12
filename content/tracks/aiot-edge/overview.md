# Track C — AIoT & Edge AI: Overview, Roadmap & Studi Kasus

**Fokus Jalur:** Analitik data sensor, telemetri Internet of Things, pemrosesan inferensi efisien on-device (*Edge AI*), kompresi model, serta pemantauan drift di CPU lokal.

---

## 1. Pengantar Jalur Spesialisasi
Jalur **AIoT & Edge AI** membekali mahasiswa kemampuan membangun model prediksi ringan dari data sensor fisik dan mengompresinya agar sesuai anggaran sumber daya terbatas. Bila mikrokontroler fisik tidak tersedia, praktikum menggunakan **simulasi di CPU lokal** dengan batasan jujur tanpa mengklaim uji silikon fisik (misalnya ESP32 atau Cortex-M) sebelum pengukuran nyata di perangkat.

## 2. Roadmap Pembelajaran Terkurasi
1. **Tahap Fondasi:** **Chapter 06** (Matematika), **Chapter 07** (Ekosistem Python), **Chapter 08** (Data Preparation).
2. **Tahap Inti Analitik Sensor (ML):** **Chapter 11–14** (ML Workflow, Regresi, Klasifikasi), **Chapter 17–18** (Clustering Anomali & Reduksi Dimensi PCA).
3. **Tahap Lanjutan Edge & Sistem:** **Chapter 23** (PyTorch), **Chapter 31** (Transfer Learning), **Chapter 36** (MLOps & Drift), **Chapter 37** (Edge AI & Kompresi), **Chapter 38** (Fusi Multimodal), **Chapter 39** (AI Safety & Tata Kelola).

## 3. Studi Kasus Domain AIoT & Edge AI

### Studi Kasus 1: Pemantauan Getaran & Suhu Motor Listrik On-Device
- **Tantangan:** Mengirim seluruh gelombang mentah akselerometer ke cloud memboroskan bandwidth dan rentan putus koneksi.
- **Solusi:** Rekayasa statistik jendela lokal (`Chapter 12`) $\to$ klasifikasi MLP terkuantisasi INT8 (`aiot_sensor_edge_lab.py`, `13.27 KB -> 7.50 KB`) $\to$ pemantauan PSI pergeseran kalibrasi (`Chapter 36`).

### Studi Kasus 2: Pemantauan Kualitas Udara / Mikroklimat Smart Campus
- **Tantangan:** Sensor suhu/kelembaban sering menghasilkan nilai hilang (*NaN*) maupun lonjakan sesaat.
- **Solusi:** Penjagaan batas fisik (`Chapter 39`), imputasi zona terkelompok (`Chapter 08`), dan klasterisasi pola anomali (`Chapter 17`).

