# Smart Sensor Data Quality Monitor — Reference Implementation

Direktori ini berisi implementasi referensi resmi untuk **Chapter 10: AI Foundations Capstone**. Sistem ini dirancang untuk memantau, memvalidasi, menganalisis, dan mendiagnosis kualitas data telemetri multi-sensor secara deterministik dan transparan menggunakan fondasi logika simbolik, statistik deskriptif, dan rekayasa data.

---

## 1. Inventaris Berkas Proyek

| Nama Berkas | Peran Arsitektur Perangkat Lunak | Bab Rujukan Terintegrasi |
| :--- | :--- | :--- |
| `dataset_generator.py` | Generator dataset telemetri multi-sensor sintetis deterministik (`seed=42`). | Bab 07 & 08 |
| `validator.py` | Validasi skema DataFrame dan batas operasional fisik (*physical bounds*). | Bab 03 & 08 |
| `statistics_analyzer.py` | Komputasi mean, median, IQR, dan Pagar Tukey (*Tukey's Fences*) per sensor. | Bab 06 |
| `rule_engine.py` | Mesin inferensi First-Order Logic & kalkulasi Data Quality Index (DQI). | Bab 05 |
| `visualization.py` | Visualisasi dashboard 3-panel Matplotlib nir-layar (*headless mode*). | Bab 07 & 08 |
| `main.py` | Pipeline runner utama terintegrasi dari data mentah hingga rekomendasi. | Bab 01–09 (Capstone) |
| `test_sensor_monitor.py` | Rangkaian pengujian unit otomatis (*automated unit tests*, 6 skenario). | Praktik Rekayasa |
| `data/sensor_telemetry_raw.csv` | Dataset 120 baris telemetri 3 instrumen sensor (`TEMP-01`, `HUMID-01`, `VIB-01`). | Bab 08 |
| `data/sensor_quality_dashboard.png` | Citra visualisasi dashboard mutu telemetri hasil eksekusi Matplotlib. | Bab 07 |

---

## 2. Persyaratan & Lingkungan Eksekusi

Sistem diuji dan diverifikasi menggunakan lingkungan Python virtual standar:
- **Python Version**: `Python 3.12.10`
- **Pustaka Utama**:
  - `pandas >= 2.0.0` (manipulasi tabel telemetri)
  - `numpy >= 1.24.0` (komputasi matriks kuartil dan random generator)
  - `matplotlib >= 3.7.0` (visualisasi visual headless backend Agg)

Aktivasi virtual environment (Windows PowerShell):
```powershell
.\.venv\Scripts\Activate.ps1
```

---

## 3. Cara Menjalankan Modul

### A. Eksekusi Pipeline Utama
Jalankan alur pemantauan dari awal hingga akhir:
```bash
python content/level-1-ai-foundations/chapter-10/code/main.py
```

### B. Menjalankan Pengujian Unit Otomatis
Jalankan pengujian unit mandiri:
```bash
python content/level-1-ai-foundations/chapter-10/code/test_sensor_monitor.py
```

### C. Pembangkitan Ulang Dataset Sintetis
Jalankan generator data:
```bash
python content/level-1-ai-foundations/chapter-10/code/dataset_generator.py
```

---

## 4. Tabel Hasil Pengujian Empiris Aktual

Tabel berikut menyajikan hasil verifikasi numerik aktual dari eksekusi pipeline utama:

| Komponen Evaluasi | Parameter / Metrik | Hasil Verifikasi Aktual | Status Validasi |
| :--- | :--- | :--- | :---: |
| **Ingesti Data** | Total Rekaman Dimuat | 120 baris (3 jenis instrumen) | **LULUS** |
| | Sensor Teridentifikasi | `TEMP-01`, `HUMID-01`, `VIB-01` | **VALID** |
| **Validasi Fisik** | Rekaman Berstatus VALID | 115 baris | **VALID** |
| | Deteksi Missing Values | 3 baris kosong (`NaN`) | **TERDETEKSI** |
| | Pelanggaran Batas Fisik | 2 baris (Suhu $-300.0^\circ\text{C}$ & $145.0^\circ\text{C}$) | **TERISOLASI** |
| **Statistik Tukey** | Sebaran Sensor Suhu | Mean: 28.68, Median: 28.05, IQR: 1.29 | **AKURAT** |
| | Pagar Tukey Suhu | $[25.31^\circ\text{C}, 30.49^\circ\text{C}]$ | **TERVERIFIKASI** |
| | Outlier Lonjakan Statistik | 2 titik terdeteksi | **TERDETEKSI** |
| **Rule Engine** | Total Alert Peringatan | 7 peringatan kausal transparan | **TERVERIFIKASI** |
| **Data Quality Index** | Skor Mutu VIB-01 | **100.00 / 100** (EXCELLENT) | **TERUKUR** |
| | Skor Mutu HUMID-01 | **97.50 / 100** (EXCELLENT) | **TERUKUR** |
| | Skor Mutu TEMP-01 | **85.00 / 100** (EXCELLENT) | **TERUKUR** |
| **Visualisasi** | Ekspor Dashboard Headless | `sensor_quality_dashboard.png` (3 panel) | **TEREKSPOR** |
| **Automated Tests** | Hasil Uji `unittest` | 6/6 pengujian unit lulus (0.029s) | **100% PASS** |

---

## 5. Batasan Sistem & Catatan Etika

1. **Bukan Industrial Grade**: Sistem ini merupakan model referensi akademik untuk pembuktian konsep, bukan perangkat pengendali keselamatan pabrik tingkat produksi.
2. **Outlier Bukan Kerusakan**: Lonjakan statistik di luar pagar Tukey (RULE-03) dapat merepresentasikan lonjakan beban kerja mesin yang valid, bukan kerusakan sensor.
3. **Pengawasan Manusia**: Peringatan sistem bersifat alat bantu penasihat (*advisory tool*). Keputusan perbaikan atau pemutusan daya sensor fisik tetap berada di bawah wewenang teknisi manusia.
