# Panduan Praktik Laboratorium — Chapter 10: AI Foundations Capstone

Selamat datang di laboratorium **Chapter 10: AI Foundations Capstone**. Modul praktikum ini membimbing Anda mengeksplorasi, menjalankan, menguji, dan memverifikasi *Reference Implementation* resmi: **Smart Sensor Data Quality Monitor**.

Sistem ini merepresentasikan integrasi utuh dari konsep-konsep Level 1:
- Perumusan Agen Cerdas (Ch 01 & 03),
- Inferensi Berbasis Aturan (Ch 05),
- Statistik Deskriptif & Deteksi Outlier Tukey IQR (Ch 06),
- Rekayasa Data Telemetri dengan Pandas & NumPy (Ch 07 & 08),
- Visualisasi Analitik Headless Matplotlib (Ch 07 & 08),
- Serta Prinsip Transparansi dan Pengawasan Manusia (*Human-in-the-Loop*) (Ch 09).

---

## 1. Tujuan Praktikum

Setelah menyelesaikan seluruh langkah praktikum mandiri ini, Anda diharapkan mampu:
1. Menjalankan pipeline pembangkitan dataset telemetri multi-sensor sintetis dengan anomali terkontrol secara deterministik (*reproducible*).
2. Memvalidasi skema data dan memisahkan pelanggaran batas fisik (*physical bounds*) dari operasi sensor normal.
3. Menghitung ringkasan statistik deskriptif dan pagar Tukey (*Tukey's Fences*) untuk mendeteksi *outlier* statistik.
4. Menerapkan mesin inferensi aturan (*Rule Engine*) untuk mengklasifikasikan anomali data ke dalam kategori *Missing*, *Physical Impossibility*, dan *Statistical Spike*.
5. Menghitung skor *Data Quality Index* (DQI) komprehensif untuk setiap instrumen sensor.
6. Merender dashboard visualisasi tiga panel secara *headless* tanpa ketergantungan layar monitor grafis.
7. Menjalankan rangkaian pengujian unit otomatis (*automated unit tests*) menggunakan pustaka `unittest` Python.

---

## 2. Struktur Direktori Praktikum

Seluruh kode sumber, dataset pengujian, dan berkas keluaran tersimpan dalam struktur modular berikut:

```text
content/level-1-ai-foundations/chapter-10/code/
├── dataset_generator.py      # Pembangkit dataset telemetri multi-sensor sintetis deterministik
├── validator.py              # Validasi skema dan batas operasional fisika pengukuran
├── statistics_analyzer.py    # Perhitungan mean, median, IQR, dan pagar Tukey per sensor
├── rule_engine.py            # Mesin inferensi First-Order Logic & kalkulasi Data Quality Index (DQI)
├── visualization.py          # Generator dashboard 3-panel Matplotlib nir-layar (headless mode)
├── main.py                   # Pipeline runner utama dari data mentah hingga rekomendasi
├── test_sensor_monitor.py    # Pengujian unit otomatis (6 skenario uji fungsional)
├── README.md                 # Katalog berkas, instruksi eksekusi, dan tabel verifikasi hasil
└── data/
    ├── sensor_telemetry_raw.csv      # Dataset 120 rekaman telemetri 3 sensor
    └── sensor_quality_dashboard.png  # Citra dashboard visualisasi mutu sensor
```

---

## 3. Prasyarat Lingkungan & Persiapan

Praktikum ini menggunakan lingkungan virtual Python standar (`.venv`) dengan dependensi:
- `python >= 3.10`
- `pandas >= 2.0.0`
- `numpy >= 1.24.0`
- `matplotlib >= 3.7.0`

Aktivasi virtual environment:
```bash
# Windows PowerShell
.\.venv\Scripts\Activate.ps1

# Windows CMD
.\.venv\Scripts\activate.bat

# macOS / Linux
source .venv/bin/activate
```

Verifikasi dependensi:
```bash
python -c "import pandas, numpy, matplotlib; print('Lingkungan Capstone AI Foundations siap!')"
```

---

## 4. Panduan Eksekusi Langkah demi Langkah

### Langkah 1: Pembangkitan Data Telemetri Sintetis
Jalankan modul pembangkit data:
```bash
python content/level-1-ai-foundations/chapter-10/code/dataset_generator.py
```
**Analisis yang Diperhatikan:**
- Program menghasilkan berkas `data/sensor_telemetry_raw.csv` berisi 120 baris data dari 3 sensor (`TEMP-01`, `HUMID-01`, `VIB-01`).
- Perhatikan bahwa *random seed* diatur ke 42 sehingga setiap eksekusi menghasilkan data yang identik secara deterministik.
- Amati keberadaan nilai kosong (`NaN`) terkontrol pada rekaman ke-12, 45, dan 88.

### Langkah 2: Eksekusi Pipeline Utama Lengkap
Jalankan alur pemantauan terintegrasi:
```bash
python content/level-1-ai-foundations/chapter-10/code/main.py
```
**Analisis Output Terminal:**
1. **Fase 1 (Ingesti)**: Memuat 120 rekaman telemetri.
2. **Fase 2 (Validasi Batas Fisik)**:
   - Mendeteksi 3 rekaman missing values (`MISSING`).
   - Mendeteksi 2 pelanggaran batas fisik (`INVALID_BOUNDS`):
     - Suhu $-300.0^\circ\text{C}$ (di bawah nol mutlak).
     - Suhu $145.0^\circ\text{C}$ (di luar batas operasional sensor).
3. **Fase 3 (Statistik & Pagar Tukey)**:
   - Menghitung mean, median, standar deviasi, dan IQR hanya dari sampel berstatus `VALID`.
   - Mengidentifikasi 2 titik *statistical outlier* pada sensor suhu.
4. **Fase 4 (Rule Engine & DQI)**:
   - Memicu 7 peringatan kualitas (*alerts*) dengan penjelasan kausal transparan.
   - Menghitung skor DQI:
     - `VIB-01`: **100.00 / 100** (EXCELLENT)
     - `HUMID-01`: **97.50 / 100** (EXCELLENT)
     - `TEMP-01`: **85.00 / 100** (EXCELLENT)
5. **Fase 5 (Visualisasi)**:
   - Mengekspor `data/sensor_quality_dashboard.png` tanpa menampilkan jendela GUI.

### Langkah 3: Menjalankan Pengujian Unit Otomatis (Automated Unit Tests)
Uji keandalan dan ketahanan sistem terhadap kasus ekstrem (*edge cases*):
```bash
python content/level-1-ai-foundations/chapter-10/code/test_sensor_monitor.py
```
**Keluaran yang Diharapkan:**
```text
......
----------------------------------------------------------------------
Ran 6 tests in 0.029s

OK
```
Enam skenario uji yang divalidasi mencakup:
1. `test_dataset_generation_shape`: Menguji dimensi dan kelengkapan kolom data.
2. `test_schema_validation`: Menguji penolakan skema saat kolom wajib hilang.
3. `test_physical_bounds_detection`: Menguji deteksi nilai di bawah nol mutlak.
4. `test_statistics_and_tukey_fences`: Menguji kebenaran perhitungan kuartil dan IQR.
5. `test_rule_engine_and_dqi`: Menguji pemicuan aturan logika dan rentang skor DQI $[0, 100]$.
6. `test_edge_case_unknown_sensor`: Menguji respons sistem saat menerima ID sensor asing.

---

## 5. Batasan Sistem & Pengawasan Manusia (Human-in-the-Loop)

Sebagai calon insinyur sistem cerdas yang bertanggung jawab, pahami batasan operasional berikut:
1. **Bukan Sistem Kelas Industri Produksi**: Prototype ini ditujukan untuk evaluasi akademik dan pembuktian konsep, bukan pengendali keselamatan turbin nuklir atau pabrik kimia skala besar.
2. **Outlier Bukan Berarti Rusak**: Nilai di luar pagar Tukey (RULE-03) bisa mencerminkan lonjakan beban operasional mesin yang nyata, bukan kerusakan instrumen.
3. **Sifat Rekomendasi (*Advisory Tool*)**: Peringatan sistem bersifat pendukung keputusan teknisi (*Human-on-the-Loop*). Keputusan mematikan mesin atau mengganti sensor fisik tetap berada di bawah wewenang insinyur manusia.

---

## 6. Tugas Mandiri & Eksplorasi Lanjutan

Untuk memperdalam pemahaman praktikum:
1. **Modifikasi Ambang Batas Fisika**: Ubah konfigurasi `PHYSICAL_BOUNDS` pada `validator.py` agar batas atas vibrasi mesin diturunkan menjadi $10.0\text{ mm/s}$. Amati bagaimana lonjakan vibrasi pada indeks 51 beralih status dari *statistical spike* menjadi *critical physical impossibility*.
2. **Penyisipan Aturan Sensor Membeku (*Frozen Sensor Rule*)**: Tambahkan aturan `RULE-04-FROZEN-SENSOR` pada `rule_engine.py` yang mendeteksi jika suatu sensor menghasilkan nilai konstan yang persis sama selama 5 interval berturut-turut dengan varians nol.
3. **Simulasi Sensor Asing**: Buat skrip kecil yang menyuntikkan data sensor `PRESSURE-01` yang belum terdaftar. Verifikasi apakah sistem memberikan label `UNKNOWN_SENSOR` dengan aman tanpa mengalami *unhandled exception*.
