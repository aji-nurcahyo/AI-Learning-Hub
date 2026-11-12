# Katalog Kode Praktik Chapter 03: Intelligent Agents & AI Problem Solving

Repositori ini memuat kumpulan contoh implementasi Python runnable mandiri untuk konsep-konsep inti Chapter 03. Seluruh skrip dirancang menggunakan pustaka standar Python (*standard library* tanpa dependensi eksternal berat) dan telah diverifikasi berjalan mulus pada Python 3.12+.

---

## Daftar Skrip Praktik

| File Script | Konsep AI yang Diimplementasikan | Karakteristik Utama |
|---|---|---|
| `01_simple_reflex_temperature.py` | Simple Reflex Agent | Pemetaan kondisi-aksi langsung (*condition-action rules*) tanpa memori internal. |
| `02_model_based_room_agent.py` | Model-Based Reflex Agent | Memelihara *internal state* (pelacakan tren suhu dan proteksi siklus kompresor). |
| `03_peas_specification_model.py` | PEAS Framework & Evaluator | Pemodelan berorientasi objek untuk PEAS dan evaluasi kuantitatif rasionalitas agen. |
| `04_grid_state_space.py` | State Space Representation | Ruang keadaan grid 2D diskret (states, legal actions, deterministic transition model). |
| `05_problem_formulation_search_prep.py` | AI Problem Formulation | Formulasi 5 elemen masalah formal Russell & Norvig sebagai jembatan menuju Chapter 04. |

---

## Petunjuk Menjalankan

Pastikan Anda berada di direktori proyek dan virtual environment aktif:

```bash
# Masuk ke direktori kode chapter 03
cd "content/level-1-ai-foundations/chapter-03/code"

# Jalankan skrip satu per satu
python 01_simple_reflex_temperature.py
python 02_model_based_room_agent.py
python 03_peas_specification_model.py
python 04_grid_state_space.py
python 05_problem_formulation_search_prep.py
```

---

## Log Output Eksekusi Aktual (Terverifikasi Python 3.12.10)

### 1. Output `01_simple_reflex_temperature.py`
```text
=== SIMULASI SIMPLE REFLEX TEMPERATURE AGENT ===
Target Suhu: 24.0°C (Toleransi: ±1.0°C)

Step   | Suhu (°C)  | Penghuni   | Tindakan Agen      | Konteks Masalah
---------------------------------------------------------------------------
1      | 28.5       | Ada        | COOLING_MAX        | Ruangan panas dan ada orang
2      | 24.2       | Ada        | FAN_IDLE           | Suhu nyaman dan ada orang
3      | 21.0       | Ada        | HEATING_MODERATE   | Suhu terlalu dingin dan ada orang
4      | 28.5       | Kosong     | ECO_STANDBY        | Ruangan panas tetapi kosong
5      | 24.0       | Kosong     | ECO_STANDBY        | Ruangan ideal dan kosong
```

### 2. Output `02_model_based_room_agent.py`
```text
=== SIMULASI MODEL-BASED ROOM AGENT ===
Fitur Internal State: Pelacakan Tren Suhu & Proteksi Siklus Kompresor

Step  | Suhu   | Penghuni | Keputusan Agen                      | Skenario Dinamis
----------------------------------------------------------------------------------------
1     | 28.0   | Ada      | COOLING_HIGH [Trend: STABLE]        | Mulai: Siang hari terik, ruangan penuh
2     | 26.5   | Ada      | COOLING_HIGH [Trend: FALLING]       | Suhu mulai turun bertahap
3     | 24.8   | Ada      | COOLING_LOW [Trend: FALLING]        | Suhu mendekati target ideal
4     | 24.0   | Ada      | COMFORT_MAINTAIN [Trend: FALLING]   | Suhu tepat di zona target
5     | 22.8   | Ada      | HEATING_MILD [Trend: FALLING]       | Fluktuasi suhu tiba-tiba turun sedikit
6     | 22.8   | Kosong   | ECO_STANDBY [Trend: STABLE]         | Penghuni meninggalkan ruangan
7     | 25.0   | Kosong   | ECO_STANDBY [Trend: RISING]         | Ruangan tetap kosong
```

### 3. Output `03_peas_specification_model.py`
```text
=== SPESIFIKASI PEAS: SMART HVAC CLIMATE AGENT ===

[P] Performance Measure (Kriteria Kinerja Objektif):
  - Meminimalkan deviasi suhu ruang dari zona kenyamanan (23°C - 25°C) saat berpenghuni.
  - Meminimalkan konsumsi energi listrik total (kWh).
  - Mencegah kerusakan aktuator dengan membatasi frekuensi siklus kompresor.
  - Waktu respon cepat (< 10 menit) mencapai suhu nyaman saat ruangan mulai terisi.

[E] Environment (Lingkungan Tugas):
  Gedung perkantoran multi-lantai dengan cuaca tropis dinamis dan fluktuasi penghuni.

[A] Actuators (Perangkat Pelaksana Aksi):
  - Katup kompresor pendingin multi-kecepatan (Inverter Compressor)
  - Motor penggerak sirip ventilasi (Air Flow Louver)
  - Relai elemen pemanas tambahan (Auxiliary Heater)

[S] Sensors (Perangkat Penerima Persepsi):
  - Sensor termokopel digital suhu udara ruangan (°C)
  - Sensor kelembapan relatif (% RH)
  - Sensor inframerah pasif kehadiran manusia (PIR Occupancy Sensor)
  - Jam waktu nyata (Real-Time Clock)

[T] Taksonomi Sifat Lingkungan:
  * Observabilitas        : Partially Observable (suhu dinding & cuaca luar tak terukur langsung)
  * Determinisasi         : Stochastic (suhu dipengaruhi bukaan pintu/jendela acak)
  * Sifat Episode         : Sequential (keputusan pendinginan saat ini memengaruhi suhu menit berikutnya)
  * Kedinamisan           : Dynamic (suhu berubah sendiri seiring waktu meskipun AC diam)
  * Diskrit/Kontinu       : Continuous (suhu dan waktu bernilai kontinu)
  * Jumlah Agen           : Single-Agent (fokus pada kendali ruangan lokal)
=======================================================

=== HASIL EVALUASI KINERJA RASIONALITAS ===
Total Siklus Simulasi     : 5 step
Total Konsumsi Energi     : 5.75 kWh
Penalti Ketidaknyamanan   : 15.00 poin
Skor Kinerja Rasional (P) : 43.5 / 100.00
```

### 4. Output `04_grid_state_space.py`
```text
=== REPRESENSTASI FORMAL RUANG KEADAAN GRID ===
Total State Legal yang Tersedia: 8 states
Himpunan State: [(0, 0), (0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1), (2, 2)]

Peta Grid (3x3):
  [ A ] [ . ] [ . ]
  [ . ] [ X ] [ . ]
  [ . ] [ . ] [ G ]
  Legenda: [A]=Agen, [G]=Tujuan, [X]=Rintangan, [.]=Lantai Kosong

Mulai eksplorasi dari Initial State: (0, 0)
Step 1: Di (0, 0) -> Aksi 'DOWN' (Legal: ['DOWN', 'RIGHT']) -> Pindah ke (1, 0)
Step 2: Di (1, 0) -> Aksi 'DOWN' (Legal: ['UP', 'DOWN']) -> Pindah ke (2, 0)
Step 3: Di (2, 0) -> Aksi 'RIGHT' (Legal: ['UP', 'RIGHT']) -> Pindah ke (2, 1)
Step 4: Di (2, 1) -> Aksi 'RIGHT' (Legal: ['LEFT', 'RIGHT']) -> Pindah ke (2, 2)

Posisi Akhir: (2, 2)
Goal Test Tercapai: YA, SUKSES!
```

### 5. Output `05_problem_formulation_search_prep.py`
```text
=== DEMONSTRASI PROBLEM FORMULATION LENGKAP ===
1. Initial State : (0, 0)
2. Goal State    : (2, 2)
3. Rintangan     : {(0, 1), (1, 1)}
4. Actions di Initial State: ['DOWN']

Uji Jalur Rencana: ['DOWN', 'DOWN', 'RIGHT', 'RIGHT']
  Step 1: Dari (0, 0) ambil 'DOWN' -> Tiba di (1, 0) (Biaya langkah: 1.0)
  Step 2: Dari (1, 0) ambil 'DOWN' -> Tiba di (2, 0) (Biaya langkah: 1.0)
  Step 3: Dari (2, 0) ambil 'RIGHT' -> Tiba di (2, 1) (Biaya langkah: 1.0)
  Step 4: Dari (2, 1) ambil 'RIGHT' -> Tiba di (2, 2) (Biaya langkah: 1.0)

Hasil Akhir Status: (2, 2)
Goal Reached?     : True
Total Path Cost   : 4.0 unit biaya
```

