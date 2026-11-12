# Panduan Praktik: Intelligent Agents & AI Problem Solving

> **Tingkat Keahlian**: Level 1 — AI Foundations  
> **Chapter**: 03 — Intelligent Agents & AI Problem Solving  
> **Jalur Spesialisasi**: Core Curriculum  
> **Lingkungan**: Python >= 3.12 (Standard Library, tanpa instalasi pustaka luar)  

---

## 1. Tujuan Praktik

Setelah menyelesaikan sesi laboratorium mandiri ini, pembelajar mampu:
1. Mengimplementasikan fungsi agen (*agent function*) dan program agen (*agent program*) menggunakan Python berorientasi objek.
2. Membedakan perilaku reaktif murni (*Simple Reflex Agent*) dengan agen yang memiliki memori status (*Model-Based Reflex Agent*).
3. Merumuskan spesifikasi PEAS (*Performance measure, Environment, Actuators, Sensors*) ke dalam struktur data yang dapat dievaluasi secara kuantitatif.
4. Membangun representasi ruang keadaan (*State Space Representation*) untuk lingkungan grid 2D diskret.
5. Memformulasikan masalah formal AI (*Problem Formulation*) yang terdiri dari lima komponen baku: *initial state*, *actions*, *transition model*, *goal test*, dan *path cost*.

---

## 2. Persyaratan Lingkungan

- **Python Runtime**: Python 3.12+ (terverifikasi kompatibel pada Python 3.12.10).
- **Dependensi Eksternal**: **Tidak ada** (murni memanfaatkan modul standar Python seperti `typing`, `dataclasses`, `math`).
- **Editor / IDE**: Visual Studio Code, PyCharm, atau Jupyter Notebook / Google Colab.
- **Kebutuhan Perangkat Keras**: CPU standar (GPU tidak diperlukan untuk materi agen simbolik dasar ini).

---

## 3. Langkah Menjalankan Kode Praktik

Buka terminal di root repositori dan pastikan virtual environment Anda aktif:

```bash
# 1. Pindah ke direktori kode praktik Chapter 03
cd content/level-1-ai-foundations/chapter-03/code

# 2. Jalankan skrip Python runnable secara mandiri
python 01_simple_reflex_temperature.py
python 02_model_based_room_agent.py
python 03_peas_specification_model.py
python 04_grid_state_space.py
python 05_problem_formulation_search_prep.py
```

---

## 4. Katalog Skrip Praktik

### `01_simple_reflex_temperature.py`
- **Konsep**: Simple Reflex Agent.
- **Mekanisme**: Memetakan percept saat ini (suhu dan status kehadiran orang) ke tindakan diskret pendinginan/pemanasan berdasarkan aturan *condition-action* statis.
- **Keterbatasan**: Tidak memiliki memori; rentan terhadap pergantian sakelar berulang jika suhu berada tepat di batas ambang toleransi (*chattering*).

### `02_model_based_room_agent.py`
- **Konsep**: Model-Based Reflex Agent dengan *Internal State*.
- **Mekanisme**: Menyimpan status tindakan sebelumnya, menghitung tren perubahan suhu (`RISING`, `FALLING`, `STABLE`), dan menerapkan proteksi kompresor (*anti-chattering*) untuk menunda pergantian drastis.
- **Keunggulan**: Mampu beroperasi di lingkungan *partially observable* dan *sequential*.

### `03_peas_specification_model.py`
- **Konsep**: Kerangka Kerja PEAS dan Evaluator Kinerja Rasional.
- **Mekanisme**: Mengenkapsulasi spesifikasi PEAS dalam `@dataclass` Python dan menghitung nilai rasionalitas objektif berdasarkan fungsi penalti energi listrik dan ketidaknyamanan termal manusia.

### `04_grid_state_space.py`
- **Konsep**: State Space Representation pada Grid 3x3.
- **Mekanisme**: Merepresentasikan semesta keadaan diskret $(r, c)$, memetakan rintangan (*obstacles*), menghasilkan daftar tindakan legal, dan mengeksekusi model transisi deterministik $T(s, a) \to s'$.

### `05_problem_formulation_search_prep.py`
- **Konsep**: AI Problem Formulation Baku (Russell & Norvig).
- **Mekanisme**: Kelas dasar `SearchProblem` dan turunan konkrit `GridNavigationProblem` yang mendefinisikan *initial state*, *actions(s)*, *result(s, a)*, *is_goal(s)*, *step_cost(s, a, s')*, dan *path_cost*.
- **Peran**: Jembatan langsung menuju algoritma pencarian rute (BFS, DFS, UCS, A*) pada Chapter 04.

---

## 5. Companion Jupyter Notebook

Tersedia berkas notebook interaktif pada:
`content/level-1-ai-foundations/chapter-03/notebooks/chapter_03_companion.ipynb`

### Cara Membuka:
1. **Di VS Code**: Buka file `.ipynb` tersebut, pilih kernel Python 3 dari virtual environment Anda, lalu klik **"Run All"**.
2. **Di Google Colab**: Unggah file `chapter_03_companion.ipynb` ke Google Colab dan jalankan cell secara berurutan. Seluruh sel bersifat *self-contained* dan bebas dari instalasi paket tambahan.

---

## 6. Dataset dan Representasi Lingkungan

Sesuai hakikat pembelajaran agen cerdas tahap awal, data lingkungan disajikan secara sintetis dan deterministik:
- **Aliran Sensor Suhu HVAC**: Deret nilai suhu sintetis antara 21.0°C hingga 28.5°C dengan variasi status kehadiran manusia (*occupancy* biner).
- **Peta Grid 2D**: Matriks koordinat berukuran $3 \times 3$ dengan dinding pembatas dan sel rintangan diskret pada koordinat $(1, 1)$ dan $(0, 1)$.
- **Lisensi**: Seluruh kode dan data sintetis dirilis di bawah lisensi terbuka MIT License.

---

## 7. Pertanyaan Eksplorasi Mandiri

1. Pada `01_simple_reflex_temperature.py`, apa yang terjadi jika sensor suhu mengalami *noise* sesaat sebesar $\pm 2^\circ\text{C}$ pada setiap langkah waktu? Bagaimana cara agen berbasis model pada `02_model_based_room_agent.py` meredam gangguan tersebut?
2. Pada `03_peas_specification_model.py`, ubah bobot penalti ketidaknyamanan termal dari 3.0 menjadi 10.0. Bagaimana pengaruh perubahan bobot ini terhadap penilaian rasionalitas tindakan agen?
3. Pada `05_problem_formulation_search_prep.py`, tambahkan biaya langkah (*step cost*) yang lebih mahal (misalnya 2.5) untuk gerakan ke arah `UP` (melawan gravitasi) dibandingkan gerakan horizontal. Hitung bagaimana total biaya jalur berubah!

