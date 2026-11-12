# Panduan Praktik: Search Algorithms — Uninformed & Informed Search

> **Tingkat Keahlian**: Level 1 — AI Foundations  
> **Chapter**: 04 — Search Algorithms: Uninformed & Informed Search  
> **Jalur Spesialisasi**: Core Curriculum  
> **Lingkungan**: Python >= 3.12 (Standard Library, tanpa instalasi pustaka luar)  

---

## 1. Tujuan Praktik

Setelah menyelesaikan sesi laboratorium mandiri ini, pembelajar mampu:
1. Mengimplementasikan dan menjalankan algoritma pencarian tanpa informasi (*Uninformed Search*): Breadth-First Search (BFS), Depth-First Search (DFS), Depth-Limited Search (DLS), Iterative Deepening Search (IDDFS), dan Uniform-Cost Search (UCS).
2. Mengimplementasikan dan menjalankan algoritma pencarian terinformasi (*Informed Search*): Greedy Best-First Search dan A\* Search.
3. Mengukur dan membandingkan performa empiris: panjang jalur (*path length*), total biaya (*path cost*), jumlah simpul yang diekspansi (*expanded nodes*), jumlah simpul yang digenerate (*generated nodes*), serta ukuran puncak memori *frontier*.
4. Menganalisis trade-off antara pencarian dengan jaminan biaya minimum (UCS, A\*) vs pencarian agresif cepat (Greedy Best-First) pada graf berbobot Peta Romania dan grid 2D.

---

## 2. Persyaratan Lingkungan

- **Python Runtime**: Python 3.12+ (terverifikasi kompatibel pada Python 3.12.10).
- **Dependensi Eksternal**: **Tidak ada** (murni memanfaatkan pustaka standar Python: `collections.deque`, `heapq`, `math`, `typing`, `time`).
- **Editor / Lingkungan**: Visual Studio Code, PyCharm, atau Jupyter Notebook / Google Colab.
- **Kebutuhan Perangkat Keras**: CPU standar (eksekusi hanya membutuhkan waktu < 1 detik).

---

## 3. Langkah Menjalankan Kode Praktik

Buka terminal di root repositori dan pastikan virtual environment Anda aktif:

```bash
# 1. Pindah ke direktori kode praktik Chapter 04
cd content/level-1-ai-foundations/chapter-04/code

# 2. Menjalankan skrip algoritma individual
python bfs_search.py
python dfs_search.py
python depth_limited_search.py
python iterative_deepening_search.py
python uniform_cost_search.py
python greedy_best_first_search.py
python astar_search.py

# 3. Menjalankan skrip evaluasi komparatif lengkap (Benchmarking)
python compare_search_algorithms.py
```

---

## 4. Ringkasan Hasil Eksperimen Nyata (Peta Romania)

| Algoritma | Status | Jumlah Langkah | Total Jarak (km) | Simpul Diekspansi | Keterangan Teoretis |
|---|:---:|:---:|:---:|:---:|---|
| **BFS** | SUCCESS | 3 hop | 450.0 km | 5 node | Optimal untuk langkah paling sedikit, bukan jarak terpendek |
| **DFS** | SUCCESS | 3 hop | 450.0 km | 4 node | Tidak menjamin optimalitas |
| **DLS (L=4)** | SUCCESS | 3 hop | 450.0 km | 3 node | Terpotong jika $L < 3$, sukses pada $L \ge 3$ |
| **IDDFS** | SUCCESS | 3 hop | 450.0 km | 8 node | Memori linier $\mathcal{O}(b \cdot d)$, optimalitas hop seperti BFS |
| **UCS** | SUCCESS | 4 hop | **418.0 km** | 13 node | **Jalur terpendek optimal sejati** (via Rimnicu Vilcea) |
| **Greedy** | SUCCESS | 3 hop | 450.0 km | 4 node | Sangat cepat/rakus, namun terjebak jalur Fagaras yang lebih jauh |
| **A\*** | SUCCESS | 4 hop | **418.0 km** | **6 node** | **Optimal terpendek**, ekspansi separuh lebih hemat dari UCS |

---

## 5. Companion Jupyter Notebook

Tersedia berkas notebook interaktif pada:
`content/level-1-ai-foundations/chapter-04/notebooks/chapter_04_companion.ipynb`

Seluruh sel telah dirancang terpadu untuk mendemonstrasikan perbandingan rute graf dan grid secara interaktif.

---

## 6. Pertanyaan Eksplorasi Mandiri

1. Pada `astar_search.py`, ganti fungsi heuristik straight-line distance dengan fungsi yang mengembalikan nilai `0.0` untuk semua kota ($h(n) = 0$). Perhatikan jumlah simpul yang diekspansi dan bandingkan hasilnya dengan `uniform_cost_search.py`!
2. Pada `depth_limited_search.py`, jalankan pencarian dengan limit = 2. Amati status return value `CUTOFF` dan jelaskan mengapa algoritma tidak mengembalikan `FAILURE`!
3. Pada `GridNavigationProblem`, tambahkan rintangan diagonal atau pergerakan diagonal dengan biaya $\sqrt{2} \approx 1.414$. Apakah heuristik Manhattan tetap memenuhi syarat *admissibility*?

