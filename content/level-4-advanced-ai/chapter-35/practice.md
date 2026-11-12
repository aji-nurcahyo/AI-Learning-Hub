# Panduan Praktik Laboratorium: Reinforcement Learning Fundamentals
**Artificial Intelligence Learning Hub — Level 4 (Chapter 35)**  
*Pengembang: Aji Pamungkas Tri Nurcahyo — Institut Teknologi Sumatera*

---

## 1. DESKRIPSI DAN PRASYARAT PRAKTIKUM

Praktikum ini membimbing mahasiswa mengimplementasikan dan memverifikasi algoritma dasar **Reinforcement Learning (RL)** secara mandiri pada lingkungan CPU lokal tanpa bergantung pada akselerator GPU, API berbayar, atau simulator 3D eksternal yang berat.

### Prasyarat Pengetahuan
- Probabilitas dasar dan nilai harapan (*expected value*).
- Kalkulus turunan parsial untuk *policy gradient*.
- Pemrograman Python, manipulasi array NumPy, serta dasar tensor & autograd PyTorch (Chapter 23).

### Lingkungan Eksekusi
- **Bahasa:** Python 3.10+ (Direkomendasikan Python 3.12).
- **Pustaka Utama:** `numpy`, `matplotlib`, `torch` (CPU edition).
- **Simulator:** `GridWorld` mandiri berbasis NumPy di `code/gridworld_environment.py`.

---

## 2. PANDUAN LANGKAH DEMI LANGKAH EKSEKUSI

Seluruh 16 script Python mandiri tersimpan di `content/level-4-advanced-ai/chapter-35/code/`:

### Langkah 1: Simulasi Lingkungan GridWorld Mandiri
```bash
python content/level-4-advanced-ai/chapter-35/code/gridworld_environment.py
```
- **Tujuan:** Memahami arsitektur `GridWorld` 4x4 beserta representasi *state* (0–15), *action* (UP, RIGHT, DOWN, LEFT), hambatan dinding `#`, bahaya `X`, dan terminal *goal* `G`.

### Langkah 2: Interaksi Agent–Environment & Pembentukan Trajektori
```bash
python content/level-4-advanced-ai/chapter-35/code/rl_foundations.py
```
- **Tujuan:** Mensimulasikan siklus `reset()` dan `step()` dengan kebijakan acak seragam untuk membentuk episode trajektori $(S_0, A_0, R_1, \dots)$.

### Langkah 3: Dinamika Transisi MDP Deterministik vs Stokastik
```bash
python content/level-4-advanced-ai/chapter-35/code/mdp_transitions.py
```
- **Tujuan:** Menguji kebenaran aksioma probabilitas $\sum_{s'} P(s' \mid s, a) = 1.0$ pada model deterministik (`slip_prob = 0.0`) dan stokastik (`slip_prob = 0.2`).

### Langkah 4: Perhitungan Discounted Return & Analisis Gamma
```bash
python content/level-4-advanced-ai/chapter-35/code/discounted_returns.py
```
- **Tujuan:** Menghitung $G_t = \sum_k \gamma^k R_{t+k+1}$ dan membandingkan pengaruh nilai $\gamma \in \{0.0, 0.5, 0.9, 0.99\}$ terhadap bobot imbalan masa depan.

### Langkah 5: Fungsi Nilai $V^\pi(s)$ dan $Q^\pi(s, a)$
```bash
python content/level-4-advanced-ai/chapter-35/code/value_functions.py
```
- **Tujuan:** Memverifikasi identitas aljabar $V^\pi(s) = \sum_a \pi(a \mid s) Q^\pi(s, a)$ secara numerik di seluruh *state*.

### Langkah 6: Perbandingan Bellman Expectation vs Bellman Optimality
```bash
python content/level-4-advanced-ai/chapter-35/code/bellman_equations.py
```
- **Tujuan:** Membuktikan perbedaan antara rata-rata ekspektasi kebijakan tetap versus operator maksimum optimalitas pada kasus numerik 3x3.

### Langkah 7: Iterative Policy Evaluation (Dynamic Programming)
```bash
python content/level-4-advanced-ai/chapter-35/code/policy_evaluation.py
```
- **Tujuan:** Mengevaluasi nilai $V^\pi(s)$ untuk kebijakan acak seragam hingga toleransi $\Delta < 10^{-5}$.

### Langkah 8: Value Iteration
```bash
python content/level-4-advanced-ai/chapter-35/code/value_iteration.py
```
- **Tujuan:** Menjalankan *Bellman Optimality Backup* hingga memperoleh $V^*(s)$ dan mengekstraksi kebijakan optimal $\pi^*(s)$ dalam 24 iterasi.

### Langkah 9: Policy Iteration & Uji Paritas
```bash
python content/level-4-advanced-ai/chapter-35/code/policy_iteration.py
```
- **Tujuan:** Mengamati konvergensi cepat *Policy Iteration* (4 iterasi luar) dan membuktikan kesetaraan solusinya dengan *Value Iteration* (selisih maks $< 10^{-4}$).

### Langkah 10: Model-Free First-Visit Monte Carlo
```bash
python content/level-4-advanced-ai/chapter-35/code/monte_carlo_returns.py
```
- **Tujuan:** Mengestimasi $V^\pi(s)$ murni dari sampel 1.500 episode tanpa mengakses model transisi $P(s' \mid s, a)$.

### Langkah 11: Temporal-Difference TD(0) vs Monte Carlo
```bash
python content/level-4-advanced-ai/chapter-35/code/td_learning.py
```
- **Tujuan:** Mempraktikkan pembaruan langkah demi langkah berbasis *bootstrapping* TD(0) dan membandingkan RMSE konvergensinya terhadap Monte Carlo.

### Langkah 12: Kontrol Off-Policy Q-Learning
```bash
python content/level-4-advanced-ai/chapter-35/code/q_learning.py
```
- **Tujuan:** Memverifikasi perhitungan manual satu langkah pembaruan Q-Learning serta melatih Q-Table 800 episode dengan evaluasi ketat $\epsilon = 0.0$.

### Langkah 13: Eksplorasi vs Eksploitasi (Jadwal Epsilon-Greedy)
```bash
python content/level-4-advanced-ai/chapter-35/code/epsilon_greedy.py
```
- **Tujuan:** Membandingkan strategi *Pure Greedy* ($\epsilon=0$), konstan tinggi ($\epsilon=0.5$), *Linear Decay*, dan *Exponential Decay* terhadap cakupan state dan return evaluasi.

### Langkah 14: Visualisasi Q-Table & Heatmap Matplotlib
```bash
python content/level-4-advanced-ai/chapter-35/code/q_table_visualization.py
```
- **Tujuan:** Mencetak matriks lengkap $Q(s, a)$, peta panah kebijakan, serta menghasilkan grafik heatmap & kurva belajar di `experiments/q_table_heatmap.png`.

### Langkah 15: Policy Gradient (REINFORCE dengan PyTorch)
```bash
python content/level-4-advanced-ai/chapter-35/code/policy_gradient_reinforce.py
```
- **Tujuan:** Melatih jaringan logits *Softmax* menggunakan gradien kebijakan $\nabla_\theta \log \pi_\theta(a \mid s)(G_t - b)$ pada CPU hingga tingkat keberhasilan 100%.

### Langkah 16: Benchmark Perbandingan Kebijakan RL
```bash
python content/level-4-advanced-ai/chapter-35/code/rl_policy_comparison.py
```
- **Tujuan:** Membandingkan Random Baseline, Value Iteration, Q-Learning, dan REINFORCE pada 50 episode stokastik identik (`slip_prob=0.1`) dan mencetak grafik perbandingan kurva belajar (`experiments/rl_policy_comparison.png`).

---

## 3. LATIHAN BERTINGKAT & PEMBAHASAN

### Tingkat Dasar (Pemahaman Konsep & Perhitungan)
1. **Latihan 1.1 (State, Action, Reward):**  
   Pada peta `GridWorld(rows=4, cols=4)` dengan `goal_pos=(3, 3)` dan `step_reward=-0.1`, mengapa `step_reward` diberi nilai negatif kecil alih-alih `0.0`?  
   *Pembahasan:* Penalti langkah `-0.1` mendorong agen meminimalkan jumlah langkah untuk mencapai tujuan (*shortest path*), mencegah agen berputar-putar tanpa tujuan.
2. **Latihan 1.2 (Discounted Return):**  
   Sebuah agen menerima urutan imbalan $[-1, +2, +5]$ dengan faktor diskonto $\gamma = 0.5$. Hitunglah $G_0$ secara manual dan verifikasi dengan `discounted_returns.py`.  
   *Pembahasan:* $G_0 = -1 + 0.5(2) + (0.5)^2(5) = -1 + 1 + 1.25 = 1.25$.

### Tingkat Menengah (Algoritma & Implementasi)
3. **Latihan 2.1 (Bellman Update):**  
   Jika pada *state* $(3, 2)$ aksi `RIGHT` mengarah langsung ke terminal *Goal* $(3, 3)$ dengan $R = +10.0$, mengapa nilai perkalian $\gamma \cdot V(\text{Goal})$ harus bernilai $0.0$?  
   *Pembahasan:* Setelah masuk ke *terminal state*, episode langsung berakhir dan tidak ada lagi imbalan di masa depan. Oleh karena itu nilai terminal wajib dikunci ke $0.0$.
4. **Latihan 2.2 (Q-Learning Update):**  
   Uji perubahan nilai learning rate $\alpha = 0.01$ vs $\alpha = 0.5$ pada `q_learning.py` dengan `slip_prob=0.2`. Manakah yang menghasilkan Q-values lebih stabil dan mengapa?  
   *Pembahasan:* Pada lingkungan stokastik (`slip_prob=0.2`), $\alpha=0.5$ terlalu reaktif terhadap fluktuasi acak sehingga Q-values berosilasi, sedangkan $\alpha$ lebih kecil merata-ratakan transisi stokastik dengan stabil.

### Tingkat Mahir (Desain & Analisis Eksperimental)
5. **Latihan 3.1 (Reward Hacking):**  
   Tambahkan imbalan sementara $+0.5$ setiap kali agen melewati sel $(1, 0)$ pada GridWorld tanpa menghapusnya setelah dilewati. Amati perilaku Q-Learning dan jelaskan fenomena yang terjadi!  
   *Pembahasan:* Agen akan memanfaatkan celah ini (*reward hacking*) dengan bergerak mondar-mandir $(0,0) \leftrightarrow (1,0)$ berulang kali untuk mengumpulkan imbalan $+0.5$ tanpa pernah menuju *Goal*.
6. **Latihan 3.2 (Evaluasi Terkendali):**  
   Buat eksperimen multi-seed (5 seed berbeda: `[1, 2, 3, 4, 5]`) pada `rl_policy_comparison.py` dan hitung rata-rata serta standar deviasi Return antara Q-Learning dan REINFORCE.

---

## 4. PANDUAN MINI PROJECT

### Judul Proyek
**Training an RL Agent in a Custom GridWorld**

### Deliverables Wajib
1. **Definisi Masalah & Desain Lingkungan:** Spesifikasi peta GridWorld kustom minimal 5x5 dengan rintangan dinding, minimal 2 zona bahaya, dan 1 tujuan.
2. **Formulasi Matematis MDP:** Definisi lengkap $(\mathcal{S}, \mathcal{A}, P, R, \gamma)$ beserta bukti kepatuhan $\sum P = 1$.
3. **Baseline Kebijakan Acak:** Pengukuran performa *random policy* sebagai batas bawah (*lower bound*).
4. **Solusi Referensi DP & Kontrol Model-Free:** Implementasi *Value Iteration* (jika model diketahui) dan *Q-Learning* (model-free) disertai strategi *epsilon decay*.
5. **Visualisasi & Evaluasi:** Kurva pembelajaran (*learning curve* asli hasil eksperimen), visualisasi panah kebijakan, pengujian ketat dengan $\epsilon = 0$, serta analisis desain *reward* dan reproduksibilitas seed.

---

## 5. RUBRIK PENILAIAN (100 POIN)

| Komponen Penilaian | Bobot | Kriteria Pemenuhan Standar Penuh |
| :--- | :---: | :--- |
| **1. Problem & Environment Design** | **20 Poin** | Desain simulator GridWorld kustom berfungsi penuh, mendukung reset/step, terminal handling benar, dan tata letak rintangan bermakna. |
| **2. Mathematical Formulation** | **20 Poin** | Formalisasi MDP akurat, perhitungan manual discounted return dan Bellman update cocok dengan output NumPy, serta penanganan terminal state tepat. |
| **3. Algorithm Implementation** | **30 Poin** | Implementasi Value Iteration dan Q-Learning bersih, bebas bug, mendukung evaluasi terpisah ($\epsilon=0$), dan mencakup uji REINFORCE/eksplorasi. |
| **4. Evaluation & Interpretation** | **20 Poin** | Perbandingan empiris multi-run terhadap baseline acak, grafik kurva pembelajaran berdasarkan data asli, serta analisis *reward design* atau *reward hacking*. |
| **5. Documentation & Reproducibility** | **10 Poin** | Kode terstruktur rapi, penetapan random seed terdokumentasi, dan seluruh skrip dapat direproduksi 100% di CPU tanpa error. |

---

## 6. REFERENSI AKADEMIK TERVERIFIKASI

1. **Sutton, R. S., & Barto, A. G. (2018).** *Reinforcement Learning: An Introduction (2nd Edition).* MIT Press.
2. **Bellman, R. (1957).** *Dynamic Programming.* Princeton University Press.
3. **Watkins, C. J. C. H., & Dayan, P. (1992).** *Q-learning.* Machine Learning, 8(3–4), 279–292.
4. **Williams, R. J. (1992).** *Simple statistical gradient-following algorithms for connectionist reinforcement learning.* Machine Learning, 8(3–4), 229–256.
5. **Paszke, A., et al. (2019).** *PyTorch: An Imperative Style, High-Performance Deep Learning Library.* Advances in Neural Information Processing Systems (NeurIPS 2019).

