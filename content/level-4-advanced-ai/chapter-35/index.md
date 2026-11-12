# Chapter 35: Reinforcement Learning Fundamentals

> **Artificial Intelligence Learning Hub — Level 4: Applied & Advanced AI Systems**  
> *Pengembang:* Aji Pamungkas Tri Nurcahyo — Institut Teknologi Sumatera  
> *Identitas Mata Kuliah:* Bab 35 dari 40 Master Curriculum  
> *Prasyarat:* Chapter 03 (Intelligent Agents), Chapter 12 (ML Foundations), Chapter 21 (Neural Networks), Chapter 34 (AI Agents)

---

## DAFTAR ISI
1. [Bagian A: Fondasi Reinforcement Learning & Perbedaan Paradigma](#bagian-a-fondasi-reinforcement-learning--perbedaan-paradigma)
2. [Bagian B: Markov Decision Process (MDP) & Dinamika Transisi](#bagian-b-markov-decision-process-mdp--dinamika-transisi)
3. [Bagian C: Return Terdiskonto & Fungsi Nilai ($V^\pi, Q^\pi$)](#bagian-c-return-terdiskonto--fungsi-nilai-vpi-qpi)
4. [Bagian D: Persamaan Bellman (Expectation vs Optimality)](#bagian-d-persamaan-bellman-expectation-vs-optimality)
5. [Bagian E: Dynamic Programming (Policy Evaluation, Policy & Value Iteration)](#bagian-e-dynamic-programming-policy-evaluation-policy--value-iteration)
6. [Bagian F: Pembelajaran Bebas Model: Monte Carlo vs Temporal-Difference](#bagian-f-pembelajaran-bebas-model-monte-carlo-vs-temporal-difference)
7. [Bagian G: Kontrol Off-Policy Q-Learning & Worked Numerical Example](#bagian-g-kontrol-off-policy-q-learning--worked-numerical-example)
8. [Bagian H: Eksplorasi vs Eksploitasi & Strategi Epsilon-Greedy](#bagian-h-eksplorasi-vs-eksploitasi--strategi-epsilon-greedy)
9. [Bagian I: Pengantar Policy Gradients & Algoritme REINFORCE](#bagian-i-pengantar-policy-gradients--algoritme-reinforce)
10. [Bagian J: Desain Reward, Reward Hacking & Pengayaan Reward Modeling](#bagian-j-desain-reward-reward-hacking--pengayaan-reward-modeling)
11. [Bagian K: Simulator GridWorld Mandiri](#bagian-k-simulator-gridworld-mandiri)
12. [Bagian L: Evaluasi Performa RL & Keterbatasan Fundamental](#bagian-l-evaluasi-performa-rl--keterbatasan-fundamental)
13. [Latihan Bertingkat & Pembahasan Lengkap](#latihan-bertingkat--pembahasan-lengkap)
14. [Panduan Mini Project & Rubrik Asesmen 100 Poin](#panduan-mini-project--rubrik-asesmen-100-poin)
15. [Referensi Akademik Terverifikasi](#referensi-akademik-terverifikasi)

---

## BAGIAN A: FONDASI REINFORCEMENT LEARNING & PERBEDAAN PARADIGMA

### 1. Belajar Melalui Interaksi Berurutan
**Reinforcement Learning (RL)** adalah cabang utama kecerdasan buatan yang mempelajari cara suatu entitas pengambil keputusan (**Agent**) belajar memilih tindakan (**Actions**) di dalam suatu lingkungan (**Environment**) guna memaksimalkan sinyal umpan balik numerik kumulatif (**Reward**) jangka panjang.

Berbeda dengan sistem *Supervised Learning* yang mengandalkan dataset berlabel $(x_i, y_i)$ statis dari pengawas manusia, dalam Reinforcement Learning tidak ada guru yang memberi tahu aksi mana yang benar pada setiap langkah. Agen harus mengeksplorasi lingkungannya sendiri melalui mekanisme *trial-and-error* dan menghadapi **konsekuensi tertunda (*delayed consequences*)**: keputusan yang diambil pada langkah $t=0$ baru menampakkan dampak positif atau negatifnya pada langkah $t=20$.

```
+-----------------------------------------------------------------------+
|          PERBANDINGAN TIGA PARADIGMA MACHINE LEARNING                 |
+-----------------------------------------------------------------------+
| 1. SUPERVISED LEARNING                                                |
|    - Data: Pasangan (X, y) berlabel, diasumsikan statis & i.i.d.     |
|    - Umpan Balik: Eksplisit & langsung pada setiap sampel latih.      |
|                                                                       |
| 2. UNSUPERVISED LEARNING                                              |
|    - Data: Fitur X tanpa label target.                                |
|    - Tujuan: Menemukan struktur tersembunyi, klaster, atau kompresi.  |
|                                                                       |
| 3. REINFORCEMENT LEARNING                                             |
|    - Data: Dihasilkan dinamis oleh interaksi agen (non-i.i.d.).      |
|    - Umpan Balik: Sinyal angka skalar (Reward) yang sering tertunda.  |
|    - Tantangan: Menyeimbangkan Exploration vs Exploitation.           |
+-----------------------------------------------------------------------+
```

### 2. Hubungan dengan Chapter 03 dan Perbedaan dari Agen LLM (Chapter 34)
Pada **Chapter 03 (*Intelligent Agents*)**, agen didefinisikan secara umum sebagai entitas yang mengamati melalui sensor dan bertindak melalui aktuator. RL memformalkan bagaimana *utility-based agent* mempelajari fungsi nilainya dari pengalaman nyata.

Penting untuk membedakan **RL Agent (Chapter 35)** dari **LLM-based Tool-Using Agent (Chapter 34)**:
- **LLM Tool-Using Agent (Bab 34):** Menggunakan pengetahuan bahasa pra-latih pada LLM sebagai *planner* semantik untuk mengoordinasikan pemanggilan API/tool terstruktur berdasarkan instruksi teks dan aturan tata kelola (*approval gate*). Bobot model bahasa tidak diperbarui saat bertugas.
- **Reinforcement Learning Agent (Bab 35):** Mempelajari kebijakan pengambilan keputusan matematis $\pi(a \mid s)$ atau fungsi nilai $Q(s, a)$ langsung dari nol melalui ribuan iterasi interaksi numerik dengan lingkungan (*state, action, reward*) guna mengoptimalkan tujuan matematis.

```
+-----------------------------------------------------------------------+
|              SIKLUS INTERAKSI AGENT–ENVIRONMENT                       |
+-----------------------------------------------------------------------+
|                                                                       |
|           +-------------------------------------------+               |
|           |                  AGENT                    |               |
|           |  (Memiliki Policy & Value / Policy Net)   |               |
|           +-------------------------------------------+               |
|                 |                                ^                    |
|     Action A_t  |                                | State S_{t+1}      |
|                 |                                | & Reward R_{t+1}   |
|                 v                                |                    |
|           +-------------------------------------------+               |
|           |               ENVIRONMENT                 |               |
|           |    (Mengubah state & memberi reward)      |               |
|           +-------------------------------------------+               |
+-----------------------------------------------------------------------+
```

### 3. Terminologi Dasar
- **State ($S_t$):** Deskripsi konfigurasi lingkungan pada waktu $t$.
- **Action ($A_t$):** Keputusan yang diambil agen dari ruang aksi $\mathcal{A}$.
- **Reward ($R_{t+1}$):** Sinyal skalar instan yang diberikan lingkungan setelah transisi $(S_t, A_t) \to S_{t+1}$.
- **Policy ($\pi$):** Strategi perilaku agen yang memetakan *state* menjadi aksi atau distribusi probabilitas aksi $\pi(a \mid s)$.
- **Trajectory ($\tau$):** Rangkaian lengkap transisi $(S_0, A_0, R_1, S_1, A_1, R_2, \dots)$.
- **Episode:** Satu babak interaksi dari *initial state* hingga mencapai *terminal state* (atau batas langkah maksimum).

---

## BAGIAN B: MARKOV DECISION PROCESS (MDP) & DINAMIKA TRANSISI

### 1. Sifat Markov (*Markov Property*)
Suatu representasi *state* $S_t$ memenuhi **Markov Property** apabila probabilitas transisi ke *state* berikutnya hanya bergantung pada *state* dan aksi saat ini, bukan pada seluruh sejarah lintasan sebelumnya:

$$P(S_{t+1} = s' \mid S_t, A_t, S_{t-1}, A_{t-1}, \dots, S_0, A_0) = P(S_{t+1} = s' \mid S_t, A_t)$$

Dengan kata lain, *state* saat ini adalah statistik yang cukup (*sufficient statistic*) merangkum seluruh informasi masa lalu yang relevan untuk masa depan.

### 2. Tuple Formal MDP
Lingkungan RL dimodelkan secara matematis sebagai **Markov Decision Process (MDP)** berbentuk tuple lima elemen:

$$\mathcal{M} = (\mathcal{S}, \mathcal{A}, P, R, \gamma)$$

1. $\mathcal{S}$: Himpunan seluruh *state* sah (*State Space*).
2. $\mathcal{A}$: Himpunan seluruh *action* yang tersedia (*Action Space*).
3. $P(s' \mid s, a) = \Pr(S_{t+1} = s' \mid S_t = s, A_t = a)$: Fungsi probabilitas transisi yang memenuhi aksioma $\sum_{s' \in \mathcal{S}} P(s' \mid s, a) = 1.0$.
4. $R(s, a, s')$: Fungsi ekspektasi *reward* yang diterima saat berpindah dari $s$ ke $s'$ melalui aksi $a$.
5. $\gamma \in [0, 1]$: Faktor diskonto (*discount factor*).

```
+-----------------------------------------------------------------------+
|          DINAMIKA TRANSISI: DETERMINISTIK VS STOKASTIK                |
+-----------------------------------------------------------------------+
| Misal agen berada di (0, 1) dan memilih aksi RIGHT:                   |
|                                                                       |
| 1. LINGKUNGAN DETERMINISTIK (slip_prob = 0.0)                         |
|    (0, 1) --(RIGHT, P=1.00)--> (0, 2)  [Pasti sampai tujuan]          |
|                                                                       |
| 2. LINGKUNGAN STOKASTIK (slip_prob = 0.20)                            |
|    (0, 1) --(P=0.80)--> (0, 2)  [Sesuai intensi RIGHT]                |
|           --(P=0.10)--> (0, 1)  [Terpeleset UP menabrak dinding atas] |
|           --(P=0.10)--> (0, 1)  [Terpeleset DOWN menabrak WALL (1,1)] |
|    Total Probabilitas: 0.80 + 0.10 + 0.10 = 1.00                      |
+-----------------------------------------------------------------------+
```

### 3. Contoh Worked MDP pada GridWorld 4x4
Dalam simulasi modul ini (`code/gridworld_environment.py`):
- **State Space:** 16 sel indeks $s \in \{0, 1, \dots, 15\}$ yang memetakan koordinat $(r, c)$ via $s = 4r + c$.
- **Action Space:** $\{0:\text{UP}, 1:\text{RIGHT}, 2:\text{DOWN}, 3:\text{LEFT}\}$.
- **Obstacles (Dinding):** $(1, 1)$ dan $(2, 2)$ — tidak dapat dimasuki; menabrak dinding membuat agen tetap di sel asal.
- **Terminal States:**
  - *Goal* $(3, 3)$: Mengakhiri episode dengan imbalan $+10.0$.
  - *Hazard* $(1, 3)$: Mengakhiri episode dengan hukuman $-5.0$.
- **Step Reward:** $-0.1$ pada setiap perpindahan ke sel biasa agar agen termotivasi mencari jalur terpendek.

---

## BAGIAN C: RETURN TERDISKONTO & FUNGSI NILAI ($V^\pi, Q^\pi$)

### 1. Discounted Return ($G_t$)
Tujuan agen bukanlah memaksimalkan imbalan sesaat, melainkan memaksimalkan ekspektasi **Discounted Return ($G_t$)** mulai dari waktu $t$:

$$G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \dots = \sum_{k=0}^{T-t-1} \gamma^k R_{t+k+1}$$

Secara rekursif, persamaan ini dapat ditulis dengan sangat ringkas:

$$G_t = R_{t+1} + \gamma G_{t+1}$$

#### Worked Example Numerik Pengaruh $\gamma$
Misalkan sebuah lintasan sepanjang 6 langkah menghasilkan barisan imbalan $R = [-0.1, -0.1, -0.1, -0.1, -0.1, +10.0]$:
- Jika $\gamma = 0.00$ (*myopic* / berpandangan sempit): $G_0 = -0.1000$ (agen hanya melihat imbalan langkah pertama).
- Jika $\gamma = 0.50$: $G_0 = -0.1 - 0.05 - 0.025 - 0.0125 - 0.00625 + (0.5)^5(10) = +0.1188$.
- Jika $\gamma = 0.90$: $G_0 = +5.4951$.
- Jika $\gamma = 0.99$: $G_0 = +9.0195$.

> **Peringatan Akademis:** Nilai $\gamma$ yang lebih besar **tidak selalu** lebih baik. Pada lingkungan berkelanjutan atau sangat stokastik, $\gamma$ mendekati $1.0$ memperbesar varians estimasi jangka panjang dan menyulitkan stabilitas pembelajaran.

### 2. Fungsi Nilai Keadaan & Aksi ($V^\pi$ dan $Q^\pi$)
1. **State-Value Function $V^\pi(s)$:** Seberapa bagus berada pada *state* $s$ apabila agen mengikuti kebijakan $\pi$:
   $$V^\pi(s) = \mathbb{E}_\pi [G_t \mid S_t = s]$$
2. **Action-Value Function $Q^\pi(s, a)$:** Seberapa bagus mengambil aksi $a$ pada *state* $s$, lalu mengikuti kebijakan $\pi$ setelahnya:
   $$Q^\pi(s, a) = \mathbb{E}_\pi [G_t \mid S_t = s, A_t = a]$$

Kedua fungsi terhubung melalui identitas ekspektasi kebijakan:

$$V^\pi(s) = \sum_{a \in \mathcal{A}} \pi(a \mid s) Q^\pi(s, a)$$

---

## BAGIAN D: PERSAMAAN BELLMAN (EXPECTATION VS OPTIMALITY)

### 1. Bellman Expectation Equation
Untuk kebijakan tetap $\pi$, nilai sebuah *state* harus konsisten dengan rata-rata nilai *state* penerusnya (*backup diagram*):

$$V^\pi(s) = \sum_{a \in \mathcal{A}} \pi(a \mid s) \sum_{s' \in \mathcal{S}} P(s' \mid s, a) \left[ R(s, a, s') + \gamma V^\pi(s') \right]$$

### 2. Bellman Optimality Equation
Nilai optimal $V^*(s) = \max_\pi V^\pi(s)$ mengasumsikan bahwa pada setiap *state*, agen selalu memilih **aksi terbaik** (*max*), bukan merata-ratakan kebijakan acak:

$$V^*(s) = \max_{a \in \mathcal{A}} \sum_{s' \in \mathcal{S}} P(s' \mid s, a) \left[ R(s, a, s') + \gamma V^*(s') \right]$$

Dan untuk $Q^*(s, a)$:

$$Q^*(s, a) = \sum_{s' \in \mathcal{S}} P(s' \mid s, a) \left[ R(s, a, s') + \gamma \max_{a' \in \mathcal{A}} Q^*(s', a') \right]$$

```
+-----------------------------------------------------------------------+
|         BELLMAN EXPECTATION BACKUP VS BELLMAN OPTIMALITY BACKUP       |
+-----------------------------------------------------------------------+
| Misal pada GridWorld 3x3 di State (2, 1) (gamma = 0.9),               |
| estimasi nilai tetangga saat ini:                                     |
|   UP -> 7.10 | RIGHT (ke Goal) -> 10.00 | DOWN -> 7.10 | LEFT -> 5.30 |
|                                                                       |
| 1. BELLMAN EXPECTATION (Kebijakan Seragam pi = 0.25)                  |
|    V(2,1) = 0.25 * (7.10 + 10.00 + 7.10 + 5.30) = 7.3750              |
|                                                                       |
| 2. BELLMAN OPTIMALITY (Operator Maksimum)                             |
|    V*(2,1) = max(7.10, 10.00, 7.10, 5.30) = 10.0000                   |
+-----------------------------------------------------------------------+
```

Persamaan Expectation berbentuk sistem linier (mengevaluasi satu kebijakan $\pi$), sedangkan persamaan Optimality bersifat non-linier karena operator `max` (mencari kebijakan terbaik $\pi^*$). Keduanya tidak identik dan tidak boleh dicampuradukkan.

---

## BAGIAN E: DYNAMIC PROGRAMMING (POLICY EVALUATION, POLICY & VALUE ITERATION)

**Dynamic Programming (DP)** memecahkan MDP secara eksak apabila **seluruh probabilitas transisi $P(s' \mid s, a)$ dan imbalan $R$ diketahui lengkap**.

### 1. Iterative Policy Evaluation
Menghitung $V^\pi(s)$ melalui penyapuan berulang terhadap seluruh $s \in \mathcal{S}$:

$$V_{k+1}(s) \leftarrow \sum_{a} \pi(a \mid s) \sum_{s'} P(s' \mid s, a) [R + \gamma V_k(s')]$$

Berhenti apabila selisih maksimum $\Delta = \max_s |V_{k+1}(s) - V_k(s)| < \theta$.

### 2. Policy Iteration
Menemukan kebijakan optimal dengan mengulang dua fase hingga kebijakan stabil:
1. **Policy Evaluation:** Hitung $V^{\pi_k}(s)$ untuk kebijakan $\pi_k$ saat ini.
2. **Policy Improvement:** Perbarui kebijakan secara serakah: $\pi_{k+1}(s) = \arg\max_a Q^{\pi_k}(s, a)$.

Pada pengujian GridWorld 4x4 stokastik (`slip=0.1`, `code/policy_iteration.py`), **Policy Iteration konvergen tepat dalam 4 langkah perbaikan luar** dengan $V^*(0,0) = 5.0157$.

### 3. Value Iteration
Menyatukan kedua fase dengan langsung menerapkan **Bellman Optimality Backup** di setiap penyapuan iterasi:

$$V_{k+1}(s) \leftarrow \max_{a \in \mathcal{A}} \sum_{s' \in \mathcal{S}} P(s' \mid s, a) [R + \gamma V_k(s')]$$

Pada pengujian `code/value_iteration.py`, Value Iteration konvergen dalam 24 penyapuan hingga toleransi $\theta = 10^{-6}$, menghasilkan nilai identik dengan Policy Iteration (selisih maks $6.45 \times 10^{-7}$).

```
+-----------------------------------------------------------------------+
|     ARAH KEBIJAKAN OPTIMAL HASIL VALUE ITERATION (GridWorld 4x4)      |
+-----------------------------------------------------------------------+
|                                                                       |
|   +-------+-------+-------+-------+                                   |
|   |   ↓   |   ←   |   ↓   |   ←   |   Baris 0: Menjauhi Hazard (1,3)  |
|   +-------+-------+-------+-------+                                   |
|   |   ↓   |   #   |   ↓   |   X   |   Baris 1: Mengitari Dinding #    |
|   +-------+-------+-------+-------+                                   |
|   |   ↓   |   ←   |   #   |   ↓   |   Baris 2: Jalur aman menuju G    |
|   +-------+-------+-------+-------+                                   |
|   |   →   |   →   |   →   |   G   |   Baris 3: Mencapai Goal (+10.0)  |
|   +-------+-------+-------+-------+                                   |
+-----------------------------------------------------------------------+
```

---

## BAGIAN F: PEMBELAJARAN BEBAS MODEL: MONTE CARLO VS TEMPORAL-DIFFERENCE

Di dunia nyata, agen jarang mengetahui matriks probabilitas transisi $P(s' \mid s, a)$. Untuk itu diperlukan pendekatan **Model-Free RL** yang belajar langsung dari sampel pengalaman.

### 1. Monte Carlo (MC) Prediction
Menunggu hingga seluruh episode berakhir, menghitung return aktual $G_t$, lalu memutakhirkan estimasi:

$$V(S_t) \leftarrow V(S_t) + \alpha [G_t - V(S_t)]$$

- **Kelebihan:** Estimasi tak bias (*unbiased*) terhadap $V^\pi(s)$, bekerja baik walaupun lingkungan tidak sepenuhnya bersifat Markov.
- **Kekurangan:** Hanya dapat diterapkan pada tugas berepisode (*episodic*), memiliki varians tinggi, dan harus menunggu hingga akhir episode sebelum belajar.

### 2. Temporal-Difference TD(0) Learning
TD(0) melakukan **bootstrapping**: memperbarui estimasi nilai pada setiap langkah berdasarkan tebakan nilai *state* berikutnya, tanpa menunggu akhir episode:

$$V(S_t) \leftarrow V(S_t) + \alpha \Big[ \underbrace{R_{t+1} + \gamma V(S_{t+1})}_{\text{TD Target}} - V(S_t) \Big]$$

Selisih $\delta_t = R_{t+1} + \gamma V(S_{t+1}) - V(S_t)$ disebut **TD Error**.

- **Perbandingan:** Tidak ada satu metode yang selalu unggul mutlak. TD(0) memiliki varians lebih rendah dan pembaruan online cepat, namun memiliki bias sementara karena bergantung pada tebakan awal $V(S_{t+1})$. MC bebas bias tetapi lambat pada episode sangat panjang.

---

## BAGIAN G: KONTROL OFF-POLICY Q-LEARNING & WORKED NUMERICAL EXAMPLE

**Q-Learning** (Watkins, 1989) adalah algoritme kontrol **Off-Policy Temporal-Difference** yang mempelajari fungsi aksi-nilai optimal $Q^*(s, a)$ secara langsung tanpa mengetahui model transisi lingkungan:

$$Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha \left[ R_{t+1} + \gamma \max_{a' \in \mathcal{A}} Q(S_{t+1}, a') - Q(S_t, A_t) \right]$$

```
+-----------------------------------------------------------------------+
|          WORKED NUMERICAL EXAMPLE: SATU LANGKAH Q-LEARNING            |
+-----------------------------------------------------------------------+
| 1. Q-Table Awal:                                                      |
|    - Q(s=0) = [UP: 0.00, RIGHT: 2.00, DOWN: 1.00, LEFT: 0.00]         |
|    - Q(s=1) = [UP: 1.00, RIGHT: 5.00, DOWN: 0.50, LEFT: 0.00]         |
|                                                                       |
| 2. Transisi yang Dialami Agen:                                        |
|    - State saat ini s = 0, Aksi a = RIGHT (indeks 1), Q_lama = 2.00   |
|    - Imbalan r = -0.10, State berikutnya s' = 1 (Non-Terminal)        |
|    - Hyperparameters: alpha = 0.20, gamma = 0.90                      |
|                                                                       |
| 3. Langkah Perhitungan:                                               |
|    - max_{a'} Q(s'=1, a') = max(1.00, 5.00, 0.50, 0.00) = 5.00        |
|    - TD Target = r + gamma * max_next_q = -0.10 + 0.90 * 5.00 = 4.40  |
|    - TD Error  = TD Target - Q_lama     = 4.40 - 2.00         = 2.40  |
|    - Q_baru(0, 1) = Q_lama + alpha * TD Error                         |
|                   = 2.00 + 0.20 * 2.40 = 2.4800                       |
+-----------------------------------------------------------------------+
```

> **Penanganan Terminal State Wajib:** Apabila $S_{t+1}$ adalah *terminal state* (seperti Goal atau Hazard), tidak ada langkah berikutnya sehingga suku $\max_{a'} Q(S_{t+1}, a')$ wajib bernilai **$0.0$**. Jika lupa menolkan suku ini pada keadaan terminal, nilai Q akan membengkak tanpa batas.

---

## BAGIAN H: EKSPLORASI VS EKSPLOITASI & STRATEGI EPSILON-GREEDY

Untuk mempelajari Q-Table yang akurat, agen harus mengunjungi seluruh *state* dan mencoba beragam aksi (*exploration*), tetapi tetap mengutamakan aksi bernilai tinggi (*exploitation*).

### Kebijakan $\epsilon$-Greedy & Pemisahan Kebijakan Evaluasi
Pada setiap langkah pelatihan:
- Dengan probabilitas $\epsilon$: Pilih aksi acak seragam dari $\mathcal{A}$ (eksplorasi).
- Dengan probabilitas $1 - \epsilon$: Pilih aksi serakah $\arg\max_a Q(s, a)$ (eksploitasi).

Pada `code/epsilon_greedy.py`, jadwal **Exponential Decay** ($\epsilon_0 = 1.0 \to 0.05$) melatih agen menjelajahi seluruh 14 *state* yang dapat dijangkau secara menyeluruh di awal, lalu mengerucutkan perilaku menuju jalur optimal di akhir pelatihan.

**Aturan Evaluasi Wajib:** Saat mengevaluasi performa akhir kebijakan hasil belajar, **matikan seluruh keacakan ($\epsilon = 0.0$)**. Membiarkan $\epsilon > 0$ selama evaluasi akan mencampur kemampuan kebijakan dengan aksi acak dan menurunkan metrik kesuksesan secara artifisial.

---

## BAGIAN I: PENGANTAR POLICY GRADIENTS & ALGORITME REINFORCE

### 1. Dari Nilai Menuju Kebijakan Terparameterisasi
Q-Learning menyimpan tabel nilai $Q(s, a)$ dan memilih aksi secara tidak langsung lewat $\arg\max$. Sebaliknya, metode **Policy Gradient** langsung memparameterisasi kebijakan stokastik menggunakan bobot parameter $\theta$:

$$\pi_\theta(a \mid s) = \text{softmax}(f_\theta(s))_a = \frac{\exp(z_\theta(s, a))}{\sum_{b \in \mathcal{A}} \exp(z_\theta(s, b))}$$

### 2. Penurunan Dasar REINFORCE (Williams, 1992)
Tujuan utama kita adalah memaksimalkan fungsi objektif $J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}[R(\tau)]$. Dengan memanfaatkan *Log-Probability Trick* ($\nabla_\theta \pi_\theta = \pi_\theta \nabla_\theta \log \pi_\theta$), gradien objektif tanpa memerlukan pengetahuan model lingkungan $P(s' \mid s, a)$ dapat ditaksir menggunakan sampel trajektori Monte Carlo:

$$\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^{T-1} \nabla_\theta \log \pi_\theta(A_t \mid S_t) (G_t - b) \right]$$

Di mana $b$ adalah **baseline** (misal rata-rata return berjalan) yang berfungsi mengurangi varians gradien secara drastis tanpa menimbulkan bias ekspektasi. Pada implementasi PyTorch (`code/policy_gradient_reinforce.py`), pembaruan gradien naik (*gradient ascent*) dieksekusi dengan meminimalkan fungsi rugi *loss*:

$$\mathcal{L}(\theta) = -\sum_{t=0}^{T-1} \log \pi_\theta(A_t \mid S_t) \cdot (G_t - b)$$

---

## BAGIAN J: DESAIN REWARD, REWARD HACKING & PENGAYAAN REWARD MODELING

### 1. Perangkap Desain Reward
Merancang fungsi imbalan (*Reward Function*) adalah bagian paling rentan terhadap kesalahan dalam proyek RL:
- **Sparse Rewards:** Hanya memberi nilai `+1` saat mencapai tujuan akhir dan `0` di tempat lain. Sangat aman dari eksploitasi, namun agen acak hampir tidak pernah menemukan tujuan pada peta besar.
- **Dense Rewards / Reward Shaping:** Menambahkan petunjuk imbalan antara (misal mendekati target). Jika tidak dirancang hati-hati sebagai fungsi potensial, agen dapat terjebak dalam **Reward Hacking**.
- **Reward Hacking (Specification Gaming):** Fenomena ketika agen mengoptimalkan angka matematis imbalan secara harfiah sambil menyimpang dari maksud perancangnya (misalnya berputar-putar mengumpulkan poin antara tanpa pernah menyelesaikan misi).

### 2. Pengayaan: Reward Modeling (Hand-Crafted vs Learned Reward)
Pada tugas kualitatif kompleks seperti penulisan teks yang membantu dan sopan, manusia tidak dapat menulis fungsi `if-else` untuk menentukan skor numerik. Dalam **Reward Modeling**, alih-alih menulis fungsi imbalan manual (*hand-crafted*), pengembang melatih jaringan saraf terpisah dari dataset preferensi pilihan manusia untuk memprediksi skor imbalan. *(Catatan: Implementasi RLHF penuh menggunakan Reward Modeling dibahas di luar cakupan bab fondasi ini).*

---

## BAGIAN K: SIMULATOR GRIDWORLD MANDIRI

Simulator `GridWorld` pada `code/gridworld_environment.py` dibangun murni menggunakan Python dan NumPy tanpa dependensi Gym eksternal wajib. Fitur utamanya meliputi:
- Konfigurasi fleksibel ukuran grid, posisi awal, posisi tujuan, daftar rintangan dinding, dan zona bahaya.
- Mendukung transisi **deterministik** (`slip_prob = 0.0`) maupun **stokastik** (`slip_prob > 0.0`).
- Antarmuka standar `reset(seed)` dan `step(action)` yang mengembalikan `(next_state, reward, done, info)` dengan batas langkah per episode (`max_steps`).
- Mengekspos tabel transisi analitik `get_transition_outcomes(s, a)` untuk pengujian langsung Dynamic Programming.

---

## BAGIAN L: EVALUASI PERFORMA RL & KETERBATASAN FUNDAMENTAL

### 1. Hasil Benchmark Empiris pada GridWorld Stokastik
Berdasarkan eksekusi aktual `code/rl_policy_comparison.py` pada lingkungan GridWorld 4x4 stokastik (`slip_prob = 0.1`, `max_steps = 40`) melintasi 50 episode evaluasi dengan seed identik (`5000..5049`):

| Paradigma Kebijakan | Tingkat Keberhasilan (*Success Rate*) | Rata-rata Return (*Avg Return*) | Rata-rata Panjang Episode (*Avg Steps*) |
| :--- | :---: | :---: | :---: |
| **1. Uniform Random (Baseline)** | 20.0% | -2.950 | 24.22 |
| **2. Value Iteration (DP Reference)** | **100.0%** | **9.456** | **6.44** |
| **3. Q-Learning (Model-Free Off-Policy)** | **100.0%** | **9.456** | **6.44** |
| **4. REINFORCE (PyTorch Policy Gradient)** | **100.0%** | **9.446** | **6.54** |

### 2. Keterbatasan Fundamental Reinforcement Learning
1. **Inefisiensi Sampel (*Sample Inefficiency*):** Membutuhkan ratusan hingga jutaan interaksi berulang untuk menguasai tugas sederhana.
2. **Ketidakstabilan & Sensitivitas Seed:** Konvergensi sering sangat sensitif terhadap jadwal *learning rate*, *random seed*, dan *exploration decay*.
3. **Kesenjangan Simulator ke Dunia Nyata (*Sim-to-Real Gap*):** Kebijakan optimal di dalam simulator matematis ideal belum tentu aman atau efektif di robotika fisik akibat derau sensor dan dinamika tak terduga.

---

## LATIHAN BERTINGKAT & PEMBAHASAN LENGKAP

### Tingkat Dasar
1. **Identifikasi MDP:** Sebutkan lima komponen tuple $\mathcal{M}$ pada kasus robot pengantar paket di gudang!  
   *Pembahasan:* $\mathcal{S}$ = koordinat posisi + status muatan paket; $\mathcal{A}$ = maju/belok/ambil/taruh; $P$ = dinamika gerak lantai licin/normal; $R$ = +100 saat paket sampai, -1 per detik, -50 jika menabrak rak; $\gamma$ = preferensi kecepatan pengiriman (misal $0.95$).
2. **Perhitungan Discounted Return:** Hitung $G_0$ untuk imbalan $[-1, -1, +12]$ dengan $\gamma = 0.9$.  
   *Pembahasan:* $G_0 = -1 + 0.9(-1) + 0.9^2(12) = -1 - 0.9 + 9.72 = 7.82$.

### Tingkat Menengah
3. **Bellman Update Manual:** Jika suatu state $s$ memiliki dua aksi dengan nilai ekspektasi aksi $Q(s, a_1) = 4.0$ dan $Q(s, a_2) = 8.0$, hitung $V^\pi(s)$ di bawah kebijakan acak seragam $\pi(a_i \mid s) = 0.5$ versus nilai optimal $V^*(s)$!  
   *Pembahasan:* $V^\pi(s) = 0.5(4.0) + 0.5(8.0) = 6.0$ (Bellman Expectation), sedangkan $V^*(s) = \max(4.0, 8.0) = 8.0$ (Bellman Optimality).
4. **Q-Learning Terminal Bug:** Mengapa sebuah implementasi Q-Learning bisa menghasilkan nilai $Q > 100$ padahal *reward* maksimum lingkungan hanya $+10$?  
   *Pembahasan:* Penyebab paling lazim adalah lupa menerapkan kondisi `0.0 if done` pada suku `max_next_q` ketika transisi berakhir pada *terminal state*, sehingga nilai Q terus mengakumulasi `+10 + gamma * 10 + ...` melewati batas akhir episode.

### Tingkat Mahir
5. **Analisis Reward Hacking:** Jelaskan bahaya menambahkan imbalan $+0.2$ setiap kali agen bergerak ke kanan ke arah target!  
   *Pembahasan:* Jika agen menabrak dinding di sebelah kanan atau mundur ke kiri ($0.0$) lalu maju ke kanan ($+0.2$) terus-menerus dalam siklus osilasi, agen dapat mengumpulkan imbalan tak terbatas tanpa pernah menyentuh *Goal* akhir.

---

## PANDUAN MINI PROJECT & RUBRIK ASESMEN 100 POIN

### Judul Proyek
**Training an RL Agent in a Custom GridWorld**

### Spesifikasi Tugas
Mahasiswa merancang lingkungan GridWorld 5x5 kustom yang memiliki rintangan, minimal dua zona bahaya, serta jalur alternatif, lalu memformalkan tuple MDP-nya secara matematis, mengevaluasi kebijakan baseline acak, membandingkan Value Iteration sebagai acuan optimal dengan Q-Learning hasil eksplorasi, serta menyusun analisis *learning curve* aktual dan risiko *reward design*.

### Rubrik Penilaian (100 Poin)
| Kriteria Asesmen | Bobot | Indikator Penilaian Maksimal |
| :--- | :---: | :--- |
| **1. Problem & Environment Design** | **20** | Simulator GridWorld kustom berfungsi penuh, mendukung reset/step, dan mendokumentasikan tata letak state/action/reward. |
| **2. Mathematical Formulation** | **20** | Formulasi tuple MDP valid, perhitungan manual *return* dan pembaruan Bellman cocok dengan verifikasi NumPy. |
| **3. Algorithm Implementation** | **30** | Implementasi Value Iteration dan Q-Learning akurat dengan penanganan terminal state tepat serta pemisahan $\epsilon = 0$ saat evaluasi. |
| **4. Evaluation & Interpretation** | **20** | Tabel komparasi multi-run vs baseline acak, kurva pembelajaran asli, dan analisis risiko *reward hacking*. |
| **5. Documentation & Reproducibility** | **10** | Kode bersih, penetapan random seed jelas, dan berjalan 100% deterministik pada CPU lokal. |

---

## REFERENSI AKADEMIK TERVERIFIKASI

1. **Sutton, R. S., & Barto, A. G. (2018).** *Reinforcement Learning: An Introduction (2nd Edition).* The MIT Press. https://mitpress.mit.edu/9780262039246/reinforcement-learning/
2. **Bellman, R. (1957).** *Dynamic Programming.* Princeton University Press.
3. **Watkins, C. J. C. H., & Dayan, P. (1992).** *Q-learning.* Machine Learning, 8(3–4), 279–292. https://doi.org/10.1007/BF00992698
4. **Williams, R. J. (1992).** *Simple statistical gradient-following algorithms for connectionist reinforcement learning.* Machine Learning, 8(3–4), 229–256. https://doi.org/10.1007/BF00992696
5. **Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J., Chanan, G., ... & Chintala, S. (2019).** *PyTorch: An Imperative Style, High-Performance Deep Learning Library.* Advances in Neural Information Processing Systems (NeurIPS 2019), 32.
