# Chapter 04: Search Algorithms: Uninformed & Informed Search

> **Tingkat Keahlian**: Level 1 — AI Foundations  
> **Jalur Spesialisasi**: Core Curriculum  
> **Prasyarat**: Chapter 03 (Intelligent Agents & AI Problem Solving)  
> **Format Pembelajaran**: Modul Bacaan Teori, Analisis Kompleksitas Matematika, Praktik Python Mandiri, dan Mini Project Evaluasi Algoritma  

---

# PART A — SEARCH FOUNDATIONS

## 1. Review Problem Formulation dan Ruang Keadaan

Pada Chapter 03, kita telah mempelajari bahwa pemecahan masalah cerdas (*AI Problem Solving*) diawali dengan proses abstraksi formal yang memetakan dunia nyata ke dalam lima komponen baku:
1. **Initial State ($s_0$)**: Keadaan awal tempat agen memulai eksplorasi.
2. **Actions ($A(s)$)**: Himpunan tindakan legal yang sah dieksekusi dari keadaan $s$.
3. **Transition Model ($\text{Result}(s, a)$)**: Fungsi deterministik yang memetakan keadaan saat ini dan tindakan ke keadaan baru $s'$.
4. **Goal Test ($\text{is\_goal}(s)$)**: Uji predikat boolean penentu keberhasilan tujuan.
5. **Path Cost ($c(s, a, s')$)**: Akumulasi biaya langkah di sepanjang jalur.

Kelima elemen ini secara implisit mendefinisikan **Ruang Keadaan (*State Space*)**, yaitu semesta seluruh keadaan fisik yang dapat dicapai dari $s_0$ melalui urutan tindakan legal apa pun. Ruang keadaan membentuk sebuah graf berarah terstruktur (*state space graph*) di mana simpul melambangkan keadaan fisik dunia (*physical state*), dan sisi (*edge*) melambangkan transisi tindakan.

---

## 2. Search Tree vs Search Graph

Pembedaan antara struktur masalah dengan struktur pencarian adalah salah satu konsep paling fundamental dalam AI:

```
+------------------------------------+      +------------------------------------+
|         STATE SPACE GRAPH          |      |            SEARCH TREE             |
|                                    |      |                                    |
|              (Arad)                |      |               [Arad]               |
|              /    \                |      |              /   |  \              |
|         (Sibiu)---(Zerind)         |      |        [Sibiu] [Timisoara] [Zerind]|
|              |                     |      |        /  |  \                     |
|          (Fagaras)                 |      |     [Arad][Fagaras][R.Vilcea]      |
|                                    |      |       ^                            |
| * Terdapat siklus (cycles)         |      | * Cabang dapat berulang tak hingga |
| * Setiap keadaan hanya muncul 1x   |      |   jika siklus tidak diproteksi     |
+------------------------------------+      +------------------------------------+
```

- **Search Tree (Pohon Pencarian)**: Pohon eksplorasi yang dibangun oleh algoritma pencarian. Akarnya adalah *initial state*, dan cabangnya adalah tindakan. Dalam pohon pencarian, **sebuah keadaan fisik yang sama dapat muncul berkali-kali di cabang yang berbeda** jika terdapat banyak jalur untuk mencapai keadaan tersebut.
- **Search Graph (Graf Pencarian)**: Mengatasi redundansi pohon pencarian dengan memastikan bahwa setiap keadaan fisik yang telah dieksplorasi tidak akan diproses ulang, mencegah algoritma terjebak dalam *infinite loop* pada graf bersiklik.

---

## 3. Search Node vs Physical State

Pemula sering kali mencampuradukkan antara *Node* pencarian dan *State* fisik. Keduanya adalah entitas yang berbeda:

$$\text{Physical State} \neq \text{Search Node}$$

- **State**: Konfigurasi fisik dunia (contoh: koordinat $(r, c)$ pada grid, atau nama kota `"Arad"`). State tidak memiliki informasi mengenai bagaimana agen sampai ke sana.
- **Node**: Struktur data akuntansi internal yang digunakan oleh algoritma pencarian untuk merekonstruksi jalur solusi. Sebuah Node memuat:
  1. `state`: Keadaan fisik yang diwakilinya.
  2. `parent`: Pointer menuju simpul induk yang membangkitkannya.
  3. `action`: Tindakan yang dieksekusi oleh simpul induk untuk mencapai simpul ini.
  4. `path_cost` ($g(n)$): Akumulasi biaya total dari root hingga node ini.
  5. `depth`: Kedalaman simpul pada pohon pencarian.

---

## 4. Frontier dan Explored Set

Dua struktur data inti pengatur proses pencarian graf (*Graph Search*):

1. **Frontier (Open List)**: Struktur data antrean yang menyimpan himpunan seluruh simpul daun (*leaf nodes*) yang telah digenerate tetapi belum diekspansi. Pilihan struktur data pada *frontier* menentukan identitas algoritma (FIFO Queue = BFS, LIFO Stack = DFS, Priority Queue = UCS / A\*).
2. **Explored Set (Closed List)**: Himpunan (*hash set*) yang mencatat seluruh keadaan fisik yang telah diekspansi. Explored set menjamin bahwa algoritma tidak akan pernah mengekspansi keadaan yang sama dua kali.

---

## 5. Tree Search vs Graph Search

Algoritma pencarian generik dapat dijalankan dalam dua varian:

```text
Fungsi GRAPH-SEARCH(problem):
    frontier = antrean_prioritas(Node(problem.initial_state))
    explored = himpunan_kosong()
    loop:
        jika frontier kosong: kembalikan FAILURE
        node = frontier.pop()
        jika problem.is_goal(node.state): kembalikan SOLUSI(node)
        tambahkan node.state ke explored
        untuk setiap action di problem.actions(node.state):
            child = Child-Node(problem, node, action)
            jika child.state bukan di explored dan bukan di frontier:
                frontier.push(child)
            selain itu jika frontier mengizinkan pembaruan jalur lebih murah (UCS/A*):
                perbarui child di frontier
```

- **Tree Search**: Tidak memelihara `explored set`. Sangat hemat memori, namun jika graf masalah memiliki siklus (*cycles*), Tree Search akan terjebak dalam loop tak berhingga (*infinite loops*).
- **Graph Search**: Memelihara `explored set`. Menjamin terminasi pada ruang keadaan berhingga (*finite state spaces*), namun membutuhkan memori untuk menyimpan seluruh simpul yang pernah dikunjungi.

---

## 6. Generated Nodes vs Expanded Nodes

Pembedaan metrik ini krusial saat menganalisis efisiensi algoritma:
- **Generated Nodes**: Jumlah total simpul yang pernah diciptakan dan dimasukkan ke dalam memori (baik yang masuk frontier maupun yang langsung diuji goal).
- **Expanded Nodes**: Jumlah simpul yang dikeluarkan dari frontier dan diperiksa seluruh anak tetangganya. Simpul yang hanya berada di frontier tanpa pernah di-pop **tidak dihitung** sebagai simpul yang diekspansi.

---

# PART B — UNINFORMED SEARCH (BLIND SEARCH)

Algoritma pencarian tanpa informasi (*Uninformed Search*) hanya memiliki akses ke definisi formal masalah ($s_0$, $A(s)$, $Result$, $GoalTest$, $StepCost$), tanpa memiliki petunjuk tambahan seberapa dekat suatu simpul non-tujuan terhadap status akhir.

---

## 7. Breadth-First Search (BFS)

### Konsep dan Mekanisme
Breadth-First Search mengeksplorasi ruang keadaan lapis demi lapis (*shallowest unexpanded nodes first*). Simpul pada kedalaman $d$ akan diekspansi seluruhnya sebelum simpul pada kedalaman $d+1$ disentuh.

- **Struktur Data**: Antrean *First-In, First-Out* (FIFO Queue) menggunakan `collections.deque`.
- **Early Goal Test**: Pada BFS graf berbobot seragam (*unweighted graph*), uji tujuan (*Goal Test*) dilakukan saat **simpul anak digenerate**, bukan saat di-pop. Pengoptimalan ini menghemat ekspansi seluruh cabang pada level kedalaman solusi $d$.
- **Sifat Teoretis**:
  - *Completeness*: Lengkap (pasti menemukan solusi jika faktor percabangan $b$ berhingga).
  - *Optimality*: Optimal terhadap **jumlah langkah/hop paling sedikit**. Hanya optimal terhadap biaya jika seluruh step cost bernilai seragam identik ($c(s, a, s') = c$).
  - *Time Complexity*: $\mathcal{O}(b^d)$.
  - *Space Complexity*: $\mathcal{O}(b^d)$ (seluruh simpul level terdalam disimpan di memori frontier). **Kelemahan terbesar BFS adalah konsumsi memorinya yang eksponensial**.

---

## 8. Depth-First Search (DFS)

### Konsep dan Mekanisme
Depth-First Search mengeksplorasi cabang terdalam dari pohon pencarian terlebih dahulu (*deepest unexpanded nodes first*), dan hanya melakukan *backtracking* saat menemui jalan buntu.

- **Struktur Data**: Antrean *Last-In, First-Out* (LIFO Stack) menggunakan list Python standar (`append()` dan `pop()`).
- **Sifat Teoretis**:
  - *Completeness*: Tidak lengkap pada ruang keadaan tak berhingga atau jika terdapat siklus pada Tree Search. Lengkap pada graf berhingga dengan Graph Search.
  - *Optimality*: **Tidak optimal**. DFS sering kali menemukan jalur solusi yang sangat panjang dan memutar sebelum sempat memeriksa jalur pendek di cabang sebelahnya.
  - *Time Complexity*: $\mathcal{O}(b^m)$, di mana $m$ adalah kedalaman maksimum ruang keadaan.
  - *Space Complexity*: $\mathcal{O}(b \cdot m)$ pada Tree Search (hanya menyimpan jalur aktif dari root ke daun beserta saudara kandungnya). Memori linier ini adalah keunggulan utama DFS.

---

## 9. Depth-Limited Search (DLS)

Depth-Limited Search adalah DFS yang dilengkapi batas kedalaman artifisial $L$ (*depth limit*). Simpul pada kedalaman $L$ diperlakukan seolah-olah tidak memiliki anak tetangga.

### Membedakan Cutoff dan Failure
Implementasi DLS yang benar wajib membedakan dua kondisi kegagalan:
1. **CUTOFF**: Solusi tidak ditemukan di dalam batas limit $L$, tetapi proses pencarian terpotong karena masih ada cabang yang lebih dalam.
2. **FAILURE**: Seluruh ruang keadaan telah dieksplorasi habis di dalam limit $L$ dan terbukti bahwa tujuan memang tidak ada di seluruh ruang keadaan.

---

## 10. Iterative Deepening Search (IDDFS)

Iterative Deepening Search memadukan keunggulan efisiensi ruang linier DFS dengan kelengkapan dan optimalitas langkah BFS.

```text
Limit = 0: [Arad]
Limit = 1: [Arad] -> [Sibiu], [Timisoara], [Zerind]
Limit = 2: [Arad] -> [Sibiu] -> [Fagaras], [Rimnicu Vilcea] ...
Limit = 3: ... -> [Bucharest] (Goal Ditemukan!)
```

Algoritma menjalankan DLS secara berulang dengan limit yang dinaikkan bertahap: $L = 0, 1, 2, 3, \dots, d$.

### Analisis Overhead Komputasi IDDFS
Banyak pemula menduga IDDFS sangat boros karena mengulang generasi simpul level atas berkali-kali. Namun, secara matematis pada pohon dengan faktor percabangan $b > 1$, sebagian besar simpul selalu berada pada level daun terbawah:

$$N(\text{IDDFS}) = (d)b^1 + (d-1)b^2 + \dots + (1)b^d \in \mathcal{O}(b^d)$$

Sebagai contoh pada $b = 10$ dan $d = 5$:
- BFS membangkitkan $10 + 100 + 1.000 + 10.000 + 100.000 = 111.110$ simpul.
- IDDFS membangkitkan $5(10) + 4(100) + 3(1.000) + 2(10.000) + 1(100.000) = 123.450$ simpul.
Overhead pengulangan hanya $\approx 11\%$, namun IDDFS menghemat memori dari $100.000$ simpul menjadi hanya $50$ simpul di memori!

---

## 11. Uniform-Cost Search (UCS / Dijkstra)

Jika BFS mencari rute dengan langkah paling sedikit, **Uniform-Cost Search mencari rute dengan akumulasi biaya numerik terendah ($g(n)$)** pada graf berbobot.

- **Struktur Data**: Priority Queue (`heapq`) dengan kunci pengurutan $g(n)$ (biaya dari start ke node $n$).
- **Aturan Krusial UCS**:
  1. **Goal Test Saat Pop**: Uji tujuan **wajib** dilakukan saat simpul di-pop dari frontier, bukan saat digenerate! Menguji tujuan saat generasi akan merusak optimalitas jika jalur pertama yang mencapai tujuan ternyata berbiaya mahal.
  2. **Better Path Updates**: Jika keadaan yang sama ditemukan kembali dengan biaya $g(n)$ yang lebih murah, entri biaya di frontier diperbarui (*cost relaxation*).
- **Sifat Teoretis**:
  - *Completeness*: Lengkap jika setiap biaya langkah $\ge \epsilon > 0$.
  - *Optimality*: **Optimal terhadap biaya minimum sejati**.
  - *Kompleksitas*: $\mathcal{O}(b^{1 + \lfloor C^* / \epsilon \rfloor})$, di mana $C^*$ adalah biaya solusi optimal dan $\epsilon$ adalah biaya langkah terkecil.

---

# PART C — INFORMED (HEURISTIC) SEARCH

Pencarian terinformasi (*Informed Search*) memanfaatkan **Fungsi Heuristik ($h(n)$)**, yaitu pengetahuan khusus domain (*domain-specific knowledge*) yang memperkirakan biaya termurah dari simpul $n$ menuju status tujuan:

$$h(n) = \text{Estimasi biaya dari } n \text{ ke Goal}$$

Sifat baku heuristik: $h(\text{Goal}) = 0$, dan $h(n) \ge 0$ untuk setiap simpul $n$.

---

## 12. Admissibility dan Consistency

Dua kriteria matematis penentu keabsahan fungsi heuristik:

### 1. Admissible Heuristic (Heuristik yang Dapat Diterima)
Sebuah heuristik $h(n)$ disebut **Admissible** jika ia tidak pernah melebih-lebihkan biaya riil minimum menuju tujuan (*never overestimates the true cost*):

$$0 \le h(n) \le h^*(n), \quad \forall n$$

di mana $h^*(n)$ adalah biaya riil terpendek yang sesungguhnya dari $n$ menuju goal. Heuristik admissible selalu bersifat optimis.

### 2. Consistent Heuristic (Heuristik Konsisten / Monoton)
Sebuah heuristik $h(n)$ disebut **Consistent** (atau *Monotone*) jika untuk setiap simpul $n$ dan setiap penerus $n'$ yang dihasilkan oleh tindakan $a$, estimasi heuristik mematuhi **Ketaksamaan Segitiga (*Triangle Inequality*)**:

$$h(n) \le c(n, a, n') + h(n')$$

```
        (n)
       /   \
c(n,a,n')   \ h(n)
     v       v
   (n') ---> [Goal]
       h(n')
```

### Hubungan Admissibility dan Consistency
- Setiap heuristik yang **Consistent** pasti bersifat **Admissible** (dengan syarat $h(\text{Goal}) = 0$).
- Heuristik yang *Admissible* **belum tentu** bersifat *Consistent*, meskipun dalam domain praktis (seperti jarak garis lurus atau Manhattan distance pada grid) hampir seluruh heuristik alami bersifat konsisten.
- Pada **Graph Search**, konsistensi menjamin bahwa ketika sebuah simpul di-pop dari frontier, biaya jalur $g(n)$ yang dicapainya dijamin sudah optimal, sehingga simpul tersebut tidak perlu dibuka kembali (*no node reopening needed*).

---

## 13. Greedy Best-First Search

Greedy Best-First Search mengevaluasi simpul murni berdasarkan estimasi jarak ke tujuan:

$$f(n) = h(n)$$

- **Karakteristik**: Bertindak serakah dengan mengekspansi simpul yang tampak paling dekat dengan tujuan.
- **Kelemahan**: **Tidak optimal dan tidak lengkap**. Pada peta Romania, dari Arad algoritma memilih Sibiu lalu Fagaras karena $h(\text{Fagaras}) = 176 < h(\text{Rimnicu Vilcea}) = 193$. Akibatnya, Greedy menghasilkan jalur via Fagaras (450 km), melewatkan jalur optimal via Rimnicu Vilcea (418 km).

---

## 14. A\* Search

A\* Search diciptakan oleh Peter Hart, Nils Nilsson, dan Bertram Raphael (1968). A\* menggabungkan keunggulan UCS (yang memperhatikan biaya masa lalu $g(n)$) dengan keunggulan Greedy Search (yang memperhatikan estimasi masa depan $h(n)$):

$$f(n) = g(n) + h(n)$$

```
f(n) = [Biaya Nyata yang Telah Ditempuh g(n)] + [Estimasi Biaya Tersisa Menuju Goal h(n)]
```

### Kondisi Optimalitas A\*
1. Pada **Tree Search**: A\* dijamin optimal jika $h(n)$ bersifat **Admissible**.
2. Pada **Graph Search**: A\* dijamin optimal jika $h(n)$ bersifat **Consistent**.
3. Jika $h(n) = 0$ untuk seluruh simpul, maka $f(n) = g(n) + 0 = g(n)$, sehingga A\* berperilaku **setara persis dengan Uniform-Cost Search (UCS)**.

---

## 15. Desain Heuristik: Manhattan vs Euclidean Distance

Pada masalah navigasi grid 2D dengan titik saat ini $(x_1, y_1)$ dan target tujuan $(x_2, y_2)$:

- **Manhattan Distance ($L_1$ Norm)**:
  $$h_{\text{Manhattan}} = |x_1 - x_2| + |y_1 - y_2|$$
  Tepat dan konsisten jika pergerakan robot hanya diizinkan ke 4 arah mata angin (Up, Down, Left, Right).
- **Euclidean Distance ($L_2$ Norm)**:
  $$h_{\text{Euclidean}} = \sqrt{(x_1 - x_2)^2 + (y_1 - y_2)^2}$$
  Jarak garis lurus Euclidean selalu $\le$ Manhattan Distance. Euclidean selalu admissible untuk pergerakan 4 arah maupun pergerakan bebas bersudut kontinu.

---

# PART D — ALGORITHM ANALYSIS & COMPARISON

Tabel berikut merangkum analisis teoretis komparatif ketujuh algoritma pencarian:

| Algoritma | Sifat Frontier | Kelengkapan (*Complete*) | Optimalitas (*Optimal*) | Kompleksitas Waktu | Kompleksitas Ruang (Memori) |
|---|---|:---:|:---:|:---:|:---:|
| **BFS** | FIFO Queue | Ya (jika $b$ berhingga) | Ya (untuk unweighted step) | $\mathcal{O}(b^d)$ | $\mathcal{O}(b^d)$ (Eksponensial) |
| **DFS** | LIFO Stack | Tidak (Tree) / Ya (Graph berhingga) | Tidak | $\mathcal{O}(b^m)$ | $\mathcal{O}(b \cdot m)$ (Linier) |
| **DLS** | Recursive / Stack | Ya jika $L \ge d$ | Tidak | $\mathcal{O}(b^L)$ | $\mathcal{O}(b \cdot L)$ (Linier) |
| **IDDFS** | Iterative DLS | Ya (jika $b$ berhingga) | Ya (untuk unweighted step) | $\mathcal{O}(b^d)$ | $\mathcal{O}(b \cdot d)$ (Linier) |
| **UCS** | Priority Queue ($g$) | Ya (jika $c \ge \epsilon > 0$) | **Ya (Biaya Minimum)** | $\mathcal{O}(b^{1 + \lfloor C^* / \epsilon \rfloor})$ | $\mathcal{O}(b^{1 + \lfloor C^* / \epsilon \rfloor})$ |
| **Greedy** | Priority Queue ($h$) | Tidak | Tidak | $\mathcal{O}(b^m)$ | $\mathcal{O}(b^m)$ |
| **A\*** | Priority Queue ($g+h$) | Ya | **Ya (Biaya Minimum)** | $\mathcal{O}(b^{\epsilon \cdot d})$ (Efisien) | $\mathcal{O}(b^d)$ (Eksponensial) |

*Keterangan parameter*: $b$ = faktor percabangan (*branching factor*), $d$ = kedalaman solusi dangkal optimal, $m$ = kedalaman maksimum ruang keadaan, $C^*$ = biaya solusi optimal, $\epsilon$ = biaya langkah terkecil.

---

# PART E — APPLICATIONS & EXPERIMENTAL BENCHMARKS

## 16. Benchmark 1: Peta Graf Berbobot Romania (Arad $\to$ Bucharest)

Masalah navigasi klasik Russell & Norvig: menemukan jalur terpendek dari kota Arad menuju ibu kota Bucharest di Romania.

```
       [Arad] --- 140 --- [Sibiu] --- 99 --- [Fagaras] --- 211 --- [Bucharest]
          |                  |                                          ^
         118                 80                                         |
          v                  v                                         101
      [Timisoara]      [Rimnicu Vilcea] --- 97 --- [Pitesti] -----------+
```

### Hasil Eksperimen Nyata (Python 3.12.10)

| Algoritma | Rute yang Ditemukan | Jarak Total | Simpul Diekspansi | Keterangan Evaluasi |
|---|---|:---:|:---:|---|
| **BFS** | Arad $\to$ Sibiu $\to$ Fagaras $\to$ Bucharest | 450 km | 5 node | Menemukan rute 3 hop (bukan jarak terpendek) |
| **DFS** | Arad $\to$ Sibiu $\to$ Fagaras $\to$ Bucharest | 450 km | 4 node | Tidak menjamin optimalitas |
| **UCS** | Arad $\to$ Sibiu $\to$ R.Vilcea $\to$ Pitesti $\to$ Bucharest | **418 km** | 13 node | **Rute optimal sejati**, mengeksplorasi 13 node |
| **Greedy** | Arad $\to$ Sibiu $\to$ Fagaras $\to$ Bucharest | 450 km | **4 node** | Cepat, tapi terjebak heuristik Fagaras |
| **A\*** | Arad $\to$ Sibiu $\to$ R.Vilcea $\to$ Pitesti $\to$ Bucharest | **418 km** | **6 node** | **Optimal sejati**, ekspansi separuh dari UCS! |

---

## 17. Ringkasan Modul

- Masalah pencarian dimodelkan menggunakan abstraksi formal $s_0$, $A(s)$, $\text{Result}(s, a)$, $\text{is\_goal}(s)$, dan $c(s, a, s')$.
- **BFS** menjamin solusi langkah tersedikit pada unweighted graph, namun rakus memori eksponensial $\mathcal{O}(b^d)$.
- **DFS** hemat memori linier $\mathcal{O}(b \cdot m)$, namun tidak menjamin kelengkapan maupun optimalitas.
- **IDDFS** adalah algoritma pencarian tanpa informasi terbaik untuk masalah berbobot seragam karena menggabungkan memori linier DFS dengan kelengkapan/optimalitas BFS.
- **UCS** menjamin biaya kumulatif terendah pada sembarang graf berbobot positif, dengan melakukan goal test saat node di-pop dari frontier.
- **Heuristik Admissible** ($0 \le h(n) \le h^*(n)$) dan **Consistent** ($h(n) \le c + h(n')$) adalah syarat mutlak jaminan optimalitas pada **A\* Search**.
- **A\* Search** adalah standar emas pencarian graf terinformasi: ia mengekspansi simpul paling minimal di antara seluruh algoritma pencarian optimal yang beroperasi dengan heuristik yang sama.

---

## 18. Latihan Terstruktur

### Latihan Tingkat Dasar (Konseptual)
1. Jelaskan perbedaan mendasar antara *Search Node* dan *Physical State*! Mengapa sebuah keadaan fisik yang sama dapat diwakili oleh lebih dari satu simpul pencarian?
2. Pada algoritma BFS unweighted, mengapa pengujian tujuan (*Goal Test*) dapat dilakukan pada saat simpul digenerate (*early goal test*), sedangkan pada UCS pengujian tujuan wajib dilakukan saat simpul di-pop dari frontier?
3. Buktikan secara matematis mengapa heuristik jarak garis lurus (*Straight-Line Distance*) pada bidang 2D selalu memenuhi sifat *Admissibility*!

### Latihan Tingkat Menengah (Penelusuran Algoritma)
4. Diberikan sebuah graf pencarian kecil:
   - Start: $S$, Goal: $G$.
   - Sisi dan biaya: $S \to A$ (biaya 2), $S \to B$ (biaya 5), $A \to C$ (biaya 4), $B \to C$ (biaya 1), $C \to G$ (biaya 3).
   - Heuristik: $h(S)=6, h(A)=5, h(B)=3, h(C)=2, h(G)=0$.
   - **Tugas**:
     a) Tuliskan urutan ekspansi simpul dan jalur yang ditemukan oleh **Uniform-Cost Search (UCS)**!
     b) Tuliskan urutan ekspansi simpul dan jalur yang ditemukan oleh **A\* Search**!
     c) Bandingkan total biaya jalur keduanya!

### Latihan Tingkat Lanjut (Analisis Heuristik)
5. Misalkan seorang insinyur AI merancang heuristik $h(n)$ di mana $h(n) = h^*(n) + 2$ untuk setiap simpul non-tujuan, dan $h(\text{Goal}) = 0$.
   - Apakah heuristik tersebut *admissible*?
   - Apakah A\* dengan heuristik tersebut dijamin menghasilkan jalur optimal? Berikan skenario pembuktian atau penyangkalannya (*counterexample*)!

---

## 19. Pembahasan dan Panduan Jawaban Latihan

### Pembahasan Latihan Tingkat Dasar
1. **Node vs State**: *State* adalah deskripsi dunia fisik (misal kota "Sibiu"). *Node* adalah struktur data pencarian yang membungkus state beserta konteks jalur historisnya (`parent`, `action`, `path_cost`, `depth`). Jika kota Sibiu dapat dicapai dari Arad (biaya 140) dan dari Oradea (biaya 222), maka di pohon pencarian terdapat dua Node berbeda yang mewakili State fisik yang sama ("Sibiu").
2. **Early Goal Test pada BFS vs UCS**: Pada BFS unweighted, setiap langkah berbiaya sama (1 hop). Begitu simpul anak goal digenerate pada kedalaman $d$, tidak mungkin ada simpul goal lain di level yang lebih dangkal. Namun pada UCS, simpul goal yang pertama kali digenerate bisa saja melalui sisi jalan pintas yang berbiaya sangat besar. Jalur yang lebih murah mungkin sedang mengantre di frontier pada kedalaman lebih dalam. Oleh karena itu, UCS wajib menahan evaluasi goal hingga simpul tersebut benar-benar memiliki nilai $g(n)$ terkecil di seluruh frontier.
3. **Bukti Admissibility Straight-Line Distance**: Berdasarkan aksioma geometri Euclidean, jarak garis lurus antara dua titik adalah panjang lintasan terpendek yang mungkin menghubungkan kedua titik tersebut. Jalan raya fisik di dunia nyata selalu meliuk, berkelok, atau terhalang kontur bumi, sehingga panjang jalan riil $h^*(n)$ selalu $\ge$ jarak garis lurus $h_{\text{SLD}}(n)$. Karena $0 \le h_{\text{SLD}}(n) \le h^*(n)$, maka heuristik garis lurus terbukti *admissible*.

### Pembahasan Latihan Tingkat Menengah
4. **Penelusuran Graf $S \to G$**:
   - Jalur yang mungkin:
     - Jalur 1: $S \to A \to C \to G$, biaya $= 2 + 4 + 3 = 9$.
     - Jalur 2: $S \to B \to C \to G$, biaya $= 5 + 1 + 3 = 9$.
   - a) **UCS**:
     - Pop $S$ ($g=0$). Masukkan $A$ ($g=2$), $B$ ($g=5$).
     - Pop $A$ ($g=2$). Masukkan $C$ via $A$ ($g=6$).
     - Pop $B$ ($g=5$). Ditemukan jalur ke $C$ via $B$ dengan $g=6$ (biaya sama).
     - Pop $C$ ($g=6$). Masukkan $G$ ($g=9$).
     - Pop $G$ ($g=9$). Goal tercapai! Jalur: $S \to A \to C \to G$ (atau $S \to B \to C \to G$), biaya $= 9$.
   - b) **A\* Search**:
     - Root $S$: $f(S) = 0 + 6 = 6$. Pop $S$.
     - Anak $A$: $g=2, h=5 \implies f(A) = 7$.
     - Anak $B$: $g=5, h=3 \implies f(B) = 8$.
     - Pop $A$ ($f=7$). Anak $C$: $g=6, h=2 \implies f(C) = 8$.
     - Frontier berisi $B$ ($f=8$) dan $C$ ($f=8$). Tie-breaking memilih $B$.
     - Pop $B$ ($f=8$). Jalur ke $C$ memiliki $g=6$, tidak memperbaiki $C$.
     - Pop $C$ ($f=8$). Anak $G$: $g=9, h=0 \implies f(G) = 9$.
     - Pop $G$ ($f=9$). Goal tercapai!
   - c) Kedua algoritma menemukan solusi optimal dengan total biaya yang sama, yaitu 9 unit.

### Pembahasan Latihan Tingkat Lanjut
5. **Heuristik $h(n) = h^*(n) + 2$**:
   - Heuristik ini **TIDAK admissible** karena untuk setiap simpul non-tujuan, nilainya melebih-lebihkan biaya riil sebesar 2 unit ($h(n) > h^*(n)$).
   - A\* dengan heuristik ini **tidak dijamin optimal**. Contoh penyangkal: Misalkan terdapat dua jalur dari $S$ ke $G$: Jalur Suboptimal $S \to G$ dengan biaya 10, dan Jalur Optimal $S \to A \to G$ dengan biaya $c(S, A)=1, c(A, G)=8$ (total biaya 9). Pada start, $h^*(A) = 8$, sehingga $h(A) = 10 \implies f(A) = 1 + 10 = 11$. Sementara $G$ langsung memiliki $f(G) = 10 + 0 = 10$. Algoritma A\* akan langsung memilih simpul $G$ ($f=10$) dan mengabaikan simpul $A$ ($f=11$), menghasilkan solusi suboptimal berbiaya 10.

---

## 20. Tugas Mini Project: Comparative Analysis of Search Algorithms for Pathfinding

### Deskripsi Tugas
Mahasiswa diminta melakukan studi komparatif dan eksperimental menyeluruh terhadap performa algoritma pencarian kecerdasan buatan dalam menyelesaikan masalah pencarian rute (*pathfinding*). Mahasiswa akan memformulasikan masalah graf/grid kustom, menjalankan algoritma, mengumpulkan metrik kinerja, dan menyusun analisis kritis.

### Ruang Lingkup dan 9 Deliverables yang Dinilai
Laporan disusun dalam format laporan teknis akademik (5–8 halaman) dengan memuat:
1. **Problem Formulation**: Formulasi formal 5 elemen masalah ($s_0$, $A(s)$, $\text{Result}$, $\text{GoalTest}$, $c$).
2. **Definisi Lingkungan Uji**: Peta graf berbobot minimal 10 simpul atau peta grid 2D minimal $6 \times 6$ dengan rintangan non-trivial.
3. **Implementasi Algoritma**: Minimal 3 algoritma yang berbeda, wajib memuat minimal 1 algoritma Uninformed (BFS/UCS) dan 1 algoritma Informed (A\*).
4. **Formulasi Heuristik**: Perumusan fungsi heuristik yang digunakan dan pembuktian sifat *admissibility*-nya.
5. **Correctness Tests**: Pengujian kasus uji batas (start = goal, tujuan tidak terjangkau, graf bersiklik).
6. **Visualisasi Rute**: Gambar/diagram jalur yang dihasilkan oleh masing-masing algoritma.
7. **Tabel Perbandingan Performa Empiris**: Membandingkan panjang jalur, total biaya, simpul diekspansi, simpul digenerate, dan waktu eksekusi (ms).
8. **Analisis Kritis & Batasan**: Menjelaskan mengapa satu algoritma mengungguli algoritma lain pada metrik tertentu.
9. **Daftar Pustaka Akademik**: Minimal 2 rujukan primer terverifikasi.

---

## 21. Rubrik Penilaian Proyek (Total 100 Poin)

| Kriteria Penilaian | Bobot | Kinerja Unggul (Penuh) | Kinerja Cukup (Parsial) | Kinerja Kurang (Tidak Terpenuhi) |
|---|---:|---|---|---|
| **Problem Formulation & Heuristic** | **15 Poin** | Formulasi 5 elemen matematis presisi; heuristik dirumuskan formal dan terbukti admissible. (14–15) | Formulasi cukup lengkap; heuristik didefinisikan namun bukti admissibility lemah. (8–13) | Formulasi salah atau tidak lengkap; heuristik tidak logis. (0–7) |
| **Implementasi Kode Algoritma** | **30 Poin** | Kode modular, runnable tanpa error, mengimplementasikan $\ge 3$ algoritma (termasuk A\*), penanganan frontier tepat. (26–30) | Kode berjalan namun kaku, terdapat kelemahan penanganan stale priority entries. (15–25) | Kode error/tidak runnable, salah menerapkan logika antrean atau evaluasi. (0–14) |
| **Correctness & Boundary Testing** | **20 Poin** | Menguji kasus batas (siklus, unreachable, start=goal), jalur yang dihasilkan terbukti valid dan optimal. (17–20) | Pengujian hanya pada kasus normal, belum menguji kasus batas atau unreachable. (10–16) | Tidak ada pengujian formal, jalur yang dihasilkan tidak valid. (0–9) |
| **Performance Analysis & Benchmarking** | **20 Poin** | Tabel metrik komparatif komprehensif (cost, expanded, generated, frontier, time) dengan analisis tajam. (17–20) | Tabel metrik ada namun analisis hanya mengulang angka tanpa interpretasi teoretis. (10–16) | Tidak ada tabel metrik atau analisis performa dangkal. (0–9) |
| **Dokumentasi & Referensi Akademik** | **15 Poin** | Laporan teknis rapi, visualisasi jalur jelas, menyertakan minimal 2 rujukan primer terverifikasi. (13–15) | Format cukup rapi, visualisasi ada namun kurang informatif, referensi kurang. (7–12) | Laporan tidak terstruktur, tanpa visualisasi, terindikasi plagiarisme. (0–6) |

---

## 22. Referensi dan Sumber Belajar Primer

1. **Russell, S., & Norvig, P. (2020)**. *Artificial Intelligence: A Modern Approach (4th ed.)*. Pearson. (Bab 3: *Solving Problems by Searching*). URL: [https://aima.cs.berkeley.edu/](https://aima.cs.berkeley.edu/)
2. **Hart, P. E., Nilsson, N. J., & Raphael, B. (1968)**. *A Formal Basis for the Heuristic Determination of Minimum Cost Paths*. IEEE Transactions on Systems Science and Cybernetics, 4(2), 100–107. DOI: [https://doi.org/10.1109/TSSC.1968.300136](https://doi.org/10.1109/TSSC.1968.300136)
3. **Korf, R. E. (1985)**. *Depth-First Iterative-Deepening: An Optimal Admissible Tree Search*. Artificial Intelligence, 27(1), 97–109.
4. **Dijkstra, E. W. (1959)**. *A Note on Two Problems in Connexion with Graphs*. Numerische Mathematik, 1, 269–271.
5. **ACM / IEEE-CS / AAAI (2024)**. *Computer Science Curricula 2023 (CS2023): Artificial Intelligence Knowledge Area*. ACM Press. URL: [https://csed.acm.org/](https://csed.acm.org/)

