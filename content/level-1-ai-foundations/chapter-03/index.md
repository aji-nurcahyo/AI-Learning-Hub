# Chapter 03: Intelligent Agents & AI Problem Solving

> **Tingkat Keahlian**: Level 1 — AI Foundations  
> **Jalur Spesialisasi**: Core Curriculum  
> **Prasyarat**: Chapter 01 (Introduction to Artificial Intelligence), Chapter 02 (History, Evolution & AI Paradigms)  
> **Format Pembelajaran**: Modul Bacaan Teori, Latihan Analisis Terstruktur, Praktik Python Mandiri, dan Tugas Perancangan Agen Cerdas  

---

## 1. Pengantar Intelligent Agents

Dalam buku teks standar dunia *Artificial Intelligence: A Modern Approach*, Stuart Russell dan Peter Norvig mendefinisikan Artificial Intelligence sebagai **studi dan perancangan agen rasional (*rational agents*)**. Pendekatan berbasis agen (*agent-based view*) ini menyatukan berbagai cabang kecerdasan buatan—mulai dari logika simbolik, sistem pakar, visi komputer, hingga pemelajaran mesin probabilistik—ke dalam satu kerangka kerja konseptual yang utuh: **bagaimana sebuah entitas komputasi dapat mempersepsikan lingkungannya dan mengambil tindakan yang paling tepat untuk mencapai tujuannya**.

Pendekatan ini bergeser dari sekadar membangun algoritma terisolasi menjadi merancang sistem otonom yang berinteraksi secara berkelanjutan dengan dunia luar. Agen tidak hanya berpikir, tetapi juga bertindak dan menanggung konsekuensi atas tindakannya di dalam lingkungan tempat ia beroperasi.

---

## 2. Konsep Agent dan Environment

Secara fundamental, **Agent** adalah segala sesuatu yang dapat dipandang sebagai pengamat lingkungannya melalui serangkaian **Sensors** dan pengubah lingkungannya melalui serangkaian **Actuators**.

```
+-------------------------------------------------------------+
|              AGENT - ENVIRONMENT INTERACTION LOOP           |
|                                                             |
|                    +-------------------+                    |
|                    |    ENVIRONMENT    |                    |
|                    +-------------------+                    |
|                       /             ^                       |
|           Percepts   /               \   Actions            |
|                     v                 \                     |
|              +-----------+       +-----------+              |
|              |  SENSORS  |       | ACTUATORS |              |
|              +-----------+       +-----------+              |
|                    \                 /                      |
|                     v               /                       |
|                    +-----------------+                      |
|                    |      AGENT      |                      |
|                    +-----------------+                      |
+-------------------------------------------------------------+
```

Interaksi ini berlangsung dalam sebuah siklus berkesinambungan:
1. Lingkungan berada pada status (*state*) fisik tertentu.
2. Sensor agen menerima sinyal dari lingkungan sebagai masukan perseptual (*percept*).
3. Agen memproses percept tersebut berdasarkan program logikanya untuk menentukan tindakan (*action*).
4. Aktuator agen mengeksekusi aksi tersebut ke lingkungan.
5. Lingkungan bertransisi ke status baru akibat aksi tersebut, dan siklus berulang kembali.

Contoh agen dalam kehidupan nyata:
- **Agen Manusia**: Memiliki mata, telinga, kulit, dan organ indra sebagai sensor; tangan, kaki, pita suara, dan gerakan tubuh sebagai aktuator.
- **Agen Robotika**: Memiliki kamera digital, sonar, LiDAR, sensor inframerah sebagai sensor; motor servo, roda, lengan gripper hidrolik sebagai aktuator.
- **Agen Perangkat Lunak (Software Agent / Bot)**: Menerima masukan penekanan tombol keyboard, paket data jaringan, berkas file sebagai sensor masukan; menampilkan grafik di layar, menulis berkas ke media penyimpanan, atau mengirim paket jaringan sebagai aktuator.

---

## 3. Sensors, Actuators, Percepts, dan Actions

Empat istilah teknis mendasar yang wajib dibedakan dengan tegas:

- **Percept**: Masukan perseptual yang diterima oleh sensor agen pada satu titik waktu instan tertentu.
- **Percept Sequence**: Riwayat lengkap (*complete history*) dari seluruh persepsi yang pernah diterima oleh agen sejak pertama kali dinyalakan hingga detik ini:

$$P^* = [p_1, p_2, p_3, \dots, p_t]$$

- **Sensors**: Perangkat keras atau antarmuka perangkat lunak yang bertugas menangkap sinyal fisik dari lingkungan dan mengubahnya menjadi data digital representasi keadaan.
- **Actions**: Himpunan keputusan atau gerakan diskret maupun kontinu yang dapat dieksekusi oleh agen ke lingkungan luar.
- **Actuators**: Mekanisme fisik atau fungsi sistem yang bertanggung jawab mengubah keputusan komputasi agen menjadi aksi nyata pada lingkungan.

---

## 4. Agent Function dan Agent Program

Pembedaan penting dalam teori agen cerdas adalah antara konsep abstrak matematis (**Agent Function**) dan perwujudan perangkat lunak nyatanya (**Agent Program**):

### Agent Function (Fungsi Matematis Abstrak)
Secara matematis, perilaku agen dapat digambarkan sebagai sebuah fungsi pemetaan dari riwayat persepsi (*percept sequence*) ke tindakan:

$$f: P^* \to A$$

Fungsi ini bersifat abstrak murni; ia mendefinisikan apa yang *seharusnya* dilakukan agen untuk setiap kemungkinan riwayat persepsi yang dapat terjadi, tanpa memedulikan bagaimana cara menghitungnya.

### Agent Program (Implementasi Komputasi Konkret)
Agent Program adalah implementasi algoritma nyata yang berjalan di atas arsitektur fisik tertentu (*physical architecture*):

$$\text{Agent} = \text{Architecture} + \text{Program}$$

Perbedaan krusialnya: **Agent Function menerima seluruh riwayat persepsi ($P^*$) sebagai domain teoritisnya**, sedangkan **Agent Program biasanya hanya menerima percept saat ini ($p_t$) sebagai masukan langsung**, karena menyimpan seluruh riwayat tak berhingga dalam memori fisik komputer adalah hal yang mustahil. Jika agen membutuhkan masa lalu, program agen harus membangun dan memperbarui status internal (*internal state*).

---

## 5. Rational Agents

Apa yang membuat sebuah agen disebut "Cerdas" atau "Rasional"?

> **Definisi Agen Rasional**:  
> Untuk setiap kemungkinan riwayat persepsi (*percept sequence*), agen rasional harus memilih sebuah tindakan yang diharapkan dapat **memaksimalkan ukuran kinerjanya (*performance measure*)**, berdasarkan bukti yang diberikan oleh riwayat persepsi dan pengetahuan bawaan apa pun yang dimiliki oleh agen tersebut.

### Membedakan Rasionalitas dari Konsep Serupa

| Konsep | Definisi | Perbedaan dengan Rasionalitas |
|---|---|---|
| **Rasionalitas (*Rationality*)** | Memaksimalkan *kinerja yang diharapkan* (*expected performance*) berdasarkan informasi yang tersedia saat itu. | Rasionalitas bergantung pada keputusan terbaik saat itu, bukan keberuntungan hasil akhir. |
| **Maha Tahu (*Omniscience*)** | Mengetahui hasil aktual dari tindakannya di masa depan secara mutlak 100%. | Mustahil di dunia nyata karena keterbatasan sensor dan ketidakpastian lingkungan. |
| **Kesempurnaan (*Perfection*)** | Menjamin hasil akhir yang selalu berhasil tanpa cela (*flawless outcome*). | Tidak realistis; agen rasional bisa saja gagal karena faktor acak lingkungan yang di luar kendalinya. |
| **Otonomi (*Autonomy*)** | Kemampuan agen untuk memelajari dan mengompensasi kekurangan pengetahuan bawaan dari pengalamannya sendiri. | Agen yang tidak otonom hanya bergantung pada aturan kaku yang ditanamkan pembuatnya sejak awal. |

*Contoh Pedagogis*: Seorang pejalan kaki menyeberang jalan saat lampu penyeberangan hijau setelah memeriksa kiri dan kanan secara saksama. Tiba-tiba sebuah pesawat jatuh menimpanya dari langit. Apakah tindakan pejalan kaki tersebut rasional? **Ya, sangat rasional**, karena ia telah memaksimalkan keselamatan berdasarkan bukti perseptual yang dapat dijangkaunya. Hasil akhir yang tragis tidak mengubah fakta bahwa keputusannya adalah rasional.

---

## 6. Performance Measures

**Performance Measure** (Ukuran Kinerja) adalah kriteria objektif eksternal yang digunakan untuk mengukur seberapa sukses sebuah agen dalam menyelesaikan tugasnya di lingkungan.

Prinsip Krusial dalam Mendesain Performance Measure:
- Ukuran kinerja harus dirancang berdasarkan **apa yang sebenarnya diinginkan di lingkungan**, bukan berdasarkan bagaimana agen berperilaku.
- *Peringatan Desain*: Jika agen pembersih vakum diberi skor "+1 setiap kali menyedot kotoran", agen yang licik dapat menyedot kotoran, membuangnya kembali ke lantai, dan menyedotnya lagi berulang-ulang untuk mengumpulkan skor tak terhingga. Ukuran kinerja yang benar adalah: *"kebersihan rata-rata seluruh lantai sepanjang waktu dikurangi konsumsi energi listrik"*.

---

## 7. PEAS Framework

Untuk merancang agen cerdas yang sistematis, langkah pertama yang wajib dilakukan oleh seorang perekayasa AI adalah mendefinisikan lingkungan tugas (*Task Environment*) menggunakan kerangka kerja **PEAS**:

1. **P — Performance Measure**: Kriteria keberhasilan objektif kuantitatif.
2. **E — Environment**: Karakteristik dunia tempat agen beroperasi.
3. **A — Actuators**: Perangkat/sarana untuk mengubah lingkungan.
4. **S — Sensors**: Perangkat/sarana untuk menangkap sinyal lingkungan.

### Contoh Analisis PEAS: Automated Taxi Driver

| Elemen PEAS | Komponen Konkret pada Automated Taxi |
|---|---|
| **Performance Measure** | Keselamatan penumpang (0 kecelakaan), kecepatan mencapai tujuan, efisiensi bahan bakar, kenyamanan penumpang (akselerasi halus), kepatuhan hukum lalu lintas, dan maksimalisasi keuntungan finansial. |
| **Environment** | Jalan raya perkotaan, jalan tol, cuaca hujan/terik, pejalan kaki tak terduga, lampu lalu lintas, polisi, kendaraan lain yang agresif. |
| **Actuators** | Kemudi setir (*steering*), pedal gas (*throttle*), sistem rem hidrolik, transmisi gigi, lampu sein/klakson, tampilan layar navigasi penumpang. |
| **Sensors** | Kamera video depan/belakang, sensor LiDAR, radar gelombang milimeter, GPS presisi tinggi, akselerometer, sensor kecepatan roda, mikrofon suara klakson. |

---

## 8. Karakteristik Task Environments

Kompleksitas desain sebuah agen cerdas tidak ditentukan oleh kecanggihan kodenya, melainkan oleh sifat lingkungan tempat ia bertugas. Russell & Norvig membagi lingkungan tugas ke dalam enam dimensi ortogonal:

```
+-------------------------------------------------------------------+
|               6 DIMENSI SIFAT TASK ENVIRONMENT                    |
|                                                                   |
|   1. Observabilitas   : Fully Observable    vs Partially Observed |
|   2. Determinisasi    : Deterministic       vs Stochastic         |
|   3. Episodik         : Episodic            vs Sequential         |
|   4. Kedinamisan      : Static              vs Dynamic            |
|   5. Diskrit/Kontinu  : Discrete            vs Continuous         |
|   6. Jumlah Agen      : Single-Agent        vs Multi-Agent        |
+-------------------------------------------------------------------+
```

Lingkungan yang paling mudah ditangani adalah: *Fully Observable, Deterministic, Episodic, Static, Discrete, dan Single-Agent* (misalnya: teka-teki tebak silang silang). Sebaliknya, lingkungan yang paling sulit dan realistis adalah: *Partially Observable, Stochastic, Sequential, Dynamic, Continuous, dan Multi-Agent* (misalnya: menyetir mobil otonom di lalu lintas Jakarta).

---

## 9. Fully Observable vs Partially Observable

- **Fully Observable (Dapat Diamati Penuh)**: Sensor agen memiliki akses ke seluruh status lingkungan pada setiap waktu yang relevan dengan pengambilan keputusan (contoh: catur fisik, teka-teki 8-puzzle). Agen tidak perlu memelihara status internal tentang dunia luar karena sensornya melihat segalanya.
- **Partially Observable (Dapat Diamati Sebagian)**: Sensor agen memiliki keterbatasan jangkauan, terhalang derau (*noise*), atau ada variabel lingkungan penting yang tidak terukur langsung (contoh: permainan kartu Poker di mana kartu lawan tertutup, atau mobil otonom yang terhalang truk di depannya). Agen **wajib** memelihara status internal (*internal state*) untuk mengingat apa yang terjadi sebelumnya.

---

## 10. Deterministic vs Stochastic

- **Deterministic (Deterministik)**: Status lingkungan berikutnya sepenuhnya ditentukan oleh status saat ini dan tindakan yang dieksekusi oleh agen (contoh: catur). Jika agen memilih memindahkan pion ke E4, pion pasti tiba di E4 tanpa kemungkinan terpeleset ke D4.
- **Stochastic (Stokastik)**: Terdapat ketidakpastian atau unsur keacakan; aksi yang sama dari status yang sama dapat menghasilkan status akhir yang berbeda dengan probabilitas tertentu (contoh: menyiram tanaman di luar ruangan yang dipengaruhi angin kencang, atau pergerakan mobil di jalan licin bersalju).

---

## 11. Episodic vs Sequential

- **Episodic (Episodik)**: Pengalaman agen dibagi ke dalam episode-episode independen. Pada setiap episode, agen menerima percept dan memilih satu aksi; kualitas aksinya hanya bergantung pada episode tersebut dan **tidak memengaruhi episode berikutnya** (contoh: sistem klasifikasi cacat botol di pabrik konveyor; membuang botol cacat #1 tidak mengubah kondisi botol cacat #2).
- **Sequential (Sekuensial)**: Keputusan saat ini akan memengaruhi seluruh percept dan keputusan di masa depan (contoh: catur, navigasi rute). Kesalahan langkah pion di awal permainan akan menjadi beban strategis hingga akhir pertandingan.

---

## 12. Static vs Dynamic

- **Static (Statis)**: Lingkungan tidak berubah sementara waktu agen sedang memproses dan memikirkan keputusan (contoh: catur tanpa jam catur). Agen dapat berpikir selama yang ia mau tanpa takut dunia berubah.
- **Dynamic (Dinamis)**: Lingkungan terus berubah dan berevolusi seiring berjalannya waktu komputasi (contoh: menyetir mobil, perdagangan saham frekuensi tinggi). Jika agen terlalu lama berpikir, mobil akan menabrak rintangan sebelum keputusan rem keluar.
- *Semidynamic*: Status lingkungan fisik tidak berubah seiring waktu, tetapi nilai performa agen berkurang jika waktu berpikirnya lambat (contoh: catur dengan *chess clock*).

---

## 13. Discrete vs Continuous

- **Discrete (Diskrit)**: Status lingkungan, representasi waktu, percept, dan aksi agen memiliki nilai terbatas dan terhitung secara pasti (contoh: catur memiliki 64 petak diskret dan jumlah kemungkinan langkah terhingga).
- **Continuous (Kontinu)**: Variabel lingkungan memiliki nilai riil tak berhingga (contoh: kecepatan mobil dalam km/jam dengan desimal kontinu, koordinat lintang-bujur GPS, atau sudut kemudi setir yang mulus).

---

## 14. Single-Agent vs Multi-Agent

- **Single-Agent (Agen Tunggal)**: Agen beroperasi sendirian di lingkungannya; entitas lain di dunia luar hanyalah objek fisik pasif yang mematuhi hukum alam (contoh: teka-teki Sudoku, pengatur suhu AC ruangan).
- **Multi-Agent (Multi-Agen)**: Terdapat entitas cerdas lain di dalam lingkungan yang juga memiliki tujuan dan memaksimalkan ukuran kinerja mereka sendiri:
  - *Kompetitif*: Agen saling bermusuhan (contoh: catur, di mana lawan berusaha meminimalkan nilai utilitas kita).
  - *Kooperatif*: Agen bekerja sama untuk mencapai tujuan bersama (contoh: armada drone pengantar barang yang mengoordinasikan rute agar tidak bertabrakan).

---

## 15. Simple Reflex Agents

**Simple Reflex Agent** adalah jenis arsitektur agen yang paling mendasar. Agen ini memilih tindakan murni berdasarkan **percept saat ini**, sepenuhnya mengabaikan riwayat persepsi masa lalu.

```
+-------------------------------------------------------------+
|                     SIMPLE REFLEX AGENT                     |
|                                                             |
|                       +-------------+                       |
|                       |   Sensors   |                       |
|                       +-------------+                       |
|                              |                              |
|                              v                              |
|                   +---------------------+                   |
|                   | What the world is   |                   |
|                   |    like right now   |                   |
|                   +---------------------+                   |
|                              |                              |
|                              v                              |
|                   +---------------------+                   |
|   Condition-Action|   Condition-Action  |                   |
|        Rules      |        Rules        |                   |
|   (IF-THEN Rules) +---------------------+                   |
|                              |                              |
|                              v                              |
|                   +---------------------+                   |
|                   |  What action I      |                   |
|                   |   should do now     |                   |
|                   +---------------------+                   |
|                              |                              |
|                              v                              |
|                       +-------------+                       |
|                       |  Actuators  |                       |
|                       +-------------+                       |
+-------------------------------------------------------------+
```

Logika operasinya dibangun di atas aturan kondisi-aksi (*Condition-Action Rules*):

$$\text{IF } \text{Kondisi Terpenuhi } \text{THEN } \text{Eksekusi Aksi}$$

*Kelebihan*: Sangat cepat, hemat komputasi, mudah diimplementasikan.  
*Kelemahan Fatal*: Hanya dapat berfungsi jika lingkungannya bersifat **Fully Observable**. Jika lingkungan berstatus *Partially Observable*, Simple Reflex Agent sangat mudah terjebak dalam **infinite loop** (perilaku berputar-putar tanpa henti).

---

## 16. Model-Based Reflex Agents

Untuk mengatasi masalah *partially observable environment*, agen wajib memiliki memori yang disebut **Internal State**. Agen ini disebut **Model-Based Reflex Agent**.

```
+-------------------------------------------------------------+
|                  MODEL-BASED REFLEX AGENT                   |
|                                                             |
|                       +-------------+                       |
|                       |   Sensors   |                       |
|                       +-------------+                       |
|                              |                              |
|                              v                              |
|                   +---------------------+                   |
|   State History   |     State Model     | <---+ How world   |
|   (Internal Memory)                     |     | evolves     |
|                   +---------------------+ <---+ What my     |
|                              |                | actions do  |
|                              v                |             |
|                   +---------------------+     |             |
|                   |   Condition-Action  |     |             |
|                   |        Rules        |     |             |
|                   +---------------------+     |             |
|                              |                |             |
|                              v                |             |
|                       +-------------+         |             |
|                       |  Actuators  | --------+             |
|                       +-------------+                       |
+-------------------------------------------------------------+
```

Agen memelihara dua jenis pengetahuan model dunia:
1. **How the world evolves independently of the agent**: Bagaimana kondisi fisik lingkungan berubah secara alami seiring waktu (misalnya: mobil di depan yang sedang melaju cepat akan berada lebih jauh dalam detik berikutnya).
2. **How the agent's own actions affect the world**: Bagaimana tindakan agen sendiri mengubah lingkungan (misalnya: memutar kemudi ke kanan akan mengubah orientasi kendaraan).

---

## 17. Goal-Based Agents

Mengetahui status dunia saat ini belum cukup untuk memutuskan apa yang harus dilakukan. Sebuah agen juga membutuhkan informasi tentang **Tujuan (*Goal*)** yang ingin dicapai.

```
+-------------------------------------------------------------+
|                      GOAL-BASED AGENT                       |
|                                                             |
|   Sensor -> State -> [ What it will be like if I do action ] |
|                                   |                         |
|                             [ GOAL TEST ]                   |
|                       (Apakah aksi mendekati goal?)         |
|                                   |                         |
|                             Actuators                       |
+-------------------------------------------------------------+
```

Perbedaan dengan Reflex Agent:
- Reflex Agent langsung bertindak begitu aturan kondisi cocok (`IF A THEN B`).
- Goal-Based Agent melakukan penalaran antisipatif: *"Jika saya memilih aksi A, dunia akan menjadi status X; apakah status X membawa saya lebih dekat ke tujuan saya?"*
- Mendukung pemecahan masalah berbasis **Pencarian (*Search*)** dan **Perencanaan (*Planning*)**. Jauh lebih fleksibel: jika tujuan diubah (misalnya rute perjalanan diganti dari Bandung ke Surabaya), program tidak perlu ditulis ulang, agen cukup mencari jalur baru.

---

## 18. Utility-Based Agents

Dalam banyak situasi nyata, mencapai tujuan saja tidak cukup; agen harus memilih **cara terbaik, tercepat, teraman, atau termurah** untuk mencapainya. Banyak tujuan yang saling bertentangan (misalnya: ingin sampai sangat cepat, tetapi juga ingin menghemat bahan bakar dan menghindari risiko kecelakaan).

Untuk menyelesaikan konflik tujuan ini, **Utility-Based Agent** menggunakan **Fungsi Utilitas (*Utility Function*)**:

$$U: S \to \mathbb{R}$$

Fungsi utilitas memetakan status dunia (*state*) ke sebuah bilangan riil yang menyatakan derajat kepuasan atau "kebahagiaan" (*degree of happiness/preference*) agen.

Agen utilitas memilih tindakan yang memaksimalkan utilitas yang diharapkan (*Expected Utility*):

$$\text{Action}^* = \arg\max_{a} \sum_{s'} P(s' \mid s, a) U(s')$$

Ini menjadi fondasi pengambilan keputusan di bawah ketidakpastian (*Decision Theory*).

---

## 19. Learning Agents

Seluruh arsitektur agen sebelumnya dapat ditransformasikan menjadi agen pembelajar (**Learning Agent**) dengan memisahkan komponen pengambil keputusan dari komponen pemelajaran.

```
+-------------------------------------------------------------+
|                       LEARNING AGENT                        |
|                                                             |
|                       +-------------+                       |
|                       |   Sensors   |                       |
|                       +-------------+                       |
|                        /           \                        |
|                       v             v                       |
|            +---------------+   +--------------------+       |
|            |    CRITIC     |   | PERFORMANCE ELEMENT|       |
|            +---------------+   | (Memilih tindakan) |       |
|                   |            +--------------------+       |
|                   v                      |                  |
|            +---------------+             |                  |
|            |LEARNING ELEM. |             v                  |
|            +---------------+      +-------------+           |
|              /           \        |  Actuators  |           |
|             v             v       +-------------+           |
|    +-------------+   +-----------+                          |
|    |Learning Goal|   |  PROBLEM  |                          |
|    +-------------+   | GENERATOR | (Menyarankan eksperimen) |
|                      +-----------+                          |
+-------------------------------------------------------------+
```

Empat komponen konseptual Learning Agent:
1. **Learning Element**: Bertanggung jawab membuat perbaikan dan pembaruan terhadap pengetahuan agen.
2. **Performance Element**: Komponen agen yang bertugas memilih aksi eksternal ke lingkungan (berisi salah satu dari 4 arsitektur sebelumnya).
3. **Critic**: Memberikan umpan balik evaluatif kepada learning element mengenai seberapa baik performa agen dibandingkan standar eksternal.
4. **Problem Generator**: Menyarankan tindakan-tindakan eksplorasi baru yang mungkin suboptimal dalam jangka pendek, tetapi menghasilkan pengalaman baru yang sangat bernilai untuk pemelajaran jangka panjang (*Exploration vs Exploitation Trade-off*).

---

## 20. Agent Architecture dan Batasannya

Program agen berjalan di atas sebuah **Arsitektur Perangkat Keras / Komputasi**:

$$\text{Agent} = \text{Architecture} + \text{Program}$$

Arsitektur menyediakan layanan fisik: membaca sensor, menyalakan aktuator, mengatur alokasi memori, dan membatasi clock siklus pemrosesan.

Batasan Arsitektur:
- **Keterbatasan Komputasi (*Bounded Rationality*)**: Herbert Simon memperkenalkan konsep *bounded rationality*: agen nyata di dunia nyata tidak memiliki waktu komputasi dan memori tak hingga untuk menghitung solusi matematis optimal murni. Agen rasional di dunia nyata harus mampu mengambil keputusan yang "cukup baik" (*satisficing*) dalam batas waktu yang tersedia.

---

## 21. AI sebagai Problem Solving

Ketika sebuah agen refleks tidak dapat memetakan percept saat ini langsung ke tindakan akhir, agen harus beralih ke paradigma **Pemecahan Masalah (*Problem Solving*)**.

Agen pemecah masalah adalah agen berbasis tujuan (*goal-based agent*) yang menemukan urutan tindakan (*sequence of actions*) yang mengarah ke status tujuan yang diinginkan melalui proses **Pencarian (*Search*)**.

Langkah-langkah Problem Solving:
1. **Goal Formulation**: Menetapkan tujuan yang ingin dicapai berdasarkan status saat ini dan ukuran kinerja.
2. **Problem Formulation**: Memutuskan status-status dan tindakan apa saja yang perlu dipertimbangkan di dunia (proses abstraksi).
3. **Search**: Menyimulasikan urutan tindakan di dalam model ruang keadaan sebelum mengeksekusinya di dunia nyata.
4. **Execution**: Mengeksekusi rangkaian aksi yang telah ditemukan satu demi satu ke lingkungan fisik.

---

## 22. Problem Formulation

**Problem Formulation** adalah proses formal mendefinisikan sebuah masalah komputasi AI ke dalam lima komponen baku:

1. **Initial State ($s_0$)**: Status awal tempat agen memulai pencarian.
2. **Actions ($A(s)$)**: Himpunan seluruh tindakan legal yang dapat dieksekusi ketika agen berada di status $s$.
3. **Transition Model ($T(s, a)$ atau $\text{Result}(s, a)$)**: Fungsi formal yang mendeskripsikan status apa yang akan dicapai jika agen mengeksekusi aksi $a$ dari status $s$:

$$s' = \text{Result}(s, a)$$

4. **Goal Test**: Sebuah fungsi predikat biner yang menguji apakah sebuah status tertentu merupakan status tujuan yang valid:

$$\text{is\_goal}(s) \to \{\text{True}, \text{False}\}$$

5. **Path Cost ($c(s, a, s')$ atau $g(n)$)**: Fungsi numerik yang mengukur biaya total perjalanan di sepanjang jalur. Biaya jalur adalah akumulasi dari biaya setiap langkah (*step cost*).

---

## 23. State Space Representation

Kelima komponen formulasi masalah di atas secara implisit mendefinisikan **Ruang Keadaan (*State Space*)**:

> **Ruang Keadaan (*State Space*)**:  
> Himpunan seluruh status yang dapat dicapai dari *initial state* melalui urutan tindakan legal apa pun. Ruang keadaan membentuk sebuah graf berarah di mana node melambangkan status (*states*) dan edge melambangkan transisi tindakan (*actions*).

```
   State Space Graph (Abstraksi Grid 3x3):
   
   (0,0) <---RIGHT---> (0,1) [Rintangan]
     ^                   
     | DOWN              
     v                   
   (1,0) <---RIGHT---> (1,1) [Rintangan]
     ^                   
     | DOWN              
     v                   
   (2,0) <---RIGHT---> (2,1) <---RIGHT---> (2,2) [GOAL]
```

---

## 24. Initial State, Actions, Transition Model, Goal Test, dan Path Cost

Berikut adalah perumusan formal notasi matematika kelima komponen:

$$\begin{aligned}
s_0 &\in S && \text{(Initial State)} \\
\text{Actions}(s) &\subseteq A && \text{(Himpunan Aksi Legal dari state } s) \\
\text{Result}: S \times A &\to S && \text{(Model Transisi Deterministik)} \\
\text{GoalTest}: S &\to \{0, 1\} && \text{(Fungsi Uji Tujuan)} \\
c(s, a, s') &\ge 0 && \text{(Step Cost non-negatif)}
\end{aligned}$$

Total biaya jalur dari urutan langkah $s_0 \xrightarrow{a_1} s_1 \xrightarrow{a_2} s_2 \dots \xrightarrow{a_k} s_k$:

$$\text{Path Cost} = \sum_{i=1}^k c(s_{i-1}, a_i, s_i)$$

---

## 25. Contoh Navigasi pada Grid Sederhana

Mari kita formulasikan masalah navigasi robot pembersih lantai pada grid $3 \times 3$:

1. **Initial State**: Koordinat sel awal, $s_0 = (0, 0)$.
2. **Actions**: Himpunan pergerakan diskret:
   $$\text{Actions} = \{\text{UP}, \text{DOWN}, \text{LEFT}, \text{RIGHT}\}$$
3. **Transition Model**:
   - Jika aksi sah (tidak keluar batas $0 \le r < 3, 0 \le c < 3$ dan tidak menabrak rintangan $(1,1)$), koordinat bertambah sesuai vektor arah.
   - Contoh: $\text{Result}((0, 0), \text{DOWN}) = (1, 0)$.
4. **Goal Test**: Robot berada di pojok kanan bawah:
   $$\text{is\_goal}((r, c)) = (r == 2 \land c == 2)$$
5. **Path Cost**: Setiap langkah perpindahan memiliki biaya seragam 1.0 unit:
   $$c(s, a, s') = 1.0$$

Jalur solusi optimal yang ditemukan:
$$(0,0) \xrightarrow{\text{DOWN}} (1,0) \xrightarrow{\text{DOWN}} (2,0) \xrightarrow{\text{RIGHT}} (2,1) \xrightarrow{\text{RIGHT}} (2,2)$$
Total biaya jalur solusi: $1.0 + 1.0 + 1.0 + 1.0 = 4.0$ unit biaya.

---

## 26. Reactive vs Planning-Based Behavior

Dua kutub filosofi dalam perancangan agen cerdas:

| Dimensi | Reactive Behavior (Refleks) | Planning-Based Behavior (Pencarian) |
|---|---|---|
| **Mekanisme** | Aturan kondisi-aksi langsung $p \to a$. | Simulasi eksplorasi jalur ruang keadaan di masa depan. |
| **Kebutuhan Waktu** | Sangat cepat (konstan $\mathcal{O}(1)$). | Membutuhkan waktu pencarian (dapat eksponensial $\mathcal{O}(b^d)$). |
| **Kebutuhan Model** | Tidak memerlukan model eksplisit dunia. | Memerlukan model transisi dan deskripsi tujuan formal. |
| **Ketahanan Dinamis** | Sangat responsif terhadap perubahan mendadak. | Lambat merespons jika lingkungan berubah saat sedang mencari. |
| **Optimasi Solusi** | Rentan jalan buntu (*local optima* / *deadlocks*). | Mampu menjamin rute optimal global (*global optimum*). |

Arsitektur modern umumnya menggunakan pendekatan hibrida (*hybrid layered architecture*): lapisan bawah reaktif untuk keselamatan refleks darurat (misal: sensor rem mendadak), lapisan atas berbasis perencanaan untuk navigasi rute jangka panjang.

---

## 27. Keterbatasan Agent Sederhana

Mengapa kita tidak bisa menyelesaikan seluruh masalah dunia hanya dengan agen sederhana?

1. **Ledakan Kombinatorik (*Combinatorial Explosion*)**: Ketika ukuran ruang keadaan membesar (misal: catur dengan $\approx 10^{120}$ status permainan), mustahil mengeksplorasi seluruh ruang keadaan secara mendalam tanpa fungsi heuristik pemandu.
2. **Ketidakpastian dan Parsialitas Dunia Nyata**: Sebagian besar sistem dunia nyata tidak bersifat deterministik dan diskret.
3. **Trade-off Eksplorasi vs Eksploitasi**: Agen pembelajar sering kali terjebak pada kepuasan semu (*suboptimal habits*) jika tidak didorong untuk mengambil risiko eksplorasi.

Tantangan-tantangan inilah yang menjadi motivasi utama lahirnya algoritma pencarian cerdas terinformasi (*Informed Search & A\**) yang akan kita pelajari secara mendalam pada **Chapter 04**.

---

## 28. Ringkasan Modul

- **Intelligent Agent** adalah entitas yang mengamati lingkungan melalui *sensors* dan bertindak melalui *actuators*.
- **Rasionalitas** berarti memaksimalkan nilai ukuran kinerja yang diharapkan (*expected performance measure*), bukan kesempurnaan mutlak (*perfection*) atau kemahatahuan (*omniscience*).
- **Kerangka Kerja PEAS** (*Performance, Environment, Actuators, Sensors*) adalah langkah pertama yang wajib didefinisikan dalam merancang agen cerdas apa pun.
- Lingkungan tugas dikarakterisasi ke dalam 6 dimensi: *observability*, *determinism*, *episodicity*, *dynamism*, *discreteness*, dan *agent cardinality*.
- Lima hierarki arsitektur agen:
  1. *Simple Reflex Agent* (aturan kondisi-aksi langsung).
  2. *Model-Based Reflex Agent* (status internal dan model perubahan dunia).
  3. *Goal-Based Agent* (mempertimbangkan tujuan di masa depan).
  4. *Utility-Based Agent* (mengukur derajat preferensi fungsi utilitas di bawah ketidakpastian).
  5. *Learning Agent* (berevolusi melalui critic, learning element, dan problem generator).
- **AI Problem Solving** memformulasikan masalah ke dalam 5 elemen: *Initial State*, *Actions*, *Transition Model*, *Goal Test*, dan *Path Cost*. Ruang keadaan membentuk graf berarah tempat algoritma pencarian beroperasi.

---

## 29. Latihan Terstruktur

### Latihan Tingkat Dasar (Konseptual & PEAS)
1. Jelaskan perbedaan mendasar antara *Agent Function* matematis dan *Agent Program* komputasional! Mengapa sebuah program agen praktis umumnya tidak menerima seluruh riwayat $P^*$ secara langsung?
2. Berikan analisis PEAS lengkap (Performance measure, Environment, Actuators, Sensors) untuk sebuah sistem **Smart Irrigation Controller** (Pengatur Irigasi Otomatis Pertanian Presisi)!
3. Identifikasi dan klasifikasikan 6 dimensi sifat lingkungan tugas (*Task Environment*) untuk permainan catur standar melawan manusia!

### Latihan Tingkat Menengah (Analisis Arsitektur & Perilaku)
4. Jelaskan mengapa *Simple Reflex Agent* dapat mengalami kegagalan fatal berupa *infinite loop* (terjebak dalam siklus aksi berulang) pada lingkungan yang bersifat *Partially Observable*! Berikan skenario ilustratifnya pada agen robot pembersih lantai!
5. Bagaimana *Model-Based Reflex Agent* mengatasi kelemahan di atas? Komponen apa saja yang wajib ditambahkan pada memorinya?

### Latihan Tingkat Lanjut (Problem Formulation Formal)
6. Sebuah robot pengantar logistik mini berada di lantai gudang berukuran $4 \times 4$ sel koordinat:
   - Status awal robot: $(0, 0)$.
   - Status target barang: $(3, 3)$.
   - Rintangan rak barang: $(1, 1), (1, 2), (2, 1)$.
   - Aksi legal: $\text{Actions} = \{\text{UP}, \text{DOWN}, \text{LEFT}, \text{RIGHT}\}$. Setiap langkah horizontal bernilai biaya 1.0, sedangkan langkah vertikal bernilai biaya 1.5.
   - **Tugas Anda**:
     a) Rumuskan kelima komponen formal masalah ini secara matematis!
     b) Hitung total *Path Cost* untuk urutan tindakan: `[DOWN, DOWN, DOWN, RIGHT, RIGHT, RIGHT]`!
     c) Apakah jalur tersebut valid dan legal? Berikan analisis transisinya langkah demi langkah!

---

## 30. Pembahasan dan Panduan Jawaban Latihan

### Pembahasan Latihan Tingkat Dasar
1. **Agent Function vs Agent Program**:
   - *Agent Function* $f: P^* \to A$ adalah pemetaan abstrak matematis dari seluruh riwayat tak berhingga percept ke tindakan ideal.
   - *Agent Program* adalah kode nyata yang dieksekusi prosesor pada siklus waktu diskret. Program agen tidak mungkin menerima seluruh riwayat $P^*$ karena kapasitas memori RAM komputer terbatas dan dimensi pencarian akan meledak secara kombinatorik. Oleh karena itu, program agen hanya menerima percept saat ini ($p_t$) dan memperbarui ringkasan keadaan masa lalu ke dalam variabel status internal (*internal state*).
2. **PEAS Smart Irrigation Controller**:
   - **P (Performance)**: Kelembapan tanah optimal di zona akar (60–80%), penghematan volume air total ($m^3$), pencegahan kebusukan akar akibat genangan, dan efisiensi konsumsi listrik pompa.
   - **E (Environment)**: Lahan pertanian terbuka, variasi cuaca dan intensitas matahari tropis, perkiraan hujan, karakteristik daya serap jenis tanah, dan laju evapotranspirasi.
   - **A (Actuators)**: Katup solenoid kran air bertekanan, motor pompa air listrik, dan nosel penyiram mikrospray / tetes (*drip*).
   - **S (Sensors)**: Sensor kelembapan tanah kapasitif (*capacitive soil moisture sensor*), sensor suhu dan kelembapan udara (DHT22), sensor radiasi cahaya matahari (lux sensor), dan antarmuka ramalan cuaca daring.
3. **Sifat Lingkungan Catur Standar**:
   - *Observabilitas*: Fully Observable (seluruh 64 petak dan posisi bidak terlihat).
   - *Determinisasi*: Deterministic (langkah bidak tidak memiliki elemen acak/probabilistik).
   - *Episodik/Sekuensial*: Sequential (langkah awal menentukan posisi akhir).
   - *Statis/Dinamis*: Semi-dynamic (papan catur fisik tidak berubah saat pemain berpikir, tetapi jam catur terus berjalan mengurangi waktu jatah).
   - *Diskrit/Kontinu*: Discrete (petak dan giliran langkah bernilai diskret terhitung).
   - *Jumlah Agen*: Multi-Agent Competitive (dua agen bersaing secara zero-sum game).

### Pembahasan Latihan Tingkat Menengah
4. **Infinite Loop pada Simple Reflex Agent**:
   - Pada *Partially Observable Environment*, dua status lingkungan yang berbeda dapat menghasilkan percept identik yang sama persis bagi sensor agen (disebut *perceptual aliasing*).
   - *Skenario*: Robot pembersih berada di sel A yang kotor, membersihkan, lalu bergerak ke kanan ke sel B yang bersih. Di sel B sensor mendeteksi lantai bersih, aturan berkata `MOVE_LEFT`. Saat kembali ke sel A yang sudah bersih, sensor mendeteksi lantai bersih dan aturan berkata `MOVE_RIGHT`. Akibatnya, agen akan berosilasi bolak-balik antara sel A dan B selamanya tanpa pernah mengeksplorasi ruangan lain.
5. **Solusi Model-Based Reflex Agent**:
   - Agen berbasis model menambahkan memori internal yang menyimpan peta status dunia yang telah dikunjungi (*visited states map*) dan model perkembangan dunia.
   - Variabel yang wajib ditambahkan: status orientasi/posisi terakhir, riwayat status sel yang telah dibersihkan, dan penghitung frekuensi kunjungan agar tidak mengulang aksi yang sama secara simetris.

### Pembahasan Latihan Tingkat Lanjut
6. **Formulasi Formal Masalah Navigasi Gudang**:
   - a) **Lima Komponen Formal**:
     - $s_0 = (0, 0)$
     - $S = \{(r, c) \mid 0 \le r < 4, 0 \le c < 4 \land (r, c) \notin \{(1, 1), (1, 2), (2, 1)\}\}$
     - $\text{Actions}((r, c)) = \{a \in \{\text{UP}, \text{DOWN}, \text{LEFT}, \text{RIGHT}\} \mid \text{Result}((r, c), a) \in S\}$
     - $\text{Result}((r, c), a) = (r + \Delta r, c + \Delta c)$ sesuai vektor arah.
     - $\text{GoalTest}(s) = (s == (3, 3))$
     - $\text{StepCost}(s, a, s') = 1.5$ jika $a \in \{\text{UP}, \text{DOWN}\}$, dan $1.0$ jika $a \in \{\text{LEFT}, \text{RIGHT}\}$.
   - b) **Kalkulasi Path Cost**:
     - 3 langkah vertikal `DOWN`: $3 \times 1.5 = 4.5$.
     - 3 langkah horizontal `RIGHT`: $3 \times 1.0 = 3.0$.
     - Total *Path Cost* $= 4.5 + 3.0 = 7.5$ unit biaya.
   - c) **Uji Validitas dan Legalitas Jalur**:
     - Langkah 1: $(0,0) \xrightarrow{\text{DOWN}} (1,0)$ (Sah, $(1,0)$ bukan rintangan).
     - Langkah 2: $(1,0) \xrightarrow{\text{DOWN}} (2,0)$ (Sah, $(2,0)$ bukan rintangan).
     - Langkah 3: $(2,0) \xrightarrow{\text{DOWN}} (3,0)$ (Sah, $(3,0)$ bukan rintangan).
     - Langkah 4: $(3,0) \xrightarrow{\text{RIGHT}} (3,1)$ (Sah, baris 3 bebas rintangan).
     - Langkah 5: $(3,1) \xrightarrow{\text{RIGHT}} (3,2)$ (Sah).
     - Langkah 6: $(3,2) \xrightarrow{\text{RIGHT}} (3,3)$ (Sah, Tiba di Goal!).
     - **Kesimpulan**: Jalur valid 100% dan berhasil mencapai tujuan tanpa menabrak rak rintangan.

---

## 31. Tugas Mini Project: Perancangan Intelligent Agent untuk Sistem Smart Environment

### Deskripsi Proyek
Mahasiswa diminta merancang arsitektur sistem agen cerdas terpadu untuk mengelola sebuah ekosistem lingkungan pintar (*Smart Environment*). Proyek ini menuntut integrasi antara analisis teoretis PEAS, perumusan sifat lingkungan tugas, penentuan arsitektur agen yang proporsional, serta formulasi masalah ruang keadaan dan prototipe kode Python / pseudocode yang dapat dievaluasi.

### Pilihan Domain Smart Environment (Pilih Salah Satu)
1. **Smart Home**: Agen manajemen energi pintar dan pencahayaan adaptif berbasis kehadiran penghuni dan tarif listrik dinamik.
2. **Smart Building**: Sistem HVAC cerdas terintegrasi untuk gedung perkantoran bertingkat dengan zona termal multi-ruangan.
3. **Smart Agriculture**: Agen otomasi rumah kaca presisi (*precision greenhouse*) pengatur nutrisi hidroponik, aerasi, dan kelembapan tanah.
4. **Smart Transportation**: Agen pengatur lampu lalu lintas adaptif pada persimpangan jalan padat berbasis deteksi kepadatan antrean kendaraan.
5. **Smart Energy Grid**: Agen penyeimbang beban mikro (*microgrid energy dispatch agent*) yang mengoptimalkan baterai surya rumah tangga dan jaringan listrik utama.

### Ruang Lingkup dan 9 Deliverables Wajib
Laporan teknis disusun dalam dokumen akademik terstruktur (5–8 halaman) memuat:
1. **Deskripsi Sistem & Problem Statement**: Latar belakang domain, tujuan operasional, dan batasan teknis lingkungan.
2. **Tabel Analisis PEAS Terperinci**:
   - 3+ metrik performa objektif terukur (*Performance Measures*).
   - Deskripsi lengkap komponen dunia luar (*Environment*).
   - Daftar seluruh perangkat pengubah lingkungan (*Actuators*).
   - Daftar seluruh perangkat penangkap persepsi (*Sensors*).
3. **Taksonomi 6 Karakteristik Task Environment**: Analisis dimensi (Fully/Partially, Deterministik/Stokastik, Episodik/Sekuensial, Statis/Dinamis, Diskrit/Kontinu, Single/Multi-Agent) disertai argumen pendukung.
4. **Pemilihan & Justifikasi Arsitektur Agen**: Pemilihan salah satu arsitektur (Reflex, Model-Based, Goal-Based, Utility-Based, Learning Agent) beserta justifikasi mengapa arsitektur yang lebih sederhana tidak memadai dan mengapa arsitektur yang lebih kompleks tidak berlebihan.
5. **Diagram Alur Agen–Lingkungan**: Diagram visual hubungan sensor, pemrosesan internal, model status, dan aktuator.
6. **Formulasi Masalah Formal (Problem Formulation)**:
   - Definisi formal *Initial State*.
   - Himpunan *Actions* legal.
   - Deskripsi *Transition Model*.
   - Kriteria *Goal Test*.
   - Perumusan fungsi biaya langkah dan *Path Cost*.
7. **Prototipe Python / Pseudocode Runnable**: Skrip Python berorientasi objek yang mensimulasikan minimal 5 langkah interaksi agen dengan lingkungannya dan menghitung skor performanya.
8. **Analisis Keterbatasan & Responsible AI**: Identifikasi skenario kegagalan (*failure modes*), keterbatasan asumsi model, dan potensi risiko etika/keamanan jika sensor mengalami gangguan.
9. **Daftar Pustaka Primer**: Minimal 2 rujukan buku teks atau artikel akademik terverifikasi.

---

## 32. Rubrik Penilaian Proyek (Total 100 Poin)

| Kriteria Penilaian | Bobot | Kinerja Unggul (Penuh) | Kinerja Cukup (Parsial) | Kinerja Kurang (Tidak Terpenuhi) |
|---|---:|---|---|---|
| **Rumusan Masalah & Analisis PEAS** | **20 Poin** | Rumusan masalah tajam; tabel PEAS komprehensif, metrik performa objektif dan tidak rancu dengan aksi. (18–20) | Masalah cukup jelas; PEAS lengkap tetapi metrik performa masih parsial. (10–17) | Analisis PEAS tidak lengkap, salah membedakan aktuator dan sensor. (0–9) |
| **Ketepatan Arsitektur & Lingkungan** | **25 Poin** | Analisis 6 dimensi task environment akurat; pemilihan arsitektur agen didukung justifikasi trade-off yang kuat. (22–25) | Karakteristik lingkungan cukup tepat; pemilihan arsitektur kurang disertai perbandingan alternatif. (12–21) | Salah mengidentifikasi sifat lingkungan; arsitektur yang dipilih tidak sesuai karakteristik masalah. (0–11) |
| **Problem Formulation Ruang Keadaan** | **20 Poin** | Formulasi 5 elemen formal matematis lengkap, presisi, dan konsisten dengan ruang keadaan masalah. (18–20) | Formulasi 5 elemen ada namun fungsi transisi atau biaya langkah kurang didefinisikan secara formal. (10–17) | Komponen formulasi masalah tidak lengkap atau tidak masuk akal. (0–9) |
| **Prototipe Python / Pseudocode** | **15 Poin** | Kode modular, runnable tanpa error, struktur data bersih, mensimulasikan interaksi nyata dan kalkulasi skor. (13–15) | Kode berjalan namun kaku, hard-coded, atau pseudocode kurang terstruktur. (7–12) | Kode error/tidak runnable, tidak mencerminkan logika agen yang dirancang. (0–6) |
| **Evaluasi Keterbatasan & Responsible AI** | **10 Poin** | Menganalisis skenario kegagalan, batasan observabilitas, dan mitigasi risiko keselamatan secara kritis. (9–10) | Analisis keterbatasan bersifat umum tanpa skenario kegagalan spesifik. (5–8) | Tidak menyertakan evaluasi keterbatasan atau menganggap sistem sempurna tanpa cela. (0–4) |
| **Kerapian Dokumentasi & Referensi** | **10 Poin** | Format dokumen akademik rapi, diagram alur jelas terbaca, menyertakan minimal 2 referensi primer terverifikasi. (9–10) | Format cukup rapi, diagram ada namun kurang informatif, referensi kurang dari 2. (5–8) | Laporan berantakan, plagiat terdeteksi, atau tanpa referensi akademik. (0–4) |

---

## 33. Referensi dan Sumber Belajar Primer

1. **Russell, S., & Norvig, P. (2020)**. *Artificial Intelligence: A Modern Approach (4th ed.)*. Pearson. (Bab 2: *Intelligent Agents* & Bab 3: *Solving Problems by Searching*). URL: [https://aima.cs.berkeley.edu/](https://aima.cs.berkeley.edu/)
2. **Wooldridge, M. (2009)**. *An Introduction to MultiAgent Systems (2nd ed.)*. John Wiley & Sons. (Bab 1: *Introduction* & Bab 2: *Intelligent Agents*). URL: [https://www.wiley.com/en-us/An+Introduction+to+MultiAgent+Systems%2C+2nd+Edition-p-9780470519462](https://www.wiley.com/en-us/An+Introduction+to+MultiAgent+Systems%2C+2nd+Edition-p-9780470519462)
3. **ACM / IEEE-CS / AAAI (2024)**. *Computer Science Curricula 2023 (CS2023): Artificial Intelligence Knowledge Area*. ACM Press. URL: [https://csed.acm.org/](https://csed.acm.org/)
4. **Simon, H. A. (1996)**. *The Sciences of the Artificial (3rd ed.)*. MIT Press. (Konsep *Bounded Rationality* dan adaptasi agen terhadap lingkungan).
5. **Franklin, S., & Graesser, A. (1996)**. *Is It an Agent, or Just a Program? A Taxonomy for Autonomous Agents*. Proceedings of the Third International Workshop on Agent Theories, Architectures, and Languages, Springer-Verlag, 21–35.

