# Chapter 05: Knowledge Representation & Reasoning

> **Tingkat Keahlian**: Level 1 — AI Foundations  
> **Jalur Spesialisasi**: Core Curriculum  
> **Prasyarat**: Chapter 04 (Search Algorithms: Uninformed & Informed Search)  
> **Format Pembelajaran**: Modul Bacaan Teori, Pembuktian Logika Formal, Praktik Python Mandiri, dan Mini Project Sistem Pakar  

---

## 1. Pengantar Knowledge Representation & Reasoning

Dalam bab-bab sebelumnya, kita telah mempelajari agen yang memecahkan masalah melalui eksplorasi jalur ruang keadaan (*Search Algorithms*). Namun, agen pencarian murni memiliki keterbatasan fundamental: mereka memperlakukan setiap status (*state*) sebagai "kotak hitam" atomik tanpa memahami struktur internal dari fakta-fakta yang menyusun dunia tersebut. 

Agar sebuah sistem kecerdasan buatan dapat bernalar (*reason*), menggeneralisasi aturan, menjelaskan keputusan, dan mengakumulasi keahlian manusia, sistem tersebut membutuhkan **Knowledge Representation & Reasoning (KRR)**:
- **Knowledge Representation (Representasi Pengetahuan)**: Studi tentang bagaimana informasi mengenai dunia luar diformalkan ke dalam simbol-simbol dan struktur data komputasi sedemikian rupa sehingga mesin dapat memprosesnya.
- **Reasoning (Penalaran)**: Proses komputasi memanipulasi representasi simbolik tersebut untuk menarik kesimpulan-kesimpulan baru yang sah secara logis (*sound logical inferences*).

KRR merupakan pilar utama dari mazhab *Symbolic AI* yang menjadi fondasi sistem pakar (*Expert Systems*), basis data graf ontologi (*Knowledge Graphs*), web semantik, hingga arsitektur hibrida modern *Neuro-Symbolic AI*.

---

## 2. Hirarki Data, Informasi, dan Pengetahuan (DIKW)

Pembedaan epistemologis antara data mentah hingga tindakan bijak:

```
               +-----------------------+
               |     WISDOM (Aksi)     |  "Gunakan rute alternatif saat hujan"
               +-----------------------+
                           ^
                           |
               +-----------------------+
               |  KNOWLEDGE (Kaidah)   |  "Kabel putus menyebabkan link padam"
               +-----------------------+
                           ^
                           |
               +-----------------------+
               |  INFORMATION (Konteks)|  "Suhu ruangan 35°C (Terlalu panas)"
               +-----------------------+
                           ^
                           |
               +-----------------------+
               |      DATA (Mentah)    |  "35", "True", "192.168.1.1"
               +-----------------------+
```

- **Data**: Simbol atau angka mentah tanpa interpretasi konteks (misal: angka `0` atau string `"down"`).
- **Informasi**: Data yang telah diberi label, struktur, dan konteks semantik (misal: `"Interface GigabitEthernet0/1 berstatus down"`).
- **Pengetahuan (*Knowledge*)**: Integrasi informasi yang dikaitkan dengan kaidah sebab-akibat, relasi dependensi, dan prosedur penalaran (misal: `"JIKA link_light_off DAN cable_plugged MAKA nic_hardware_fault"`).

---

## 3. Arsitektur Knowledge-Based Agent

Sebuah agen berbasis pengetahuan (*Knowledge-Based Agent*) memiliki dua komponen utama yang terpisah secara modular:

```
+-------------------------------------------------------------+
|                 KNOWLEDGE-BASED AGENT                       |
|                                                             |
|   Sensors ----> [ TELL(KB, Percept) ]                       |
|                          |                                  |
|                          v                                  |
|               +-----------------------+                     |
|               |    KNOWLEDGE BASE     | <---+               |
|               | (Himpunan Kalimat /   |     |               |
|               |  Facts & Rules)       |     | Inferensi     |
|               +-----------------------+     | Logika        |
|                          ^                  | Deduktif      |
|                          |                  |               |
|               +-----------------------+     |               |
|               |   INFERENCE ENGINE    | ----+               |
|               |  (Forward / Backward) |                     |
|               +-----------------------+                     |
|                          |                                  |
|                          v                                  |
|   Actuators <-- [ ASK(KB, Action) ]                         |
+-------------------------------------------------------------+
```

1. **Knowledge Base (KB)**: Himpunan representasi formal dari apa yang diketahui oleh agen tentang dunianya. Setiap representasi individual disebut sebagai *kalimat* (*sentence*) atau aksioma.
2. **Inference Engine (Mesin Inferensi)**: Algoritma penalaran mandiri yang bertugas menurunkan kalimat baru dari kalimat-kalimat yang telah ada di KB.
3. **Dua Operasi Inti**:
   - `TELL(KB, sentence)`: Memasukkan fakta atau aturan baru ke dalam basis pengetahuan.
   - `ASK(KB, query)`: Mengajukan pertanyaan apakah suatu hipotesis bernilai benar berdasarkan isi basis pengetahuan.

---

## 4. Declarative vs Procedural Knowledge

Dua paradigma dalam merekayasa kecerdasan buatan:
- **Declarative Knowledge (Pengetahuan Deklaratif)**: Merepresentasikan **APA** yang benar tentang dunia dalam bentuk fakta dan aturan logika eksplisit. Mesin inferensi umum yang bertugas mencari cara menurunkannya.
  - *Keunggulan*: Sangat modular, mudah diperbarui (cukup menambah fakta baru tanpa mengubah kode algoritma), dan menyediakan jejak penjelasan transparan.
- **Procedural Knowledge (Pengetahuan Prosedural)**: Mengkodekan **BAGAIMANA** menyelesaikan masalah secara langsung ke dalam urutan langkah instruksi kode pemrograman (seperti tumpukan percabangan `if-else` bersarang).
  - *Kelemahan*: Kaku, rentan error jika aturan bertambah ratusan, dan sulit diaudit jalurnya.

---

## 5. Logika Proposisional (Propositional Logic)

Logika proposisional (atau kalkulus proposisi) adalah bahasa representasi pengetahuan formal yang paling sederhana.

### Sintaksis
Kalimat dalam logika proposisional dibangun dari:
- **Simbol Proposisi Atomik**: Dilambangkan dengan huruf besar ($P, Q, R, \dots$) yang mewakili pernyataan fakta yang hanya dapat bernilai benar (*True*) atau salah (*False*).
- **Operator Logika (*Connectives*)**:
  1. **Negasi ($\neg$ / `NOT`)**: Pembalikan nilai kebenaran ($\neg P$).
  2. **Konjungsi ($\land$ / `AND`)**: Bernilai True jika dan hanya jika kedua proposisi True ($P \land Q$).
  3. **Disjungsi ($\lor$ / `OR`)**: Bernilai True jika minimal salah satu proposisi True ($P \lor Q$).
  4. **Implikasi Material ($\to$ / `IMPLIES`)**: Hubungan kondisional *"JIKA P MAKA Q"* ($P \to Q$).
  5. **Bikondisional / Ekuivalensi ($\leftrightarrow$ / `IFF`)**: Bernilai True jika kedua proposisi memiliki nilai kebenaran identik ($P \leftrightarrow Q$).

---

## 6. Tabel Kebenaran (Truth Tables)

Semantik dari logika proposisional didefinisikan secara presisi melalui tabel kebenaran:

| $P$ | $Q$ | $\neg P$ | $P \land Q$ | $P \lor Q$ | $P \to Q$ | $P \leftrightarrow Q$ |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| True | True | False | True | True | **True** | True |
| True | False | False | False | True | **False** | False |
| False | True | True | False | True | **True** | False |
| False | False | True | False | False | **True** | True |

---

## 7. Peringatan: Implikasi Material vs Kausalitas

Banyak pemula terjebak pada baris ketiga dan keempat tabel kebenaran implikasi: mengapa ketika $P$ bernilai False, kalimat $P \to Q$ otomatis bernilai **True**?

> **Prinsip Implikasi Material**:  
> Kalimat $P \to Q$ menyatakan janji: *"Jika $P$ terjadi, maka saya jamin $Q$ pasti terjadi"*. Janji tersebut **hanya dilanggar (False)** jika premis $P$ benar-benar terjadi tetapi kesimpulan $Q$ tidak terbukti. Jika premis $P$ tidak pernah terjadi, janji tersebut tidak pernah dilanggar, sehingga secara matematis bernilai *vacuously true*.
> 
> Implikasi material ($P \to Q \equiv \neg P \lor Q$) **bukan hubungan sebab-akibat fisik (*causation*)**, melainkan hubungan nilai kebenaran formal.

---

## 8. Logical Entailment ($KB \models \alpha$) dan Model Checking

Hubungan paling fundamental dalam logika AI adalah **Entailment** (Keterikatan Logis):

$$\alpha \models \beta$$

Artinya: *"Kalimat $\alpha$ meng-entail kalimat $\beta$ jika dan hanya jika dalam setiap model (kemungkinan dunia) di mana $\alpha$ bernilai True, $\beta$ juga wajib bernilai True"*.

Secara formal berbasis himpunan model:
$$M(\alpha) \subseteq M(\beta)$$

### Model Checking
Metode pembuktian entailment paling mendasar pada logika proposisional adalah **Model Checking**:
1. Identifikasi seluruh $n$ simbol proposisi yang terlibat dalam $KB$ dan query $\alpha$.
2. Bangkitkan seluruh $2^n$ kemungkinan kombinasi nilai kebenaran (*Truth Table enumeration*).
3. Untuk setiap model di mana $KB$ bernilai True, verifikasi apakah $\alpha$ juga bernilai True.
4. Jika ya untuk semua model tersebut, maka terbukti $KB \models \alpha$. Jika ditemukan satu saja model di mana $KB = \text{True}$ namun $\alpha = \text{False}$, maka entailment gugur.

---

## 9. Aturan Inferensi Baku: Modus Ponens

Model checking membutuhkan waktu eksponensial $\mathcal{O}(2^n)$. Untuk bernalar secara efisien, AI menggunakan **Aturan Inferensi Deduktif (*Inference Rules*)**.

Aturan inferensi yang paling terkenal dan menjadi tulang punggung sistem pakar adalah **Modus Ponens**:

$$\frac{\alpha \to \beta, \quad \alpha}{\beta}$$

Artinya: Jika kita mengetahui bahwa kalimat implikasi $\alpha \to \beta$ bernilai benar, dan kita juga mengetahui bahwa anteseden $\alpha$ bernilai benar, maka kita berhak menyimpulkan bahwa konsekuen $\beta$ bernilai benar.

### Generalized Modus Ponens (Untuk Kaidah IF-THEN)
Dalam sistem berbasis aturan praktis, premis sering kali berbentuk konjungsi beberapa fakta:

$$\frac{(P_1 \land P_2 \land \dots \land P_k) \to Q, \quad P_1, P_2, \dots, P_k}{Q}$$

---

## 10. Pengantar First-Order Logic (FOL)

Meskipun logika proposisional memiliki semantik yang kokoh, ia memiliki keterbatasan daya ekspresi yang parah: **kurangnya mekanisme generalisasi objek**.

*Contoh Masalah*: Untuk menyatakan bahwa *"Setiap antarmuka jaringan yang kabelnya putus akan berstatus down"*, dalam logika proposisional kita harus menulis ribuan proposisi terpisah:
- $\text{KabelPutus\_Eth0} \to \text{Down\_Eth0}$
- $\text{KabelPutus\_Eth1} \to \text{Down\_Eth1}$
- $\text{KabelPutus\_Wlan0} \to \text{Down\_Wlan0}$ ... dan seterusnya.

### Unsur-Unsur First-Order Logic
First-Order Logic (FOL) mengatasi keterbatasan ini dengan memodelkan dunia ke dalam:
1. **Objek (*Objects*)**: Entitas konkret di dunia (misal: `Eth0`, `RouterA`, `Server1`).
2. **Predikat (*Predicates*)**: Menyatakan relasi atau properti objek (misal: $\text{Connected}(\text{Eth0}, \text{RouterA})$, $\text{IsDown}(\text{Eth0})$).
3. **Fungsi (*Functions*)**: Memetakan objek ke objek lain (misal: $\text{IPAddressOf}(\text{Eth0})$).
4. **Variabel (*Variables*)**: Huruf kecil yang mewakili objek sembarang ($x, y, z$).
5. **Kuantor (*Quantifiers*)**:
   - **Kuantor Universal ($\forall$)**: *"Untuk semua"* ($\forall x \, (\text{CableCut}(x) \to \text{IsDown}(x))$).
   - **Kuantor Eksistensial ($\exists$)**: *"Ada / terdapat minimal satu"* ($\exists x \, (\text{Interface}(x) \land \text{IsDown}(x))$).

---

## 11. Arsitektur Rule-Based Expert Systems

Sistem Pakar Berbasis Aturan (*Rule-Based Expert System*) merepresentasikan keahlian manusia ke dalam basis pengetahuan berbentuk kaidah:

$$\text{IF } \langle \text{Kondisi / Gejala} \rangle \text{ THEN } \langle \text{Kesimpulan / Tindakan} \rangle$$

Untuk mengeksekusi kaidah-kaidah ini, terdapat dua strategi mesin inferensi:

```
+-------------------------------------------------------------+
|              FORWARD vs BACKWARD CHAINING                   |
|                                                             |
|   FORWARD CHAINING (Bottom-Up / Data-Driven)                |
|   Fakta Gejala Awal =====> Kaidah Terpicu =====> Kesimpulan |
|                                                             |
|   BACKWARD CHAINING (Top-Down / Goal-Driven)                |
|   Fakta Dasar <===== Sub-Goal Premis <===== Hipotesis Goal  |
+-------------------------------------------------------------+
```

---

## 12. Forward Chaining (Data-Driven / Bottom-Up)

Forward Chaining dimulai dari himpunan fakta yang telah diketahui, kemudian menerapkan aturan Modus Ponens secara berulang untuk menarik kesimpulan baru, hingga tujuan tercapai atau tidak ada lagi aturan baru yang dapat dipicu (*Fixpoint*).

### Algoritma Forward Chaining
1. Inisialisasi himpunan fakta yang diketahui $F = \text{Facts}$.
2. Ulangi:
   - Cari kaidah $R: (\text{Premises} \to C)$ di mana $\text{Premises} \subseteq F$ dan $C \notin F$.
   - Jika tidak ada kaidah yang memenuhi, berhenti (*Titik Tetap Tercapai*).
   - Tambahkan $C$ ke $F$, dan catat $R$ sebagai bukti penurunannya.
   - Jika $C == \text{Query}$, kembalikan status *Sukses*.

*Karakteristik*: Sangat cocok untuk tugas pemantauan (*monitoring*), diagnosis komprehensif, dan perencanaan sintesis.

---

## 13. Backward Chaining (Goal-Driven / Top-Down)

Backward Chaining bekerja dari arah berlawanan: dimulai dari hipotesis tujuan (*Goal / Query*) yang ingin dibuktikan, kemudian mencari kaidah yang memiliki kesimpulan tersebut, dan secara rekursif berusaha membuktikan premis-premisnya sebagai sub-tujuan baru.

### Algoritma Backward Chaining
1. Untuk membuktikan $G$:
   - Jika $G \in \text{Facts}$, kembalikan *True*.
   - Jika $G$ ada dalam tumpukan pemanggilan aktif (*call stack*), **potong cabang rekursi** (terdeteksi siklus!).
   - Cari seluruh kaidah $R$ yang konklusinya adalah $G$.
   - Untuk setiap kaidah $R$, buktikan seluruh premisnya secara rekursif. Jika semua premis terbukti *True*, maka $G$ terbukti *True*.
   - Jika seluruh kaidah gagal, kembalikan *False*.

*Karakteristik*: Sangat efisien jika tujuan spesifik telah diketahui sejak awal (misal: *"Apakah sistem mengalami kegagalan kartu jaringan?"*). Algoritma hanya mengevaluasi kaidah yang relevan dengan tujuan.

---

## 14. Perbandingan Forward vs Backward Chaining

| Dimensi | Forward Chaining (Maju) | Backward Chaining (Mundur) |
|---|---|---|
| **Arah Penalaran** | *Data-Driven* (dari data ke kesimpulan). | *Goal-Driven* (dari hipotesis ke data). |
| **Titik Awal** | Fakta-fakta yang diobservasi pengguna. | Hipotesis atau pertanyaan yang diajukan pengguna. |
| **Eksplorasi Aturan** | Menarik seluruh kesimpulan yang mungkin diturunkan. | Hanya memeriksa aturan yang relevan dengan pembuktian hipotesis. |
| **Sifat Interaksi** | Pasif: Menunggu data lengkap lalu menghasilkan seluruh diagnosis. | Aktif: Menanyakan fakta spesifik hanya saat dibutuhkan (*query on demand*). |
| **Kasus Terbaik** | Sistem monitoring lalu lintas, sistem peringatan dini gempa. | Sistem diagnosis medis klinis, audit kelayakan kredit perbankan. |

---

## 15. Closed-World vs Open-World Assumptions

Asumsi epistemologis penentu interpretasi ketidaktahuan (*absence of information*):

- **Closed-World Assumption (CWA)**: Mengasumsikan bahwa seluruh pengetahuan tentang dunia telah tercatat lengkap di basis pengetahuan. Jika sebuah pernyataan tidak dapat dibuktikan bernilai True dari KB, maka pernyataan tersebut **otomatis dianggap bernilai False**. (Standar sistem pakar klasik dan basis data relasional/SQL).
- **Open-World Assumption (OWA)**: Mengakui bahwa basis pengetahuan mungkin tidak lengkap. Jika sebuah fakta tidak tercatat di KB, statusnya adalah **Unknown (Tidak Diketahui)**, bukan False. (Standar ontologi Semantic Web dan penalaran medis modern).

---

## 16. Keterbatasan Penalaran Simbolik Murni

Meskipun penalaran berbasis kaidah sangat transparan dan dapat diaudit secara hukum (*explainable*), ia menghadapi tantangan besar:
1. **Ketidakpastian Dunia Nyata**: Logika biner sulit menangani fakta yang memiliki derajat probabilitas (misal: *"Pasien kemungkinan 70% terkena demam berdarah"*).
2. **Knowledge Acquisition Bottleneck**: Menulis dan memelihara puluhan ribu kaidah konsisten membutuhkan biaya rekayasa yang sangat masif.
3. **Ketiadaan Adaptabilitas Mandiri**: Sistem tidak dapat memelajari kaidah baru secara induktif dari data mentah tanpa intervensi pakar manusia.

Tantangan probabilitas dan ketidakpastian inilah yang mendorong evolusi AI menuju **Probabilitas dan Statistika** yang akan kita pelajari pada **Chapter 06**.

---

## 17. Ringkasan Modul

- **Knowledge-Based Agent** memisahkan secara tegas antara Basis Pengetahuan (*Knowledge Base*) dan Mesin Inferensi (*Inference Engine*).
- **Logika Proposisional** menyediakan formalisasi semantik ketat untuk fakta atomik menggunakan operator $\neg, \land, \lor, \to, \leftrightarrow$.
- **Model Checking** membuktikan *Logical Entailment* ($KB \models \alpha$) melalui evaluasi seluruh $2^n$ model interpretasi pada tabel kebenaran.
- **Modus Ponens** adalah aturan inferensi deduktif baku: dari $\alpha \to \beta$ dan $\alpha$, tarik kesimpulan $\beta$.
- **Forward Chaining** mengeksekusi penalaran maju dari fakta ke kesimpulan hingga titik tetap (*fixpoint*).
- **Backward Chaining** mengeksekusi pembuktian mundur dari hipotesis ke fakta dasar, dengan pencegahan siklus rekursif.
- **Closed-World Assumption (CWA)** memperlakukan ketiadaan bukti sebagai kegagalan/False.

---

## 18. Latihan Terstruktur

### Latihan Tingkat Dasar (Logika Proposisional)
1. Diberikan proposisi:
   - $P$: "Kabel jaringan terpasang pada port."
   - $Q$: "Lampu indikator ethernet menyala."
   - $R$: "Data dapat ditransmisikan ke switch."
   Tuliskan formula logika proposisional formal untuk pernyataan: *"Kabel jaringan terpasang pada port dan jika lampu indikator menyala maka data dapat ditransmisikan ke switch!"*
2. Susun tabel kebenaran lengkap untuk formula kontraposisi: $(P \to Q) \leftrightarrow (\neg Q \to \neg P)$. Apakah formula tersebut merupakan sebuah tautologi? Jelaskan!

### Latihan Tingkat Menengah (Inferensi Kaidah)
3. Sebuah basis pengetahuan memiliki fakta: $\{A, B\}$.
   Kaidah-kaidah yang tersedia:
   - $R_1: (A \land B) \to C$
   - $R_2: (C \land D) \to E$
   - $R_3: (C) \to F$
   - $R_4: (F \land A) \to G$
   - $R_5: (E) \to H$
   a) Lakukan penelusuran **Forward Chaining** langkah demi langkah hingga mencapai titik tetap (*fixpoint*)! Sebutkan seluruh fakta baru yang berhasil disimpulkan!
   b) Jika diajukan query: *"Apakah G bernilai benar?"*, tunjukkan pohon penjelasannya (*proof tree*)!

### Latihan Tingkat Lanjut (Analisis Kasus Kritis)
4. Diberikan basis aturan siklik:
   - $R_A: B \to A$
   - $R_B: C \to B$
   - $R_C: A \to C$
   Jika basis fakta awal kosong ($\text{Facts} = \emptyset$) dan diajukan query pembuktian hipotesis $A$ menggunakan **Backward Chaining**:
   a) Apa yang terjadi jika algoritma tidak memiliki deteksi siklus (*cycle detection*)?
   b) Jelaskan bagaimana mekanisme tumpukan pemanggilan aktif (*active call stack*) mencegah *infinite recursion* pada kasus ini!
   c) Apa kesimpulan akhir yang dihasilkan di bawah *Closed-World Assumption*?

---

## 19. Pembahasan dan Panduan Jawaban Latihan

### Pembahasan Latihan Tingkat Dasar
1. **Formula Logika**: $P \land (Q \to R)$.
2. **Tabel Kebenaran Kontraposisi**:
   - Untuk semua 4 kemungkinan kombinasi nilai $P$ dan $Q$, nilai $P \to Q$ selalu bernilai identik dengan $\neg Q \to \neg P$ (True pada model (T,T), (F,T), (F,F); False pada model (T,F)).
   - Karena kedua sisi selalu bernilai sama pada seluruh baris, maka $(P \to Q) \leftrightarrow (\neg Q \to \neg P)$ selalu menghasilkan nilai **True** di setiap model.
   - **Kesimpulan**: Formula kontraposisi adalah sebuah **Tautologi** (hukum ekuivalensi logis valid).

### Pembahasan Latihan Tingkat Menengah
3. **Penelusuran Forward Chaining**:
   - Fakta awal: $F_0 = \{A, B\}$.
   - **Iterasi 1**:
     - Kaidah $R_1: (A \land B) \to C$ terpicu karena $\{A, B\} \subseteq F_0$.
     - Fakta baru yang ditarik: $C$.
     - Himpunan fakta terbarui: $F_1 = \{A, B, C\}$.
   - **Iterasi 2**:
     - Kaidah $R_3: (C) \to F$ terpicu karena $\{C\} \subseteq F_1$.
     - Fakta baru yang ditarik: $F$.
     - Himpunan fakta terbarui: $F_2 = \{A, B, C, F\}$.
     - *(Catatan: Kaidah $R_2$ belum bisa terpicu karena fakta $D$ tidak ada)*.
   - **Iterasi 3**:
     - Kaidah $R_4: (F \land A) \to G$ terpicu karena $\{F, A\} \subseteq F_2$.
     - Fakta baru yang ditarik: $G$.
     - Himpunan fakta terbarui: $F_3 = \{A, B, C, F, G\}$.
   - **Iterasi 4**:
     - Tidak ada lagi kaidah yang premisnya terpenuhi. Titik tetap (*fixpoint*) tercapai!
   - a) Fakta-fakta baru yang berhasil disimpulkan: $\{C, F, G\}$.
   - b) **Pohon Penjelasan untuk $G$**:
     - $G$ diturunkan dari $R_4$ dengan premis $F$ dan $A$.
     - $F$ diturunkan dari $R_3$ dengan premis $C$.
     - $C$ diturunkan dari $R_1$ dengan premis $A$ dan $B$.
     - $A$ dan $B$ adalah fakta dasar masukan.

### Pembahasan Latihan Tingkat Lanjut
4. **Analisis Basis Aturan Siklik**:
   - a) Tanpa deteksi siklus: Algoritma akan mencari cara membuktikan $A \implies$ mencoba $R_A$ membutuhkan $B \implies$ mencoba $R_B$ membutuhkan $C \implies$ mencoba $R_C$ membutuhkan $A \implies$ memanggil pembuktian $A$ kembali secara tak berhingga. Akibatnya, program akan mengalami *Stack Overflow Exception* (*Infinite Recursion*).
   - b) Mekanisme *active call stack*: Algoritma mencatat simpul yang sedang aktif dalam rantai pembuktian: $\text{Stack} = \{A, B, C\}$. Ketika kaidah $R_C$ mencoba mengevaluasi sub-goal $A$ yang sudah ada di dalam Stack, algoritma langsung memotong cabang pencarian dan mengembalikan nilai False untuk cabang siklus tersebut.
   - c) Di bawah *Closed-World Assumption*, karena tidak ada fakta dasar yang mendukung siklus tersebut, seluruh kaidah kandidat gagal dibuktikan, sehingga hipotesis $A$ disimpulkan bernilai **False** secara aman tanpa crash.

---

## 20. Tugas Mini Project: Rule-Based Expert System for Network Troubleshooting

### Deskripsi Tugas
Mahasiswa diminta merancang dan mengimplementasikan sebuah prototipe Sistem Pakar Berbasis Aturan (*Rule-Based Expert System*) yang berfungsi mendiagnosis permasalahan konektivitas jaringan komputer pada workstation kantor. Sistem harus mampu menerima masukan gejala fisik/logis dari teknisi, mengeksekusi inferensi logika, dan memberikan rekomendasi tindakan perbaikan beserta penjelasan penalarannya.

### Ruang Lingkup dan 8 Deliverables Wajib
Laporan teknis disusun dalam dokumen akademik (4–6 halaman) dengan memuat:
1. **Definisi Masalah & Ruang Lingkup Diagnosis**: Skenario jaringan lokal (LAN) kantor dengan router gateway, server DHCP, switch, dan DNS resolver.
2. **Katalog Fakta Atomik**: Minimal 8 fakta gejala/observasi teknis yang terdefinisi dengan jelas.
3. **Katalog Kaidah Inferensi (Knowledge Base)**: Minimal 6 aturan IF-THEN yang mencakup relasi kausalitas bertingkat (multi-level inference).
4. **Alur Penalaran & Diagram Pohon Inferensi**: Visualisasi alur penurunan kesimpulan dari fakta awal menuju kesimpulan akhir (*root cause*).
5. **Implementasi Python Runnable**: Menggunakan kerangka kerja berbasis objek yang memuat mesin Forward Chaining atau Backward Chaining (tanpa menggunakan fungsi `eval()`).
6. **Skenario Pengujian Sistem Pakar**:
   - Skenario A: Masalah fisik kabel / kartu jaringan (NIC hardware fault).
   - Skenario B: Masalah konfigurasi IP / DHCP failure.
   - Skenario C: Masalah server DNS luar kantor.
7. **Fasilitas Penjelasan (*Explanation Facility*)**: Kemampuan sistem menjelaskan *"MENGAPA"* suatu kesimpulan diambil berdasarkan kaidah yang terpicu.
8. **Analisis Keterbatasan & Referensi**: Evaluasi batasan pendekatan simbolik murni dan daftar pustaka primer.

---

## 21. Rubrik Penilaian Proyek (Total 100 Poin)

| Kriteria Penilaian | Bobot | Kinerja Unggul (Penuh) | Kinerja Cukup (Parsial) | Kinerja Kurang (Tidak Terpenuhi) |
|---|---:|---|---|---|
| **Problem Definition & Scope** | **15 Poin** | Skenario jaringan terdefinisi sangat tajam, relasi dependensi komponen realistis. (14–15) | Skenario jaringan cukup jelas, cakupan masalah agak sempit. (8–13) | Masalah kabur atau tidak realistis. (0–7) |
| **Knowledge Representation** | **25 Poin** | Fakta atomik presisi; $\ge 6$ kaidah IF-THEN terstruktur dengan relasi bertingkat tanpa inkonsistensi. (22–25) | Fakta dan aturan lengkap, namun relasi penalaran masih bersifat datar/satu lapis. (13–21) | Fakta tidak baku, terdapat kontradiksi logika dalam kaidah. (0–12) |
| **Inference Correctness & Safety** | **25 Poin** | Mesin inferensi mengeksekusi Modus Ponens secara valid (*sound*), bebas dari infinite loop pada siklus, penanganan CWA tepat. (22–25) | Inferensi menghasilkan kesimpulan yang benar, namun penanganan siklus masih rentan. (13–21) | Mesin inferensi salah menarik kesimpulan, menggunakan `eval()` tidak aman. (0–12) |
| **Testing & Explanation Facility** | **20 Poin** | Menguji minimal 3 skenario berbeda; fasilitas penjelasan mampu merinci jejak kaidah pendukung secara transparan. (17–20) | Pengujian hanya 1–2 skenario, penjelasan inferensi kurang terperinci. (10–16) | Tidak ada pengujian skenario, tidak ada fasilitas penjelasan. (0–9) |
| **Dokumentasi & Referensi** | **15 Poin** | Laporan teknis rapi, diagram pohon inferensi jelas, menyertakan minimal 2 referensi primer terverifikasi. (13–15) | Format cukup rapi, diagram ada namun kurang informatif, referensi kurang. (7–12) | Laporan tidak terstruktur, plagiarisme terdeteksi. (0–6) |

---

## 22. Referensi dan Sumber Belajar Primer

1. **Russell, S., & Norvig, P. (2020)**. *Artificial Intelligence: A Modern Approach (4th ed.)*. Pearson. (Bab 7: *Logical Agents* & Bab 9: *Inference in First-Order Logic*). URL: [https://aima.cs.berkeley.edu/](https://aima.cs.berkeley.edu/)
2. **Brachman, R., & Levesque, H. (2004)**. *Knowledge Representation and Reasoning*. Morgan Kaufmann. DOI: [https://doi.org/10.1016/B978-1-55860-932-7.X5000-4](https://doi.org/10.1016/B978-1-55860-932-7.X5000-4)
3. **Giarratano, J. C., & Riley, G. D. (2005)**. *Expert Systems: Principles and Programming (4th ed.)*. Thomson Course Technology.
4. **ACM / IEEE-CS / AAAI (2024)**. *Computer Science Curricula 2023 (CS2023): Artificial Intelligence Knowledge Area*. ACM Press. URL: [https://csed.acm.org/](https://csed.acm.org/)
5. **Shortliffe, E. H. (1976)**. *Computer-Based Medical Consultations: MYCIN*. Elsevier.

