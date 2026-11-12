# Katalog Kode Praktik Chapter 05: Knowledge Representation & Reasoning

Repositori ini memuat implementasi runnable dari komponen-komponen inti penalaran logika simbolik dan sistem pakar berbasis aturan:
1. `facts_and_rules.py`: Struktur data representasi Fact, Rule, dan Knowledge Base (KB).
2. `truth_table.py`: Pembangkit tabel kebenaran lengkap dan verifikasi *Logical Entailment* ($KB \models \alpha$) via model checking.
3. `modus_ponens.py`: Aturan inferensi deduktif klasik dan Generalized Modus Ponens.
4. `forward_chaining.py`: Mesin inferensi Forward Chaining (*bottom-up/data-driven*) dengan fasilitas penjelasan (*How Facility*).
5. `backward_chaining.py`: Mesin inferensi Backward Chaining (*top-down/goal-driven*) dengan proteksi ketergantungan siklik (*cycle cut*).

---

## Petunjuk Menjalankan

Masuk ke direktori kode praktik Chapter 05:

```bash
cd content/level-1-ai-foundations/chapter-05/code

# Menjalankan masing-masing modul
python facts_and_rules.py
python truth_table.py
python modus_ponens.py
python forward_chaining.py
python backward_chaining.py
```

---

## Log Output Eksekusi Aktual (Terverifikasi Python 3.12.10)

### 1. Model Checking Entailment & Truth Table (`truth_table.py`)
```text
TABEL KEBENARAN: P -> Q
-----------------------
  P   |   Q   || P -> Q
-----------------------
True  | True  ||    True   
True  | False ||   False   
False | True  ||    True   
False | False ||    True   
-----------------------
Sifat Formula: Tautologi = False, Satisfiable = True

TABEL KEBENARAN: NOT(P AND Q) <-> (NOT P OR NOT Q)
--------------------------------------------------
  P   |   Q   || NOT(P AND Q) <-> (NOT P OR NOT Q)
--------------------------------------------------
True  | True  ||    True   
True  | False ||    True   
False | True  ||    True   
False | False ||    True   
--------------------------------------------------
Sifat Formula: Tautologi = True, Satisfiable = True

Uji Model Checking: KB { P, P -> Q } |= Q (Modus Ponens): True
```

### 2. Forward Chaining Engine (`forward_chaining.py`)
```text
--- MEMULAI INFERENSI FORWARD CHAINING ---
[Iterasi 1] 2 kaidah terpicu:
  -> Firing [R1] IF (cable_plugged AND link_light_off) THEN nic_hardware_fault ==> Menyimpulkan: 'nic_hardware_fault'
  -> Firing [R2] IF (link_light_off) THEN interface_down ==> Menyimpulkan: 'interface_down'
[Iterasi 2] 1 kaidah terpicu:
  -> Firing [R3] IF (interface_down AND ping_gateway_fails) THEN local_network_disconnected ==> Menyimpulkan: 'local_network_disconnected'
[Iterasi 3] 1 kaidah terpicu:
  -> Firing [R4] IF (local_network_disconnected) THEN internet_inaccessible ==> Menyimpulkan: 'internet_inaccessible'

[SUKSES] Query 'internet_inaccessible' berhasil dibuktikan pada iterasi 3!

Hasil Pengujian Query 'internet_inaccessible': TERBUKTI (TRUE)
Fakta-fakta Baru yang Ditarik: ['nic_hardware_fault', 'interface_down', 'local_network_disconnected', 'internet_inaccessible']

=== FASILITAS PENJELASAN (HOW) UNTUK: 'internet_inaccessible' ===
Fakta 'internet_inaccessible' diturunkan melalui Kaidah [R4] IF (local_network_disconnected) THEN internet_inaccessible:
  Premis yang mendukung: ['local_network_disconnected']
    * 'local_network_disconnected' diturunkan dari Kaidah [R3] IF (interface_down AND ping_gateway_fails) THEN local_network_disconnected
```

### 3. Backward Chaining Engine & Cyclic Safety (`backward_chaining.py`)
```text
Membuktikan Hipotesis Utama: 'internet_inaccessible'
[?] Menguji Sub-Goal: 'internet_inaccessible'
  [!] Mencoba Kaidah [R4] IF (local_network_disconnected) THEN internet_inaccessible:
  [?] Menguji Sub-Goal: 'local_network_disconnected'
    [!] Mencoba Kaidah [R3] IF (interface_down AND ping_gateway_fails) THEN local_network_disconnected:
    [?] Menguji Sub-Goal: 'interface_down'
      [!] Mencoba Kaidah [R2] IF (link_light_off) THEN interface_down:
      [?] Menguji Sub-Goal: 'link_light_off'
        => SUKSES: 'link_light_off' adalah FAKTA DIKETAHUI.
      => SUKSES: Kaidah [R2] IF (link_light_off) THEN interface_down BERHASIL MEMBUKTIKAN 'interface_down'!
    [?] Menguji Sub-Goal: 'ping_gateway_fails'
      => SUKSES: 'ping_gateway_fails' adalah FAKTA DIKETAHUI.
    => SUKSES: Kaidah [R3] IF (interface_down AND ping_gateway_fails) THEN local_network_disconnected BERHASIL MEMBUKTIKAN 'local_network_disconnected'!
  => SUKSES: Kaidah [R4] IF (local_network_disconnected) THEN internet_inaccessible BERHASIL MEMBUKTIKAN 'internet_inaccessible'!

Hasil Akhir Pembuktian: HIPOTESIS TERBUKTI (TRUE)

--- UJI PENANGANAN KETERGANTUNGAN SIKLIK (CYCLIC DEPENDENCIES) ---
[?] Menguji Sub-Goal: 'A'
  [!] Mencoba Kaidah [R_A] IF (B) THEN A:
  [?] Menguji Sub-Goal: 'B'
    [!] Mencoba Kaidah [R_B] IF (A) THEN B:
    [?] Menguji Sub-Goal: 'A'
      => PERINGATAN: Siklus terdeteksi pada 'A'! Memotong cabang rekursi.
    => GAGAL: Seluruh kaidah untuk 'B' tidak dapat dipenuhi.
  => GAGAL: Seluruh kaidah untuk 'A' tidak dapat dipenuhi.
Hasil Pembuktian Siklik: GAGAL DENGAN AMAN (BEBAS INFINITE LOOP)
```

