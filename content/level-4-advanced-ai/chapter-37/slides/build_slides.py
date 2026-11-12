"""Slide Deck Generator for Chapter 37: Edge AI & Model Optimization.

Generates 37 slides conforming strictly to deck schema:
- Semantic kinds: cover, content, split, code, case, summary
- Body <= 1000 chars, code blocks <= 18 lines, substantive speaker notes.
"""

import json
import os

slides = [
    {
        "kind": "cover",
        "title": "Edge AI & Model Optimization",
        "body": "On-Device Inference, Kuantisasi FP32-ke-INT8, Weight Pruning, Knowledge Distillation, dan Keterbatasan Hardware Tepi\n\n**Artificial Intelligence Learning Hub — Level 4 (Chapter 37)**\n*Aji Pamungkas Tri Nurcahyo — Institut Teknologi Sumatera*",
        "notes": "Selamat datang di Chapter 37. Bab ini membahas teknik kompresi dan optimasi model jaringan saraf agar mampu berjalan hemat memori serta berlatensi rendah pada perangkat tepi (Edge AI)."
    },
    {
        "kind": "content",
        "title": "Mengapa Edge AI Dibutuhkan?",
        "body": "- **Latensi Real-Time:** Sistem otonom, kamera pengawas industri, dan drone tidak dapat menunggu 150–300 ms perjalanan jaringan ke server cloud.\n- **Privasi Data Lokal:** Pemrosesan biometrik atau citra medis langsung di perangkat mencegah pengiriman data pribadi ke luar.\n- **Ketangguhan Offline:** Stasiun sensor pertanian pintar (*smart agriculture*) tetap beroperasi tanpa konektivitas internet stabil.",
        "notes": "Jelaskan motivasi ekonomi dan teknis di balik pemindahan inferensi dari pusat data awan ke perangkat dekat sumber data."
    },
    {
        "kind": "content",
        "title": "Tujuan Pembelajaran Bab 37",
        "body": "1. **Membedakan** arsitektur Cloud Inference vs On-Device Edge Inference.\n2. **Mengukur** *parameter count*, ukuran artefak `.pt`, serta latensi P50/P95 aktual.\n3. **Menghitung** transformasi matematis *Quantization* FP32 $\\to$ INT8 (*scale*, *zero point*, galat).\n4. **Menerapkan** Dynamic & Post-Training Static Quantization di PyTorch CPU.\n5. **Membandingkan** *Unstructured* vs *Structured Pruning* dan *Knowledge Distillation*.\n6. **Menganalisis** kesesuaian model terhadap batasan SRAM/Flash mikrokontroler.",
        "notes": "Pastikan mahasiswa memahami bahwa optimasi selalu menyangkut kompromi antara ukuran, latensi, dan akurasi."
    },
    {
        "kind": "split",
        "title": "Cloud Inference vs On-Device Edge AI",
        "body": "- **Cloud Inference:**\n  - Sumber daya hampir tak terbatas (GPU/TPU besar).\n  - Membutuhkan jaringan kontinu, biaya bandwidth, dan berisiko paparan privasi.\n- **On-Device Edge AI:**\n  - Eksekusi lokal di dekat sensor.\n  - Dibatasi ketat oleh **RAM (KB–MB)**, **Flash storage**, frekuensi CPU, serta **anggaran daya baterai (mW–W)**.",
        "notes": "Tidak semua tugas cocok dipindah ke edge; sering kali arsitektur hibrida (edge deteksi awal + cloud analisis berat) menjadi pilihan terbaik."
    },
    {
        "kind": "content",
        "title": "Kategori Perangkat Keras Edge AI",
        "body": "1. **Mikrokontroler (MCU - TinyML):** ARM Cortex-M7, ESP32-S3. RAM hanya ratusan KB; tanpa sistem operasi penuh.\n2. **Single-Board Computers (SBC):** Raspberry Pi 4/5, Jetson Nano. Embedded Linux, RAM 2–8 GB.\n3. **Mobile SoC & NPU Khusus:** Prosesor ponsel dengan akselerator matriks INT8 (*Neural Processing Unit*).\n- *Aturan Kejujuran:* Simulasi di CPU PC tidak membuktikan kelayakan fisik tanpa pengujian di target hardware.",
        "notes": "Tekankan pemisahan tegas antara CPU simulation dan physical hardware-in-the-loop testing."
    },
    {
        "kind": "content",
        "title": "Keterbatasan Utama Sistem Embedded",
        "body": "- **SRAM Terbatas (*Tensor Arena*):** Berbeda dengan server ber-RAM ratusan GB, MCU hanya menyediakan 256 KB – 512 KB SRAM untuk menampung aktivasi kerja (*intermediate tensors*).\n- **Flash Storage:** Kapasitas penyimpan bobot statis (1–4 MB) yang juga harus berbagi ruang dengan firmware RTOS dan tumpukan WiFi.\n- **Dukungan Operator:** Banyak operator kompleks PyTorch tidak didukung oleh runtime *embedded* seperti TFLite Micro.",
        "notes": "Tensor Arena adalah alokasi memori statis pre-alokasi agar tidak terjadi fragmentasi heap selama inferensi real-time."
    },
    {
        "kind": "content",
        "title": "Baseline Model Profiling (Hasil Aktual)",
        "body": "Model Acuan `EdgeTeacherMLP (16 -> 256 -> 128 -> 64 -> 4)` pada dataset sensor IoT 4-kelas:\n- **Parameter Count:** `45,764` parameter.\n- **Raw FP32 Weight RAM:** `178.77 KB` (4 byte per parameter).\n- **Serialized `.pt` Disk Size:** `183.94 KB`.\n- **Test Accuracy:** `92.33%` (N=600).\n- **Warm CPU Latency:** Mean `0.0779 ms` | P95 `0.1025 ms` (Windows 11, 32 logical CPUs).",
        "notes": "Seluruh perbandingan optimasi pada bab ini diukur terhadap model acuan FP32 ini."
    },
    {
        "kind": "content",
        "title": "Alur Kerja Kompresi & Optimasi Model",
        "body": "```\n[FP32 Baseline (183.94 KB)] --+--> [1. Quantization (INT8)] ---> 52.6 - 62.7 KB\n                              |\n                              +--> [2. Weight Pruning]       ---> 54.6 KB (Struct)\n                              |\n                              +--> [3. Knowledge Distill.]   ---> 8.82 KB (Student)\n```",
        "notes": "Tiga keluarga besar kompresi model: mengurangi presisi bit (Quantization), menghapus bobot redundan (Pruning), dan melatih model kecil baru (Distillation)."
    },
    {
        "kind": "content",
        "title": "Pengantar Kuantisasi (Quantization)",
        "body": "- **Quantization** memetakan bilangan pecahan kontinu berpresisi tinggi (**FP32**, 32-bit) menjadi himpunan bilangan bulat diskret berpresisi rendah (**INT8**, 8-bit).\n- **Keuntungan Teoretis:**\n  - Reduksi jejak memori bobot hingga **$4\\times$** (32-bit $\\to$ 8-bit).\n  - Efisiensi transfer cache/bus memori.\n  - Instruksi aritmatika integer (*SIMD / VNNI / NEON*) pada perangkat keras yang mendukung.",
        "notes": "Penting diingat: tidak semua nilai FP32 dapat direpresentasikan secara persis dalam 256 tingkat INT8, sehingga pasti timbul galat kuantisasi."
    },
    {
        "kind": "split",
        "title": "Perbandingan Format FP32, FP16, dan INT8",
        "body": "- **FP32 (Single Precision, 4 byte):** 1 bit tanda + 8 bit eksponen + 23 bit mantisa. Rentang dinamis masif ($\\approx \\pm 3.4\\times 10^{38}$).\n- **FP16 (Half Precision, 2 byte):** 1 bit tanda + 5 bit eksponen + 10 bit mantisa. Memotong ukuran $2\\times$.\n- **INT8 (Signed 8-bit Integer, 1 byte):** Hanya menampung bilangan bulat diskret $[-128, 127]$ sebanyak 256 tingkat.",
        "notes": "Karena INT8 hanya memiliki 256 angka yang mungkin, penentuan skala pemetaan (scale dan zero point) menjadi krusial."
    },
    {
        "kind": "content",
        "title": "Matematika Kuantisasi: Scale & Zero Point",
        "body": "- Pemetaan linear dari nilai riil $x$ menuju integer terkuantisasi $q$ dan sebaliknya (*dequantization*):\n$$q = \\text{clip}\\left(\\text{round}\\left(\\frac{x}{s}\\right) + z,\\; q_{\\min},\\; q_{\\max}\\right), \\quad \\hat{x} = s \\cdot (q - z)$$\n- **Symmetric ($z = 0$):** $s = \\frac{\\max(|x|)}{127}$ untuk $[-127, 127]$. Cepat secara komputasi.\n- **Asymmetric ($z \\neq 0$):** $s = \\frac{x_{\\max} - x_{\\min}}{255}$ untuk $[-128, 127]$. Memanfaatkan penuh 256 tingkat pada aktivasi miring (*skewed* seperti ReLU).",
        "notes": "Worked example pada [-0.50, 0.00, 0.25, 1.20, 2.54, 4.00] membuktikan Asymmetric memiliki MSE jauh lebih kecil karena tidak membuang rentang negatif."
    },
    {
        "kind": "content",
        "title": "Galat Kuantisasi, Outlier & Granularitas",
        "body": "- **Ancaman Outlier:** Satu bobot tunggal bernilai `15.0` memaksa skala $s$ membesar drastis ($0.11811$), sehingga bobot normal $[-0.3, 0.3]$ terkompresi menjadi hanya beberapa bilangan bulat (*SQNR anjlok*).\n- **Per-Tensor vs Per-Channel:**\n  - *Per-Tensor:* Satu $(s, z)$ untuk seluruh matriks bobot.\n  - *Per-Channel:* $(s_c, z_c)$ terpisah bagi setiap baris saluran keluar, menyelamatkan presisi saluran berskala kecil.",
        "notes": "Eksperimen di quantization_error.py menunjukkan SQNR saluran kecil pulih drastis saat beralih dari Per-Tensor ke Per-Channel."
    },
    {
        "kind": "content",
        "title": "Post-Training Dynamic Quantization",
        "body": "- **Mekanisme:** Bobot `nn.Linear` dikonversi statis menjadi `INT8` saat konversi model, sedangkan aktivasi tetap dibaca `FP32` dan **dikuantisasi secara dinamis (*on-the-fly*)** pada saat inferensi.\n- **Kelebihan:** Tanpa perlu data kalibrasi; ukuran `.pt` menyusut dari `183.94 KB` $\\to$ **`52.63 KB` ($3.49\\times$ lebih kecil)** dengan akurasi terjaga `92.17%`.\n- **Kelemahan:** Overhead mencari `min/max` aktivasi secara dinamis saat runtime.",
        "notes": "Dynamic quantization paling efektif pada model berukuran besar seperti LSTM atau Transformer Linear layers di mana bandwidth memori menjadi bottleneck."
    },
    {
        "kind": "split",
        "title": "Post-Training Static Quantization",
        "body": "- **Mekanisme:** Baik bobot maupun **skala aktivasi** dikunci sebelumnya (*pre-computed*) ke `INT8` menggunakan `QuantStub` dan `DeQuantStub`.\n- Menghindari overhead kalkulasi min/max aktivasi saat runtime.\n- Membutuhkan tahap **Kalibrasi** menggunakan kumpulan data representatif.",
        "notes": "Pada backend onednn PyTorch 2.15, ukuran .pt menyusut menjadi 62.70 KB (2.93x lebih kecil) dan mempertahankan akurasi uji 92.67%."
    },
    {
        "kind": "code",
        "title": "Alur Kalibrasi Static Quantization (PyTorch)",
        "body": "```python\nimport torch.ao.quantization as tq\n\nstatic_m.qconfig = tq.get_default_qconfig(\"onednn\")\ntq.prepare(static_m, inplace=True)  # Sisipkan observer min/max\n\n# Jalankan kalibrasi tanpa label pada ~250 sampel representatif\nwith torch.no_grad():\n    static_m(X_train[:250])\n\nstatic_q_model = tq.convert(static_m, inplace=False)\n```",
        "notes": "Catatan versi: torch.ao.quantization memancarkan DeprecationWarning di PyTorch 2.15 dengan jalur migrasi masa depan menuju torchao."
    },
    {
        "kind": "content",
        "title": "Pengantar Quantization-Aware Training (QAT)",
        "body": "- Jika Post-Training Quantization menurunkan akurasi secara drastis (sering pada model super kompak seperti MobileNetV1/INT4), gunakan **QAT**.\n- **Fake Quantization:** Menyisipkan simulasian pembulatan kuantisasi ke dalam grafik *forward pass* selama pelatihan, sementara gradien tetap diperbarui dalam `FP32` menggunakan *Straight-Through Estimator* (STE).\n- Model belajar mengkompensasi derau kuantisasi langsung di bobotnya.",
        "notes": "QAT menuntut komputasi pelatihan ulang lebih tinggi dibanding PTQ, tetapi memberikan akurasi tertinggi di presisi rendah."
    },
    {
        "kind": "content",
        "title": "Weight Pruning: Pemangkasan Bobot Redundan",
        "body": "- **Hipotesis Lottery Ticket / Redundansi:** Jaringan saraf padat biasanya penuh dengan bobot bernilai mendekati nol yang berkontribusi sangat kecil terhadap keluaran.\n- **Magnitude-Based Pruning:** Menghitung $|w_{ij}|$, lalu mengubah bobot di bawah ambang batas tertentu menjadi `0.0` melalui masker biner (*sparsity ratio*).\n- Tujuan: Mengurangi kompleksitas model dengan kerusakan akurasi minimal.",
        "notes": "Pemangkasan agresif hampir selalu diikuti oleh fine-tuning singkat (beberapa epoch) agar bobot tersisa beradaptasi."
    },
    {
        "kind": "split",
        "title": "Structured vs Unstructured Pruning",
        "body": "- **Unstructured Pruning:** Menghapus bobot individu acak di mana saja $\\to$ menghasilkan matriks jarang (*sparse*) berpola tak beraturan.\n- **Structured Pruning:** Menghapus seluruh **neuron, baris, atau filter konvolusi** ber-norma L1 terendah sekaligus $\\to$ langsung mengubah dimensi arsitektur menjadi jaringan padat yang benar-benar lebih ramping!",
        "notes": "Perbedaan struktural ini menentukan apakah ukuran file fisik dan kecepatan inferensi benar-benar berkurang di CPU biasa."
    },
    {
        "kind": "content",
        "title": "Realitas Unstructured Pruning di CPU Standar",
        "body": "- **Peringatan Kritis:** Mengubah 70% bobot menjadi nol lewat *Unstructured Pruning* **TIDAK OTOMATIS mengecilkan berkas `.pt` ataupun mempercepat CPU GEMM!**\n- **Bukti Eksperimen (`magnitude_pruning.py`):**\n  - FP32 Baseline: `183.94 KB`, Mean Latency `0.0592 ms`.\n  - Unstructured Pruned (70% nol): **`183.94 KB` (rasio kompresi mentah $1.00\\times$)**.\n- Membutuhkan format indeks jarang (*CSR/CSC*), *gzip* (`70.27 KB`), atau akselerator *sparse hardware* agar ruang benar-benar hemat.",
        "notes": "Sebaliknya, Structured Pruning memangkas 50% neuron tersembunyi (45.7K -> 12.6K param) dan mengecilkan .pt fisik menjadi 54.57 KB."
    },
    {
        "kind": "content",
        "title": "Sensitivitas Sparsity vs Akurasi",
        "body": "Hasil sapuan kejarangan (`pruning_sparsity_analysis.py`):\n- **0% Sparsity:** Akurasi `92.33%`\n- **50% Sparsity:** Zero-Shot `90.33%` $\\to$ +5-Epoch FT **`92.00%+`**\n- **70% Sparsity:** Zero-Shot `87.17%` $\\to$ +5-Epoch FT **`89.17%`**\n- **95% Sparsity:** Kapasitas jaringan runtuh parah jika seluruh koneksi vital terputus.",
        "notes": "Fine-tuning pemulihan terbukti sangat efektif memulihkan akurasi pada tingkat kejarangan 50%-70%."
    },
    {
        "kind": "content",
        "title": "Knowledge Distillation (Guru–Murid)",
        "body": "- **Konsep (Hinton et al., 2015):** Alih-alih mengompres jaringan besar yang sudah jadi, kita melatih jaringan kecil baru (**Student Model**) agar meniru perilaku jaringan besar yang akurat (**Teacher Model**).\n- **Soft Targets (*Dark Knowledge*):** Probabilitas keluaran Guru mengungkapkan informasi kemiripan antarkelas yang tidak pernah ada dalam label keras one-hot `[1, 0, 0, 0]`.",
        "notes": "Pastikan bobot Teacher selalu dibekukan (requires_grad=False) selama proses distilasi."
    },
    {
        "kind": "content",
        "title": "Arsitektur Teacher vs Compact Student",
        "body": "- **Teacher Model (`EdgeTeacherMLP`):** `16 -> 256 -> 128 -> 64 -> 4` (**45.764 parameter**, `.pt` = `183.94 KB`).\n- **Student Model (`EdgeStudentMLP`):** `16 -> 32 -> 16 -> 4` (**1.140 parameter**, `.pt` = **`8.82 KB`** — **$40\\times$ lebih sedikit parameter** dan **$20.85\\times$ lebih kecil di disk**).\n- Student yang dilatih dengan distilasi ($T=3.0, \\alpha=0.6$) mencapai akurasi uji **`91.33%`** (hanya selisih `1.00%` dari Teacher!).",
        "notes": "Student 8.82 KB sangat kecil hingga bahkan mudah muat dalam memori SRAM mikrokontroler kelas TinyML."
    },
    {
        "kind": "content",
        "title": "Temperature Scaling pada Softmax",
        "body": "- Probabilitas lembut dihitung dengan membagi logit $z_i$ menggunakan suhu $T > 1$:\n$$p_i = \\frac{\\exp(z_i / T)}{\\sum_j \\exp(z_j / T)}$$\n- **Worked Example Logits `[4.0, 2.0, 0.5, -1.0]`:**\n  - **$T = 1.0$ (Tajam):** `[0.8530, 0.1154, 0.0258, 0.0057]`\n  - **$T = 3.0$ (Lembut):** `[0.4966, 0.2550, 0.1546, 0.0938]`\n- Kelas kedua meningkat informasinya dari $0.1154 \\to 0.2550$!",
        "notes": "Suhu tinggi meratakan puncak distribusi sehingga sinyal keserupaan antar-kelas terlihat jelas oleh Student."
    },
    {
        "kind": "content",
        "title": "Fungsi Rugi Knowledge Distillation ($T^2$ Scaling)",
        "body": "- Menggabungkan **KL Divergence** (melawan target lembut Guru) dan **Cross-Entropy** (melawan label asli):\n$$\\mathcal{L}_{\\text{total}} = \\alpha \\cdot T^2 \\cdot D_{\\text{KL}}\\left(\\sigma\\left(\\frac{z_T}{T}\\right) \\;\\middle\\|\\; \\sigma\\left(\\frac{z_S}{T}\\right)\\right) + (1 - \\alpha) \\cdot \\text{CE}(y, z_S)$$\n- **Mengapa dikali $T^2$?** Karena turunan logit yang dibagi $T$ menyusut dengan faktor $1/T^2$; perkalian $T^2$ menjaga keseimbangan besaran gradien kedua suku!",
        "notes": "Tanpa faktor skala T^2, pengaruh sinyal guru akan melemah drastis saat suhu T diperbesar."
    },
    {
        "kind": "content",
        "title": "Kartu Skor Komparasi Optimasi (Hasil Aktual)",
        "body": "Pengukuran aktual pada Windows 11 CPU (32 logical CPUs, PyTorch `2.15.0+cpu`):\n\n| Strategi Optimasi | Test Acc | `.pt` Size | Rasio Ukuran | Mean Lat (ms) |\n| :--- | :---: | :---: | :---: | :---: |\n| **FP32 Baseline** | **92.33%** | 183.94 KB | 1.00x | 0.0779 ms |\n| **Dynamic INT8** | 92.17% | 52.63 KB | **3.49x** | 1.0943 ms |\n| **Static INT8** | **92.67%** | 62.70 KB | **2.93x** | 2.0225 ms |\n| **Unstruct Pruned (70%)** | 89.17% | 183.94 KB | 1.00x | 0.0724 ms |\n| **Struct Pruned (50%)** | 89.67% | 54.57 KB | **3.37x** | 0.0714 ms |\n| **Distilled Student** | 91.33% | **8.82 KB** | **20.85x** | **0.0503 ms** |",
        "notes": "Tabel ini 100% hasil eksekusi nyata di optimization_comparison.py dan model_size_comparison.py."
    },
    {
        "kind": "content",
        "title": "Analisis Parameter Count",
        "body": "- **FP32 Baseline & Quantized & Unstruct Pruned:** Seluruhnya tetap memiliki **45.764** posisi bobot (kuantisasi mengecilkan bit per bobot; *unstructured pruning* mengubah nilai ke nol).\n- **Structured Pruned (50% neuron):** Menyusut secara arsitektural menjadi **12.644 parameter** ($3.62\\times$ lebih sedikit).\n- **Distilled Student:** Hanya **1.140 parameter** ($40.1\\times$ lebih sedikit).",
        "notes": "Bedakan pengurangan jumlah parameter (parameter reduction) dari pengurangan lebar bit (bit-width reduction)."
    },
    {
        "kind": "content",
        "title": "Analisis Ukuran Artefak Disk (Raw vs Gzip)",
        "body": "- **Dynamic INT8:** `52.63 KB` mentah ($3.49\\times$) $\\to$ `43.99 KB` gzip.\n- **Unstructured Pruned (70%):** `183.94 KB` mentah ($1.00\\times$), namun setelah dikompresi `gzip` turun tajam ke **`70.27 KB`** karena banyaknya pengulangan pola nol.\n- **Distilled Student:** Hanya `8.82 KB` mentah ($20.85\\times$) dan `6.02 KB` gzip!",
        "notes": "Hasil gzip membuktikan bahwa bobot jarang memiliki entropi rendah namun memerlukan encoder khusus agar hemat ruang penyimpanan."
    },
    {
        "kind": "content",
        "title": "Interpretasi Kejujuran Latensi Inferensi",
        "body": "- **Fakta Empiris:** Pada model MLP kecil ($45$K param, batch=1) di CPU x86 Windows, **Dynamic INT8 (`1.09 ms`) dan Static INT8 (`2.02 ms`) justru lebih lambat dari FP32 (`0.078 ms`)** akibat overhead konversi stub dan dispatch kernel `onednn`!\n- **Kecepatan Nyata:** Diperoleh melalui penyusutan dimensi matriks pada **Distilled Student (`0.0503 ms`)** dan **Structured Pruning (`0.0714 ms`)**.\n- *Pelajaran:* **Jangan pernah mengasumsikan kuantisasi pasti mempercepat CPU tanpa pengukuran berulang.**",
        "notes": "Kuantisasi INT8 baru memberikan speedup latensi besar pada model CNN/Transformer besar atau pada NPU/DSP perangkat keras yang dirancang khusus untuk integer."
    },
    {
        "kind": "content",
        "title": "Jejak Memori Puncak & Konteks Hardware",
        "body": "- Menggunakan `tracemalloc`, satu *forward pass* single-sample memakan alokasi heap sangat kecil (`< 20 KB`), namun di perangkat mikrokontroler memori *Tensor Arena* harus menampung aktivasi terbesar sekaligus bobot.\n- Konteks Pengukuran:\n  - OS: `Windows-11-10.0.26300-SP0` | `32 Logical CPUs` | PyTorch `2.15.0+cpu`.\n  - Setiap angka latensi dirata-rata dari 3 pengulangan $\\times$ 400 eksekusi setelah 50 *warm-up*.",
        "notes": "Pembuangan warm-up dan pengulangan uji menjamin stabilitas hasil benchmark dari gangguan penjadwal OS."
    },
    {
        "kind": "content",
        "title": "Trade-Off Akurasi vs Ukuran vs Latensi",
        "body": "- Tidak ada satu teknik yang menang di semua dimensi:\n  - Butuh akurasi tertinggi & hemat memori penyimpanan $3.5\\times$ pada arsitektur tetap? $\\to$ **Static / Dynamic INT8** (`92.67%` / `92.17%`).\n  - Butuh jejak RAM sekecil mungkin & inferensi sub-50 mikrodetik untuk mikrokontroler? $\\to$ **Knowledge Distillation** (`8.82 KB`, `91.33%`, `0.0503 ms`).",
        "notes": "Gunakan grafik multi-panel optimization_tradeoffs.png di kelas untuk mendiskusikan batas Pareto."
    },
    {
        "kind": "case",
        "title": "Kesesuaian Simulasi Terhadap Target Mikrokontroler",
        "body": "Evaluasi simulasi anggaran memori (`edge_hardware_constraints.py`):\n- **ARM Cortex-M7 (256 KB SRAM, 1 MB Flash):** `FP32 Teacher` (179.5 KB bobot + 120 KB overhead runtime) hampir menyentuh batas RAM/Flash; **`Distilled Student` (4.5 KB)** dan **`INT8` (45.9 KB)** muat sangat leluasa.\n- **Heavy Vision CNN (ResNet18, 44 MB):** Gagal total di MCU (*RAM/Flash overflow*), wajib menggunakan SBC seperti Raspberry Pi 4.\n- *(Catatan: Validasi simulasi, bukan uji flashing perangkat fisik).*",
        "notes": "Tekankan pentingnya menghitung overhead runtime TFLite Micro (~120 KB) dan RTOS WiFi stack (~220 KB di ESP32)."
    },
    {
        "kind": "content",
        "title": "Keterbatasan Fundamental Edge AI",
        "body": "1. **Batas Kapasitas Model:** Model bahasa miliaran parameter (LLM) mustahil berjalan utuh pada mikrokontroler ber-RAM kilobyte.\n2. **Ketiadaan Pembaruan di Tempat (*No On-Device Backprop*):** Perangkat tepi biasanya murni menjalankan inferensi baca-saja karena keterbatasan daya dan presisi integer.\n3. **Degradasi Pergeseran Domain:** Model edge sulit diperbarui masal secara serentak ketika ribuan sensor di lapangan mengalami *Concept Drift*.",
        "notes": "Kaitkan keterbatasan pembaruan edge ini dengan tantangan MLOps terdistribusi."
    },
    {
        "kind": "code",
        "title": "Demonstrasi Kode Python: Distilasi Guru–Murid",
        "body": "```python\n# Pembekuan guru & pelatihan murid dengan KL Divergence bertemperatur T=3.0\nwith torch.no_grad():\n    soft_targets = F.softmax(teacher(xb) / T, dim=1)\n\nlog_soft_student = F.log_softmax(student(xb) / T, dim=1)\nloss_kd = kl_loss_fn(log_soft_student, soft_targets) * (T ** 2)\nloss_ce = ce_loss_fn(student(xb), yb)\ntotal_loss = alpha * loss_kd + (1.0 - alpha) * loss_ce\n```",
        "notes": "Tunjukkan bagaimana hanya dalam 10 baris kode, Knowledge Distillation mentransfer pengetahuan model guru ke arsitektur murid yang 40x lebih ramping."
    },
    {
        "kind": "case",
        "title": "Studi Kasus Mini Project Bab 37",
        "body": "- **Judul:** *Optimizing a Neural Network for CPU-Based Edge Inference.*\n- Mahasiswa membangun model baseline, menguji kuantisasi FP32 $\\to$ INT8, pemangkasan terstruktur/tak terstruktur, serta *Knowledge Distillation*, lalu menyusun laporan benchmark ukuran artefak, latensi berulang, akurasi, dan kelayakan anggaran mikrokontroler.",
        "notes": "Rubrik 100 poin: Baseline (15), Implementasi Optimasi (30), Benchmarking Latensi/Ukuran (25), Analisis Trade-off & Hardware (20), Dokumentasi (10)."
    },
    {
        "kind": "content",
        "title": "Diskusi Reflektif Kelas",
        "body": "1. Jika hasil benchmark menunjukkan *Dynamic INT8* membuat model Anda `2x` lebih lambat di CPU, kondisi apa yang membuat Anda tetap memilih menyimpannya dalam format terkuantisasi?\n2. Mengapa **Structured Pruning** mampu langsung mempercepat perkalian matriks CPU tanpa mengubah kernel BLAS, sedangkan **Unstructured Pruning** tidak?",
        "notes": "Jawaban 1: Apabila keterbatasan penyimpanan Flash/RAM atau bandwidth jaringan distribusi firmware lebih kritis dari waktu CPU tunggal."
    },
    {
        "kind": "summary",
        "title": "Ringkasan Pembelajaran Bab 37",
        "body": "- **Edge AI** memindahkan inferensi ke perangkat lokal demi latensi rendah, privasi data, dan keandalan offline.\n- **Quantization (FP32 $\\to$ INT8)** memangkas memori $3-3.5\\times$ lewat skala dan *zero point*, tetapi speedup CPU bergantung pada ukuran matriks & dukungan kernel.\n- **Structured Pruning** memangkas neuron fisik, sedangkan **Unstructured Pruning** membutuhkan format jarang khusus.\n- **Knowledge Distillation** mencapai kompresi tertinggi (**$20.85\\times$ lebih kecil, $91.33\\%$ akurasi**).",
        "notes": "Bab 38 selanjutnya akan memperluas arsitektur AI untuk menggabungkan modalitas teks, citra, dan sensor dalam Multimodal AI Systems."
    },
    {
        "kind": "content",
        "title": "Referensi Akademik Terverifikasi",
        "body": "1. **Hinton, G., Vinyals, O., & Dean, J. (2015).** *Distilling the Knowledge in a Neural Network.* arXiv:1503.02531.\n2. **Jacob, B., et al. (2018).** *Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference.* CVPR 2018.\n3. **Han, S., Mao, H., & Dally, W. J. (2016).** *Deep Compression: Compressing Deep Neural Networks with Pruning, Trained Quantization and Huffman Coding.* ICLR 2016.\n4. **Warden, P., & Situnayake, D. (2019).** *TinyML: Machine Learning with TensorFlow Lite on Arduino and Ultra-Low-Power Microcontrollers.* O'Reilly.\n5. **PyTorch Team. (2024).** *PyTorch Quantization & Pruning Documentation.* pytorch.org.",
        "notes": "Seluruh referensi merupakan publikasi seminal di bidang kompresi model dan TinyML."
    }
]


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    out_file = os.path.join(script_dir, "deck.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(slides, f, indent=2, ensure_ascii=False)
    print(f"Generated {len(slides)} slides to {out_file}")


if __name__ == "__main__":
    main()

