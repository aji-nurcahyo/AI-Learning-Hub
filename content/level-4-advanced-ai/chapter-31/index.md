# Bab 31: Transfer Learning & Fine-Tuning

## Paradigma Adaptasi Model Modern: Dari Ekstraksi Fitur Klasik hingga Parameter-Efficient Fine-Tuning (LoRA)

---

## 1. Hakikat dan Landasan Transfer Learning

Dalam rekayasa sistem kecerdasan buatan terapan (*applied AI systems*), melatih arsitektur jaringan saraf tiruan mendalam dari nol (*training from scratch*) sering kali menjadi strategi yang tidak realistis. Pendekatan ini menuntut jutaan data beranotasi tinggi, ratusan jam komputasi akselerator perangkat keras (GPU/TPU), serta konsumsi energi yang sangat besar. Sebaliknya, praktisi industri hampir selalu dihadapkan pada regim data terbatas (*low-data regime*) dengan batasan komputasi yang ketat.

**Transfer Learning** merevolusi lanskap ini dengan membalikkan paradigma pelatihan: alih-alih mempelajari representasi fitur dari kondisi acak (*random weight initialization*), model memanfaatkan representasi fitur yang telah dipelajari dari tugas sumber (*Source Task*) pada domain berskala masif (*Source Domain*) untuk menyelesaikan tugas sasaran (*Target Task*) pada domain spesifik (*Target Domain*).

```
+---------------------------------------------------------------------------------------------------------+
|                                    PARADIGMA TRANSFER LEARNING                                          |
+---------------------------------------------------------------------------------------------------------+
|                                                                                                         |
|       [ SOURCE TASK & DOMAIN ]                                   [ TARGET TASK & DOMAIN ]               |
|       - Dataset Skala Masif                                      - Dataset Spesifik / Terbatas          |
|       - Tugas Umum (e.g. ImageNet, Web Corpus)                   - Tugas Akhir (e.g. Deteksi Cacat)     |
|                                                                                                         |
|                   │                                                          │                          |
|                   v                                                          v                          |
|       ┌───────────────────────┐                                  ┌───────────────────────┐              |
|       │  PRE-TRAINING PHASE   │                                  │   ADAPTATION PHASE    │              |
|       │  Pelatihan Bobot Awal │ ──(Transfer Representasi Fitur)──>  Fine-Tuning / PEFT   │              |
|       │  Dasar (Backbone W0)  │                                  │   Pada Data Target    │              |
|       └───────────────────────┘                                  └───────────────────────┘              |
|                                                                                                         |
+---------------------------------------------------------------------------------------------------------+
```

### 1.1 Formulasi Matematis Domain dan Tugas
Secara formal, sebuah **Domain** $\mathcal{D}$ didefinisikan oleh dua komponen: ruang fitur $\mathcal{X}$ dan distribusi probabilitas marginal $P(X)$:
$$\mathcal{D} = \{\mathcal{X}, \, P(X)\}, \quad X = \{x_1, \dots, x_n\} \in \mathcal{X}$$

Sebuah **Tugas** $\mathcal{T}$ pada domain tertentu didefinisikan oleh ruang label $\mathcal{Y}$ dan fungsi prediktif kondisional $P(Y \mid X)$:
$$\mathcal{T} = \{\mathcal{Y}, \, P(Y \mid X)\}$$

Dalam Transfer Learning, terdapat empat kemungkinan relasi antara sumber (*source*, indeks $S$) dan sasaran (*target*, indeks $T$):
1. **Perbedaan Ruang Fitur**: $\mathcal{X}_S \neq \mathcal{X}_T$ (*e.g.*, teks bahasa Inggris vs teks bahasa Indonesia).
2. **Perbedaan Distribusi Fitur (Covariate Shift)**: $P(X_S) \neq P(X_T)$ (*e.g.*, citra kamera jalan raya siang hari vs malam hari).
3. **Perbedaan Ruang Label**: $\mathcal{Y}_S \neq \mathcal{Y}_T$ (*e.g.*, 1000 kelas objek umum ImageNet vs 3 kelas cacat manufaktur).
4. **Perbedaan Fungsi Kondisional (Concept Shift)**: $P(Y_S \mid X_S) \neq P(Y_T \mid X_T)$.

### 1.2 Hirarki Representasi Fitur (*Feature Reuse*)
Studi empiris klasik (Yosinski et al., 2014) membuktikan bahwa jaringan saraf tiruan mendalam mempelajari representasi bertingkat (*hierarchical features*):
- **Lapisan Bawah (Early Layers)**: Mempelajari fitur primitif umum (*edges, corners, color blobs, basic token statistics*). Fitur ini memiliki transferabilitas universal lintas domain.
- **Lapisan Tengah (Middle Layers)**: Mempelajari motif dan tekstur komposit (*patterns, shapes, syntactic dependencies*).
- **Lapisan Atas (Top Layers)**: Mempelajari fitur semantik abstrak yang sangat terspesialisasi pada tugas sumber tertentu.

```
       INPUT RAW DATA
             │
             v
   ┌───────────────────┐
   │   Lapisan Bawah   │ ──> Fitur Primitif Umum (Garis, Sudut, Frekuensi Dasar) -> Sangat Dapat Ditransfer
   └─────────┬─────────┘
             v
   ┌───────────────────┐
   │   Lapisan Tengah  │ ──> Pola dan Tekstur Komposit -> Cukup Dapat Ditransfer
   └─────────┬─────────┘
             v
   ┌───────────────────┐
   │   Lapisan Atas    │ ──> Konsep Semantik Spesifik Tugas Sumber -> Butuh Fine-Tuning Khusus
   └─────────┬─────────┘
             v
   ┌───────────────────┐
   │  Classification   │ ──> Pemetaan Label Tugas Asal -> Wajib Diganti untuk Tugas Target
   │       Head        │
   └───────────────────┘
```

---

## 2. Ekstraksi Fitur & Linear Probing

Strategi paling konservatif dan efisien dalam Transfer Learning adalah **Feature Extraction** (sering disebut sebagai **Linear Probing**).

### 2.1 Mekanisme Kerja Linear Probing
Pada pendekatan ini:
1. Bobot representasi backbone pra-terlatih dibekukan secara utuh:
   $$\forall \theta_{\text{backbone}}, \quad \frac{\partial \mathcal{L}}{\partial \theta_{\text{backbone}}} = 0 \quad (\text{requires\_grad} = \text{False})$$
2. Lapisan kepala klasifikasi tugas asal dilepas dan digantikan oleh modul klasifikasi baru:
   $$y_{\text{pred}} = \text{Softmax}(W_{\text{head}} \cdot \Phi(x) + b_{\text{head}})$$
   Di mana $\Phi(x)$ adalah vektor representasi fitur yang dihasilkan oleh backbone beku.
3. Hanya bobot $W_{\text{head}}$ dan $b_{\text{head}}$ yang diperbarui oleh algoritma optimasi.

```
+---------------------------------------------------------------------------------------------------------+
|                                    ARSITEKTUR LINEAR PROBING                                            |
+---------------------------------------------------------------------------------------------------------+
|                                                                                                         |
|       Input x ───> [ BACKBONE FEATURE EXTRACTOR ] ───> Vektor Fitur Laten ───> [ NEW CLASSIFIER HEAD ]  |
|                         (DIBEKUKAN UTUH)                         Phi(x)           (DILATIH PENUH)       |
|                    param.requires_grad = False                                    lr = 1e-2             |
|                                                                                                         |
+---------------------------------------------------------------------------------------------------------+
```

### 2.2 Distingsi Kritis Runtime: Mode `train()` vs `eval()`
Salah satu kesalahan rekayasa paling fatal dalam implementasi PyTorch adalah menganggap bahwa menyetel `param.requires_grad = False` otomatis membekukan seluruh perilaku dinamis model.

```python
# PERINGATAN REKAYASA PYTORCH:
backbone.train()  # SALAH untuk feature extraction murni!
# Meskipun bobot dibekukan:
# 1. Lapisan nn.Dropout tetap aktif membuang aktivasi secara acak!
# 2. Lapisan nn.BatchNorm tetap memperbarui statistik running_mean & running_var!
```

Untuk menjamin inferensi deterministik dan mencegah rusaknya kalibrasi statistik normalisasi batch dari domain target, backbone **wajib dikunci ke mode evaluasi**:
```python
backbone.eval()   # BENAR: Dropout nonaktif, BatchNorm membeku pada statistik pra-pelatihan
new_head.train()  # Kepala baru tetap aktif dalam mode pelatihan
```

---

## 3. Strategi Fine-Tuning: Spektrum Adaptasi Model

Jika domain target memiliki perbedaan representasi yang signifikan dari domain sumber, fitur laten beku $\Phi(x)$ mungkin tidak cukup diskriminatif. Pengembang dapat menaiki spektrum adaptasi menuju **Fine-Tuning**.

```
+---------------------------------------------------------------------------------------------------------+
|                                   SPEKTRUM STRATEGI ADAPTASI MODEL                                      |
+---------------------------------------------------------------------------------------------------------+
|                                                                                                         |
|  [ 1. LINEAR PROBING ]             [ 2. PARTIAL FINE-TUNING ]           [ 3. FULL FINE-TUNING ]         |
|  - 0% parameter backbone dilatih   - 10-30% lapisan atas dilatih        - 100% parameter diperbarui     |
|  - Sangat cepat, hemat memori      - Menyesuaikan semantik tinggi       - Fleksibilitas representasi    |
|  - Aman dari overfitting           - Mempertahankan fitur dasar         - Risiko tinggi kelupaan katastropik |
|                                                                                                         |
+---------------------------------------------------------------------------------------------------------+
```

### 3.1 Partial Fine-Tuning & Progressive Unfreezing
Pada **Partial Fine-Tuning**, lapisan bawah yang bersifat universal tetap dibekukan, sementara lapisan atas dicairkan (*unfrozen*) secara bertahap (*Progressive Unfreezing*). 

Strategi ini didampingi oleh **Differential Learning Rates** (laju pembelajaran bertingkat):
$$\eta_{\text{lower}} = 0, \quad \eta_{\text{upper}} = 10^{-4}, \quad \eta_{\text{head}} = 10^{-2}$$

Dengan konfigurasi ini, fitur tingkat tinggi bergeser secara halus menyesuaikan target task tanpa mengalami disrupsi struktural.

### 3.2 Full Fine-Tuning & Bahaya Catastrophic Forgetting
**Full Fine-Tuning** memperbarui seluruh parameter model secara bersamaan. Meskipun menawarkan fleksibilitas tertinggi, metode ini memiliki risiko kritis: **Catastrophic Forgetting** (kelupaan katastropik).

Ketika model dipaksa mengoptimalkan fungsi rugi pada dataset target baru dengan laju pembelajaran agresif ($\eta \ge 10^{-2}$), gradien liar akan menimpa bobot koneksi yang menyimpan pengetahuan dasar tugas sumber:
$$\mathcal{L}_{\text{target}}(\theta) \to \min \implies \mathcal{L}_{\text{source}}(\theta) \gg 0$$

```
Akurasi Tugas Sumber
 1.0 ┼──────* (Performa Awal: 95%)
     │       \
 0.8 ┼        \
     │         \
 0.6 ┼          * (Fine-Tuning Konservatif: Retensi 88%)
     │           \
 0.4 ┼            \
     │             \
 0.2 ┼              *──────> Fine-Tuning Agresif: Runtuh ke 25% (Catastrophic Forgetting!)
   0 ┴──────────────┴──────────────┴───> Langkah Pelatihan Target
```

Mitigasi ilmiah mencakup:
1. Menggunakan **laju pembelajaran konservatif** ($\eta \le 10^{-4}$).
2. Menerapkan **Weight Decay** (regularisasi L2) untuk menjaga bobot tetap berada di sekitar konfigurasi pra-terlatih.
3. Menggunakan paradigma **Parameter-Efficient Fine-Tuning (PEFT)**.

---

## 4. Pergeseran Domain (*Domain Shift*) & Transfer Negatif

Model pembelajaran mendalam mengasumsikan bahwa data latih dan data uji ditarik dari distribusi probabilitas yang identik dan independen (*i.i.d. assumption*). Dalam penerapan nyata, asumsi ini kerap dilanggar.

### 4.1 Taksonomi Pergeseran Distribusi
1. **Covariate Shift**: Distribusi input $P(X)$ berubah antara lingkungan sumber dan sasaran, namun hubungan pemetaan sebab-akibat $P(Y \mid X)$ tetap konstan.
   $$P_S(X) \neq P_T(X), \quad P_S(Y \mid X) = P_T(Y \mid X)$$
   *Contoh*: Sensor turbin pabrik diuji pada suhu ruangan normal ($25^\circ\text{C}$), namun dioperasikan di fasilitas gurun ($50^\circ\text{C}$).
2. **Prior Probability Shift (Label Shift)**: Distribusi frekuensi label target $P(Y)$ berubah secara ekstrem.
   $$P_S(Y) \neq P_T(Y), \quad P_S(X \mid Y) = P_T(X \mid Y)$$
3. **Concept Drift**: Karakteristik fisik dari label itu sendiri berubah seiring waktu.

### 4.2 Fenomena Transfer Negatif (*Negative Transfer*)
Transfer Learning tidak selalu menguntungkan. Jika representasi dari tugas sumber tidak memiliki relasi sebab-akibat dengan tugas sasaran, atau justru memiliki korelasi terbalik (*spurious correlation*), maka pemanfaatan bobot pra-terlatih akan menghambat pembelajaran:
$$\text{Akurasi}(\text{Model Transfer}) < \text{Akurasi}(\text{Model Scratch})$$

```
+---------------------------------------------------------------------------------------------------------+
|                                    ANALISIS TRANSFER NEGATIF                                            |
+---------------------------------------------------------------------------------------------------------+
|                                                                                                         |
|       Kasus A: Transfer Positif                               Kasus B: Transfer Negatif                 |
|       - Tugas Sumber: Klasifikasi Hewan Umum                  - Tugas Sumber: Klasifikasi Berlawanan    |
|       - Tugas Sasaran: Klasifikasi Ras Kucing Liar            - Tugas Sasaran: Deteksi Anomali Acak     |
|       - Hasil: Akurasi Transfer > Akurasi Scratch             - Hasil: Akurasi Transfer < Akurasi Scratch|
|         (Pengetahuan fitur visual sangat bermanfaat)            (Fitur sumber menyesatkan batas kelas)  |
|                                                                                                         |
+---------------------------------------------------------------------------------------------------------+
```

---

## 5. Paradigma Parameter-Efficient Fine-Tuning (PEFT)

Seiring membesarnya ukuran model fondasi (dari ResNet dan BERT berparameter jutaan hingga LLM berparameter ratusan milyar), Full Fine-Tuning menghadapi kendala memori yang tidak dapat diatasi:

### 5.1 Hambatan Memori Pelatihan Skala Besar
Dalam optimizer modern seperti AdamW, memori pelatihan untuk setiap parameter presisi FP32 (4 byte) adalah:
- Nilai bobot model: $4$ byte
- Gradien balik: $4$ byte
- Momen gradien orde pertama ($m_t$): $4$ byte
- Momen gradien orde kedua ($v_t$): $4$ byte
$$\text{Total Kebutuhan Memori} = 16 \text{ byte per parameter}$$

Untuk model dengan 7 milyar parameter (7B), menyimpan status optimizer saja membutuhkan minimal $7 \times 16 = 112\text{ GB}$ memori murni di luar aktivasi! Menyimpan berkas *checkpoint* penuh sebesar 28 GB untuk setiap variasi tugas spesifik juga tidak layak secara infrastruktur.

**PEFT** memecahkan kebuntuan ini dengan membekukan $99\%+$ parameter model dasar dan hanya melatih modul adaptor tambahan berdimensi kecil.

---

## 6. Fondasi Matematis Low-Rank Adaptation (LoRA)

Diperkenalkan oleh Hu et al. (ICLR 2022), **Low-Rank Adaptation (LoRA)** merupakan terobosan paling berpengaruh dalam adaptasi efisien parameter.

```
+---------------------------------------------------------------------------------------------------------+
|                                 MEKANISME DEKOMPOSISI MATRIKS LoRA                                      |
+---------------------------------------------------------------------------------------------------------+
|                                                                                                         |
|                                         Output h = W0 x + Delta W x                                     |
|                                                ▲                                                        |
|                                                │                                                        |
|                                              ( + ) ◄────────────────────────┐                           |
|                                                │                            │                           |
|                                                │               [ Matriks B: d_out x r ]                 |
|                                                │               (Inisialisasi Nol Mutlak)                |
|                                                │                            ▲                           |
|                                   [ Matriks W0: d_out x d_in ]              │ r (Rank Rendah)           |
|                                      (BOBOT ASLI DIBEKUKAN)                 │                           |
|                                                │               [ Matriks A: r x d_in ]                  |
|                                                │               (Inisialisasi Gaussian)                  |
|                                                │                            ▲                           |
|                                                │                            │                           |
|                                                └──────────────┬─────────────┘                           |
|                                                               │                                         |
|                                                          Input x [Batch, d_in]                          |
|                                                                                                         |
+---------------------------------------------------------------------------------------------------------+
```

### 6.1 Hipotesis Rank Intrinsik Rendah
Aghajanyan et al. (2020) membuktikan bahwa proses adaptasi model terlatih terhadap tugas baru memiliki **dimensi intrinsik yang sangat rendah** (*low intrinsic dimension*). Artinya, perubahan bobot $\Delta W$ tidak memerlukan seluruh derajat kebebasan ruang matriks penuh $\mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$, melainkan dapat diaproksimasi secara akurat pada sub-ruang rank rendah.

### 6.2 Perumusan Aljabar Dekomposisi LoRA
Untuk lapisan transformasi linier $h = W_0 x$ dengan bobot beku $W_0 \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$, LoRA memodelkan pembaruan bobot sebagai perkalian dua matriks rank rendah:
$$W' = W_0 + \Delta W = W_0 + \frac{\alpha}{r} (B \cdot A)$$

Di mana:
- $A \in \mathbb{R}^{r \times d_{\text{in}}}$ adalah matriks proyeksi turun (*down-projection*).
- $B \in \mathbb{R}^{d_{\text{out}} \times r}$ adalah matriks proyeksi naik (*up-projection*).
- $r \ll \min(d_{\text{in}}, d_{\text{out}})$ adalah hiperparameter rank intrinsik (misal $r \in \{2, 4, 8, 16\}$).
- $\alpha$ adalah hiperparameter skalar konstan (*LoRA scaling factor*).

### 6.3 Inisialisasi Kritis & Paritas Langkah Awal
Agar proses fine-tuning dimulai tepat dari representasi pra-pelatihan tanpa distorsi acak awal, LoRA menetapkan skema inisialisasi asimetris:
$$A \sim \mathcal{N}\left(0, \, \frac{1}{r}\right), \quad B = 0$$

Akibatnya, pada iterasi awal $t=0$:
$$\Delta W = \frac{\alpha}{r} (B \cdot A) = \frac{\alpha}{r} (0 \cdot A) = 0$$
$$\implies h = W_0 x + 0 = W_0 x$$
Model adaptor terbukti secara matematis memiliki luaran yang identik 100% dengan model dasar sebelum langkah optimasi pertama dimulai.

### 6.4 Komputasi Forward Hemat Memori
Dalam eksekusi perangkat lunak, matriks penuh $\Delta W$ **tidak pernah dibentuk secara eksplisit** di memori selama pelatihan. Sebaliknya, perkalian matriks dikelompokkan secara asosiatif memanfaatkan ukuran rank $r$ yang kecil:
$$\Delta h = \frac{\alpha}{r} \left( (x \cdot A^T) \cdot B^T \right)$$
1. $x \cdot A^T \implies [\text{Batch}, d_{\text{in}}] \times [d_{\text{in}}, r] \to [\text{Batch}, r]$ (Proyeksi ke dimensi mini $r$).
2. $(\dots) \cdot B^T \implies [\text{Batch}, r] \times [r, d_{\text{out}}] \to [\text{Batch}, d_{\text{out}}]$.

Kompleksitas komputasi dan konsumsi memori aktivasi menyusut secara drastis dari $\mathcal{O}(d_{\text{in}} \cdot d_{\text{out}})$ menjadi $\mathcal{O}(r \cdot (d_{\text{in}} + d_{\text{out}}))$.

### 6.5 Penggabungan Bobot Tanpa Latensi (*Zero-Latency Merging*)
Setelah siklus pelatihan selesai, matriks adaptor dapat disatukan secara permanen ke dalam bobot dasar sebelum disebarkan ke server produksi:
$$W_{\text{merged}} = W_0 + \frac{\alpha}{r} (B \cdot A)$$
Setelah penggabungan, arsitektur model kembali menjadi modul linier tunggal standar. Tidak ada penambahan latensi komputasi sama sekali (*zero inference latency*) saat melayani permintaan pengguna.

---

## 7. Desain Implementasi PyTorch Mandiri pada CPU

Katalog kode Bab 31 menyediakan implementasi kelas modul `LoRALinear` yang sepenuhnya kompatibel dengan ekosistem PyTorch standar:

```python
import torch
import torch.nn as nn

class LoRALinear(nn.Module):
    def __init__(self, in_features, out_features, r=4, lora_alpha=8.0, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / r
        
        # 1. Base Layer yang Dibekukan
        self.base_layer = nn.Linear(in_features, out_features, bias=bias)
        self.base_layer.weight.requires_grad = False
        if bias:
            self.base_layer.bias.requires_grad = False
            
        # 2. Parameter Rendah LoRA
        self.lora_A = nn.Parameter(torch.empty(r, in_features))
        nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))
        
        self.merged = False

    def forward(self, x):
        if self.merged:
            return self.base_layer(x)
        # Jalur adaptif
        base_out = self.base_layer(x)
        adapter_out = (x @ self.lora_A.t()) @ self.lora_B.t() * self.scaling
        return base_out + adapter_out

    def merge_weights(self):
        if not self.merged:
            delta_w = (self.lora_B @ self.lora_A) * self.scaling
            self.base_layer.weight.data += delta_w
            self.merged = True
```

---

## 8. Evaluasi Empiris & Perbandingan Komprehensif

Dalam skrip laboratorium `fine_tuning_comparison.py`, lima paradigma adaptasi diuji pada dataset target yang identik, partisi data uji terisolasi, dan anggaran komputasi seimbang:

| Paradigma Adaptasi | Parameter Latih | % Trainable | Memori Optimizer | Test Akurasi | Karakteristik Utama |
|---|:---:|:---:|:---:|:---:|---|
| **1. Training from Scratch** | $1.506$ | $100.0\%$ | $100\%$ | $81.67\%$ | Rentan overfitting pada data sedikit |
| **2. Linear Probing** | $66$ | $4.4\%$ | $4.4\%$ | $83.33\%$ | Ekstrem cepat, stabil, konvergensi instan |
| **3. Partial Fine-Tuning** | $1.154$ | $76.6\%$ | $76.6\%$ | $88.33\%$ | Menyesuaikan representasi atas secara aman |
| **4. Full Fine-Tuning** | $1.506$ | $100.0\%$ | $100\%$ | $86.67\%$ | Risiko catastrophic forgetting & degradasi |
| **5. LoRA Adaptation ($r=4$)** | $322$ | **$21.4\%$** | **$21.4\%$** | **$88.33\%$** | **Akurasi optimal dengan parameter sangat hemat** |

Hasil empiris membuktikan bahwa **LoRA ($r=4$)** mencapai akurasi puncak ($88.33\%$) setara dengan Partial Fine-Tuning, namun hanya membutuhkan pembaruan pada sebagian kecil parameter terbobot.

---

## 9. Latihan Bertingkat (3-Tier Exercises)

### Latihan Tingkat 1: Pemahaman Konseptual & Aljabar
1. **Soal**: Sebuah modul atensi Multi-Head memiliki proyeksi matriks Query $W_q \in \mathbb{R}^{768 \times 768}$. Jika diadaptasi menggunakan LoRA dengan rank $r=4$ dan scaling factor $\alpha = 8.0$:
   a. Berapa jumlah elemen pada matriks asli $W_0$?
   b. Berapa jumlah parameter gabungan matriks $A$ dan $B$?
   c. Berapa faktor pengali skalar pembaruan saat forward pass?
2. **Soal**: Mengapa inisialisasi matriks $B$ harus disetel ke nol mutlak ($B=0$) pada awal pelatihan LoRA? Jelaskan apa yang terjadi jika kedua matriks $A$ dan $B$ sama-sama diinisialisasi secara acak Gaussian.

*Pembahasan*:
1. a. Parameter $W_0 = 768 \times 768 = 589.824$ elemen.
   b. Matriks $A \in \mathbb{R}^{4 \times 768} = 3.072$ elemen. Matriks $B \in \mathbb{R}^{768 \times 4} = 3.072$ elemen. Total $= 6.144$ parameter.
   c. Skalar pengali $= \frac{\alpha}{r} = \frac{8.0}{4} = 2.0$.
2. Inisialisasi $B=0$ menjamin $\Delta W = \frac{\alpha}{r} (0 \cdot A) = 0$. Model adaptor memulai pelatihan dari titik representasi pra-pelatihan murni tanpa distorsi. Jika $B$ diinisialisasi acak, maka $\Delta W \neq 0$, menyuntikkan derau acak besar yang merusak kapabilitas representasi awal model.

### Latihan Tingkat 2: Implementasi Pemrograman PyTorch
**Soal**: Tuliskan fungsi Python `freeze_model_except_head(model, head_name='classifier')` yang menerima model PyTorch, membekukan seluruh parameter selain lapisan kepala, serta mengembalikan jumlah parameter beku dan parameter aktif.

*Pembahasan*:
```python
def freeze_model_except_head(model, head_name='classifier'):
    frozen_count = 0
    trainable_count = 0
    for name, param in model.named_parameters():
        if head_name in name:
            param.requires_grad = True
            trainable_count += param.numel()
        else:
            param.requires_grad = False
            frozen_count += param.numel()
    return frozen_count, trainable_count
```

### Latihan Tingkat 3: Evaluasi Kasus & Arsitektur
**Soal**: Sebuah rumah sakit ingin mengadaptasi model segmentasi organ tubuh dari citra CT-Scan umum ke citra MRI spesifik. Mereka memiliki 50 sampel MRI. Diskusikan risiko teknis, pilihan strategi adaptasi terbaik, dan protokol pengujian data yang harus diterapkan.

*Pembahasan*:
- **Risiko**: Perbedaan fisika akuisisi antara CT-Scan (densitas radiasi x-ray) dan MRI (resonansi magnetik spin hidrogen) menimbulkan *Covariate Shift* parah. Jumlah sampel yang sangat sedikit (50 sampel) menimbulkan risiko *overfitting* ekstrem bila dilakukan Full Fine-Tuning.
- **Strategi Terbaik**: Gunakan *Data Augmentation* intensif (rotasi, distorsi elastis) untuk mencegah memorisasi spasial. Terapkan LoRA ($r=4$) pada lapisan proyeksi atau *Linear Probing* awal, lalu evaluasi konvergensi validasi.
- **Protokol Evaluasi**: Wajib menggunakan *Patient-Level Group K-Fold* atau isolasi Test Set berbasis pasien unik guna mencegah kebocoran data irisan citra (*slice leakage*).

---

## 10. Tugas Proyek Mandiri, Mini Project, dan Enam Milestone Capstone

### 10.1 Deskripsi Mini Project
Mahasiswa wajib mengimplementasikan proyek adaptasi model mandiri bertajuk:  
**"Efficient Model Adaptation Using Transfer Learning & LoRA"**

### 10.2 Enam Milestone Pelaksanaan
1. **Milestone 1**: Identifikasi tugas sumber (*Source Task*) dan tugas sasaran (*Target Task*), penyusunan hipotesis ilmiah, dan dokumentasi asal-usul data.
2. **Milestone 2**: Partisi bertingkat 3-way (*Train 70%, Validation 15%, Test 15%*) bebas kebocoran data.
3. **Milestone 3**: Pelatihan model baseline pembanding (*Scratch Model*).
4. **Milestone 4**: Implementasi *Feature Extraction* (*Linear Probing*) dengan pembekuan parameter terverifikasi.
5. **Milestone 5**: Implementasi *Partial Fine-Tuning* atau modul kustom *LoRA* beserta pembuktian aliran gradien.
6. **Milestone 6**: Pengujian pada Test Set terisolasi, pembuatan tabel komparasi sumber daya (parameter, memori, akurasi), dan penyusunan *Model Card*.

---

## 11. Rubrik Penilaian Proyek Mini (Skala 100 Poin)

| Dimensi Penilaian | Bobot | Kriteria Sempurna (Full Credit) | Kriteria Cukup (Partial Credit) | Kriteria Kurang (Insufficient) |
|---|:---:|---|---|---|
| **1. Pemahaman Masalah & Partisi Data** | 15 Poin | Perumusan masalah jelas; dataset terdokumentasi; partisi 3-way bertingkat bebas kebocoran data terverifikasi. | Masalah jelas namun pemisahan data uji kurang terisolasi atau terjadi kebocoran skala kecil. | Tidak ada pemisahan data uji independen; kebocoran data parah. |
| **2. Implementasi Transfer Learning & Freezing** | 25 Poin | Pembekuan parameter (`requires_grad=False`) tepat; mode `eval()` diposisikan benar; Linear Probing berhasil dieksekusi. | Pembekuan parameter berhasil namun mode runtime `train()` vs `eval()` diabaikan. | Pembekuan gagal; kode runtime error; tidak ada pemisahan head. |
| **3. Implementasi Fine-Tuning / LoRA** | 25 Poin | Menerapkan Partial FT dengan differential LRs atau LoRA dengan dekomposisi rank rendah; paritas inisialisasi matematis terbukti. | Menerapkan adaptasi namun laju pembelajaran tidak terkontrol atau gradien tidak terverifikasi. | Implementasi salah aljabar; parameter tidak terupdate; tidak ada LoRA. |
| **4. Evaluasi & Analisis Sumber Daya** | 25 Poin | Tabel komparasi lengkap (parameter latih, waktu latih, akurasi, F1); analisis trade-off performa vs memori mendalam. | Komparasi hanya melaporkan akurasi tanpa analisis waktu komputasi atau persentase parameter. | Klaim performa tanpa bukti tabel evaluasi; evaluasi dilaporkan pada data latih. |
| **5. Dokumentasi & Referensi Akademik** | 10 Poin | Kode bersih, modular, terdokumentasi rapi; Model Card ringkas tersedia; menyertakan $\ge 2$ referensi akademik terverifikasi. | Kode dapat dijalankan namun dokumentasi minim; referensi kurang lengkap. | Tidak ada dokumentasi; referensi fiktif atau tidak relevan. |
| **Total Skor Maksimal** | **100 Poin** | **Standar Rekayasa Industri & Akademik Teruji** | — | — |

---

## 12. Ringkasan Bab & Referensi Akademik Terverifikasi

### 12.1 Ringkasan Bab 31
1. **Transfer Learning** mentransfer representasi fitur yang telah dipelajari pada domain sumber besar untuk mempercepat konvergensi dan meningkatkan akurasi pada domain sasaran kecil.
2. **Feature Extraction (Linear Probing)** membekukan backbone secara utuh (`requires_grad=False`, `eval()` mode), melatih hanya kepala klasifikasi baru dengan efisiensi komputasi maksimal.
3. **Partial Fine-Tuning** mencairkan lapisan atas secara bertahap menggunakan *differential learning rates* untuk menyesuaikan semantik tinggi secara terkontrol.
4. **Catastrophic Forgetting** adalah risiko utama Full Fine-Tuning di mana representasi tugas asal runtuh akibat laju pembelajaran yang terlalu agresif.
5. **Low-Rank Adaptation (LoRA)** memodelkan pembaruan bobot sebagai $\Delta W = \frac{\alpha}{r} BA$, menghemat $>95\%$ parameter dan memori komputasi tanpa mengorbankan kualitas representasi, serta mendukung *zero-latency weight merging* untuk lingkungan produksi.

### 12.2 Referensi Akademik Terverifikasi
1. **Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., & Chen, W. (2021).** *LoRA: Low-Rank Adaptation of Large Language Models.* In International Conference on Learning Representations (ICLR 2022). [arXiv:2106.09685](https://arxiv.org/abs/2106.09685)
2. **Yosinski, J., Clune, J., Bengio, Y., & Lipson, H. (2014).** *How transferable are features in deep neural networks?* In Advances in Neural Information Processing Systems (NeurIPS 2014), Vol. 27, pp. 3320–3328. [NeurIPS Proceedings](https://proceedings.neurips.cc/paper/2014/file/375c71349b295fbe2dcdca9206f20a06-Paper.pdf)
3. **Pan, S. J., & Yang, Q. (2010).** *A Survey on Transfer Learning.* IEEE Transactions on Knowledge and Data Engineering, 22(10), pp. 1345–1359. [DOI: 10.1109/TKDE.2009.191](https://doi.org/10.1109/TKDE.2009.191)
4. **He, K., Zhang, X., Ren, S., & Sun, J. (2016).** *Deep Residual Learning for Image Recognition.* In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pp. 770–778. [arXiv:1512.03385](https://arxiv.org/abs/1512.03385)
5. **Howard, J., & Ruder, S. (2018).** *Universal Language Model Fine-tuning for Text Classification.* In Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (ACL 2018), pp. 328–339. [arXiv:1801.06146](https://arxiv.org/abs/1801.06146)

