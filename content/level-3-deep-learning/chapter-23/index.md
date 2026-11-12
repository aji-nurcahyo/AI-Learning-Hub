# Bab 23: Deep Learning with PyTorch

## Fondasi Komputasi Tensor, Autograd Dinamis, Arsitektur nn.Module, dan Alur Kerja Pelatihan Jaringan Saraf Tiruan Modern

---

## 1. Peran Framework Deep Learning & Transisi dari NumPy

Pada Bab 21 dan 22, kita telah membedah seluruh fondasi komputasi jaringan saraf tiruan lapis banyak (*Multilayer Perceptron*) dari dasar matematika murni menggunakan pustaka **NumPy**: mulai dari kombinasi linier terbobot $\mathbf{z} = \mathbf{W}\mathbf{x} + \mathbf{b}$, fungsi aktivasi non-linier, penurunan aturan rantai kalkulus multivariat (*backpropagation*), hingga implementasi optimizer adaptif (*Adam*).

Meskipun pendekatan dari awal (*from scratch*) sangat krusial untuk membangun intuisi matematis yang kokoh, mengimplementasikan model komputasi mendalam skala besar pada masalah dunia nyata menggunakan NumPy menghadapi tiga keterbatasan praktis yang parah:
1. **Ketiadaan Diferensiasi Otomatis Tergeneralisasi**: Menuliskan turunan parsial $\frac{\partial \mathcal{L}}{\partial \mathbf{W}}$ secara manual untuk arsitektur kompleks dengan puluhan lapisan (seperti ResNet atau Transformer) memakan waktu lama dan sangat rentan terhadap *silent gradient bugs*.
2. **Keterbatasan Akselerasi Perangkat Keras**: Operasi array NumPy terikat erat pada pemrosesan CPU tunggal dan tidak mendukung eksekusi terakselerasi pada prosesor grafis paralel masif (*Graphics Processing Units* - GPU atau TPU) tanpa pustaka pihak ketiga khusus.
3. **Ketiadaan Abstraksi Pipeline Data & Siklus Hidup Model**: Mengelola pengelompokan batch (*mini-batching*), pengacakan (*shuffling*), paralelisasi multi-proses (*multi-worker streaming*), serialisasi bobot, dan isolasi mode evaluasi membutuhkan ribuan baris kode infrastruktur yang rumit.

```
+---------------------------------------------------------------------------------------------------------+
|                               EVOLUSI PARADIGMA: NUMPY FROM SCRATCH KE PYTORCH                          |
+---------------------------------------------------------------------------------------------------------+
|                                                                                                         |
|   [ PENDEKATAN NUMPY MURNI (BAB 21-22) ]                                                                |
|   - Forward Pass    : Matriks dot product manual z = W @ x + b                                          |
|   - Backward Pass   : Turunan analitis ditulis tangan dL/dW = delta @ a.T                               |
|   - Hardware        : Terbatas pada thread CPU standar                                                  |
|   - Kompleksitas    : Sangat tinggi bila arsitektur memiliki percabangan graf                           |
|                                                                                                         |
|                                         │                                                               |
|                                         ▼ (Transisi ke Framework Modern)                                |
|                                                                                                         |
|   [ PENDEKATAN FRAMEWORK PYTORCH (BAB 23) ]                                                             |
|   - Abstraksi Tensor: Array N-Dimensi dengan dukungan komputasi heterogen (CPU & GPU)                   |
|   - Mesin Autograd  : Graf komputasi dinamis yang otomatis menurunkan gradien (loss.backward())         |
|   - Lapisan Moduler : Pustaka torch.nn dengan parameter terkelola secara otomatis                       |
|   - Ekosistem Data  : Abstraksi Dataset dan DataLoader untuk streaming batch optimal                    |
|                                                                                                         |
+---------------------------------------------------------------------------------------------------------+
```

Di sinilah **PyTorch** hadir sebagai kerangka kerja (*framework*) standar emas dalam penelitian kecerdasan buatan global. Dikembangkan oleh Meta AI Research dan kini dikelola di bawah Linux Foundation, PyTorch menggabungkan kemudahan sintaksis bergaya Pythonik (*imperative execution*) dengan performa komputasi C++ dan CUDA di tingkat mesin.

---

## 2. Representasi Tensor, Dimensi (Shape), Dtype, dan Device

Struktur data paling mendasar dalam PyTorch adalah **Tensor**. Secara matematis, tensor adalah generalisasi aljabar dari skalar (tensor rank-0), vektor (tensor rank-1), dan matriks (tensor rank-2) ke ruang berdimensi $N$ arbitrer.

```
+---------------------------------------------------------------------------------------------------------+
|                                   HIERARKI REPRESENTASI TENSOR PYTORCH                                  |
+---------------------------------------------------------------------------------------------------------+
|                                                                                                         |
|     Skalar (0D)           Vektor (1D)                 Matriks (2D)               Tensor Citra (3D/4D)   |
|                                                                                                         |
|       [ 42.0 ]         [ 1.0, 2.0, 3.0 ]          [[ 1.0, 2.0, 3.0 ],        (Batch, Channels, H, W)    |
|                                                    [ 4.0, 5.0, 6.0 ]]        [ 32,    3,       224, 224]|
|                                                                                                         |
|     Shape: ()          Shape: (3,)                Shape: (2, 3)              Shape: (32, 3, 224, 224)   |
|     ndim: 0            ndim: 1                    ndim: 2                    ndim: 4                    |
|                                                                                                         |
+---------------------------------------------------------------------------------------------------------+
```

### Empat Karakteristik Utama Tensor PyTorch:
1. **Shape / Size**: Tuple yang menyatakan panjang setiap sumbu tensor (misalnya `torch.Size([32, 6])` menyatakan 32 baris sampel dengan 6 dimensi fitur).
2. **Rank / ndim**: Jumlah dimensi yang dimiliki tensor (`len(tensor.shape)`).
3. **Dtype (Data Type)**:
   - `torch.float32` (atau `torch.float`): Tipe standar floating-point 32-bit untuk representasi fitur dan parameter bobot neural network.
   - `torch.int64` (atau `torch.long`): Tipe bilangan bulat 64-bit wajib untuk label target klasifikasi pada fungsi rugi `nn.CrossEntropyLoss`.
   - `torch.bool`: Nilai boolean untuk *masking* pengindeksan data.
4. **Device**: Penunjuk lokasi memori fisik tempat data tensor dialokasikan (`cpu` atau `cuda:0`).

```
+---------------------------------------------------------------------------------------------------------+
|                               ALOKASI PERANGKAT KOMPUTASI (DEVICE MANAGEMENT)                           |
+---------------------------------------------------------------------------------------------------------+
|                                                                                                         |
|          [ Host Memory: RAM ]                           [ Device Memory: VRAM ]                         |
|         +--------------------+                         +------------------------+                       |
|         |  Tensor CPU        | ── tensor.to(device) ─> |  Tensor GPU (CUDA)     |                       |
|         |  (device = 'cpu')  |                         |  (device = 'cuda:0')   |                       |
|         +--------------------+ <── tensor.cpu() ────── +------------------------+                       |
|                                                                                                         |
|  PERINGATAN KRUSIAL: Seluruh tensor dalam suatu operasi matematika (misal: input dan model)             |
|  WAJIB berada pada perangkat fisik yang sama. Operasi lintas device akan memicu RuntimeError!           |
|                                                                                                         |
+---------------------------------------------------------------------------------------------------------+
```

### Jembatan Zero-Copy dengan NumPy
PyTorch menyediakan interoperabilitas langsung dengan ekosistem data science Python (seperti NumPy, Pandas, dan Scikit-Learn) melalui jembatan **Zero-Copy**:
- `torch.from_numpy(arr)`: Mengonversi array NumPy menjadi tensor PyTorch tanpa menyalin memori fisik (*shared memory buffer*).
- `tensor.numpy()`: Mengonversi tensor CPU PyTorch kembali menjadi array NumPy.

> [!IMPORTANT]
> Karena berbagi blok memori yang sama, mutasi *in-place* pada tensor PyTorch akan secara instan mengubah nilai array NumPy yang terhubung, dan sebaliknya. Fitur ini memaksimalkan efisiensi memori tetapi menuntut kehati-hatian pengembang.

---

## 3. Manipulasi Bentuk (Reshaping) dan Aturan Broadcasting

### Operasi Reshaping Tanpa Salin Memori
Dalam arsitektur deep learning, tensor masukan sering kali harus diubah susunan dimensinya sebelum dialirkan ke lapisan tertentu:

```
+---------------------------------------------------------------------------------------------------------+
|                               RAGAM METODE MANIPULASI BENTUK TENSOR                                     |
+---------------------------------------------------------------------------------------------------------+
|  Operasi                  Deskripsi Fungsi                                 Contoh Transformasi          |
|  -----------------------------------------------------------------------------------------------------  |
|  view(shape)              Mengubah bentuk jika data kontigu di memori      (12,) -> (3, 4)              |
|  reshape(shape)           Mengubah bentuk secara fleksibel (aman)          (12,) -> (3, 4)              |
|  unsqueeze(dim)           Menyisipkan dimensi ukuran 1 pada indeks dim     (3, 4) -> (1, 3, 4)          |
|  squeeze(dim)             Menghilangkan dimensi ukuran 1 pada indeks dim   (1, 3, 4) -> (3, 4)          |
|  permute(dims)            Mengatur ulang urutan sumbu dimensi tensor       (B, H, W, C) -> (B, C, H, W) |
|  flatten(start, end)      Meratakan dimensi tertentu menjadi vektor 1D     (B, C, H, W) -> (B, C*H*W)   |
+---------------------------------------------------------------------------------------------------------+
```

### Aturan Broadcasting (Penyiaran Dimensi)
Broadcasting memungkinkan PyTorch melakukan operasi biner (seperti penjumlahan atau perkalian) antara dua tensor yang memiliki dimensi berbeda tanpa harus menduplikasi data di memori secara eksplisit:

```
+---------------------------------------------------------------------------------------------------------+
|                                   VISUALISASI BROADCASTING PENJUMLAHAN                                  |
+---------------------------------------------------------------------------------------------------------+
|                                                                                                         |
|       Tensor A: Shape (3, 1)                      Tensor B: Shape (1, 4)                                |
|          [[ 10.0 ],                                 [[ 1.0, 2.0, 3.0, 4.0 ]]                            |
|           [ 20.0 ],                                                                                     |
|           [ 30.0 ]]                                                                                     |
|                                                                                                         |
|                                              │                                                          |
|                                              ▼ (Operasi: A + B)                                         |
|                                                                                                         |
|                               Tensor Hasil: Shape (3, 4)                                                |
|                     [[ 11.0, 12.0, 13.0, 14.0 ],                                                        |
|                      [ 21.0, 22.0, 23.0, 24.0 ],                                                        |
|                      [ 31.0, 32.0, 33.0, 34.0 ]]                                                        |
|                                                                                                         |
+---------------------------------------------------------------------------------------------------------+
```

Dua dimensi dinilai kompatibel untuk penyiaran jika:
1. Kedua dimensi memiliki panjang yang sama, ATAU
2. Salah satu dimensi bernilai tepat 1 (dimensi ini akan diperluas secara virtual).

---

## 4. Mesin Autograd & Graf Komputasi Dinamis

### Konsep Autograd (Automatic Differentiation)
Di jantung PyTorch terdapat mesin **Autograd**, yaitu sistem kalkulus diferensiasi otomatis berbasis graf berarah tak berputar (*Directed Acyclic Graph* - DAG).

Setiap kali sebuah tensor dibuat dengan parameter `requires_grad=True`, PyTorch mulai melacak setiap operasi matematika yang melibatkan tensor tersebut. Setiap simpul hasil operasi menyimpan atribut `grad_fn` (*gradient function*) yang mencatat turunan lokal operasi elementer yang bersangkutan.

```
+---------------------------------------------------------------------------------------------------------+
|                                 GRAF KOMPUTASI DINAMIS UNTUK y = (w*x + b)^2                            |
+---------------------------------------------------------------------------------------------------------+
|                                                                                                         |
|       [ w ] (requires_grad=True) ──┐                                                                    |
|                                    ├── (*) [MulBackward0] ──> [ p = w*x ] ──┐                           |
|       [ x ] (Konstanta input)   ───┘                                        ├── (+) [AddBackward0] ─┐   |
|                                                                             │                       │   |
|       [ b ] (requires_grad=True) ───────────────────────────────────────────┘                       │   |
|                                                                                                     ▼   |
|                                                                                           [ u = p + b ] |
|                                                                                                     │   |
|                                                                                                     ▼   |
|                                                                                           ( ** 2 ) [Pow]|
|                                                                                                     │   |
|                                                                                                     ▼   |
|                                                                                                   [ y ] |
|                                                                                                         |
|       ALIRAN MAJU (FORWARD PASS)      ─────────────────────────────────────────────────────────────>    |
|       ALIRAN BALIK (y.backward())     <─────────────────────────────────────────────────────────────    |
|                                                                                                         |
+---------------------------------------------------------------------------------------------------------+
```

### Jebakan Akumulasi Gradien Bawaan (Gradient Accumulation)
Salah satu karakteristik internal PyTorch yang paling sering memicu bug fatal bagi pemula adalah:
> **PyTorch TIDAK PERNAH menimpa gradien lama saat `backward()` dipanggil; melainkan MENJUMLAHKAN (mengakumulasikan) gradien baru ke dalam atribut `param.grad` yang sudah ada.**

$$\mathbf{g}_{\text{aktual}} = \mathbf{g}_{\text{sebelumnya}} + \nabla_{\boldsymbol{\theta}} \mathcal{L}$$

Jika pengembang lupa memanggil `optimizer.zero_grad()` sebelum langkah `backward()`, gradien akan terus membengkak di setiap batch, menyebabkan parameter bobot diperbarui secara keliru dan meledak (*exploding gradients*).

```
+---------------------------------------------------------------------------------------------------------+
|                                   SIKLUS PERBAIKAN AKUMULASI GRADIEN                                    |
+---------------------------------------------------------------------------------------------------------+
|                                                                                                         |
|    Batch 1:  dL/dw = 10.0  ──>  w.grad = 10.0                                                           |
|                                                                                                         |
|    [ TANPA zero_grad() ]                                                                                |
|    Batch 2:  dL/dw = 15.0  ──>  w.grad = 10.0 + 15.0 = 25.0  (SALAH! Gradien terkontaminasi masa lalu)  |
|                                                                                                         |
|    [ DENGAN optimizer.zero_grad() ]                                                                     |
|    Batch 2:  w.grad.zero_() ──> dL/dw = 15.0  ──>  w.grad = 15.0 (BENAR! Gradien murni batch aktif)     |
|                                                                                                         |
+---------------------------------------------------------------------------------------------------------+
```

### Menonaktifkan Graf dengan `torch.no_grad()` dan `.detach()`
Pada tahap inferensi atau evaluasi model, kita tidak memerlukan turunan parsial untuk pembaruan parameter. Membiarkan autograd mencatat graf komputasi akan memboroskan alokasi memori RAM/GPU:
- `with torch.no_grad():`: Blok konteks yang menghentikan pembuatan DAG secara global, menghemat memori hingga lebih dari 50% dan mempercepat inferensi.
- `tensor.detach()`: Memutuskan tensor dari graf komputasi aktif dan menghasilkan tensor baru yang memiliki nilai sama tetapi dengan `requires_grad=False`.

---

## 5. Arsitektur Model Berorientasi Objek dengan `torch.nn.Module`

Seluruh arsitektur jaringan saraf di PyTorch dibangun dengan mewarisi kelas dasar `torch.nn.Module`. Pendekatan berorientasi objek ini mengisolasi deklarasi parameter dari aliran data komputasi.

```
+---------------------------------------------------------------------------------------------------------+
|                                STRUKTUR KELAS NEURAL NETWORK (NN.MODULE)                                |
+---------------------------------------------------------------------------------------------------------+
|                                                                                                         |
|  class ModelKlasifikasi(nn.Module):                                                                     |
|      def __init__(self, in_features, hidden_dim, num_classes):                                          |
|          super().__init__()  # Wajib memanggil inisialisasi kelas induk                                 |
|          # 1. Deklarasi Lapisan Parameter Terbobot                                                      |
|          self.fc1 = nn.Linear(in_features, hidden_dim)                                                  |
|          self.relu = nn.ReLU()                                                                          |
|          self.fc2 = nn.Linear(hidden_dim, num_classes)                                                  |
|                                                                                                         |
|      def forward(self, x):                                                                              |
|          # 2. Definisi Aliran Graf Komputasi (Forward Pass)                                             |
|          out = self.fc1(x)                                                                              |
|          out = self.relu(out)                                                                           |
|          logits = self.fc2(out)                                                                         |
|          return logits  # Kembalikan logits mentah!                                                     |
|                                                                                                         |
+---------------------------------------------------------------------------------------------------------+
```

### Mengapa Logits Mentah Tanpa Softmax?
Sebuah kesalahan umum adalah memasang `nn.Softmax(dim=-1)` pada lapisan terakhir model klasifikasi saat menggunakan `nn.CrossEntropyLoss`.

Di PyTorch, `nn.CrossEntropyLoss` telah menggabungkan fungsi `nn.LogSoftmax` dan `nn.NLLLoss` (*Negative Log-Likelihood Loss*) secara simultan. Dengan menerima **logits mentah**, PyTorch dapat menerapkan trik stabilitas numerik *subtract-max*:

$$\text{LogSoftmax}(z_i) = \ln \left( \frac{e^{z_i - \max(\mathbf{z})}}{\sum_j e^{z_j - \max(\mathbf{z})}} \right) = (z_i - \max(\mathbf{z})) - \ln \sum_j e^{z_j - \max(\mathbf{z})}$$

Jika pengembang menambahkan fungsi `Softmax` di akhir model, maka probabilitas $[0, 1]$ akan dihitung kembali logaritma dan softmax-nya untuk kedua kalinya di dalam fungsi rugi, yang mengakibatkan nilai gradien menjadi tidak tepat.

---

## 6. Pipa Data Modular: `Dataset` dan `DataLoader`

PyTorch memisahkan tanggung jawab penyimpanan sampel data dari logika pembuatan batch melalui dua abstraksi:

```
+---------------------------------------------------------------------------------------------------------+
|                                    ARSITEKTUR ALIRAN DATA PYTORCH                                       |
+---------------------------------------------------------------------------------------------------------+
|                                                                                                         |
|   [ DATA MENTAH / CSV / TENSOR ]                                                                        |
|                 │                                                                                       |
|                 ▼                                                                                       |
|   +---------------------------------------+                                                             |
|   |   torch.utils.data.Dataset            |  Menyimpan pasangan sampel dan label individual             |
|   |   - __len__() -> Total sampel         |  Menerima indeks i dan mengembalikan (X[i], y[i])           |
|   |   - __getitem__(idx) -> (X_i, y_i)    |                                                             |
|   +---------------------------------------+                                                             |
|                 │                                                                                       |
|                 ▼                                                                                       |
|   +---------------------------------------+                                                             |
|   |   torch.utils.data.DataLoader         |  Mengelompokkan sampel menjadi mini-batch (batch_size)      |
|   |   - batch_size = 32                   |  Mengacak urutan sampel setiap epoch (shuffle=True)         |
|   |   - shuffle = True / False            |  Mendukung multi-process data loading (num_workers)         |
|   |   - drop_last = False                 |  Otomatis menggabungkan skalar menjadi tensor batch         |
|   +---------------------------------------+                                                             |
|                 │                                                                                       |
|                 ▼ (Streaming per Iterasi)                                                               |
|   ( Batch_X: Shape (32, 6), Batch_y: Shape (32,) ) ───> [ Masuk ke Model & Training Loop ]              |
|                                                                                                         |
+---------------------------------------------------------------------------------------------------------+
```

---

## 7. Protokol Partisi 3-Way & Pencegahan Data Leakage

Dalam pembelajaran mendalam, kumpulan data harus dipartisi secara ketat menjadi tiga subset terisolasi:

```
+---------------------------------------------------------------------------------------------------------+
|                               PROTOKOL PARTISI 3-WAY & DISTRIBUSI PERAN DATA                            |
+---------------------------------------------------------------------------------------------------------+
|                                                                                                         |
|   ========================================= TOTAL DATASET =========================================     |
|   |                                       |                           |                           |     |
|   |        TRAINING SET (70%)             |   VALIDATION SET (15%)    |      TEST SET (15%)       |     |
|   |                                       |                           |                           |     |
|   | - Komputasi gradien autograd          | - Evaluasi performa epoch | - Uji generalisasi akhir  |     |
|   | - Pembaruan bobot optimizer           | - Pemilihan checkpoint    | - Terisolasi total        |     |
|   | - Fit statistik normalisasi           | - Tuning hiperparameter   | - Tidak disentuh tuning   |     |
|   =================================================================================================     |
|                                                                                                         |
+---------------------------------------------------------------------------------------------------------+
```

### Pencegahan Kebocoran Data (Data Leakage)
Salah satu pelanggaran integritas metodologis paling berbahaya adalah menghitung parameter statistik pra-pemrosesan (seperti rata-rata $\mu$ dan standar deviasi $\sigma$ untuk standardisasi) dari seluruh data sebelum partisi dilakukan.

$$\mu_{\text{train}} = \frac{1}{N_{\text{train}}} \sum_{i \in \text{Train}} x_i, \qquad \sigma_{\text{train}} = \sqrt{\frac{1}{N_{\text{train}}} \sum_{i \in \text{Train}} (x_i - \mu_{\text{train}})^2}$$

Nilai $\mu_{\text{train}}$ dan $\sigma_{\text{train}}$ tersebut kemudian digunakan untuk mentransformasikan data latih, data validasi, dan data uji:
$$x_{\text{val}}^{\text{norm}} = \frac{x_{\text{val}} - \mu_{\text{train}}}{\sigma_{\text{train}} + \epsilon}, \qquad x_{\text{test}}^{\text{norm}} = \frac{x_{\text{test}} - \mu_{\text{train}}}{\sigma_{\text{train}} + \epsilon}$$

Dengan cara ini, model tidak pernah terpapar informasi statistik apapun dari data validasi maupun data uji selama proses pembelajaran berlangsung.

---

## 8. Anatomi Lengkap Training Loop dan Validation Loop

Proses pelatihan model di PyTorch mengikuti pola deterministik lima langkah pada setiap batch:

```
+---------------------------------------------------------------------------------------------------------+
|                               SIKLUS STANDAR 5 LANGKAH TRAINING LOOP PYTORCH                            |
+---------------------------------------------------------------------------------------------------------+
|                                                                                                         |
|   model.train()  <── Mengaktifkan mode pelatihan                                                        |
|   for batch_X, batch_y in train_loader:                                                                 |
|                                                                                                         |
|       [ LANGKAH 1: RESET GRADIEN ]                                                                      |
|       optimizer.zero_grad()  ──> Membersihkan akumulator param.grad agar tidak bertambah ganda         |
|                                                                                                         |
|       [ LANGKAH 2: FORWARD PASS ]                                                                       |
|       logits = model(batch_X)  ──> Menghitung skor luaran jaringan                                      |
|                                                                                                         |
|       [ LANGKAH 3: EVALUASI LOSS ]                                                                      |
|       loss = criterion(logits, batch_y)  ──> Mengevaluasi skalar penalti kesalahan                     |
|                                                                                                         |
|       [ LANGKAH 4: BACKWARD PASS ]                                                                      |
|       loss.backward()  ──> Menelusuri graf balik & menghitung seluruh dL/dw                             |
|                                                                                                         |
|       [ LANGKAH 5: UPDATE PARAMETER ]                                                                   |
|       optimizer.step()  ──> Menggeser bobot searah gradien (misal: Adam / SGD)                         |
|                                                                                                         |
+---------------------------------------------------------------------------------------------------------+
```

### Perbedaan Krusial `model.train()` vs `model.eval()`
- `model.train()`: Mengaktifkan perilaku pelatihan khusus pada lapisan regulasi, seperti penjatuhan acak pada `nn.Dropout` dan pencatatan rata-rata berjalan (*running mean/variance*) pada `nn.BatchNorm`.
- `model.eval()`: Menonaktifkan dropout (seluruh neuron aktif dengan skala penuh) dan membekukan lapisan batch normalization pada nilai estimasi populasi.
- **PENTING**: `model.eval()` **TIDAK** mematikan autograd! Pelacakan gradien hanya dimatikan dengan menyelimuti kode di dalam blok `with torch.no_grad():`.

---

## 9. Checkpointing: Serialisasi `state_dict` & Inferensi Aman

Dalam sistem produksi, bobot model yang telah dilatih harus disimpan ke media penyimpanan permanen dan dimuat kembali saat proses inferensi.

```
+---------------------------------------------------------------------------------------------------------+
|                             TATA KELOLA SERIALISASI MODEL: STATE_DICT                                   |
+---------------------------------------------------------------------------------------------------------+
|                                                                                                         |
|  [ PENDEKATAN SALAH & RENTAN ]                                                                          |
|  torch.save(model, "model_utuh.pt")                                                                     |
|  Bahaya: Bergantung pada path kelas Python asli; rentan rusak saat refactoring kode; celah keamanan     |
|          arbitrary code execution melalui modul pickle!                                                 |
|                                                                                                         |
|  [ STANDAR TERBAIK INDUSTRI ]                                                                           |
|  1. Penyimpanan:                                                                                        |
|     torch.save(model.state_dict(), "model_weights.pt")                                                  |
|     (Hanya menyimpan kamus parameter bobot dan bias dalam format pasangan kunci-nilai)                 |
|                                                                                                         |
|  2. Pemuatan Kembali yang Aman:                                                                         |
|     new_model = TabularMLP(...)                                                                         |
|     checkpoint = torch.load("model_weights.pt", weights_only=True)                                      |
|     new_model.load_state_dict(checkpoint)                                                               |
|     new_model.eval()                                                                                    |
|                                                                                                         |
+---------------------------------------------------------------------------------------------------------+
```

Penggunaan parameter `weights_only=True` pada `torch.load` (standar wajib PyTorch versi 2.4+) memastikan bahwa proses deserialisasi hanya membaca tensor numerik murni dan menolak eksekusi fungsi arbitrer yang dapat disusupi kode berbahaya (*malicious payload*).

---

## 10. Studi Kasus Empiris: MLP Non-Linier vs Simple Baseline

Untuk membuktikan efektivitas arsitektur neural network di atas model linier klasik pada protokol evaluasi yang adil, dilakukan eksperimen komparasi langsung menggunakan dataset klasifikasi sintetis non-linier ($N = 1000$ sampel, 6 fitur masukan):

```
+---------------------------------------------------------------------------------------------------------+
|                               RINGKASAN HASIL EKSPERIMEN AKTUAL (TEST SET N=150)                        |
+---------------------------------------------------------------------------------------------------------+
|  Model                 Arsitektur Lapisan       Total Parameter   Test Loss    Test Akurasi   Matriks TP/TN |
|  -----------------------------------------------------------------------------------------------------  |
|  SimpleBaseline        Linear(6, 2)             14 parameter      0.4274       83.33%         TP:58, TN:67  |
|  TabularMLP            6 -> 16 -> 8 -> 2        266 parameter     0.3234       88.67%         TP:61, TN:72  |
|  -----------------------------------------------------------------------------------------------------  |
|  Keunggulan Representasi Non-Linier MLP: +5.34% Akurasi, Reduksi Loss -0.1040                           |
+---------------------------------------------------------------------------------------------------------+
```

### Analisis Hasil:
1. **Model Linier Baseline**: Meskipun cepat dan sederhana, model linier hanya mampu memisahkan bidang datar hiperplane tunggal sehingga menghasilkan galat klasifikasi pada sampel-sampel yang berada di area interaksi non-linier fitur ($x_0 \cdot x_3$ dan $x_2^2$).
2. **TabularMLP**: Lapisan tersembunyi dengan fungsi aktivasi ReLU berhasil memetakan fitur masukan ke ruang representasi berdimensi lebih tinggi di mana batas keputusan non-linier dapat dipisahkan secara lebih akurat, meningkatkan akurasi uji hingga **88.67%**.

---

## 11. Batas Reproduksibilitas & Checklist Debugging PyTorch

### Mengapa Seed Tidak Menjamin Hasil Identik 100%?
Meskipun pengembang telah menetapkan seed acak global via `torch.manual_seed(42)` dan `np.random.seed(42)`, hasil pelatihan terkadang tetap menunjukkan variasi minor ketika dijalankan di komputer yang berbeda:
1. **Non-Asosiatif Floating-Point**: Dalam aritmatika komputer, $(a + b) + c \neq a + (b + c)$ pada batas presisi $10^{-7}$. Rutinitas paralelisasi multi-thread CPU (seperti OpenMP atau Intel MKL) mengakumulasikan gradien dalam urutan thread yang berbeda-beda.
2. **Kernel Non-Deterministik**: Algoritma konvolusi dan perkalian matriks atomik tertentu pada GPU memprioritaskan kecepatan komputasi di atas determinisme urutan penjumlahan floating-point.
3. **Praktik Terbaik**: Dokumentasikan seluruh versi perangkat lunak, sistem operasi, seed, dan laporkan nilai rata-rata $\pm$ standar deviasi dari beberapa *run* eksperimen.

### Checklist 7 Kesalahan Umum Pemula di PyTorch:
1. **Lupa `optimizer.zero_grad()`**: Menyebabkan gradien terakumulasi antar-batch dan meledak.
2. **Memanggil `nn.Softmax` sebelum `nn.CrossEntropyLoss`**: Menghasilkan perhitungan log-loss ganda yang salah.
3. **Mengabaikan `model.eval()` dan `with torch.no_grad():` saat Evaluasi**: Mengakibatkan evaluasi menggunakan dropout acak dan memboroskan memori graf komputasi.
4. **Salah Tipe Data Label Klasifikasi**: Label target untuk `CrossEntropyLoss` wajib bertipe `torch.long`, bukan `torch.float32`.
5. **Data Leakage pada Normalisasi**: Menghitung mean/std dari seluruh data sebelum partisi train/val/test.
6. **Perangkat Tidak Cocok (Device Mismatch)**: Tensor input berada di `cpu` sementara model berada di `cuda:0`.
7. **Penyimpanan Objek Model Penuh**: Mengabaikan format `state_dict` dan menyimpan seluruh objek Python yang rapuh.

---

## 12. Rangkuman & Latihan Soal Mandiri

### Intisari Bab:
- PyTorch memadukan tensor berkinerja tinggi, autograd dinamis, dan abstraksi modular `torch.nn.Module`.
- Pipa data `Dataset` dan `DataLoader` menjamin isolasi data tanpa kebocoran (*zero data leakage*).
- Siklus pelatihan terstandar memerlukan koordinasi ketat antara `model.train()`, `optimizer.zero_grad()`, `backward()`, dan `step()`.
- Evaluasi yang aman dan hemat memori membutuhkan kombinasi `model.eval()` dan `torch.no_grad()`.
- Model disimpan menggunakan kamus bobot `state_dict` dan dimuat kembali secara aman melalui `weights_only=True`.

---

### Latihan Soal 3-Tingkat Mandiri

#### Tingkat 1: Pemahaman Konseptual (Dasar)
1. Diberikan arsitektur jaringan saraf sederhana: `fc1 = nn.Linear(10, 32)` diikuti `fc2 = nn.Linear(32, 4)`. Hitunglah total parameter terbobot (bobot dan bias) yang harus dilatih pada model ini!
   - **Solusi**:
     - Lapisan 1 (`fc1`): Matriks bobot berukuran $32 \times 10 = 320$, vektor bias berukuran $32$. Total = $320 + 32 = 352$ parameter.
     - Lapisan 2 (`fc2`): Matriks bobot berukuran $4 \times 32 = 128$, vektor bias berukuran $4$. Total = $128 + 4 = 132$ parameter.
     - Total Keseluruhan = $352 + 132 = 484$ parameter *trainable*.

#### Tingkat 2: Analisis Matematis & Graf (Menengah)
1. Jelaskan bagaimana mesin Autograd PyTorch mengevaluasi turunan pada fungsi `z = torch.relu(x)` pada titik $x = 0$ di mana secara teori kalkulus klasik fungsi ReLU tidak memiliki turunan (*non-differentiable*)!
   - **Solusi**:
     Fungsi ReLU didefinisikan sebagai $f(x) = \max(0, x)$. Pada titik $x = 0$, terdapat diskontinuitas turunan (turunan kiri bernilai 0 dan turunan kanan bernilai 1). Dalam teori optimasi modern, rentang $[0, 1]$ disebut sebagai *subgradient*. Untuk menjaga determinisme komputasi perangkat lunak, PyTorch secara konvensi menetapkan turunan ReLU pada $x = 0$ bernilai tepat **0.0** (mengikuti aturan turunan sisi kiri).

#### Tingkat 3: Rekayasa Komputasional (Lanjut)
1. Mengapa memanggil `loss_tracker += loss` di dalam loop pelatihan PyTorch dapat menyebabkan komputer kehabisan memori (*Out-Of-Memory* / OOM error), dan bagaimana cara memperbaikinya?
   - **Solusi**:
     Variabel `loss` yang dihasilkan oleh fungsi rugi merupakan tensor PyTorch yang masih terikat pada seluruh riwayat graf komputasi aktif (`grad_fn`). Jika kita menjumlahkannya langsung ke variabel Python biasa (`loss_tracker += loss`), PyTorch akan mempertahankan seluruh graf komputasi dari batch-batch sebelumnya di memori RAM/GPU tanpa pernah membebaskannya (*garbage collection* terhalang). Perbaikannya adalah dengan mengekstrak nilai skalar murni menggunakan method `.item()`:
     ```python
     loss_tracker += loss.item() * len(batch_y)
     ```
     Dengan memanggil `.item()`, nilai numerik skalar float Python diekstrak dan graf komputasi batch tersebut dapat langsung dihapus dari memori setelah `optimizer.step()`.

---

### Mini Project Chapter 23

#### Judul Tugas Mini Project:
**"Building an End-to-End Classification Pipeline Using PyTorch"**

#### Deskripsi Tugas:
Rancang dan bangun pipeline klasifikasi terpadu dari awal menggunakan pustaka **PyTorch** pada jalur CPU:
1. Bangkitkan dataset klasifikasi sintetis dengan seed tetap, bagi menjadi partisi 3-way (Train 70%, Val 15%, Test 15%) secara terisolasi, dan terapkan standardisasi fitur bebas kebocoran data.
2. Bangun kelas model jaringan saraf lapis banyak `TabularMLP` berbasis `nn.Module` dan model acuan linier sederhana `SimpleBaseline`.
3. Tuliskan loop pelatihan dan validasi terpisah dengan metrik loss dan akurasi per epoch.
4. Lakukan evaluasi akhir pada test set terisolasi, hitung matriks konfusi, dan bandingkan performa MLP terhadap baseline.
5. Simpan checkpoint bobot model menggunakan `state_dict`, muat kembali dengan aman menggunakan `weights_only=True`, dan buktikan konsistensi inferensi $100\%$ identik.

#### Rubrik Penilaian Proyek (Total: 100 Poin):

| Kriteria Penilaian | Bobot | Deskripsi Skor Maksimal |
|---|:---:|---|
| **1. Data & Protokol Split** | 20 Poin | Pembangkitan data terdefinisi (`seed=42`), pemisahan partisi 3-way (Train, Val, Test) terbukti tidak tumpang tindih, dan normalisasi fitur terisolasi hanya pada statistik data latih (zero data leakage). |
| **2. Model & Tensor Correctness** | 20 Poin | Subclassing `nn.Module` tepat, penentuan dimensi lapisan dan jumlah parameter akurat, serta luaran model berupa logits mentah tanpa Softmax prematur yang cocok untuk `nn.CrossEntropyLoss`. |
| **3. Training & Validation Loop** | 25 Poin | Implementasi loop pelatihan baku (`optimizer.zero_grad()`, `backward()`, `step()`), penggunaan tepat `model.train()` dan `model.eval()`, serta pembungkusan evaluasi di dalam `with torch.no_grad()`. |
| **4. Evaluation & Reproducibility** | 20 Poin | Evaluasi akhir pada test set terisolasi, perbandingan performa objektif terhadap baseline linier, matriks konfusi 2x2, serta penyimpanan dan pemuatan aman checkpoint `state_dict` (`weights_only=True`). |
| **5. Penjelasan Hasil & Keterbatasan** | 15 Poin | Analisis keunggulan representasi non-linier, dokumentasi batas reproduksibilitas floating-point antar-perangkat keras, dan pembahasan keterbatasan data sintetis secara jujur. |

---

## 13. Referensi Ilmiah Terverifikasi

1. **Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J., Chanan, G., ... & Chintala, S. (2019)**. *PyTorch: An Imperative Style, High-Performance Deep Learning Library*. Advances in Neural Information Processing Systems (NeurIPS 32), 8024-8035. URL: [https://proceedings.neurips.cc/paper/2019/hash/bdbca288fee7f92f2bfa9f7012727740-Abstract.html](https://proceedings.neurips.cc/paper/2019/hash/bdbca288fee7f92f2bfa9f7012727740-Abstract.html)
2. **PyTorch Foundation (2024)**. *PyTorch Documentation and Official Tutorials (v2.x)*. URL: [https://pytorch.org/docs/stable/index.html](https://pytorch.org/docs/stable/index.html)
3. **Goodfellow, I., Bengio, Y., & Courville, A. (2016)**. *Deep Learning* (Chapter 6: Deep Feedforward Networks). MIT Press. URL: [https://www.deeplearningbook.org/](https://www.deeplearningbook.org/)
4. **Baydin, A. G., Pearlmutter, B. A., Radul, A. A., & Siskind, J. M. (2018)**. *Automatic differentiation in machine learning: a survey*. Journal of Machine Learning Research, 18(153), 1-43. URL: [https://jmlr.org/papers/v18/17-468.html](https://jmlr.org/papers/v18/17-468.html)
5. **ACM / IEEE-CS (2024)**. *Computer Science Curricula 2023 (CS2023): Artificial Intelligence Knowledge Area*. ACM Press. URL: [https://csed.acm.org/](https://csed.acm.org/)

