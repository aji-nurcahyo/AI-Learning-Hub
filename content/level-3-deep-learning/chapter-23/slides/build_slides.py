"""
Chapter 23 - Deep Learning with PyTorch
Pembangkit Slide Presentasi Reveal.js (deck.json)
Menghasilkan 22 slide presentasi modular, semantik, dan valid.
"""

import json
from pathlib import Path

slides = [
    {
        "kind": "cover",
        "title": "Deep Learning with PyTorch",
        "body": "Membangun Jaringan Saraf Tiruan Modern Menggunakan Ekosistem Komputasi Tensor dan Autograd PyTorch.",
        "notes": "Slide pembuka Bab 23. Sampaikan bahwa setelah menguasai dasar matematika di Bab 21-22, kita kini melangkah ke framework standar industri deep learning modern yaitu PyTorch."
    },
    {
        "kind": "content",
        "title": "Peran Framework Deep Learning",
        "body": "Transisi dari komputasi manual ke pustaka terakselerasi:\n- Abstraksi Tensor berdimensi N dengan dukungan eksekusi heterogen (CPU dan GPU/CUDA).\n- Diferensiasi otomatis dinamis (Autograd) berbasis Computational Graph terarah.\n- Lapisan neural teruji (torch.nn) dan optimizer adaptif terintegrasi.\n- Ekosistem pemrosesan batch, multi-threading, dan pipeline data terukur.",
        "notes": "Jelaskan bahwa framework deep learning menghilangkan kerumitan penurunan aljabar matriks manual dan memungkinkan peneliti berfokus pada inovasi arsitektur model."
    },
    {
        "kind": "content",
        "title": "Anatomi Tensor PyTorch",
        "body": "Struktur data inti untuk merepresentasikan sinyal, bobot, dan gradien:\n- Shape / Size: Tuple dimensi tensor, misalnya (Batch, Channels, Height, Width).\n- Dtype: Tipe data numerik (torch.float32 untuk bobot/fitur, torch.long untuk label klasifikasi).\n- Device: Lokasi memori komputasi fisik (cpu atau cuda:0).\n- Layout memori kontigu mendukung paralelisasi instruksi SIMD prosesor.",
        "notes": "Tekankan pentingnya memeriksa shape dan dtype sebelum melakukan operasi matriks untuk menghindari bug penyiaran dimensi yang tidak disengaja."
    },
    {
        "kind": "split",
        "title": "Perbandingan: PyTorch Tensor vs NumPy ndarray",
        "body": "Persamaan dan Perbedaan Utama:\n- Keduanya menyediakan array n-dimensi berkinerja tinggi dengan sintaksis pengirisan (slicing) serupa.\n- Jembatan zero-copy: torch.from_numpy() dan tensor.numpy() berbagi blok memori fisik yang sama.\n- PyTorch Tensor unggul dalam akselerasi perangkat keras GPU dan pelacakan graf gradien autograd.\n- NumPy terbatas pada komputasi CPU murni tanpa kapabilitas backpropagation otomatis.",
        "notes": "Beri contoh bahwa mutasi in-place pada tensor PyTorch akan langsung mengubah array NumPy yang terhubung karena keduanya menggunakan penunjuk memori yang sama."
    },
    {
        "kind": "content",
        "title": "Manipulasi Bentuk Tensor",
        "body": "Operasi reshaping fundamental tanpa penyalinan memori yang mahal:\n- view(shape): Mengubah sudut pandang tensor jika memori berurutan (contiguous).\n- reshape(shape): Fleksibel mengubah bentuk, melakukan cloning bila memori tidak berurutan.\n- unsqueeze(dim): Menyisipkan dimensi ukuran 1 pada posisi indeks tertentu.\n- squeeze(dim): Menghilangkan dimensi ukuran 1 yang tidak perlu.\n- permute(dims): Mengatur ulang urutan sumbu dimensi tensor.",
        "notes": "Jelaskan perbedaan antara view dan reshape. Sarankan menggunakan reshape bila pembelajar tidak yakin apakah memori tensor kontigu atau tidak."
    },
    {
        "kind": "content",
        "title": "Autograd: Graf Komputasi Dinamis",
        "body": "Mekanisme kalkulus diferensial otomatis bawaan:\n- Parameter requires_grad=True menginstruksikan PyTorch untuk mencatat seluruh operasi ke graf komputasi (DAG).\n- Setiap operasi menghasilkan fungsi gradien grad_fn yang tersambung ke simpul induk.\n- Pemanggilan loss.backward() menelusuri graf balik secara terbalik (reverse topological sort).\n- Gradien parsial disimpan secara otomatis pada atribut param.grad.",
        "notes": "Hubungkan dengan Bab 22 di mana kita mempelajari aturan rantai multivariat. Autograd adalah otomasi perangkat lunak dari aturan rantai tersebut."
    },
    {
        "kind": "case",
        "title": "Jebakan Fatal: Akumulasi Gradien Bawaan",
        "body": "Kasus Nyata pada Pelatihan Jaringan Saraf:\n- Masalah: PyTorch secara default MENJUMLAHKAN (akumulasi) gradien baru ke param.grad yang sudah ada.\n- Gejala: Jika loop pelatihan tidak mereset gradien, langkah optimasi menjadi berlipat ganda dan meledak.\n- Solusi Wajib: Panggil optimizer.zero_grad() di awal setiap langkah iterasi batch.\n- Catatan Ahli: Akumulasi gradien hanya sengaja dipakai saat menyimulasikan batch size raksasa pada RAM terbatas.",
        "notes": "Ini adalah salah satu kesalahan paling umum bagi pemula di PyTorch. Tekankan urutan: zero_grad() -> forward -> loss -> backward() -> step()."
    },
    {
        "kind": "code",
        "title": "Contoh Autograd & Turunan Analitis",
        "body": "Verifikasi perhitungan gradien otomatis terhadap diferensiasi kalkulus manual.",
        "code": "import torch\nw = torch.tensor(2.0, requires_grad=True)\nx = torch.tensor(3.0)\nb = torch.tensor(1.0, requires_grad=True)\n\n# Forward pass: y = (w * x + b)^2\ny = (w * x + b) ** 2  # (2*3 + 1)^2 = 49.0\ny.backward()\n\n# Analitis: dy/dw = 2*(w*x + b)*x = 2*7*3 = 42.0\n# Analitis: dy/db = 2*(w*x + b)*1 = 2*7*1 = 14.0\nprint(f\"dy/dw Autograd: {w.grad.item():.1f}\")\nprint(f\"dy/db Autograd: {b.grad.item():.1f}\")",
        "notes": "Tunjukkan bahwa autograd PyTorch menghasilkan angka 42.0 dan 14.0 yang persis cocok dengan penurunan rumus diferensiasi rantai."
    },
    {
        "kind": "content",
        "title": "Membangun Model dengan torch.nn.Module",
        "body": "Fondasi berorientasi objek untuk seluruh model deep learning:\n- Subclassing: Setiap arsitektur mewarisi kelas dasar torch.nn.Module.\n- Konstruktor (__init__): Mendefinisikan lapisan parameter (nn.Linear, nn.Dropout, dll.) dan memanggil super().__init__().\n- Forward (__call__): Menentukan aliran komputasi dari masukan tensor menuju luaran prediksi.\n- Registrasi Parameter: Seluruh atribut layer secara otomatis terdaftar di dalam named_parameters().",
        "notes": "Ingatkan mahasiswa untuk selalu memanggil model(x), bukan model.forward(x), karena method __call__ mengeksekusi hooks internal PyTorch."
    },
    {
        "kind": "content",
        "title": "Mengapa Logits Mentah Tanpa Softmax?",
        "body": "Pemisahan lapisan luaran dan perhitungan fungsi rugi:\n- Model klasifikasi PyTorch lazimnya mengeluarkan logits mentah (unnormalized scores).\n- nn.CrossEntropyLoss menggabungkan log_softmax dan NLLLoss secara bersamaan.\n- Manfaat: Menggunakan trik stabilitas subtract-max secara internal untuk mencegah floating-point overflow.\n- Jika Softmax dipasang di akhir model lalu dimasukkan ke CrossEntropyLoss, fungsi rugi akan terhitung ganda dan salah.",
        "notes": "Tautkan ke Bab 21 di mana kita membahas subtract-max trick. PyTorch mengimplementasikan trik ini langsung di dalam C++ kernel CrossEntropyLoss."
    },
    {
        "kind": "split",
        "title": "Dataset vs DataLoader",
        "body": "Pemisahan Tanggung Jawab Aliran Data:\n- torch.utils.data.Dataset: Bertanggung jawab menyimpan satu sampel data dan labelnya (__len__ dan __getitem__).\n- torch.utils.data.DataLoader: Bertanggung jawab mengelompokkan sampel menjadi mini-batch, pengacakan (shuffling), dan multi-processing.\n- Pemisahan ini menjaga kode data modular dan tidak bergantung pada arsitektur model.",
        "notes": "Jelaskan bahwa Dataset seperti perpustakaan buku, sedangkan DataLoader adalah kurir yang mengantarkan sekotak buku per giliran."
    },
    {
        "kind": "case",
        "title": "Protokol Partisi 3-Way & Pencegahan Data Leakage",
        "body": "Tata Kelola Data Pelatihan yang Bersih:\n- Training Set (70%): Untuk menghitung gradien dan memperbarui bobot parameter.\n- Validation Set (15%): Untuk tuning hiperparameter dan pemilihan checkpoint model terbaik.\n- Test Set (15%): Terisolasi sepenuhnya untuk penilaian akhir yang tidak bias.\n- Aturan Kebocoran Data: Skaler normalisasi (mean dan std) WAJIB dihitung hanya dari data latih, lalu diaplikasikan ke val dan test.",
        "notes": "Tekankan bahwa menghitung mean dan std dari seluruh dataset sebelum partisi merupakan bentuk kebocoran data yang memalsukan hasil evaluasi."
    },
    {
        "kind": "code",
        "title": "Implementasi Custom Dataset Modular",
        "body": "Struktur subclassing Dataset untuk data tabular terstandar.",
        "code": "from torch.utils.data import Dataset, DataLoader\n\nclass TabularDataset(Dataset):\n    def __init__(self, X, y):\n        self.X = torch.as_tensor(X, dtype=torch.float32)\n        self.y = torch.as_tensor(y, dtype=torch.long)\n\n    def __len__(self):\n        return len(self.X)\n\n    def __getitem__(self, idx):\n        return self.X[idx], self.y[idx]\n\n# shuffle=True HANYA pada train_loader!\nloader = DataLoader(ds, batch_size=32, shuffle=True)",
        "notes": "Jelaskan mengapa shuffle bernilai True hanya untuk train set dan harus bernilai False untuk validation serta test set agar evaluasi bersifat deterministik."
    },
    {
        "kind": "content",
        "title": "Anatomi Lengkap Training Loop",
        "body": "Lima langkah mutlak dalam setiap iterasi batch pelatihan:\n1. Model Train Mode: Mengaktifkan perilaku pelatihan via model.train().\n2. Reset Gradien: Memanggil optimizer.zero_grad() untuk membersihkan memori akumulator.\n3. Forward Pass: Mengevaluasi output logits = model(inputs).\n4. Evaluasi Loss: Menghitung fungsi rugi skalar loss = criterion(logits, targets).\n5. Propagasi Balik & Update: Mengeksekusi loss.backward() disusul optimizer.step().",
        "notes": "Gambarkan kelima langkah ini sebagai urutan mantra suci yang tidak boleh ditukar posisinya."
    },
    {
        "kind": "split",
        "title": "model.train() vs model.eval()",
        "body": "Mengontrol Perilaku Internal Lapisan Neural:\n- model.train(): Mengaktifkan dropout acak dan pelacakan running statistics pada batch normalization.\n- model.eval(): Menonaktifkan dropout (skala penuh) dan mengunci parameter batch normalization pada nilai estimasi populasi.\n- PENTING: model.eval() TIDAK menonaktifkan komputasi gradien autograd!\n- Untuk menghentikan pelacakan graf gradien, gabungkan dengan blok with torch.no_grad().",
        "notes": "Banyak pemula mengira model.eval() sudah otomatis menonaktifkan gradien. Tegaskan bahwa model.eval() dan torch.no_grad() memiliki tugas yang berbeda dan saling melengkapi."
    },
    {
        "kind": "content",
        "title": "Peran Blok torch.no_grad()",
        "body": "Optimasi efisiensi memori dan kecepatan pada tahap evaluasi:\n- Menginstruksikan autograd engine untuk tidak mengalokasikan graf komputasi (DAG).\n- Mengurangi konsumsi memori RAM/VRAM hingga lebih dari 50% saat inferensi batch besar.\n- Mempercepat waktu eksekusi evaluasi karena menghilangkan overhead pelacakan grad_fn.\n- Wajib digunakan pada validation loop, test loop, dan sistem inferensi produksi.",
        "notes": "Tunjukkan bahwa tanpa torch.no_grad(), memori GPU/RAM akan membengkak saat evaluasi karena graf komputasi terus terakumulasi."
    },
    {
        "kind": "code",
        "title": "Siklus Evaluasi Model Terisolasi",
        "body": "Pola baku loop evaluasi yang hemat memori dan bebas pembaruan bobot.",
        "code": "def evaluate(model, loader, criterion):\n    model.eval()\n    total_loss, correct, total = 0.0, 0, 0\n    with torch.no_grad():\n        for X_batch, y_batch in loader:\n            logits = model(X_batch)\n            loss = criterion(logits, y_batch)\n            total_loss += loss.item() * len(y_batch)\n            preds = torch.argmax(logits, dim=-1)\n            correct += (preds == y_batch).sum().item()\n            total += len(y_batch)\n    return total_loss / total, (correct / total) * 100",
        "notes": "Soroti penggunaan loss.item() agar kita hanya mengambil nilai skalar float tanpa mempertahankan graf komputasi di variabel total_loss."
    },
    {
        "kind": "content",
        "title": "Penyimpanan dan Pemuatan: state_dict",
        "body": "Praktek terbaik serialisasi model deep learning:\n- state_dict: Dictionary Python yang memetakan setiap layer ke tensor bobot dan biasnya.\n- Jangan simpan seluruh objek model secara langsung (torch.save(model)) karena rentan terhadap perubahan kode kelas Python.\n- Pemuatan Aman: Gunakan torch.load(path, weights_only=True) untuk memitigasi celah keamanan eksekusi kode acak (CVE pickle).\n- Urutan Pemuatan: Inisialisasi arsitektur kelas baru -> load_state_dict() -> model.eval().",
        "notes": "Tekankan alasan keamanan PyTorch versi modern yang merekomendasikan weights_only=True."
    },
    {
        "kind": "case",
        "title": "Studi Kasus: MLP Non-Linier vs Baseline Linier",
        "body": "Hasil Eksperimen Terverifikasi pada Data Sintetis Non-Linier (N=1000):\n- Arsitektur MLP: 6 masukan -> 16 hidden (ReLU) -> 8 hidden (ReLU) -> 2 output (266 parameter).\n- Model Acuan: Regresi Linier tunggal 6 masukan -> 2 output (14 parameter).\n- Hasil Test Set (N=150 terisolasi):\n  - SimpleBaseline Linier: Akurasi 83.33% (Loss: 0.4274)\n  - TabularMLP Non-Linier: Akurasi 88.67% (Loss: 0.3234)\n- Kesimpulan: Kapasitas representasi non-linier MLP memberikan keunggulan terukur +5.33%.",
        "notes": "Paparkan data eksperimen nyata dari file experiments/results_ch23.json. Tunjukkan bahwa hasil ini diperoleh dari pengujian CPU yang dapat direproduksi."
    },
    {
        "kind": "content",
        "title": "Batas Reproduksibilitas PyTorch",
        "body": "Memahami batasan komputasi floating-point non-deterministik:\n- torch.manual_seed(42) menjamin urutan inisialisasi acak identik pada mesin yang sama.\n- Keterbatasan: Operasi aljabar CUDA/cuDNN dan pustaka BLAS multi-threading memiliki sifat non-asosiatif floating-point.\n- Hasil pelatihan tidak dijamin identik 100% pada versi PyTorch, arsitektur CPU, atau GPU yang berbeda.\n- Praktik Terbaik: Dokumentasikan versi perangkat lunak, sistem operasi, seed, dan konfigurasi hiperparameter secara transparan.",
        "notes": "Jelaskan mengapa seed sama di laptop intel dan macbook M-series bisa menghasilkan loss yang sedikit berbeda karena urutan akumulasi floating point berbeda."
    },
    {
        "kind": "content",
        "title": "Daftar Periksa Debugging PyTorch (Checklist)",
        "body": "Tujuh pemeriksaan cepat saat menghadapi bug pelatihan:\n1. Tensor Shape Mismatch: Periksa dimensi input layer fc1 cocok dengan jumlah fitur masukan.\n2. Lupa Reset Gradien: Apakah optimizer.zero_grad() terpanggil sebelum backward?\n3. Mode Evaluasi Tertinggal: Apakah model.eval() aktif saat inferensi dan model.train() saat pelatihan?\n4. Target Dtype Klasifikasi: Pastikan label bertipe torch.long, bukan float.\n5. Logits vs Softmax: Pastikan tidak memanggil nn.Softmax sebelum nn.CrossEntropyLoss.\n6. Device Mismatch: Pastikan tensor masukan dan bobot model berada pada device yang sama.\n7. Detach Graph: Gunakan loss.item() saat mencatat metrik agar tidak memicu memory leak.",
        "notes": "Beri tahu mahasiswa bahwa 90% kesalahan coding deep learning pemula berasal dari ketujuh poin ini."
    },
    {
        "kind": "summary",
        "title": "Rangkuman Bab 23",
        "body": "Poin Inti Pembelajaran:\n- PyTorch menyediakan ekosistem terintegrasi untuk komputasi tensor dan graf diferensiasi dinamis.\n- Abstraksi Dataset dan DataLoader menjamin modularitas dan isolasi partisi data tanpa leakage.\n- Arsitektur berbasis nn.Module memisahkan definisi layer dengan aliran komputasi forward pass.\n- Manajemen siklus hidup (model.train, model.eval, torch.no_grad, state_dict) menjamin reproduksibilitas dan keandalan sistem inferensi produksi.",
        "notes": "Tutup sesi dan beri transisi menuju Bab 24 di mana kita akan menerapkan PyTorch untuk arsitektur Convolutional Neural Networks pada data visual."
    }
]

output_path = Path(__file__).parent / "deck.json"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(slides, f, indent=2, ensure_ascii=False)

print(f"Berhasil membuat {len(slides)} slide Reveal.js di: {output_path}")

