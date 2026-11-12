# Panduan Berkas Kode Chapter 31: Transfer Learning & Fine-Tuning

Katalog ini berisi 12 skrip Python mandiri yang mendemonstrasikan konsep, perumusan aljabar, implementasi PyTorch, dan studi komparasi Transfer Learning, Feature Extraction, Fine-Tuning, serta Low-Rank Adaptation (LoRA). Seluruh skrip dirancang khusus agar dapat dieksekusi secara instan pada CPU komputer standar tanpa ketergantungan kartu grafis (GPU).

---

## 1. Daftar dan Deskripsi Skrip

| No | Nama Berkas | Topik Utama | Kompleksitas / Dependensi |
|---|---|---|:---:|
| 1 | `transfer_learning_basics.py` | Perbedaan Training from Scratch vs Pretrained Model, Source vs Target Task | PyTorch, CPU |
| 2 | `pretrained_feature_extraction.py` | Pembekuan bobot (`requires_grad=False`), penggantian head, dan mode `train()` vs `eval()` | PyTorch, CPU |
| 3 | `freeze_unfreeze_layers.py` | Inspeksi status lapisan bergradien dan progresive unfreezing bertingkat | PyTorch, CPU |
| 4 | `linear_probing.py` | Pelatihan kepala klasifikasi linier murni pada representasi fitur beku | PyTorch, SGD, CPU |
| 5 | `partial_fine_tuning.py` | Pencairan lapisan atas dengan *differential learning rates* via parameter groups | PyTorch, AdamW, CPU |
| 6 | `full_fine_tuning.py` | Pembaruan seluruh parameter model dan demonstrasi *Catastrophic Forgetting* | PyTorch, CPU |
| 7 | `domain_shift_demo.py` | Simulasi *Covariate Shift*, degradasi performa, dan pemulihan via adaptasi domain | PyTorch, CPU |
| 8 | `negative_transfer_demo.py` | Kondisi kegagalan transfer akibat representasi tugas sumber yang kontradiktif | PyTorch, CPU |
| 9 | `lora_from_scratch.py` | Formulasi matematis $W' = W_0 + \frac{\alpha}{r} BA$, inisialisasi $B=0$, dan bukti rank | NumPy, CPU |
| 10 | `lora_parameter_count.py` | Analisis penghematan parameter dan memori lintas dimensi $d$ dan rank $r$ | Python murni |
| 11 | `lora_pytorch_demo.py` | Implementasi kelas kustom `LoRALinear` di PyTorch dan verifikasi aliran gradien | PyTorch, CPU |
| 12 | `fine_tuning_comparison.py` | Komparasi empiris 5 metode (Scratch, Probing, Partial FT, Full FT, LoRA) pada data identik | PyTorch, CPU |

---

## 2. Instruksi Eksekusi

Jalankan seluruh skrip secara individual melalui terminal menggunakan Python 3.10+:

```bash
# Contoh menjalankan demonstrasi dasar transfer learning
python content/level-4-advanced-ai/chapter-31/code/transfer_learning_basics.py

# Contoh menjalankan perbandingan 5 paradigma adaptasi
python content/level-4-advanced-ai/chapter-31/code/fine_tuning_comparison.py
```

Seluruh skrip menggunakan seed deterministik (`torch.manual_seed(42)` dan `np.random.seed(42)`) untuk menjamin reproduksibilitas numerik 100%.

