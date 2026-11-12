"""
build_slides.py
Script pembuat presentasi Reveal.js untuk Chapter 31: Transfer Learning & Fine-Tuning.
Menghasilkan slides/deck.json berisi 32 slide terstruktur dengan semantic kinds:
['cover', 'content', 'code', 'split', 'case', 'summary']
"""

import json
import os

slides = [
    {
        "kind": "cover",
        "title": "Transfer Learning & Fine-Tuning",
        "body": "Pondasi Adaptasi Model Modern: Pretraining, Feature Extraction, Progressive Unfreezing, hingga Parameter-Efficient Fine-Tuning (LoRA)\n\nLevel 4: Applied & Advanced AI Systems\nArtificial Intelligence Learning Hub",
        "notes": "Selamat datang di Bab 31 AI Learning Hub. Bab ini membuka Level 4 dengan fokus pada teknik adaptasi model modern: dari transfer learning klasik hingga LoRA."
    },
    {
        "kind": "content",
        "title": "Mengapa Transfer Learning?",
        "body": "- Melatih model skala besar dari awal (from scratch) membutuhkan jutaan data berlabel dan ribuan GPU-hours.\n- Kebanyakan masalah dunia nyata berada dalam regim data terbatas (low-data regime).\n- Transfer Learning memanfaatkan pengetahuan (representasi fitur) yang telah dipelajari pada tugas sumber (Source Task) untuk mempercepat dan meningkatkan performa pada tugas target (Target Task).",
        "notes": "Jelaskan analogi transfer belajar manusia: seseorang yang sudah bisa mengendarai sepeda motor akan lebih cepat belajar mengendarai mobil dibanding orang yang belum pernah mengemudi sama sekali."
    },
    {
        "kind": "content",
        "title": "Tujuan Pembelajaran Bab 31",
        "body": "1. Membedakan pelatihan from scratch vs pemanfaatan pretrained model.\n2. Menguasai alur kerja Feature Extraction dan Linear Probing.\n3. Mengatur pembekuan lapisan (layer freezing) dan progressive unfreezing.\n4. Menerapkan Partial dan Full Fine-Tuning dengan differential learning rates.\n5. Mengidentifikasi domain shift, negative transfer, dan catastrophic forgetting.\n6. Merumuskan dan mengimplementasikan Low-Rank Adaptation (LoRA) secara matematis.",
        "notes": "Soroti bahwa pemahaman matematis LoRA dan manajemen status komputasi PyTorch (train vs eval) adalah kompetensi inti bab ini."
    },
    {
        "kind": "split",
        "title": "From Scratch vs Transfer Learning",
        "body": "Kiri: Training from Scratch\n- Bobot diinisialisasi acak.\n- Membutuhkan data latih masif.\n- Konvergensi lambat dan rentan overfitting.\n- Biaya komputasi tinggi.\n\nKanan: Transfer Learning\n- Bobot awal membawa prior representasi visual/bahasa yang kaya.\n- Efektif pada dataset target kecil.\n- Konvergensi jauh lebih cepat.\n- Menghemat energi dan sumber daya komputasi.",
        "notes": "Tekankan perbedaan kurva loss dan kebutuhan jumlah sampel data antara scratch vs transfer learning."
    },
    {
        "kind": "content",
        "title": "Anatomi Model Pra-terlatih: Backbone & Head",
        "body": "- **Backbone (Feature Extractor)**: Tumpukan lapisan konvolusi atau Transformer yang memetakan masukan mentah ke ruang representasi laten berdimensi tinggi.\n- **Classification Head (Task-Specific)**: Lapisan proyeksi linear teratas yang memetakan vektor representasi fitur ke probabilitas kelas tugas target.\n- Dalam adaptasi model, kepala tugas sumber dibuang dan digantikan oleh kepala baru yang sesuai dengan dimensi kelas tugas target.",
        "notes": "Jelaskan bahwa representasi di lapisan bawah bersifat umum (garis, tekstur, token), sementara lapisan atas lebih spesifik tugas."
    },
    {
        "kind": "content",
        "title": "Feature Extraction & Linear Probing",
        "body": "- Seluruh bobot backbone dibekukan secara permanen: `param.requires_grad = False`.\n- Hanya bobot kepala klasifikasi baru (Linear Probe) yang diperbarui gradiennya.\n- Sangat cepat, hemat komputasi, dan tidak memerlukan memori untuk status optimizer pada lapisan backbone.\n- Cocok jika domain data target sangat mirip dengan domain data sumber.",
        "notes": "Linear Probing adalah teknik baseline wajib sebelum mencoba fine-tuning yang lebih agresif."
    },
    {
        "kind": "code",
        "title": "Implementasi Freezing di PyTorch",
        "body": "Membekukan bobot backbone dan mengganti kepala klasifikasi baru:",
        "code": "# 1. Bekukan backbone\nfor param in backbone.parameters():\n    param.requires_grad = False\n\n# 2. Pasang kepala baru\nmodel = nn.Sequential(\n    backbone,\n    nn.Linear(feature_dim, num_target_classes)\n)\n\n# 3. Optimizer hanya melatih kepala baru\noptimizer = optim.Adam(\n    model[1].parameters(), lr=1e-2\n)",
        "notes": "Perhatikan bahwa hanya parameter kepala baru yang diteruskan ke optimizer, menghemat status momen Adam."
    },
    {
        "kind": "split",
        "title": "Jebakan Runtime: Mode train() vs eval()",
        "body": "Kiri: Mengapa Membekukan Saja Tidak Cukup?\n- `requires_grad = False` hanya mencegah perhitungan gradien dan pembaruan bobot.\n- Lapisan seperti Dropout tetap membuang neuron secara acak jika model berada dalam mode `train()`!\n- Lapisan BatchNorm tetap memperbarui running mean dan running variance.\n\nKanan: Praktik Terbaik Feature Extraction\n- Wajib memanggil `backbone.eval()` untuk mengunci perilaku deterministik.\n- Kepala baru tetap dapat disetel ke mode `train()` saat pelatihan berlangsung.",
        "notes": "Ini adalah salah satu bug paling umum dalam rekayasa transfer learning di PyTorch yang sering diabaikan praktisi."
    },
    {
        "kind": "content",
        "title": "Strategi Fine-Tuning: Spektrum Adaptasi",
        "body": "Tiga tingkatan adaptasi model pra-terlatih:\n\n1. **Linear Probing**: 0% backbone dilatih (Hanya kepala baru).\n2. **Partial Fine-Tuning**: 10-30% lapisan teratas backbone dilatih, lapisan bawah dibekukan.\n3. **Full Fine-Tuning**: 100% parameter model diperbarui dengan laju pembelajaran kecil.",
        "notes": "Bahas spektrum trade-off antara fleksibilitas adaptasi versus risiko overfitting dan catastrophic forgetting."
    },
    {
        "kind": "content",
        "title": "Differential Learning Rates",
        "body": "- Memberikan perlakuan laju pembelajaran berbeda untuk setiap komponen:\n  $$\\eta_{\\text{backbone}} \\ll \\eta_{\\text{head}}$$\n- Contoh praktis: $\\eta_{\\text{backbone}} = 10^{-4}$ (penyesuaian halus), sedangkan $\\eta_{\\text{head}} = 10^{-2}$ (pembelajaran cepat dari nol).\n- Meminimalkan distorsi pada fitur fondasi sekaligus mempercepat konvergensi pemetaan kelas target.",
        "notes": "Gunakan optimizer parameter groups di PyTorch untuk menerapkan differential learning rates secara rapi."
    },
    {
        "kind": "code",
        "title": "Konfigurasi Parameter Groups di PyTorch",
        "body": "Mengatur laju pembelajaran bertingkat pada optimizer AdamW:",
        "code": "optimizer = optim.AdamW([\n    {'params': model.top_layers.parameters(),\n     'lr': 1e-4,\n     'weight_decay': 1e-4},\n    {'params': model.classifier_head.parameters(),\n     'lr': 1e-2,\n     'weight_decay': 1e-3}\n])",
        "notes": "Tunjukkan fleksibilitas optimizer PyTorch dalam mengelola hiperparameter per grup modul."
    },
    {
        "kind": "split",
        "title": "Risiko 1: Catastrophic Forgetting",
        "body": "Kiri: Apa Itu Catastrophic Forgetting?\n- Fenomena di mana jaringan saraf melupakan representasi tugas sumber secara tiba-tiba setelah dilatih pada tugas baru.\n- Sering terjadi pada Full Fine-Tuning dengan laju pembelajaran terlalu tinggi.\n\nKanan: Mitigasi Saintifik\n- Gunakan conservative learning rate ($10^{-5}$ hingga $10^{-4}$).\n- Terapkan Weight Decay (L2 penalty) untuk menjaga bobot tetap dekat dengan inisialisasi.\n- Gunakan Parameter-Efficient Fine-Tuning (LoRA).",
        "notes": "Kaitkan dengan hasil eksperimen full_fine_tuning.py di mana akurasi source task runtuh saat LR agresif digunakan."
    },
    {
        "kind": "split",
        "title": "Risiko 2: Domain Shift & Covariate Shift",
        "body": "Kiri: Karakteristik Pergeseran Domain\n- Distribusi masukan bergeser: $P_{\\text{target}}(X) \\neq P_{\\text{source}}(X)$.\n- Hubungan label kondisional $P(Y|X)$ diasumsikan relatif invarian.\n- Contoh: Sensor beroperasi di lingkungan suhu ekstrem atau citra kamera dengan kondisi pencahayaan berbeda.\n\nKanan: Dampak & Solusi\n- Akurasi model pra-terlatih dapat turun drastis tanpa adaptasi.\n- Solusi: Domain adaptation menggunakan sedikit sampel target terlabel untuk mengkalibrasi ulang batas keputusan.",
        "notes": "Rujuk hasil simulasi domain_shift_demo.py di mana pergeseran kovariat menyebabkan penurunan akurasi sebelum adaptasi."
    },
    {
        "kind": "split",
        "title": "Risiko 3: Transfer Negatif (Negative Transfer)",
        "body": "Kiri: Kapan Transfer Gagal?\n- Terjadi ketika tugas sumber dan tugas target tidak memiliki kesamaan representasi atau memiliki korelasi terbalik (spurious correlation).\n- Representasi pra-terlatih bertindak sebagai bias yang menyesatkan.\n\nKanan: Indikasi & Penanganan\n- Model transfer berkinerja lebih buruk daripada model yang dilatih dari awal (Scratch).\n- Wajib melakukan benchmarking terhadap model scratch untuk memastikan transfer learning memberikan nilai tambah nyata.",
        "notes": "Ini alasan mengapa model baseline scratch selalu wajib dibangun sebagai pembanding empiris."
    },
    {
        "kind": "content",
        "title": "Mengapa Butuh PEFT?",
        "body": "- Model fondasi modern memiliki milyaran hingga ratusan milyar parameter.\n- Full fine-tuning membutuhkan memori GPU/RAM masif untuk menyimpan status optimizer (Adam membutuhkan 16 byte per parameter).\n- Menyimpan model penuh terpisah untuk setiap tugas pelanggan sangat tidak efisien (ratusan gigabyte per model).\n- **Solusi**: Parameter-Efficient Fine-Tuning (PEFT) hanya melatih sebagian kecil parameter (< 1%).",
        "notes": "PEFT mengubah paradigma deployment AI: satu base model raksasa melayani puluhan adapter kecil."
    },
    {
        "kind": "content",
        "title": "Konsep Low-Rank Adaptation (LoRA)",
        "body": "- Diperkenalkan oleh Hu et al. (ICLR 2022).\n- Hipotesis Inti: Perubahan bobot selama adaptasi tugas ($\\Delta W$) memiliki rank intrinsik yang rendah (low intrinsic dimension).\n- Bobot dasar $W_0 \\in \\mathbb{R}^{d_{\\text{out}} \\times d_{\\text{in}}}$ dibekukan secara permanen.\n- Pembaruan dimodelkan melalui dekomposisi dua matriks rank rendah:\n  $$\\Delta W = \\frac{\\alpha}{r} (B \\cdot A)$$",
        "notes": "Dekomposisi ini memangkas jumlah parameter yang dilatih secara dramatis."
    },
    {
        "kind": "split",
        "title": "Dimensi Matriks & Dekomposisi LoRA",
        "body": "Kiri: Struktur Matriks\n- $W_0 \\in \\mathbb{R}^{d \\times d}$: Matriks bobot asli (BEKU).\n- $A \\in \\mathbb{R}^{r \\times d}$: Down-projection matrix (TRAINABLE).\n- $B \\in \\mathbb{R}^{d \\times r}$: Up-projection matrix (TRAINABLE).\n- Rank $r \\ll d$ (misal $r=4$ pada $d=768$).\n\nKanan: Jumlah Parameter\n- Parameter $W_0$: $d \\times d = 589.824$.\n- Parameter Adaptor: $2 \\times r \\times d = 2 \\times 4 \\times 768 = 6.144$.\n- Pengurangan parameter: **98.96% lebih hemat!**",
        "notes": "Tunjukkan visualisasi perkalian matriks (d x r) dikali (r x d) menghasilkan matriks berukuran (d x d)."
    },
    {
        "kind": "content",
        "title": "Inisialisasi Matematis LoRA",
        "body": "- Matriks $A$ diinisialisasi secara acak menggunakan distribusi Gaussian:\n  $$A \\sim \\mathcal{N}\\left(0, \\frac{1}{r}\\right)$$\n- Matriks $B$ diinisialisasi dengan **Nol Mutlak**:\n  $$B = 0$$\n- Konsekuensi Kritis:\n  $$\\Delta W = \\frac{\\alpha}{r} (B \\cdot A) = \\frac{\\alpha}{r} (0 \\cdot A) = 0$$\n- Model pada awal pelatihan berperilaku persis sama dengan base model tanpa distorsi!",
        "notes": "Tekankan pentingnya inisialisasi B=0 agar training dimulai dari titik representasi pretrained murni."
    },
    {
        "kind": "content",
        "title": "Faktor Penskalaan Alpha (Scaling Factor)",
        "body": "- Formula forward pass:\n  $$h = W_0 x + \\frac{\\alpha}{r} B A x$$\n- Konvensi praktis: $\\alpha$ umumnya disetel ke $2 \\times r$ atau konstan (misal $\\alpha = 16$).\n- Peran $\\alpha$: Menstabilkan magnitudo pembaruan saat bereksperimen dengan variasi rank $r$ yang berbeda, sehingga laju pembelajaran tidak perlu di-tuning ulang.",
        "notes": "Jelaskan mengapa scaling alpha/r penting ketika mengubah nilai r dari 4 ke 8 atau 16."
    },
    {
        "kind": "code",
        "title": "Implementasi LoRALinear di PyTorch",
        "body": "Mekanisme forward pass hemat memori tanpa membentuk matriks penuh:",
        "code": "class LoRALinear(nn.Module):\n    def __init__(self, in_features, out_features, r=4, alpha=8.0):\n        super().__init__()\n        self.base = nn.Linear(in_features, out_features, bias=False)\n        self.base.weight.requires_grad = False\n        self.scaling = alpha / r\n        self.A = nn.Parameter(torch.randn(r, in_features) * (1/r**0.5))\n        self.B = nn.Parameter(torch.zeros(out_features, r))\n\n    def forward(self, x):\n        base_out = self.base(x)\n        adapter_out = (x @ self.A.t()) @ self.B.t() * self.scaling\n        return base_out + adapter_out",
        "notes": "Perhatikan bahwa x dikalikan ke A terlebih dahulu (Batch, r) baru ke B (Batch, out), bukan B @ A penuh."
    },
    {
        "kind": "split",
        "title": "Zero-Latency Inference: Weight Merging",
        "body": "Kiri: Fase Pelatihan (Modular)\n- Matriks $A$ dan $B$ disimpan terpisah dari $W_0$.\n- Sangat fleksibel untuk bertukar adaptor antar tugas yang berbeda dalam hitungan milidetik.\n\nKanan: Fase Deployment (Merged)\n- Sebelum dideploy ke server produksi, satukan bobot:\n  $$W_{\\text{deploy}} = W_0 + \\frac{\\alpha}{r} (B \\cdot A)$$\n- Model kembali menjadi struktur Linear standar tanpa overhead komputasi tambahan saat melayani pengguna (zero inference latency).",
        "notes": "Jelaskan fitur merge_weights() dan unmerge_weights() yang telah kita uji di lora_pytorch_demo.py."
    },
    {
        "kind": "content",
        "title": "Analisis Penghematan Memori VRAM/RAM",
        "body": "- Dalam optimizer AdamW, setiap parameter trainable membutuhkan:\n  1. Bobot parameter (4 byte float32)\n  2. Nilai gradien (4 byte)\n  3. Momen orde pertama $m_t$ (4 byte)\n  4. Momen orde kedua $v_t$ (4 byte)\n  $$\\text{Total Memori per Param} = 16 \\text{ byte}$$\n- Dengan LoRA, $99\\%$ parameter berstatus beku (hanya 4 byte untuk bobot, 0 byte untuk gradien dan optimizer).\n- Memungkinkan pelatihan model besar pada perangkat CPU atau kartu grafis konsumen sederhana.",
        "notes": "Tunjukkan kalkulasi numerik dari skrip lora_parameter_count.py."
    },
    {
        "kind": "split",
        "title": "Hasil Komparasi Empiris 5 Paradigma",
        "body": "Hasil pengujian terstandarisasi (fine_tuning_comparison.py):\n\n1. **Scratch**: 1.506 param latih (100%), Akurasi: ~81.7%\n2. **Linear Probing**: 66 param latih (4.4%), Akurasi: ~83.3%\n3. **Partial FT**: 1.154 param latih (76.6%), Akurasi: ~88.3%\n4. **Full FT**: 1.506 param latih (100%), Akurasi: ~86.7%\n5. **LoRA (r=4)**: 322 param latih (21.4%), Akurasi: ~88.3%",
        "notes": "LoRA mencapai akurasi tertinggi setara Partial FT dengan parameter terbobot yang jauh lebih sedikit."
    },
    {
        "kind": "case",
        "title": "Studi Kasus: Pemantauan Turbin Industri",
        "body": "- **Skenario**: Model deteksi anomali dilatih pada turbin gas tipe A (Source Domain) di lingkungan uji terkontrol.\n- **Tantangan Target**: Diterapkan pada turbin tipe B di lingkungan laut lepas dengan kelembaban dan salinitas tinggi.\n- **Solusi Rekayasa**: Membekukan backbone ekstraksi sinyal sensor dan menerapkan adaptor LoRA ($r=4$) pada lapisan rekurensi/atensi menggunakan 50 rekaman data lokal.\n- **Hasil**: Degradasi akurasi pulih dari 68% menjadi 91% tanpa merusak model referensi pabrik.",
        "notes": "Bahas relevansi industri dari adaptasi domain dan transfer learning hemat sumber daya."
    },
    {
        "kind": "content",
        "title": "Pedoman Pemilihan Strategi Adaptasi",
        "body": "| Kondisi Data Target | Kesamaan dengan Domain Sumber | Strategi yang Disarankan |\n|---|---|---|\n| Data Sedikit (< 100 sampel) | Sangat Mirip | Linear Probing (Mencegah Overfitting) |\n| Data Sedikit (< 500 sampel) | Agak Berbeda | LoRA ($r=2$ atau $r=4$) |\n| Data Sedang (1.000+ sampel) | Cukup Berbeda | Partial FT atau LoRA ($r=8$) |\n| Data Masif (10.000+ sampel) | Sangat Berbeda | Full Fine-Tuning atau Scratch |",
        "notes": "Tabel ini menjadi panduan praktis mahasiswa dalam memilih strategi untuk proyek mandiri mereka."
    },
    {
        "kind": "content",
        "title": "Responsible Fine-Tuning & Mitigasi Bias",
        "body": "- **Inherited Biases**: Pretrained model dapat membawa bias representasi sosial, bahasa, atau sensor yang terserap saat pre-training.\n- **Label Quality**: Fine-tuning pada dataset target dengan label kotor atau tidak representatif akan merusak performa model secara keseluruhan.\n- **Audit Subpopulasi**: Selalu periksa apakah fine-tuning meningkatkan performa agregat namun mengorbankan subkelompok tertentu (*fairness trade-off*).",
        "notes": "Kaitkan dengan prinsip Responsible AI yang dipelajari pada Chapter 20 dan Chapter 30."
    },
    {
        "kind": "content",
        "title": "Keterbatasan Model Adaptation",
        "body": "1. **Bukan Solusi Universal**: Jika domain target memiliki fisika atau semantik yang berlawanan, pretraining tidak memberikan manfaat (*negative transfer*).\n2. **Ketergantungan Representasi**: Kualitas ekstraksi fitur bergantung mutlak pada keragaman dataset sumber.\n3. **Sensitivitas Hiperparameter**: Nilai rank $r$, $\\alpha$, dan laju pembelajaran LoRA membutuhkan validasi cermat pada validation set terisolasi.",
        "notes": "Ingatkan mahasiswa untuk selalu mengevaluasi secara kritis dan tidak berasumsi pretrained model selalu unggul."
    },
    {
        "kind": "summary",
        "title": "Ringkasan Bab 31",
        "body": "- Transfer Learning mengubah paradigma rekayasa AI dengan memanfaatkan representasi fitur terlatih.\n- Feature Extraction dan Linear Probing membekukan backbone untuk efisiensi komputasi maksimal.\n- Full Fine-Tuning memiliki fleksibilitas tinggi namun rentan terhadap Catastrophic Forgetting.\n- Low-Rank Adaptation (LoRA) mendekomposisi pembaruan bobot $\\Delta W = \\frac{\\alpha}{r} BA$, menghemat $>95\\%$ parameter dan memori tanpa mengorbankan performa target.",
        "notes": "Tutup dengan menegaskan bahwa teknik adaptasi model ini adalah landasan penting sebelum mempelajari Vector Search di Bab 32 dan RAG di Bab 33."
    },
    {
        "kind": "content",
        "title": "Referensi Akademik Terverifikasi",
        "body": "1. Hu, E. J., et al. (2021). LoRA: Low-Rank Adaptation of Large Language Models. ICLR 2022.\n2. Yosinski, J., et al. (2014). How transferable are features in deep neural networks? NeurIPS 2014.\n3. Pan, S. J., & Yang, Q. (2010). A Survey on Transfer Learning. IEEE TKDE.\n4. He, K., et al. (2016). Deep Residual Learning for Image Recognition. CVPR 2016.\n5. Howard, J., & Ruder, S. (2018). Universal Language Model Fine-tuning for Text Classification (ULMFiT). ACL 2018.",
        "notes": "Seluruh referensi ini adalah makalah primer yang membentuk fondasi transfer learning dan fine-tuning modern."
    }
]

def main():
    out_dir = os.path.join("content", "level-4-advanced-ai", "chapter-31", "slides")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "deck.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(slides, f, indent=2, ensure_ascii=False)
    print(f"Berhasil membuat {len(slides)} slide Reveal.js di {out_path}")

if __name__ == "__main__":
    main()

