import json
import sys
import io

cells = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# Chapter 01: Introduction to Artificial Intelligence\n",
            "## Notebook Pendamping & Eksperimen Konseptual\n",
            "\n",
            "Notebook ini mendampingi materi Chapter 01 untuk mengeksplorasi perbedaan antara sistem berbasis aturan (*Rule-Based System*), klasifikasi kata kunci, kerapuhan *exact matching*, serta metrik akurasi dan analisis matriks kontingensi sederhana (*confusion counts*).\n",
            "\n",
            "> **Catatan Integritas Data:** Seluruh dataset dalam notebook ini adalah data sintetis berukuran ringkas untuk tujuan ilustrasi konseptual, bukan model terlatih berskala penuh."
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "---\n",
            "### Bagian 1: Sistem Rekomendasi Berbasis Aturan (Rule-Based Recommender)\n",
            "Pada sistem berbasis aturan, manusia menuliskan seluruh logika *if-else* secara eksplisit. Komputer hanya mengeksekusi logika tersebut tanpa proses belajar dari data."
        ]
    },
    {
        "cell_type": "code",
        "metadata": {},
        "source": [
            "from typing import Dict, Any, List\n",
            "\n",
            "def recommend_study_path(profile: Dict[str, Any]) -> List[str]:\n",
            "    interest = profile.get('interest', '').lower()\n",
            "    exp = profile.get('experience_level', '').lower()\n",
            "    recs = []\n",
            "    if exp in ['pemula', 'beginner']:\n",
            "        recs.append('Fondasi Python dan Struktur Data Dasar')\n",
            "    if interest == 'computer vision':\n",
            "        recs.append('Convolutional Neural Networks (CNN)' if exp == 'menengah' else 'Dasar Representasi Matriks Citra')\n",
            "    elif interest == 'natural language processing':\n",
            "        recs.append('Large Language Models dan Fine-Tuning' if exp == 'menengah' else 'Pengantar Tokenisasi Teks')\n",
            "    else:\n",
            "        recs.append('Dasar Algoritma AI dan Representasi Pengetahuan')\n",
            "    recs.append('Prinsip Responsible AI dan Mitigasi Bias')\n",
            "    return recs\n",
            "\n",
            "mhs = {'user': 'Budi', 'interest': 'computer vision', 'experience_level': 'pemula'}\n",
            "print(f\"Rekomendasi untuk {mhs['user']}:\")\n",
            "for i, r in enumerate(recommend_study_path(mhs), 1):\n",
            "    print(f\"{i}. {r}\")\n"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "---\n",
            "### Bagian 2: Keyword-Based Text Classifier dan Kasus Salah\n",
            "Pencocokan kata kunci tanpa memahami konteks sering menghasilkan *False Positive* (alarm palsu) dan *False Negative* (ancaman nyata lolos)."
        ]
    },
    {
        "cell_type": "code",
        "metadata": {},
        "source": [
            "SPAM_KEYWORDS = {'hadiah', 'gratis', 'klaim', 'menang', 'promo', 'transfer'}\n",
            "\n",
            "def classify_text(text: str) -> str:\n",
            "    return 'SPAM' if any(k in text.lower() for k in SPAM_KEYWORDS) else 'HAM'\n",
            "\n",
            "samples = [\n",
            "    ('Buku referensi kuliah ini gratis dipinjam di perpustakaan.', 'HAM'),\n",
            "    ('Tolong segera kirimkan dana darurat ke rekening ini.', 'SPAM')\n",
            "]\n",
            "\n",
            "for text, actual in samples:\n",
            "    pred = classify_text(text)\n",
            "    status = 'BENAR' if pred == actual else 'ERROR/SALAH'\n",
            "    print(f\"Teks: '{text}'\")\n",
            "    print(f\"Label Aktual: {actual} | Prediksi: {pred} -> {status}\\n\")\n"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "---\n",
            "### Bagian 3: Exact Matching vs Variasi Bahasa Alami\n",
            "Manusia berkomunikasi dengan variasi kapitalisasi, salah ketik, dan sinonim. Aturan pencocokan string eksak sangat rapuh (*brittle*)."
        ]
    },
    {
        "cell_type": "code",
        "metadata": {},
        "source": [
            "FAQ = {'kapan jadwal uas': 'UAS dimulai pada minggu ke-16 semester.'}\n",
            "\n",
            "queries = [\n",
            "    'kapan jadwal uas',\n",
            "    'Kapan jadwal UAS?',\n",
            "    'kapan jadual uas',\n",
            "    'kapan ujian akhir semester dimulai'\n",
            "]\n",
            "\n",
            "for q in queries:\n",
            "    ans = FAQ.get(q.strip().lower())\n",
            "    status = 'Ditemukan' if ans else 'TIDAK DITEMUKAN'\n",
            "    print(f\"Kueri: '{q}' -> Hasil: {status}\")\n"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "---\n",
            "### Bagian 4: Kalkulasi Akurasi Eksplisit\n",
            "Rumus akurasi menghitung rasio prediksi benar terhadap total sampel: $\\text{Akurasi} = \\frac{\\text{Prediksi Benar}}{\\text{Total Sampel}}$."
        ]
    },
    {
        "cell_type": "code",
        "metadata": {},
        "source": [
            "y_true = [1, 0, 0, 1, 0, 0, 1, 1, 0, 1]\n",
            "y_pred = [1, 0, 1, 1, 0, 0, 0, 1, 0, 1]\n",
            "\n",
            "correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)\n",
            "total = len(y_true)\n",
            "acc = correct / total\n",
            "print(f\"Total Sampel : {total}\")\n",
            "print(f\"Prediksi Benar: {correct}\")\n",
            "print(f\"Akurasi Sistem: {acc:.2f} ({acc*100:.1f}%)\")\n"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "---\n",
            "### Bagian 5: Matriks Kontingensi Sederhana (Confusion Counts)\n",
            "Membongkar akurasi menjadi 4 kuadran: TP, FP, TN, FN untuk memahami dampak kesalahan pada data yang tidak seimbang."
        ]
    },
    {
        "cell_type": "code",
        "metadata": {},
        "source": [
            "y_actual = [0]*16 + [1]*4\n",
            "y_predicted = [0]*14 + [1]*2 + [1]*2 + [0]*2\n",
            "\n",
            "tp = sum(1 for t, p in zip(y_actual, y_predicted) if t == 1 and p == 1)\n",
            "fp = sum(1 for t, p in zip(y_actual, y_predicted) if t == 0 and p == 1)\n",
            "tn = sum(1 for t, p in zip(y_actual, y_predicted) if t == 0 and p == 0)\n",
            "fn = sum(1 for t, p in zip(y_actual, y_predicted) if t == 1 and p == 0)\n",
            "\n",
            "print(f\"TP: {tp} | FP: {fp}\")\n",
            "print(f\"FN: {fn} | TN: {tn}\")\n",
            "print(f\"Akurasi: {(tp+tn)/len(y_actual)*100:.1f}%\")\n",
            "print(f\"Penipuan lolos (False Negative): {fn} dari {y_actual.count(1)} ({fn/y_actual.count(1)*100:.1f}% gagal ditangkap)\")\n"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "---\n",
            "### Pertanyaan Eksplorasi Mandiri\n",
            "1. Pada Bagian 2, bagaimana cara Anda memodifikasi sistem agar kata 'gratis' tidak otomatis dianggap spam?\n",
            "2. Pada Bagian 5, jika biaya kerugian dari 1 transaksi penipuan (FN) adalah Rp 10.000.000, sedangkan biaya verifikasi manual FP adalah Rp 50.000, apakah akurasi 80% dapat diterima oleh bisnis perbankan?"
        ]
    }
]

# Eksekusi cell kode dan rekam output aktual
exec_scope = {}
execution_count = 1

for cell in cells:
    if cell['cell_type'] == 'code':
        code_text = ''.join(cell['source'])
        stdout_trap = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = stdout_trap
        try:
            exec(code_text, exec_scope)
            output_str = stdout_trap.getvalue()
            cell['execution_count'] = execution_count
            cell['outputs'] = [
                {
                    'name': 'stdout',
                    'output_type': 'stream',
                    'text': output_str.splitlines(keepends=True)
                }
            ]
        except Exception as e:
            cell['execution_count'] = execution_count
            cell['outputs'] = [
                {
                    'output_type': 'error',
                    'ename': type(e).__name__,
                    'evalue': str(e),
                    'traceback': [str(e)]
                }
            ]
        finally:
            sys.stdout = old_stdout
            execution_count += 1
    else:
        cell['metadata'] = {}

nb = {
    'cells': cells,
    'metadata': {
        'language_info': {
            'name': 'python',
            'version': sys.version.split()[0]
        },
        'kernelspec': {
            'display_name': 'Python 3',
            'language': 'python',
            'name': 'python3'
        }
    },
    'nbformat': 4,
    'nbformat_minor': 5
}

output_path = 'content/level-1-ai-foundations/chapter-01/notebooks/chapter_01_companion.ipynb'
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=2, ensure_ascii=False)

print(f'PASS: {output_path} generated and executed successfully.')

