# Mini Project: Analisis Kelayakan Solusi AI untuk Masalah Nyata

## Deskripsi Tugas

Sebagai calon perekayasa sistem kecerdasan buatan, keterampilan paling mendasar sebelum menulis kode atau melatih model adalah **kemampuan analisis kelayakan sistem** (*problem framing & feasibility study*). Banyak proyek AI di industri mengalami kegagalan bukan karena algoritma yang buruk, melainkan karena memaksakan penerapan AI pada masalah yang sebenarnya lebih efisien diselesaikan dengan metode non-AI, atau karena mengabaikan risiko data dan etika sejak awal.

Pada mini project ini, Anda diminta memilih satu studi kasus permasalahan nyata di sekitar Anda (misalnya di lingkungan kampus, layanan publik lokal, sistem transportasi, pertanian, atau UMKM), lalu menyusun **Laporan Analisis Kelayakan Solusi AI**.

> **Catatan Penting Pelaksanaan:**
> Pada Chapter 01, tugas ini **TIDAK mewajibkan Anda melatih model Machine Learning atau menulis kode aplikasi penuh**. Fokus utama tugas adalah ketajaman perumusan masalah, penalaran pemilihan pendekatan, desain arsitektur data/evaluasi, serta analisis risiko sistem cerdas secara mendalam dan terukur.

---

## Sembilan Deliverables Laporan

Laporan Anda wajib memuat sembilan bab pembahasan berikut:

1. **Deskripsi Masalah**: Jelaskan persoalan nyata yang ingin diatasi, mengapa persoalan tersebut penting, dan apa dampak negatif jika masalah dibiarkan.
2. **Pengguna dan Konteks Operasional**: Siapa yang akan menggunakan sistem ini, di mana sistem dijalankan (web, smartphone, edge computing/perangkat tertanam), dan apa kendala lingkungan operasinya.
3. **Solusi Non-AI sebagai Pembanding (Baseline)**: Jelaskan bagaimana masalah tersebut diselesaikan saat ini tanpa AI (misalnya menggunakan aturan Excel manual, formulir kertas, atau query database sederhana). Apa kelemahan utama solusi eksisting ini?
4. **Pendekatan AI yang Dipertimbangkan**: Evaluasi kesesuaian antara Rule-Based System, Machine Learning Klasik, Deep Learning, atau Generative AI untuk masalah ini. Jelaskan mengapa pendekatan terpilih lebih rasional dibanding pendekatan lainnya.
5. **Spesifikasi Input dan Output**:
   - *Input*: Jenis data mentah yang diterima dan bagaimana representasi fitur numerik dibentuk.
   - *Output*: Format hasil prediksi atau tindakan yang dihasilkan sistem bagi pengguna.
6. **Kebutuhan Data dan Tata Kelola**:
   - Sumber asal data (*data provenance*).
   - Estimasi volume data yang dibutuhkan.
   - Cara pelabelan data (*annotation protocol*).
   - Pertimbangan privasi data dan perizinan lisensi.
7. **Protokol Evaluasi dan Metrik Keberhasilan**:
   - Metrik kuantitatif yang digunakan (misalnya Accuracy, Precision, Recall, MAE).
   - Penjelasan mengapa metrik tersebut relevan dengan konsekuensi masalah.
   - Cara membagi data untuk pengujian independen.
8. **Analisis Risiko dan Strategi Mitigasi**:
   - Apa dampak dari kesalahan False Positive vs False Negative?
   - Potensi bias sosial atau diskriminasi historis dalam data.
   - Strategi mitigasi dan perancangan *Human-in-the-Loop*.
9. **Keterbatasan Sistem**: Nyatakan secara jujur pada kondisi apa sistem ini diprediksi akan gagal (*failure modes*) dan kapan pengguna dilarang mempercayai sistem secara buta.

---

## Rubrik Penilaian (Total 100 Poin)

| Kriteria Penilaian | Bobot | Nilai Penuh (Full Marks) | Nilai Parsial (Partial Marks) | Tidak Terpenuhi (Unsatisfactory) |
|---|---|---|---|---|
| **1. Perumusan Masalah & Konteks** | **20 poin** | Masalah didefinisikan secara spesifik, berbasis fakta nyata, konteks pengguna jelas, dan perbandingan solusi non-AI (*baseline*) diuraikan secara tajam dengan kelemahannya. | Masalah dipaparkan umum atau kurang kontekstual; solusi non-AI disebutkan tetapi kurang mendalam perbandingannya. | Masalah kabur, artifisial, atau solusi non-AI tidak dibahas sama sekali. |
| **2. Ketepatan Konsep & Pemilihan Pendekatan** | **25 poin** | Membedakan secara akurat antara Rule-Based, ML, DL, dan GenAI; memberikan justifikasi rekayasa yang kokoh mengapa pendekatan AI terpilih memang dibutuhkan dan tidak berlebihan (*over-engineering*). | Memilih pendekatan AI yang masuk akal namun pembedaan paradigma kurang mendalam atau justifikasi kebutuhan AI masih lemah. | Salah mengidentifikasi konsep AI/ML, atau memaksakan penggunaan AI yang tidak masuk akal untuk masalah yang sangat sederhana. |
| **3. Desain Input, Output & Evaluasi** | **20 poin** | Input, representasi fitur, dan output didefinisikan secara konkret; metrik evaluasi dipilih secara tepat dengan mempertimbangkan trade-off False Positive vs False Negative. | Input/output didefinisikan tetapi representasi teknis kurang jelas; metrik evaluasi hanya menyebutkan "akurasi" tanpa analisis trade-off error. | Desain alur data tidak jelas dan metrik evaluasi tidak diuraikan atau tidak relevan dengan masalah. |
| **4. Analisis Risiko & Keterbatasan** | **20 poin** | Menelaah secara kritis dampak kegagalan sistem, bias data, privasi pengguna, mekanisme Human-in-the-Loop yang konkret, serta batasan kondisi operasional sistem. | Mengidentifikasi beberapa risiko umum (misal sekadar menyebut 'bisa bias') tanpa strategi mitigasi nyata atau mekanisme manusia yang jelas. | Mengabaikan analisis risiko, menganggap sistem selalu benar, atau tidak menyebutkan keterbatasan sistem. |
| **5. Kejelasan Komunikasi & Referensi** | **15 poin** | Laporan terstruktur rapi, bahasa Indonesia formal yang alami, istilah teknis bahasa Inggris digunakan secara konsisten dan tepat, menyertakan minimal 2 referensi primer yang kredibel. | Struktur laporan cukup baik; ada beberapa ketidakkonsistenan istilah teknis; referensi kurang lengkap. | Laporan berantakan, tidak terbaca, plagiarisme, atau tanpa referensi rujukan ilmiah. |

---

## Petunjuk Pengumpulan

- **Format Dokumen**: PDF atau Markdown (`.md`).
- **Panjang Laporan**: Direkomendasikan 3 hingga 5 halaman (sekitar 1.200 – 2.000 kata).
- **Integritas Akademik**: Jika Anda memanfaatkan asisten berbasis LLM untuk membantu merapikan kalimat laporan, Anda wajib mencantumkan lampiran pernyataan transparansi penggunaan AI beserta prompt utama yang digunakan, sesuai panduan Responsible AI.

