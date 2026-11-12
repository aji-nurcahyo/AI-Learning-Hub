# Chapter 39: AI Safety, Security & Governance
**Level 4: Applied & Advanced AI Systems — Artificial Intelligence Learning Hub**  
*Created & Developed by Aji Pamungkas Tri Nurcahyo — Institut Teknologi Sumatera*

---

## PART A — FONDASI AI SAFETY, SECURITY & GOVERNANCE

Peningkatan kapabilitas model pembelajaran mesin dan *Generative AI* tidak otomatis menjamin keandalan maupun keamanan penerapannya. Sistem berakurasi tinggi tetap dapat menimbulkan kerugian serius jika menghadapi distribusi input tak terduga, ketimpangan historis pada data latih, penyalahgunaan oleh pihak bermaksud jahat, atau absennya pengawasan manusia (*human oversight*).

### 1. Pembedaan Tegas: Safety vs Security vs Governance
Ketiga disiplin ini sering tumpang tindih di diskusi umum, tetapi memiliki definisi serta tujuan rekayasa yang berbeda dan **tidak boleh disamakan**:

```text
+-----------------------+--------------------------+---------------------------+
|      AI SAFETY        |       AI SECURITY        |       AI GOVERNANCE       |
|  (Keselamatan Sistem) |    (Keamanan Sistem)     |   (Tata Kelola & Kontrol) |
+-----------------------+--------------------------+---------------------------+
| Mencegah atau         | Melindungi aset AI       | Menetapkan kebijakan,     |
| mengurangi dampak     | (data, bobot, API, alat) | peran akuntabilitas,      |
| merugikan yang TIDAK  | dari ANCAMAN &           | proses audit, persetujuan |
| DISENGAJA akibat      | PENYALAHGUNAAN aktor     | manusia, dan kepatuhan    |
| cacat/kegagalan model | bermaksud jahat          | sepanjang siklus hidup    |
+-----------------------+--------------------------+---------------------------+
```

### 2. Pilar Trustworthy AI
Mengacu kerangka **NIST AI RMF 1.0**, sistem AI tepercaya bertumpu pada tujuh karakteristik:
1. **Valid & Reliable:** Berperilaku konsisten, teruji ulang, dan terukur ketepatannya pada lingkup yang dimaksud.
2. **Safe:** Meminimalkan bahaya fisik, psikologis, finansial, maupun lingkungan termasuk lewat mekanisme *fail-safe*.
3. **Secure & Resilient:** Tahan terhadap manipulasi adversarial, injeksi, serta mampu memulihkan operasi pasca insiden.
4. **Accountable & Transparent:** Memiliki penanggung jawab manusia/institusi jelas serta keterbukaan fungsi dan batasannya.
5. **Explainable & Interpretable:** Menyediakan penafsiran faktor pendorong keputusan yang relevan bagi pengaudit dan pengguna.
6. **Privacy-Enhanced:** Meminimalkan pengumpulan data pribadi serta menjaga kerahasiaan informasi pelatihan.
7. **Fair with Harmful Bias Managed:** Mengukur dan mengendalikan ketimpangan perlakuan sistematis antar kelompok terdampak.

---

## PART B — AI RISK MANAGEMENT (NIST AI RMF 1.0)

Manajemen risiko AI memetakan secara sistematis bahaya yang mungkin muncul, mengukur probabilitas dan keparahannya, menerapkan kontrol mitigasi, lalu terus memantau **Residual Risk** (sisa risiko yang tetap ada sesudah kontrol diterapkan).

### 1. Empat Fungsi NIST AI RMF 1.0 (NIST AI 100-1)
```text
    +---------------+       +---------------+       +---------------+       +---------------+
    |   1. GOVERN   | ----> |    2. MAP     | ----> |  3. MEASURE   | ----> |   4. MANAGE   |
    | Kebijakan,    |       | Konteks,      |       | Metrik bias,  |       | Prioritas,    |
    | peran &       |       | pemangku &    |       | robustness &  |       | mitigasi &    |
    | akuntabilitas |       | sumber bahaya |       | keamanan      |       | residual risk |
    +---------------+       +---------------+       +---------------+       +---------------+
            ^                                                                       |
            +----------------------- Pemantauan Kontinu ----------------------------+
```

### 2. Matriks Risiko & Contoh Kerja Risk Register
Skor risiko dihitung sebagai hasil kali **Likelihood** ($1\dots 5$) dan **Severity** ($1\dots 5$), dengan prioritas: `CRITICAL` ($\ge 15$), `HIGH` ($9–14$), `MEDIUM` ($4–8$), dan `LOW` ($<4$). Eksekusi aktual `code/ai_risk_register.py` & `code/risk_scoring.py` menghasilkan contoh kerja berikut:

| ID | Fungsi NIST RMF | Bahaya Teridentifikasi | Skor Inheren ($L\times S$) | Kontrol Mitigasi Utama | Pemilik (*Owner*) | Skor Residual |
| :--- | :--- | :--- | :---: | :--- | :--- | :---: |
| `RSK-01` | `Map / Manage` | *Indirect Prompt Injection* dari dokumen RAG | `20` (`CRITICAL`) | Isolasi XML + pemisahan data–instruksi | *Security Lead* | `6` |
| `RSK-02` | `Measure` | Bias representasi menekan TPR subkelompok | `16` (`CRITICAL`) | Audit EO/DP + *Human Oversight* | *ML Data Steward* | `6` |
| `RSK-03` | `Govern / Manage` | Aksi alat tak berizin (*Excessive Agency*) | `15` (`CRITICAL`) | RBAC allowlist + *Human Approval Gate* | *Agent Owner* | `3` |
| `RSK-04` | `Map / Measure` | Re-identifiksi PII via *quasi-identifier* | `12` (`HIGH`) | *Data minimization* + generalisasi | *Privacy Officer* | `4` |
| `RSK-05` | `Measure / Manage` | *Concept drift* / input OOD tak terkalibrasi | `12` (`HIGH`) | Monitor PSI/KS + *human escalation* | *MLOps Engineer* | `4` |

Total poin risiko inheren turun dari **`75`** menjadi **`23`** (penurunan **`69.33%`**, divisualisasikan pada `experiments/risk_matrix_heatmap.png`).

> [!NOTE]
> **Penafian Kejujuran:** Contoh tabel skor di atas murni merupakan **skenario kerja edukatif sintetis** untuk latihan kelas, **bukan** merupakan hasil audit risiko suatu organisasi nyata.

---

## PART C — DATA QUALITY, BIAS & ALGORITHMIC FAIRNESS

Bias dapat menyusup ke sistem AI sebelum satu baris model pun dilatih:
1. **Sampling & Representation Bias:** Sebagian kelompok terlalu sedikit terwakili (`code/dataset_bias_analysis.py` menunjukkan rasio representasi sintetis `Group B / Group A = 0.346`, yakni `25.7%` vs `74.3%`).
2. **Measurement Bias:** Variabel *proxy* mencatat nilai sistematis lebih rendah untuk satu kelompok akibat alat ukur bias.
3. **Historical & Label Bias:** Label masa lalu merekam keputusan manusia yang timpang (kesenjangan laju label positif sintetis mencapai `14.64` poin persentase).
4. **Distribution Shift:** Populasi operasional bergeser menjauhi distribusi latih.

### Perhitungan Manual Metrik Fairness & Konflik Matematika (`code/fairness_metrics.py`)
Tiga metrik keadilan utama didefinisikan sebagai selisih mutlak antar kelompok $A \in \{0, 1\}$:
- **Demographic Parity Difference:** $\left|P(\hat{Y}=1 \mid A=0) - P(\hat{Y}=1 \mid A=1)\right|$
- **Equal Opportunity Difference:** $\left|\text{TPR}_0 - \text{TPR}_1\right|$ dengan $\text{TPR}_g = P(\hat{Y}=1 \mid Y=1, A=g)$
- **Equalized Odds Gap:** $\max\left(\left|\text{TPR}_0 - \text{TPR}_1\right|,\ \left|\text{FPR}_0 - \text{FPR}_1\right|\right)$

Pada contoh manual terverifikasi `N=20` (`fairness_metrics.py`):
- **Kelompok 0 (`n=10`, `6` positif):** $\text{Sel}_0 = 5/10 = 0.5000$, $\text{TPR}_0 = 4/6 = 0.6667$, $\text{FPR}_0 = 1/4 = 0.2500$.
- **Kelompok 1 (`n=10`, `3` positif):** $\text{Sel}_1 = 3/10 = 0.3000$, $\text{TPR}_1 = 2/3 = 0.6667$, $\text{FPR}_1 = 1/7 = 0.1429$.
- **Hasil Audit:** **Equal Opportunity Diff** $= |0.6667 - 0.6667| = \mathbf{0.0000}$ (terpenuhi sempurna!), sedangkan **Demographic Parity Diff** $= |0.5000 - 0.3000| = \mathbf{0.2000}$ dan **Equalized Odds Gap** $= \mathbf{0.1071}$.

```text
  [Base Rates Berbeda: 60% vs 30%]
         |
         +--> Equal Opportunity Terpenuhi (0.0000)  =/=>  Demographic Parity Gagal (0.2000)!
```

> [!IMPORTANT]
> **Teorema Ketidakmungkinan Fairness & Etika Sintetis:** Tidak ada satu pun metrik keadilan yang mutlak "paling benar" di segala situasi karena metrik-metrik tersebut secara matematis dapat saling bertentangan ketika *base rate* berbeda. Selain itu, demonstrasi pada `Group A` & `Group B` **dilarang keras** dipakai untuk menarik kesimpulan tentang kelompok demografis manusia di dunia nyata.

---

## PART D — PRIVACY & DATA PROTECTION

Dalam siklus hidup AI, perlindungan data pribadi menuntut pembedaan antara **Personal Data** umum dan **Sensitive / Specific Data** (seperti data biometrik, rekam medis, atau keuangan). Prinsip rekayasa utamanya meliputi **Data Minimization**, **Consent**, **Purpose Limitation**, **Data Retention**, dan **Access Control**.

### 1. Bahaya Re-Identifikasi & Differential Privacy
Menghapus atribut pengenal langsung (`nama`) saja **bukan jaminan mutlak** privasi (*anonymization is never an absolute guarantee*). Pada `code/privacy_data_minimization.py`, penghapusan nama dari `4` rekaman sintetis menyisakan `1` rekaman dengan kombinasi *quasi-identifier* unik (`zip3 + tahun lahir + dept`) $\implies$ risiko re-identifikasinya **`25.0%`**. Mitigasi dilakukan lewat generalisasi (*k-anonymity*) dan **Differential Privacy**, di mana noise Laplace terkontrol ditambahkan pada hasil agregasi:

$$
\tilde{\mu} = \mu + \text{Laplace}\!\left(0, \frac{\Delta f}{\epsilon}\right)
$$

Semakin kecil anggaran privasi $\epsilon$ (`2.0 -> 0.5 -> 0.1`), jaminan privasi semakin kuat namun simpangan akurasi statistik semakin besar.

### 2. Konteks Hukum Perlindungan Data (Terverifikasi Yurisdiksi)
- **EU General Data Protection Regulation (GDPR — Regulation (EU) 2016/679):** Regulasi mengikat di Uni Eropa yang mengatur dasar pemrosesan sah, *data minimization*, hak subjek data (*erasure*, transparansi), dan pembatasan keputusan terotomatisasi penuh.
- **Indonesia — UU No. 27 Tahun 2022 tentang Pelindungan Data Pribadi (UU PDP):** Undang-undang resmi di Indonesia yang membagi data pribadi umum dan spesifik, menetapkan hak subjek data, kewajiban pengendali/prosesor data, serta aturan transfer data lintas batas.

---

## PART E — AI THREAT MODELING (DEFENSIF & SINTETIS)

**Threat Modeling** memetakan **Assets** (dataset, bobot model, API, alat agen, log), **Trust Boundaries** (titik perlintasan dari input luar ke zona eksekusi tepercaya), **Attack Surfaces**, dan **Threat Actors**.

```text
  [Input Luar Tak Tepercaya] ===( Trust Boundary 1 )===> [Input Validator & Sanitizer]
                                                                    |
  [Model Weights & Log] <======( Trust Boundary 2 )====== [RBAC & Human Approval Gate]
```

Kategori ancaman AI utama:
1. **Input Manipulation & Adversarial Examples:** Memasukkan nilai patologis (`NaN`, tipe boolean tersamar, string path traversal) atau perturbasi kecil untuk menjatuhkan prediksi.
2. **Data Poisoning:** Menyisipkan sampel/label beracun ke dalam sumber pelatihan.
3. **Model Extraction & Model Inversion:** Mengkloning perilaku model melalui kueri masif atau merekonstruksi atribut pelatihan sensitif dari probabilitas keluaran.
4. **Data Exfiltration:** Membocorkan konteks sensitif keluar melalui keluaran teks atau alat agen.

---

## PART F — KEAMANAN LLM & AI AGENT (DEFENSE IN DEPTH)

Mengintegrasikan materi **Chapter 33 (RAG)** dan **Chapter 34 (AI Agents)**, bahaya paling kritis pada aplikasi LLM modern adalah **Indirect Prompt Injection** melalui dokumen korpus atau keluaran alat eksternal, serta **Excessive Agency**.

### Prinsip Pertahanan Berlapis (*Defense in Depth*)
1. **Strict Input Schema (`code/input_validation.py`):** Blokir tipe non-valid (`5/5` payload berbahaya ditolak bersih).
2. **Instruction Hierarchy & Data Isolation (`code/prompt_injection_defense_demo.py`):** Teks hasil retrieval RAG dibungkus ke dalam tag `<untrusted_document>` dan pola injeksi disaring—seluruh teks eksternal diperlakukan mutlak sebagai **data inert**, bukan perintah (`Executed Directives = 0`).
3. **Tool Allowlisting & RBAC (`code/tool_permission_checks.py`):** Peran `viewer` ditolak memanggil alat berisiko lebih tinggi; alat tak terdaftar diblokir mutlak.
4. **Human Approval Gate:** Setiap aksi `STATE_CHANGING` (`update_threshold`) wajib tertahan hingga `human_approved=True`.
5. **Output & URL Sanitization (`code/untrusted_content_handling.py`) + Audit Log SHA-256:** Menyensor pola PII sintetis dan menegakkan allowlist domain HTTPS (`arxiv.org`, `nist.gov`, `iso.org`, `oecd.org`).

---

## PART G — ROBUSTNESS & RELIABILITY

Pada `code/model_robustness_checks.py`, uji stres pada `RandomForestClassifier` memperlihatkan kerentanan model terhadap pergeseran distribusi dan pentingnya mekanisme **Human Escalation Fallback** bila keyakinan rendah (`confidence < 0.70`) atau input berada di luar distribusi latih (`|z| > 4.0`):

| Kondisi Uji Sintetis | Akurasi Uji | Rata-rata Keyakinan | Laju Eskalasi ke Manusia |
| :--- | :---: | :---: | :---: |
| *Clean In-Distribution Test* | `84.50%` | `0.813` | `22.0%` |
| *Noisy Shift* ($\sigma = 1.2$) | `69.00%` | `0.727` | `40.5%` |
| *Missing Features* (`30% NaN` + imputasi aman) | `72.50%` | `0.727` | `41.0%` |
| *Extreme OOD Stress* (`+6.0 shift`) | `50.00%` | `0.599` | **`100.0%`** (Seluruhnya ditahan!) |

> [!WARNING]
> Lolos uji ketahanan di skenario sintetis lokal **tidak otomatis membuktikan** ketangguhan model pada lingkungan operasional dunia nyata.

---

## PART H — EXPLAINABILITY & DOKUMENTASI TRANSPARAN

Melanjutkan **Chapter 19** (tanpa mengulang teori rinci SHAP/LIME), *explainability* global maupun lokal berfungsi sebagai bukti audit untuk memverifikasi bahwa model belajar fitur domain yang sah dan bukan fitur proksi bias. Dokumentasi formal mencakup dua artefak kembar:
1. **Model Cards (`Mitchell et al., 2019`, `code/model_card_generator.py` $\to$ `experiments/sample_model_card.json`):** Memaparkan detail arsitektur, *Intended Use*, daftar eksplisit **Out-of-Scope / Prohibited Use**, pemecahan metrik per subkelompok, serta keterbatasan.
2. **Datasheets for Datasets (`Gebru et al., 2021`):** Mencatat motivasi pembuatan dataset, komposisi, metode pengumpulan, pembersihan, pemeliharaan, dan lisensi.

---

## PART I — AI GOVERNANCE & INSIDEN

Tata kelola AI di tingkat proyek maupun institusi menuntut **Model Ownership**, **Data Stewardship**, **Risk Acceptance** berjenjang, **Change Management**, prosedur **Incident Reporting & Rollback**, serta protokol **Retirement & Decommissioning** saat model kedaluwarsa.

Skrip `code/governance_checklist.py` menyediakan 6 butir kesiapan tata kelola proyek akademik (`6/6 Verified`). **Penting:** Daftar periksa proyek pendidikan ini bertujuan membiasakan dokumentasi bertanggung jawab dan **bukan** merupakan sertifikasi kepatuhan hukum atau audit regulasi formal.

---

## PART J — STANDAR & KERANGKA INTERNASIONAL TERVERIFIKASI

| Standar / Kerangka | Penerbit | Sifat & Status Resmi | Cakupan Utama |
| :--- | :--- | :--- | :--- |
| **NIST AI RMF 1.0 (AI 100-1)** | NIST (Jan 2023) | **Panduan Sukarela** (tanpa sertifikasi) | Manajemen risiko siklus hidup AI melalui 4 fungsi: *Govern, Map, Measure, Manage*. |
| **ISO/IEC 42001:2023** | ISO / IEC (Des 2023) | **Standar Sistem Manajemen** (dapat diaudit & disertifikasi) | Persyaratan membangun dan memelihara *AI Management System* (AIMS) organisasi. |
| **ISO/IEC 23894:2023** | ISO / IEC (Feb 2023) | **Panduan Teknis Sukarela** (non-sertifikasi) | Panduan integrasi proses manajemen risiko khusus AI dalam organisasi. |
| **OECD AI Principles** | OECD (2019, pembaruan 2024) | **Standar Antar-Pemerintah** (panduan kebijakan) | Pertumbuhan inklusif, transparansi, keadilan, ketahanan, keamanan, dan akuntabilitas. |
| **UNESCO Ethics of AI** | UNESCO (Nov 2021) | **Rekomendasi Global** (194 negara anggota) | Etika berbasis HAM & kelestarian lingkungan (*Ethical Impact Assessment* / EIA). |
| **EU GDPR & RI UU PDP No. 27/2022** | Pemerintah / Legislatif | **Kewajiban Hukum Mengikat** (*Statutory Law*) | Pelindungan data pribadi, hak subjek data, dasar hukum pemrosesan, & sanksi hukum. |

---

## PART K — STUDI KASUS RESPONSIBLE AI, LATIHAN & MINI PROJECT

### 1. Tujuh Studi Kasus Analisis Dampak Lintas Domain

1. **Scholarship Screening (Ilustratif):** *Intended use:* membantu penyortiran administratif awal; *Stakeholders:* pelamar, komite beasiswa; *Harms:* penolakan tak adil; *Data/Model risks:* bias historis & *EO gap*; *Security:* manipulasi berkas PDF; *Mitigasi & Oversight:* audit fairness + keputusan final mutlak oleh panel manusia; *Batas:* tidak boleh terotomasi penuh.
2. **Predictive Maintenance Industri:** *Intended use:* peringatan dini keausan motor; *Stakeholders:* teknisi, manajer pabrik; *Harms:* henti mesin mendadak (*FN*) atau biaya cek palsu (*FP*); *Data/Model risks:* sensor kotor & *concept drift*; *Security:* injeksi `NaN`/spoofing sensor; *Mitigasi & Oversight:* monitor PSI/KS + inspeksi teknisi sebelum shutdown; *Batas:* simulasi laboratorium berbeda dari mesin nyata.
3. **Computer Vision Monitoring:** *Intended use:* deteksi APD helm di area konstruksi; *Stakeholders:* pekerja, tim K3; *Harms:* alarm keselamatan terlewat atau pengawasan privasi berlebihan; *Data/Model risks:* bias pencahayaan & oklusi; *Security:* akses tidak sah ke umpan CCTV; *Mitigasi & Oversight:* pemrosesan di edge tanpa menyimpan wajah (*data minimization*) + petugas K3; *Batas:* tidak menjamin 100% di cuaca ekstrem.
4. **AI-Based Learning Assistant:** *Intended use:* membantu mahasiswa memahami konsep modul terbuka; *Stakeholders:* mahasiswa, dosen; *Harms:* ketergantungan & jawaban keliru (*false authority*); *Data/Model risks:* topik di luar korpus; *Security:* *direct prompt injection* untuk memperoleh kunci ujian; *Mitigasi & Oversight:* jawaban tergrounded sumber + dosen sebagai penilai sah; *Batas:* asisten tidak menggantikan bimbingan akademik.
5. **RAG Question Answering:** *Intended use:* Q&A dokumen kebijakan kampus; *Stakeholders:* civitas akademika, administrator dokumen; *Harms:* kebocoran dokumen internal atau salah sitasi; *Data/Model risks:* potongan konteks terpotong (*chunking*); *Security:* **Indirect Prompt Injection** di korpus; *Mitigasi & Oversight:* ACL per dokumen, `<untrusted_document>` isolation, serta tombol penolakan (*abstention*); *Batas:* tidak bebas halusinasi mutlak.
6. **Tool-Using AI Agent:** *Intended use:* asisten otomasi laporan harian lokal; *Stakeholders:* analis, tim keamanan TI; *Harms:* penghapusan/mutasi basis data; *Data/Model risks:* loop pemanggilan berlebih; *Security:* **Excessive Agency** & injeksi argumen; *Mitigasi & Oversight:* RBAC allowlist + JSON schema check + **Human Approval Gate** wajib; *Batas:* membutuhkan operator siaga.
7. **Edge AI Monitoring:** *Intended use:* deteksi anomali getaran lokal di perangkat hemat daya; *Stakeholders:* operator lapangan, perancang IoT; *Harms:* penurunan presisi pasca kuantisasi INT8 (*false sense of security*); *Data/Model risks:* *quantization error* pada *outlier*; *Security:* modifikasi bobot firmware fisik; *Mitigasi & Oversight:* audit kesetiaan logit & fallback *threshold* aman; *Batas:* simulasi CPU PC bukan verifikasi silikon fisik.

### 2. Ringkasan, Latihan, Mini Project & Referensi
- Ringkasan terpadu [`responsible_ai_audit.py`](file:///D:/Workstation/Repo%20My%20Corses/Kecerdasan%20Buatan/content/level-4-advanced-ai/chapter-39/code/responsible_ai_audit.py) memverifikasi **5/5 kontrol lulus** bersama total 15 skrip di [`practice.md`](file:///D:/Workstation/Repo%20My%20Corses/Kecerdasan%20Buatan/content/level-4-advanced-ai/chapter-39/practice.md) dan [`chapter_39_companion.ipynb`](file:///D:/Workstation/Repo%20My%20Corses/Kecerdasan%20Buatan/content/level-4-advanced-ai/chapter-39/notebooks/chapter_39_companion.ipynb).
- **Latihan Bertingkat (Dasar, Menengah, Lanjut)** lengkap dengan kunci pembahasan serta **Mini Project "Responsible AI Risk Assessment & Security Audit"** (14 deliverables & **Rubrik 100 Poin**: `15 + 25 + 20 + 25 + 15`) tersedia pada [`practice.md`](file:///D:/Workstation/Repo%20My%20Corses/Kecerdasan%20Buatan/content/level-4-advanced-ai/chapter-39/practice.md).

### Referensi Resmi & Akademik Terverifikasi
1. NIST (2023). *Artificial Intelligence Risk Management Framework (AI RMF 1.0), NIST AI 100-1*. https://doi.org/10.6028/NIST.AI.100-1
2. Mitchell, M., et al. (2019). *Model Cards for Model Reporting*. ACM FAT* 2019. https://arxiv.org/abs/1810.03993
3. Gebru, T., et al. (2021). *Datasheets for Datasets*. Communications of the ACM. https://arxiv.org/abs/1803.09010
4. Hardt, M., Price, E., & Srebro, N. (2016). *Equality of Opportunity in Supervised Learning*. NeurIPS 2016. https://arxiv.org/abs/1610.02413
5. OWASP Foundation (2024). *OWASP Top 10 for Large Language Model Applications*. https://owasp.org/www-project-top-10-for-large-language-model-applications/
6. OECD (2024). *Recommendation of the Council on Artificial Intelligence (OECD AI Principles)*. https://legalinstruments.oecd.org/en/instruments/OECD-LEGAL-0449
7. UNESCO (2021). *Recommendation on the Ethics of Artificial Intelligence*. https://unesdoc.unesco.org/ark:/48223/pf0000381137

