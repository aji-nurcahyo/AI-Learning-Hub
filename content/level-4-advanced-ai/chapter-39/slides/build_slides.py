"""Generates slides/deck.json (37 slides) for Chapter 39: AI Safety, Security & Governance."""

import json
import os


def make_slides():
    slides = [
        {
            "kind": "cover",
            "title": "Chapter 39: AI Safety, Security & Governance",
            "body": "Level 4: Applied & Advanced AI Systems — Artificial Intelligence Learning Hub\nCreated & Developed by Aji Pamungkas Tri Nurcahyo (Institut Teknologi Sumatera)",
            "notes": "Selamat datang di Chapter 39. Materi ini membahas pilar keselamatan, keamanan teknis, keadilan algoritmis, privasi, dan tata kelola yang diperlukan sebelum sistem AI digunakan secara bertanggung jawab."
        },
        {
            "kind": "content",
            "title": "Mengapa AI Safety & Governance Penting?",
            "body": "Model berakurasi tinggi tetap bisa gagal secara berbahaya bila:\n- Terpapar data di luar distribusi latih (OOD shift) atau bias historis.\n- Dimanipulasi melalui Prompt Injection, Data Poisoning, atau ekstraksi model.\n- Beroperasi tanpa Human Oversight pada keputusan berisiko tinggi.",
            "notes": "Buka kuliah dengan membedakan performa statistik rata-rata di set uji dari keandalan dan tanggung jawab dampak di dunia nyata."
        },
        {
            "kind": "content",
            "title": "Tujuan Pembelajaran Chapter 39",
            "body": "Setelah mempelajari chapter ini, mahasiswa mampu:\n- Membedakan AI Safety, AI Security, dan AI Governance secara tegas.\n- Menyusun Risk Register berbasis NIST AI RMF 1.0 (Govern, Map, Measure, Manage).\n- Menghitung metrik Fairness (DP, EO, Equalized Odds) serta risiko privasi.\n- Menerapkan pertahanan defensif (validasi input, isolasi prompt, izin alat) & Model Card.",
            "notes": "Tekankan keempat kompetensi inti: kerangka risiko, keadilan/privasi, keamanan defensif offline, dan dokumentasi tata kelola."
        },
        {
            "kind": "split",
            "title": "Safety vs Security vs Governance",
            "body": "**AI Safety (Keselamatan)**\nMencegah atau mengurangi dampak merugikan yang tidak disengaja (kegagalan akurasi, OOD, kesalahan keputusan).\n\n**AI Security (Keamanan)**\nMelindungi aset model, data, dan alat dari ancaman aktor bermaksud jahat (injection, poisoning).\n\n---\n\n**AI Governance (Tata Kelola)**\nKebijakan organisasi, peran kepemilikan, audit, dan kontrol proses pengawasan di seluruh siklus hidup.\n\nKetiganya saling melengkapi dan **tidak boleh disamakan**.",
            "notes": "Sering kali mahasiswa mencampuradukkan safety dengan security. Tegaskan perbedaan niat ancaman (accidental harm vs adversarial threat) dan tata kelola institusi."
        },
        {
            "kind": "content",
            "title": "Pilar Trustworthy AI",
            "body": "Mengacu NIST AI RMF 1.0, sistem AI tepercaya memiliki 7 karakteristik utama:\n- **Valid & Reliable** dan **Safe** (meminimalkan bahaya).\n- **Secure & Resilient** terhadap serangan atau gangguan.\n- **Accountable & Transparent** serta **Explainable & Interpretable**.\n- **Privacy-Enhanced** dan **Fair with Harmful Bias Managed**.",
            "diagram": ["Valid & Safe", "Secure & Resilient", "Fair & Privacy", "Explainable & Accountable"],
            "notes": "Ingatkan bahwa karakteristik ini sering melibatkan trade-off teknis, misalnya antara tingkat privasi Differential Privacy dan akurasi prediksi."
        },
        {
            "kind": "content",
            "title": "Reliability (Keandalan Operasional)",
            "body": "- Konsistensi perilaku model dalam kondisi operasional yang diharapkan maupun terdefinisi.\n- Mengukur variabilitas lintas seed, kestabilan kalibrasi probabilitas, dan ketersediaan layanan.\n- Memerlukan pemisahan jelas antara kesalahan prediksi biasa vs kegagalan sistem kritis.",
            "notes": "Hubungkan kembali dengan Chapter 36 (MLOps) tentang pengujian berulang sebelum model di promosikan ke lingkungan lokal."
        },
        {
            "kind": "content",
            "title": "Robustness & Keterbatasan Uji Sintetis",
            "body": "- **Robustness:** Ketahanan performa terhadap noise sensor, data hilang (missing values), dan pergeseran distribusi.\n- `RandomForest` dalam demo Ch 39 mendeteksi input OOD (`z > 4.0`) atau `conf < 0.70` untuk memicu **Human Escalation**.\n- **Batasan Penting:** Lolos pengujian ketahanan pada data sintetis **tidak otomatis menjamin** ketahanan di lingkungan produksi nyata.",
            "notes": "Tekankan kejujuran ilmiah: uji stres sintetis adalah minimum sanity check, bukan bukti ketahanan kondisi dunia nyata."
        },
        {
            "kind": "content",
            "title": "Transparency (Transparansi)",
            "body": "- Kejujuran keterbukaan mengenai batas kemampuan, sumber data, metrik kegagalan, dan peran otomasi.\n- Pengguna berhak mengetahui ketika mereka berinteraksi dengan sistem AI.\n- Transparansi tidak selalu berarti membuka bobot proprietary atau kode rahasia ke publik.",
            "notes": "Bedakan transparansi kepada pengguna akhir (notice & caveats) dari transparansi audit internal (artefak kode dan log)."
        },
        {
            "kind": "content",
            "title": "Accountability & Human Oversight",
            "body": "- Algoritma tidak memikul tanggung jawab hukum atau etika; manusia dan institusi yang bertanggung jawab.\n- **Model Ownership & Data Stewardship** menetapkan siapa pemilik risiko tiap komponen.\n- **Human-in-the-Loop (HITL)** menahan keputusan berisiko atau keyakinan rendah agar ditinjau manusia.",
            "notes": "Tekankan bahwa human oversight yang efektif menuntut operator paham batas model, bukan sekadar rubber-stamping."
        },
        {
            "kind": "content",
            "title": "Siklus Manajemen Risiko AI",
            "body": "Manajemen risiko bersifat berkelanjutan di seluruh siklus hidup model:\n1. Identifikasi sumber risiko (data, model, keamanan, manusia).\n2. Penilaian **Likelihood** (1–5) dan **Severity** (1–5).\n3. Prioritas & kontrol mitigasi.\n4. Pelacakan **Residual Risk** & pemantauan kontinu.",
            "diagram": ["1. Identify", "2. Assess (L x S)", "3. Mitigate", "4. Monitor Residual"],
            "notes": "Tegaskan bahwa kontrol mitigasi menurunkan risiko menjadi Residual Risk, tetapi tidak pernah membuat risiko menjadi nol mutlak."
        },
        {
            "kind": "content",
            "title": "NIST AI Risk Management Framework (RMF 1.0)",
            "body": "Diterbitkan NIST (AI 100-1, Jan 2023) sebagai panduan sukarela empat fungsi:\n- **Govern:** Budaya risiko, kebijakan, peran akuntabilitas organisasi.\n- **Map:** Memetakan konteks penggunaan, pemangku kepentingan, dan potensi bahaya.\n- **Measure:** Menguji metrik keadilan, ketahanan, keamanan secara empiris.\n- **Manage:** Menentukan prioritas respons, mitigasi, dan pemantauan residual risk.",
            "notes": "Pastikan mahasiswa paham perbedaan statusnya: NIST AI RMF 1.0 adalah voluntary guidance framework, bukan regulasi mengikat atau sertifikasi ISO."
        },
        {
            "kind": "content",
            "title": "Identifikasi Risiko & Matriks Skor 5x5",
            "body": "Setiap bahaya dikuantifikasi lewat:\n- **Inherent Risk Score** $= \\text{Likelihood} \\times \\text{Severity}$ (skala 1–25).\n- **Priority:** `CRITICAL` ($\\ge 15$), `HIGH` ($9–14$), `MEDIUM` ($4–8$), `LOW` ($<4$).\n- Pada contoh kerja `risk_scoring.py`, 5 skenario risiko turun dari total poin inheren ke *residual risk* setelah kontrol defensif diterapkan.",
            "tex": "\\text{Risk Score} = \\text{Likelihood} \\times \\text{Severity} \\in [1, 25]",
            "notes": "Beri peringatan eksplisit agar contoh skor kerja pendidikan tidak diklaim sebagai hasil audit organisasi nyata."
        },
        {
            "kind": "code",
            "title": "Struktur AI Risk Register (Python)",
            "body": "Representasi terstruktur item Risk Register memetakan bahaya ke NIST RMF dan pemilik risiko.",
            "code": "@dataclass\nclass AIRiskItem:\n    risk_id: str\n    title: str\n    nist_function: str\n    likelihood: int\n    severity: int\n    mitigation: str\n    owner: str\n    residual_likelihood: int\n    residual_severity: int",
            "notes": "Tunjukkan pentingnya kolom `owner` dan `residual_*` sehingga setiap risiko punya penanggung jawab dan evaluasi ulang pasca kontrol."
        },
        {
            "kind": "content",
            "title": "Sumber Bias pada Dataset",
            "body": "Bias dapat menyusup jauh sebelum pelatihan model dimulai:\n- **Sampling & Representation Bias:** Subkelompok terwakili jauh lebih sedikit (`25%` vs `75%`).\n- **Measurement Bias:** Fitur proxy sistematis bergeser antar kelompok.\n- **Historical & Label Bias:** Pelabelan masa lalu mencerminkan ketidaksetaraan historis.\n- **Distribution Shift:** Populasi uji berubah dari populasi latih.",
            "notes": "Ingatkan mahasiswa: menghapus kolom grup sensitif (fairness through unawareness) tidak menghilangkan bias jika fitur proxy tetap ada."
        },
        {
            "kind": "split",
            "title": "Keadilan Algoritmis (Fairness) & Batasannya",
            "body": "**Apa itu Algorithmic Fairness?**\nMengevaluasi apakah kinerja dan laju keputusan model adil antar subkelompok tanpa diskriminasi merugikan.\n\n---\n\n**Etika Uji Sintetis:**\nDemonstrasi pada `Group A` dan `Group B` sintetis berguna untuk memverifikasi rumus matematika saja—**dilarang** menarik kesimpulan mengenai kelompok demografis dunia nyata.",
            "notes": "Sampaikan aturan akademik ketat ini agar mahasiswa tidak menyalahgunakan dataset mainan untuk klaim sosiologis nyata."
        },
        {
            "kind": "content",
            "title": "Metrik Fairness & Trade-off Matematika",
            "body": "Tiga metrik evaluasi utama (`fairness_metrics.py`):\n- **Demographic Parity Diff:** $|P(\\hat{Y}=1|A=0) - P(\\hat{Y}=1|A=1)|$\n- **Equal Opportunity Diff:** $|\\text{TPR}_0 - \\text{TPR}_1|$\n- **Equalized Odds:** $\\max(|\\text{TPR}_0 - \\text{TPR}_1|, |\\text{FPR}_0 - \\text{FPR}_1|)$\n\nPada contoh manual `N=20`: `EO Diff = 0.0000`, namun `DP Diff = 0.2000` karena *base rate* berbeda (`60%` vs `30%`).",
            "tex": "\\text{EO Diff} = |P(\\hat{Y}=1|Y=1, A=0) - P(\\hat{Y}=1|Y=1, A=1)|",
            "notes": "Jelaskan teorema ketidakmungkinan: kecuali base rate sama atau prediksi sempurna, beberapa definisi fairness secara matematis saling bertentangan."
        },
        {
            "kind": "content",
            "title": "Privasi & Pelindungan Data (GDPR & UU PDP)",
            "body": "- Membedakan **Personal Data** umum vs **Sensitive/Specific Data** (biometrik, kesehatan).\n- **EU GDPR (2016/679)** dan **Indonesia UU PDP No. 27 Tahun 2022** mewajibkan dasar pemrosesan sah, transparansi tujuan, dan hak subjek data.\n- **Risiko Re-Identifikasi:** Sekadar menghapus kolom nama tidak cukup jika kombinasi *quasi-identifier* (`zip3 + tahun lahir + prodi`) tetap unik!",
            "notes": "Pastikan konteks yurisdiksi selalu disebutkan (UU PDP No. 27/2022 untuk Indonesia; GDPR untuk Uni Eropa) tanpa mengarang pasal."
        },
        {
            "kind": "content",
            "title": "Data Minimization & Differential Privacy",
            "body": "- **Data Minimization:** Kumpulkan hanya fitur yang benar-benar relevan; terapkan *purpose limitation* & *data retention*.\n- **De-Identification & Generalisasi:** Ubah tanggal lahir spesifik menjadi rentang 5 tahun (*k-anonymity*).\n- **Differential Privacy Concept:** Tambahkan noise terkalibrasi Laplace ($\\text{scale} = \\Delta f / \\epsilon$) pada statistik keluaran—$\\epsilon$ kecil meningkatkan privasi tetapi menurunkan presisi.",
            "tex": "\\tilde{\\mu} = \\mu + \\text{Laplace}\\!\\left(0, \\frac{\\Delta f}{\\epsilon}\\right)",
            "notes": "Tekankan bahwa anonimisasi bukanlah jaminan privasi mutlak dan selalu ada kompromi antara anggaran privasi epsilon dan kegunaan analitik."
        },
        {
            "kind": "content",
            "title": "AI Threat Modeling",
            "body": "Pendekatan sistematis mengidentifikasi apa yang perlu dilindungi:\n- **Assets:** Dataset pelatihan, bobot model, API/endpoint, alat agen, log audit.\n- **Trust Boundaries:** Batas antara zona tepercaya dan input eksternal tak tepercaya.\n- **Attack Surfaces:** Input API tabular, korpus dokumen RAG, keluaran alat eksternal.\n- Semua praktik keamanan di modul ini bersifat **100% defensif & offline**.",
            "diagram": ["Untrusted Input", "Trust Boundary", "Schema Guard", "Trusted AI Core"],
            "notes": "Tegaskan larangan membuat exploit kit atau menyerang layanan nyata; fokus pada pemodelan ancaman untuk membangun kontrol berlapis."
        },
        {
            "kind": "content",
            "title": "Ancaman Model: Poisoning, Adversarial & Extraction",
            "body": "- **Data Poisoning:** Menyuntikkan label/sampel beracun ke dataset latih (Chapter 39 fokus pada verifikasi provenance).\n- **Adversarial Examples & Input Manipulation:** Perturbasi terarah guna mengecoh klasifikasi.\n- **Model Extraction & Inversion:** Meniru model lewat kueri berulang atau merekonstruksi fitur sensitif pelatihan.",
            "notes": "Jelaskan bahwa kontrol defensif mencakup rate limiting, logging kueri mencurigakan, validasi integritas data, dan penolakan input anomali."
        },
        {
            "kind": "content",
            "title": "Prompt Injection: Direct vs Indirect",
            "body": "- **Direct Prompt Injection:** Pengguna langsung menulis instruksi untuk membajak aturan sistem.\n- **Indirect Prompt Injection:** Instruksi berbahaya tersembunyi di dalam **dokumen RAG** atau **output alat web** eksternal!\n- **Aturan Emas:** Teks tak tepercaya harus diperlakukan murni sebagai **data inert**, **bukan** instruksi yang boleh dieksekusi.",
            "notes": "Hubungkan langsung dengan materi RAG (Chapter 33) dan AI Agents (Chapter 34) agar tampak integrasi sistemnya."
        },
        {
            "kind": "code",
            "title": "Isolasi Konten RAG Tak Tepercaya",
            "body": "Strategi defensif memisahkan instruksi sistem dari teks korpus melalui pemindaian pola dan pembatas XML.",
            "code": "def inspect_and_isolate_retrieved_doc(doc_id: str, raw_text: str):\n    for line in raw_text.splitlines():\n        if any(pat.search(line) for pat in INJECTION_PATTERNS):\n            sanitized.append(\"[FILTERED_SUSPICIOUS_DIRECTIVE]\")\n    return f\"<untrusted_document id='{doc_id}'>\\n\" + txt + \"\\n</untrusted_document>\"",
            "notes": "Peragakan bagaimana `executed_directives` dijaga mutlak bernilai 0 pada `prompt_injection_defense_demo.py`."
        },
        {
            "kind": "content",
            "title": "Keamanan AI Agent & Excessive Agency",
            "body": "Risiko terbesar AI Agent adalah **Excessive Agency**: agen memiliki akses alat tulis/hapus tanpa pembatasan.\nKontrol wajib:\n- **Tool Allowlisting:** Tolak nama alat di luar daftar resmi.\n- **Input & Output Schema Validation:** Cegah injeksi tipe dan *path traversal* (`../etc/passwd`).\n- **Audit Logging:** Rekam checksum SHA-256 setiap permintaan aksi.",
            "notes": "Ingatkan mahasiswa insiden dunia nyata sering terjadi ketika model diberi hak akses tulis terlalu luas."
        },
        {
            "kind": "content",
            "title": "Tool Permissions & Least Privilege",
            "body": "- Terapkan **Role-Based Access Control (RBAC)**: peran `viewer` hanya diizinkan memanggil alat `READ_ONLY`, tidak boleh mengeksekusi `export_report` atau alat mutasi.\n- Pisahkan risiko `READ_ONLY` vs `STATE_CHANGING` (`update_threshold`).\n- Prinsip **Least Privilege**: tiap komponen hanya mendapat izin minimum yang diperlukan.",
            "notes": "Rujuk output skrip `tool_permission_checks.py` yang menguji 5 skenario lintas peran."
        },
        {
            "kind": "content",
            "title": "Human Approval Gate untuk Aksi Berisiko",
            "body": "- Seluruh operasi `STATE_CHANGING` atau berdampak eksternal wajib tertahan pada **Human Approval Gate** hingga operator terotorisasi menyetujuinya secara eksplisit (`human_approved=True`).\n- Apabila ditolak, sistem harus mengembalikan status aman tanpa mutasi parsial.",
            "diagram": ["Agent Proposes Tool", "RBAC + Schema Check", "STATE_CHANGING?", "Human Approval Gate", "Execute + Audit Log"],
            "notes": "Tegaskan kembali aturan keamanan: tidak ada eksekusi aksi kritikal otomatis tanpa izin manusia."
        },
        {
            "kind": "content",
            "title": "Defense in Depth (Pertahanan Berlapis)",
            "body": "Tidak ada satu filter pun yang sempurna 100%; oleh karena itu terapkan kontrol berlapis:\n1. **Lapisan Input:** Validasi tipe (tolak boolean tersamar/NaN/Inf) & rentang nilai.\n2. **Lapisan Konteks:** Isolasi XML dan sanitasi karakter kontrol.\n3. **Lapisan Kebijakan:** RBAC, Tool Allowlist, dan Human Approval.\n4. **Lapisan Output & Monitoring:** Redaksi PII, HTTPS allowlist, OOD fallback.",
            "notes": "Gunakan analogi keju Swiss (Swiss Cheese Model): bila satu lapisan tertembus pola baru, lapisan berikutnya menahan bahaya."
        },
        {
            "kind": "content",
            "title": "Explainability & Interpretability (Link Ch 19)",
            "body": "- Mengacu Chapter 19: **Interpretability** adalah kejelasan mekanisme internal model (misalnya koefisien linear); **Explainability** mencakup teknik penjelas lokal/global (seperti *permutation feature importance*).\n- Penjelasan yang baik membantu audit bias fitur, deteksi kebocoran data, dan keputusan eskalasi operator.",
            "notes": "Jangan mengulang teori SHAP/LIME panjang lebar; fokus pada kegunaan explainability sebagai alat audit tata kelola."
        },
        {
            "kind": "code",
            "title": "Struktur Model Card Terstandar",
            "body": "Mengadaptasi Mitchell et al. (2019), `model_card_generator.py` mendokumentasikan batas penggunaan secara jujur.",
            "code": "card = {\n  \"model_details\": {\"name\": \"SyntheticScreeningBaseline\", \"version\": \"1.0.0\"},\n  \"intended_use\": {\"primary_domain\": \"Academic fairness diagnostics\"},\n  \"out_of_scope_and_prohibited\": [\"Real scholarship or credit decisions\"],\n  \"ethical_and_safety_considerations\": {\"regulatory_status\": \"Educational only\"}\n}",
            "notes": "Soroti bagian `out_of_scope_and_prohibited` yang sering dilupakan pengembang teknis padahal krusial mencegah penyalahgunaan."
        },
        {
            "kind": "content",
            "title": "Datasheets for Datasets & Audit Dokumentasi",
            "body": "- **Datasheets for Datasets (Gebru et al., 2021):** Mendokumentasikan motivasi pembuatan, komposisi, proses koleksi, pra-pemrosesan, pemeliharaan, dan lisensi.\n- Mempermudah pelacakan *data provenance* dan audit kelayakan privasi sebelum dataset dibagikan antar tim.",
            "notes": "Tunjukkan bagaimana dokumentasi dataset melengkapi Model Card untuk transparansi end-to-end."
        },
        {
            "kind": "content",
            "title": "AI Governance Lifecycle & Decommissioning",
            "body": "- Tata kelola mencakup penunjukan **Model Owner**, **Change Management**, pemantauan metrik, serta prosedur **Retirement & Decommissioning** saat model kedaluwarsa.\n- **Penegasan Penting:** Daftar periksa tata kelola proyek akademik (`governance_checklist.py`) **bukan** merupakan sertifikasi kepatuhan regulasi formal.",
            "notes": "Mahasiswa perlu sadar bahwa proyek kampus dan kepatuhan regulasi organisasi punya syarat audit formal yang sangat berbeda."
        },
        {
            "kind": "split",
            "title": "Standar & Kerangka Internasional Terverifikasi",
            "body": "**Panduan Sukarela vs Standar Manajemen:**\n- **NIST AI RMF 1.0 (2023):** Panduan sukarela risiko AI (tanpa sertifikasi).\n- **ISO/IEC 42001:2023:** Standar Sistem Manajemen AI (AIMS) yang **dapat diaudit & disertifikasi**.\n- **ISO/IEC 23894:2023:** Panduan proses manajemen risiko AI.\n\n---\n\n**Prinsip Etika & Kewajiban Hukum:**\n- **OECD AI Principles (2019/2024)** & **UNESCO Ethics of AI (2021)**.\n- **EU GDPR (2016/679)** & **Indonesia UU PDP No. 27/2022**: Kewajiban hukum perlindungan data pribadi.",
            "notes": "Tekankan pembedaan tegas antara voluntary guidance, auditable management standard, dan statutory legal obligations."
        },
        {
            "kind": "content",
            "title": "AI Incident Management & Eskalasi",
            "body": "Ketika penyimpangan terdeteksi di lingkungan aktif:\n1. **Deteksi & Isolasi:** Picu alert drift/keamanan atau *fallback* ke manusia.\n2. **Klasifikasi Keparahan & Pelaporan:** Catat kronologi dengan log tak terbantahkan.\n3. **Rollback & Perbaikan:** Kembalikan versi stabil yang disetujui manusia (Chapter 36) serta perbarui Risk Register.",
            "notes": "Hubungkan kembali ke skrip `model_rollback_simulation.py` pada Chapter 36."
        },
        {
            "kind": "case",
            "title": "Studi Kasus: RAG QA vs Screening Beasiswa",
            "body": "- **RAG Question Answering:** Risiko utama = *indirect prompt injection* & halusinasi kutipan; mitigasi = isolasi delimiter XML, sitasi `[Doc: ID]`, dan ACL dokumen.\n- **Screening Beasiswa (Ilustratif):** Risiko utama = bias historis & *equal opportunity gap*; mitigasi = audit subkelompok wajib & penolakan otomasi penuh tanpa tinjauan panel manusia.",
            "notes": "Bandingkan profil bahaya kedua kasus untuk melatih kemampuan mahasiswa menyusun analisis dampak spesifik konteks."
        },
        {
            "kind": "code",
            "title": "Demonstrasi Responsible AI Audit (Python)",
            "body": "Suite `responsible_ai_audit.py` memverifikasi 5 kontrol utama secara terotomatisasi.",
            "code": "results = run_responsible_ai_audit()\n# risk_register_complete      : PASS\n# input_validation_blocks_nan : PASS\n# prompt_injection_isolated   : PASS\n# human_approval_gate_enforced: PASS\n# fairness_audit_operational  : PASS",
            "notes": "Tunjukkan bagaimana pengujian keselamatan dan keamanan diintegrasikan layaknya unit test perangkat lunak biasa."
        },
        {
            "kind": "content",
            "title": "Diskusi: Kapan Sistem AI Harus Ditolak Rilis?",
            "body": "Pertimbangkan skenario diskusi kelas:\n- Bagaimana bila akurasi keseluruhan `92%`, tetapi `Recall` kelompok rentan hanya `45%`?\n- Bagaimana bila model tidak memiliki batas penolakan *abstention* untuk input OOD di domain medis?\n- Keputusan berani dalam AI Engineering sering kali berarti **menunda rilis** demi keselamatan.",
            "notes": "Ajak mahasiswa berdiskusi bahwa mengatakan 'model ini belum siap' adalah sikap rekayasa yang paling profesional."
        },
        {
            "kind": "summary",
            "title": "Ringkasan Chapter 39",
            "body": "- **Safety**, **Security**, dan **Governance** memiliki peran berbeda yang saling mengunci.\n- Risiko dikelola sistematis lewat **NIST AI RMF** dan **Risk Register**, dengan *residual risk* selalu ada.\n- Keadilan memiliki *trade-off* matematis; privasi menuntut *minimization* dan kesadaran risiko re-identifikasinya.\n- Terapkan **Defense in Depth** defensif, **Human Approval Gate**, dan dokumentasi jujur **Model Card**.",
            "notes": "Tutup materi dengan mengarahkan mahasiswa mengerjakan latihan dan Mini Project audit sistem dari Chapter 21–38."
        },
        {
            "kind": "content",
            "title": "Referensi Akademik & Standar Resmi",
            "body": "1. NIST (2023). *AI Risk Management Framework (AI RMF 1.0)*. https://doi.org/10.6028/NIST.AI.100-1\n2. Mitchell, M. et al. (2019). *Model Cards for Model Reporting*. https://arxiv.org/abs/1810.03993\n3. Gebru, T. et al. (2021). *Datasheets for Datasets*. https://arxiv.org/abs/1803.09010\n4. Hardt, M. et al. (2016). *Equality of Opportunity in Supervised Learning*. https://arxiv.org/abs/1610.02413\n5. OWASP (2024). *OWASP Top 10 for LLM Applications*. https://owasp.org/www-project-top-10-for-large-language-model-applications/",
            "notes": "Seluruh referensi mengarah ke dokumen standar dan makalah resmi terverifikasi untuk pendalaman lanjutan."
        }
    ]

    assert 30 <= len(slides) <= 38
    for i, s in enumerate(slides):
        assert len(s["body"]) <= 1000, f"Slide {i} body too long: {len(s['body'])}"
        if "code" in s:
            assert len(s["code"].split("\n")) <= 18, f"Slide {i} code too long"

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "deck.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(slides, f, indent=2, ensure_ascii=False)
    print(f"Generated {len(slides)} slides at {out_path}")


if __name__ == "__main__":
    make_slides()

