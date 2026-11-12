# Chapter 39: AI Safety, Security & Governance — Code Catalog

Koleksi 15 script Python mandiri, deterministik, offline, dan murni **defensif** menggunakan NumPy, Scikit-learn, Matplotlib, dan Python Standard Library (tanpa kredensial asli, tanpa data pribadi nyata, dan tanpa teknik penyerangan sistem nyata).

## Daftar Script

1. `ai_risk_register.py`: Pembuatan dan pencatatan *AI Risk Register* contoh terpetakan ke 4 fungsi **NIST AI RMF 1.0** (*Govern, Map, Measure, Manage*) ke `experiments/risk_register_sample.json`.
2. `risk_scoring.py`: Perhitungan skor risiko $5\times 5$ (Likelihood $\times$ Severity), *residual risk* pasca mitigasi, serta heatmap `experiments/risk_matrix_heatmap.png`.
3. `dataset_bias_analysis.py`: Analisis *representation bias*, *measurement proxy noise*, dan kesenjangan label pada dataset sintetis aman (`Group A` vs `Group B`).
4. `fairness_metrics.py`: Perhitungan manual & verifikasi kode **Demographic Parity Difference**, **Equal Opportunity Difference**, dan **Equalized Odds**, serta bukti ketidakmungkinan pemenuhan serentak (*impossibility trade-off*) ketika *base rate* berbeda.
5. `group_performance_analysis.py`: Audit akurasi & *TPR/FPR* subkelompok pada `LogisticRegression` + grafik trade-off ambang keputusan (`experiments/fairness_metrics_comparison.png`).
6. `privacy_data_minimization.py`: Prinsip *Data Minimization*, pembuktian risiko re-identifikasi lewat *quasi-identifier* tunggal, dan mekanisme dasar *Laplace Differential Privacy*.
7. `input_validation.py`: Penjagaan batas (*boundary guard*) menolak field hilang, boolean tersembunyi, `NaN`/`Inf`, angka di luar jangkauan, dan string patologis (`../etc/passwd`).
8. `prompt_injection_defense_demo.py`: Isolasi konten eksternal tak tepercaya dalam delimiter XML `<untrusted_document>`, perlakuan instruksi berbahaya murni sebagai **data inert**, dan jaminan 0 eksekusi perintah.
9. `tool_permission_checks.py`: Penegakan *Role-Based Access Control* (RBAC), *tool allowlisting*, dan *Human Approval Gate* wajib untuk alat yang mengubah status.
10. `untrusted_content_handling.py`: Sanitasi teks tak tepercaya (karakter kontrol, redaksi PII sintetis, serta *HTTPS domain allowlist*).
11. `model_robustness_checks.py`: Uji degradasi `RandomForestClassifier` terhadap noise geser, `30%` data hilang, serta masukan OOD ekstrem dengan pemicu **Human Escalation Fallback**.
12. `model_card_generator.py`: Generator struktur **Model Card** standar (`experiments/sample_model_card.json`) mencakup *Intended Use*, *Prohibited Use*, metrik subkelompok, dan batasan.
13. `responsible_ai_audit.py`: Suite pengujian otomatis lintas validasi masukan, isolasi prompt, izin alat, risiko, dan keadilan.
14. `governance_checklist.py`: Daftar periksa tata kelola proyek AI akademik beserta penafian jujur non-sertifikasi regulasi.
15. `ai_safety_evaluation.py`: Ringkasan terpadu seluruh metrik keselamatan, keamanan, keadilan, dan tata kelola Chapter 39.

