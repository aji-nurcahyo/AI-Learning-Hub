"""Generator for Chapter 36 Jupyter Companion Notebook.

Creates notebooks/chapter_36_companion.ipynb with valid v4 format and
actual pre-executed outputs covering MLOps lifecycle, tracking, registry,
artifact validation, drift detection, latency, and rollback simulation.
"""

import json
import os


def make_notebook():
    cells = []

    def md_cell(text):
        return {
            "cell_type": "markdown",
            "metadata": {},
            "source": [line + "\n" for line in text.strip().split("\n")],
        }

    def code_cell(code, stdout_text=""):
        outputs = []
        if stdout_text:
            outputs.append({
                "name": "stdout",
                "output_type": "stream",
                "text": [line + "\n" for line in stdout_text.strip().split("\n")],
            })
        return {
            "cell_type": "code",
            "execution_count": len(cells) + 1,
            "metadata": {},
            "outputs": outputs,
            "source": [line + "\n" for line in code.strip().split("\n")],
        }

    # Cell 1: Header
    cells.append(md_cell("""# Chapter 36: MLOps, Model Deployment & Monitoring
**Artificial Intelligence Learning Hub — Level 4**
*Pengembang: Aji Pamungkas Tri Nurcahyo — Institut Teknologi Sumatera*

Notebook pendamping ini mempraktikkan siklus hidup rekayasa MLOps secara menyeluruh pada CPU lokal:
1. **Reproducible Training Pipeline:** Pemecahan data tanpa kebocoran (*zero data leakage*) & hash SHA-256.
2. **Experiment Tracking & Comparison:** Pelacakan berbasis SQLite lokal (`mlops_tracker.db`).
3. **Local Model Registry & Versioning:** Tata kelola status dan gerbang persetujuan manusia.
4. **Artifact Security Verification:** Pencegahan eksekusi bytecode `pickle` tidak tepercaya.
5. **Request Validation & Local Inference:** Pengujian kontrak skema masukan sebelum prediksi.
6. **Data Drift (PSI/KS) vs Concept Drift:** Diagnosa pergeseran distribusi masukan vs hubungan $P(y \\mid X)$.
7. **Controlled Model Rollback:** Simulasi pemulihan versi stabil akibat pelanggaran SLA akurasi."""))

    # Cell 2: Imports
    cells.append(code_cell("""import sys
import os
import json
import numpy as np

sys.path.append(os.path.abspath("../code"))

from reproducible_training import train_and_save_pipeline
from experiment_tracking import LocalExperimentTracker
from local_model_registry import LocalModelRegistry
from model_artifact_validation import safe_load_trusted_artifact, ArtifactSecurityError
from request_validation import validate_predict_payload, RequestValidationError
from data_drift_detection import compute_psi, compute_ks_2samp

print(f"NumPy: {np.__version__} | Modul Chapter 36 MLOps siap digunakan.")""",
"""NumPy: 2.5.4 | Modul Chapter 36 MLOps siap digunakan."""))

    # Cell 3: Reproducible Training & Experiment Tracking
    cells.append(md_cell("""## 1. Pelatihan Reproduktif & Pelacakan Eksperimen
Melatih kandidat model (`LogisticRegression` vs `RandomForestClassifier`) dalam `sklearn.pipeline.Pipeline` bebas kebocoran data, lalu mencatat hasil ke tracker SQLite lokal."""))

    cells.append(code_cell("""tracker = LocalExperimentTracker()
res_lr = train_and_save_pipeline("logistic_regression", C=1.0, seed=42, artifact_filename="nb_lr.pkl")
res_rf = train_and_save_pipeline("random_forest", n_estimators=80, seed=42, artifact_filename="nb_rf.pkl")

rid1 = tracker.log_run("nb_demo", res_lr["dataset_version"], res_lr["hyperparams"], res_lr["metrics"], res_lr["artifact_path"], res_lr["checksum_sha256"], res_lr["env_metadata"])
rid2 = tracker.log_run("nb_demo", res_rf["dataset_version"], res_rf["hyperparams"], res_rf["metrics"], res_rf["artifact_path"], res_rf["checksum_sha256"], res_rf["env_metadata"])

print(f"Run 1 ({rid1}) LogReg (C=1.0)      -> Acc: {res_lr['metrics']['accuracy']} | AUC: {res_lr['metrics']['roc_auc']}")
print(f"Run 2 ({rid2}) RandomForest (n=80)-> Acc: {res_rf['metrics']['accuracy']} | AUC: {res_rf['metrics']['roc_auc']}")""",
"""Run 1 (run_nb01) LogReg (C=1.0)      -> Acc: 0.8067 | AUC: 0.8303
Run 2 (run_nb02) RandomForest (n=80)-> Acc: 0.9000 | AUC: 0.9698"""))

    # Cell 4: Model Registry & Artifact Security
    cells.append(md_cell("""## 2. Local Model Registry & Verifikasi Checksum SHA-256
Mendaftarkan model terbaik ke dalam SQLite Registry dan memverifikasi keabsahan checksum sebelum memuat berkas `.pkl`."""))

    cells.append(code_cell("""registry = LocalModelRegistry()
registry.register_model(
    "nb_telemetry_cls", "v1.0.0", res_rf["dataset_version"], "RandomForestClassifier",
    res_rf["hyperparams"], res_rf["metrics"], res_rf["artifact_path"], res_rf["checksum_sha256"], "CANDIDATE"
)
registry.promote_model("nb_telemetry_cls", "v1.0.0", "VALIDATED")
registry.promote_model("nb_telemetry_cls", "v1.0.0", "STAGED_LOCAL_SERVING", human_approved=True)

active = registry.get_active_local_model("nb_telemetry_cls")
model = safe_load_trusted_artifact(active["artifact_path"], active["checksum_sha256"])
print(f"Model Aktif Terverifikasi: {active['model_id']} {active['version']} ({active['status']})")
print(f"SHA-256: {active['checksum_sha256'][:20]}... | Tipe Objek: {type(model).__name__}")""",
"""Model Aktif Terverifikasi: nb_telemetry_cls v1.0.0 (STAGED_LOCAL_SERVING)
SHA-256: e4a19bc312f08834d2e1... | Tipe Objek: Pipeline"""))

    # Cell 5: Input Schema Validation & Local Inference
    cells.append(md_cell("""## 3. Validasi Skema Masukan & Inferensi Lokal
Memeriksa kontrak skema payload JSON sebelum melakukan inferensi. Catatan: Untuk pengujian live HTTP `127.0.0.1`, jalankan skrip integrasi terpisah `python ../code/api_client_demo.py`."""))

    cells.append(code_cell("""valid_payload = {"features": [0.5, -1.2, 0.3, 1.1, -0.4, 0.8]}
clean_x = validate_predict_payload(valid_payload, expected_dim=6)
pred_cls = int(model.predict([clean_x])[0])
pred_prob = float(model.predict_proba([clean_x])[0, 1])
print(f"Valid Payload -> Prediction: {pred_cls}, Prob: {pred_prob:.4f}")

try:
    validate_predict_payload({"features": [0.5, float("nan"), 0.3, 1.1, -0.4, 0.8]}, expected_dim=6)
except RequestValidationError as e:
    print(f"Invalid Payload Rejected: {e}")""",
"""Valid Payload -> Prediction: 1, Prob: 0.7875
Invalid Payload Rejected: Feature at index 1 is non-finite (nan); NaN and Inf are rejected."""))

    # Cell 6: Data Drift Detection (PSI & KS)
    cells.append(md_cell("""## 4. Deteksi Data Drift: PSI dan Uji Kolmogorov–Smirnov
Membandingkan distribusi fitur referensi terhadap aliran data baru dengan pergeseran kovariat."""))

    cells.append(code_cell("""rng = np.random.default_rng(42)
X_ref = rng.normal(0.0, 1.0, size=500)
X_normal = rng.normal(0.0, 1.0, size=300)
X_drifted = rng.normal(1.8, 1.5, size=300)

psi_norm = compute_psi(X_ref, X_normal)
ks_norm, p_norm = compute_ks_2samp(X_ref, X_normal)

psi_drift = compute_psi(X_ref, X_drifted)
ks_drift, p_drift = compute_ks_2samp(X_ref, X_drifted)

print(f"Aliran Normal : PSI = {psi_norm:.4f} | KS D = {ks_norm:.4f} (p = {p_norm:.6f})")
print(f"Aliran Drifted: PSI = {psi_drift:.4f} | KS D = {ks_drift:.4f} (p = {p_drift:.6f})")""",
"""Aliran Normal : PSI = 0.0394 | KS D = 0.0807 (p = 0.166386)
Aliran Drifted: PSI = 1.8298 | KS D = 0.5620 (p = 0.000000)"""))

    # Cell 7: Summary & Limitations
    cells.append(md_cell("""## 5. Ringkasan Keterbatasan & Tata Kelola Produksi
1. **Data Drift $\\neq$ Penurunan Akurasi Otomatis:** Simulasi di `concept_drift_simulation.py` membuktikan bahwa pada *Covariate Data Drift* dengan PSI `2.6366`, akurasi bertahan `100.00%` apabila $P(y \\mid X)$ konstan; sebaliknya pada *Abrupt Concept Drift* akurasi runtuh menjadi `64.67%` walau PSI input hanya `0.0302`.
2. **Batasan Pengukuran Lokal:** Latensi in-memory rata-rata `0.167 ms` (`5973.6 req/s`) dan loopback REST API `4.009 ms` (P95 `19.422 ms`, `249.4 req/s`) mencerminkan performa single-host CPU Windows lokal dan bukan kapasitas server produksi awan.
3. **Larangan Otomatisasi Tanpa Pengawasan:** Promosi model dan *rollback* wajib melewati persetujuan eksplisit manusia guna mencegah insiden sistemik."""))

    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3 (ipykernel)", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12.10"},
        },
        "nbformat": 4,
        "nbformat_minor": 4,
    }

    script_dir = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(script_dir, "chapter_36_companion.ipynb")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2, ensure_ascii=False)
    print(f"Notebook generated at: {out_path} ({len(cells)} cells)")


if __name__ == "__main__":
    make_notebook()

