"""Generates pre-executed Chapter 40 Capstone companion notebook (nbformat v4)."""

import json
import os
import subprocess
import sys


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CODE_DIR = os.path.join(BASE_DIR, "code")


def run_snippet(script_name: str) -> str:
    res = subprocess.run([sys.executable, os.path.join(CODE_DIR, script_name)], cwd=CODE_DIR, capture_output=True, text=True, check=True)
    return (res.stdout or "") + (res.stderr or "")


def md_cell(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(keepends=True)}


def code_cell(source: str, output_text: str, exec_count: int) -> dict:
    return {
        "cell_type": "code",
        "execution_count": exec_count,
        "metadata": {},
        "outputs": [{"name": "stdout", "output_type": "stream", "text": output_text.splitlines(keepends=True)}],
        "source": source.splitlines(keepends=True),
    }


def main():
    steps = [
        ("Stage 1–2: Dataset Provenance, EDA & Zero-Leakage Splits",
         "Menghasilkan telemetri motor sintetis `1.500` sampel $\\times$ `6` fitur (`seed=42`), menyimpan `sensor_data_dictionary.json`, lalu melakukan pemisahan *Train / Validation / Test* berstrata sebelum `StandardScaler` di-*fit* murni pada training set.",
         "import dataset_generator, preprocessing\ndataset_generator.main()\npreprocessing.main()",
         run_snippet("dataset_generator.py") + "\n" + run_snippet("preprocessing.py")),
        ("Stage 3–5: Baseline vs Candidate Models & Resource Profiling",
         "Membandingkan `DummyClassifier`, `LogisticRegression`, `RandomForest`, dan `MLPClassifier` pada *Validation Set* berikut profil ukuran artefak (`.pkl`) dan latensi CPU (Mean / P95).",
         "import baseline_model, candidate_models\nbaseline_model.main()\ncandidate_models.main()",
         run_snippet("baseline_model.py") + "\n" + run_snippet("candidate_models.py")),
        ("Stage 6: Final Held-Out Test Evaluation, Explainability & Error Analysis",
         "Memverifikasi SHA-256 artefak pemenang seleksi validasi, lalu mengevaluasi satu kali pada held-out test set (`N=300`) lengkap dengan *Permutation Feature Importance* dan sampel kasus keliru.",
         "import evaluation_and_error_analysis\nevaluation_and_error_analysis.main()",
         run_snippet("evaluation_and_error_analysis.py")),
        ("Stage 7: Validated Local Inference & Post-Deployment Drift Simulation",
         "Menguji penjagaan skema input (`NaN`/boolean/out-of-bounds) serta menyimulasikan pemantauan aliran telemetri nominal vs tergeser (`+9.5C` & `+1.1 mm/s` getaran) menggunakan **PSI** dan **Kolmogorov-Smirnov**.",
         "import local_inference, monitoring_and_drift\nlocal_inference.main()\nmonitoring_and_drift.main()",
         run_snippet("local_inference.py") + "\n" + run_snippet("monitoring_and_drift.py")),
        ("Stage 8: Risk Register, Model Card, Unified Pipeline & Automated Tests",
         "Mencetak **Risk Register** Capstone, mengeksekusi orkestrator terpadu `run_capstone_pipeline.py`, dan memverifikasi suite unit test `test_capstone_suite.py`.",
         "import responsible_ai_assessment, run_capstone_pipeline, unittest, test_capstone_suite\nresponsible_ai_assessment.main()\nrun_capstone_pipeline.main()",
         run_snippet("responsible_ai_assessment.py") + "\n" + run_snippet("run_capstone_pipeline.py") + "\n" + run_snippet("test_capstone_suite.py")),
    ]

    cells = [
        md_cell(
            "# Companion Notebook — Chapter 40: Advanced AI Capstone Project\n"
            "**Intelligent Sensor Monitoring with Machine Learning, Evaluation & Responsible AI**\n"
            "**Artificial Intelligence Learning Hub — Level 4 (ITERA)**\n\n"
            "Notebook end-to-end ini mengeksekusi proyek referensi Capstone di CPU lokal mulai dari perumusan dataset, *zero-leakage split*, pembandingan baseline & kandidat, *explainability*, *error analysis*, kontrol keamanan inferensi, *PSI/KS drift monitoring*, hingga Risk Register dan Model Card.\n"
        )
    ]
    for idx, (heading, desc, code, out) in enumerate(steps, start=1):
        cells.append(md_cell(f"## {heading}\n\n{desc}"))
        cells.append(code_cell(f"import sys\nsys.path.append('../code')\n{code}", out, idx))

    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12.10"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chapter_40_companion.ipynb")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2, ensure_ascii=False)
    print(f"Notebook generated cleanly: {out_path} ({len(cells)} cells)")


if __name__ == "__main__":
    main()

