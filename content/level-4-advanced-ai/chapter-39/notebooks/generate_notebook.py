"""Generates pre-executed Chapter 39 companion notebook (nbformat v4)."""

import json
import os
import subprocess
import sys


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CODE_DIR = os.path.join(BASE_DIR, "code")


def run_snippet(script_name: str) -> str:
    res = subprocess.run([sys.executable, os.path.join(CODE_DIR, script_name)], cwd=CODE_DIR, capture_output=True, text=True, check=True)
    return res.stdout


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
        ("1. Safety, Security, Governance & Risk Register",
         "Menjalankan konstruksi **AI Risk Register** terpetakan ke 4 fungsi **NIST AI RMF 1.0** (*Govern, Map, Measure, Manage*) beserta skor risiko $5\\times 5$.",
         "import ai_risk_register, risk_scoring\nai_risk_register.main()\nrisk_scoring.main()",
         run_snippet("ai_risk_register.py") + "\n" + run_snippet("risk_scoring.py")),
        ("2. Synthetic Dataset Bias & Fairness Metrics Conflict",
         "Menganalisis ketimpangan representasi dan *measurement proxy* pada dataset sintetis, lalu membuktikan bahwa **Equal Opportunity** (`0.0000`) dapat terpenuhi sementara **Demographic Parity** (`0.2000`) tidak.",
         "import dataset_bias_analysis, fairness_metrics, group_performance_analysis\ndataset_bias_analysis.main()\nfairness_metrics.main()\ngroup_performance_analysis.main()",
         run_snippet("dataset_bias_analysis.py") + "\n" + run_snippet("fairness_metrics.py") + "\n" + run_snippet("group_performance_analysis.py")),
        ("3. Privacy, Data Minimization & Re-Identification Risks",
         "Membuktikan risiko *quasi-identifier singleton* pada rekaman yang hanya menghapus kolom nama, serta konsep **Laplace Differential Privacy**.",
         "import privacy_data_minimization\nprivacy_data_minimization.main()",
         run_snippet("privacy_data_minimization.py")),
        ("4. Input Validation & Prompt Injection Trust Boundaries",
         "Memvalidasi batas input numerik (`NaN`/`Inf`/boolean) dan mengisolasi teks korpus berbahaya sebagai data inert di dalam `<untrusted_document>`.",
         "import input_validation, prompt_injection_defense_demo, untrusted_content_handling\ninput_validation.main()\nprompt_injection_defense_demo.main()\nuntrusted_content_handling.main()",
         run_snippet("input_validation.py") + "\n" + run_snippet("prompt_injection_defense_demo.py") + "\n" + run_snippet("untrusted_content_handling.py")),
        ("5. Tool Permissions & Human Approval Gate",
         "Menegakkan **RBAC allowlisting** serta memblokir aksi `STATE_CHANGING` sebelum operator memberikan persetujuan eksplisit.",
         "import tool_permission_checks\ntool_permission_checks.main()",
         run_snippet("tool_permission_checks.py")),
        ("6. Robustness, OOD Escalation & Model Card",
         "Menguji degradasi model di bawah noise, `30%` missing features, dan input OOD (`+6.0` shift), lalu mencetak **Model Card** terstruktur.",
         "import model_robustness_checks, model_card_generator\nmodel_robustness_checks.main()\nmodel_card_generator.main()",
         run_snippet("model_robustness_checks.py") + "\n" + run_snippet("model_card_generator.py")),
        ("7. Governance Checklist & Final Responsible AI Audit",
         "Menjalankan daftar periksa tata kelola akademik dan suite audit keamanan/risiko terpadu.",
         "import governance_checklist, responsible_ai_audit, ai_safety_evaluation\ngovernance_checklist.main()\nresponsible_ai_audit.main()\nai_safety_evaluation.main()",
         run_snippet("governance_checklist.py") + "\n" + run_snippet("responsible_ai_audit.py") + "\n" + run_snippet("ai_safety_evaluation.py")),
    ]

    cells = [
        md_cell(
            "# Companion Notebook — Chapter 39: AI Safety, Security & Governance\n"
            "**Artificial Intelligence Learning Hub — Level 4 (ITERA)**\n\n"
            "Notebook ini mengintegrasikan demonstrasi defensif offline untuk manajemen risiko NIST AI RMF, keadilan algoritmis, privasi, validasi input, isolasi *Prompt Injection*, izin alat, dokumentasi **Model Card**, dan audit tata kelola.\n"
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
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chapter_39_companion.ipynb")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2, ensure_ascii=False)
    print(f"Notebook generated cleanly: {out_path} ({len(cells)} cells)")


if __name__ == "__main__":
    main()

