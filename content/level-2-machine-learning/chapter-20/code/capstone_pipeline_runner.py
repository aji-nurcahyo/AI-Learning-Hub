"""
Chapter 20 - Machine Learning Capstone
Script 9: Capstone Pipeline Master Runner
Menjalankan siklus hidup lengkap (end-to-end) Machine Learning Capstone
secara berurutan, terintegrasi, dan dapat direproduksi 100%.
"""

import warnings
warnings.filterwarnings('ignore')
import sys
from pathlib import Path

from generate_capstone_dataset import generate_capstone_dataset
from baseline_models import run_baseline_evaluation
from candidate_models import run_candidate_model_benchmark
from cross_validation_tuning import run_tuning
from final_test_evaluation import evaluate_test_set
from error_analysis_diagnostics import run_diagnostics
from explainability_analysis import run_explainability
from responsible_ai_assessment import run_responsible_ai

def run_full_pipeline():
    print("#" * 90)
    print("AI LEARNING HUB - LEVEL 2 MACHINE LEARNING CAPSTONE")
    print("MASTER PIPELINE RUNNER: INDUSTRIAL PREDICTIVE MAINTENANCE SYSTEM")
    print("#" * 90)

    print("\n>>> TAHAP 1: PEMBUATAN & VERIFIKASI DATASET TELEMETRI")
    generate_capstone_dataset()

    print("\n>>> TAHAP 2: BENCHMARK DUMMY BASELINE & SIMPLE MODEL")
    run_baseline_evaluation()

    print("\n>>> TAHAP 3: SELEKSI & BENCHMARK KANDIDAT MODEL (5-FOLD CV)")
    run_candidate_model_benchmark()

    print("\n>>> TAHAP 4: HYPERPARAMETER TUNING (GRIDSEARCH CV)")
    champion_name, best_model = run_tuning()

    print("\n>>> TAHAP 5: EVALUASI AKHIR PADA UNTOUCHED TEST SET")
    evaluate_test_set()

    print("\n>>> TAHAP 6: ERROR ANALYSIS & DECISION THRESHOLD COST OPTIMIZATION")
    run_diagnostics()

    print("\n>>> TAHAP 7: MODEL EXPLAINABILITY & FEATURE IMPORTANCE")
    run_explainability()

    print("\n>>> TAHAP 8: RESPONSIBLE AI, FAIRNESS & GOVERNANCE AUDIT")
    run_responsible_ai()

    print("\n" + "#" * 90)
    print("SELURUH 8 TAHAPAN MACHINE LEARNING CAPSTONE SELESAI DENGAN STATUS: SUKSES VERIFIKASI")
    print("#" * 90)

if __name__ == "__main__":
    run_full_pipeline()
