"""
Chapter 30 - Deep Learning Capstone
Script 8: Capstone End-to-End Pipeline Orchestrator
Menjalankan seluruh tahapan proyek capstone secara terintegrasi:
Data Loading -> Preprocessing -> Baseline Benchmark -> Deep Learning Training ->
Test Set Evaluation -> Error Analysis -> Checkpointing -> Responsible AI Audit -> JSON Logging.
"""

import time
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

from dataset_loader import generate_capstone_dataset, split_data_leakage_free
from baseline_model import train_and_eval_baseline
from deep_learning_model import DeepPredictiveMaintenanceNet
from training_engine import train_model_with_early_stopping
from evaluation_diagnostics import evaluate_model_on_test_set
from checkpoint_manager import save_capstone_checkpoint
from responsible_ai_audit import audit_subgroup_disparity, evaluate_sensor_noise_robustness


def run_full_capstone_pipeline() -> dict:
    t_pipeline_start = time.perf_counter()
    print("=" * 80)
    print("AI LEARNING HUB — LEVEL 3 DEEP LEARNING CAPSTONE")
    print("REFERENCE IMPLEMENTATION: END-TO-END NEURAL NETWORK CLASSIFICATION")
    print("=" * 80)

    # 1. Dataset Loading & Partitioning
    print("\n[TAHAP 1: DATASET LOADING & LEAKAGE-FREE SPLITTING]")
    df = generate_capstone_dataset(n_samples=1000, seed=42)
    train_df, val_df, test_df = split_data_leakage_free(df, seed=42)
    print(f"Dataset dimuat: {len(df)} sampel telemetri.")
    print(f"Partisi 3-Way : Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}.")

    # 2. Feature Preprocessing (Fit on Train ONLY)
    features = ["temperature_c", "vibration_mms", "pressure_bar", "current_amp"]
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_df[features].values)
    X_val = scaler.transform(val_df[features].values)
    X_test = scaler.transform(test_df[features].values)

    y_train = train_df["condition_label"].values
    y_val = val_df["condition_label"].values
    y_test = test_df["condition_label"].values

    # 3. Baseline Benchmark Evaluation
    print("\n[TAHAP 2: BASELINE BENCHMARK EVALUATION (LOGISTIC REGRESSION)]")
    baseline_res = train_and_eval_baseline()
    print(f"Baseline Test Accuracy: {baseline_res['test_acc']:.2%}")
    print(f"Baseline Test Macro F1: {baseline_res['test_f1']:.4f}")

    # 4. PyTorch Deep Neural Network Setup
    print("\n[TAHAP 3: DEEP LEARNING ARCHITECTURE & TRAINING LOOP]")
    torch.manual_seed(42)
    model = DeepPredictiveMaintenanceNet(in_features=4, hidden_dim=64, num_classes=3, dropout_rate=0.15)
    
    train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
    val_ds = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long))
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.005, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

    trained_model, history = train_model_with_early_stopping(
        model, train_loader, val_loader, criterion, optimizer, scheduler=scheduler, max_epochs=35, patience=6
    )

    # 5. Final Evaluation on Held-Out Test Set
    print("\n[TAHAP 4: FINAL TEST SET EVALUATION & ERROR ANALYSIS]")
    eval_res = evaluate_model_on_test_set(trained_model, X_test, y_test, test_df)
    print(f"Deep Learning Test Accuracy : {eval_res['accuracy']:.2%}")
    print(f"Deep Learning Test Macro F1 : {eval_res['macro_f1']:.4f}")
    
    acc_gain = eval_res['accuracy'] - baseline_res['test_acc']
    f1_gain = eval_res['macro_f1'] - baseline_res['test_f1']
    print(f"Peningkatan vs Baseline     : Akurasi {acc_gain:+.2%}, Macro F1 {f1_gain:+.4f}")
    print(f"Total Galat Prediksi        : {eval_res['total_errors']} dari {eval_res['total_test_samples']} sampel")

    # 6. Checkpoint Persistence
    print("\n[TAHAP 5: MODEL CHECKPOINT PERSISTENCE]")
    ckpt_path = Path("content/level-3-deep-learning/chapter-30/experiments/capstone_best_model.pt")
    meta = {
        "dataset": "Simulated Predictive Maintenance Multi-Sensor",
        "test_accuracy": eval_res["accuracy"],
        "test_f1": eval_res["macro_f1"],
        "device": "cpu"
    }
    save_capstone_checkpoint(trained_model, scaler.mean_, scaler.scale_, meta, ckpt_path)

    # 7. Responsible AI Assessment
    print("\n[TAHAP 6: RESPONSIBLE AI AUDIT]")
    subgroup_audit = audit_subgroup_disparity(trained_model, X_test, y_test, test_df)
    noise_audit = evaluate_sensor_noise_robustness(trained_model, X_test, y_test)
    print(f"Disparitas Macro F1 Subgrup: {subgroup_audit['f1_disparity']:.4f} (Status Adil: {subgroup_audit['is_fair']})")

    total_time = time.perf_counter() - t_pipeline_start
    print(f"\nPipeline Capstone Selesai Penuh dalam: {total_time:.3f} detik pada CPU.")

    # 8. Save Full Pipeline Log to JSON
    summary_data = {
        "chapter": 30,
        "title": "Deep Learning Capstone Reference Pipeline",
        "timestamp": "2026-11-08",
        "total_runtime_seconds": round(total_time, 4),
        "dataset_samples": len(df),
        "split_counts": {"train": len(train_df), "val": len(val_df), "test": len(test_df)},
        "baseline": {
            "model": "LogisticRegression",
            "test_accuracy": round(baseline_res["test_acc"], 4),
            "test_macro_f1": round(baseline_res["test_f1"], 4)
        },
        "deep_learning": {
            "model": "DeepPredictiveMaintenanceNet",
            "epochs_trained": len(history["train_loss"]),
            "final_train_loss": round(history["train_loss"][-1], 4),
            "final_val_loss": round(history["val_loss"][-1], 4),
            "test_accuracy": round(eval_res["accuracy"], 4),
            "test_macro_f1": round(eval_res["macro_f1"], 4),
            "accuracy_gain_vs_baseline": round(acc_gain, 4),
            "f1_gain_vs_baseline": round(f1_gain, 4)
        },
        "responsible_ai": {
            "f1_disparity": round(subgroup_audit["f1_disparity"], 4),
            "is_fair": subgroup_audit["is_fair"],
            "noise_robustness": noise_audit
        }
    }

    out_json = Path("content/level-3-deep-learning/chapter-30/experiments/results_ch30.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)
    print(f"Log eksperimen capstone berhasil disimpan di: {out_json}")

    return summary_data


def main():
    run_full_capstone_pipeline()


if __name__ == "__main__":
    main()

