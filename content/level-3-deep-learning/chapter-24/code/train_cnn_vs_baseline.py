"""
Chapter 24 - Convolutional Neural Networks
Contoh 5: Pelatihan Komparatif CNN vs Baseline (Fashion-MNIST)

Skrip ini mendemonstrasikan:
1. Pelatihan terpadu FashionCNN vs FlattenMLPBaseline pada protokol eksperimen identik.
2. Opsi mode ganda: --smoke-test (verifikasi cepat) atau mode standar edukatif.
3. Partisi data 3-way terisolasi (Train, Validation, Test) dengan seed deterministik.
4. Pemantauan kurva loss dan akurasi per-epoch (training dan validation).
5. Pengukuran waktu eksekusi nyata pada CPU.
6. Evaluasi komprehensif pada test set terisolasi: akurasi per-kelas, matriks konfusi,
   dan penyimpanan log hasil eksperimen ke experiments/results_ch24.json.
"""

import sys
import time
import json
import argparse
from pathlib import Path
import torch
import torch.nn as nn

from cnn_architecture import FashionCNN, FlattenMLPBaseline
from data_loader_fashion_mnist import get_fashion_mnist_loaders, FASHION_CLASSES


def train_epoch(model, loader, criterion, optimizer, device):
    """Menjalankan satu siklus epoch pelatihan."""
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for bx, by in loader:
        bx, by = bx.to(device), by.to(device)
        optimizer.zero_grad()
        logits = model(bx)
        loss = criterion(logits, by)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(by)
        preds = torch.argmax(logits, dim=-1)
        correct += (preds == by).sum().item()
        total += len(by)
    return total_loss / total, (correct / total) * 100.0


def evaluate_model(model, loader, criterion, device):
    """Mengevaluasi model pada validasi atau pengujian terisolasi."""
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_targets = [], []

    with torch.no_grad():
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            logits = model(bx)
            loss = criterion(logits, by)

            total_loss += loss.item() * len(by)
            preds = torch.argmax(logits, dim=-1)
            correct += (preds == by).sum().item()
            total += len(by)

            all_preds.extend(preds.cpu().tolist())
            all_targets.extend(by.cpu().tolist())

    return total_loss / total, (correct / total) * 100.0, all_preds, all_targets


def compute_multiclass_metrics(preds, targets, num_classes=10):
    """Menghitung akurasi per-kelas dan matriks konfusi 10x10."""
    cm = [[0] * num_classes for _ in range(num_classes)]
    for p, t in zip(preds, targets):
        cm[t][p] += 1

    per_class_acc = {}
    for c in range(num_classes):
        total_class = sum(cm[c])
        correct_class = cm[c][c]
        acc = (correct_class / total_class * 100.0) if total_class > 0 else 0.0
        per_class_acc[FASHION_CLASSES[c]] = round(acc, 2)

    return cm, per_class_acc


def run_experiment(smoke_test=False):
    print("=" * 70)
    print("CNN VS FLATTEN MLP EXPERIMENTAL BENCHMARK")
    print("=" * 70)

    # 1. Konfigurasi
    seed = 42
    torch.manual_seed(seed)
    device = torch.device("cpu")

    if smoke_test:
        print("[MODE: SMOKE TEST CEPAT]")
        train_samples = 600
        val_samples = 200
        test_samples = 400
        epochs = 2
        batch_size = 32
    else:
        print("[MODE: EKSPERIMEN EDUKATIF LENGKAP]")
        train_samples = 6000
        val_samples = 1500
        test_samples = 2000
        epochs = 5
        batch_size = 64

    lr = 0.001
    print(f"Konfigurasi: Train={train_samples}, Val={val_samples}, Test={test_samples}, Epochs={epochs}, LR={lr}")

    # 2. Pemuatan Data
    loaders, is_synthetic = get_fashion_mnist_loaders(
        root="data/fashion_mnist",
        batch_size=batch_size,
        val_size=val_samples,
        subset_train=train_samples,
        seed=seed
    )
    train_loader, val_loader, full_test_loader = loaders

    # Batasi data uji jika smoke test atau subset
    if smoke_test or test_samples < len(full_test_loader.dataset):
        test_sub_indices = list(range(min(test_samples, len(full_test_loader.dataset))))
        test_ds = torch.utils.data.Subset(full_test_loader.dataset, test_sub_indices)
        test_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    else:
        test_loader = full_test_loader

    criterion = nn.CrossEntropyLoss()

    # 3. Latih FashionCNN
    print(f"\n--- [1] Melatih Model: FashionCNN ({epochs} Epochs) ---")
    torch.manual_seed(seed)
    cnn = FashionCNN(num_classes=10).to(device)
    cnn_opt = torch.optim.Adam(cnn.parameters(), lr=lr)

    cnn_history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    t0_cnn = time.perf_counter()

    for epoch in range(1, epochs + 1):
        tr_l, tr_a = train_epoch(cnn, train_loader, criterion, cnn_opt, device)
        va_l, va_a, _, _ = evaluate_model(cnn, val_loader, criterion, device)
        cnn_history["train_loss"].append(round(tr_l, 4))
        cnn_history["train_acc"].append(round(tr_a, 2))
        cnn_history["val_loss"].append(round(va_l, 4))
        cnn_history["val_acc"].append(round(va_a, 2))
        print(f"  CNN Epoch {epoch:2d}/{epochs:2d} | Train Loss: {tr_l:.4f} Acc: {tr_a:5.2f}% | Val Loss: {va_l:.4f} Acc: {va_a:5.2f}%")

    cnn_train_time = round(time.perf_counter() - t0_cnn, 2)
    print(f"  Waktu Pelatihan CNN (CPU): {cnn_train_time} detik")

    # 4. Latih FlattenMLPBaseline
    print(f"\n--- [2] Melatih Model Acuan: FlattenMLPBaseline ({epochs} Epochs) ---")
    torch.manual_seed(seed)
    mlp = FlattenMLPBaseline(num_classes=10).to(device)
    mlp_opt = torch.optim.Adam(mlp.parameters(), lr=lr)

    mlp_history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    t0_mlp = time.perf_counter()

    for epoch in range(1, epochs + 1):
        tr_l, tr_a = train_epoch(mlp, train_loader, criterion, mlp_opt, device)
        va_l, va_a, _, _ = evaluate_model(mlp, val_loader, criterion, device)
        mlp_history["train_loss"].append(round(tr_l, 4))
        mlp_history["train_acc"].append(round(tr_a, 2))
        mlp_history["val_loss"].append(round(va_l, 4))
        mlp_history["val_acc"].append(round(va_a, 2))
        print(f"  MLP Epoch {epoch:2d}/{epochs:2d} | Train Loss: {tr_l:.4f} Acc: {tr_a:5.2f}% | Val Loss: {va_l:.4f} Acc: {va_a:5.2f}%")

    mlp_train_time = round(time.perf_counter() - t0_mlp, 2)
    print(f"  Waktu Pelatihan MLP (CPU): {mlp_train_time} detik")

    # 5. Evaluasi Uji Akhir Terisolasi
    print(f"\n--- [3] Evaluasi Akhir pada Test Set Terisolasi (N={len(test_loader.dataset)}) ---")
    test_loss_cnn, test_acc_cnn, preds_cnn, targets_cnn = evaluate_model(cnn, test_loader, criterion, device)
    test_loss_mlp, test_acc_mlp, preds_mlp, targets_mlp = evaluate_model(mlp, test_loader, criterion, device)

    cm_cnn, per_class_cnn = compute_multiclass_metrics(preds_cnn, targets_cnn)
    cm_mlp, per_class_mlp = compute_multiclass_metrics(preds_mlp, targets_mlp)

    print(f"  FashionCNN         -> Test Loss: {test_loss_cnn:.4f} | Test Akurasi: {test_acc_cnn:.2f}%")
    print(f"  FlattenMLPBaseline -> Test Loss: {test_loss_mlp:.4f} | Test Akurasi: {test_acc_mlp:.2f}%")
    print(f"  Selisih Akurasi (CNN vs MLP): {test_acc_cnn - test_acc_mlp:+.2f}%")

    print("\n[Performa Per-Kelas (Akurasi %)]:")
    print(f"{'Nama Kelas':<15} | {'CNN (%)':<10} | {'MLP (%)':<10} | {'Delta':<8}")
    print("-" * 50)
    for cls_name in FASHION_CLASSES:
        acc_c = per_class_cnn[cls_name]
        acc_m = per_class_mlp[cls_name]
        print(f"{cls_name:<15} | {acc_c:>8.2f}% | {acc_m:>8.2f}% | {acc_c - acc_m:>+6.2f}%")

    # 6. Simpan Hasil Eksperimen
    results_dir = Path(__file__).parent.parent / "experiments"
    results_dir.mkdir(parents=True, exist_ok=True)
    results_path = results_dir / "results_ch24.json"

    experiment_report = {
        "chapter": 24,
        "title": "Convolutional Neural Networks",
        "dataset": "Fashion-MNIST (Zalando Research, MIT License)" if not is_synthetic else "Synthetic Fashion Data",
        "is_synthetic": is_synthetic,
        "protocol": {
            "train_samples": len(train_loader.dataset),
            "val_samples": len(val_loader.dataset),
            "test_samples": len(test_loader.dataset),
            "batch_size": batch_size,
            "epochs": epochs,
            "learning_rate": lr,
            "optimizer": "Adam",
            "device": "CPU"
        },
        "cnn": {
            "parameters": sum(p.numel() for p in cnn.parameters()),
            "train_time_sec": cnn_train_time,
            "final_val_acc": cnn_history["val_acc"][-1],
            "test_loss": round(test_loss_cnn, 4),
            "test_acc": round(test_acc_cnn, 2),
            "per_class_acc": per_class_cnn,
            "confusion_matrix": cm_cnn,
            "history": cnn_history
        },
        "mlp": {
            "parameters": sum(p.numel() for p in mlp.parameters()),
            "train_time_sec": mlp_train_time,
            "final_val_acc": mlp_history["val_acc"][-1],
            "test_loss": round(test_loss_mlp, 4),
            "test_acc": round(test_acc_mlp, 2),
            "per_class_acc": per_class_mlp,
            "confusion_matrix": cm_mlp,
            "history": mlp_history
        }
    }

    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(experiment_report, f, indent=2)
    print(f"\nBerhasil menyimpan log eksperimen terstruktur ke: {results_path}")

    # Simpan bobot checkpoint CNN untuk ekstraksi feature maps di skrip berikutnya
    models_dir = Path(__file__).parent.parent / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = models_dir / "fashion_cnn_weights.pt"
    torch.save(cnn.state_dict(), ckpt_path)
    print(f"Checkpoint bobot CNN disimpan ke: {ckpt_path.name}")

    print("\nKESIMPULAN: Eksperimen komparatif selesai dan seluruh artefak tervalidasi.")
    return experiment_report


def main():
    parser = argparse.ArgumentParser(description="Pelatihan komparatif CNN vs Baseline")
    parser.add_argument("--smoke-test", action="store_true", help="Jalankan smoke test cepat 2 epoch")
    args = parser.parse_args()

    run_experiment(smoke_test=args.smoke_test)


if __name__ == "__main__":
    main()

