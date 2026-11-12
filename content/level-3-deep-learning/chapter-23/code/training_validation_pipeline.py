"""
Chapter 23 - Deep Learning with PyTorch
Contoh 5: Training, Validation, dan Testing Pipeline Lengkap

Skrip ini mendemonstrasikan:
1. Alur siklus pelatihan (training loop) terstandar industri dengan model.train().
2. Siklus validasi (validation loop) terisolasi dengan model.eval() dan torch.no_grad().
3. Pemantauan metrik per-epoch: Loss (CrossEntropy) dan Akurasi Klasifikasi.
4. Perbandingan performa objektif: Deep MLP vs Simple Baseline pada protokol evaluasi identik.
5. Evaluasi akhir pada test set terisolasi, matriks konfusi, dan penyimpanan log hasil terstruktur.
"""

import json
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset_dataloader import generate_synthetic_data, split_data_zero_leakage, SyntheticTabularDataset
from neural_network_module import TabularMLP, SimpleBaseline


def train_one_epoch(model, loader, criterion, optimizer, device):
    """Menjalankan satu siklus epoch pelatihan."""
    model.train()  # Mengaktifkan mode training (penting untuk dropout/batchnorm)
    total_loss = 0.0
    correct = 0
    total_samples = 0

    for batch_X, batch_y in loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)

        # 1. Reset gradien akumulator
        optimizer.zero_grad()

        # 2. Forward pass (menghasilkan logits mentah)
        logits = model(batch_X)

        # 3. Hitung loss
        loss = criterion(logits, batch_y)

        # 4. Backward pass
        loss.backward()

        # 5. Pembaruan bobot optimizer
        optimizer.step()

        # Akumulasi statistik
        total_loss += loss.item() * len(batch_y)
        preds = torch.argmax(logits, dim=-1)
        correct += (preds == batch_y).sum().item()
        total_samples += len(batch_y)

    epoch_loss = total_loss / total_samples
    epoch_acc = (correct / total_samples) * 100.0
    return epoch_loss, epoch_acc


def evaluate(model, loader, criterion, device):
    """Mengevaluasi model pada validasi atau pengujian tanpa komputasi gradien."""
    model.eval()  # Mengaktifkan mode evaluasi
    total_loss = 0.0
    correct = 0
    total_samples = 0

    all_preds = []
    all_targets = []

    with torch.no_grad():  # Menghentikan pelacakan graf komputasi (hemat memori)
        for batch_X, batch_y in loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            logits = model(batch_X)
            loss = criterion(logits, batch_y)

            total_loss += loss.item() * len(batch_y)
            preds = torch.argmax(logits, dim=-1)
            correct += (preds == batch_y).sum().item()
            total_samples += len(batch_y)

            all_preds.extend(preds.cpu().tolist())
            all_targets.extend(batch_y.cpu().tolist())

    eval_loss = total_loss / total_samples
    eval_acc = (correct / total_samples) * 100.0
    return eval_loss, eval_acc, all_preds, all_targets


def train_model(model, train_loader, val_loader, criterion, optimizer, epochs=25, device="cpu"):
    """Loop pelatihan penuh melintasi seluruh epoch."""
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        va_loss, va_acc, _, _ = evaluate(model, val_loader, criterion, device)

        history["train_loss"].append(round(tr_loss, 4))
        history["train_acc"].append(round(tr_acc, 2))
        history["val_loss"].append(round(va_loss, 4))
        history["val_acc"].append(round(va_acc, 2))

        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:2d}/{epochs:2d} | Train Loss: {tr_loss:.4f} Acc: {tr_acc:5.2f}% | Val Loss: {va_loss:.4f} Acc: {va_acc:5.2f}%")

    return history


def compute_confusion_matrix(preds, targets):
    """Menghitung matriks konfusi biner 2x2 tanpa dependensi sklearn."""
    tp = sum(1 for p, t in zip(preds, targets) if p == 1 and t == 1)
    tn = sum(1 for p, t in zip(preds, targets) if p == 0 and t == 0)
    fp = sum(1 for p, t in zip(preds, targets) if p == 1 and t == 0)
    fn = sum(1 for p, t in zip(preds, targets) if p == 0 and t == 1)
    return {"TP": tp, "TN": tn, "FP": fp, "FN": fn}


def main():
    print("=" * 70)
    print("PYTORCH TRAINING & VALIDATION PIPELINE")
    print("=" * 70)

    # 1. Konfigurasi Eksperimen
    seed = 42
    torch.manual_seed(seed)
    device = torch.device("cpu")
    print(f"Konfigurasi: Seed = {seed}, Device = {device}, Batch Size = 32, Epochs = 25")

    # 2. Persiapan Data Bebas Leakage
    X_raw, y_raw = generate_synthetic_data(n_samples=1000, n_features=6, seed=seed)
    (X_tr, y_tr), (X_va, y_va), (X_te, y_te) = split_data_zero_leakage(X_raw, y_raw, seed=seed)

    train_loader = DataLoader(SyntheticTabularDataset(X_tr, y_tr), batch_size=32, shuffle=True)
    val_loader = DataLoader(SyntheticTabularDataset(X_va, y_va), batch_size=32, shuffle=False)
    test_loader = DataLoader(SyntheticTabularDataset(X_te, y_te), batch_size=32, shuffle=False)

    criterion = nn.CrossEntropyLoss()

    # 3. Pelatihan Model 1: TabularMLP
    print("\n--- [1] Melatih Model: TabularMLP (Non-Linier) ---")
    torch.manual_seed(seed)
    mlp_model = TabularMLP(in_features=6, hidden_1=16, hidden_2=8, num_classes=2).to(device)
    mlp_optimizer = torch.optim.Adam(mlp_model.parameters(), lr=0.01)
    mlp_history = train_model(mlp_model, train_loader, val_loader, criterion, mlp_optimizer, epochs=25, device=device)

    # 4. Pelatihan Model 2: SimpleBaseline (Linier)
    print("\n--- [2] Melatih Model Acuan: SimpleBaseline (Linier) ---")
    torch.manual_seed(seed)
    baseline_model = SimpleBaseline(in_features=6, num_classes=2).to(device)
    baseline_optimizer = torch.optim.Adam(baseline_model.parameters(), lr=0.01)
    baseline_history = train_model(baseline_model, train_loader, val_loader, criterion, baseline_optimizer, epochs=25, device=device)

    # 5. Evaluasi Akhir pada Test Set Terisolasi
    print("\n--- [3] Evaluasi Akhir pada Test Set Terisolasi (N=150) ---")
    test_loss_mlp, test_acc_mlp, preds_mlp, targets_mlp = evaluate(mlp_model, test_loader, criterion, device)
    test_loss_base, test_acc_base, preds_base, targets_base = evaluate(baseline_model, test_loader, criterion, device)

    cm_mlp = compute_confusion_matrix(preds_mlp, targets_mlp)
    cm_base = compute_confusion_matrix(preds_base, targets_base)

    print(f"  TabularMLP    -> Test Loss: {test_loss_mlp:.4f} | Test Akurasi: {test_acc_mlp:.2f}% | CM: {cm_mlp}")
    print(f"  Baseline      -> Test Loss: {test_loss_base:.4f} | Test Akurasi: {test_acc_base:.2f}% | CM: {cm_base}")

    # Verifikasi keunggulan representasi non-linier MLP pada data sintetis non-linier
    print(f"\n  Selisih Akurasi Test (MLP vs Baseline): {test_acc_mlp - test_acc_base:+.2f}%")
    assert test_acc_mlp > test_acc_base, "MLP seharusnya mengungguli baseline linier pada fungsi sintetis non-linier"

    # 6. Simpan Hasil Eksperimen Terstruktur
    results_dir = Path(__file__).parent.parent / "experiments"
    results_dir.mkdir(parents=True, exist_ok=True)
    results_file = results_dir / "results_ch23.json"

    experiment_data = {
        "chapter": 23,
        "title": "Deep Learning with PyTorch",
        "dataset": "Synthetic Non-Linear Tabular (1000 samples, 6 features)",
        "split": {"train": len(X_tr), "val": len(X_va), "test": len(X_te)},
        "hyperparameters": {"batch_size": 32, "epochs": 25, "lr": 0.01, "optimizer": "Adam", "loss": "CrossEntropyLoss"},
        "mlp": {
            "parameters": sum(p.numel() for p in mlp_model.parameters()),
            "final_val_acc": mlp_history["val_acc"][-1],
            "test_loss": round(test_loss_mlp, 4),
            "test_acc": round(test_acc_mlp, 2),
            "confusion_matrix": cm_mlp,
            "history": mlp_history
        },
        "baseline": {
            "parameters": sum(p.numel() for p in baseline_model.parameters()),
            "final_val_acc": baseline_history["val_acc"][-1],
            "test_loss": round(test_loss_base, 4),
            "test_acc": round(test_acc_base, 2),
            "confusion_matrix": cm_base,
            "history": baseline_history
        }
    }

    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(experiment_data, f, indent=2)
    print(f"\nBerhasil menyimpan log eksperimen terstruktur ke: {results_file}")

    print("\nKESIMPULAN: Pipeline pelatihan PyTorch lengkap terverifikasi dan siap digunakan.")


if __name__ == "__main__":
    main()

