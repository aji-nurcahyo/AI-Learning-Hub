"""
Chapter 30 - Deep Learning Capstone
Script 4: Training Engine, Validation Loop & Early Stopping
Implementasi siklus pelatihan industri: AdamW optimizer, gradient clipping,
pemantauan validasi terisolasi, pereduksi learning rate, dan early stopping.
"""

import time
import copy
import torch
import torch.nn as nn
from typing import Dict, List, Tuple
from torch.utils.data import DataLoader, TensorDataset


def train_model_with_early_stopping(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler = None,
    max_epochs: int = 50,
    patience: int = 8,
    max_grad_norm: float = 1.0
) -> Tuple[nn.Module, Dict[str, List[float]]]:
    """
    Siklus pelatihan profesional dengan early stopping dan model checkpointing di memori.
    """
    history = {
        "train_loss": [],
        "val_loss": [],
        "val_acc": [],
        "val_f1": []
    }

    best_val_loss = float("inf")
    best_model_weights = copy.deepcopy(model.state_dict())
    epochs_no_improve = 0

    t_start = time.perf_counter()

    for epoch in range(1, max_epochs + 1):
        # 1. Fase Pelatihan (Training Phase)
        model.train()
        running_train_loss = 0.0
        for X_b, y_b in train_loader:
            optimizer.zero_grad()
            logits = model(X_b)
            loss = criterion(logits, y_b)
            loss.backward()

            # Gradient Clipping
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)

            optimizer.step()
            running_train_loss += loss.item() * X_b.size(0)

        epoch_train_loss = running_train_loss / len(train_loader.dataset)

        # 2. Fase Validasi Terisolasi (Validation Phase)
        model.eval()
        running_val_loss = 0.0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for X_b, y_b in val_loader:
                logits = model(X_b)
                loss = criterion(logits, y_b)
                running_val_loss += loss.item() * X_b.size(0)

                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(y_b.cpu().numpy())

        epoch_val_loss = running_val_loss / len(val_loader.dataset)
        val_acc = (torch.tensor(all_preds) == torch.tensor(all_targets)).float().mean().item()

        # Update Scheduler
        if scheduler is not None:
            scheduler.step(epoch_val_loss)

        # Simpan metrik riwayat
        history["train_loss"].append(epoch_train_loss)
        history["val_loss"].append(epoch_val_loss)
        history["val_acc"].append(val_acc)

        # 3. Logika Early Stopping
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_model_weights = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
            star = "*"
        else:
            epochs_no_improve += 1
            star = ""

        if epoch % 5 == 0 or epoch == 1 or star == "*":
            print(f"Epoch {epoch:02d}/{max_epochs:02d} | Train Loss: {epoch_train_loss:.4f} | "
                  f"Val Loss: {epoch_val_loss:.4f} | Val Acc: {val_acc:.2%} {star}")

        if epochs_no_improve >= patience:
            print(f"--> Early stopping dipicu pada epoch {epoch} (Tanpa perbaikan selama {patience} epoch berturut-turut).")
            break

    total_time = time.perf_counter() - t_start
    print(f"Pelatihan Selesai dalam {total_time:.3f} detik. Memulihkan bobot model terbaik (Best Val Loss: {best_val_loss:.4f}).")

    # Muat kembali bobot terbaik
    model.load_state_dict(best_model_weights)
    return model, history


def main():
    print("=" * 70)
    print("CHAPTER 30 CAPSTONE: TRAINING ENGINE VERIFICATION")
    print("=" * 70)

    from dataset_loader import generate_capstone_dataset, split_data_leakage_free
    from deep_learning_model import DeepPredictiveMaintenanceNet
    from sklearn.preprocessing import StandardScaler

    df = generate_capstone_dataset(n_samples=1000, seed=42)
    train_df, val_df, _ = split_data_leakage_free(df, seed=42)

    features = ["temperature_c", "vibration_mms", "pressure_bar", "current_amp"]
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(train_df[features].values)
    X_v = scaler.transform(val_df[features].values)

    y_tr = train_df["condition_label"].values
    y_v = val_df["condition_label"].values

    train_ds = TensorDataset(torch.tensor(X_tr, dtype=torch.float32), torch.tensor(y_tr, dtype=torch.long))
    val_ds = TensorDataset(torch.tensor(X_v, dtype=torch.float32), torch.tensor(y_v, dtype=torch.long))

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)

    torch.manual_seed(42)
    model = DeepPredictiveMaintenanceNet(in_features=4, hidden_dim=64, num_classes=3)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.005, weight_decay=1e-4)

    trained_model, hist = train_model_with_early_stopping(
        model, train_loader, val_loader, criterion, optimizer, max_epochs=25, patience=5
    )

    print("\n[VERIFIKASI HASIL PELATIHAN]")
    print(f"Epoch Dilatih           : {len(hist['train_loss'])}")
    print(f"Loss Awal (Epoch 1)     : {hist['train_loss'][0]:.4f}")
    print(f"Loss Akhir              : {hist['train_loss'][-1]:.4f}")
    print(f"Akurasi Validasi Terbaik: {max(hist['val_acc']):.2%}")
    assert hist['train_loss'][-1] < hist['train_loss'][0], "Pelatihan gagal menurunkan loss!"
    print("✓ Model berhasil konvergen dan menurunkan loss secara konsisten.")


if __name__ == "__main__":
    main()

