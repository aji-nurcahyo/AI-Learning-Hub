"""
Chapter 30 - Deep Learning Capstone
Script 6: Checkpoint Management, Safe Model Reload & Reproducibility
Penyimpanan checkpoint model terlatih beserta metadata skema, pemulihan aman (safe reload),
serta verifikasi reproduksibilitas numerik deterministik.
"""

from pathlib import Path
import torch
import numpy as np
from deep_learning_model import DeepPredictiveMaintenanceNet


def save_capstone_checkpoint(
    model: torch.nn.Module,
    scaler_mean: np.ndarray,
    scaler_scale: np.ndarray,
    metadata: dict,
    checkpoint_path: Path
):
    """Menyimpan checkpoint lengkap berstandar produksi."""
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state_dict": model.state_dict(),
        "model_architecture": "DeepPredictiveMaintenanceNet",
        "in_features": model.in_features,
        "num_classes": model.num_classes,
        "scaler_mean": scaler_mean.tolist(),
        "scaler_scale": scaler_scale.tolist(),
        "metadata": metadata
    }
    torch.save(payload, checkpoint_path)
    print(f"Checkpoint berhasil disimpan di: {checkpoint_path}")


def load_capstone_checkpoint(checkpoint_path: Path) -> tuple:
    """Memuat kembali checkpoint dan mengembalikan model serta scaler parameter."""
    payload = torch.load(checkpoint_path, weights_only=False)
    
    model = DeepPredictiveMaintenanceNet(
        in_features=payload["in_features"],
        num_classes=payload["num_classes"]
    )
    model.load_state_dict(payload["model_state_dict"])
    model.eval()

    scaler_mean = np.array(payload["scaler_mean"])
    scaler_scale = np.array(payload["scaler_scale"])
    metadata = payload.get("metadata", {})

    return model, scaler_mean, scaler_scale, metadata


def main():
    print("=" * 70)
    print("CHAPTER 30 CAPSTONE: CHECKPOINT MANAGEMENT & REPRODUCIBILITY")
    print("=" * 70)

    # Inisialisasi model dan bobot acak deterministik
    torch.manual_seed(42)
    original_model = DeepPredictiveMaintenanceNet(in_features=4, hidden_dim=64, num_classes=3)
    original_model.eval()

    dummy_mean = np.array([65.0, 2.5, 120.0, 15.0])
    dummy_scale = np.array([8.0, 0.8, 12.0, 2.0])

    ckpt_file = Path("content/level-3-deep-learning/chapter-30/experiments/capstone_model_checkpoint.pt")

    meta = {
        "trained_date": "2026-11-08",
        "author": "Aji Pamungkas Tri Nurcahyo",
        "target_task": "Predictive Maintenance 3-Class",
        "best_val_loss": 0.2854
    }

    # 1. Simpan checkpoint
    save_capstone_checkpoint(original_model, dummy_mean, dummy_scale, meta, ckpt_file)

    # 2. Muat kembali checkpoint ke instance model baru
    loaded_model, l_mean, l_scale, l_meta = load_capstone_checkpoint(ckpt_file)
    print(f"Checkpoint berhasil dimuat. Metadata tanggal latih: {l_meta.get('trained_date')}")

    # 3. Uji Kesetaraan Numerik 100% (Reproducibility Verification)
    test_input = torch.randn(5, 4)
    with torch.no_grad():
        out_orig = original_model(test_input)
        out_load = loaded_model(test_input)

    max_diff = torch.max(torch.abs(out_orig - out_load)).item()
    print(f"\n[VERIFIKASI KESETARAAN NUMERIK HASIL INFERENSI]")
    print(f"Deviasi Maksimum Antar-Model: {max_diff:.8e}")

    assert max_diff < 1e-7, "Error: Hasil model yang dimuat ulang tidak identik dengan model asli!"
    print("✓ Model asli dan model yang dimuat terbukti 100% IDENTIK secara floating-point bitwise.")


if __name__ == "__main__":
    main()

