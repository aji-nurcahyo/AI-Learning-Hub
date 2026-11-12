"""
Chapter 23 - Deep Learning with PyTorch
Contoh 6: Checkpointing (Saving/Loading state_dict) dan Inference Pipeline

Skrip ini mendemonstrasikan:
1. Praktik terbaik penyimpanan bobot model: state_dict vs serialisasi penuh.
2. Pemuatan aman menggunakan parameter weights_only=True (mencegah arbitary code execution).
3. Verifikasi konsistensi inferensi sebelum vs setelah reload (torch.allclose).
4. Pembuatan fungsi inferensi produksi end-to-end dengan probabilitas keyakinan (Softmax).
"""

import tempfile
from pathlib import Path
import torch
import torch.nn as nn

from neural_network_module import TabularMLP


def predict_single_or_batch(model, features_tensor):
    """
    Fungsi inferensi produksi: menerima input tensor dan mengembalikan
    prediksi kelas serta tingkat keyakinan (confidence score).
    """
    model.eval()
    with torch.no_grad():
        if features_tensor.ndim == 1:
            features_tensor = features_tensor.unsqueeze(0)  # Bentuk (1, D)
        logits = model(features_tensor)
        probs = torch.softmax(logits, dim=-1)
        confidences, predictions = torch.max(probs, dim=-1)
    return predictions.tolist(), confidences.tolist(), probs.tolist()


def main():
    print("=" * 70)
    print("PYTORCH CHECKPOINTING & INFERENCE PIPELINE")
    print("=" * 70)

    # 1. Inisialisasi Model Awal
    torch.manual_seed(42)
    original_model = TabularMLP(in_features=6, hidden_1=16, hidden_2=8, num_classes=2)

    # Contoh input untuk uji konsistensi
    test_input = torch.tensor([
        [0.5, -1.2, 0.8, 1.5, -0.4, 0.2],
        [-1.0, 0.4, -0.5, 0.0, 1.2, -0.8]
    ], dtype=torch.float32)

    # Prediksi sebelum disimpan
    original_model.eval()
    with torch.no_grad():
        original_logits = original_model(test_input)
    print("\n[1] Prediksi Model Awal (Sebelum Disimpan):")
    print(f"  Logits Asli:\n{original_logits}")

    # 2. Simpan Checkpoint state_dict
    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_path = Path(tmpdir) / "tabular_mlp_state_dict.pt"
        print(f"\n[2] Menyimpan Bobot Model ke Checkpoint:")
        print(f"  Jalur Berkas: {checkpoint_path.name}")
        
        # Simpan HANYA state_dict, BUKAN seluruh objek model
        torch.save(original_model.state_dict(), checkpoint_path)
        print("  Status: state_dict berhasil disimpan (hanya parameter bobot dan bias)")

        # 3. Buat Model Baru Bersih & Muat State Dict
        print("\n[3] Memuat Bobot ke Model Baru yang Masih Kosong:")
        loaded_model = TabularMLP(in_features=6, hidden_1=16, hidden_2=8, num_classes=2)
        
        # Muat dengan aman menggunakan weights_only=True (rekomendasi keamanan PyTorch modern)
        checkpoint_data = torch.load(checkpoint_path, weights_only=True)
        loaded_model.load_state_dict(checkpoint_data)
        loaded_model.eval()
        print("  Status: load_state_dict berhasil dieksekusi dengan weights_only=True")

        # 4. Uji Verifikasi Konsistensi Inferensi
        with torch.no_grad():
            loaded_logits = loaded_model(test_input)
        print("\n[4] Prediksi Model Setelah Dimuat (Reloaded):")
        print(f"  Logits Baru:\n{loaded_logits}")

        # Verifikasi numerik: logits harus 100% cocok
        diff = torch.abs(original_logits - loaded_logits).max().item()
        print(f"  Selisih Maksimum Logits: {diff:.2e}")
        assert torch.allclose(original_logits, loaded_logits, atol=1e-7), "Konsistensi reload gagal!"
        print("  Status Konsistensi: 100% IDENTIK (Presisi Floating-Point)")

    # 5. Pipeline Inferensi Produksi
    print("\n[5] Simulasi Inferensi Produksi:")
    preds, confs, probs = predict_single_or_batch(loaded_model, test_input)
    for i, (pred, conf, prob) in enumerate(zip(preds, confs, probs)):
        print(f"  Sampel {i+1}:")
        print(f"    Probabilitas Kelas: [Kelas 0: {prob[0]:.4f}, Kelas 1: {prob[1]:.4f}]")
        print(f"    Prediksi Akhir:     Kelas {pred}")
        print(f"    Tingkat Keyakinan:  {conf*100:.2f}%")

    print("\nKESIMPULAN: Checkpointing state_dict dan inferensi tervalidasi aman dan konsisten.")


if __name__ == "__main__":
    main()

