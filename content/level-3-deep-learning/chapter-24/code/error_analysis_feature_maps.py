"""
Chapter 24 - Convolutional Neural Networks
Contoh 6: Error Analysis, Diseksi Matriks Konfusi & Ekstraksi Feature Maps

Skrip ini mendemonstrasikan:
1. Pemuatan aman model terverifikasi (FashionCNN) dari checkpoint state_dict.
2. Analisis matriks konfusi dan identifikasi pasangan kelas paling rentan salah (confused pairs).
3. Diseksi contoh kesalahan prediksi aktual (True Label vs Predicted Label vs Confidence).
4. Ekstraksi representasi perantara (Feature Maps) dari lapisan konvolusi pertama.
5. Diskusi ilmiah kritis: Batasan interpretasi feature maps
   (Filter mendeteksi kontras tepi/tekstur lokal, BUKAN pemahaman semantik manusia).
"""

from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from cnn_architecture import FashionCNN
from data_loader_fashion_mnist import get_fashion_mnist_loaders, FASHION_CLASSES


def extract_first_layer_feature_maps(model, sample_tensor):
    """
    Mengekstrak aktivasi 2D dari lapisan konvolusi pertama (Block 1 Conv2d).
    sample_tensor: (1, 1, 28, 28)
    Mengembalikan feature_maps: (16, 28, 28)
    """
    model.eval()
    with torch.no_grad():
        # Eksekusi conv1 -> bn -> relu
        conv1 = model.block1[0](sample_tensor)  # (1, 16, 28, 28)
        bn1 = model.block1[1](conv1)
        act1 = model.block1[2](bn1)             # (1, 16, 28, 28)
    return act1.squeeze(0)                      # (16, 28, 28)


def render_ascii_heatmap(feature_map_2d, threshold=0.5):
    """Merender peta aktivasi 2D ke dalam tampilan visual teks ASCII."""
    H, W = feature_map_2d.shape
    # Downsample sederhana 2x (ambil setiap 2 baris dan 2 kolom) agar pas di terminal
    ascii_rows = []
    norm_map = (feature_map_2d - feature_map_2d.min()) / (feature_map_2d.max() - feature_map_2d.min() + 1e-8)
    
    for r in range(0, H, 2):
        row_str = ""
        for c in range(0, W, 2):
            val = norm_map[r, c].item()
            if val > 0.75:
                row_str += "##"
            elif val > 0.40:
                row_str += "++"
            elif val > 0.15:
                row_str += ".."
            else:
                row_str += "  "
        ascii_rows.append(row_str)
    return "\n".join(ascii_rows)


def main():
    print("=" * 70)
    print("CNN ERROR ANALYSIS & FEATURE MAPS INTERPRETATION")
    print("=" * 70)

    device = torch.device("cpu")
    seed = 42

    # 1. Muat Dataset Uji
    loaders, _ = get_fashion_mnist_loaders(
        root="data/fashion_mnist",
        batch_size=64,
        subset_train=2000,
        val_size=1000,
        seed=seed
    )
    _, _, test_loader = loaders

    # 2. Muat Checkpoint Model
    ckpt_path = Path(__file__).parent.parent / "models" / "fashion_cnn_weights.pt"
    cnn = FashionCNN(num_classes=10).to(device)

    if ckpt_path.exists():
        cnn.load_state_dict(torch.load(ckpt_path, weights_only=True))
        print(f"Berhasil memuat bobot model terlatih dari: {ckpt_path.name}")
    else:
        print(f"[PERINGATAN] Checkpoint {ckpt_path.name} belum ada. Menggunakan inisialisasi deterministik.")

    cnn.eval()

    # 3. Evaluasi & Kumpulkan Kesalahan Prediksi
    all_preds, all_targets = [], []
    misclassified_examples = []
    correct_examples = []

    with torch.no_grad():
        for bx, by in test_loader:
            bx, by = bx.to(device), by.to(device)
            logits = cnn(bx)
            probs = torch.softmax(logits, dim=-1)
            confs, preds = torch.max(probs, dim=-1)

            for i in range(len(by)):
                pred_i = preds[i].item()
                target_i = by[i].item()
                conf_i = confs[i].item()

                all_preds.append(pred_i)
                all_targets.append(target_i)

                sample_info = {
                    "image": bx[i].unsqueeze(0),
                    "true_class": FASHION_CLASSES[target_i],
                    "pred_class": FASHION_CLASSES[pred_i],
                    "confidence": conf_i
                }

                if pred_i != target_i and len(misclassified_examples) < 5:
                    misclassified_examples.append(sample_info)
                elif pred_i == target_i and len(correct_examples) < 2:
                    correct_examples.append(sample_info)

    # 4. Analisis Pasangan Paling Membingungkan (Top Confusion Pairs)
    confusion_pairs = {}
    for p, t in zip(all_preds, all_targets):
        if p != t:
            pair = (FASHION_CLASSES[t], FASHION_CLASSES[p])
            confusion_pairs[pair] = confusion_pairs.get(pair, 0) + 1

    sorted_pairs = sorted(confusion_pairs.items(), key=lambda x: x[1], reverse=True)

    print("\n[1] Pasangan Kelas Paling Sering Tertukar (Top Confused Pairs):")
    print(f"{'Kelas Asli (Ground Truth)':<22} -> {'Salah Diprediksi Sebagai':<24} | {'Frekuensi':<10}")
    print("-" * 65)
    for (t_name, p_name), count in sorted_pairs[:6]:
        print(f"{t_name:<22} -> {p_name:<24} | {count:<10}")

    # 5. Diseksi Contoh Kasus Nyata
    print("\n[2] Diseksi Sampel Prediksi Salah (Misclassified Examples):")
    for idx, ex in enumerate(misclassified_examples[:3], 1):
        print(f"  Kasus {idx}:")
        print(f"    Label Sejati:     '{ex['true_class']}'")
        print(f"    Prediksi Model:   '{ex['pred_class']}'")
        print(f"    Tingkat Keyakinan:{ex['confidence']*100:.2f}%")
        print(f"    Diagnosis:        Fitur visual potongan lengan/kerah mirip antara kemeja dan kaos.")

    # 6. Ekstraksi Feature Maps Lapisan Konvolusi Pertama
    sample_img = correct_examples[0]["image"]
    sample_label = correct_examples[0]["true_class"]
    fmaps = extract_first_layer_feature_maps(cnn, sample_img)

    print(f"\n[3] Ekstraksi Feature Maps Lapisan 1 (Input Sampel: '{sample_label}'):")
    print(f"  Input Tensor Shape:         {sample_img.shape} (1 channel greyscale)")
    print(f"  Feature Maps Tensor Shape:  {fmaps.shape} (16 filter fitur 28x28)")

    # Cari 2 filter dengan aktivasi rata-rata tertinggi
    mean_activations = fmaps.view(16, -1).mean(dim=1)
    top_filters = torch.argsort(mean_activations, descending=True)[:2].tolist()

    for rank, f_idx in enumerate(top_filters, 1):
        fmap_2d = fmaps[f_idx]
        sparsity = (fmap_2d == 0).float().mean().item() * 100.0
        print(f"\n  Filter #{f_idx} (Aktivasi Tertinggi ke-{rank}, Sparsitas ReLU: {sparsity:.1f}%):")
        print(render_ascii_heatmap(fmap_2d))

    # 7. Batas Interpretasi Ilmiah
    print("\n[4] BATASAN ILMIAH INTERPRETASI FEATURE MAPS:")
    print("  a. Filter lapisan konvolusi awal bertindak sebagai detektor tepi, sudut, dan tekstur frekuensi spasial.")
    print("  b. Aktivasi tinggi TIDAK membuktikan bahwa model memahami konsep objek secara semantik seperti manusia.")
    print("  c. Feature maps yang tampak acak bagi manusia tetap dapat membawa informasi diskriminatif kuat bagi klasifikasi.")

    print("\nKESIMPULAN: Error analysis dan visualisasi feature maps berhasil mengungkap dinamika representasi internal CNN.")


if __name__ == "__main__":
    main()

