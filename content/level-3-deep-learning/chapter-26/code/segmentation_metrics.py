"""
Chapter 26 - Computer Vision Applications
Skrip 7: Image Segmentation Evaluation Metrics (Mask IoU, Dice Score, and Pixel Accuracy)
Menghitung metrik evaluasi segmentasi citra tingkat piksel:
Mask Intersection over Union (Jaccard Index), Dice Similarity Coefficient (F1-Score),
Pixel Accuracy, serta pembuktian hubungan aljabar Dice vs IoU.
"""

from typing import Tuple, Dict
import numpy as np


def compute_segmentation_metrics(
    pred_mask: np.ndarray,
    gt_mask: np.ndarray
) -> Dict[str, float]:
    """
    Menghitung metrik evaluasi antara mask biner prediksi dan ground-truth.
    Mask bernilai boolean atau {0, 1} atau {0, 255}.
    """
    p = (pred_mask > 0).astype(bool)
    g = (gt_mask > 0).astype(bool)

    # 1. Elemen Matriks Kontingensi Piksel
    true_positive = np.logical_and(p, g).sum()
    false_positive = np.logical_and(p, np.logical_not(g)).sum()
    false_negative = np.logical_and(np.logical_not(p), g).sum()
    true_negative = np.logical_and(np.logical_not(p), np.logical_not(g)).sum()

    total_pixels = pred_mask.size

    # 2. Mask IoU (Jaccard Index)
    intersection = true_positive
    union = true_positive + false_positive + false_negative
    iou = intersection / union if union > 0 else 1.0

    # 3. Dice Similarity Coefficient (F1-Score)
    dice = (2.0 * intersection) / (p.sum() + g.sum()) if (p.sum() + g.sum()) > 0 else 1.0

    # 4. Pixel Accuracy
    pixel_acc = (true_positive + true_negative) / total_pixels

    # 5. Precision & Recall tingkat piksel
    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 1.0
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 1.0

    return {
        "mask_iou": float(iou),
        "dice_coefficient": float(dice),
        "pixel_accuracy": float(pixel_acc),
        "precision": float(precision),
        "recall": float(recall),
        "tp_pixels": int(true_positive),
        "fp_pixels": int(false_positive),
        "fn_pixels": int(false_negative)
    }


if __name__ == "__main__":
    print("=== Eksplorasi Metrik Evaluasi Segmentasi (Mask IoU & Dice Score) ===")

    # Skenario Contoh Numerik Terhitung Eksplisit:
    # Kanvas 100x100 (10.000 piksel)
    # Ground Truth Mask: Kotak 50x50 di koordinat [20:70, 20:70] -> Luas = 2.500 piksel
    # Prediksi Mask    : Kotak 50x50 di koordinat [30:80, 20:70] -> Luas = 2.500 piksel
    # Irisan (Intersection): [30:70, 20:70] -> 40 x 50 = 2.000 piksel
    # Gabungan (Union)     : 2.500 + 2.500 - 2.000 = 3.000 piksel
    #
    # Nilai Teoritis:
    # IoU  = 2.000 / 3.000 = 2/3 ≈ 0.666667
    # Dice = (2 * 2.000) / (2.500 + 2.500) = 4.000 / 5.000 = 0.800000
    gt = np.zeros((100, 100), dtype=np.uint8)
    gt[20:70, 20:70] = 1

    pred = np.zeros((100, 100), dtype=np.uint8)
    pred[30:80, 20:70] = 1

    metrics = compute_segmentation_metrics(pred, gt)

    print("Hasil Perhitungan Metrik Evaluasi:")
    print(f"  Piksel Irisan (TP)       : {metrics['tp_pixels']:,} piksel (Teoritis: 2.000)")
    print(f"  Piksel Prediksi Salah(FP): {metrics['fp_pixels']:,} piksel (Teoritis: 500)")
    print(f"  Piksel Objek Terlewat(FN): {metrics['fn_pixels']:,} piksel (Teoritis: 500)")
    print(f"  Mask IoU (Jaccard Index) : {metrics['mask_iou']:.6f} (Teoritis 2/3: {2/3:.6f})")
    print(f"  Dice Coefficient (F1)    : {metrics['dice_coefficient']:.6f} (Teoritis: 0.800000)")
    print(f"  Pixel Accuracy Total     : {metrics['pixel_accuracy'] * 100:.2f}%")
    print(f"  Precision Piksel         : {metrics['precision']:.4f}")
    print(f"  Recall Piksel            : {metrics['recall']:.4f}")

    # Pembuktian Hubungan Aljabar Dice vs IoU:
    # Dice = 2*IoU / (1 + IoU)
    # Dice = 2*(2/3) / (1 + 2/3) = (4/3) / (5/3) = 4/5 = 0.8000
    derived_dice = (2.0 * metrics["mask_iou"]) / (1.0 + metrics["mask_iou"])
    print(f"\nVerifikasi Hubungan Aljabar:")
    print(f"  Dice dari Rumus Hubungan (2*IoU / (1+IoU)): {derived_dice:.6f}")
    assert abs(metrics["dice_coefficient"] - derived_dice) < 1e-6
    print("  Status: HUBUNGAN MATEMATIS DICE VS IOU TERBUKTI COCOK 100%!")
    print("\nSeluruh perhitungan metrik evaluasi segmentasi berhasil diverifikasi!")

