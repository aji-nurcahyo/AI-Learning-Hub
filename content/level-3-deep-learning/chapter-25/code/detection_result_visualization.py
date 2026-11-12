"""
Chapter 25 - Object Detection & YOLO
Skrip 10: Detection Result Diagnostic Visualization (TP, FP, FN Visual Breakdown)
Menghasilkan visualisasi diagnostik perbandingan antara kotak Ground-Truth dan Prediksi Model:
- Hijau: True Positive (Deteksi Akurat)
- Merah: False Positive (Deteksi Palsu / False Alarm)
- Kuning: False Negative (Objek Terlewat / Missed Detection)
"""

from typing import List, Tuple
from pathlib import Path
import numpy as np
import cv2
from iou_calculation import compute_iou_single
from bounding_box_visualization import create_synthetic_detection_scene


def render_diagnostic_overlay(
    image: np.ndarray,
    gt_boxes: List[List[float]],
    pred_boxes: List[List[float]],
    pred_scores: List[float],
    iou_threshold: float = 0.5
) -> Tuple[np.ndarray, dict]:
    """
    Menggambar kotak diagnostik TP (Hijau), FP (Merah), dan FN (Kuning) pada kanvas citra.
    """
    canvas = image.copy()
    h_img, w_img = canvas.shape[:2]

    num_preds = len(pred_boxes)
    num_gts = len(gt_boxes)

    gt_matched = [False] * num_gts
    pred_matched = [False] * num_preds
    pred_iou_with_gt = [0.0] * num_preds

    # Urutkan prediksi berdasarkan skor menurun
    order = np.argsort(-np.array(pred_scores)) if num_preds > 0 else []

    for i in order:
        best_iou = 0.0
        best_gt = -1
        for j, g in enumerate(gt_boxes):
            iou, _, _ = compute_iou_single(pred_boxes[i], g)
            if iou > best_iou:
                best_iou = iou
                best_gt = j
        
        pred_iou_with_gt[i] = best_iou
        if best_iou >= iou_threshold and not gt_matched[best_gt]:
            pred_matched[i] = True
            gt_matched[best_gt] = True

    # 1. Gambar Ground Truth yang terlewat (False Negative - Kuning)
    fn_count = 0
    for j, g in enumerate(gt_boxes):
        if not gt_matched[j]:
            fn_count += 1
            x1, y1, x2, y2 = map(int, g)
            # Garis putus-putus atau tebal warna Kuning (BGR: 0, 255, 255)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.putText(canvas, f"FN: Ground Truth {j}", (x1, max(15, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)

    # 2. Gambar Prediksi TP (Hijau) dan FP (Merah)
    tp_count = 0
    fp_count = 0
    for i in range(num_preds):
        x1, y1, x2, y2 = map(int, pred_boxes[i])
        score = pred_scores[i]
        iou_val = pred_iou_with_gt[i]

        if pred_matched[i]:
            tp_count += 1
            color = (0, 255, 0) # Hijau
            label = f"TP conf:{score:.2f} iou:{iou_val:.2f}"
        else:
            fp_count += 1
            color = (0, 0, 255) # Merah
            label = f"FP conf:{score:.2f} iou:{iou_val:.2f}"

        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        cv2.putText(canvas, label, (x1, max(15, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

    metrics = {
        "true_positives": tp_count,
        "false_positives": fp_count,
        "false_negatives": fn_count,
        "precision": float(tp_count / (tp_count + fp_count)) if (tp_count + fp_count) > 0 else 0.0,
        "recall": float(tp_count / num_gts) if num_gts > 0 else 0.0
    }

    return canvas, metrics


if __name__ == "__main__":
    print("=== Visualisasi Diagnostik Hasil Deteksi (TP, FP, FN) ===")

    # Siapkan scene sintetis
    img, gts, _, _ = create_synthetic_detection_scene(640, 480)
    # Tambahkan satu GT ketiga di sudut kanan bawah
    gts.append([450.0, 360.0, 580.0, 440.0]) # GT 2

    # Skenario Prediksi Model:
    # Pred 0: Cocok GT 0 (TP)
    # Pred 1: Prediksi salah tempat di tengah (FP)
    # Pred 2: Duplikat deteksi pada GT 0 (FP)
    # Catatan: GT 1 dan GT 2 terlewat (FN)!
    preds = [
        [85.0, 122.0, 275.0, 318.0],  # Cocok GT 0 -> TP (IoU ~0.88, Skor 0.92)
        [220.0, 20.0, 340.0, 100.0],  # Salah tempat -> FP (Skor 0.65)
        [90.0, 125.0, 270.0, 315.0]   # Duplikat GT 0 -> FP (Skor 0.50)
    ]
    scores = [0.92, 0.65, 0.50]

    overlay, summary = render_diagnostic_overlay(img, gts, preds, scores, iou_threshold=0.5)

    print(f"Hasil Evaluasi Diagnostik:")
    print(f"  True Positives  (Hijau) : {summary['true_positives']}")
    print(f"  False Positives (Merah) : {summary['false_positives']}")
    print(f"  False Negatives (Kuning): {summary['false_negatives']}")
    print(f"  Precision               : {summary['precision']:.4f}")
    print(f"  Recall                  : {summary['recall']:.4f}")

    out_path = Path("experiments/diagnostic_detection_overlay.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), overlay)
    print(f"\nBerkas diagnostik tersimpan di: {out_path}")

    assert summary['true_positives'] == 1 and summary['false_positives'] == 2 and summary['false_negatives'] == 2
    print("Pengujian visualisasi diagnostik deteksi objek sukses!")

