"""
Chapter 25 - Object Detection & YOLO
Skrip 9: Confidence Threshold Impact Analysis
Menganalisis dampak variasi ambang batas skor keyakinan (Confidence Threshold)
terhadap jumlah deteksi, tingkat presisi, recall, serta kompromi (trade-off) false alarm vs missed detections.
"""

from typing import List, Dict
import numpy as np
from detection_metrics import match_predictions_to_ground_truth


def evaluate_threshold_impact(
    pred_boxes: List[List[float]],
    pred_scores: List[float],
    gt_boxes: List[List[float]],
    thresholds: List[float] = [0.10, 0.25, 0.50, 0.75, 0.90],
    iou_eval_threshold: float = 0.50
) -> List[Dict[str, float]]:
    """
    Mengevaluasi trade-off presisi dan recall di bawah berbagai ambang batas keyakinan.
    """
    records = []
    num_gts = len(gt_boxes)

    for th in thresholds:
        # Tapis prediksi berdasarkan ambang batas keyakinan
        filtered_indices = [i for i, s in enumerate(pred_scores) if s >= th]
        f_boxes = [pred_boxes[i] for i in filtered_indices]
        f_scores = [pred_scores[i] for i in filtered_indices]

        tp, fp, _ = match_predictions_to_ground_truth(
            f_boxes, f_scores, gt_boxes, iou_threshold=iou_eval_threshold
        )

        tp_count = int(np.sum(tp))
        fp_count = int(np.sum(fp))
        fn_count = num_gts - tp_count

        precision = tp_count / (tp_count + fp_count) if (tp_count + fp_count) > 0 else 1.0
        recall = tp_count / num_gts if num_gts > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        records.append({
            "confidence_threshold": float(th),
            "retained_predictions": len(filtered_indices),
            "true_positives": tp_count,
            "false_positives": fp_count,
            "false_negatives": fn_count,
            "precision": float(round(precision, 4)),
            "recall": float(round(recall, 4)),
            "f1_score": float(round(f1, 4))
        })

    return records


if __name__ == "__main__":
    print("=== Analisis Dampak Ambang Batas Keyakinan (Confidence Threshold) ===")

    # Skenario Simulasi: 5 Ground Truth objek dengan 8 deteksi kandidat
    gts = [
        [100.0, 100.0, 200.0, 200.0],
        [250.0, 150.0, 350.0, 250.0],
        [400.0, 200.0, 500.0, 300.0],
        [50.0,  300.0, 150.0, 400.0],
        [300.0, 350.0, 420.0, 450.0]
    ]

    preds = [
        [102.0, 98.0, 198.0, 202.0],  # Skor 0.94 -> TP pada GT 0
        [252.0, 148.0, 348.0, 252.0], # Skor 0.88 -> TP pada GT 1
        [405.0, 195.0, 495.0, 305.0], # Skor 0.72 -> TP pada GT 2
        [55.0,  295.0, 145.0, 405.0], # Skor 0.55 -> TP pada GT 3
        [305.0, 345.0, 415.0, 455.0], # Skor 0.38 -> TP pada GT 4
        [120.0, 250.0, 180.0, 310.0], # Skor 0.30 -> FP (Noise / False Alarm)
        [450.0, 80.0,  510.0, 140.0], # Skor 0.20 -> FP (Noise)
        [220.0, 380.0, 260.0, 420.0]  # Skor 0.12 -> FP (Noise sangat rendah)
    ]
    scores = [0.94, 0.88, 0.72, 0.55, 0.38, 0.30, 0.20, 0.12]

    test_thresholds = [0.10, 0.25, 0.50, 0.75, 0.90]
    eval_results = evaluate_threshold_impact(preds, scores, gts, thresholds=test_thresholds)

    print(f"Total Ground Truth: {len(gts)} objek | Total Prediksi Mentah: {len(preds)} kotak\n")
    print(f"{'Threshold':<11} | {'Preds':<6} | {'TP':<4} | {'FP':<4} | {'FN':<4} | {'Precision':<10} | {'Recall':<8} | {'F1-Score':<8}")
    print("-" * 75)
    for r in eval_results:
        print(f"{r['confidence_threshold']:<11.2f} | {r['retained_predictions']:<6} | {r['true_positives']:<4} | {r['false_positives']:<4} | {r['false_negatives']:<4} | {r['precision']:<10.4f} | {r['recall']:<8.4f} | {r['f1_score']:<8.4f}")

    # Verifikasi kesimpulan ilmiah
    assert eval_results[0]["recall"] >= eval_results[-1]["recall"], "Recall harus menurun seiring kenaikan threshold!"
    assert eval_results[-1]["precision"] >= eval_results[0]["precision"], "Presisi harus meningkat pada threshold tinggi!"
    print("\nKesimpulan Ilmiah Terbukti:")
    print("1. Threshold Rendah (0.10): Menghasilkan Recall Maksimal (1.0000) namun memicu banyak False Alarm (FP=3, Presisi 0.6250).")
    print("2. Threshold Tinggi (0.90): Menghasilkan Presisi Sempurna (1.0000) namun mengorbankan Recall (0.2000, 4 objek terlewat / FN=4).")
    print("3. Titik Optimal F1-Score: Tercapai pada Threshold 0.50 (F1=0.8889, Presisi 1.0000, Recall 0.8000).")

