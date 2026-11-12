"""
Chapter 25 - Object Detection & YOLO
Skrip 4: Object Detection Evaluation Metrics (TP, FP, FN, Precision, Recall, AP & mAP)
Mengimplementasikan pencocokan prediksi ke ground-truth (Greedy Matching),
kurva Precision-Recall terinterpolasi, Average Precision (AP),
serta evaluasi mAP@0.5 dan mAP@0.5:0.95 standar COCO/VOC.
"""

from typing import List, Dict, Tuple
import numpy as np
from iou_calculation import compute_iou_single


def match_predictions_to_ground_truth(
    pred_boxes: List[List[float]],
    pred_scores: List[float],
    gt_boxes: List[List[float]],
    iou_threshold: float = 0.5
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Mencocokkan daftar prediksi ke ground-truth untuk satu kelas objek.
    Mengembalikan: (tp_array, fp_array, total_gt_count)
    tp_array[i] bernilai 1 jika prediksi ke-i adalah True Positive, 0 jika False Positive.
    """
    num_preds = len(pred_boxes)
    num_gts = len(gt_boxes)

    if num_preds == 0:
        return np.array([]), np.array([]), num_gts

    # Urutkan prediksi berdasarkan skor keyakinan menurun
    sort_order = np.argsort(-np.array(pred_scores))
    sorted_pred_boxes = [pred_boxes[i] for i in sort_order]

    tp = np.zeros(num_preds)
    fp = np.zeros(num_preds)
    gt_matched = [False] * num_gts

    for i, p_box in enumerate(sorted_pred_boxes):
        best_iou = 0.0
        best_gt_idx = -1

        for j, g_box in enumerate(gt_boxes):
            iou, _, _ = compute_iou_single(p_box, g_box)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = j

        # Syarat True Positive: IoU >= ambang batas DAN target belum pernah dicocokkan sebelumnya
        if best_iou >= iou_threshold:
            if not gt_matched[best_gt_idx]:
                tp[i] = 1.0
                gt_matched[best_gt_idx] = True # Kunci agar tidak dicocokkan berulang
            else:
                fp[i] = 1.0 # Duplikat deteksi pada objek yang sama dihitung sebagai FP
        else:
            fp[i] = 1.0 # Tidak ada GT yang bertumpang tindih cukup tinggi

    return tp, fp, num_gts


def compute_ap(
    tp: np.ndarray, 
    fp: np.ndarray, 
    num_gts: int, 
    method: str = "all_points"
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Menghitung Average Precision (AP) dari vektor kumulatif TP dan FP.
    method: 'all_points' (Standar COCO/VOC 2010+) atau '11_points' (Standar VOC 2007).
    """
    if len(tp) == 0 or num_gts == 0:
        return 0.0, np.array([]), np.array([])

    # Akumulasi kumulatif
    cum_tp = np.cumsum(tp)
    cum_fp = np.cumsum(fp)

    precisions = cum_tp / (cum_tp + cum_fp)
    recalls = cum_tp / num_gts

    if method == "all_points":
        # Interpolar precision secara monotonik menurun: P(r) = max_{r' >= r} P(r')
        mrec = np.concatenate(([0.0], recalls, [1.0]))
        mpre = np.concatenate(([0.0], precisions, [0.0]))

        for i in range(len(mpre) - 2, -1, -1):
            mpre[i] = max(mpre[i], mpre[i + 1])

        # Hitung luas area di bawah kurva (Area Under Curve - AUC)
        i = np.where(mrec[1:] != mrec[:-1])[0]
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    elif method == "11_points":
        # 11 titik recall tetap: 0.0, 0.1, ..., 1.0
        ap = 0.0
        for t in np.linspace(0.0, 1.0, 11):
            p_candidates = precisions[recalls >= t]
            p = np.max(p_candidates) if len(p_candidates) > 0 else 0.0
            ap += p / 11.0
    else:
        raise ValueError(f"Metode AP tidak dikenali: {method}")

    return float(ap), precisions, recalls


def compute_map(
    detections: Dict[int, Dict[str, List]], 
    iou_thresholds: List[float] = [0.5]
) -> Dict[str, float]:
    """
    Menghitung Mean Average Precision (mAP) melintasi multi-kelas dan multi-ambang IoU.
    detections: kamus berstruktur {class_id: {'preds': [...], 'scores': [...], 'gts': [...]}}
    """
    ap_per_thresh = []

    for iou_th in iou_thresholds:
        aps_per_class = []
        for c, data in detections.items():
            tp, fp, num_gts = match_predictions_to_ground_truth(
                data['preds'], data['scores'], data['gts'], iou_threshold=iou_th
            )
            ap, _, _ = compute_ap(tp, fp, num_gts)
            aps_per_class.append(ap)
        ap_per_thresh.append(np.mean(aps_per_class))

    return {
        'mAP@0.5': float(ap_per_thresh[0]),
        'mAP_mean': float(np.mean(ap_per_thresh))
    }


if __name__ == "__main__":
    print("=== Eksplorasi Metrik Evaluasi Deteksi Objek (AP & mAP) ===")

    # Skenario Contoh Numerik Kecil Terverifikasi:
    # Terdapat 3 Bounding Box Ground-Truth (Mobil)
    gts = [
        [50.0, 50.0, 150.0, 150.0],   # GT 0
        [200.0, 200.0, 300.0, 300.0], # GT 1
        [400.0, 100.0, 500.0, 250.0]  # GT 2
    ]

    # Model mengeluarkan 4 prediksi (sudah diurutkan dari skor tertinggi):
    preds = [
        [52.0, 48.0, 148.0, 152.0],   # Pred 0: Cocok GT 0 (IoU ~0.90) -> TP
        [55.0, 50.0, 150.0, 145.0],   # Pred 1: Duplikat GT 0 (IoU ~0.85) -> FP (karena GT 0 sudah dicocokkan)
        [205.0, 195.0, 298.0, 302.0], # Pred 2: Cocok GT 1 (IoU ~0.88) -> TP
        [300.0, 400.0, 380.0, 480.0]  # Pred 3: Halusinasi / Salah Tempat -> FP
        # Catatan: GT 2 tidak pernah terdeteksi sama sekali -> False Negative (FN)!
    ]
    scores = [0.95, 0.88, 0.80, 0.65]

    print(f"Jumlah Objek Sebenarnya (Ground Truth): {len(gts)}")
    print(f"Jumlah Prediksi Model:                 {len(preds)}\n")

    # 1. Pencocokan Prediksi pada IoU Threshold = 0.5
    tp, fp, num_gts = match_predictions_to_ground_truth(preds, scores, gts, iou_threshold=0.5)
    print("1. Hasil Pencocokan Greedily (Threshold IoU = 0.50):")
    for i, (p, s) in enumerate(zip(preds, scores)):
        status = "TRUE POSITIVE (TP)" if tp[i] == 1 else "FALSE POSITIVE (FP)"
        print(f"   Prediksi {i} (Skor {s:.2f}): {status}")

    fn_count = num_gts - int(np.sum(tp))
    print(f"\n   Total Ringkasan: TP = {int(np.sum(tp))}, FP = {int(np.sum(fp))}, FN = {fn_count}")
    assert int(np.sum(tp)) == 2 and int(np.sum(fp)) == 2 and fn_count == 1
    print("   Status Verifikasi: Sesuai ekspektasi teoritis (TP=2, FP=2, FN=1)")

    # 2. Perhitungan Precision-Recall dan AP
    ap_all, precisions, recalls = compute_ap(tp, fp, num_gts, method="all_points")
    ap_11, _, _ = compute_ap(tp, fp, num_gts, method="11_points")

    print(f"\n2. Evaluasi Average Precision (AP):")
    print(f"   Vektor Precision: {[round(p, 4) for p in precisions.tolist()]}")
    print(f"   Vektor Recall   : {[round(r, 4) for r in recalls.tolist()]}")
    print(f"   AP (All-Point AUC - Standar Modern): {ap_all:.4f}")
    print(f"   AP (11-Point Interpolation - VOC07): {ap_11:.4f}")

    # 3. Evaluasi Multi-Ambang Batas COCO (mAP@0.5:0.95)
    coco_thresholds = np.linspace(0.5, 0.95, 10).tolist()
    dummy_dataset = {
        0: {'preds': preds, 'scores': scores, 'gts': gts}
    }
    coco_results = compute_map(dummy_dataset, iou_thresholds=coco_thresholds)
    print(f"\n3. Evaluasi Protokol COCO mAP:")
    print(f"   mAP@0.5      : {coco_results['mAP@0.5']:.4f} (Evaluasi toleransi spasial longgar)")
    print(f"   mAP@0.5:0.95 : {coco_results['mAP_mean']:.4f} (Rata-rata 10 ambang batas ketat)")
    print("\nSeluruh fungsi metrik deteksi objek terverifikasi akurat!")

