"""
Chapter 25 - Object Detection & YOLO
Skrip 2: Intersection over Union (IoU) Calculation & Verification
Menghitung luas irisan (Intersection Area), luas gabungan (Union Area),
serta rasio IoU skalar maupun batch tensor multivariat.
"""

from typing import Union, List, Tuple
import numpy as np


def compute_iou_single(box_a: List[float], box_b: List[float]) -> Tuple[float, float, float]:
    """
    Menghitung IoU antara dua bounding box berformat [x1, y1, x2, y2].
    Mengembalikan tuple: (iou, intersection_area, union_area).
    """
    # 1. Koordinat irisan (Intersection)
    x1_inter = max(box_a[0], box_b[0])
    y1_inter = max(box_a[1], box_b[1])
    x2_inter = min(box_a[2], box_b[2])
    y2_inter = min(box_a[3], box_b[3])

    # Luas irisan (0 jika tidak berpotongan)
    w_inter = max(0.0, x2_inter - x1_inter)
    h_inter = max(0.0, y2_inter - y1_inter)
    area_inter = w_inter * h_inter

    # 2. Luas masing-masing kotak
    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])

    # 3. Luas gabungan (Union)
    area_union = area_a + area_b - area_inter

    # 4. Rasio IoU
    if area_union <= 0.0:
        return 0.0, 0.0, 0.0
    
    iou = area_inter / area_union
    return float(iou), float(area_inter), float(area_union)


def compute_iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """
    Menghitung matriks IoU bervektor antara N kotak dan M kotak.
    boxes_a: shape (N, 4) dalam format [x1, y1, x2, y2]
    boxes_b: shape (M, 4) dalam format [x1, y1, x2, y2]
    Output: shape (N, M) berisi nilai IoU setiap pasangan.
    """
    boxes_a = np.asarray(boxes_a, dtype=np.float32)
    boxes_b = np.asarray(boxes_b, dtype=np.float32)

    # Area masing-masing kotak
    area_a = (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1]) # (N,)
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1]) # (M,)

    # Perluasan dimensi untuk penyiaran (broadcasting): (N, 1) vs (1, M)
    x1 = np.maximum(boxes_a[:, None, 0], boxes_b[None, :, 0])
    y1 = np.maximum(boxes_a[:, None, 1], boxes_b[None, :, 1])
    x2 = np.minimum(boxes_a[:, None, 2], boxes_b[None, :, 2])
    y2 = np.minimum(boxes_a[:, None, 3], boxes_b[None, :, 3])

    w_inter = np.maximum(0.0, x2 - x1)
    h_inter = np.maximum(0.0, y2 - y1)
    area_inter = w_inter * h_inter # (N, M)

    area_union = area_a[:, None] + area_b[None, :] - area_inter # (N, M)
    
    # Hindari pembagian nol
    iou_matrix = np.where(area_union > 0, area_inter / area_union, 0.0)
    return iou_matrix


if __name__ == "__main__":
    print("=== Eksplorasi & Verifikasi Numerik Intersection over Union (IoU) ===")

    # Contoh Terhitung Manual (Worked Example)
    # Kotak A: [100, 100, 300, 300] -> Lebar 200, Tinggi 200 -> Luas = 40.000
    # Kotak B: [200, 200, 400, 400] -> Lebar 200, Tinggi 200 -> Luas = 40.000
    # Irisan:  [200, 200, 300, 300] -> Lebar 100, Tinggi 100 -> Luas = 10.000
    # Gabungan: 40.000 + 40.000 - 10.000 = 70.000
    # IoU: 10.000 / 70.000 = 1 / 7 ≈ 0.142857
    box_a = [100.0, 100.0, 300.0, 300.0]
    box_b = [200.0, 200.0, 400.0, 400.0]

    iou_val, inter_area, union_area = compute_iou_single(box_a, box_b)
    print("\n1. Studi Kasus Parsial Bertindih:")
    print(f"   Kotak A: {box_a}")
    print(f"   Kotak B: {box_b}")
    print(f"   Luas Irisan (Intersection Area) : {inter_area:,.1f}")
    print(f"   Luas Gabungan (Union Area)      : {union_area:,.1f}")
    print(f"   Nilai IoU Terhitung             : {iou_val:.6f}")
    print(f"   Nilai Teoritis (1/7)            : {1.0 / 7.0:.6f}")
    assert abs(iou_val - (1.0 / 7.0)) < 1e-6, "Galat numerik pada perhitungan IoU!"
    print("   Status Verifikasi: COCOK DENGAN PREDIKSI TEORITIS (Delta < 1e-6)")

    # 2. Kotak Identik (100% overlap)
    iou_identical, _, _ = compute_iou_single(box_a, box_a)
    print(f"\n2. Kotak Identik: IoU = {iou_identical:.4f} (Harus 1.0000)")
    assert abs(iou_identical - 1.0) < 1e-6

    # 3. Kotak Terpisah Total (Disjoint)
    box_c = [500.0, 500.0, 600.0, 600.0]
    iou_disjoint, _, _ = compute_iou_single(box_a, box_c)
    print(f"3. Kotak Terpisah: IoU = {iou_disjoint:.4f} (Harus 0.0000)")
    assert abs(iou_disjoint - 0.0) < 1e-6

    # 4. Kotak Bersarang (Nested)
    # Kotak D: [150, 150, 250, 250] -> Luas = 100 x 100 = 10.000
    # Irisan = 10.000, Gabungan = 40.000 -> IoU = 10.000 / 40.000 = 0.2500
    box_d = [150.0, 150.0, 250.0, 250.0]
    iou_nested, _, _ = compute_iou_single(box_a, box_d)
    print(f"4. Kotak Bersarang: IoU = {iou_nested:.4f} (Harus 0.2500)")
    assert abs(iou_nested - 0.25) < 1e-6

    # 5. Uji Batch Matriks IoU
    query_boxes = np.array([box_a, box_c])
    target_boxes = np.array([box_a, box_b, box_d])
    iou_mat = compute_iou_matrix(query_boxes, target_boxes)
    print("\n5. Matriks IoU Vektor (2 Kueri x 3 Target):")
    print(iou_mat)
    assert iou_mat.shape == (2, 3)
    print("\nSeluruh perhitungan dan pengujian matematis IoU lulus 100%!")

