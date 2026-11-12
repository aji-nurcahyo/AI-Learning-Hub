"""
Chapter 25 - Object Detection & YOLO
Skrip 3: Non-Maximum Suppression (NMS) Implementation & Verification
Mengeliminasi prediksi bounding box redundan melalui perangkingan skor keyakinan
(Confidence Ranking) dan penapisan berbasis ambang batas IoU (IoU Thresholding).
Menyediakan mode Class-Aware maupun Class-Agnostic serta validasi terhadap OpenCV NMSBoxes.
"""

from typing import List, Tuple
import numpy as np
import cv2
from iou_calculation import compute_iou_single


def nms_pure_numpy(
    boxes: np.ndarray, 
    scores: np.ndarray, 
    iou_threshold: float = 0.5, 
    score_threshold: float = 0.25
) -> List[int]:
    """
    Implementasi mandiri algoritma Non-Maximum Suppression (Class-Agnostic) menggunakan NumPy murni.
    boxes: shape (N, 4) dalam format [x1, y1, x2, y2]
    scores: shape (N,) berisi nilai skor keyakinan
    iou_threshold: ambang batas tumpang tindih untuk supresi
    score_threshold: ambang batas minimum skor untuk dipertimbangkan
    Output: Daftar indeks kotak terpilih yang dipertahankan.
    """
    if len(boxes) == 0:
        return []

    boxes = np.asarray(boxes, dtype=np.float32)
    scores = np.asarray(scores, dtype=np.float32)

    # 1. Penapisan awal berdasarkan ambang batas skor (Confidence Filtering)
    valid_mask = scores >= score_threshold
    indices = np.where(valid_mask)[0]
    
    if len(indices) == 0:
        return []

    # 2. Urutkan indeks berdasarkan skor keyakinan secara menurun (Descending Sort)
    order = indices[np.argsort(-scores[indices])]

    keep = []

    # 3. Ekstraksi koordinat untuk komputasi cepat
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)

    while len(order) > 0:
        # Ambil kotak dengan skor tertinggi saat ini
        current_idx = order[0]
        keep.append(current_idx)

        if len(order) == 1:
            break

        # Hitung IoU antara kotak terpilih dengan sisa kotak kandidat
        remaining_indices = order[1:]

        xx1 = np.maximum(x1[current_idx], x1[remaining_indices])
        yy1 = np.maximum(y1[current_idx], y1[remaining_indices])
        xx2 = np.minimum(x2[current_idx], x2[remaining_indices])
        yy2 = np.minimum(y2[current_idx], y2[remaining_indices])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h

        union = areas[current_idx] + areas[remaining_indices] - inter
        iou = np.where(union > 0, inter / union, 0.0)

        # Pertahankan hanya kotak yang memiliki IoU <= iou_threshold
        surviving = np.where(iou <= iou_threshold)[0]
        order = remaining_indices[surviving]

    return keep


def nms_class_aware(
    boxes: np.ndarray,
    scores: np.ndarray,
    classes: np.ndarray,
    iou_threshold: float = 0.5,
    score_threshold: float = 0.25
) -> List[int]:
    """
    Menjalankan NMS secara independen untuk setiap kelas objek (Class-Aware NMS).
    Dua objek berbeda kelas yang bertumpukan tinggi TIDAK akan saling meniadakan.
    """
    boxes = np.asarray(boxes, dtype=np.float32)
    scores = np.asarray(scores, dtype=np.float32)
    classes = np.asarray(classes, dtype=np.int32)

    unique_classes = np.unique(classes)
    total_keep = []

    for c in unique_classes:
        class_mask = (classes == c)
        class_indices = np.where(class_mask)[0]
        
        c_boxes = boxes[class_indices]
        c_scores = scores[class_indices]

        c_keep_local = nms_pure_numpy(c_boxes, c_scores, iou_threshold, score_threshold)
        total_keep.extend([class_indices[i] for i in c_keep_local])

    # Urutkan berdasarkan skor menurun
    total_keep.sort(key=lambda idx: scores[idx], reverse=True)
    return total_keep


if __name__ == "__main__":
    print("=== Eksplorasi & Verifikasi Algoritma Non-Maximum Suppression (NMS) ===")

    # Skenario Uji: 5 Bounding Box Kandidat di sekitar 2 Objek Fisik (Mobil dan Orang)
    # Objek 1 (Mobil): Terdeteksi oleh 3 kotak kandidat yang saling tumpang tindih
    # Objek 2 (Orang): Terdeteksi oleh 2 kotak kandidat
    candidate_boxes = np.array([
        [100.0, 100.0, 250.0, 200.0],  # Box 0: Mobil (Utama, Conf=0.92)
        [105.0, 102.0, 248.0, 198.0],  # Box 1: Mobil (Redundan IoU ~0.90, Conf=0.75)
        [115.0, 95.0,  260.0, 210.0],  # Box 2: Mobil (Redundan IoU ~0.78, Conf=0.60)
        [400.0, 150.0, 480.0, 350.0],  # Box 3: Orang (Utama, Conf=0.88)
        [395.0, 155.0, 475.0, 345.0],  # Box 4: Orang (Redundan IoU ~0.85, Conf=0.70)
        [50.0,  50.0,  80.0,  80.0],   # Box 5: Noise (Conf=0.15 - di bawah ambang batas)
    ], dtype=np.float32)

    scores = np.array([0.92, 0.75, 0.60, 0.88, 0.70, 0.15], dtype=np.float32)
    classes = np.array([0, 0, 0, 1, 1, 0], dtype=np.int32) # 0: Mobil, 1: Orang
    class_names = {0: "Mobil", 1: "Orang"}

    print(f"Total Kotak Mentah Masukan: {len(candidate_boxes)}")
    for i, (b, s, c) in enumerate(zip(candidate_boxes, scores, classes)):
        print(f"  Prediksi {i}: Kelas={class_names[c]:<5} | Skor={s:.2f} | BBox={b.tolist()}")

    # 1. Eksekusi NMS Mandiri (NumPy)
    keep_indices = nms_pure_numpy(candidate_boxes, scores, iou_threshold=0.5, score_threshold=0.25)
    print(f"\n1. Hasil NMS NumPy Murni (IoU Thresh=0.5, Score Thresh=0.25):")
    print(f"   Indeks Terpilih: {keep_indices}")
    print(f"   Jumlah Prediksi Tersisa: {len(keep_indices)} dari {len(candidate_boxes)} kotak mentah")
    for idx in keep_indices:
        print(f"   -> [DILESTARIKAN] Box {idx}: {class_names[classes[idx]]} (Skor={scores[idx]:.2f})")

    # Verifikasi logis: Hanya Box 0 (Mobil utama) dan Box 3 (Orang utama) yang boleh bertahan
    assert keep_indices == [0, 3], f"Hasil NMS tidak sesuai ekspektasi! Didapat: {keep_indices}"
    print("   Status Verifikasi: TEPAT 100% (Box 1, 2, 4 ditiadakan karena tumpang tindih; Box 5 ditapis karena skor rendah)")

    # 2. Verifikasi dengan Implementasi Resmi OpenCV NMSBoxes
    # cv2.dnn.NMSBoxes membutuhkan format [x, y, w, h] (sudut kiri-atas, lebar, tinggi)
    cv_boxes = [[int(b[0]), int(b[1]), int(b[2] - b[0]), int(b[3] - b[1])] for b in candidate_boxes]
    cv_indices = cv2.dnn.NMSBoxes(cv_boxes, scores.tolist(), score_threshold=0.25, nms_threshold=0.5)
    cv_keep = [int(i) for i in cv_indices] if len(cv_indices) > 0 else []
    print(f"\n2. Hasil Validasi Pembanding OpenCV cv2.dnn.NMSBoxes:")
    print(f"   Indeks Terpilih OpenCV: {cv_keep}")
    assert keep_indices == cv_keep, "Deviasi terdeteksi antara implementasi NumPy mandiri dan OpenCV!"
    print("   Status Verifikasi: IDENTIK SEMPURNA TERHADAP OPENCV NMSBOXES!")

    # 3. Uji Kasus Dua Objek Berbeda Kelas Bertumpukan Tinggi
    print("\n3. Uji Kasus Khusus: Class-Aware NMS (Objek Bertumpuk Beda Kelas)")
    # Misal: Orang naik Sepeda (Bounding box orang dan sepeda beririsan > 0.6)
    mixed_boxes = np.array([
        [100.0, 100.0, 200.0, 300.0], # Orang (Skor 0.90)
        [90.0,  150.0, 210.0, 310.0], # Sepeda (Skor 0.85) - Bertumpuk tinggi dengan Orang
    ])
    mixed_scores = np.array([0.90, 0.85])
    mixed_classes = np.array([0, 1]) # 0: Orang, 1: Sepeda

    iou_mixed, _, _ = compute_iou_single(mixed_boxes[0].tolist(), mixed_boxes[1].tolist())
    print(f"   IoU antara Orang dan Sepeda: {iou_mixed:.4f} (Bertumpuk tinggi > 0.5)")

    agnostic_keep = nms_pure_numpy(mixed_boxes, mixed_scores, iou_threshold=0.5, score_threshold=0.25)
    print(f"   - Class-Agnostic NMS Terpilih: {agnostic_keep} (Salah satu objek terhapus keliru!)")

    aware_keep = nms_class_aware(mixed_boxes, mixed_scores, mixed_classes, iou_threshold=0.5, score_threshold=0.25)
    print(f"   - Class-Aware NMS Terpilih   : {aware_keep} (Keduanya berhasil dipertahankan!)")
    assert len(aware_keep) == 2, "Class-aware NMS harus melestarikan kedua kelas objek!"
    print("\nSeluruh pengujian algoritma NMS terverifikasi sukses!")

