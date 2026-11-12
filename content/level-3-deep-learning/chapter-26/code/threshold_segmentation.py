"""
Chapter 26 - Computer Vision Applications
Skrip 6: Threshold-Based Segmentation & Connected Component Analysis
Mendemonstrasikan segmentasi citra berbasis ambang batas (Global, Otsu, dan Adaptive Thresholding),
mengatasi variasi pencahayaan non-seragam, serta mengekstrak instans objek biner,
koordinat bounding box, dan titik pusat massa (Centroid) melalui Connected Components with Stats.
"""

from typing import Dict, List, Tuple
from pathlib import Path
import numpy as np
import cv2


def apply_segmentation_thresholds(gray_img: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Menerapkan tiga metode penentuan ambang batas (Thresholding) pada citra grayscale:
    1. Global Thresholding: Ambang batas statis tetap (misal T = 127).
    2. Otsu's Thresholding: Penentuan ambang batas optimal otomatis melalui minimisasi variansi intra-kelas.
    3. Adaptive Gaussian: Ambang batas lokal adaptif untuk menangani gradien pencahayaan tak seragam.
    """
    # 1. Global Threshold
    _, global_thresh = cv2.threshold(gray_img, 127, 255, cv2.THRESH_BINARY)

    # 2. Otsu Threshold (Menghitung ambang optimal otomatis)
    otsu_val, otsu_thresh = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 3. Adaptive Threshold (Ukuran blok lokal 11x11, konstanta C = 2)
    adaptive_thresh = cv2.adaptiveThreshold(
        gray_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )

    return {
        "global": global_thresh,
        "otsu": otsu_thresh,
        "adaptive": adaptive_thresh,
        "otsu_optimal_val": float(otsu_val)
    }


def extract_connected_components(binary_mask: np.ndarray, min_area: int = 50) -> List[Dict[str, any]]:
    """
    Mengekstrak setiap objek terhubung (Connected Components) dari mask biner.
    Mengembalikan daftar kamus yang memuat label ID, luas area, bounding box [x, y, w, h], dan centroid (cx, cy).
    """
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
    objects = []

    # Indeks 0 adalah latar belakang (Background), objek dimulai dari indeks 1
    for i in range(1, num_labels):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < min_area:
            continue # Tapis derau bintik kecil
        
        x = int(stats[i, cv2.CC_STAT_LEFT])
        y = int(stats[i, cv2.CC_STAT_TOP])
        w = int(stats[i, cv2.CC_STAT_WIDTH])
        h = int(stats[i, cv2.CC_STAT_HEIGHT])
        cx, cy = centroids[i]

        objects.append({
            "label_id": i,
            "area": area,
            "bbox_xywh": [x, y, w, h],
            "bbox_xyxy": [x, y, x + w, y + h],
            "centroid": (float(cx), float(cy))
        })

    return objects


if __name__ == "__main__":
    print("=== Eksplorasi Segmentasi Berbasis Thresholding & Connected Components ===")

    # 1. Buat citra grayscale dengan gradien pencahayaan miring (Non-uniform illumination)
    # Latar kiri terang (180), latar kanan gelap (40)
    h, w = 240, 320
    grad = np.linspace(180, 40, w, dtype=np.uint8)
    background = np.tile(grad, (h, 1))

    # Letakkan dua objek kontras tinggi (Objek A di kiri gelap 50, Objek B di kanan terang 220)
    canvas = background.copy()
    cv2.circle(canvas, (80, 120), 30, 40, -1)   # Objek A (Lingkaran)
    cv2.rectangle(canvas, (220, 80), (280, 160), 220, -1) # Objek B (Kotak)

    print("1. Citra Uji dengan Gradien Pencahayaan Non-Seragam Berhasil Dibuat.")

    # 2. Jalankan Segmentasi Thresholding
    thresh_results = apply_segmentation_thresholds(canvas)
    print(f"\nHasil Evaluasi Thresholding:")
    print(f"  Nilai Ambang Batas Optimal Otsu : T = {thresh_results['otsu_optimal_val']:.1f}")

    # 3. Ekstraksi Komponen Terhubung pada Mask Otsu
    components = extract_connected_components(thresh_results["otsu"], min_area=100)
    print(f"\nHasil Ekstraksi Komponen Terhubung (Connected Components):")
    print(f"  Total Objek Terdeteksi (Area > 100 piksel): {len(components)}")

    for obj in components:
        print(f"  -> Objek {obj['label_id']}: Luas = {obj['area']} piksel | BBox = {obj['bbox_xywh']} | Centroid = ({obj['centroid'][0]:.1f}, {obj['centroid'][1]:.1f})")

    assert len(components) >= 1, "Gagal mengekstrak objek dari mask biner!"
    print("\nSegmentasi dan analisis komponen terhubung berhasil diverifikasi!")

