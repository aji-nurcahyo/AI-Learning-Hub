"""
Chapter 26 - Computer Vision Applications
Skrip 4: Gradient Operators & Edge Detection (Sobel, Laplacian, and Canny)
Mendemonstrasikan ekstraksi batas kontur citra melalui operator diferensial:
Sobel horizontal dan vertikal, kalkulasi magnitudo gradien, Laplacian of Gaussian,
serta algoritma multi-tahap Canny Edge Detector (Hysteresis Thresholding).
"""

from typing import Dict, Tuple
from pathlib import Path
import numpy as np
import cv2


def compute_gradient_operators(gray_img: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Menghitung vektor gradien dan detektor tepi pada citra grayscale:
    1. Sobel X: Turunan parsial spasial horizontal dI/dx.
    2. Sobel Y: Turunan parsial spasial vertikal dI/dy.
    3. Gradient Magnitude: sqrt(Gx^2 + Gy^2).
    4. Laplacian: Turunan orde kedua d^2I/dx^2 + d^2I/dy^2.
    """
    # 1. Sobel dalam presisi float64 untuk menghindari saturasi negatif
    sobel_x = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize=3)

    # 2. Magnitudo gradien
    magnitude = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
    magnitude_uint8 = np.clip(magnitude, 0, 255).astype(np.uint8)

    # 3. Laplacian (Orde kedua)
    laplacian = cv2.Laplacian(gray_img, cv2.CV_64F, ksize=3)
    laplacian_uint8 = np.clip(np.abs(laplacian), 0, 255).astype(np.uint8)

    return {
        "sobel_x": np.clip(np.abs(sobel_x), 0, 255).astype(np.uint8),
        "sobel_y": np.clip(np.abs(sobel_y), 0, 255).astype(np.uint8),
        "magnitude": magnitude_uint8,
        "laplacian": laplacian_uint8
    }


def apply_canny_edge_detector(
    gray_img: np.ndarray,
    low_thresh: int = 50,
    high_thresh: int = 150
) -> np.ndarray:
    """
    Menerapkan algoritma Canny Edge Detector 5 tahap:
    1. Penghalusan Gaussian (Noise filtering)
    2. Kalkulasi magnitudo dan orientasi gradien
    3. Non-Maximum Suppression tipis 1 piksel
    4. Penapisan ambang ganda (Double Thresholding: T_low dan T_high)
    5. Pelacakan tepi histeresis (Hysteresis Edge Tracking)
    """
    blurred = cv2.GaussianBlur(gray_img, (5, 5), sigmaX=1.4)
    edges = cv2.Canny(blurred, threshold1=low_thresh, threshold2=high_thresh)
    return edges


if __name__ == "__main__":
    print("=== Eksplorasi Operator Gradien & Deteksi Tepi (Edge Detection) ===")

    # 1. Buat citra sintetis objek geometris (persegi panjang putih di tengah kanvas hitam)
    canvas = np.zeros((200, 200), dtype=np.uint8)
    cv2.rectangle(canvas, (40, 40), (160, 160), 255, -1)

    print("1. Citra Sintetis Kotak Putih 120x120 pada Latar Hitam Dibuat.")

    # 2. Hitung Operator Sobel dan Laplacian
    grads = compute_gradient_operators(canvas)

    # Sobel X mendeteksi tepi vertikal (kiri dan kanan kotak)
    # Sobel Y mendeteksi tepi horizontal (atas dan bawah kotak)
    nonzero_x = np.count_nonzero(grads["sobel_x"])
    nonzero_y = np.count_nonzero(grads["sobel_y"])
    print(f"\n2. Respons Operator Gradien:")
    print(f"   Piksel Tepi Sobel X (Garis Vertikal)  : {nonzero_x} piksel aktif")
    print(f"   Piksel Tepi Sobel Y (Garis Horizontal): {nonzero_y} piksel aktif")
    print(f"   Piksel Magnitudo Gabungan             : {np.count_nonzero(grads['magnitude'])} piksel aktif")
    print(f"   Piksel Laplacian (Orde Kedua)         : {np.count_nonzero(grads['laplacian'])} piksel aktif")

    # 3. Deteksi Tepi Canny
    canny_edges = apply_canny_edge_detector(canvas, low_thresh=50, high_thresh=150)
    canny_pixel_count = np.count_nonzero(canny_edges)
    print(f"\n3. Hasil Canny Edge Detector (T_low=50, T_high=150):")
    print(f"   Jumlah Piksel Garis Tepi Tertipis     : {canny_pixel_count} piksel")

    # Keliling teoritis kotak 120x120 adalah ~480 piksel
    print(f"   Estimasi Panjang Garis Tepi Kontur     : ~{canny_pixel_count} piksel (Mendekati keliling 4x120=480 piksel)")
    assert 400 <= canny_pixel_count <= 550, "Hasil Canny detector menyimpang dari perimeter objek!"

    out_path = Path("experiments/canny_edges_sample.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), canny_edges)
    print(f"Citra hasil tepi Canny tersimpan di: {out_path}")
    print("\nSeluruh pengujian deteksi tepi dan operator gradien lulus 100%!")

