"""
Chapter 26 - Computer Vision Applications
Skrip 2: Color Space Conversions (Grayscale, HSV, and LAB) & Color Thresholding
Mengonversi ruang warna citra, menjelaskan keunggulan ruang warna HSV terhadap variasi pencahayaan,
rumus luminansi Grayscale tertimbang, serta segmentasi berbasis rentang warna (Color Thresholding).
"""

from typing import Tuple
from pathlib import Path
import numpy as np
import cv2
from image_loading import create_sample_color_pattern_image


def convert_color_spaces(img_bgr: np.ndarray) -> dict:
    """
    Mengonversi citra BGR ke tiga ruang warna fundamental Computer Vision:
    1. Grayscale: Reduksi dimensi ke 1 kanal luminansi.
    2. HSV (Hue, Saturation, Value): Pemisahan kromatisitas dari intensitas cahaya.
    3. LAB (Luminance, A-green/red, B-blue/yellow): Ruang warna persepsi visual seragam.
    """
    # 1. Grayscale (Luminansi tertimbang: Y = 0.114*B + 0.587*G + 0.299*R)
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # 2. HSV (Hue: 0-179, Saturation: 0-255, Value: 0-255)
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    # 3. LAB (L: 0-255, A: 0-255, B: 0-255)
    img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)

    return {
        "gray": img_gray,
        "hsv": img_hsv,
        "lab": img_lab
    }


def segment_color_in_hsv(
    img_bgr: np.ndarray,
    lower_hsv: Tuple[int, int, int],
    upper_hsv: Tuple[int, int, int]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Melakukan segmentasi warna pada ruang HSV menggunakan batas bawah dan batas atas.
    Mengembalikan: (mask_biner, citra_terisolasi)
    """
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    lower = np.array(lower_hsv, dtype=np.uint8)
    upper = np.array(upper_hsv, dtype=np.uint8)

    # Buat mask biner: 255 jika berada dalam rentang warna, 0 jika di luar
    mask = cv2.inRange(img_hsv, lower, upper)
    # Terapkan operasi bitwise AND untuk mengekstrak hanya objek berwarna target
    isolated = cv2.bitwise_and(img_bgr, img_bgr, mask=mask)

    return mask, isolated


if __name__ == "__main__":
    print("=== Eksplorasi Konversi Ruang Warna & Segmentasi HSV ===")
    sample_path = Path("experiments/sample_color_pattern.png")
    if not sample_path.exists():
        create_sample_color_pattern_image(sample_path)

    img = cv2.imread(str(sample_path))
    print(f"Citra Masukan Dimuat: Shape {img.shape}")

    # 1. Konversi ke Grayscale, HSV, dan LAB
    converted = convert_color_spaces(img)
    print(f"\nHasil Konversi Ruang Warna:")
    print(f"  Grayscale Shape : {converted['gray'].shape} (1 Kanal Luminansi)")
    print(f"  HSV Shape       : {converted['hsv'].shape} (3 Kanal: Hue, Saturation, Value)")
    print(f"  LAB Shape       : {converted['lab'].shape} (3 Kanal: Lightness, A, B)")

    # 2. Verifikasi Formula Grayscale Tertimbang
    # Piksel Hijau Murni: B=0, G=255, R=0
    # Rumus Grayscale OpenCV: Y = 0.114*0 + 0.587*255 + 0.299*0 ≈ 149.685 -> 150
    green_pixel_bgr = np.array([[[0, 255, 0]]], dtype=np.uint8)
    green_gray = cv2.cvtColor(green_pixel_bgr, cv2.COLOR_BGR2GRAY)[0, 0]
    expected_gray = int(round(0.587 * 255))
    print(f"\nVerifikasi Formula Grayscale pada Piksel Hijau Murni:")
    print(f"  Hasil OpenCV cvtColor : {green_gray}")
    print(f"  Hasil Teoritis (0.587*G): {expected_gray}")
    assert abs(int(green_gray) - expected_gray) <= 1, "Deviasi rumus luminansi melebihi toleransi!"
    print("  Status: FORMULA LUMINANSI TERTIMBANG SESUAI STANDAR ITU-R BT.601!")

    # 3. Segmentasi Warna Hijau pada Ruang HSV
    # Rentang Hue Hijau di OpenCV: ~35 hingga ~85
    lower_green = (35, 100, 100)
    upper_green = (85, 255, 255)
    mask_green, isolated_green = segment_color_in_hsv(img, lower_green, upper_green)

    green_pixels_count = int(np.count_nonzero(mask_green))
    print(f"\nHasil Segmentasi Balok Hijau (HSV Thresholding):")
    print(f"  Piksel Hijau Terdeteksi : {green_pixels_count:,} piksel")
    print(f"  Persentase Kanvas Aktif : {(green_pixels_count / (img.shape[0] * img.shape[1])) * 100:.2f}%")

    out_path = Path("experiments/isolated_green_hsv.png")
    cv2.imwrite(str(out_path), isolated_green)
    print(f"  Citra Terisolasi Disimpan ke: {out_path}")

    assert green_pixels_count > 0, "Segmentasi warna hijau gagal mendeteksi piksel!"
    print("\nSeluruh pengujian konversi ruang warna dan segmentasi HSV berhasil!")

