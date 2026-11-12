"""
Chapter 26 - Computer Vision Applications
Skrip 1: Image Loading, Dimensions, and BGR vs RGB Representation
Mendemonstrasikan pemuatan citra menggunakan OpenCV, inspeksi dimensi (H, W, C),
tipe data piksel (uint8), penanganan format kanal BGR bawaan OpenCV vs RGB standar,
serta penanganan galat jika berkas tidak ditemukan.
"""

from typing import Tuple
from pathlib import Path
import numpy as np
import cv2


def load_and_inspect_image(image_path: str) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    Memuat citra dari path, mengekstrak informasi metadata, dan menghasilkan
    dua representasi: format asli OpenCV (BGR) dan format standar Matplotlib (RGB).
    """
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Berkas citra tidak ditemukan pada path: {image_path}")

    # Membaca citra menggunakan OpenCV (bawaan: BGR uint8)
    img_bgr = cv2.imread(str(path))
    if img_bgr is None:
        raise ValueError(f"Gagal mendecode citra dari berkas: {image_path}")

    h, w, c = img_bgr.shape
    metadata = {
        "height": h,
        "width": w,
        "channels": c,
        "dtype": str(img_bgr.dtype),
        "min_pixel": int(img_bgr.min()),
        "max_pixel": int(img_bgr.max()),
        "mean_pixel": float(np.mean(img_bgr))
    }

    # Konversi kanal BGR ke RGB
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    return img_bgr, img_rgb, metadata


def create_sample_color_pattern_image(output_path: Path, width: int = 400, height: int = 300) -> Path:
    """
    Membuat citra sintetis uji yang memuat 3 balok warna primer murni (Merah, Hijau, Biru)
    untuk menguji secara visual dan numerik fenomena inversi kanal BGR vs RGB.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas = np.zeros((height, width, 3), dtype=np.uint8)

    # Balok 1: Merah Murni (BGR: [0, 0, 255])
    canvas[20:100, 20:380] = [0, 0, 255]
    # Balok 2: Hijau Murni (BGR: [0, 255, 0])
    canvas[110:190, 20:380] = [0, 255, 0]
    # Balok 3: Biru Murni (BGR: [255, 0, 0])
    canvas[200:280, 20:380] = [255, 0, 0]

    cv2.imwrite(str(output_path), canvas)
    return output_path


if __name__ == "__main__":
    print("=== Eksplorasi Pemuatan Citra & Representasi BGR vs RGB ===")
    sample_img_path = Path("experiments/sample_color_pattern.png")
    create_sample_color_pattern_image(sample_img_path)
    print(f"Citra Sampel Warna Dibuat di: {sample_img_path}")

    # 1. Pemuatan dan Analisis Dimensi
    bgr_img, rgb_img, meta = load_and_inspect_image(str(sample_img_path))
    print(f"\nMetadata Citra:")
    print(f"  Tinggi (Height)   : {meta['height']} piksel")
    print(f"  Lebar (Width)     : {meta['width']} piksel")
    print(f"  Kanal (Channels)  : {meta['channels']} (3-channel color)")
    print(f"  Tipe Data         : {meta['dtype']} (Rentang [0, 255])")
    print(f"  Intensitas [Min, Max] : [{meta['min_pixel']}, {meta['max_pixel']}]")

    # 2. Pembuktian Inversi Kanal BGR vs RGB
    # Pada Balok 1 (Merah Murni):
    # - Format BGR: Kanal ke-0 (B)=0, Kanal ke-1 (G)=0, Kanal ke-2 (R)=255
    # - Format RGB: Kanal ke-0 (R)=255, Kanal ke-1 (G)=0, Kanal ke-2 (B)=0
    bgr_pixel = bgr_img[50, 200]
    rgb_pixel = rgb_img[50, 200]
    print(f"\nPerbandingan Nilai Piksel Balok Merah (Koordinat y=50, x=200):")
    print(f"  OpenCV BGR Array : [B={bgr_pixel[0]}, G={bgr_pixel[1]}, R={bgr_pixel[2]}]")
    print(f"  Matplotlib RGB   : [R={rgb_pixel[0]}, G={rgb_pixel[1]}, B={rgb_pixel[2]}]")

    assert bgr_pixel[0] == 0 and bgr_pixel[2] == 255, "Nilai piksel BGR tidak sesuai format OpenCV!"
    assert rgb_pixel[0] == 255 and rgb_pixel[2] == 0, "Nilai piksel RGB tidak sesuai format standar!"
    print("  Status Verifikasi: INVERSI KANAL BGR -> RGB TERBUKTI TEPAT!")

    # 3. Penanganan Galat Berkas Hilang
    try:
        load_and_inspect_image("berkas_tidak_ada.jpg")
    except FileNotFoundError as e:
        print(f"\nUji Penanganan Galat: Berhasil menangkap pengecualian -> {e}")

    print("\nSeluruh pengujian pemuatan citra dan representasi warna berhasil diverifikasi!")

