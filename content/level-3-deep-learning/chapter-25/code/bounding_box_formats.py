"""
Chapter 25 - Object Detection & YOLO
Skrip 1: Bounding Box Formats, Coordinate Conversions, and Validation
Mendemonstrasikan representasi koordinat Bounding Box (xyxy vs xywh, absolut vs normalisasi),
konversi antar-format, validasi integritas koordinat, dan clipping ke batas citra.
"""

from typing import Tuple, List, Union
import numpy as np


def validate_bounding_box(box: List[float], img_shape: Tuple[int, int] = None, fmt: str = "xyxy") -> bool:
    """
    Memvalidasi apakah koordinat bounding box valid secara geometris dan berada dalam batas citra.
    fmt: 'xyxy' [x1, y1, x2, y2] atau 'xywh' [xc, yc, w, h]
    """
    if len(box) != 4:
        return False
    
    if fmt == "xyxy":
        x1, y1, x2, y2 = box
        if x2 <= x1 or y2 <= y1:
            return False
        if img_shape:
            h, w = img_shape
            if x1 < 0 or y1 < 0 or x2 > w or y2 > h:
                return False
    elif fmt == "xywh":
        xc, yc, w, h = box
        if w <= 0 or h <= 0:
            return False
        if img_shape:
            img_h, img_w = img_shape
            x1 = xc - w / 2
            y1 = yc - h / 2
            x2 = xc + w / 2
            y2 = yc + h / 2
            if x1 < 0 or y1 < 0 or x2 > img_w or y2 > img_h:
                return False
    else:
        raise ValueError(f"Format tidak dikenali: {fmt}")
    
    return True


def xyxy_to_xywh(box: Union[List[float], np.ndarray]) -> np.ndarray:
    """
    Mengonversi format [x1, y1, x2, y2] (sudut kiri-atas & kanan-bawah)
    menjadi format [x_center, y_center, width, height].
    """
    box = np.array(box, dtype=np.float32)
    x1, y1, x2, y2 = box[..., 0], box[..., 1], box[..., 2], box[..., 3]
    w = x2 - x1
    h = y2 - y1
    xc = x1 + w / 2.0
    yc = y1 + h / 2.0
    return np.stack([xc, yc, w, h], axis=-1)


def xywh_to_xyxy(box: Union[List[float], np.ndarray]) -> np.ndarray:
    """
    Mengonversi format [x_center, y_center, width, height]
    menjadi format [x1, y1, x2, y2] (sudut kiri-atas & kanan-bawah).
    """
    box = np.array(box, dtype=np.float32)
    xc, yc, w, h = box[..., 0], box[..., 1], box[..., 2], box[..., 3]
    x1 = xc - w / 2.0
    y1 = yc - h / 2.0
    x2 = xc + w / 2.0
    y2 = yc + h / 2.0
    return np.stack([x1, y1, x2, y2], axis=-1)


def normalize_coordinates(box: np.ndarray, img_w: int, img_h: int) -> np.ndarray:
    """
    Menskalakan koordinat piksel absolut menjadi rentang relatif [0.0, 1.0].
    Bisa digunakan untuk format xyxy maupun xywh.
    """
    box = np.array(box, dtype=np.float32)
    scaled = box.copy()
    scaled[..., 0] /= img_w
    scaled[..., 1] /= img_h
    scaled[..., 2] /= img_w
    scaled[..., 3] /= img_h
    return np.clip(scaled, 0.0, 1.0)


def denormalize_coordinates(norm_box: np.ndarray, img_w: int, img_h: int) -> np.ndarray:
    """
    Mengembalikan koordinat ternormalisasi [0.0, 1.0] ke nilai piksel absolut.
    """
    norm_box = np.array(norm_box, dtype=np.float32)
    abs_box = norm_box.copy()
    abs_box[..., 0] *= img_w
    abs_box[..., 1] *= img_h
    abs_box[..., 2] *= img_w
    abs_box[..., 3] *= img_h
    return abs_box


def clip_bounding_box(box: List[float], img_w: int, img_h: int) -> List[float]:
    """
    Memotong koordinat [x1, y1, x2, y2] agar tetap berada dalam kanvas citra [0, img_w], [0, img_h].
    """
    x1 = max(0.0, min(float(box[0]), float(img_w)))
    y1 = max(0.0, min(float(box[1]), float(img_h)))
    x2 = max(0.0, min(float(box[2]), float(img_w)))
    y2 = max(0.0, min(float(box[3]), float(img_h)))
    return [x1, y1, x2, y2]


if __name__ == "__main__":
    print("=== Eksplorasi Format dan Validasi Bounding Box ===")
    img_w, img_h = 640, 480
    print(f"Dimensi Citra Uji: {img_w} x {img_h} piksel (Lebar x Tinggi)\n")

    # 1. Kotak Absolut [x1, y1, x2, y2]
    box_xyxy = [100.0, 120.0, 340.0, 360.0]
    print(f"1. Bounding Box Absolut [xyxy]: {box_xyxy}")
    print(f"   Validitas: {validate_bounding_box(box_xyxy, (img_h, img_w), 'xyxy')}")
    
    # 2. Konversi ke xywh
    box_xywh = xyxy_to_xywh(box_xyxy)
    print(f"2. Hasil Konversi ke [xywh]:   {box_xywh.tolist()}")
    print(f"   (Pusat X: {box_xywh[0]:.1f}, Pusat Y: {box_xywh[1]:.1f}, Lebar: {box_xywh[2]:.1f}, Tinggi: {box_xywh[3]:.1f})")

    # 3. Konversi Balik ke xyxy
    recovered_xyxy = xywh_to_xyxy(box_xywh)
    print(f"3. Konversi Balik ke [xyxy]:   {recovered_xyxy.tolist()}")
    diff = np.max(np.abs(np.array(box_xyxy) - recovered_xyxy))
    print(f"   Deviasi Rekonstruksi:       {diff:.2e} (Presisi numerik sempurna!)")

    # 4. Normalisasi ke format YOLO [0.0, 1.0]
    box_yolo = normalize_coordinates(box_xywh, img_w, img_h)
    print(f"4. Format YOLO Ternormalisasi:  {box_yolo.tolist()}")

    # 5. Denormalisasi kembali ke piksel
    recovered_abs = denormalize_coordinates(box_yolo, img_w, img_h)
    print(f"5. Denormalisasi ke Piksel:     {recovered_abs.tolist()}")

    # 6. Pengujian Kotak Tidak Valid & Clipping
    invalid_box = [400.0, 300.0, 200.0, 100.0] # x2 < x1 dan y2 < y1
    print(f"\n6. Uji Kotak Tidak Valid (x2 < x1): {invalid_box}")
    print(f"   Validitas: {validate_bounding_box(invalid_box, (img_h, img_w), 'xyxy')}")

    out_of_bounds = [-20.0, 50.0, 700.0, 520.0]
    clipped = clip_bounding_box(out_of_bounds, img_w, img_h)
    print(f"7. Uji Kotak Melebihi Batas: {out_of_bounds}")
    print(f"   Hasil Pemotongan (Clipped): {clipped}")
    print("\nSeluruh pengujian format dan koordinat bounding box berhasil diverifikasi!")

