"""
Chapter 25 - Object Detection & YOLO
Skrip 5: Bounding Box Visualization Utilities
Menyediakan modul visualisasi untuk menggambar bounding box, label kelas,
skor keyakinan, dan palet warna representatif pada citra menggunakan OpenCV dan Matplotlib.
"""

from typing import List, Dict, Tuple, Optional
import numpy as np
import cv2


# Palet warna kontras tinggi standar Computer Vision (format BGR untuk OpenCV)
COLOR_PALETTE = [
    (0, 255, 0),     # Hijau terang (Kelas 0)
    (255, 0, 0),     # Biru terang (Kelas 1)
    (0, 165, 255),   # Oranye (Kelas 2)
    (0, 255, 255),   # Kuning (Kelas 3)
    (255, 0, 255),   # Magenta (Kelas 4)
    (0, 0, 255),     # Merah (Kelas 5)
    (255, 255, 0),   # Cyan (Kelas 6)
]


def draw_bounding_boxes(
    image: np.ndarray,
    boxes: List[List[float]],
    scores: Optional[List[float]] = None,
    classes: Optional[List[int]] = None,
    class_names: Optional[Dict[int, str]] = None,
    box_format: str = "xyxy",
    line_thickness: int = 2,
    font_scale: float = 0.5
) -> np.ndarray:
    """
    Menggambar bounding box dan kartu label pada salinan citra.
    image: array citra berformat BGR (H, W, 3)
    boxes: daftar koordinat kotak
    box_format: 'xyxy' [x1, y1, x2, y2]
    """
    canvas = image.copy()
    h_img, w_img = canvas.shape[:2]

    for i, box in enumerate(boxes):
        if box_format == "xyxy":
            x1, y1, x2, y2 = map(int, box)
        elif box_format == "xywh":
            xc, yc, w, h = box
            x1 = int(xc - w / 2)
            y1 = int(yc - h / 2)
            x2 = int(xc + w / 2)
            y2 = int(yc + h / 2)
        else:
            raise ValueError(f"Format tidak didukung: {box_format}")

        # Pastikan koordinat berada dalam batas citra
        x1 = max(0, min(x1, w_img - 1))
        y1 = max(0, min(y1, h_img - 1))
        x2 = max(0, min(x2, w_img - 1))
        y2 = max(0, min(y2, h_img - 1))

        # Tentukan kelas dan warna
        cls_id = classes[i] if classes is not None and i < len(classes) else 0
        color = COLOR_PALETTE[cls_id % len(COLOR_PALETTE)]

        # Gambar persegi panjang bounding box
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, line_thickness)

        # Susun teks label
        label_parts = []
        if class_names and cls_id in class_names:
            label_parts.append(class_names[cls_id])
        elif classes is not None:
            label_parts.append(f"Class {cls_id}")
        
        if scores is not None and i < len(scores):
            label_parts.append(f"{scores[i]:.2f}")

        if label_parts:
            label_text = " ".join(label_parts)
            (text_w, text_h), baseline = cv2.getTextSize(
                label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1
            )
            
            # Buat latar belakang kartu teks agar mudah terbaca
            label_y1 = max(0, y1 - text_h - baseline - 4)
            label_y2 = y1
            label_x2 = min(w_img, x1 + text_w + 6)
            
            cv2.rectangle(canvas, (x1, label_y1), (label_x2, label_y2), color, -1)
            cv2.putText(
                canvas,
                label_text,
                (x1 + 3, y1 - baseline - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (0, 0, 0), # Teks hitam kontras
                thickness=1,
                lineType=cv2.LINE_AA
            )

    return canvas


def create_synthetic_detection_scene(width: int = 640, height: int = 480) -> Tuple[np.ndarray, List[List[float]], List[int], Dict[int, str]]:
    """
    Membuat citra sintetis kanvas gelap dengan objek geometris (persegi dan lingkaran)
    sebagai ground truth simulasi untuk pengujian visual tanpa memerlukan aset eksternal.
    """
    img = np.full((height, width, 3), 40, dtype=np.uint8) # Abu-abu gelap (RGB 40, 40, 40)
    
    # Objek 1: Bangunan/Kendaraan Kotak Biru
    cv2.rectangle(img, (80, 120), (280, 320), (220, 100, 50), -1)
    # Objek 2: Rambu/Bola Lingkaran Kuning
    cv2.circle(img, (480, 240), 80, (50, 220, 220), -1)

    gt_boxes = [
        [80.0, 120.0, 280.0, 320.0], # Kotak 1
        [400.0, 160.0, 560.0, 320.0] # Kotak 2 (Bounding box lingkaran: pusat 480, radius 80)
    ]
    gt_classes = [0, 1]
    class_names = {0: "Kendaraan", 1: "Rambu"}

    return img, gt_boxes, gt_classes, class_names


if __name__ == "__main__":
    print("=== Eksplorasi Visualisasi Bounding Box ===")
    img, gt_boxes, gt_classes, class_names = create_synthetic_detection_scene(640, 480)
    print(f"Citra Sintetis Berhasil Dibuat: Shape {img.shape}")
    print(f"Ground Truth Objek 1 (Kendaraan): {gt_boxes[0]}")
    print(f"Ground Truth Objek 2 (Rambu)    : {gt_boxes[1]}")

    # Visualisasikan kotak ground-truth
    vis_img = draw_bounding_boxes(
        img,
        boxes=gt_boxes,
        scores=[1.0, 1.0],
        classes=gt_classes,
        class_names=class_names,
        line_thickness=2
    )

    # Validasi bahwa kanvas terisi dan piksel berubah
    diff = np.sum(np.abs(vis_img.astype(np.int32) - img.astype(np.int32)))
    print(f"Piksel Teranotasi: {diff:,} perbedaan nilai intensitas terdeteksi")
    assert diff > 0, "Visualisasi gagal memodifikasi kanvas citra!"
    print("Modul visualisasi bounding box berhasil diverifikasi!")

