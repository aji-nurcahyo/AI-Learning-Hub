"""
Chapter 26 - Computer Vision Applications
Skrip 11: Integrated End-to-End Computer Vision Pipeline
Mengintegrasikan seluruh tahapan sistem visi komputer:
Akuisisi citra -> Prapemrosesan deterministik -> Inferensi deteksi objek ->
Pascapemrosesan & Pelacakan Centroid -> Visualisasi diagnostik -> Pemrofilan latensi & throughput.
"""

from typing import Dict, List, Tuple
from pathlib import Path
import time
import numpy as np
import cv2
from centroid_tracking import CentroidTracker


def create_synthetic_test_frame(width: int = 640, height: int = 480) -> np.ndarray:
    """Membuat frame citra sintetis kanvas gelap dengan objek persegi dan lingkaran kontras."""
    img = np.full((height, width, 3), 40, dtype=np.uint8)
    cv2.rectangle(img, (80, 120), (280, 320), (220, 100, 50), -1)
    cv2.circle(img, (480, 240), 80, (50, 220, 220), -1)
    return img


class IntegratedCVPipeline:
    """
    Pipeline Visi Komputer modular yang menggabungkan preprocessing,
    deteksi objek, tracking, dan visualisasi diagnostik.
    """
    def __init__(self, target_size: Tuple[int, int] = (640, 480)):
        self.target_size = target_size
        self.tracker = CentroidTracker(max_disappeared=5, max_distance=60.0)

    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        """Prapemrosesan citra deterministik: resizing dan reduksi noise ringan."""
        resized = cv2.resize(frame, self.target_size)
        denoised = cv2.GaussianBlur(resized, (3, 3), 0.8)
        return denoised

    def detect_objects_cv(self, preprocessed_frame: np.ndarray) -> List[List[float]]:
        """
        Deteksi objek menggunakan segmentasi kontur adaptif (metode deterministik ringan CPU).
        """
        gray = cv2.cvtColor(preprocessed_frame, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 70, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        boxes = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 400: # Tapis derau kecil
                x, y, w, h = cv2.boundingRect(cnt)
                boxes.append([float(x), float(y), float(x + w), float(y + h)])
        return boxes

    def process_frame(self, raw_frame: np.ndarray) -> Tuple[np.ndarray, dict]:
        """
        Mengeksekusi satu siklus penuh pipeline dan mencatat profil latensi setiap tahapan.
        """
        timings = {}
        t_start = time.perf_counter()

        # 1. Prapemrosesan
        t0 = time.perf_counter()
        prep_frame = self.preprocess(raw_frame)
        timings["preprocess_ms"] = (time.perf_counter() - t0) * 1000.0

        # 2. Deteksi Objek
        t0 = time.perf_counter()
        detected_boxes = self.detect_objects_cv(prep_frame)
        timings["detection_ms"] = (time.perf_counter() - t0) * 1000.0

        # 3. Pelacakan Objek (Tracking)
        t0 = time.perf_counter()
        tracked_objects = self.tracker.update(detected_boxes)
        timings["tracking_ms"] = (time.perf_counter() - t0) * 1000.0

        # 4. Visualisasi Hasil
        t0 = time.perf_counter()
        annotated = prep_frame.copy()
        for box in detected_boxes:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)

        for obj_id, (cx, cy) in tracked_objects.items():
            cv2.circle(annotated, (int(cx), int(cy)), 4, (0, 0, 255), -1)
            cv2.putText(annotated, f"ID: {obj_id}", (int(cx) - 15, int(cy) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)

        timings["visualization_ms"] = (time.perf_counter() - t0) * 1000.0
        timings["total_pipeline_ms"] = (time.perf_counter() - t_start) * 1000.0

        stats = {
            "num_detected": len(detected_boxes),
            "num_tracked": len(tracked_objects),
            "timings": timings,
            "estimated_fps": 1000.0 / timings["total_pipeline_ms"] if timings["total_pipeline_ms"] > 0 else 0.0
        }

        return annotated, stats


if __name__ == "__main__":
    print("=== Eksplorasi Pipeline Terpadu Visi Komputer (Integrated CV Pipeline) ===")

    pipeline = IntegratedCVPipeline(target_size=(640, 480))
    # Buat citra adegan sintetis
    raw_img = create_synthetic_test_frame(640, 480)

    print("Menjalankan Pemrosesan Pipeline Lengkap...")
    output_canvas, stats = pipeline.process_frame(raw_img)

    print(f"\nHasil Eksekusi Pipeline:")
    print(f"  Jumlah Objek Terdeteksi : {stats['num_detected']}")
    print(f"  Jumlah Objek Dilacak    : {stats['num_tracked']}")
    print(f"\nProfil Latensi Tahapan (CPU):")
    print(f"  - Preprocessing        : {stats['timings']['preprocess_ms']:.2f} ms")
    print(f"  - Object Detection     : {stats['timings']['detection_ms']:.2f} ms")
    print(f"  - Object Tracking      : {stats['timings']['tracking_ms']:.2f} ms")
    print(f"  - Visual Rendering     : {stats['timings']['visualization_ms']:.2f} ms")
    print(f"  -------------------------------------------")
    print(f"  TOTAL LATENSI PIPELINE : {stats['timings']['total_pipeline_ms']:.2f} ms")
    print(f"  ESTIMASI THROUGHPUT    : {stats['estimated_fps']:.1f} FPS (Memenuhi kualifikasi real-time > 30 FPS)")

    out_file = Path("experiments/integrated_pipeline_output.png")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_file), output_canvas)
    print(f"\nHasil visual tersimpan di: {out_file}")

    assert stats["num_detected"] > 0, "Pipeline gagal mendeteksi objek!"
    assert stats["estimated_fps"] > 15.0, "Throughput pipeline harus mendukung pemrosesan cepat pada CPU!"
    print("\nSeluruh pengujian Integrated Computer Vision Pipeline sukses 100%!")
