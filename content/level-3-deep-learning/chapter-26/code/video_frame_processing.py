"""
Chapter 26 - Computer Vision Applications
Skrip 9: Video Frame Stream Processing & Latency Profiling
Mendemonstrasikan pemrosesan aliran bingkai video (Video Stream Processing):
pembangkitan klip video sintetis, iterasi frame demi frame, penghitungan FPS pemrosesan aktual,
pencatatan stempel waktu (timestamps), serta analisis latensi pemrosesan tanpa memerlukan webcam fisik.
"""

from typing import Tuple, List
from pathlib import Path
import time
import numpy as np
import cv2


def generate_synthetic_video_clip(
    output_path: Path,
    num_frames: int = 30,
    width: int = 320,
    height: int = 240,
    fps: int = 15
) -> Path:
    """
    Membangkitkan klip video sintetis singkat yang memuat lingkaran bergerak
    untuk keperluan pengujian pipeline video secara offline tanpa kamera fisik.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Gunakan fourcc MJPG atau mp4v
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    for i in range(num_frames):
        frame = np.full((height, width, 3), 40, dtype=np.uint8)
        # Objek bergerak melintasi layar: x bertambah, y sinusoidal
        cx = int(40 + (i / num_frames) * (width - 80))
        cy = int(height / 2 + 30 * np.sin(i * 0.3))
        # Gambar bola hijau bergerak
        cv2.circle(frame, (cx, cy), 20, (0, 220, 100), -1)
        # Tulis stempel frame
        cv2.putText(frame, f"Frame {i:02d}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        writer.write(frame)

    writer.release()
    return output_path


def process_video_stream(
    video_path: Path,
    max_frames: int = 30
) -> dict:
    """
    Membaca dan memproses sekuens video, mengukur waktu latensi per frame,
    serta menghitung Processing Throughput (FPS Pemrosesan Nyata).
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Gagal membuka berkas video: {video_path}")

    input_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    latencies_ms = []
    processed_count = 0
    start_total = time.perf_counter()

    while cap.isOpened() and processed_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        t0 = time.perf_counter()
        
        # Simulasi alur kerja analisis visi: Grayscale -> Gaussian Blur -> Deteksi Kontur
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 1.0)
        _, thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        t1 = time.perf_counter()
        latencies_ms.append((t1 - t0) * 1000.0)
        processed_count += 1

    cap.release()
    total_elapsed_s = time.perf_counter() - start_total
    processing_fps = processed_count / total_elapsed_s if total_elapsed_s > 0 else 0.0

    return {
        "input_video_fps": float(input_fps),
        "total_clip_frames": total_frames,
        "processed_frames": processed_count,
        "mean_latency_ms": float(np.mean(latencies_ms)),
        "p95_latency_ms": float(np.percentile(latencies_ms, 95)),
        "processing_fps": float(processing_fps),
        "is_realtime_capable": bool(processing_fps >= input_fps)
    }


if __name__ == "__main__":
    print("=== Eksplorasi Pemrosesan Aliran Bingkai Video (Video Stream) ===")
    vid_path = Path("experiments/synthetic_motion_clip.mp4")
    print(f"1. Membangkitkan Video Sintetis di: {vid_path} ...")
    generate_synthetic_video_clip(vid_path, num_frames=30, width=320, height=240, fps=15)

    print("2. Memproses Aliran Frame dan Mengukur Profil Latensi...")
    profile = process_video_stream(vid_path)

    print(f"\nHasil Pemrofilan Kinerja Video:")
    print(f"  FPS Video Masukan     : {profile['input_video_fps']} FPS")
    print(f"  Bingkai Terproses     : {profile['processed_frames']} frame")
    print(f"  Rerata Latensi Frame  : {profile['mean_latency_ms']:.2f} ms")
    print(f"  Persentil 95 Latensi  : {profile['p95_latency_ms']:.2f} ms")
    print(f"  Throughput Pemrosesan : {profile['processing_fps']:.1f} FPS")
    print(f"  Kemampuan Real-Time   : {profile['is_realtime_capable']}")

    assert profile["processed_frames"] == 30, "Jumlah frame terproses tidak cocok!"
    assert profile["processing_fps"] > profile["input_video_fps"], "Pemrosesan harus lebih cepat dari FPS video!"
    print("\nPemrosesan video dan pengukuran latensi berhasil diverifikasi!")

