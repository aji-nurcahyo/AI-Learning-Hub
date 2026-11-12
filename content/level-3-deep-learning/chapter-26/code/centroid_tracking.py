"""
Chapter 26 - Computer Vision Applications
Skrip 8: Centroid Object Tracking (Tracking-by-Detection Paradigm)
Mengimplementasikan pelacak objek berbasis centroid:
mengasosiasikan deteksi antarbingkai melalui jarak Euclidean,
mengelola siklus hidup objek (Registrasi ID baru, Pembaruan koordinat, dan Deregistrasi objek hilang).
"""

from typing import Dict, List, Tuple
from collections import OrderedDict
import numpy as np


class CentroidTracker:
    """
    Pelacak objek visual berbasis jarak centroid Euclidean antarbingkai.
    """
    def __init__(self, max_disappeared: int = 5, max_distance: float = 50.0):
        # Menyimpan ID objek -> koordinat centroid (cx, cy)
        self.objects = OrderedDict()
        # Menyimpan ID objek -> jumlah frame berturut-turut objek hilang
        self.disappeared = OrderedDict()
        self.next_object_id = 0
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance

    def register(self, centroid: Tuple[float, float]) -> int:
        """Mendaftarkan objek baru dengan ID unik inkremental."""
        assigned_id = self.next_object_id
        self.objects[assigned_id] = centroid
        self.disappeared[assigned_id] = 0
        self.next_object_id += 1
        return assigned_id

    def deregister(self, object_id: int) -> None:
        """Menghapus objek dari registri setelah hilang melebihi batas toleransi."""
        del self.objects[object_id]
        del self.disappeared[object_id]

    def update(self, rects: List[List[float]]) -> Dict[int, Tuple[float, float]]:
        """
        Menerima daftar bounding box [x1, y1, x2, y2] pada frame saat ini
        dan memperbarui asosiasi ID objek.
        """
        # Jika tidak ada deteksi pada frame saat ini
        if len(rects) == 0:
            for obj_id in list(self.disappeared.keys()):
                self.disappeared[obj_id] += 1
                if self.disappeared[obj_id] > self.max_disappeared:
                    self.deregister(obj_id)
            return self.objects

        # Hitung centroid dari setiap kotak masukan
        input_centroids = np.zeros((len(rects), 2), dtype=np.float32)
        for i, (x1, y1, x2, y2) in enumerate(rects):
            input_centroids[i] = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

        # Jika belum ada objek yang dilacak, daftarkan seluruhnya
        if len(self.objects) == 0:
            for i in range(len(input_centroids)):
                self.register(tuple(input_centroids[i]))
            return self.objects

        # Hitung matriks jarak Euclidean antara objek lama dan deteksi baru
        object_ids = list(self.objects.keys())
        object_centroids = list(self.objects.values())

        # Matriks jarak berdimensi (num_existing_objects, num_input_centroids)
        dists = np.linalg.norm(
            np.array(object_centroids)[:, np.newaxis] - input_centroids[np.newaxis, :], axis=2
        )

        # Pencocokan Greedy berdasarkan jarak terdekat
        rows = dists.min(axis=1).argsort()
        cols = dists.argmin(axis=1)[rows]

        used_rows = set()
        used_cols = set()

        for (row, col) in zip(rows, cols):
            if row in used_rows or col in used_cols:
                continue

            if dists[row, col] > self.max_distance:
                continue

            obj_id = object_ids[row]
            self.objects[obj_id] = tuple(input_centroids[col])
            self.disappeared[obj_id] = 0

            used_rows.add(row)
            used_cols.add(col)

        # Tangani objek yang belum terpasangkan (hilang atau baru)
        unused_rows = set(range(len(object_ids))) - used_rows
        for row in unused_rows:
            obj_id = object_ids[row]
            self.disappeared[obj_id] += 1
            if self.disappeared[obj_id] > self.max_disappeared:
                self.deregister(obj_id)

        unused_cols = set(range(len(input_centroids))) - used_cols
        for col in unused_cols:
            self.register(tuple(input_centroids[col]))

        return self.objects


if __name__ == "__main__":
    print("=== Eksplorasi Pelacakan Objek Visual (Centroid Tracking) ===")

    tracker = CentroidTracker(max_disappeared=2, max_distance=60.0)

    # Simulasi Sekuens 4 Frame Pergerakan Dua Objek (Mobil A dan Orang B)
    # Objek 0 bergerak ke kanan (+20 piksel per frame)
    # Objek 1 bergerak ke bawah (+15 piksel per frame)
    frames_detections = [
        # Frame 1: Dua objek muncul pertama kali
        [[100.0, 100.0, 160.0, 160.0], [300.0, 200.0, 350.0, 300.0]],
        # Frame 2: Kedua objek bergeser
        [[120.0, 100.0, 180.0, 160.0], [300.0, 215.0, 350.0, 315.0]],
        # Frame 3: Objek 1 terhalang pohon (Occlusion / Deteksi hilang sementara)
        [[140.0, 100.0, 200.0, 160.0]],
        # Frame 4: Objek 1 muncul kembali setelah oklusi
        [[160.0, 100.0, 220.0, 160.0], [300.0, 245.0, 350.0, 345.0]],
    ]

    for frame_idx, detections in enumerate(frames_detections, start=1):
        tracked = tracker.update(detections)
        print(f"Frame {frame_idx}: {len(detections)} Kotak Terdeteksi -> {len(tracked)} Objek Aktif Dilacak")
        for obj_id, centroid in tracked.items():
            print(f"  [ID {obj_id}] Centroid = ({centroid[0]:.1f}, {centroid[1]:.1f})")

    # Verifikasi konsistensi ID
    # ID 0 dan ID 1 harus tetap mempertahankan ID-nya tanpa tertukar (Identity Switch)
    assert 0 in tracker.objects and 1 in tracker.objects
    print("\nStatus Verifikasi: KONSISTENSI ID OBJEK TERBUKTI STABIL MESKIPUN TERJADI OKLUSI SEMENTARA!")

