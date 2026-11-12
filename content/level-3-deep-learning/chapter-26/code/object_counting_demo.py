"""
Chapter 26 - Computer Vision Applications
Skrip 10: Object Counting with Virtual Tripwire Line Crossing
Mengimplementasikan penghitungan objek otomatis pada aliran video menggunakan garis pembatas virtual (Tripwire):
mendeteksi arah perlintasan objek (In/Out), mencegah penghitungan ganda (Double Counting Mitigation),
dan melacak status lintasan setiap ID objek.
"""

from typing import Dict, List, Tuple, Set
import numpy as np
from centroid_tracking import CentroidTracker


class TripwireObjectCounter:
    """
    Penghitung objek otomatis yang memantau perlintasan garis virtual horizontal.
    """
    def __init__(self, line_y: int = 120, direction_tolerance: int = 5):
        self.line_y = line_y
        self.direction_tolerance = direction_tolerance
        self.tracker = CentroidTracker(max_disappeared=5, max_distance=60.0)
        
        # Riwayat posisi Y sebelumnya untuk setiap ID objek: {obj_id: [y_history]}
        self.track_history: Dict[int, List[float]] = {}
        # Himpunan ID objek yang telah dihitung agar tidak dihitung ganda
        self.counted_ids: Set[int] = set()
        
        self.count_down = 0 # Objek bergerak ke bawah melintasi garis
        self.count_up = 0   # Objek bergerak ke atas melintasi garis

    def process_frame(self, detected_bboxes: List[List[float]]) -> Tuple[int, int, Dict[int, Tuple[float, float]]]:
        """
        Memproses satu frame: memperbarui pelacak, mendeteksi perlintasan garis,
        dan memperbarui counter.
        """
        tracked_objects = self.tracker.update(detected_bboxes)

        for obj_id, centroid in tracked_objects.items():
            cx, cy = centroid
            if obj_id not in self.track_history:
                self.track_history[obj_id] = []
            self.track_history[obj_id].append(cy)

            # Jika objek memiliki riwayat minimal 2 posisi
            if len(self.track_history[obj_id]) >= 2 and obj_id not in self.counted_ids:
                prev_y = self.track_history[obj_id][-2]
                curr_y = self.track_history[obj_id][-1]

                # Periksa apakah objek memotong garis horizontal
                # 1. Melintas ke bawah: posisi sebelumnya < line_y dan posisi sekarang >= line_y
                if prev_y < self.line_y <= curr_y:
                    self.count_down += 1
                    self.counted_ids.add(obj_id)
                # 2. Melintas ke atas: posisi sebelumnya > line_y dan posisi sekarang <= line_y
                elif prev_y > self.line_y >= curr_y:
                    self.count_up += 1
                    self.counted_ids.add(obj_id)

        return self.count_down, self.count_up, tracked_objects


if __name__ == "__main__":
    print("=== Eksplorasi Penghitungan Objek Otomatis (Tripwire Line Crossing) ===")

    counter = TripwireObjectCounter(line_y=120)
    print(f"Garis Virtual Tripwire Ditetapkan pada Koordinat Y = {counter.line_y} piksel.\n")

    # Simulasi Sekuens 5 Frame:
    # Objek A (Mobil): Mulai dari Y=80, turun ke Y=140 melintasi garis Y=120
    # Objek B (Motor): Mulai dari Y=180, naik ke Y=90 melintasi garis Y=120
    video_sequence = [
        # Frame 1: Sebelum melintasi garis
        [[100.0, 70.0,  150.0, 90.0],  [250.0, 170.0, 280.0, 190.0]], # Centroid A ~80, B ~180
        # Frame 2: Mendekati garis
        [[100.0, 95.0,  150.0, 115.0], [250.0, 145.0, 280.0, 165.0]], # Centroid A ~105, B ~155
        # Frame 3: Melintasi garis (Cross Event)
        [[100.0, 120.0, 150.0, 140.0], [250.0, 110.0, 280.0, 130.0]], # Centroid A ~130, B ~120
        # Frame 4: Melewati garis
        [[100.0, 145.0, 150.0, 165.0], [250.0, 75.0,  280.0, 95.0]],  # Centroid A ~155, B ~85
        # Frame 5: Menjauh (Verifikasi tidak terjadi double counting)
        [[100.0, 170.0, 150.0, 190.0], [250.0, 45.0,  280.0, 65.0]],  # Centroid A ~180, B ~55
    ]

    for f_idx, bboxes in enumerate(video_sequence, start=1):
        down, up, objects = counter.process_frame(bboxes)
        print(f"Frame {f_idx}: Objek Terlacak={len(objects)} | Total Melintas Ke Bawah={down} | Melintas Ke Atas={up}")

    print(f"\nHasil Akhir Penghitungan Tripwire:")
    print(f"  Total Kendaraan Masuk (Ke Bawah) : {counter.count_down}")
    print(f"  Total Kendaraan Keluar (Ke Atas) : {counter.count_up}")
    print(f"  Total ID Objek Terhitung Unik     : {len(counter.counted_ids)}")

    assert counter.count_down == 1, "Penghitungan arah bawah salah!"
    assert counter.count_up == 1, "Penghitungan arah atas salah!"
    assert len(counter.counted_ids) == 2, "Mitigasi double-counting gagal!"
    print("Status: PENGHITUNGAN OBJEK DAN MITIGASI DOUBLE COUNTING LULUS 100%!")

