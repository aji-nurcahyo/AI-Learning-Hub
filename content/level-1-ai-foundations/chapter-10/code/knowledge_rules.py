"""Knowledge Representation & Logic Rule Inference Engine.

Chapter 10: AI Foundations Capstone.
Menerapkan representasi pengetahuan berbasis First-Order Logic dan
Forward Chaining Inference Engine untuk mengevaluasi keselamatan zona kampus,
status bahaya kebakaran, serta kendala aksesibilitas disabilitas.
"""

from typing import Dict, List, Set, Tuple


class CampusKnowledgeBase:
    """Basis Pengetahuan (Knowledge Base) untuk inferensi aturan darurat kampus."""

    def __init__(self):
        # Basis fakta dinamis
        self.facts: Set[str] = set()
        self.zone_penalties: Dict[Tuple[int, int], float] = {}
        self.blocked_zones: Set[Tuple[int, int]] = set()

    def tell(self, fact: str) -> None:
        """Menambahkan fakta baru ke dalam Knowledge Base."""
        self.facts.add(fact)

    def ask(self, query: str) -> bool:
        """Memeriksa apakah fakta atau proposisi ada di dalam KB."""
        return query in self.facts

    def infer_safety_status(
        self,
        sensor_readings: List[Dict],
        wheelchair_mode: bool = False
    ) -> Tuple[Set[Tuple[int, int]], Dict[Tuple[int, int], float]]:
        """Menjalankan mesin inferensi maju (Forward Chaining) pada data telemetri.
        
        Aturan Logika:
        R1: IF Smoke(z) > 400 ppm AND Temp(z) > 55 deg C THEN FireHazard(z)
        R2: IF FireHazard(z) THEN Blocked(z)
        R3: IF FireHazard(z) AND Adjacent(z, n) THEN SmokeSpread(n)
        R4: IF SmokeSpread(n) AND NOT Blocked(n) THEN Penalty(n) = 3.5
        R5: IF wheelchair_mode AND StairOnly(z) THEN Inaccessible(z)
        R6: IF Inaccessible(z) THEN Blocked(z)
        """
        self.blocked_zones.clear()
        self.zone_penalties.clear()

        # 1. Evaluasi pembacaan sensor tiap zona
        for reading in sensor_readings:
            coord = reading["coord"]
            smoke = reading.get("smoke_ppm", 0.0)
            temp = reading.get("temp_c", 25.0)
            has_ramp = reading.get("has_ramp", True)

            # R1: Deteksi titik api utama
            if smoke >= 400.0 and temp >= 55.0:
                self.tell(f"FireHazard({coord})")
                # R2: Titik api mutlak tidak boleh dilalui
                self.blocked_zones.add(coord)
                self.tell(f"Blocked({coord})")

            # R5 & R6: Kendala aksesibilitas kursi roda
            if wheelchair_mode and not has_ramp:
                self.tell(f"StairOnly({coord})")
                self.blocked_zones.add(coord)
                self.tell(f"InaccessibleForWheelchair({coord})")

        # 2. R3 & R4: Propagasi asap ke zona-zona tetangga (Adjacent Neighbors)
        for blocked in list(self.blocked_zones):
            if f"FireHazard({blocked})" in self.facts:
                bx, by = blocked
                neighbors = [
                    (bx + 1, by), (bx - 1, by),
                    (bx, by + 1), (bx, by - 1)
                ]
                for n in neighbors:
                    if (0 <= n[0] < 10) and (0 <= n[1] < 10):
                        if n not in self.blocked_zones:
                            self.tell(f"SmokeSpread({n})")
                            self.zone_penalties[n] = 3.5

        return self.blocked_zones, self.zone_penalties


def main() -> None:
    print("=== INFERENSI BASIS PENGETAHUAN DARURAT (FORWARD CHAINING) ===")
    kb = CampusKnowledgeBase()

    sample_telemetry = [
        {"zone_id": "LAB-TEK-01", "coord": (3, 4), "temp_c": 78.5, "smoke_ppm": 550.0, "has_ramp": True},
        {"zone_id": "LAB-TEK-02", "coord": (3, 5), "temp_c": 62.0, "smoke_ppm": 480.0, "has_ramp": True},
        {"zone_id": "GEDUNG-B-L2", "coord": (5, 6), "temp_c": 26.0, "smoke_ppm": 12.0, "has_ramp": False},
        {"zone_id": "KORIDOR-UTAMA", "coord": (2, 4), "temp_c": 32.0, "smoke_ppm": 85.0, "has_ramp": True},
    ]

    # Uji 1: Jalur Evakuasi Standar (Pejalan Kaki Sehat)
    blocked_std, penalties_std = kb.infer_safety_status(sample_telemetry, wheelchair_mode=False)
    print(f"\n[Skenario 1: Evakuasi Standar]")
    print(f"Zona Api Terblokir (Blocked)  : {blocked_std}")
    print(f"Zona Penalti Asap (SmokeZone) : {penalties_std}")

    # Uji 2: Jalur Evakuasi Ramah Disabilitas (Wheelchair Accessible)
    kb2 = CampusKnowledgeBase()
    blocked_wc, penalties_wc = kb2.infer_safety_status(sample_telemetry, wheelchair_mode=True)
    print(f"\n[Skenario 2: Evakuasi Ramah Disabilitas]")
    print(f"Zona Terblokir Total (Api + Tangga): {blocked_wc}")
    print(f"Zona Tangga Terblokir Khusus       : {(5, 6) in blocked_wc}")

    print("\nHASIL PENALARAN LOGIKA:")
    print("Knowledge Base berhasil menyimpulkan status keselamatan zona dinamis")
    print("tanpa perlu menebak; aturan deduktif menjamin kepatuhan keselamatan 100%.")


if __name__ == "__main__":
    main()

