"""
02_model_based_room_agent.py
Implementasi Model-Based Reflex Agent untuk sistem HVAC cerdas.
Karakteristik: Memelihara internal state (status kompresor, riwayat perubahan suhu,
dan durasi ruangan kosong) untuk mencegah osilasi cepat (chattering) dan menghemat energi.

Kurikulum AI Learning Hub - Institut Teknologi Sumatera
"""

from typing import Dict, Any, List


class ModelBasedRoomAgent:
    """Agen berbasis model yang memanfaatkan status internal dan model transisi."""

    def __init__(self, target_temp: float = 24.0, tolerance: float = 1.0, min_cycle_steps: int = 2):
        self.target_temp = target_temp
        self.tolerance = tolerance
        self.min_cycle_steps = min_cycle_steps
        
        # Internal State agen
        self.last_action = "OFF"
        self.steps_in_current_action = 0
        self.previous_temp = None
        self.history_actions: List[str] = []

    def update_state(self, current_temp: float) -> str:
        """Memperbarui model dunia internal: menghitung tren perubahan suhu."""
        if self.previous_temp is None:
            trend = "STABLE"
        elif current_temp > self.previous_temp + 0.2:
            trend = "RISING"
        elif current_temp < self.previous_temp - 0.2:
            trend = "FALLING"
        else:
            trend = "STABLE"
        
        self.previous_temp = current_temp
        return trend

    def select_action(self, percept: Dict[str, Any]) -> str:
        """Memilih tindakan berdasarkan percept dan internal state."""
        temp = percept.get("temperature", 24.0)
        occupied = percept.get("occupied", False)
        
        trend = self.update_state(temp)
        self.steps_in_current_action += 1

        # Aturan Model-Based:
        # Jika ruangan kosong, prioritaskan efisiensi energi
        if not occupied:
            desired_action = "ECO_STANDBY"
        else:
            # Jika suhu di atas batas nyaman
            if temp > self.target_temp + self.tolerance:
                desired_action = "COOLING_HIGH"
            # Jika suhu di bawah batas nyaman
            elif temp < self.target_temp - self.tolerance:
                desired_action = "HEATING_MILD"
            else:
                # Di dalam zona toleransi nyaman:
                # Periksa tren suhu menggunakan model internal untuk menghindari switching dini
                if trend == "FALLING" and self.last_action == "COOLING_HIGH":
                    desired_action = "COOLING_LOW"  # Perlambat pendinginan
                elif trend == "RISING" and self.last_action == "HEATING_MILD":
                    desired_action = "FAN_CIRCULATE"
                else:
                    desired_action = "COMFORT_MAINTAIN"

        # Model Pelindung Kompresor (Anti-Chattering):
        # Mencegah kompresor mati-nyala terlalu cepat dalam waktu singkat
        if (
            self.last_action.startswith("COOLING") and desired_action == "HEATING_MILD"
            and self.steps_in_current_action < self.min_cycle_steps
        ):
            # Tunda transisi drastis; jalankan sirkulasi kipas sementara
            actual_action = "FAN_CIRCULATE (PROTECTED)"
        else:
            actual_action = desired_action

        # Update status aksi internal
        if actual_action != self.last_action:
            self.last_action = actual_action
            self.steps_in_current_action = 1
        
        self.history_actions.append(actual_action)
        return f"{actual_action} [Trend: {trend}]"


def run_simulation():
    agent = ModelBasedRoomAgent(target_temp=24.0, tolerance=1.0, min_cycle_steps=2)

    # Rangkaian persepsi berurutan (Sequential Environment)
    sensor_stream = [
        {"temp": 28.0, "occ": True, "event": "Mulai: Siang hari terik, ruangan penuh"},
        {"temp": 26.5, "occ": True, "event": "Suhu mulai turun bertahap"},
        {"temp": 24.8, "occ": True, "event": "Suhu mendekati target ideal"},
        {"temp": 24.0, "occ": True, "event": "Suhu tepat di zona target"},
        {"temp": 22.8, "occ": True, "event": "Fluktuasi suhu tiba-tiba turun sedikit"},
        {"temp": 22.8, "occ": False, "event": "Penghuni meninggalkan ruangan"},
        {"temp": 25.0, "occ": False, "event": "Ruangan tetap kosong"},
    ]

    print("=== SIMULASI MODEL-BASED ROOM AGENT ===")
    print("Fitur Internal State: Pelacakan Tren Suhu & Proteksi Siklus Kompresor\n")
    print(f"{'Step':<5} | {'Suhu':<6} | {'Penghuni':<8} | {'Keputusan Agen':<35} | {'Skenario Dinamis'}")
    print("-" * 88)

    for step, item in enumerate(sensor_stream, start=1):
        percept = {"temperature": item["temp"], "occupied": item["occ"]}
        decision = agent.select_action(percept)
        occ_str = "Ada" if item["occ"] else "Kosong"
        print(f"{step:<5} | {item['temp']:<6.1f} | {occ_str:<8} | {decision:<35} | {item['event']}")


if __name__ == "__main__":
    run_simulation()

