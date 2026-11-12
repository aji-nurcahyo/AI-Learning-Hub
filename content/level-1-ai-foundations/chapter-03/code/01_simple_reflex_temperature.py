"""
01_simple_reflex_temperature.py
Implementasi Simple Reflex Agent untuk sistem pengatur suhu ruangan sederhana.
Karakteristik: Memilih tindakan murni berdasarkan persepsi saat ini (condition-action rules),
tanpa memelihara status internal masa lalu.

Kurikulum AI Learning Hub - Institut Teknologi Sumatera
"""

from typing import Dict, Any, Tuple


class SimpleReflexTemperatureAgent:
    """Agen refleks sederhana yang mengeksekusi aturan kondisi-aksi langsung."""

    def __init__(self, target_temp: float = 24.0, tolerance: float = 1.0):
        self.target_temp = target_temp
        self.tolerance = tolerance

    def condition_action_rule(self, current_temp: float, is_occupied: bool) -> str:
        """Memetakan percept saat ini ke tindakan diskret."""
        if not is_occupied:
            return "ECO_STANDBY"
        
        # Kondisi saat ruangan berpenghuni
        if current_temp > self.target_temp + self.tolerance:
            return "COOLING_MAX"
        elif current_temp < self.target_temp - self.tolerance:
            return "HEATING_MODERATE"
        else:
            return "FAN_IDLE"

    def select_action(self, percept: Dict[str, Any]) -> str:
        """Fungsi agen utama: f(percept) -> action."""
        temp = percept.get("temperature", 24.0)
        occupied = percept.get("occupied", False)
        return self.condition_action_rule(temp, occupied)


def run_simulation():
    agent = SimpleReflexTemperatureAgent(target_temp=24.0, tolerance=1.0)
    
    # Rangkaian persepsi uji (Time step, Suhu Ruang, Status Kehadiran Orang)
    scenarios = [
        {"step": 1, "temperature": 28.5, "occupied": True, "desc": "Ruangan panas dan ada orang"},
        {"step": 2, "temperature": 24.2, "occupied": True, "desc": "Suhu nyaman dan ada orang"},
        {"step": 3, "temperature": 21.0, "occupied": True, "desc": "Suhu terlalu dingin dan ada orang"},
        {"step": 4, "temperature": 28.5, "occupied": False, "desc": "Ruangan panas tetapi kosong"},
        {"step": 5, "temperature": 24.0, "occupied": False, "desc": "Ruangan ideal dan kosong"},
    ]

    print("=== SIMULASI SIMPLE REFLEX TEMPERATURE AGENT ===")
    print(f"Target Suhu: {agent.target_temp}°C (Toleransi: ±{agent.tolerance}°C)\n")
    print(f"{'Step':<6} | {'Suhu (°C)':<10} | {'Penghuni':<10} | {'Tindakan Agen':<18} | {'Konteks Masalah'}")
    print("-" * 75)

    for sc in scenarios:
        percept = {"temperature": sc["temperature"], "occupied": sc["occupied"]}
        action = agent.select_action(percept)
        occ_str = "Ada" if sc["occupied"] else "Kosong"
        print(f"{sc['step']:<6} | {sc['temperature']:<10.1f} | {occ_str:<10} | {action:<18} | {sc['desc']}")


if __name__ == "__main__":
    run_simulation()

