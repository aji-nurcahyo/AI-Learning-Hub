"""
03_peas_specification_model.py
Representasi berorientasi objek untuk kerangka kerja PEAS
(Performance measure, Environment, Actuators, Sensors).
Menunjukkan bagaimana spesifikasi konseptual PEAS diterjemahkan ke dalam
struktur data komputasional yang dapat dievaluasi secara kuantitatif.

Kurikulum AI Learning Hub - Institut Teknologi Sumatera
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any


@dataclass
class PEASDescriptor:
    """Spesifikasi formal komponen PEAS sistem cerdas."""
    agent_name: str
    performance_metrics: List[str]
    environment_description: str
    actuators: List[str]
    sensors: List[str]
    task_environment_properties: Dict[str, str]

    def display_specification(self):
        print(f"=== SPESIFIKASI PEAS: {self.agent_name.upper()} ===")
        print("\n[P] Performance Measure (Kriteria Kinerja Objektif):")
        for m in self.performance_metrics:
            print(f"  - {m}")
        print(f"\n[E] Environment (Lingkungan Tugas):\n  {self.environment_description}")
        print("\n[A] Actuators (Perangkat Pelaksana Aksi):")
        for a in self.actuators:
            print(f"  - {a}")
        print("\n[S] Sensors (Perangkat Penerima Persepsi):")
        for s in self.sensors:
            print(f"  - {s}")
        print("\n[T] Taksonomi Sifat Lingkungan:")
        for prop, val in self.task_environment_properties.items():
            print(f"  * {prop:<22}: {val}")
        print("=" * 55)


class EnvironmentalPerformanceEvaluator:
    """Evaluator kinerja agen berdasarkan metrik objektif eksternal."""

    def __init__(self, ideal_temp: float = 24.0):
        self.ideal_temp = ideal_temp
        self.total_energy_kwh = 0.0
        self.discomfort_penalty = 0.0
        self.step_count = 0

    def record_step(self, temp: float, occupied: bool, action: str):
        self.step_count += 1
        
        # 1. Kalkulasi konsumsi energi per tindakan
        energy_costs = {
            "COOLING_MAX": 2.5,
            "COOLING_HIGH": 2.0,
            "COOLING_LOW": 1.0,
            "HEATING_MILD": 1.2,
            "FAN_IDLE": 0.2,
            "FAN_CIRCULATE": 0.3,
            "ECO_STANDBY": 0.05,
        }
        self.total_energy_kwh += energy_costs.get(action.split()[0], 0.5)

        # 2. Penalti ketidaknyamanan termal jika ruangan berpenghuni
        if occupied:
            dev = abs(temp - self.ideal_temp)
            if dev > 1.5:
                self.discomfort_penalty += (dev - 1.5) * 5.0

    def compute_score(self) -> float:
        """Skor performa: Semakin tinggi semakin optimal (reward tinggi, penalti rendah)."""
        base_score = 100.0
        # Formula evaluasi rasionalitas agen: Utilitas = 100 - (Bobot Energi + Bobot Ketidaknyamanan)
        score = base_score - (self.total_energy_kwh * 2.0 + self.discomfort_penalty * 3.0)
        return max(0.0, round(score, 2))


def main():
    # 1. Definisi PEAS untuk Smart HVAC Agent
    smart_hvac_peas = PEASDescriptor(
        agent_name="Smart HVAC Climate Agent",
        performance_metrics=[
            "Meminimalkan deviasi suhu ruang dari zona kenyamanan (23°C - 25°C) saat berpenghuni.",
            "Meminimalkan konsumsi energi listrik total (kWh).",
            "Mencegah kerusakan aktuator dengan membatasi frekuensi siklus kompresor.",
            "Waktu respon cepat (< 10 menit) mencapai suhu nyaman saat ruangan mulai terisi."
        ],
        environment_description="Gedung perkantoran multi-lantai dengan cuaca tropis dinamis dan fluktuasi penghuni.",
        actuators=[
            "Katup kompresor pendingin multi-kecepatan (Inverter Compressor)",
            "Motor penggerak sirip ventilasi (Air Flow Louver)",
            "Relai elemen pemanas tambahan (Auxiliary Heater)"
        ],
        sensors=[
            "Sensor termokopel digital suhu udara ruangan (°C)",
            "Sensor kelembapan relatif (% RH)",
            "Sensor inframerah pasif kehadiran manusia (PIR Occupancy Sensor)",
            "Jam waktu nyata (Real-Time Clock)"
        ],
        task_environment_properties={
            "Observabilitas": "Partially Observable (suhu dinding & cuaca luar tak terukur langsung)",
            "Determinisasi": "Stochastic (suhu dipengaruhi bukaan pintu/jendela acak)",
            "Sifat Episode": "Sequential (keputusan pendinginan saat ini memengaruhi suhu menit berikutnya)",
            "Kedinamisan": "Dynamic (suhu berubah sendiri seiring waktu meskipun AC diam)",
            "Diskrit/Kontinu": "Continuous (suhu dan waktu bernilai kontinu)",
            "Jumlah Agen": "Single-Agent (fokus pada kendali ruangan lokal)"
        }
    )

    smart_hvac_peas.display_specification()

    # 2. Pengujian metrik performa objektif
    evaluator = EnvironmentalPerformanceEvaluator(ideal_temp=24.0)
    
    test_log = [
        (28.0, True, "COOLING_MAX"),
        (26.0, True, "COOLING_HIGH"),
        (24.5, True, "COOLING_LOW"),
        (24.0, True, "FAN_IDLE"),
        (24.0, False, "ECO_STANDBY"),
    ]

    for temp, occ, act in test_log:
        evaluator.record_step(temp, occ, act)

    print("\n=== HASIL EVALUASI KINERJA RASIONALITAS ===")
    print(f"Total Siklus Simulasi     : {evaluator.step_count} step")
    print(f"Total Konsumsi Energi     : {evaluator.total_energy_kwh:.2f} kWh")
    print(f"Penalti Ketidaknyamanan   : {evaluator.discomfort_penalty:.2f} poin")
    print(f"Skor Kinerja Rasional (P) : {evaluator.compute_score()} / 100.00")


if __name__ == "__main__":
    main()

