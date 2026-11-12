"""
Smart Sensor Data Quality Monitor - Rule Engine & Alerts
Chapter 10: AI Foundations Capstone Reference Implementation

Menerapkan aturan inferensi deterministik berbasis Knowledge Base untuk menghasilkan
peringatan kualitas data (Data Quality Alerts) yang dapat dijelaskan logikanya (Explainable).
"""

from typing import Dict, Any, List
import pandas as pd


class DataQualityRuleEngine:
    """Mesin inferensi berbasis aturan untuk mengevaluasi kualitas data sensor."""
    
    def __init__(self, stats_summary: Dict[str, Any]):
        self.stats = stats_summary
        
    def evaluate_records(self, df_validated: pd.DataFrame) -> List[Dict[str, Any]]:
        """Mengevaluasi rekaman telemetri dan memicu alert peringatan."""
        alerts = []
        
        for idx, row in df_validated.iterrows():
            sid = row["sensor_id"]
            val = row["measurement_value"]
            status = row["validation_status"]
            ts = row["timestamp"]
            
            # ATURAN 1: Missing Data Alert
            if status == "MISSING":
                alerts.append({
                    "timestamp": ts,
                    "sensor_id": sid,
                    "severity": "WARNING",
                    "rule_id": "RULE-01-MISSING-BURST",
                    "title": "Paket Data Telemetri Hilang",
                    "explanation": f"Nilai sensor {sid} kosong pada timestamp {ts}. Kemungkinan gangguan jaringan LoRa/Wi-Fi.",
                    "recommended_action": "Periksa konektivitas jaringan gateway dan buffer memori sensor."
                })
                continue
                
            # ATURAN 2: Batas Fisika Mustahil (Physical Impossibility)
            if status == "INVALID_BOUNDS":
                alerts.append({
                    "timestamp": ts,
                    "sensor_id": sid,
                    "severity": "CRITICAL",
                    "rule_id": "RULE-02-PHYSICAL-IMPOSSIBLE",
                    "title": "Kegagalan Fisik / Kerusakan Sensor Terdeteksi",
                    "explanation": f"Nilai {val} berada di luar batas operasional fisika sistem untuk {sid}.",
                    "recommended_action": "Kirim teknisi lapangan untuk kalibrasi perangkat keras atau periksa kabel grounding sensor."
                })
                continue
                
            # ATURAN 3: Lonjakan Statistik (Statistical Outlier Spike)
            if sid in self.stats:
                st = self.stats[sid]
                if val > st["upper_fence"] or val < st["lower_fence"]:
                    alerts.append({
                        "timestamp": ts,
                        "sensor_id": sid,
                        "severity": "MEDIUM",
                        "rule_id": "RULE-03-STATISTICAL-SPIKE",
                        "title": "Lonjakan Statistik Terdeteksi (Tukey Fence)",
                        "explanation": f"Nilai {val} menyimpang dari kuartil normal (Batas Tukey: [{st['lower_fence']}, {st['upper_fence']}]).",
                        "recommended_action": "Verifikasi log operasional mesin apakah terjadi beban puncak sementara sebelum menandai sensor rusak."
                    })
                    
        return alerts

    def calculate_data_quality_index(self, df_validated: pd.DataFrame) -> Dict[str, float]:
        """Menghitung Data Quality Index (DQI) komprehensif dalam rentang [0, 100]."""
        dqi_scores = {}
        for sid, group in df_validated.groupby("sensor_id"):
            total = len(group)
            if total == 0:
                dqi_scores[sid] = 0.0
                continue
            missing = (group["validation_status"] == "MISSING").sum()
            invalid = (group["validation_status"] == "INVALID_BOUNDS").sum()
            
            # Hitung outlier dari group
            st = self.stats.get(sid, {})
            uf = st.get("upper_fence", float("inf"))
            lf = st.get("lower_fence", float("-inf"))
            outliers = (
                (group["validation_status"] == "VALID") & 
                ((group["measurement_value"] > uf) | (group["measurement_value"] < lf))
            ).sum()
            
            penalty = (missing * 1.0) + (invalid * 1.5) + (outliers * 0.5)
            score = max(0.0, 100.0 * (1.0 - (penalty / total)))
            dqi_scores[sid] = round(score, 2)
            
        return dqi_scores

