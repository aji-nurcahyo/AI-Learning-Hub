"""
Unit Tests for Smart Sensor Data Quality Monitor
Chapter 10: AI Foundations Capstone Reference Implementation
"""

import unittest
import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Pastikan modul direktori lokal terbaca
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dataset_generator import generate_synthetic_sensor_data
from validator import validate_schema, validate_sensor_records
from statistics_analyzer import compute_sensor_statistics
from rule_engine import DataQualityRuleEngine


class TestSensorDataQualityMonitor(unittest.TestCase):
    
    def setUp(self):
        self.tmp_csv = Path(__file__).resolve().parent / "data" / "test_telemetry.csv"
        self.df = generate_synthetic_sensor_data(self.tmp_csv, n_records=60, seed=42)
        
    def tearDown(self):
        if self.tmp_csv.exists():
            self.tmp_csv.unlink()
            
    def test_dataset_generation_shape(self):
        self.assertEqual(len(self.df), 60)
        self.assertIn("measurement_value", self.df.columns)
        self.assertIn("timestamp", self.df.columns)
        # Verifikasi keberadaan missing values terkontrol
        self.assertGreater(self.df["measurement_value"].isna().sum(), 0)
        
    def test_schema_validation(self):
        is_valid, msg = validate_schema(self.df)
        self.assertTrue(is_valid)
        
        # Test missing column
        bad_df = self.df.drop(columns=["sensor_id"])
        is_valid_bad, msg_bad = validate_schema(bad_df)
        self.assertFalse(is_valid_bad)
        self.assertIn("sensor_id", msg_bad)
        
    def test_physical_bounds_detection(self):
        val_df, report = validate_sensor_records(self.df)
        self.assertIn("validation_status", val_df.columns)
        # Rekaman indeks 24 adalah -300 C (suhu di bawah nol mutlak)
        row_24 = val_df.iloc[24]
        self.assertEqual(row_24["validation_status"], "INVALID_BOUNDS")
        self.assertGreater(report["invalid_bounds_count"], 0)
        
    def test_statistics_and_tukey_fences(self):
        val_df, _ = validate_sensor_records(self.df)
        stats = compute_sensor_statistics(val_df)
        self.assertIn("TEMP-01", stats)
        temp_st = stats["TEMP-01"]
        self.assertGreater(temp_st["upper_fence"], temp_st["median"])
        self.assertLess(temp_st["lower_fence"], temp_st["median"])
        self.assertEqual(temp_st["iqr"], round(temp_st["q3"] - temp_st["q1"], 2))
        
    def test_rule_engine_and_dqi(self):
        val_df, _ = validate_sensor_records(self.df)
        stats = compute_sensor_statistics(val_df)
        engine = DataQualityRuleEngine(stats)
        alerts = engine.evaluate_records(val_df)
        dqi = engine.calculate_data_quality_index(val_df)
        
        self.assertGreater(len(alerts), 0)
        rule_ids = {a["rule_id"] for a in alerts}
        self.assertIn("RULE-01-MISSING-BURST", rule_ids)
        self.assertIn("RULE-02-PHYSICAL-IMPOSSIBLE", rule_ids)
        
        for sid, score in dqi.items():
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 100.0)
            
    def test_edge_case_unknown_sensor(self):
        unknown_df = pd.DataFrame([{
            "timestamp": "2026-11-05 08:00:00",
            "sensor_id": "GHOST-99",
            "measurement_value": 50.0,
            "unit": "unknown"
        }])
        val_df, report = validate_sensor_records(unknown_df)
        self.assertEqual(val_df.iloc[0]["validation_status"], "UNKNOWN_SENSOR")


if __name__ == "__main__":
    unittest.main()
