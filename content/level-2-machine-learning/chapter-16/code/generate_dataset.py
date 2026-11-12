"""Pembangkit Dataset Status Jaringan Sensor Nirkabel (Deterministik)
Untuk Chapter 16: Support Vector Machines & k-Nearest Neighbors
"""

import pathlib
import numpy as np
import pandas as pd

def generate_sensor_network_data(output_path: pathlib.Path, random_seed: int = 42) -> pd.DataFrame:
    np.random.seed(random_seed)
    n_samples = 360

    # Fitur-fitur telemetri simpul sensor nirkabel (Wireless Sensor Network / WSN)
    # 1. Signal strength (RSSI) dalam dBm: -90 dBm (lemah) s/d -30 dBm (sangat kuat)
    signal_strength = np.random.uniform(-85.0, -35.0, n_samples)

    # 2. Temperature dalam Celcius: 20°C s/d 65°C
    temperature = np.random.uniform(22.0, 58.0, n_samples)

    # 3. Humidity dalam persen: 30% s/d 95%
    humidity = np.random.uniform(35.0, 90.0, n_samples)

    # 4. Packet loss percentage: 0% s/d 25%
    packet_loss = np.random.exponential(scale=2.5, size=n_samples)
    packet_loss = np.clip(packet_loss, 0.0, 30.0)

    # 5. Latency dalam ms: 15 ms s/d 350 ms
    latency = 20.0 + 3.5 * packet_loss + np.random.normal(0, 10.0, n_samples)
    latency = np.clip(latency, 10.0, 400.0)

    # 6. Sensor voltage dalam Volt: 2.4V s/d 3.6V
    voltage = np.random.normal(3.3, 0.25, n_samples)
    voltage = np.clip(voltage, 2.1, 3.8)

    # Penetapan kategori status simpul jaringan:
    # NORMAL: Sinyal kuat, paket loss rendah, latensi rendah, suhu normal, tegangan prima
    # WARNING: Penurunan sinyal, latensi meningkat, atau baterai mulai aus
    # CRITICAL: Paket loss parah, baterai kritis (< 2.5V), suhu ekstrem (> 50°C), atau sinyal putus-putus
    categories = []

    for i in range(n_samples):
        sig = signal_strength[i]
        tmp = temperature[i]
        hum = humidity[i]
        pkt = packet_loss[i]
        lat = latency[i]
        volt = voltage[i]

        is_critical = (pkt > 12.0) or (volt < 2.5) or (tmp > 52.0) or (sig < -80.0 and lat > 180.0)
        is_warning = (pkt > 5.0) or (volt < 2.8) or (tmp > 42.0) or (sig < -72.0) or (lat > 90.0)

        # Injeksi variasi deterministik teratur
        if i % 6 == 0:
            pkt += np.random.uniform(6.0, 12.0)
            is_critical = True
        elif i % 10 == 0:
            volt -= np.random.uniform(0.4, 0.7)
            is_critical = True

        if is_critical:
            categories.append("CRITICAL")
        elif is_warning:
            categories.append("WARNING")
        else:
            categories.append("NORMAL")

    df = pd.DataFrame({
        "signal_strength_dbm": np.round(signal_strength, 1),
        "temperature_c": np.round(temperature, 1),
        "humidity_pct": np.round(humidity, 1),
        "packet_loss_pct": np.round(packet_loss, 2),
        "latency_ms": np.round(latency, 1),
        "sensor_voltage_v": np.round(voltage, 2),
        "node_status": categories
    })

    # Encode integer
    mapping = {"NORMAL": 0, "WARNING": 1, "CRITICAL": 2}
    df["status_code"] = df["node_status"].map(mapping)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return df

if __name__ == "__main__":
    out = pathlib.Path(__file__).parent / "data" / "wireless_sensor_network_state.csv"
    data = generate_sensor_network_data(out)
    print(f"Dataset generated at: {out}")
    print(f"Total samples: {len(data)}")
    print(f"Distribution:\n{data['node_status'].value_counts()}")

