"""Pembangkit Dataset Segmentasi Telemetri Sensor Nirkabel (Deterministik)
Untuk Chapter 17: Unsupervised Learning & Clustering
dan Chapter 18: Dimensionality Reduction
"""

import pathlib
import numpy as np
import pandas as pd

def generate_telemetry_data(output_path: pathlib.Path, random_seed: int = 42) -> pd.DataFrame:
    np.random.seed(random_seed)
    n_samples = 360

    # Pembangkitan data segmentasi tanpa label terawasi:
    # Terdiri dari 3 pola perilaku operasional alamiah (latent clusters) + noise:
    # Cluster 0: Simpul Normal / Hemat Daya (Sinyal prima, latensi rendah, paket loss nihil, suhu adem)
    # Cluster 1: Simpul Jauh / Beban Tinggi (Sinyal agak lemah, latensi menengah, konsumsi daya tinggi)
    # Cluster 2: Simpul Kritis / Lingkungan Buruk (Suhu tinggi, paket loss tinggi, baterai tiris)
    # Outliers / Noise: Sebaran acak sporadis

    n_cluster = 110
    n_noise = 30

    # Pola 1: Normal / Hemat Daya
    c1_sig = np.random.normal(-45.0, 4.0, n_cluster)
    c1_pkt = np.random.exponential(0.8, n_cluster)
    c1_lat = np.random.normal(25.0, 5.0, n_cluster)
    c1_tmp = np.random.normal(28.0, 3.0, n_cluster)
    c1_vlt = np.random.normal(3.5, 0.1, n_cluster)
    c1_eng = np.random.normal(25.0, 5.0, n_cluster)

    # Pola 2: Jauh / Beban Tinggi
    c2_sig = np.random.normal(-68.0, 5.0, n_cluster)
    c2_pkt = np.random.normal(3.5, 1.0, n_cluster)
    c2_lat = np.random.normal(95.0, 15.0, n_cluster)
    c2_tmp = np.random.normal(36.0, 4.0, n_cluster)
    c2_vlt = np.random.normal(3.2, 0.15, n_cluster)
    c2_eng = np.random.normal(90.0, 12.0, n_cluster)

    # Pola 3: Kritis / Termal Tinggi
    c3_sig = np.random.normal(-78.0, 6.0, n_cluster)
    c3_pkt = np.random.normal(12.0, 3.0, n_cluster)
    c3_lat = np.random.normal(210.0, 30.0, n_cluster)
    c3_tmp = np.random.normal(54.0, 5.0, n_cluster)
    c3_vlt = np.random.normal(2.6, 0.2, n_cluster)
    c3_eng = np.random.normal(140.0, 18.0, n_cluster)

    # Noise Points
    cn_sig = np.random.uniform(-85.0, -35.0, n_noise)
    cn_pkt = np.random.uniform(0.0, 22.0, n_noise)
    cn_lat = np.random.uniform(15.0, 320.0, n_noise)
    cn_tmp = np.random.uniform(20.0, 62.0, n_noise)
    cn_vlt = np.random.uniform(2.2, 3.7, n_noise)
    cn_eng = np.random.uniform(10.0, 180.0, n_noise)

    sig = np.concatenate([c1_sig, c2_sig, c3_sig, cn_sig])
    pkt = np.clip(np.concatenate([c1_pkt, c2_pkt, c3_pkt, cn_pkt]), 0.0, 35.0)
    lat = np.clip(np.concatenate([c1_lat, c2_lat, c3_lat, cn_lat]), 10.0, 400.0)
    tmp = np.clip(np.concatenate([c1_tmp, c2_tmp, c3_tmp, cn_tmp]), 15.0, 70.0)
    vlt = np.clip(np.concatenate([c1_vlt, c2_vlt, c3_vlt, cn_vlt]), 2.0, 3.8)
    eng = np.clip(np.concatenate([c1_eng, c2_eng, c3_eng, cn_eng]), 5.0, 200.0)

    # Latent profile hanya untuk evaluasi eksternal demonstrasi, BUKAN untuk input clustering!
    latent = ["ECO_NORMAL"] * n_cluster + ["HIGH_LOAD"] * n_cluster + ["THERMAL_CRITICAL"] * n_cluster + ["OUTLIER_NOISE"] * n_noise

    df = pd.DataFrame({
        "signal_strength_dbm": np.round(sig, 1),
        "packet_loss_pct": np.round(pkt, 2),
        "latency_ms": np.round(lat, 1),
        "temperature_c": np.round(tmp, 1),
        "voltage_v": np.round(vlt, 2),
        "energy_consumption_mah": np.round(eng, 1),
        "latent_profile": latent
    })

    # Acak urutan baris agar tidak terurut rapi per cluster
    df = df.sample(frac=1.0, random_state=random_seed).reset_index(drop=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return df

if __name__ == "__main__":
    out = pathlib.Path(__file__).parent / "data" / "wireless_sensor_telemetry.csv"
    data = generate_telemetry_data(out)
    print(f"Dataset generated at: {out}")
    print(f"Total samples: {len(data)}")
    print(f"Latent profile distribution:\n{data['latent_profile'].value_counts()}")

