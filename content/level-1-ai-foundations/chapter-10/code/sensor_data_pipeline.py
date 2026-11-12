"""Campus Sensor Data Preparation & Headless EDA Visualization.

Chapter 10: AI Foundations Capstone.
Memuat telemetri sensor kampus mentah (campus_telemetry_raw.csv), membersihkan
inkonsistensi string, mengimputasi missing values berbasis tipe fasilitas,
dan menghasilkan peta grid evakuasi kampus (campus_evacuation_map.png).
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def generate_campus_telemetry_raw(csv_path: Path) -> pd.DataFrame:
    """Menghasilkan dataset telemetri sensor 100 node grid kampus ITERA (10 x 10)."""
    np.random.seed(42)
    records = []
    facility_types = ["Gedung Kuliah", "Laboratorium", "Koridor Luar", "Lapangan Terbuka"]

    for x in range(10):
        for y in range(10):
            # Tentukan tipe fasilitas berdasarkan lokasi
            if (x, y) in [(0, 9), (9, 9), (0, 0), (9, 0)]:
                ftype = "Lapangan Terbuka" # Titik kumpul aman
                has_ramp = True
            elif 2 <= x <= 4 and 3 <= y <= 6:
                ftype = "Laboratorium" # Kawasan rawan api
                has_ramp = (x % 2 == 0) # Sebagian bertangga
            elif x >= 6 and y >= 6:
                ftype = "Gedung Kuliah"
                has_ramp = (y % 2 == 0)
            else:
                ftype = "Koridor Luar"
                has_ramp = True

            # Kondisi termal & asap normal baseline
            temp = float(np.random.normal(loc=26.5, scale=2.5))
            smoke = float(np.clip(np.random.normal(loc=15.0, scale=5.0), 0, 50))
            door_status = "OPEN"

            # Titik api darurat nyata di laboratorium (3, 4) dan (3, 5)
            if (x, y) in [(3, 4), (3, 5)]:
                temp = float(np.random.normal(loc=82.0, scale=3.0))
                smoke = float(np.random.normal(loc=550.0, scale=25.0))
                door_status = "BLOCKED"
            elif (x, y) in [(2, 4), (4, 4), (3, 3), (3, 6)]:
                # Zona dispersi asap tebal
                temp = float(np.random.normal(loc=42.0, scale=4.0))
                smoke = float(np.random.normal(loc=210.0, scale=30.0))

            # Sematan inkonsistensi string pada sebagian rekaman
            cased_ftype = ftype
            if np.random.rand() < 0.15:
                cased_ftype = ftype.upper()
            elif np.random.rand() < 0.15:
                cased_ftype = ftype.lower()

            # Sematan missing values terkontrol (~4% MCAR)
            val_temp = temp if np.random.rand() > 0.04 else np.nan
            val_smoke = smoke if np.random.rand() > 0.04 else np.nan

            records.append({
                "node_id": f"NODE-{x:02d}-{y:02d}",
                "grid_x": x,
                "grid_y": y,
                "facility_type": cased_ftype,
                "temp_c": np.round(val_temp, 2) if not np.isnan(val_temp) else np.nan,
                "smoke_ppm": np.round(val_smoke, 1) if not np.isnan(val_smoke) else np.nan,
                "door_status": door_status,
                "has_ramp": has_ramp,
                "occupancy": int(np.random.poisson(lam=12))
            })

    df = pd.DataFrame(records)
    df.to_csv(csv_path, index=False)
    return df


def clean_telemetry_data(df_raw: pd.DataFrame, clean_path: Path) -> pd.DataFrame:
    """Melakukan standardisasi string dan imputasi terkelompok."""
    df = df_raw.copy()

    # 1. Normalisasi string facility_type
    df["facility_type"] = df["facility_type"].astype(str).str.strip().str.title()
    df["door_status"] = df["door_status"].astype(str).str.strip().str.upper()

    # 2. Imputasi median terkelompok berdasarkan tipe fasilitas
    df["temp_c"] = df.groupby("facility_type")["temp_c"].transform(
        lambda s: s.fillna(s.median())
    )
    df["smoke_ppm"] = df.groupby("facility_type")["smoke_ppm"].transform(
        lambda s: s.fillna(s.median())
    )

    df.to_csv(clean_path, index=False)
    return df


def plot_campus_evacuation_map(df_clean: pd.DataFrame, output_img: Path) -> None:
    """Menghasilkan peta visualisasi grid termal dan titik kumpul aman kampus."""
    grid_temp = np.zeros((10, 10))
    grid_smoke = np.zeros((10, 10))

    for _, row in df_clean.iterrows():
        gx = int(row["grid_x"])
        gy = int(row["grid_y"])
        grid_temp[gy, gx] = row["temp_c"]  # Matplotlib origin: row=y, col=x
        grid_smoke[gy, gx] = row["smoke_ppm"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Panel 1: Distribusi Suhu Ruang
    c1 = ax1.imshow(grid_temp, origin='lower', cmap='YlOrRd', vmin=20, vmax=90)
    fig.colorbar(c1, ax=ax1, label='Suhu (deg C)')
    ax1.set_title('Peta Suhu Grid Kampus ITERA (Thermal Heatmap)')
    ax1.set_xlabel('Koordinat X (Grid)')
    ax1.set_ylabel('Koordinat Y (Grid)')

    # Tandai titik api dan titik kumpul aman
    ax1.scatter([3, 3], [4, 5], color='black', marker='x', s=120, label='Titik Api (Fire Hazard)')
    ax1.scatter([0, 9], [9, 9], color='green', marker='s', s=140, label='Titik Kumpul (Safe Assembly)')
    ax1.legend(loc='lower left', fontsize=9)

    # Panel 2: Distribusi Kerapatan Asap (Smoke Density)
    c2 = ax2.imshow(grid_smoke, origin='lower', cmap='Purples', vmin=0, vmax=600)
    fig.colorbar(c2, ax=ax2, label='Kerapatan Asap (PPM)')
    ax2.set_title('Peta Kerapatan Asap & Gas Beracun (Smoke PPM)')
    ax2.set_xlabel('Koordinat X (Grid)')
    ax2.set_ylabel('Koordinat Y (Grid)')

    ax2.scatter([3, 3], [4, 5], color='red', marker='x', s=120, label='Pusat Kebakaran')
    ax2.scatter([0, 9], [9, 9], color='green', marker='s', s=140, label='Zona Aman')
    ax2.legend(loc='lower left', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_img, dpi=200)
    plt.close()


def main() -> None:
    data_dir = Path(__file__).resolve().parent / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    raw_csv = data_dir / "campus_telemetry_raw.csv"
    clean_csv = data_dir / "campus_telemetry_clean.csv"
    map_png = data_dir / "campus_evacuation_map.png"

    df_raw = generate_campus_telemetry_raw(raw_csv)
    print(f"Dataset mentah digenerasi: {raw_csv.name} ({len(df_raw)} baris)")
    print(f"Nilai hilang awal suhu : {df_raw['temp_c'].isnull().sum()}")
    print(f"Nilai hilang awal asap : {df_raw['smoke_ppm'].isnull().sum()}")

    df_clean = clean_telemetry_data(df_raw, clean_csv)
    print(f"\nDataset bersih disimpan : {clean_csv.name}")
    print(f"Sisa nilai hilang suhu : {df_clean['temp_c'].isnull().sum()} (Tuntas)")
    print(f"Sisa nilai hilang asap : {df_clean['smoke_ppm'].isnull().sum()} (Tuntas)")

    plot_campus_evacuation_map(df_clean, map_png)
    print(f"Peta grid diekspor ke   : {map_png.name} (Headless Agg Mode)")


if __name__ == "__main__":
    main()
