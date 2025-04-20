import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"


def load_data():
    # Load the raw pedestrian data and sensors data
    ped = pd.read_csv(DATA_RAW / "pedestrian-counting-system-monthly-counts-per-hour.csv")
    sensors = pd.read_csv(DATA_RAW / "pedestrian-counting-system-sensor-locations.csv")

    return ped, sensors


def clean_data():
    # Filter the data to remove old entries
    ped, sensors = load_data()
    cutoff = pd.Timestamp('2020-01-01')
    ped["Sensing_Date"] = pd.to_datetime(ped["Sensing_Date"], errors='coerce')
    ped = (ped.drop_duplicates(["Sensor_Name", "Sensing_Date", "HourDay"]))
    ped_recent = ped.loc[ped["Sensing_Date"] >= cutoff].copy()

    return ped_recent, sensors


def make_recent_snapshot():
    ped, _ = clean_data()

    ped = ped[ped["Sensing_Date"] >= "2024-01-01"]
    ped_recent = (
        ped.sort_values(["Sensor_Name", "Sensing_Date"])
        .groupby("Sensor_Name")
        .tail(168)
    )

    out_path = Path(ROOT / "data/interim/pedestrian_recent.parquet")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ped_recent.to_parquet(out_path, index=False)
    print(f"Snapshot saved: {out_path}  ({len(ped_recent):,} rows)")

