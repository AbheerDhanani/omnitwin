#!/usr/bin/env python3
"""
Generate realistic historical environmental dataset for Pune Urban Area (2022-2024).
Creates:
  1. data/omnitwin_pune_historical.csv (for Streamlit app & Python engine)
  2. public/data.json (for instant Vercel web deployment)
"""

import json
import os
import numpy as np
import pandas as pd

# Define 5 distinct Pune neighborhoods with coordinates and localized source weights
STATIONS = {
    "Shivajinagar": {
        "lat": 18.5308,
        "lon": 73.8475,
        "traffic_mult": 1.25,
        "industrial_mult": 0.45,
        "dust_mult": 0.85,
        "description": "Dense urban commercial core & heavy transport junction"
    },
    "Hadapsar": {
        "lat": 18.5089,
        "lon": 73.9260,
        "traffic_mult": 1.10,
        "industrial_mult": 1.35,
        "dust_mult": 1.05,
        "description": "Eastern corridor with manufacturing units & IT corridors"
    },
    "Pimpri-Chinchwad": {
        "lat": 18.6298,
        "lon": 73.7997,
        "traffic_mult": 1.00,
        "industrial_mult": 1.95,
        "dust_mult": 0.90,
        "description": "Heavy industrial & automotive manufacturing hub"
    },
    "Katraj": {
        "lat": 18.4575,
        "lon": 73.8677,
        "traffic_mult": 0.90,
        "industrial_mult": 0.40,
        "dust_mult": 1.55,
        "description": "Southern highway junction with active construction & dust"
    },
    "Kothrud": {
        "lat": 18.5074,
        "lon": 73.8077,
        "traffic_mult": 0.85,
        "industrial_mult": 0.30,
        "dust_mult": 0.70,
        "description": "Western residential sector with high canopy cover"
    },
}


def generate_dataset(seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2022-01-01", "2024-12-31", freq="D")
    n = len(dates)
    doy = dates.dayofyear.values

    # Meteorological seasonality in Pune (Deccan Plateau)
    # Winter inversion (Nov-Feb): doy < 60 or doy > 305
    winter = np.clip(np.cos(2 * np.pi * (doy - 15) / 365), 0, None) ** 1.5
    # Monsoon season (June to September: doy 155 to 275)
    monsoon = np.exp(-0.5 * ((doy - 215) / 35) ** 2)

    # Precipitation (monsoon concentrated)
    rain_prob = 0.04 + 0.65 * monsoon
    rain_amount = rng.gamma(shape=2.2, scale=5.0, size=n) * (rng.random(n) < rain_prob)

    # Ambient Temperature (°C)
    temp = 25.5 + 7.5 * np.cos(2 * np.pi * (doy - 130) / 365) + rng.normal(0, 1.2, n)

    # Relative Humidity (%)
    humidity = np.clip(
        42.0 + 40.0 * monsoon + 15.0 * (rain_amount > 1.0) - 10.0 * winter + rng.normal(0, 5.0, n),
        15.0,
        98.0
    )

    # Wind speed (km/h) - higher during monsoon, lower/calm during winter
    wind = np.clip(
        7.5 + 4.5 * monsoon - 2.8 * winter + rng.normal(0, 2.0, n),
        1.2,
        28.0
    )

    weekday = (dates.dayofweek.values < 5).astype(float)
    # Dry dust proxy: high wind speed on dry days with low humidity
    dust_proxy = wind * (1.0 - humidity / 100.0) * (rain_amount < 0.5)

    # Dilution factor: thermal inversion in winter traps pollutants; monsoon washes them out
    dilution_trap = np.clip(1.0 + 0.65 * winter - 0.45 * monsoon - 0.025 * (wind - 7.0), 0.35, 2.1)
    wet_scavenging = np.exp(-0.07 * rain_amount)

    frames = []
    for station_name, meta in STATIONS.items():
        tm = meta["traffic_mult"]
        im = meta["industrial_mult"]
        dm = meta["dust_mult"]

        # Traffic index (0-100 scale, higher on weekdays and rush hours)
        traffic_idx = np.clip(
            (56.0 + 18.0 * weekday) * tm + rng.normal(0, 5.0, n) - 0.5 * rain_amount,
            10.0,
            120.0
        )

        # Industrial activity index (relatively stable daily baseline with small noise)
        industrial_idx = np.clip(
            52.0 * im + 4.0 * weekday + rng.normal(0, 2.5, n),
            5.0,
            130.0
        )

        # PM2.5 calculation based on physics & emissions formula
        traffic_contrib = 0.32 * traffic_idx
        industrial_contrib = 0.24 * industrial_idx
        dust_contrib = 4.2 * dm * dust_proxy
        regional_bg = 16.0

        raw_emissions = traffic_contrib + industrial_contrib + dust_contrib + regional_bg
        pm25_expected = raw_emissions * dilution_trap * wet_scavenging

        # Autoregressive persistence
        ar_noise = np.zeros(n)
        for t in range(1, n):
            ar_noise[t] = 0.62 * ar_noise[t - 1] + rng.normal(0, 4.5)

        pm25 = np.clip(pm25_expected + ar_noise, 8.0, 320.0)

        df_station = pd.DataFrame({
            "date": dates.strftime("%Y-%m-%d"),
            "station": station_name,
            "lat": meta["lat"],
            "lon": meta["lon"],
            "pm25": pm25.round(1),
            "temp": temp.round(1),
            "humidity": humidity.round(1),
            "wind": wind.round(1),
            "rain": rain_amount.round(1),
            "traffic_index": traffic_idx.round(1),
            "industrial_index": industrial_idx.round(1),
        })
        frames.append(df_station)

    df_all = pd.concat(frames, ignore_index=True)
    return df_all


def export_for_vercel(df: pd.DataFrame, output_json_path: str):
    """Exports compact JSON dataset optimized for fast client-side loading on Vercel."""
    stations_data = {}
    for st_name, meta in STATIONS.items():
        st_df = df[df.station == st_name].sort_values("date")
        stations_data[st_name] = {
            "meta": meta,
            "records": st_df.to_dict(orient="records")
        }

    # City average
    avg_df = df.groupby("date").agg({
        "pm25": "mean",
        "temp": "mean",
        "humidity": "mean",
        "wind": "mean",
        "rain": "mean",
        "traffic_index": "mean",
        "industrial_index": "mean",
    }).round(1).reset_index()

    avg_df["station"] = "City average"
    avg_df["lat"] = 18.5204
    avg_df["lon"] = 73.8567

    stations_data["City average"] = {
        "meta": {
            "lat": 18.5204,
            "lon": 73.8567,
            "description": "Urban composite average across all 5 monitored neighborhoods"
        },
        "records": avg_df.to_dict(orient="records")
    }

    full_payload = {
        "area": "Pune Urban Region",
        "stations": STATIONS,
        "stationList": ["City average"] + list(STATIONS.keys()),
        "data": stations_data,
        "defaultReplayDate": "2024-11-15",
        "assumptions": {
            "attribution": "Assumption: Industrial baseline is constant; traffic scales with rush hour; dust correlates with wind speed.",
            "actions": [
                {"id": "action1", "name": "Action 1: Odd-Even Traffic Rule", "source": "Vehicular", "defaultCut": 0.30},
                {"id": "action2", "name": "Action 2: Halt Heavy Industry", "source": "Industrial", "defaultCut": 0.80},
                {"id": "action3", "name": "Action 3: Mandate Construction Sprinklers", "source": "Dust/Weather", "defaultCut": 0.40}
            ]
        }
    }

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(full_payload, f, separators=(',', ':'))
    print(f"Exported Vercel data JSON to: {output_json_path} ({os.path.getsize(output_json_path) // 1024} KB)")


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "data")
    public_dir = os.path.join(base_dir, "public")

    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(public_dir, exist_ok=True)

    csv_path = os.path.join(data_dir, "omnitwin_pune_historical.csv")
    json_path = os.path.join(public_dir, "data.json")

    print("Generating Pune urban environmental dataset (2022-2024)...")
    df = generate_dataset()
    df.to_csv(csv_path, index=False)
    print(f"Saved CSV dataset to: {csv_path} ({len(df)} rows across {df['station'].nunique()} stations)")

    export_for_vercel(df, json_path)
    print("Dataset generation complete!")
