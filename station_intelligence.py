
import os
import pandas as pd
import matplotlib.pyplot as plt

def refresh_station_intelligence(project_dir):
    path = os.path.join(project_dir, "Live_Station_AQI_History.csv")

    if not os.path.exists(path):
        raise FileNotFoundError("Live station history is missing.")

    df = pd.read_csv(path)
    if df.empty:
        raise ValueError("Station history is empty.")

    df["aqi"] = pd.to_numeric(df["aqi"], errors="coerce")

    # API observation timestamps are Indian local time when timezone-naive.
    timestamps = pd.to_datetime(df["observation_time"], errors="coerce")

    if timestamps.dt.tz is None:
        timestamps = timestamps.dt.tz_localize(
            "Asia/Kolkata", ambiguous="NaT", nonexistent="NaT"
        )

    df["observation_time"] = timestamps.dt.tz_convert("UTC")

    df = df.sort_values("observation_time")
    latest = df.groupby("station", as_index=False).tail(1).copy()

    now = pd.Timestamp.now(tz="UTC")
    latest["age_hours"] = (
        now - latest["observation_time"]
    ).dt.total_seconds() / 3600

    latest["freshness"] = latest["age_hours"].apply(
        lambda x: "Unknown" if pd.isna(x)
        else "Fresh" if 0 <= x <= 2
        else "Aging" if 2 < x <= 6
        else "Stale" if x > 6
        else "Future timestamp — verify source"
    )

    latest["AQI availability"] = latest["aqi"].apply(
        lambda x: "Available" if pd.notna(x) else "Unavailable"
    )

    columns = [
        c for c in [
            "station", "aqi", "category", "dominant_pollutant",
            "observation_time", "age_hours", "freshness",
            "AQI availability"
        ] if c in latest.columns
    ]

    report = (
        "### Station Intelligence Summary\n"
        f"- Stations found: **{len(latest)}**\n"
        f"- AQI available: **{latest['aqi'].notna().sum()}**\n"
        f"- AQI unavailable: **{latest['aqi'].isna().sum()}**\n"
        f"- Fresh readings (2 hours or less): "
        f"**{(latest['freshness'] == 'Fresh').sum()}**\n"
        f"- Stale readings (over 6 hours): "
        f"**{(latest['freshness'] == 'Stale').sum()}**\n\n"
        "*Freshness is calculated using India Standard Time correctly.*"
    )

    chart_data = latest.dropna(subset=["aqi"]).sort_values("aqi")
    fig, ax = plt.subplots(figsize=(10, 4))

    if not chart_data.empty:
        ax.barh(chart_data["station"], chart_data["aqi"])
        ax.set_xlabel("Provider-reported AQI")
        ax.set_title("Latest Station AQI")
    else:
        ax.text(0.5, 0.5, "No valid AQI readings",
                ha="center", va="center")

    fig.tight_layout()
    return report, latest[columns], fig
