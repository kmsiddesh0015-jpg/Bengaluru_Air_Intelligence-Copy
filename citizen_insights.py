
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

POLLUTANTS = ["pm25", "pm10", "no2", "so2", "co", "o3", "nh3"]

def build_citizen_insights_with_pollutant_chart(project_dir):
    path = os.path.join(project_dir, "Live_Station_AQI_History.csv")

    if not os.path.exists(path):
        raise FileNotFoundError(
            "Live_Station_AQI_History.csv is missing."
        )

    df = pd.read_csv(path)

    if df.empty:
        raise ValueError("Station history contains no records.")

    df["aqi"] = pd.to_numeric(df["aqi"], errors="coerce")
    df["observation_time"] = pd.to_datetime(
        df["observation_time"], errors="coerce"
    )

    latest = (
        df.sort_values("observation_time")
        .groupby("station", as_index=False)
        .tail(1)
        .copy()
    )

    valid = latest.dropna(subset=["aqi"]).sort_values(
        "aqi", ascending=False
    )

    average = valid["aqi"].mean()
    highest = valid.iloc[0] if not valid.empty else None
    lowest = valid.iloc[-1] if not valid.empty else None

    summary = f"""
### Bengaluru Citizen Air-Quality Insights

- Stations monitored: **{len(latest)}**
- Stations with valid AQI: **{len(valid)}**
- Stations without AQI: **{latest['aqi'].isna().sum()}**
- Average reported AQI: **{average:.1f}** if pd.notna(average) else "Unavailable"
- Highest AQI: **{highest['aqi']} at {highest['station']}** if highest is not None else "Unavailable"
- Lowest AQI: **{lowest['aqi']} at {lowest['station']}** if lowest is not None else "Unavailable"

*These are provider-reported station readings, not predictions.*
"""

    table = valid.reset_index(drop=True)

    fig1, ax1 = plt.subplots(figsize=(10, 5))
    if not valid.empty:
        ax1.barh(valid["station"], valid["aqi"])
        ax1.invert_yaxis()
    ax1.set_xlabel("Reported AQI")
    ax1.set_title("Station AQI Ranking")
    fig1.tight_layout()

    pollutant_columns = [
        f"{p}_avg" for p in POLLUTANTS
        if f"{p}_avg" in latest.columns
    ]

    fig2, ax2 = plt.subplots(figsize=(10, 5))
    if pollutant_columns:
        matrix = latest.set_index("station")[pollutant_columns]
        matrix.columns = [c.replace("_avg", "").upper()
                          for c in matrix.columns]
        image = ax2.imshow(matrix, aspect="auto", cmap="YlOrRd")
        ax2.set_yticks(range(len(matrix.index)))
        ax2.set_yticklabels(matrix.index)
        ax2.set_xticks(range(len(matrix.columns)))
        ax2.set_xticklabels(matrix.columns)
        fig2.colorbar(image, ax=ax2, label="Provider-reported sub-index")
    else:
        ax2.text(0.5, 0.5, "Pollutant data unavailable",
                 ha="center", va="center")
    ax2.set_title("Station-wise Pollutant Comparison")
    fig2.tight_layout()

    return summary, table, fig1, fig2


def build_citizen_insights_with_chart(project_dir):
    summary, table, chart, _ = (
        build_citizen_insights_with_pollutant_chart(project_dir)
    )
    return summary, table, chart


def build_citizen_insights(project_dir):
    summary, table, chart, _ = (
        build_citizen_insights_with_pollutant_chart(project_dir)
    )
    plt.close(chart)
    return summary, table


def generate_air_quality_insights(project_dir):
    path = os.path.join(project_dir, "Live_Station_AQI_History.csv")
    if not os.path.exists(path):
        return "Station history is not available yet."

    df = pd.read_csv(path)
    if df.empty:
        return "No station readings are available."

    df["aqi"] = pd.to_numeric(df["aqi"], errors="coerce")
    latest = (
        df.sort_values("updated")
        .groupby("station", as_index=False)
        .tail(1)
    )
    valid = latest.dropna(subset=["aqi"])

    if valid.empty:
        return "No valid station AQI readings are available."

    highest = valid.loc[valid["aqi"].idxmax()]
    lowest = valid.loc[valid["aqi"].idxmin()]

    return (
        "### Automated Air-Quality Insights\n"
        f"- Highest reported AQI: **{highest['aqi']}** "
        f"at **{highest['station']}**\n"
        f"- Lowest reported AQI: **{lowest['aqi']}** "
        f"at **{lowest['station']}**\n"
        f"- Valid station readings: **{len(valid)}**\n\n"
        "*Automatically generated descriptive insights, not an AI prediction.*"
    )
