
import os
import pandas as pd
import folium

def build_citizen_hotspot_map(project_dir):
    path = os.path.join(project_dir, "Live_Station_AQI_History.csv")

    if not os.path.exists(path):
        raise FileNotFoundError(
            "Live_Station_AQI_History.csv is required for the hotspot map."
        )

    df = pd.read_csv(path)

    required = ["station", "latitude", "longitude", "aqi"]
    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(
            "Station history is missing map fields: " + ", ".join(missing)
        )

    df["aqi"] = pd.to_numeric(df["aqi"], errors="coerce")
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")

    latest = (
        df.sort_values("updated")
        .groupby("station", as_index=False)
        .tail(1)
    )

    latest = latest.dropna(subset=["latitude", "longitude"])

    if latest.empty:
        raise ValueError("No valid station coordinates are available.")

    center = [
        latest["latitude"].mean(),
        latest["longitude"].mean()
    ]

    map_view = folium.Map(
        location=center,
        zoom_start=11,
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}",
        attr="Esri World Street Map"
    )

    for _, row in latest.iterrows():
        aqi = row["aqi"]

        if pd.isna(aqi):
            color = "gray"
            label = "AQI unavailable"
        elif aqi <= 50:
            color = "green"
            label = "Good"
        elif aqi <= 100:
            color = "orange"
            label = "Moderate"
        elif aqi <= 200:
            color = "red"
            label = "Poor"
        else:
            color = "darkred"
            label = "Very Poor"

        popup = (
            f"<b>{row['station']}</b><br>"
            f"AQI: {aqi if pd.notna(aqi) else 'Unavailable'}<br>"
            f"Category: {label}<br>"
            f"Updated: {row.get('updated', 'Unknown')}"
        )

        folium.CircleMarker(
            location=[row["latitude"], row["longitude"]],
            radius=9,
            color=color,
            fill=True,
            fill_opacity=0.8,
            popup=folium.Popup(popup, max_width=300)
        ).add_to(map_view)

    bounds = [
        [latest["latitude"].min(), latest["longitude"].min()],
        [latest["latitude"].max(), latest["longitude"].max()]
    ]

    map_view.fit_bounds(bounds, max_zoom=13)

    return map_view
