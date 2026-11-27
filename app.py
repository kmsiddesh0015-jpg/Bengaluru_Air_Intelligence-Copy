
import os
import requests
import pandas as pd
import numpy as np
import gradio as gr
import joblib
import matplotlib.pyplot as plt

from station_intelligence import refresh_station_intelligence
from citizen_dashboard import build_complete_citizen_dashboard


from datetime import timedelta
from datetime import datetime
from zoneinfo import ZoneInfo

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(
    BASE_DIR, "Forecasting_Features_and_Targets.csv"
)

MODEL_PATH = os.path.join(
    BASE_DIR, "AQI_24_Hour_Enhanced_XGBoost.pkl"
)

LIVE_LOG = os.path.join(
    BASE_DIR, "Live_City_AQI_History.csv"
)

# Historical forecasting model
data = pd.read_csv(DATA_PATH, parse_dates=["Timestamp"])
bundle = joblib.load(MODEL_PATH)

model = bundle["model"]
imputer = bundle["imputer"]
features = bundle["features"]
encoded_columns = bundle["encoded_columns"]

stations = sorted(data["Device"].dropna().unique().tolist())

latest_data = (
    data.sort_values("Timestamp")
    .groupby("Device")
    .tail(1)
)

def forecast_aqi(station, horizon):

    rows = latest_data[latest_data["Device"] == station]

    if rows.empty:
        return "No forecasting data available."

    row = rows.iloc[-1].copy()

    current = float(row["current_aqi"])
    timestamp = pd.Timestamp(row["Timestamp"])

    hours = {
        "1 Hour": 1,
        "6 Hours": 6,
        "24 Hours": 24
    }

    if horizon == "24 Hours":

        row["aqi_change_1h"] = (
            row["current_aqi"] - row["aqi_lag_1"]
        )
        row["aqi_change_3h"] = (
            row["current_aqi"] - row["aqi_lag_3"]
        )
        row["aqi_change_6h"] = (
            row["current_aqi"] - row["aqi_lag_6"]
        )
        row["aqi_change_12h"] = (
            row["current_aqi"] - row["aqi_lag_12"]
        )
        row["station_city_difference"] = (
            row["current_aqi"] - row["city_aqi_mean"]
        )
        row["station_city_median_difference"] = (
            row["current_aqi"] - row["city_aqi_median"]
        )

        X = pd.DataFrame(
            [row[features].to_dict()],
            columns=features
        )

        X = pd.get_dummies(
            X, columns=["Device"], dtype=float
        )

        X = X.reindex(
            columns=encoded_columns,
            fill_value=0
        )

        X = X.replace([np.inf, -np.inf], np.nan)
        X = imputer.transform(X)

        predicted_change = float(model.predict(X)[0])
        predicted = max(0, current + predicted_change)

        model_name = "Enhanced XGBoost"

    else:
        predicted = current
        model_name = "Persistence"

    change = predicted - current
    forecast_time = timestamp + timedelta(hours=hours[horizon])

    # Actual age, rather than age relative to the historical dataset
    now = pd.Timestamp.now()
    age = (now - timestamp).total_seconds() / 3600

    if age > 48:
        freshness = (
            f"WARNING: This observation is approximately "
            f"{age:.0f} hours old. It is historical data, "
            "not a current AQI reading."
        )
    else:
        freshness = (
            f"Observation age: {age:.1f} hours. "
            "Check the timestamp before operational use."
        )

    return f"""
# Bengaluru Smart AQI Forecast

**Station:** {station}

**Forecast horizon:** {horizon}

## AQI Prediction

- Recorded AQI: {current:.1f}
- Predicted AQI: {predicted:.1f}
- Expected AQI change: {change:+.1f}

## Forecast Information

- Observation time: {timestamp}
- Forecast target time: {forecast_time}
- Forecasting method: {model_name}

## Data Freshness

{freshness}

**Important:** Historical station data is used here.
The 1-hour and 6-hour forecasts use persistence.
The 24-hour forecast uses Enhanced XGBoost.
"""

# Live city-level modelled AQI
def get_aqi_category(aqi):

    if aqi is None or pd.isna(aqi):
        return "Unavailable"
    if aqi <= 50:
        return "Good"
    if aqi <= 100:
        return "Moderate"
    if aqi <= 150:
        return "Unhealthy for Sensitive Groups"
    if aqi <= 200:
        return "Unhealthy"
    if aqi <= 300:
        return "Very Unhealthy"
    return "Hazardous"


def fetch_live_city_aqi():

    url = "https://air-quality-api.open-meteo.com/v1/air-quality"

    params = {
        "latitude": 12.9716,
        "longitude": 77.5946,
        "current": "us_aqi,pm2_5,pm10",
        "hourly": "us_aqi,pm2_5,pm10",
        "forecast_days": 2,
        "timezone": "Asia/Kolkata"
    }

    try:
        response = requests.get(
            url, params=params, timeout=30
        )
        response.raise_for_status()

        payload = response.json()
        current = payload["current"]
        hourly = payload["hourly"]

        model_time = pd.Timestamp(current["time"])
        fetched_at = datetime.now(
            ZoneInfo("Asia/Kolkata")
        )

        forecast = pd.DataFrame({
            "forecast_time": pd.to_datetime(hourly["time"]),
            "modelled_us_aqi": hourly["us_aqi"],
            "pm2_5": hourly["pm2_5"],
            "pm10": hourly["pm10"]
        })

        forecast = forecast[
            forecast["forecast_time"] > model_time
        ].copy()

        forecast = forecast.sort_values("forecast_time")

        # Save successful current readings as snapshots
        record = pd.DataFrame([{
            "snapshot_time": fetched_at.isoformat(),
            "model_time": current["time"],
            "modelled_us_aqi": current.get("us_aqi"),
            "pm2_5": current.get("pm2_5"),
            "pm10": current.get("pm10"),
            "latitude": 12.9716,
            "longitude": 77.5946,
            "location_label": "Bengaluru City Centre",
            "data_type": "Modelled city-level estimate"
        }])

        if os.path.exists(LIVE_LOG):
            old = pd.read_csv(LIVE_LOG)
            history = pd.concat(
                [old, record], ignore_index=True
            )
        else:
            history = record

        history = history.drop_duplicates(
            subset=["model_time"], keep="last"
        )

        history.to_csv(LIVE_LOG, index=False)

        return {
            "success": True,
            "current": current,
            "model_time": model_time,
            "fetched_at": fetched_at,
            "forecast": forecast
        }

    except Exception as error:
        return {
            "success": False,
            "message": str(error)
        }


def refresh_live_dashboard():

    result = fetch_live_city_aqi()

    if not result["success"]:
        return (
            "## Live AQI unavailable\n\n" + result["message"],
            None,
            pd.DataFrame()
        )

    current = result["current"]
    forecast = result["forecast"]

    aqi = current.get("us_aqi")
    category = get_aqi_category(aqi)

    summary = f"""
# Bengaluru Live Air Quality

**Location:** Bengaluru City Centre

**Data type:** Modelled city-level estimate

**Model timestamp:** {result["model_time"]}

**Last successful fetch:** {result["fetched_at"].strftime("%d %b %Y, %I:%M:%S %p %Z")}

# Current US AQI: {aqi} — {category}

| Measurement | Current value |
|---|---:|
| US AQI | {aqi} |
| PM2.5 | {current.get("pm2_5")} µg/m³ |
| PM10 | {current.get("pm10")} µg/m³ |

**AQI category reference:** Good (0–50), Moderate (51–100),
Unhealthy for Sensitive Groups (101–150), Unhealthy (151–200),
Very Unhealthy (201–300), Hazardous (301+).

*This is a modelled city-level estimate, not a physical monitoring-station reading.*
"""

    fig, ax = plt.subplots(figsize=(11, 4))

    if not forecast.empty:
        ax.plot(
            forecast["forecast_time"],
            forecast["modelled_us_aqi"],
            marker="o"
        )

    ax.set_title("Bengaluru Future Modelled AQI Forecast")
    ax.set_xlabel("Forecast time")
    ax.set_ylabel("US AQI")
    ax.grid(True, alpha=0.3)

    fig.autofmt_xdate()
    fig.tight_layout()

    return summary, fig, forecast


# Unified application
with gr.Blocks(
    title="Bengaluru Smart AQI Forecasting System"
) as app:

    gr.Markdown("""
    # Bengaluru Smart AQI Forecasting System

    Area-wise historical station forecasting and live city-level
    modelled air-quality forecasts.
    """)

    with gr.Tabs():

        with gr.Tab("Station Forecasting"):

            gr.Markdown("""
            ## Historical Monitoring Station Forecasts

            Forecasts for 13 stations using the trained project models.
            """)

            with gr.Row():
                station = gr.Dropdown(
                    choices=stations,
                    value=stations[0],
                    label="Select Monitoring Station"
                )

                horizon = gr.Dropdown(
                    choices=["1 Hour", "6 Hours", "24 Hours"],
                    value="24 Hours",
                    label="Forecast Horizon"
                )

            button = gr.Button(
                "Generate Station Forecast",
                variant="primary"
            )

            station_output = gr.Markdown()

            button.click(
                forecast_aqi,
                inputs=[station, horizon],
                outputs=station_output
            )

        with gr.Tab("Live Bengaluru AQI"):

            gr.Markdown("""
            ## Live City-Level Air Quality

            External modelled estimates for Bengaluru city centre.
            These values are separate from the 13 station models.
            """)

            refresh_button = gr.Button(
                "Refresh Live AQI",
                variant="primary"
            )

            live_summary = gr.Markdown()
            live_plot = gr.Plot(label="Future AQI Forecast")

            live_table = gr.Dataframe(
                label="Future Hourly Forecast",
                interactive=False
            )

            refresh_button.click(
                refresh_live_dashboard,
                outputs=[
                    live_summary,
                    live_plot,
                    live_table
                ]
            )

            app.load(
                refresh_live_dashboard,
                outputs=[
                    live_summary,
                    live_plot,
                    live_table
                ]
            )


        with gr.Tab("Station Intelligence"):

            gr.Markdown("## Bengaluru Station Intelligence")
            gr.Markdown(
                "Latest provider-reported station readings, "
                "AQI availability and data freshness."
            )

            station_refresh = gr.Button(
                "Refresh Station Intelligence",
                variant="primary"
            )

            station_summary = gr.Markdown()
            station_table = gr.Dataframe(
                label="Latest Station Status",
                interactive=False
            )
            station_plot = gr.Plot(label="Station AQI Overview")

            station_refresh.click(
                lambda: refresh_station_intelligence(BASE_DIR),
                outputs=[
                    station_summary,
                    station_table,
                    station_plot
                ]
            )

            app.load(
                lambda: refresh_station_intelligence(BASE_DIR),
                outputs=[
                    station_summary,
                    station_table,
                    station_plot
                ]
            )

        with gr.Tab("Citizen Insights"):

            gr.Markdown("## Bengaluru Citizen Air-Quality Insights")

            citizen_button = gr.Button(
                "Generate Citizen Insights",
                variant="primary"
            )

            citizen_summary = gr.Markdown()
            citizen_table = gr.Dataframe(
                label="Station AQI Ranking",
                interactive=False
            )
            citizen_plot = gr.Plot(
                label="Station AQI Comparison"
            )

            gr.Markdown("### Bengaluru AQI Hotspot Map")
            citizen_map = gr.HTML()

            citizen_pollutant_plot = gr.Plot(
                label="Station-wise Pollutant Sub-index Comparison"
            )

            gr.Markdown("### Automated Air-Quality Insights")
            citizen_ai_insights = gr.Markdown()

            citizen_outputs = [
                citizen_summary,
                citizen_table,
                citizen_plot,
                citizen_map,
                citizen_pollutant_plot,
                citizen_ai_insights
            ]

            citizen_button.click(
                lambda: build_complete_citizen_dashboard(BASE_DIR),
                outputs=citizen_outputs
            )

            app.load(
                lambda: build_complete_citizen_dashboard(BASE_DIR),
                outputs=citizen_outputs
            )


if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
