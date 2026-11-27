
BENGALURU SMART AQI FORECASTING SYSTEM
======================================

PROJECT OVERVIEW
----------------
A two-part AQI forecasting application:

1. Historical monitoring station forecasting across 13 stations.
2. Live Bengaluru city-centre modelled air-quality estimates.

STATION FORECASTING
-------------------
1-hour: Persistence baseline
6-hour: Persistence baseline
24-hour: Enhanced XGBoost

Previously evaluated independent test results:
1-hour  MAE: 1.9849 | RMSE: 3.8212 | R2: 0.9823
6-hour  MAE: 7.2169 | RMSE: 11.2783 | R2: 0.8505
24-hour MAE: 14.1341 | RMSE: 18.8733 | R2: 0.5853

LIVE CITY-LEVEL AIR QUALITY
---------------------------
Location: Bengaluru city centre
Coordinates: 12.9716, 77.5946
Provider: Open-Meteo Air Quality API
Variables: US AQI, PM2.5, PM10
Forecast: Hourly forecast records

IMPORTANT DATA DISTINCTION
--------------------------
Live city-level values are modelled estimates.
They are not physical monitoring-station measurements.

Live city-level estimates are kept separate from the
13-station historical forecasting models.

DATA LIMITATIONS
----------------
The historical station dataset ends on 17 October 2026.
Station observations are not currently updated automatically.
The dashboard displays data freshness warnings.

The 1-hour and 6-hour models use persistence.
The 24-hour model uses Enhanced XGBoost.
Future weather forecast inputs are not yet integrated
into the trained 24-hour station model.

The live city-level modelled forecast is a separate
external forecast and is not produced by the trained
station-level XGBoost model.

RUNNING THE APPLICATION
-----------------------
Install dependencies:

pip install -r requirements.txt

Run:

python app.py

FILES
-----
app.py
Forecasting_Features_and_Targets.csv
AQI_24_Hour_Enhanced_XGBoost.pkl
AQI_24_Hour_Enhanced_Features.pkl
AQI_24_Hour_Hybrid_XGBoost.pkl
AQI_24_Hour_Features.pkl
Cleaned_AQI_Weather_Data.csv
Final_Model_Performance.csv
Dashboard_Validation_Results.csv
README.txt
requirements.txt

Generated live audit and test files are also included
when available.
