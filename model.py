import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
import warnings
warnings.filterwarnings("ignore")

def load_and_train_model():
    df = pd.read_csv("data/household_power_consumption.txt",
                     sep=';',
                     parse_dates=[[0, 1]],
                     infer_datetime_format=True,
                     low_memory=False,
                     na_values=['?'])

    df.rename(columns={"Date_Time": "datetime"}, inplace=True)
    df.set_index("datetime", inplace=True)
    df["Global_active_power"] = pd.to_numeric(df["Global_active_power"], errors="coerce")
    df.dropna(subset=["Global_active_power"], inplace=True)
    df = df.resample("H").mean()

    history = df["Global_active_power"][-24:].round(2)

    model = ARIMA(df["Global_active_power"], order=(2, 1, 2))
    model_fit = model.fit()

    return model_fit, history

model_fit, history_data = load_and_train_model()

def get_forecast(user_input):
    forecast_days = int(user_input.get("forecast_days", 1))
    household_size = int(user_input.get("household_size", 1))
    preference = user_input.get("preference", "normal")
    peak_hours = user_input.get("peak_hours", [])

     scale = 1.0
    if preference == "high":
        scale = 1.3
    elif preference == "low":
        scale = 0.8

    
    steps = forecast_days * 24
    forecast = model_fit.forecast(steps=steps)
    forecast_scaled = forecast * household_size * scale
    forecast_scaled = forecast_scaled.round(2)
