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
