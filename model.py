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