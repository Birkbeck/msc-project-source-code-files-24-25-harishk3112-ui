# LSTM  vs ARIMA for 24h, 72h, 168h
import inspect
from pprint import pprint

import model as lstm_mod
import arima_model as arima_mod

def _find_arima_fn():
    """
    Return the ARIMA backtest function and a dict describing its parameters.
    """
    fn = getattr(arima_mod, "_backtest_recent_arima", None)
    if fn is None:
        fn = getattr(arima_mod, "backtest_recent_arima", None)
    if fn is None:
        raise AttributeError(
            "Could not find ARIMA backtest function. "
            "Expected '_backtest_recent_arima' or 'backtest_recent_arima' in model.py"
        )
    sig = inspect.signature(fn)
    params = list(sig.parameters.keys())
    return fn, params