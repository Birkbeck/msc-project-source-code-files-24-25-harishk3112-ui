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
            "Expected '_backtest_recent_arima' or 'backtest_recent_arima' "
        )
    sig = inspect.signature(fn)
    params = list(sig.parameters.keys())
    return fn, params

def _call_arima_backtest(n_hours, within_pct, arima_order=(2, 1, 2)):
    """
    Call ARIMA backtest using whichever signature your file defines.
    """
    fn, params = _find_arima_fn()

    kwargs = {}
    if "n_hours" in params:
        kwargs["n_hours"] = n_hours
    if "order" in params:
        kwargs["order"] = arima_order
    # prefer 'within_pct' if available; otherwise try 'pct'
    if "within_pct" in params:
        kwargs["within_pct"] = within_pct
    elif "pct" in params:
        kwargs["pct"] = within_pct

    return fn(**kwargs)