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

    def _flatten_metrics(tag, m):
    return {
        "Model": tag,
        "Window(h)": m.get("window_hours"),
        "RMSE": m.get("RMSE"),
        "MAE": m.get("MAE"),
        "MAPE%": m.get("MAPE_%"),
        "sMAPE%": m.get("sMAPE_%"),
        "R2": m.get("R2"),
        "Within±10%": m.get("Within10pct_%"),
    }


def _widths(rows, headers):
    return {h: max(len(str(h)), *(len(str(r[h])) for r in rows)) for h in headers}


def _print_row(row, headers, widths):
    print("  ".join(str(row[h]).ljust(widths[h]) for h in headers))


def _divider(title):
    bar = "=" * 96
    print(f"\n{bar}\n{title}\n{bar}")

    def compare_all(eval_windows=(24, 72, 168), within_pct=10.0, arima_order=(2, 1, 2)):
    for n_hours in eval_windows:
        # LSTM backtest
        lstm_eval = lstm_mod._backtest_recent(n_hours=n_hours)
        if lstm_eval is None:
            # ensure trained/loaded, then retry
            lstm_mod.load_and_train_model(n_runs=1)
            lstm_eval = lstm_mod._backtest_recent(n_hours=n_hours)

        # ARIMA backtest 
        arima_eval = _call_arima_backtest(n_hours=n_hours, within_pct=within_pct, arima_order=arima_order)

        _divider(f"Evaluation Window: Last {n_hours} hours  |  Within ±{within_pct}%")

        if not lstm_eval or not arima_eval:
            print("Comparison not available for this window (missing evaluation).")
            continue

        lstm_row = _flatten_metrics("LSTM", lstm_eval["metrics"])
        arima_row = _flatten_metrics(f"ARIMA{arima_order}", arima_eval["metrics"])

        headers = ["Model", "Window(h)", "RMSE", "MAE", "MAPE%", "sMAPE%", "R2", "Within±10%"]
        widths = _widths([lstm_row, arima_row], headers)

        _print_row({h: h for h in headers}, headers, widths)
        _print_row({h: "-" * widths[h] for h in headers}, headers, widths)
        _print_row(lstm_row, headers, widths)
        _print_row(arima_row, headers, widths)


if __name__ == "__main__":
    # Runs all three windows in one go
    compare_all(eval_windows=(24, 72, 168), within_pct=10.0, arima_order=(2, 1, 2))