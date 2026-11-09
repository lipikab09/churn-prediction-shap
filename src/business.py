"""Profit maths, threshold search, sensitivity grid and gain curves."""
import numpy as np
import pandas as pd


def expected_profit(y_true, proba, thr, value, cost, success):
    """Contact everyone with proba >= thr.
    Gain = success * V_i for each contacted TRUE churner; cost = `cost` per contact."""
    y_true, proba, value = map(np.asarray, (y_true, proba, value))
    contact = proba >= thr
    saved = (contact & (y_true == 1)) * success * value
    return float(saved.sum() - contact.sum() * cost)


def profit_curve(y_true, proba, value, cost, success, thresholds):
    profits = np.array([expected_profit(y_true, proba, t, value, cost, success)
                        for t in thresholds])
    return np.asarray(thresholds), profits


def best_threshold(y_true, proba, value, cost, success, thresholds):
    thr, profits = profit_curve(y_true, proba, value, cost, success, thresholds)
    i = int(np.argmax(profits))
    return float(thr[i]), float(profits[i])


def sensitivity_grid(y_true, proba, value, costs, successes, thresholds):
    rows = []
    for c in costs:
        for s in successes:
            t, p = best_threshold(y_true, proba, value, c, s, thresholds)
            rows.append(dict(cost=c, success=s, best_threshold=t, best_profit=p))
    return pd.DataFrame(rows)


def theoretical_threshold(value, cost, success):
    """With calibrated probabilities, contacting customer i pays off iff
    p_i * success * V_i > cost, i.e. p_i > cost / (success * V_i)."""
    return cost / (success * np.asarray(value))


def gain_curve(y_true, proba):
    y_true = np.asarray(y_true)
    order = np.argsort(-np.asarray(proba))
    cum = np.cumsum(y_true[order]) / y_true.sum()
    frac = np.arange(1, len(y_true) + 1) / len(y_true)
    return frac, cum


def capture_at(y_true, proba, fraction=0.2):
    frac, cum = gain_curve(y_true, proba)
    return float(cum[np.searchsorted(frac, fraction) - 1])
