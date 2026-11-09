"""Step 9: one-time evaluation on the held-out TEST set + fairness audit.

Run:  python -m src.evaluate
"""
import json
from functools import partial

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from fairlearn.metrics import MetricFrame, count, selection_rate
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)

from . import config as cfg
from .business import capture_at, expected_profit, gain_curve
from .preprocessing import get_splits


def main():
    cfg.ensure_dirs()
    s = get_splits()
    X, y, sens = s["X_test"], s["y_test"].values, s["s_test"]
    cal = joblib.load(cfg.MODELS_DIR / "calibrated_model.joblib")
    meta = json.loads((cfg.MODELS_DIR / "config.json").read_text())
    thr = meta["threshold"]          # chosen on VALIDATION data, never on test
    a = meta["assumptions"]

    p = cal.predict_proba(X)[:, 1]
    pred = (p >= thr).astype(int)
    value = X["MonthlyCharges"].values * a["expected_months"]
    prof = lambda t: expected_profit(y, p, t, value, a["retention_cost"], a["offer_success_rate"])

    tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
    res = dict(
        threshold=thr,
        always_no_accuracy=float((y == 0).mean()),
        pr_auc=average_precision_score(y, p),
        roc_auc=roc_auc_score(y, p),
        recall=recall_score(y, pred),
        precision=precision_score(y, pred, zero_division=0),
        f1=f1_score(y, pred),
        brier=brier_score_loss(y, p),
        confusion=dict(tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp)),
        profit_at_optimal=prof(thr),
        profit_at_0_5=prof(0.5),
        profit_contact_all=prof(0.0),
        top20pct_captures_churners=capture_at(y, p, 0.2),
    )
    res["uplift_vs_0_5_pct"] = ((res["profit_at_optimal"] - res["profit_at_0_5"])
                                / abs(res["profit_at_0_5"]) * 100
                                if res["profit_at_0_5"] else None)

    # ---- gain / lift chart ----
    frac, cum = gain_curve(y, p)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(frac, cum, label="model")
    ax.plot([0, 1], [0, 1], "k--", label="random")
    ax.axvline(0.2, color="grey", ls=":")
    ax.set_xlabel("Fraction of customers contacted (riskiest first)")
    ax.set_ylabel("Fraction of churners captured")
    ax.set_title(f"Cumulative gain: top 20% -> {res['top20pct_captures_churners']:.0%}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(cfg.FIG_DIR / "gain_chart.png", dpi=150)
    plt.close(fig)

    # ---- fairness (recall / precision / selection rate by group) ----
    metrics = {"recall": recall_score,
               "precision": partial(precision_score, zero_division=0),
               "selection_rate": selection_rate, "n": count}
    fairness = {}
    for col in ("gender", "SeniorCitizen"):
        mf = MetricFrame(metrics=metrics, y_true=y, y_pred=pred,
                         sensitive_features=sens[col].values)
        bg = mf.by_group
        gaps = (bg[["recall", "precision", "selection_rate"]].max()
                - bg[["recall", "precision", "selection_rate"]].min())
        print(f"\n== {col} ==\n{bg.round(3)}\nmax gap:\n{gaps.round(3)}")
        bg.round(4).to_csv(cfg.REPORTS_DIR / f"fairness_{col}.csv")
        fairness[col] = {"by_group": bg.round(4).to_dict(), "gaps": gaps.round(4).to_dict()}
    res["fairness"] = fairness

    (cfg.REPORTS_DIR / "test_metrics.json").write_text(json.dumps(res, indent=2, default=float))
    print("\n== TEST RESULTS ==")
    for k, v in res.items():
        if k != "fairness":
            print(f"{k}: {v}")


if __name__ == "__main__":
    main()
