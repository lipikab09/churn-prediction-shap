"""Steps 4-7: compare imbalance strategies, tune with Optuna, calibrate,
pick the profit-optimal threshold, save artifacts.

Run from the project root:  python -m src.train --trials 50
"""
import argparse
import json

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import seaborn as sns
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             precision_recall_curve, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score
from sklearn.pipeline import Pipeline

from . import config as cfg
from .business import best_threshold, expected_profit, profit_curve, sensitivity_grid
from .preprocessing import build_preprocessor, get_splits


def make_lgbm(**params):
    base = dict(n_estimators=300, learning_rate=0.05, num_leaves=15,
                min_child_samples=20, random_state=cfg.RANDOM_STATE,
                verbose=-1, n_jobs=-1)
    base.update(params)
    return LGBMClassifier(**base)


def make_pipe(model, scale=False):
    return Pipeline([("prep", build_preprocessor(scale)), ("model", model)])


# ---------------------------------------------------------------- Step 4
def compare_strategies(X, y):
    skf = StratifiedKFold(5, shuffle=True, random_state=cfg.RANDOM_STATE)
    spw = (y == 0).sum() / (y == 1).sum()
    strategies = {
        "1_logreg_class_weight": make_pipe(
            LogisticRegression(class_weight="balanced", max_iter=2000), scale=True),
        "2_lgbm_scale_pos_weight": make_pipe(make_lgbm(scale_pos_weight=spw)),
        # SMOTE lives INSIDE the imblearn pipeline -> applied to training folds only
        "3_lgbm_smote_in_cv": ImbPipeline([
            ("prep", build_preprocessor()),
            ("smote", SMOTE(random_state=cfg.RANDOM_STATE)),
            ("model", make_lgbm())]),
        "4_lgbm_plain_threshold_tuning": make_pipe(make_lgbm()),
    }
    rows = []
    for name, pipe in strategies.items():
        print(f"  CV: {name}")
        oof = cross_val_predict(pipe, X, y, cv=skf, method="predict_proba")[:, 1]
        prec, rec, thr = precision_recall_curve(y, oof)
        f1 = 2 * prec * rec / (prec + rec + 1e-12)
        k = int(np.argmax(f1[:-1]))
        pred05 = oof >= 0.5
        rows.append(dict(
            strategy=name,
            pr_auc=average_precision_score(y, oof),
            roc_auc=roc_auc_score(y, oof),
            brier=brier_score_loss(y, oof),
            recall_at_0_5=recall_score(y, pred05),
            precision_at_0_5=precision_score(y, pred05, zero_division=0),
            best_f1_threshold=thr[k], recall_at_best_f1=rec[k],
            precision_at_best_f1=prec[k], best_f1=f1[k]))
    df = pd.DataFrame(rows).round(4)
    df.to_csv(cfg.REPORTS_DIR / "imbalance_comparison.csv", index=False)
    print(df.to_string(index=False))
    print(f"\n  'Always predict no' accuracy: {(y == 0).mean():.3f} (recall = 0)")
    return df


# ---------------------------------------------------------------- Step 5
def tune(X, y, n_trials):
    skf = StratifiedKFold(5, shuffle=True, random_state=cfg.RANDOM_STATE)
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        params = dict(
            n_estimators=trial.suggest_int("n_estimators", 150, 700),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            num_leaves=trial.suggest_int("num_leaves", 6, 48),
            max_depth=trial.suggest_int("max_depth", 3, 8),
            min_child_samples=trial.suggest_int("min_child_samples", 10, 60),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 10, log=True),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10, log=True),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
            subsample_freq=1,
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
        )
        pipe = make_pipe(make_lgbm(**params))
        return cross_val_score(pipe, X, y, cv=skf, scoring="average_precision").mean()

    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=cfg.RANDOM_STATE))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    print(f"  best CV PR-AUC = {study.best_value:.4f}")
    return {**study.best_params, "subsample_freq": 1}, float(study.best_value)


# ---------------------------------------------------------------- Step 6
def reliability_plot(y, p_raw, p_cal, path):
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    for p, label in ((p_raw, "raw LightGBM"), (p_cal, "isotonic calibrated")):
        frac_pos, mean_pred = calibration_curve(y, p, n_bins=10, strategy="quantile")
        ax.plot(mean_pred, frac_pos, marker="o", label=label)
    ax.plot([0, 1], [0, 1], "k--", label="perfect")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed churn rate")
    ax.set_title("Reliability curve (validation set)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=50)
    ap.add_argument("--skip-compare", action="store_true")
    ap.add_argument("--skip-tune", action="store_true")
    args = ap.parse_args()

    cfg.ensure_dirs()
    s = get_splits()
    Xtr, ytr, Xval, yval = s["X_train"], s["y_train"], s["X_val"], s["y_val"]
    print(f"train={len(Xtr)}  val={len(Xval)}  test={len(s['X_test'])}  "
          f"churn rate={s['y_train_full'].mean():.3f}")

    if not args.skip_compare:
        print("\n[Step 4] Imbalance strategies (5-fold CV on train+val, SMOTE inside folds)")
        compare_strategies(s["X_train_full"], s["y_train_full"])

    best_params, cv_pr_auc = {}, None
    if not args.skip_tune:
        print(f"\n[Step 5] Optuna ({args.trials} trials, objective = CV PR-AUC)")
        best_params, cv_pr_auc = tune(Xtr, ytr, args.trials)

    print("\n[Step 6] Fit + calibrate")
    raw_pipe = make_pipe(make_lgbm(**best_params)).fit(Xtr, ytr)      # for SHAP
    cal = CalibratedClassifierCV(make_pipe(make_lgbm(**best_params)),
                                 method="isotonic", cv=5).fit(Xtr, ytr)
    p_raw = raw_pipe.predict_proba(Xval)[:, 1]
    p_cal = cal.predict_proba(Xval)[:, 1]
    brier_raw, brier_cal = brier_score_loss(yval, p_raw), brier_score_loss(yval, p_cal)
    print(f"  Brier raw={brier_raw:.4f}  calibrated={brier_cal:.4f}  "
          f"val PR-AUC={average_precision_score(yval, p_cal):.4f}")
    reliability_plot(yval, p_raw, p_cal, cfg.FIG_DIR / "reliability_curve.png")

    print("\n[Step 7] Profit-optimal threshold (validation set)")
    value = Xval["MonthlyCharges"].values * cfg.EXPECTED_MONTHS
    C, S = cfg.RETENTION_COST, cfg.OFFER_SUCCESS_RATE
    thr, profit_best = best_threshold(yval, p_cal, value, C, S, cfg.THRESHOLD_GRID)
    profit_05 = expected_profit(yval, p_cal, 0.5, value, C, S)
    profit_all = expected_profit(yval, p_cal, 0.0, value, C, S)
    uplift = (profit_best - profit_05) / abs(profit_05) * 100 if profit_05 else float("nan")
    print(f"  best threshold={thr:.2f}  profit={profit_best:,.0f}  "
          f"@0.5={profit_05:,.0f}  contact-all={profit_all:,.0f}  uplift vs 0.5={uplift:.1f}%")

    grid, profits = profit_curve(yval, p_cal, value, C, S, cfg.THRESHOLD_GRID)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(grid, profits)
    ax.axvline(thr, color="green", ls="--", label=f"optimal = {thr:.2f}")
    ax.axvline(0.5, color="red", ls=":", label="default = 0.50")
    ax.set_xlabel("Decision threshold")
    ax.set_ylabel("Expected campaign profit ($)")
    ax.set_title("Profit vs threshold (validation set)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(cfg.FIG_DIR / "profit_vs_threshold.png", dpi=150)
    plt.close(fig)

    sens = sensitivity_grid(yval, p_cal, value, [10, 15, 25, 40, 60],
                            [0.10, 0.20, 0.30, 0.40, 0.50], cfg.THRESHOLD_GRID)
    sens.to_csv(cfg.REPORTS_DIR / "sensitivity.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    sns.heatmap(sens.pivot(index="cost", columns="success", values="best_profit"),
                annot=True, fmt=".0f", cmap="YlGn", ax=axes[0])
    axes[0].set_title("Max expected profit ($)")
    sns.heatmap(sens.pivot(index="cost", columns="success", values="best_threshold"),
                annot=True, fmt=".2f", cmap="Blues", ax=axes[1])
    axes[1].set_title("Optimal threshold")
    fig.tight_layout()
    fig.savefig(cfg.FIG_DIR / "sensitivity_heatmap.png", dpi=150)
    plt.close(fig)

    # ---- save artifacts ----
    joblib.dump(raw_pipe, cfg.MODELS_DIR / "raw_pipeline.joblib")
    joblib.dump(cal, cfg.MODELS_DIR / "calibrated_model.joblib")
    meta = dict(
        threshold=thr,
        assumptions=dict(retention_cost=C, offer_success_rate=S,
                         expected_months=cfg.EXPECTED_MONTHS),
        best_params=best_params, cv_pr_auc=cv_pr_auc,
        val=dict(brier_raw=brier_raw, brier_calibrated=brier_cal,
                 profit_best=profit_best, profit_at_0_5=profit_05,
                 profit_contact_all=profit_all, uplift_vs_0_5_pct=uplift),
    )
    (cfg.MODELS_DIR / "config.json").write_text(json.dumps(meta, indent=2))
    s["raw"].loc[s["X_test"].index[:200]].to_csv(
        cfg.MODELS_DIR / "sample_customers.csv", index=False)
    print(f"\nSaved models + reports to {cfg.MODELS_DIR} and {cfg.REPORTS_DIR}")


if __name__ == "__main__":
    main()
