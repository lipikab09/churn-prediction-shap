"""Step 8: SHAP at four levels - global, local, interactions, churn archetypes.

Run:  python -m src.shap_analysis
SHAP runs on the RAW LightGBM model (not the calibrated wrapper). Values are in
log-odds of the raw model and describe the MODEL, not causal effects.
"""
import json

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

from . import config as cfg
from .explain import positive_class
from .preprocessing import get_splits


def save(name):
    plt.gcf().savefig(cfg.FIG_DIR / name, dpi=150, bbox_inches="tight")
    plt.close("all")


def name_and_action(p):
    """Rule-based naming. Inspect archetype_profile.csv and rename by hand if needed."""
    if p["share_month_to_month"] > 0.7 and p["share_fiber"] > 0.5 and p["tenure"] < 18:
        return ("New month-to-month fiber customers paying a lot",
                "Discounted 1-2 year contract + service-quality check")
    if p["tenure"] >= 30 and p["charge_delta"] > 0:
        return ("Long-tenure customers with rising bills",
                "Loyalty price lock / bill review")
    if p["num_services"] <= 1:
        return ("Low-engagement customers with few add-ons",
                "Free add-on trial / bundle offer")
    if p["share_echeck"] > 0.5:
        return ("Electronic-check payers at risk", "Autopay incentive")
    return ("Mixed-risk segment", "Targeted retention call; review top drivers manually")


def main():
    cfg.ensure_dirs()
    s = get_splits()
    X_test, y_test = s["X_test"], s["y_test"].values
    raw = joblib.load(cfg.MODELS_DIR / "raw_pipeline.joblib")
    cal = joblib.load(cfg.MODELS_DIR / "calibrated_model.joblib")
    thr = json.loads((cfg.MODELS_DIR / "config.json").read_text())["threshold"]

    prep, model = raw.named_steps["prep"], raw.named_steps["model"]
    Xt = prep.transform(X_test)
    explainer = shap.TreeExplainer(model)
    sv = positive_class(explainer(Xt))
    proba = cal.predict_proba(X_test)[:, 1]

    # 1. Global
    shap.plots.beeswarm(sv, max_display=15, show=False)
    save("shap_beeswarm.png")
    shap.plots.bar(sv, max_display=15, show=False)
    save("shap_bar.png")
    imp = pd.Series(np.abs(sv.values).mean(0), index=Xt.columns).sort_values(ascending=False)
    imp.to_csv(cfg.REPORTS_DIR / "shap_global_importance.csv", header=["mean_abs_shap"])
    print("Top drivers:\n", imp.head(10).round(3))

    # 2. Local: high-risk, low-risk, borderline
    for label, i in (("high", int(np.argmax(proba))), ("low", int(np.argmin(proba))),
                     ("borderline", int(np.argmin(np.abs(proba - thr))))):
        shap.plots.waterfall(sv[i], max_display=12, show=False)
        save(f"shap_waterfall_{label}.png")

    # 3. Dependence + interactions
    if {"tenure", "Contract_Month_to_month"} <= set(Xt.columns):
        shap.plots.scatter(sv[:, "tenure"], color=sv[:, "Contract_Month_to_month"], show=False)
        save("shap_dependence_tenure_contract.png")
    if {"MonthlyCharges", "InternetService_Fiber_optic"} <= set(Xt.columns):
        shap.plots.scatter(sv[:, "MonthlyCharges"],
                           color=sv[:, "InternetService_Fiber_optic"], show=False)
        save("shap_dependence_monthlycharges.png")
    try:
        iv = explainer.shap_interaction_values(Xt)
        if isinstance(iv, list):
            iv = iv[1]
        if iv.ndim == 4:
            iv = iv[..., 1]
        m = np.abs(iv).mean(0)
        np.fill_diagonal(m, 0)
        iu = np.triu_indices_from(m, k=1)
        pairs = pd.DataFrame({"feature_a": Xt.columns[iu[0]], "feature_b": Xt.columns[iu[1]],
                              "mean_abs_interaction": m[iu]})
        pairs = pairs.sort_values("mean_abs_interaction", ascending=False).head(15)
        pairs.to_csv(cfg.REPORTS_DIR / "shap_top_interactions.csv", index=False)
        print("\nTop interactions:\n", pairs.head(8).round(4).to_string(index=False))
    except Exception as e:  # interaction values are an optional extra
        print("Skipping interaction values:", e)

    # 4. Churn archetypes: cluster SHAP vectors of predicted churners
    mask = proba >= thr
    if mask.sum() < 30:
        print("Too few predicted churners to cluster.")
        return
    S = sv.values[mask]
    best_k, best_sil = 2, -1
    for k in range(2, 7):
        sil = silhouette_score(S, KMeans(k, n_init=10, random_state=cfg.RANDOM_STATE).fit_predict(S))
        print(f"  k={k} silhouette={sil:.3f}")
        if sil > best_sil:
            best_k, best_sil = k, sil
    labels = KMeans(best_k, n_init=10, random_state=cfg.RANDOM_STATE).fit_predict(S)
    print(f"Chosen k={best_k}")

    feats = Xt[mask].reset_index(drop=True)
    shap_df = pd.DataFrame(S, columns=Xt.columns)
    rows = []
    for c in range(best_k):
        idx = labels == c
        prof = dict(
            cluster=c, n=int(idx.sum()),
            actual_churn_rate=float(y_test[mask][idx].mean()),
            mean_proba=float(proba[mask][idx].mean()),
            tenure=float(feats.loc[idx, "tenure"].mean()),
            MonthlyCharges=float(feats.loc[idx, "MonthlyCharges"].mean()),
            num_services=float(feats.loc[idx, "num_services"].mean()),
            charge_delta=float(feats.loc[idx, "charge_delta"].mean()),
            share_month_to_month=float(feats.loc[idx, "Contract_Month_to_month"].mean()),
            share_fiber=float(feats.loc[idx, "InternetService_Fiber_optic"].mean()),
            share_echeck=float(feats.loc[idx, "PaymentMethod_Electronic_check"].mean()),
            top_drivers=", ".join(shap_df[idx].mean().sort_values(ascending=False).head(4).index),
        )
        prof["archetype"], prof["suggested_intervention"] = name_and_action(prof)
        rows.append(prof)
    out = pd.DataFrame(rows)
    out.round(3).to_csv(cfg.REPORTS_DIR / "archetype_profile.csv", index=False)
    print(out[["cluster", "n", "actual_churn_rate", "archetype",
               "suggested_intervention", "top_drivers"]].to_string(index=False))

    xy = PCA(2, random_state=cfg.RANDOM_STATE).fit_transform(S)
    fig, ax = plt.subplots(figsize=(7, 5))
    for c in range(best_k):
        ax.scatter(*xy[labels == c].T, s=12, alpha=0.6, label=out.loc[c, "archetype"])
    ax.set_title("Churn archetypes (PCA of SHAP vectors)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(cfg.FIG_DIR / "archetypes_pca.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
