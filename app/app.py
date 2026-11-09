"""Streamlit app: per-customer churn risk + SHAP explanation.

Run from the project root:  streamlit run app/app.py
"""
import json
import sys
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import shap
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.explain import positive_class  # noqa: E402
from src.preprocessing import build_features  # noqa: E402

st.set_page_config(page_title="Churn Risk Explorer", page_icon="📉", layout="wide")

ACTIONS = [  # (feature substring, recommended action) - first positive driver that matches wins
    ("Contract_Month_to_month", "Offer a discounted 1- or 2-year contract."),
    ("InternetService_Fiber_optic", "Check service quality; offer a fiber price lock."),
    ("PaymentMethod_Electronic_check", "Give an incentive to switch to autopay."),
    ("tenure", "New customer: proactive onboarding call and welcome offer."),
    ("num_services", "Offer a bundle with security / tech-support add-ons."),
    ("TechSupport", "Offer free tech-support trial."),
    ("OnlineSecurity", "Offer free online-security trial."),
    ("MonthlyCharges", "Review the plan; offer a loyalty discount or price lock."),
    ("charge_delta", "Bill is above the customer's average: offer a price lock."),
]


@st.cache_resource
def load_artifacts():
    raw = joblib.load(ROOT / "models" / "raw_pipeline.joblib")
    cal = joblib.load(ROOT / "models" / "calibrated_model.joblib")
    meta = json.loads((ROOT / "models" / "config.json").read_text())
    explainer = shap.TreeExplainer(raw.named_steps["model"])
    return raw, cal, meta, explainer


@st.cache_data
def load_samples():
    p = ROOT / "models" / "sample_customers.csv"
    return pd.read_csv(p) if p.exists() else None


try:
    raw_pipe, cal, meta, explainer = load_artifacts()
except FileNotFoundError:
    st.error("Model files not found. Run `python -m src.train` first.")
    st.stop()

thr = meta["threshold"]
A = meta["assumptions"]
samples = load_samples()

st.title("📉 Customer Churn Risk Explorer")
st.caption("Calibrated LightGBM · profit-optimised threshold · SHAP explanations. "
           "SHAP explains the model, not causal effects.")

row, key = None, "custom"
if samples is not None:
    choice = st.sidebar.selectbox("Load a sample customer", ["Custom"] + samples["customerID"].tolist())
    if choice != "Custom":
        row, key = samples[samples["customerID"] == choice].iloc[0], choice


def sel(label, col, options):
    d = row[col] if row is not None else options[0]
    return st.selectbox(label, options, index=options.index(d) if d in options else 0,
                        key=f"{col}_{key}")


YN, YNP, YNI = ["No", "Yes"], ["No", "Yes", "No phone service"], ["No", "Yes", "No internet service"]
c1, c2, c3 = st.columns(3)
with c1:
    senior_default = 1 if (row is not None and int(row["SeniorCitizen"]) == 1) else 0
    senior = 1 if st.selectbox("Senior citizen", ["No", "Yes"], index=senior_default,
                               key=f"senior_{key}") == "Yes" else 0
    partner = sel("Partner", "Partner", YN)
    dependents = sel("Dependents", "Dependents", YN)
    tenure = st.slider("Tenure (months)", 0, 72, int(row["tenure"]) if row is not None else 12,
                       key=f"tenure_{key}")
    phone = sel("Phone service", "PhoneService", YN)
    lines = sel("Multiple lines", "MultipleLines", YNP)
with c2:
    internet = sel("Internet service", "InternetService", ["DSL", "Fiber optic", "No"])
    addons = {c: sel(c, c, YNI) for c in ["OnlineSecurity", "OnlineBackup", "DeviceProtection",
                                          "TechSupport", "StreamingTV", "StreamingMovies"]}
with c3:
    contract = sel("Contract", "Contract", ["Month-to-month", "One year", "Two year"])
    paperless = sel("Paperless billing", "PaperlessBilling", YN)
    pay = sel("Payment method", "PaymentMethod",
              ["Electronic check", "Mailed check", "Bank transfer (automatic)",
               "Credit card (automatic)"])
    monthly = st.number_input("Monthly charges ($)", 0.0, 200.0,
                              float(row["MonthlyCharges"]) if row is not None else 70.0,
                              key=f"monthly_{key}")
    tc_default = (pd.to_numeric(row["TotalCharges"], errors="coerce") if row is not None
                  else tenure * monthly)
    total = st.number_input("Total charges ($)", 0.0, 10000.0,
                            float(0 if pd.isna(tc_default) else tc_default), key=f"total_{key}")

raw_row = pd.DataFrame([dict(
    SeniorCitizen=senior, Partner=partner, Dependents=dependents, tenure=tenure,
    PhoneService=phone, MultipleLines=lines, InternetService=internet, **addons,
    Contract=contract, PaperlessBilling=paperless, PaymentMethod=pay,
    MonthlyCharges=monthly, TotalCharges=total, gender="NA")])
X = build_features(raw_row)

proba = float(cal.predict_proba(X)[0, 1])
high_cut = max(0.5, 2 * thr)
tier = "High" if proba >= high_cut else "Medium" if proba >= thr else "Low"
value = monthly * A["expected_months"]
exp_profit = proba * A["offer_success_rate"] * value - A["retention_cost"]

st.divider()
m1, m2, m3, m4 = st.columns(4)
m1.metric("Churn probability", f"{proba:.1%}")
m2.metric("Risk tier", {"High": "🔴 High", "Medium": "🟠 Medium", "Low": "🟢 Low"}[tier])
m3.metric("Decision threshold", f"{thr:.2f}")
m4.metric("Expected profit if contacted", f"${exp_profit:,.0f}")

Xt = raw_pipe.named_steps["prep"].transform(X)
sv = positive_class(explainer(Xt))
contrib = pd.Series(sv.values[0], index=Xt.columns).sort_values(ascending=False)

action = "No retention action needed."
if proba >= thr:
    action = "Contact with a generic retention offer."
    for feat in contrib[contrib > 0].head(3).index:
        hit = next((a for k, a in ACTIONS if k in feat), None)
        if hit:
            action = hit
            break
st.subheader("Recommended action")
st.info(action)

left, right = st.columns([3, 2])
with left:
    st.subheader("Why? (SHAP waterfall, log-odds of raw model)")
    shap.plots.waterfall(sv[0], max_display=10, show=False)
    st.pyplot(plt.gcf(), clear_figure=True)
with right:
    st.subheader("Top risk drivers")
    st.dataframe(contrib.head(5).rename("SHAP").to_frame().round(3))
    st.subheader("Top protective factors")
    st.dataframe(contrib.tail(5).rename("SHAP").to_frame().round(3))
