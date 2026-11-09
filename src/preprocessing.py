"""Cleaning, feature engineering, splits and the sklearn preprocessor."""
import re

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import config as cfg

ADDON_COLS = [
    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies",
]

NUMERIC = [
    "tenure", "MonthlyCharges", "TotalCharges", "num_services",
    "avg_monthly_charge", "charge_delta", "SeniorCitizen", "is_autopay",
]
CATEGORICAL = [
    "Partner", "Dependents", "PhoneService", "MultipleLines", "InternetService",
    *ADDON_COLS, "Contract", "PaperlessBilling", "PaymentMethod", "tenure_group",
]
FEATURES = NUMERIC + CATEGORICAL
# `gender` is deliberately NOT a feature. It is kept only for fairness auditing.


def _sanitize(value):
    """'Bank transfer (automatic)' -> 'Bank_transfer_automatic' (safe for LightGBM names)."""
    if isinstance(value, str):
        return re.sub(r"[^0-9A-Za-z]+", "_", value).strip("_")
    return value


def build_features(raw: pd.DataFrame) -> pd.DataFrame:
    """Raw Telco-format rows -> model features. Used in training AND in the app."""
    df = raw.drop(columns=[c for c in ("customerID", "Churn") if c in raw.columns]).copy()

    # TotalCharges is a string with blanks for tenure == 0 customers -> 0
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0.0)

    # 'No internet service' / 'No phone service' are redundant -> 'No'
    for c in ADDON_COLS:
        df[c] = df[c].replace("No internet service", "No")
    df["MultipleLines"] = df["MultipleLines"].replace("No phone service", "No")

    for c in df.columns:
        if not pd.api.types.is_numeric_dtype(df[c]):
            df[c] = df[c].map(_sanitize)

    # ---- engineered features ----
    df["tenure_group"] = pd.cut(
        df["tenure"], bins=[-1, 12, 24, 48, np.inf],
        labels=["0_12", "12_24", "24_48", "48_plus"],
    ).astype(str)
    df["num_services"] = sum((df[c] == "Yes").astype(int) for c in ADDON_COLS)
    df["avg_monthly_charge"] = df["TotalCharges"] / (df["tenure"] + 1)
    df["charge_delta"] = df["MonthlyCharges"] - df["avg_monthly_charge"]
    df["is_autopay"] = df["PaymentMethod"].str.contains("automatic", case=False).astype(int)
    return df[FEATURES]


def load_dataset(path=cfg.DATA_PATH):
    raw = pd.read_csv(path)
    y = (raw["Churn"] == "Yes").astype(int)
    X = build_features(raw)
    sens = pd.DataFrame({
        "gender": raw["gender"],
        "SeniorCitizen": raw["SeniorCitizen"].map({0: "No", 1: "Yes"}),
    })
    return raw, X, y, sens


def get_splits(path=cfg.DATA_PATH):
    """60 / 20 / 20 train / validation / test, stratified. Test is touched once."""
    raw, X, y, sens = load_dataset(path)
    X_tf, X_test, y_tf, y_test, _, s_test = train_test_split(
        X, y, sens, test_size=0.2, stratify=y, random_state=cfg.RANDOM_STATE)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tf, y_tf, test_size=0.25, stratify=y_tf, random_state=cfg.RANDOM_STATE)
    return dict(
        raw=raw,
        X_train_full=X_tf, y_train_full=y_tf,
        X_train=X_train, y_train=y_train,
        X_val=X_val, y_val=y_val,
        X_test=X_test, y_test=y_test, s_test=s_test,
    )


def build_preprocessor(scale: bool = False) -> ColumnTransformer:
    """One-hot categoricals; optionally scale numerics (needed for logistic regression)."""
    num = StandardScaler() if scale else "passthrough"
    pre = ColumnTransformer(
        [
            ("num", num, NUMERIC),
            ("cat", OneHotEncoder(handle_unknown="ignore", drop="if_binary",
                                  sparse_output=False), CATEGORICAL),
        ],
        verbose_feature_names_out=False,
    )
    pre.set_output(transform="pandas")
    return pre
