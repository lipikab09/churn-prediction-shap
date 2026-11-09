"""SHAP helpers shared by the analysis script and the Streamlit app."""


def positive_class(sv):
    """Different SHAP/LightGBM versions return (n, f) or (n, f, 2). Normalise to (n, f)."""
    return sv[..., 1] if sv.values.ndim == 3 else sv
