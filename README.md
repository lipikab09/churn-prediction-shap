# Profit-Aware Churn Prediction with SHAP Customer Archetypes

Calibrated LightGBM churn model on the Telco Customer Churn dataset (~7,000 customers),
with a profit-optimised decision threshold, SHAP explanations at four levels
(global, local, interactions, churn archetypes), a fairness audit, a model card, and a Streamlit app.

**Live app:** _add your Streamlit Community Cloud link here_

## Why not accuracy?
~73% of customers stay, so "predict nobody churns" scores ~73% accuracy and catches zero churners.
This project uses PR-AUC, recall/precision, lift/gain, Brier score and **expected profit** instead.

## Setup
```bash
python -m venv venv && source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```
Download `WA_Fn-UseC_-Telco-Customer-Churn.csv` from Kaggle (blastchar/telco-customer-churn)
and put it in `data/`.

## Run order
```bash
python -m src.eda            # EDA figures + baseline accuracy
python -m src.train          # imbalance comparison, Optuna (50 trials), calibration, profit threshold
python -m src.evaluate       # one-time TEST evaluation + gain chart + fairness
python -m src.shap_analysis  # SHAP global/local/interaction + archetypes
streamlit run app/app.py     # app
```
Quick smoke test: `python -m src.train --trials 5 --skip-compare`.
Outputs: `models/` (pipelines, threshold), `reports/` (CSV/JSON), `reports/figures/` (PNG).

## Method
| Step | What happens |
|---|---|
| Split | 60/20/20 stratified train/val/test. Test touched once (`evaluate.py`). |
| Imbalance | Compare class weights, `scale_pos_weight`, SMOTE (inside CV folds via `imblearn.Pipeline`) and plain model + threshold tuning. |
| Tuning | Optuna, 5-fold CV, objective = PR-AUC. |
| Calibration | Isotonic `CalibratedClassifierCV`; Brier score and reliability curve before/after. |
| Threshold | Maximises expected profit on the **validation** set. |
| SHAP | `TreeExplainer` on the raw LightGBM model; KMeans on SHAP vectors of predicted churners (k by silhouette). |
| Fairness | `fairlearn.MetricFrame` by `gender` and `SeniorCitizen` (gender is not a model feature). |

## Business assumptions (edit in `src/config.py`)
Retention offer cost **$25**, offer success rate **25%**, value saved = `MonthlyCharges x 12` months.
These are assumptions, not facts; see `reports/sensitivity.csv` and `sensitivity_heatmap.png`.
With calibrated probabilities, contacting customer *i* pays off iff `p_i * s * V_i > C`.

## Results (fill in from `reports/test_metrics.json`)
- Test PR-AUC: _X_   ROC-AUC: _X_   Brier: _X_
- At threshold _t_: recall _X_, precision _X_
- Expected profit vs default 0.5 threshold: _+X%_
- Top 20% riskiest customers capture _Y%_ of churners
- Top SHAP drivers: _fill in from `reports/shap_global_importance.csv`_

## Limitations
- SHAP explains the model, not causal drivers; proving interventions work needs an A/B test or uplift modelling.
- Single static dataset, no time dimension; business numbers are assumptions.
- SMOTE interpolates one-hot columns, a known limitation.
- The archetype names are rule-based; check `reports/archetype_profile.csv` and rename by hand.

## Deploy
Commit `models/`, `app/`, `src/`, `requirements.txt`, push to GitHub, then create an app on
Streamlit Community Cloud pointing at `app/app.py`.
