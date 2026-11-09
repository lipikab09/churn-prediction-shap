# Model Card: Telco Churn Predictor

## Intended use
Rank existing telecom customers by churn risk to prioritise retention offers. Not intended for
decisions that affect pricing fairness, credit, or eligibility for essential services.

## Data
Telco Customer Churn (Kaggle, blastchar): ~7,043 customers, 21 columns, ~26.5% churn.
Split 60/20/20 stratified (train/validation/test, seed 42). `customerID` dropped; `gender` excluded
from features and used only for auditing.

## Model
LightGBM tuned with Optuna (objective: CV PR-AUC), isotonic-calibrated. Threshold chosen to
maximise expected campaign profit on the validation set under stated assumptions
(`src/config.py`: cost, success rate, months of value).

## Metrics (test set, fill from `reports/test_metrics.json`)
| Metric | Value |
|---|---|
| PR-AUC | |
| ROC-AUC | |
| Recall / Precision / F1 at threshold | |
| Brier score | |
| Expected profit (optimal vs 0.5) | |
| Top-20% churner capture | |

## Fairness (fill from `reports/fairness_*.csv`)
Recall, precision and selection rate by `gender` and `SeniorCitizen`. Report gaps honestly and
note any group where the model under-detects churners or over-targets customers.

## Limitations
- One dataset, no time dimension, so no drift or temporal-validation evidence.
- Some features (e.g. TotalCharges, tenure) may partially encode churn timing.
- Profit results depend on assumed cost, success rate and customer value.
- SHAP is descriptive of the model, not causal.

## Ethical considerations
Retention offers should not systematically exclude protected groups. Monitor subgroup recall and
selection rate after deployment and re-audit when data changes.
