"""Step 1: quick EDA figures + the 'always predict no' baseline.

Run:  python -m src.eda
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from . import config as cfg
from .preprocessing import load_dataset


def main():
    cfg.ensure_dirs()
    raw, X, y, _ = load_dataset()
    df = X.assign(Churn=y.values)
    print(f"Rows: {len(df)}  churn rate: {y.mean():.3f}")
    print(f"'Always predict no' accuracy: {(y == 0).mean():.3f}  (recall = 0.0)")

    cols = ["Contract", "tenure_group", "PaymentMethod", "InternetService"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, col in zip(axes.ravel(), cols):
        rate = df.groupby(col)["Churn"].mean().sort_values()
        sns.barplot(x=rate.values, y=rate.index, ax=ax, color="steelblue")
        ax.set_title(f"Churn rate by {col}")
        ax.set_xlabel("churn rate")
    fig.tight_layout()
    fig.savefig(cfg.FIG_DIR / "eda_churn_by_category.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    sns.kdeplot(data=df, x="MonthlyCharges", hue="Churn", common_norm=False, fill=True, ax=ax)
    ax.set_title("MonthlyCharges distribution by churn")
    fig.tight_layout()
    fig.savefig(cfg.FIG_DIR / "eda_monthlycharges.png", dpi=150)
    plt.close(fig)
    print(f"Figures saved to {cfg.FIG_DIR}")


if __name__ == "__main__":
    main()
