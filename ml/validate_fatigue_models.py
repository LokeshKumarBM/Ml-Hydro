"""Leave-one-group-out validation of baseline, Ridge and Random Forest models.

Each fold holds out one complete material/environment combination, so the
score shows how the model does on test conditions it has never seen. This is
much stricter (and more honest) than a random row split, where near-identical
neighbouring points from the same test curve end up in train and test.

Outputs:
  data/processed/group_cv_results.csv      per-fold scores for every model
  data/processed/plots/predicted_vs_actual.png
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import LeaveOneGroupOut

from fatigue_common import FEATURES, ML_DIR, build_pipeline, load_fatigue_data

OUTPUT_DIR = ML_DIR / "data" / "processed"
RESULTS_FILE = OUTPUT_DIR / "group_cv_results.csv"
PLOT_FILE = OUTPUT_DIR / "plots" / "predicted_vs_actual.png"


def make_models():
    return {
        "Baseline (training mean)": DummyRegressor(strategy="mean"),
        "Ridge regression": Ridge(alpha=1.0),
        "Random Forest": RandomForestRegressor(
            n_estimators=300, min_samples_leaf=3, random_state=42, n_jobs=-1
        ),
    }


def main():
    df = load_fatigue_data()
    X, y, groups = df[FEATURES], df["log_rate"], df["group"]
    cv = LeaveOneGroupOut()

    print("=" * 64)
    print("LEAVE-ONE-GROUP-OUT VALIDATION (log10 da/dN)")
    print("=" * 64)
    print(f"Records: {len(df)}   Groups: {groups.nunique()}")
    print()

    rows = []
    ridge_predictions = pd.Series(np.nan, index=df.index)

    for name, regressor in make_models().items():
        for fold, (train_idx, test_idx) in enumerate(cv.split(X, y, groups), start=1):
            model = build_pipeline(regressor)
            model.fit(X.iloc[train_idx], y.iloc[train_idx])
            pred = model.predict(X.iloc[test_idx])
            actual = y.iloc[test_idx]

            if name == "Ridge regression":
                ridge_predictions.iloc[test_idx] = pred

            rows.append({
                "model": name,
                "fold": fold,
                "held_out_group": groups.iloc[test_idx].iloc[0],
                "test_records": len(test_idx),
                "MAE": mean_absolute_error(actual, pred),
                "RMSE": float(np.sqrt(mean_squared_error(actual, pred))),
                "R2": r2_score(actual, pred),
            })

    results = pd.DataFrame(rows)
    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(RESULTS_FILE, index=False)

    pd.set_option("display.width", 120)
    print(results.round(3).to_string(index=False))
    print("\nMean across folds:")
    print(results.groupby("model")[["MAE", "RMSE", "R2"]].mean().round(3))

    # Predicted vs actual for Ridge, coloured by held-out group.
    PLOT_FILE.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.5, 6))
    for group, part in df.groupby("group"):
        ax.scatter(part["log_rate"], ridge_predictions[part.index], s=12, alpha=0.7, label=group)
    lo = min(df["log_rate"].min(), ridge_predictions.min()) - 0.2
    hi = max(df["log_rate"].max(), ridge_predictions.max()) + 0.2
    ax.plot([lo, hi], [lo, hi], color="grey", linewidth=1, linestyle="--")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("Measured log10(da/dN) [m/cycle]")
    ax.set_ylabel("Predicted log10(da/dN) [m/cycle]")
    ax.set_title("Ridge: leave-one-group-out predictions")
    ax.legend(fontsize=8, title="Held-out group", title_fontsize=8)
    fig.tight_layout()
    fig.savefig(PLOT_FILE, dpi=150)

    print(f"\nSaved results: {RESULTS_FILE}")
    print(f"Saved plot:    {PLOT_FILE}")
    print("\nResearch evaluation only; not a safety certification.")


if __name__ == "__main__":
    main()
