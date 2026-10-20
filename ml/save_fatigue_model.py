"""Train the Ridge fatigue model on all data and save it with its metadata.

Run validate_fatigue_models.py first so the metadata can include the
grouped cross-validation scores.
"""

import json

import joblib
import pandas as pd
import sklearn

from fatigue_common import (
    CATEGORICAL_FEATURES,
    FEATURES,
    METADATA_PATH,
    MODEL_DIR,
    MODEL_PATH,
    NUMERIC_FEATURES,
    build_pipeline,
    load_fatigue_data,
    tested_conditions,
)

CV_RESULTS = MODEL_DIR.parent / "data" / "processed" / "group_cv_results.csv"


def main():
    df = load_fatigue_data()

    model = build_pipeline()
    model.fit(df[FEATURES], df["log_rate"])

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    validation = None
    if CV_RESULTS.exists():
        cv = pd.read_csv(CV_RESULTS)
        ridge = cv[cv["model"].eq("Ridge regression")]
        if not ridge.empty:
            validation = {
                "method": "Leave-one-group-out (material|environment)",
                "folds": int(len(ridge)),
                "mean_mae_log10": round(float(ridge["MAE"].mean()), 4),
                "mean_r2": round(float(ridge["R2"].mean()), 4),
            }

    metadata = {
        "model": "Ridge regression (alpha=1.0)",
        "target": "log10(da_dn_m_per_cycle)",
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "training_records": int(len(df)),
        "sklearn_version": sklearn.__version__,
        "materials": sorted(df["material"].astype(str).unique().tolist()),
        "environments": sorted(df["environment"].astype(str).unique().tolist()),
        "orientations": sorted(df["orientation"].dropna().astype(str).unique().tolist()),
        "test_methods": sorted(df["test_method"].dropna().astype(str).unique().tolist()),
        "tested_conditions": tested_conditions(df),
        "validation": validation,
        "purpose": "Exploratory fatigue crack-growth research only",
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("Model training complete.")
    print(f"Training records: {len(df)}")
    print(f"Saved model:    {MODEL_PATH}")
    print(f"Saved metadata: {METADATA_PATH}")
    if validation is None:
        print("Note: run validate_fatigue_models.py first to record CV scores.")
    print("This model is not a safety certification.")


if __name__ == "__main__":
    main()
