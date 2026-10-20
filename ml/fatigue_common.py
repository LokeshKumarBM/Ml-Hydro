"""Shared dataset loading and model pipeline for the fatigue crack-growth model.

Every training, comparison and validation script imports from here so the
features, filtering and preprocessing stay identical everywhere.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ML_DIR = Path(__file__).resolve().parent
DATA_FILE = ML_DIR / "data" / "processed" / "fatigue_dataset.csv"
MODEL_DIR = ML_DIR / "models"
MODEL_PATH = MODEL_DIR / "fatigue_ridge_model.joblib"
METADATA_PATH = MODEL_DIR / "fatigue_model_metadata.json"

TARGET = "da_dn_m_per_cycle"
NUMERIC_FEATURES = [
    "pressure_mpa",
    "delta_k_mpa_sqrt_m",
    "r_ratio",
    "frequency_hz",
]
CATEGORICAL_FEATURES = [
    "material",
    "environment",
    "orientation",
    "test_method",
]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def load_fatigue_data(path=DATA_FILE):
    """Load standard fatigue-test rows with a positive crack-growth rate.

    Adds:
      log_rate - log10(da/dN), the regression target
      group    - "material|environment", used for grouped validation
    """
    df = pd.read_csv(path)
    df = df[df["sheet"].eq("Mechanical Properties-Fatigue")].copy()
    df = df.dropna(subset=[TARGET, "material", "environment"])
    df = df[df[TARGET] > 0].copy()
    df["log_rate"] = np.log10(df[TARGET])
    df["group"] = df["material"].astype(str) + "|" + df["environment"].astype(str)
    return df.reset_index(drop=True)


def build_preprocessor():
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("numeric", numeric, NUMERIC_FEATURES),
        ("categorical", categorical, CATEGORICAL_FEATURES),
    ])


def build_pipeline(regressor=None):
    """Preprocessing + regressor. Defaults to Ridge(alpha=1.0)."""
    return Pipeline([
        ("preprocessor", build_preprocessor()),
        ("regressor", regressor if regressor is not None else Ridge(alpha=1.0)),
    ])


def tested_conditions(df):
    """Describe every material/environment combination present in the data.

    The API and dashboard use this so users can only request conditions the
    model was actually trained on.
    """
    conditions = []
    for (material, environment), rows in df.groupby(["material", "environment"]):
        conditions.append({
            "material": str(material),
            "environment": str(environment),
            "pressure_mpa": float(rows["pressure_mpa"].median()),
            "orientations": sorted(rows["orientation"].dropna().astype(str).unique().tolist()),
            "test_methods": sorted(rows["test_method"].dropna().astype(str).unique().tolist()),
            "r_ratios": sorted(float(v) for v in rows["r_ratio"].dropna().unique()),
            "frequencies_hz": sorted(float(v) for v in rows["frequency_hz"].dropna().unique()),
            "delta_k_range": [
                round(float(rows["delta_k_mpa_sqrt_m"].min()), 3),
                round(float(rows["delta_k_mpa_sqrt_m"].max()), 3),
            ],
            "records": int(len(rows)),
        })
    return conditions
