"""Visualise a rule-based material screen as a decision tree.

What this does (and does not do):
  1. Labels each material in a general engineering-materials table as
     "suitable" if all six properties fall inside fixed ranges chosen by hand
     (strength, stiffness, Poisson's ratio, density close to a reference
     carbon steel).
  2. Fits a decision tree to those labels and draws it.

Because the labels come from the hand-written rule, the tree simply
re-learns that rule. It is a way to *visualise* the screening criteria,
not a predictive model, and it does not use any hydrogen data.

Outputs:
  data/processed/material_screening.csv
  data/processed/plots/material_screening_tree.png
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree

BASE_DIR = Path(__file__).resolve().parent
INPUT_FILE = BASE_DIR / "data" / "raw" / "material_properties" / "material_properties.csv"
OUTPUT_FILE = BASE_DIR / "data" / "processed" / "material_screening.csv"
PLOT_FILE = BASE_DIR / "data" / "processed" / "plots" / "material_screening_tree.png"

FEATURES = ["Su", "Sy", "E", "G", "mu", "Ro"]

# Hand-chosen acceptance ranges (units as in the source table:
# MPa for Su/Sy/E/G, kg/m^3 for Ro).
RULES = {
    "Su": (336, 505),          # ultimate tensile strength
    "Sy": (251, 376),          # yield strength
    "E": (165000, 248000),     # Young's modulus
    "G": (63000, 94000),       # shear modulus
    "mu": (0.24, 0.36),        # Poisson's ratio
    "Ro": (6200, 9400),        # density
}


def main():
    df = pd.read_csv(INPUT_FILE)

    df["Material"] = (
        df[["Std", "Material", "Heat treatment"]]
        .fillna("")
        .astype(str)
        .agg(" ".join, axis=1)
        .str.strip()
    )
    df["Sy"] = pd.to_numeric(
        df["Sy"].astype(str).str.replace(" max", "", regex=False), errors="coerce"
    )
    df = df.dropna(subset=FEATURES).copy()

    suitable = pd.Series(True, index=df.index)
    for column, (low, high) in RULES.items():
        suitable &= df[column].between(low, high)
    df["Use"] = suitable

    out = df[["Material"] + FEATURES + ["Use"]]
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT_FILE, index=False)

    tree = DecisionTreeClassifier(random_state=42)
    tree.fit(out[FEATURES], out["Use"])

    print(f"Materials screened: {len(out)}   Suitable: {int(out['Use'].sum())}")
    print("\nLearned tree (mirrors the hand-written rule):")
    print(export_text(tree, feature_names=FEATURES))

    PLOT_FILE.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(16, 9))
    plot_tree(
        tree,
        feature_names=FEATURES,
        class_names=["Not suitable", "Suitable"],
        filled=True,
        rounded=True,
        fontsize=8,
        ax=ax,
    )
    fig.tight_layout()
    fig.savefig(PLOT_FILE, dpi=130)

    print(f"Saved: {OUTPUT_FILE}")
    print(f"Saved: {PLOT_FILE}")


if __name__ == "__main__":
    main()
