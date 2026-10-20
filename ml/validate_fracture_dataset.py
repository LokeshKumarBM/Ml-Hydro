"""Sanity checks for the fracture-toughness dataset built by build_fracture_dataset.py."""

from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "data" / "processed" / "fracture_dataset.csv"

NUMERIC = ["pressure_mpa", "yield_strength_mpa", "jic_kj_m2", "kic_mpa_sqrt_m"]


def main():
    df = pd.read_csv(DATA_FILE, encoding="utf-8-sig")

    print("=" * 60)
    print("FRACTURE DATASET VALIDATION")
    print("=" * 60)

    print(f"\nRows: {len(df)}   Columns: {len(df.columns)}")

    print("\nMissing values in key columns:")
    print(df[["material", "environment"] + NUMERIC].isna().sum().to_string())

    print("\nRecords by material and environment:")
    print(df.groupby(["material", "environment"], dropna=False).size().to_string())

    print("\nValue ranges:")
    for column in NUMERIC:
        values = pd.to_numeric(df[column], errors="coerce")
        print(f"  {column:20s} min={values.min():.4g}  max={values.max():.4g}")
        if (values <= 0).any() and column != "pressure_mpa":
            print(f"    WARNING: non-positive values in {column}")

    print("\nDuplicate rows:", int(df.duplicated().sum()))

    # Hydrogen effect: KIC in H2 relative to the reference environment.
    print("\nKIC in hydrogen vs reference (same material):")
    for material, rows in df.groupby("material"):
        h2 = rows[rows["environment"].astype(str).str.startswith("H2")]["kic_mpa_sqrt_m"]
        ref = rows[~rows["environment"].astype(str).str.startswith("H2")]["kic_mpa_sqrt_m"]
        if len(h2.dropna()) and len(ref.dropna()):
            ratio = h2.mean() / ref.mean()
            print(f"  {material}: H2/reference = {ratio:.2f}")

    print("\nValidation complete.")


if __name__ == "__main__":
    main()
