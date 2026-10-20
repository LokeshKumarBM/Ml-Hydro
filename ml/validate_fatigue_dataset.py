
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "data" / "processed" / "fatigue_dataset.csv"

df = pd.read_csv(DATA_FILE)

print("=" * 60)
print("FATIGUE DATASET VALIDATION")
print("=" * 60)

print("\n1. Dataset dimensions")
print(f"Rows: {len(df)}")
print(f"Columns: {len(df.columns)}")

print("\n2. Missing values")
print(df.isna().sum().to_string())

print("\n3. Measurements by material")
print(df.groupby("material").size().to_string())

print("\n4. Measurements by environment")
print(
    df.groupby(["material", "environment"], dropna=False)
    .size()
    .to_string()
)

print("\n5. Measurement ranges")
for column in [
    "pressure_mpa",
    "r_ratio",
    "frequency_hz",
    "delta_k_mpa_sqrt_m",
    "da_dn_m_per_cycle",
]:
    if column in df.columns:
        print(f"\n{column}")
        print(df[column].describe().to_string())

print("\n6. Invalid or non-positive crack-growth values")
rate = pd.to_numeric(df["da_dn_m_per_cycle"], errors="coerce")
print("Missing rates:", rate.isna().sum())
print("Rates <= 0:", (rate <= 0).sum())

print("\n7. Duplicate measurement rows")
print("Duplicates:", df.duplicated().sum())

print("\n8. Worksheet and data type counts")
print(df.groupby(["material", "sheet", "data_type"]).size().to_string())

print("\nValidation complete.")
