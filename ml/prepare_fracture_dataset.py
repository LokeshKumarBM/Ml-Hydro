
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "processed"

input_path = DATA_DIR / "fracture_dataset.csv"
output_path = DATA_DIR / "fracture_ml_ready.csv"

df = pd.read_csv(input_path)

# These are the measurements required for this initial analysis.
required = [
    "material",
    "environment",
    "pressure_mpa",
    "yield_strength_mpa",
    "jic_kj_m2",
    "kic_mpa_sqrt_m",
]

# Preserve complete records only in the new analysis file.
# The original dataset is not modified.
clean = df.dropna(subset=required).copy()

# Keep the original temperature and test metadata.
clean.to_csv(output_path, index=False)

print("Original records:", len(df))
print("Complete records:", len(clean))
print("Excluded due to missing required values:", len(df) - len(clean))
print("\nRecords by material and environment:")

if clean.empty:
    print("No complete records available.")
else:
    print(
        clean[
            ["material", "environment", "pressure_mpa",
             "yield_strength_mpa", "jic_kj_m2", "kic_mpa_sqrt_m"]
        ].to_string(index=False)
    )

print("\nSaved:", output_path)
