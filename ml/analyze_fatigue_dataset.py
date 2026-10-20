
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "data" / "processed" / "fatigue_dataset.csv"
OUTPUT_DIR = BASE_DIR / "data" / "processed" / "plots"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA_FILE)

# Analyze the standard fatigue sheets first.
# Keep Fatigue2 separate because its test format is different.
standard = df[
    df["sheet"].eq("Mechanical Properties-Fatigue")
].copy()

# Assign simple display labels based on the recorded source environment.
def environment_label(value):
    value = str(value).upper()

    if "H2" in value:
        return "Hydrogen"
    if "N2" in value:
        return "Nitrogen reference"
    if "AIR" in value:
        return "Air reference"

    return "Other / unclassified"


standard["environment_group"] = standard["environment"].apply(
    environment_label
)

print("=" * 60)
print("FATIGUE DATA EXPLORATORY ANALYSIS")
print("=" * 60)

print("\nMeasurements by material and environment:")
print(
    standard.groupby(
        ["material", "environment_group"]
    ).size().to_string()
)

print("\nMedian crack-growth rate by material and environment:")
print(
    standard.groupby(
        ["material", "environment_group"]
    )["da_dn_m_per_cycle"]
    .median()
    .to_string()
)

print("\nGenerating plots...")

for material in sorted(standard["material"].unique()):
    subset = standard[standard["material"] == material]

    plt.figure(figsize=(9, 6))

    for environment, group in subset.groupby("environment_group"):
        plt.scatter(
            group["delta_k_mpa_sqrt_m"],
            group["da_dn_m_per_cycle"],
            label=environment,
            alpha=0.75,
            s=25,
        )

    plt.yscale("log")
    plt.xlabel("Stress-intensity-factor range ΔK (MPa√m)")
    plt.ylabel("Crack-growth rate da/dN (m/cycle)")
    plt.title(f"{material}: Fatigue crack-growth measurements")
    plt.legend()
    plt.grid(True, which="both", alpha=0.25)
    plt.tight_layout()

    output_file = OUTPUT_DIR / f"{material}_fatigue.png"
    plt.savefig(output_file, dpi=160)
    plt.close()

    print(f"Saved plot: {output_file}")

print("\nExploratory analysis complete.")
print("Note: These plots are descriptive, not safety certifications.")
