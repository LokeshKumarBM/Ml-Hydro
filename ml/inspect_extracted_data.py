
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "processed"

files = [
    "hydrogen_workbook_cells.csv",
    "hydrogen_fracture_fatigue_cells.csv",
]

for filename in files:
    path = DATA_DIR / filename

    print("\n" + "=" * 70)
    print(f"FILE: {filename}")
    print("=" * 70)

    if not path.exists():
        print(f"File not found: {path}")
        continue

    df = pd.read_csv(path)

    print(f"Total CSV records: {len(df)}")
    print("\nRecords per workbook:")
    print(df.groupby("workbook").size().to_string())

    print("\nRecords per sheet:")
    print(
        df.groupby(["workbook", "sheet"])
        .size()
        .to_string()
    )

    # Display cells in their original row/column order.
    for (workbook, sheet), group in df.groupby(
        ["workbook", "sheet"],
        sort=False
    ):
        if "fracture" not in sheet.lower():
            continue

        print(f"\n--- {workbook} | {sheet} ---")

        preview = group[
            ["row_number", "column_letter", "cell_value"]
        ]

        print(preview.to_string(index=False))

        # Limit the console output for each sheet.
        if len(group) > 100:
            print("Preview limited to the first 100 cells.")
            break
