
from pathlib import Path
import pandas as pd

DATA_DIR = (
    Path(__file__).parent
    / "data"
    / "raw"
    / "hydrogen_compatibility"
)


def main():
    if not DATA_DIR.exists():
        print(f"Folder not found: {DATA_DIR}")
        print("Please check the folder structure.")
        return

    files = sorted(DATA_DIR.glob("*.xlsx"))

    if not files:
        print(f"No Excel files found in: {DATA_DIR}")
        print("Copy your downloaded .xlsx files into this folder.")
        return

    for file_path in files:
        print("\n" + "=" * 60)
        print(f"FILE: {file_path.name}")
        print("=" * 60)

        try:
            excel = pd.ExcelFile(file_path)
            print("Worksheets:", excel.sheet_names)

            for sheet in excel.sheet_names:
                df = pd.read_excel(
                    file_path,
                    sheet_name=sheet,
                    header=None,
                )

                print(f"\n--- Worksheet: {sheet} ---")
                print(f"Rows: {len(df)}, Columns: {len(df.columns)}")

                nonempty = df.dropna(how="all")
                print(nonempty.head(15).to_string(
                    index=False,
                    header=False,
                ))

        except Exception as error:
            print(f"Could not inspect workbook: {error}")


if __name__ == "__main__":
    main()
