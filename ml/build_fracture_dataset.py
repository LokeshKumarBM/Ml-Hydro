
from pathlib import Path
import csv
import re
from openpyxl import load_workbook

BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "data" / "raw" / "hydrogen_compatibility"
OUTPUT_DIR = BASE_DIR / "data" / "processed"

WORKBOOKS = [
    "1100cia85tenfrafat.xlsx",
    "1211nib10frafat.xlsx",
    "3230san11frafat.xlsx",
]


def text(value):
    if value is None:
        return ""
    return str(value).strip()


def number(value):
    """Parse a numeric value; preserve missing or qualified values."""
    raw = text(value)

    if not raw or raw.lower() in {"n/r", "nr", "n.a.", "na"}:
        return None

    # Do not silently remove footnote markers from values such as 140†.
    if "†" in raw or "‡" in raw:
        raw = raw.replace("†", "").replace("‡", "").strip()

    match = re.fullmatch(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", raw)
    if not match:
        return None

    return float(raw)


def parse_environment(value):
    """
    Examples:
      H2: 45 / RT
      air: 0.1 / RT
      N2: 6.9 / RT
    """
    raw = text(value)
    match = re.match(
        r"^\s*([^:]+):\s*([-+]?\d*\.?\d+)\s*/\s*(.*?)\s*$",
        raw,
        flags=re.IGNORECASE,
    )

    if not match:
        return {
            "environment": raw or None,
            "pressure_mpa": None,
            "temperature": None,
        }

    return {
        "environment": match.group(1).strip(),
        "pressure_mpa": float(match.group(2)),
        "temperature": match.group(3).strip() or None,
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    records = []

    for filename in WORKBOOKS:
        path = RAW_DIR / filename

        if not path.exists():
            print(f"WARNING: Missing workbook: {path}")
            continue

        workbook = load_workbook(
            path,
            read_only=True,
            data_only=True,
        )

        for sheet in workbook.worksheets:
            if "fracture" not in sheet.title.lower():
                continue

            # These workbooks use fixed columns in their fracture sheets:
            # A material, D test-condition label, F environment,
            # G orientation, H method, I rate, J yield strength,
            # K JIC, L KIC.
            for row_index, row in enumerate(
                sheet.iter_rows(min_row=7, values_only=True),
                start=7,
            ):
                # Convert the row's first 13 values to A:M positions.
                values = list(row[:13])
                values += [None] * (13 - len(values))

                label = text(values[3])   # D
                environment_raw = text(values[5])  # F

                # Keep the named reference and gas-test rows only.
                # Exclude individual trial rows and footnotes.
                if label.lower() != "reference" and not label.lower().startswith(
                    "gas environment"
                ):
                    continue

                # A test row needs an environment field to be interpretable.
                if not environment_raw:
                    continue

                parsed = parse_environment(environment_raw)

                # Identify material and temper from workbook metadata.
                material = text(sheet["A1"].value)
                form = text(sheet["A3"].value)
                temper = text(sheet["A4"].value)

                jic_raw = text(values[10])  # K
                kic_raw = text(values[11])  # L

                record = {
                    "workbook": filename,
                    "material": material or None,
                    "form": form or None,
                    "temper": temper or None,
                    "test_row": row_index,
                    "test_label": label,
                    "environment": parsed["environment"],
                    "pressure_mpa": parsed["pressure_mpa"],
                    "temperature": parsed["temperature"],
                    "orientation": text(values[6]) or None,
                    "test_method": text(values[7]) or None,
                    "test_rate": text(values[8]) or None,
                    "yield_strength_mpa": number(values[9]),
                    "jic_kj_m2": number(values[10]),
                    "kic_mpa_sqrt_m": number(values[11]),
                    "jic_original": jic_raw or None,
                    "kic_original": kic_raw or None,
                    "source_sheet": sheet.title,
                }

                records.append(record)

        workbook.close()

    output_path = OUTPUT_DIR / "fracture_dataset.csv"

    fields = [
        "workbook",
        "material",
        "form",
        "temper",
        "test_row",
        "test_label",
        "environment",
        "pressure_mpa",
        "temperature",
        "orientation",
        "test_method",
        "test_rate",
        "yield_strength_mpa",
        "jic_kj_m2",
        "kic_mpa_sqrt_m",
        "jic_original",
        "kic_original",
        "source_sheet",
    ]

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)

    print(f"Saved: {output_path}")
    print(f"Number of extracted test rows: {len(records)}")

    if records:
        for record in records:
            print(
                record["material"],
                "|", record["environment"],
                "| pressure:", record["pressure_mpa"],
                "| JIC:", record["jic_original"],
                "| KIC:", record["kic_original"],
            )


if __name__ == "__main__":
    main()
