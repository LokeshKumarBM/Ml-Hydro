
from pathlib import Path
import csv
import re
import warnings

from openpyxl import load_workbook


BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "data" / "raw" / "hydrogen_compatibility"
OUTPUT_DIR = BASE_DIR / "data" / "processed"

WORKBOOKS = [
    "1100cia85tenfrafat.xlsx",
    "1211nib10frafat.xlsx",
    "3230san11frafat.xlsx",
]

OUTPUT_FILE = OUTPUT_DIR / "fatigue_dataset.csv"


def clean(value):
    if value is None:
        return None
    if isinstance(value, str):
        value = value.strip()
        return value if value else None
    return value


def number(value):
    value = clean(value)

    if value is None:
        return None

    try:
        return float(value)
    except (ValueError, TypeError):
        return None


def parse_environment(value):
    """Preserve the original environment and extract pressure/temp if possible."""
    text = str(value or "").strip()

    pressure = None
    temperature = None

    match = re.search(r"([-+]?\d*\.?\d+)\s*/\s*(RT|[-+]?\d*\.?\d+)", text, re.I)

    if match:
        pressure = float(match.group(1))
        temp_text = match.group(2)

        if temp_text.upper() != "RT":
            temperature = float(temp_text)

    return pressure, temperature


def extract_standard_fatigue(workbook, workbook_name, material):
    """Extract paired ΔK and da/dN columns from standard fatigue sheets."""
    records = []

    for sheet in workbook.worksheets:
        if sheet.title.lower() != "mechanical properties-fatigue":
            continue

        # Metadata is in rows 5–9; paired measurement headers are in row 12.
        header_row = 12

        for col in range(5, sheet.max_column):
            delta_header = clean(sheet.cell(header_row, col).value)
            rate_header = clean(sheet.cell(header_row, col + 1).value)

            if not delta_header or not rate_header:
                continue

            if "da/dn" not in str(rate_header).lower():
                continue

            if "k" not in str(delta_header).lower():
                continue

            # The value for each test condition is in the second column
            # of its measurement pair.
            env = clean(sheet.cell(5, col + 1).value)
            orientation = clean(sheet.cell(6, col + 1).value)
            test_method = clean(sheet.cell(7, col + 1).value)
            r_ratio = number(sheet.cell(8, col + 1).value)
            frequency = number(sheet.cell(9, col + 1).value)

            pressure, temperature = parse_environment(env)

            for row in range(14, sheet.max_row + 1):
                delta_k = number(sheet.cell(row, col).value)
                rate = number(sheet.cell(row, col + 1).value)

                if delta_k is None or rate is None:
                    continue

                records.append({
                    "material": material,
                    "workbook": workbook_name,
                    "sheet": sheet.title,
                    "data_type": "measured_crack_growth",
                    "environment": env,
                    "pressure_mpa": pressure,
                    "temperature_c": temperature,
                    "orientation": orientation,
                    "test_method": test_method,
                    "r_ratio": r_ratio,
                    "frequency_hz": frequency,
                    "delta_k_mpa_sqrt_m": delta_k,
                    "da_dn_m_per_cycle": rate,
                    "lower_bound_m_per_cycle": None,
                    "upper_bound_m_per_cycle": None,
                    "reference_flag": clean(
                        sheet.cell(10, col + 1).value
                    ),
                })

    return records


def extract_fatigue2(workbook, workbook_name, material):
    """Extract constant-ΔK fatigue tests with varying stress ratios."""
    records = []

    for sheet in workbook.worksheets:
        if sheet.title.lower() != "mechanical properties-fatigue2":
            continue

        current = None

        for row in range(7, sheet.max_row + 1):
            label = clean(sheet.cell(row, 4).value)

            # A new test block begins with Reference or gas environment.
            if label:
                env = clean(sheet.cell(row, 6).value)
                pressure, temperature = parse_environment(env)

                current = {
                    "environment": env,
                    "pressure_mpa": pressure,
                    "temperature_c": temperature,
                    "orientation": clean(sheet.cell(row, 7).value),
                    "test_method": clean(sheet.cell(row, 8).value),
                    "delta_k_mpa_sqrt_m": number(
                        sheet.cell(row, 9).value
                    ),
                    "reference_flag": clean(
                        sheet.cell(row, 14).value
                    ),
                }

            if current is None:
                continue

            r_ratio = number(sheet.cell(row, 10).value)
            rate = number(sheet.cell(row, 11).value)

            if r_ratio is None or rate is None:
                continue

            records.append({
                "material": material,
                "workbook": workbook_name,
                "sheet": sheet.title,
                "data_type": "constant_delta_k_crack_growth",
                **current,
                "r_ratio": r_ratio,
                "frequency_hz": None,
                "da_dn_m_per_cycle": rate,
                "lower_bound_m_per_cycle": number(
                    sheet.cell(row, 12).value
                ),
                "upper_bound_m_per_cycle": number(
                    sheet.cell(row, 13).value
                ),
            })

    return records


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_records = []

    for filename in WORKBOOKS:
        path = RAW_DIR / filename

        if not path.exists():
            print(f"WARNING: Missing workbook: {path}")
            continue

        # Material names come from the first cell of the first worksheet.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Unknown extension is not supported",
            )

            workbook = load_workbook(
                path,
                read_only=True,
                data_only=True,
            )

        material = clean(workbook.worksheets[0]["A1"].value)

        all_records.extend(
            extract_standard_fatigue(workbook, filename, material)
        )
        all_records.extend(
            extract_fatigue2(workbook, filename, material)
        )

        workbook.close()

        print(f"Processed: {filename} ({material})")

    if not all_records:
        print("\nNo fatigue records extracted.")
        print("Check the workbook filenames and data folder.")
        return

    columns = [
        "material",
        "workbook",
        "sheet",
        "data_type",
        "environment",
        "pressure_mpa",
        "temperature_c",
        "orientation",
        "test_method",
        "r_ratio",
        "frequency_hz",
        "delta_k_mpa_sqrt_m",
        "da_dn_m_per_cycle",
        "lower_bound_m_per_cycle",
        "upper_bound_m_per_cycle",
        "reference_flag",
    ]

    with OUTPUT_FILE.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(all_records)

    print(f"\nTotal extracted measurements: {len(all_records)}")
    print(f"Saved dataset: {OUTPUT_FILE}")

    print("\nMeasurements by material and sheet:")
    counts = {}

    for record in all_records:
        key = (record["material"], record["sheet"])
        counts[key] = counts.get(key, 0) + 1

    for (material, sheet), count in sorted(
    counts.items(),
    key=lambda item: (str(item[0][0]), str(item[0][1]))
):
        print(f"{material} | {sheet}: {count}")


if __name__ == "__main__":
    main()
