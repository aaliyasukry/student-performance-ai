from pathlib import Path
from openpyxl import load_workbook
import pandas as pd
import re

# paths

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_QUALITY_DIR = PROJECT_ROOT / "reports" / "data_quality"
DATA_QUALITY_DIR.mkdir(parents=True, exist_ok=True)
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
REPORTS_DIR = PROJECT_ROOT / "reports"

REPORTS_DIR.mkdir(exist_ok=True)

OUTPUT_FILE = DATA_QUALITY_DIR / "data_inventory.csv"

# Helper functions
def classify_sheet(sheet_name):
    """Classify sheets relevant to Grades 6-10."""

    name = sheet_name.strip()

    # examples: 6A, 6B. 10A, 10B
    if re.match(r"(6|7|8|9|10)[A-Za-z]+", name):
        return "class_sheet"

    if "summary" in name.lower():
        return "summary_sheet"

    if "calendar" in name.lower():
        return "calendar_sheet"

    return "other"

def inspect_workbook(filepath):
    """Inspect workbook metadata only."""

    workbook = load_workbook(
        filepath,
        read_only=False,
        data_only=False
    )

    results = []

    for worksheet in workbook.worksheets:

        results.append({
            "file_name": filepath.name,
            "sheet_name": worksheet.title,
            "sheet_type": classify_sheet(worksheet.title),
            "rows": worksheet.max_row,
            "columns": worksheet.max_column,
            "sheet_state": worksheet.sheet_state
        })

        workbook.close()

    return results

# main inventory
excel_files = sorted(
    list(RAW_DATA_DIR.glob("*.xlsx"))
    + list(RAW_DATA_DIR.glob("*.xlsm"))
)

if not excel_files:
    raise FileNotFoundError(
        f"No Excel files found in: {RAW_DATA_DIR}"
    )

all_results = []

print("=" * 70)
print("STUDENT PERFORMANCE AI - RAW DATA INVENTORY")
print("=" * 70)

for filepath in excel_files:

    print(f"\nInspecting: {filepath.name}")

    try:
        workbook_results = inspect_workbook(filepath)

        all_results.extend(workbook_results)

        print(f"  Sheets found: {len(workbook_results)}")

        target_sheets = [
            row["sheet_name"]
            for row in workbook_results
            if row["sheet_type"] == "class_sheet"
        ]

        print(
            "  Grade 6-10 class sheets:",
            ", ".join(target_sheets) if target_sheets else "None"
        )

    except Exception as exc:
        print(f"  ERROR: {exc}")


# save inventory

inventory_df = pd.DataFrame(all_results)

inventory_df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\n" + "=" * 70)
print("INVENTORY COMPLETE")
print("=" * 70)

print(f"\nExcel files found: {len(excel_files)}")
print(f"Total sheets found: {len(inventory_df)}")

if not inventory_df.empty:

    target_count = (
        inventory_df["sheet_type"]
        .eq("class_sheet")
        .sum()
    )

    print(f"Grade 6-10 class sheets found: {target_count}")

print(f"\nInventory saved to:")
print(OUTPUT_FILE)
