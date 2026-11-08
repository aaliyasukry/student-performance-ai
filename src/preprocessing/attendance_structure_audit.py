from pathlib import Path
from openpyxl import load_workbook
from openpyxl.utils import get_column_letter
import pandas as pd
import re


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_QUALITY_DIR = PROJECT_ROOT / "reports" / "data_quality"
DATA_QUALITY_DIR.mkdir(parents=True, exist_ok=True)
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
REPORTS_DIR = PROJECT_ROOT / "reports"

REPORTS_DIR.mkdir(parents=True, exist_ok=True)

ATTENDANCE_AUDIT_OUTPUT = DATA_QUALITY_DIR / "attendance_structure_audit.csv"
TERM_SUMMARY_OUTPUT = DATA_QUALITY_DIR / "school_days_term_summary.csv"
ISSUES_OUTPUT = REPORTS_DIR / "attendance_audit_issues.csv"

# Old diagnostic reports are no longer part of the normal pipeline.
LEGACY_DIAGNOSTIC_OUTPUTS = [
    REPORTS_DIR / "school_days_audit.csv",
    REPORTS_DIR / "attendance_identifier_audit.csv",
]


ATTENDANCE_FILES = {
    "attendance_2025.xlsx": 2025,
    "attendance_2026.xlsx": 2026,
}


RESEARCH_TERMS_BY_YEAR = {
    2025: {1, 2, 3},
    2026: {1, 2},
}


RESEARCH_MONTHS_BY_YEAR = {
    2025: [
        "Jan", "Feb", "Mar", "Apr",
        "May", "Jun", "Jul",
        "Aug", "Sep", "Oct", "Nov", "Dec"
    ],
    2026: [
        "Jan", "Feb", "Mar", "Apr",
        "May", "Jun", "Jul"
    ],
}


KNOWN_UNAVAILABLE_ATTENDANCE = {
    (2026, "9A"):
        "Historical attendance records were unavailable for Grade 9A in 2026."
}


MONTH_ALIASES = {
    "jan": "Jan", "january": "Jan",
    "feb": "Feb", "february": "Feb",
    "mar": "Mar", "march": "Mar",
    "apr": "Apr", "april": "Apr",
    "may": "May",
    "jun": "Jun", "june": "Jun",
    "jul": "Jul", "july": "Jul",
    "aug": "Aug", "august": "Aug",
    "sep": "Sep", "sept": "Sep", "september": "Sep",
    "oct": "Oct", "october": "Oct",
    "nov": "Nov", "november": "Nov",
    "dec": "Dec", "december": "Dec",
}


def clean_text(value):
    if value is None:
        return ""
    return str(value).strip()


def normalized(value):
    text = clean_text(value).lower()
    text = re.sub(r"[\n\r\t]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def clean_number(value):
    if value is None:
        return None
    try:
        return float(clean_text(value))
    except (ValueError, TypeError):
        return None


def clean_integer(value):
    number = clean_number(value)

    if number is None:
        return None

    if not float(number).is_integer():
        return None

    return int(number)


def is_target_class_sheet(sheet_name):
    return bool(
        re.fullmatch(
            r"(6|7|8|9|10)[A-Za-z]+",
            sheet_name.strip()
        )
    )


def parse_grade_class(sheet_name):
    match = re.fullmatch(
        r"(6|7|8|9|10)([A-Za-z]+)",
        sheet_name.strip()
    )

    if not match:
        return None, None

    return int(match.group(1)), match.group(2).upper()


def get_academic_stage(grade):
    if grade in {6, 7, 8, 9}:
        return "SECONDARY"

    if grade == 10:
        return "OL"

    return "UNKNOWN"


def detect_header_row(ws, search_rows=25):
    best_row = None
    best_score = -1

    for row_num in range(
        1,
        min(search_rows, ws.max_row) + 1
    ):
        score = 0

        for col_num in range(
            1,
            ws.max_column + 1
        ):
            value = normalized(
                ws.cell(
                    row=row_num,
                    column=col_num
                ).value
            )

            if "index" in value:
                score += 5

            if "admission" in value:
                score += 5

            if value in MONTH_ALIASES:
                score += 2

            if value == "total":
                score += 2

        if score > best_score:
            best_score = score
            best_row = row_num

    return best_row if best_score > 0 else None


def detect_month_columns(ws, header_row):
    month_columns = {}

    if header_row is None:
        return month_columns

    for col_num in range(
        1,
        ws.max_column + 1
    ):
        value = normalized(
            ws.cell(
                row=header_row,
                column=col_num
            ).value
        )

        if value in MONTH_ALIASES:
            month_columns[
                MONTH_ALIASES[value]
            ] = col_num

    return month_columns


def find_index_column(ws, header_row):
    if header_row is None:
        return None

    for col_num in range(
        1,
        ws.max_column + 1
    ):
        value = normalized(
            ws.cell(
                row=header_row,
                column=col_num
            ).value
        )

        if "index" in value:
            return col_num

    return None


def find_admission_column(ws, header_row):
    if header_row is None:
        return None

    for col_num in range(
        1,
        ws.max_column + 1
    ):
        value = normalized(
            ws.cell(
                row=header_row,
                column=col_num
            ).value
        )

        if "admission" in value:
            return col_num

    return None


def looks_like_name(value):
    text = clean_text(value)

    if not text:
        return False

    if not re.search(r"[A-Za-z]", text):
        return False

    forbidden = {
        "index",
        "index no",
        "admission",
        "admission no",
        "total",
        "teacher",
        "principal",
        "signature",
    }

    return normalized(text) not in forbidden


def classify_attendance_value(value):
    if value is None:
        return "EMPTY"

    text = clean_text(value)

    if not text:
        return "EMPTY"

    if text == "-":
        return "DASH"

    try:
        number = float(text)

        if number == 0:
            return "ZERO"

        if number > 0:
            return "NUMERIC_POSITIVE"

        return "NUMERIC_NEGATIVE"

    except (ValueError, TypeError):
        return "OTHER"


def row_has_attendance_data(
    ws,
    row_num,
    month_columns,
    research_months
):
    for month in research_months:
        col_num = month_columns.get(month)

        if col_num is None:
            continue

        value = ws.cell(
            row=row_num,
            column=col_num
        ).value

        if value is not None and clean_text(value):
            return True

    return False


def detect_student_rows(
    ws,
    header_row,
    index_col,
    admission_col,
    month_columns,
    research_months
):
    records = []

    for row_num in range(
        header_row + 1,
        ws.max_row + 1
    ):
        raw_index = (
            clean_text(
                ws.cell(
                    row=row_num,
                    column=index_col
                ).value
            )
            if index_col
            else ""
        )

        admission_no = (
            clean_text(
                ws.cell(
                    row=row_num,
                    column=admission_col
                ).value
            )
            if admission_col
            else ""
        )

        if not row_has_attendance_data(
            ws,
            row_num,
            month_columns,
            research_months
        ):
            continue

        name_identifier = ""

        if raw_index and looks_like_name(raw_index):
            name_identifier = raw_index
            raw_index = ""

        if not raw_index and not admission_no and not name_identifier:
            continue

        records.append(
            {
                "excel_row": row_num,
                "index_no": raw_index,
                "admission_no": admission_no,
                "name_identifier": name_identifier,
            }
        )

    return records


def detect_school_days_header(ws):
    for row_num in range(
        1,
        min(ws.max_row, 20) + 1
    ):
        values = [
            normalized(
                ws.cell(
                    row=row_num,
                    column=col
                ).value
            )
            for col in range(
                1,
                ws.max_column + 1
            )
        ]

        if (
            "month" in values
            and
            any(
                "school day" in value
                for value in values
            )
        ):
            return row_num

    return None


def audit_school_days_sheet(
    ws,
    file_name,
    year
):
    records = []
    terms = []
    issues = []

    header_row = detect_school_days_header(ws)

    if header_row is None:
        issues.append(
            {
                "file_name": file_name,
                "sheet_name": ws.title,
                "excel_row": None,
                "issue_type": "SCHOOL_DAYS_HEADER_NOT_FOUND",
                "details": "",
            }
        )
        return records, terms, issues

    month_col = None
    days_col = None
    term_col = None

    for col in range(
        1,
        ws.max_column + 1
    ):
        value = normalized(
            ws.cell(
                row=header_row,
                column=col
            ).value
        )

        if value == "month":
            month_col = col

        elif "school day" in value:
            days_col = col

        elif value in {"term", "terms"}:
            term_col = col

    research_months = (
        RESEARCH_MONTHS_BY_YEAR[year]
    )

    research_terms = (
        RESEARCH_TERMS_BY_YEAR[year]
    )

    for row_num in range(
        header_row + 1,
        ws.max_row + 1
    ):
        raw_month = normalized(
            ws.cell(
                row=row_num,
                column=month_col
            ).value
        )

        if raw_month not in MONTH_ALIASES:
            continue

        month = MONTH_ALIASES[raw_month]

        if month not in research_months:
            continue

        days = clean_number(
            ws.cell(
                row=row_num,
                column=days_col
            ).value
        )

        term = clean_integer(
            ws.cell(
                row=row_num,
                column=term_col
            ).value
        )

        records.append(
            {
                "file_name": file_name,
                "year": year,
                "month": month,
                "term": term,
                "school_days": days,
            }
        )

        if days is None or days < 0:
            issues.append(
                {
                    "file_name": file_name,
                    "sheet_name": ws.title,
                    "excel_row": row_num,
                    "issue_type": "INVALID_SCHOOL_DAYS",
                    "details": month,
                }
            )

        if term not in research_terms:
            issues.append(
                {
                    "file_name": file_name,
                    "sheet_name": ws.title,
                    "excel_row": row_num,
                    "issue_type": "INVALID_TERM",
                    "details": month,
                }
            )

    df = pd.DataFrame(records)

    if not df.empty:
        for term, group in df.groupby("term"):
            terms.append(
                {
                    "file_name": file_name,
                    "year": year,
                    "term": int(term),
                    "months": ", ".join(
                        group["month"].tolist()
                    ),
                    "school_days": group[
                        "school_days"
                    ].sum(),
                }
            )

    return records, terms, issues


attendance_records = []
identifier_records = []
school_days_records = []
term_summary_records = []
issue_records = []


print("=" * 110)
print(
    "STUDENT PERFORMANCE AI - "
    "ATTENDANCE STRUCTURE AUDIT"
)
print("=" * 110)


for file_name, year in ATTENDANCE_FILES.items():

    filepath = RAW_DATA_DIR / file_name

    workbook = load_workbook(
        filepath,
        data_only=True
    )

    research_months = (
        RESEARCH_MONTHS_BY_YEAR[year]
    )

    print(
        f"\n{file_name} | Year {year}"
    )

    for ws in workbook.worksheets:

        if not is_target_class_sheet(
            ws.title
        ):
            continue

        grade, class_name = (
            parse_grade_class(ws.title)
        )

        known_unavailable = (
            (year, ws.title)
            in KNOWN_UNAVAILABLE_ATTENDANCE
        )

        header_row = detect_header_row(ws)
        month_columns = detect_month_columns(
            ws,
            header_row
        )

        index_col = find_index_column(
            ws,
            header_row
        )

        admission_col = (
            find_admission_column(
                ws,
                header_row
            )
        )

        students = detect_student_rows(
            ws,
            header_row,
            index_col,
            admission_col,
            month_columns,
            research_months
        )

        counts = {
            "EMPTY": 0,
            "DASH": 0,
            "ZERO": 0,
            "NUMERIC_POSITIVE": 0,
            "NUMERIC_NEGATIVE": 0,
            "OTHER": 0,
        }

        for student in students:
            row_num = student["excel_row"]

            for month in research_months:
                col_num = month_columns.get(
                    month
                )

                if col_num is None:
                    continue

                category = (
                    classify_attendance_value(
                        ws.cell(
                            row=row_num,
                            column=col_num
                        ).value
                    )
                )

                counts[category] += 1

        admission_only = sum(
            1
            for student in students
            if (
                not student["index_no"]
                and student["admission_no"]
                and not student[
                    "name_identifier"
                ]
            )
        )

        names = sum(
            1
            for student in students
            if student["name_identifier"]
        )

        missing_admission = sum(
            1
            for student in students
            if not student["admission_no"]
        )

        attendance_records.append(
            {
                "file_name": file_name,
                "year": year,
                "sheet_name": ws.title,
                "grade": grade,
                "class": class_name,
                "academic_stage":
                    get_academic_stage(grade),
                "availability_status":
                    (
                        "KNOWN_UNAVAILABLE"
                        if known_unavailable
                        else "AVAILABLE"
                    ),
                "header_row": header_row,
                "index_column":
                    (
                        get_column_letter(index_col)
                        if index_col
                        else None
                    ),
                "admission_column":
                    (
                        get_column_letter(
                            admission_col
                        )
                        if admission_col
                        else None
                    ),
                "students_detected":
                    len(students),
                "admission_only_students":
                    admission_only,
                "name_identifiers":
                    names,
                "missing_admission_no":
                    missing_admission,
                "positive_numeric_values":
                    counts[
                        "NUMERIC_POSITIVE"
                    ],
                "zero_values":
                    counts["ZERO"],
                "dash_values":
                    counts["DASH"],
                "empty_values":
                    counts["EMPTY"],
            }
        )

        for student in students:
            identifier_records.append(
                {
                    "file_name": file_name,
                    "year": year,
                    "sheet_name": ws.title,
                    "excel_row":
                        student["excel_row"],
                    "index_no":
                        student["index_no"],
                    "admission_no":
                        student["admission_no"],
                    "name_identifier":
                        student[
                            "name_identifier"
                        ],
                }
            )

        if (
            admission_col is None
            and not known_unavailable
        ):
            issue_records.append(
                {
                    "file_name": file_name,
                    "sheet_name": ws.title,
                    "excel_row": header_row,
                    "issue_type":
                        "ADMISSION_COLUMN_NOT_FOUND",
                    "details": "",
                }
            )

        print(
            f"{ws.title:<4} | "
            f"Students: {len(students):<3} | "
            f"Admission-only: "
            f"{admission_only:<2} | "
            f"Missing Admission: "
            f"{missing_admission:<3} | "
            f"Names: {names:<2} | "
            f"Status: "
            f"{'UNAVAILABLE' if known_unavailable else 'OK'}"
        )

    if "School Days Total" in workbook.sheetnames:

        school_records, term_records, issues = (
            audit_school_days_sheet(
                workbook[
                    "School Days Total"
                ],
                file_name,
                year
            )
        )

        school_days_records.extend(
            school_records
        )

        term_summary_records.extend(
            term_records
        )

        issue_records.extend(
            issues
        )

    workbook.close()


attendance_df = pd.DataFrame(
    attendance_records
)

identifier_df = pd.DataFrame(
    identifier_records
)

school_days_df = pd.DataFrame(
    school_days_records
)

term_summary_df = pd.DataFrame(
    term_summary_records
)

issues_df = pd.DataFrame(
    issue_records
)


if issues_df.empty:
    issues_df = pd.DataFrame(
        columns=[
            "file_name",
            "sheet_name",
            "excel_row",
            "issue_type",
            "details",
        ]
    )


attendance_df.to_csv(
    ATTENDANCE_AUDIT_OUTPUT,
    index=False
)

term_summary_df.to_csv(
    TERM_SUMMARY_OUTPUT,
    index=False
)

# Only retain the issue report when the audit finds a real problem.
if not issues_df.empty:
    issues_df.to_csv(
        ISSUES_OUTPUT,
        index=False
    )
elif ISSUES_OUTPUT.exists():
    ISSUES_OUTPUT.unlink()

# Remove stale diagnostic files created by older versions.
for legacy_output in LEGACY_DIAGNOSTIC_OUTPUTS:
    if legacy_output.exists():
        legacy_output.unlink()


print("\n" + "=" * 110)
print("ATTENDANCE STRUCTURE AUDIT COMPLETE")
print("=" * 110)

print(
    f"\nClass sheets audited: "
    f"{len(attendance_df)}"
)

print(
    f"Student rows detected: "
    f"{attendance_df['students_detected'].sum()}"
)

print(
    f"Rows missing Admission No.: "
    f"{attendance_df['missing_admission_no'].sum()}"
)

print(
    f"Audit issues detected: "
    f"{len(issues_df)}"
)
