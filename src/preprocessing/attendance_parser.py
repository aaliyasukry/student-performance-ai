from pathlib import Path
from openpyxl import load_workbook
from openpyxl.utils import get_column_letter
import pandas as pd
import re


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_QUALITY_DIR = PROJECT_ROOT / "reports" / "data_quality"
DATA_QUALITY_DIR.mkdir(parents=True, exist_ok=True)

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
CLEANED_DATA_DIR = PROJECT_ROOT / "data" / "cleaned"
REPORTS_DIR = PROJECT_ROOT / "reports"

CLEANED_DATA_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


MONTHLY_OUTPUT = CLEANED_DATA_DIR / "attendance_monthly.csv"
TERM_OUTPUT = CLEANED_DATA_DIR / "attendance_term.csv"
CALENDAR_OUTPUT = CLEANED_DATA_DIR / "attendance_calendar.csv"

PARSE_SUMMARY_OUTPUT = DATA_QUALITY_DIR / "attendance_parse_summary.csv"
ISSUES_OUTPUT = REPORTS_DIR / "attendance_parse_issues.csv"
ADJUSTMENTS_OUTPUT = REPORTS_DIR / "attendance_adjustments.csv"


ATTENDANCE_FILES = {
    "attendance_2025.xlsx": 2025,
    "attendance_2026.xlsx": 2026,
}


RESEARCH_MONTHS_BY_YEAR = {
    2025: [
        "Jan", "Feb", "Mar", "Apr",
        "May", "Jun", "Jul",
        "Aug", "Sep", "Oct", "Nov", "Dec",
    ],
    2026: [
        "Jan", "Feb", "Mar", "Apr",
        "May", "Jun", "Jul",
    ],
}


KNOWN_UNAVAILABLE_ATTENDANCE = {
    (2026, "9A"):
        "Historical attendance records were unavailable."
}


MAX_REASONABLE_OVERAGE = 5


MONTH_ALIASES = {
    "jan": "Jan",
    "january": "Jan",
    "feb": "Feb",
    "february": "Feb",
    "mar": "Mar",
    "march": "Mar",
    "apr": "Apr",
    "april": "Apr",
    "may": "May",
    "jun": "Jun",
    "june": "Jun",
    "jul": "Jul",
    "july": "Jul",
    "aug": "Aug",
    "august": "Aug",
    "sep": "Sep",
    "sept": "Sep",
    "september": "Sep",
    "oct": "Oct",
    "october": "Oct",
    "nov": "Nov",
    "november": "Nov",
    "dec": "Dec",
    "december": "Dec",
}


def clean_text(value):

    if value is None:
        return ""

    return str(value).strip()


def normalized(value):

    text = clean_text(value).lower()

    text = re.sub(
        r"[\n\r\t]+",
        " ",
        text
    )

    return re.sub(
        r"\s+",
        " ",
        text
    ).strip()


def clean_identifier(value):

    if value is None:
        return ""

    if isinstance(value, int):
        return str(value)

    if isinstance(value, float):

        if value.is_integer():
            return str(int(value))

    text = clean_text(value)

    if re.fullmatch(
        r"\d+\.0",
        text
    ):
        return text[:-2]

    return text


def canonical_index(value):

    text = clean_identifier(value)

    if not text:
        return ""

    parts = re.findall(
        r"\d+",
        text
    )

    if not parts:
        return ""

    result = parts[-1].lstrip("0")

    return result or "0"


def clean_number(value):

    if value is None:
        return None

    try:

        return float(
            clean_text(value)
        )

    except (
        ValueError,
        TypeError
    ):

        return None


def clean_integer(value):

    number = clean_number(value)

    if number is None:
        return None

    if not number.is_integer():
        return None

    return int(number)


def is_target_sheet(sheet_name):

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

    return (
        int(match.group(1)),
        match.group(2).upper()
    )


def get_academic_stage(grade):

    if grade in {6, 7, 8, 9}:
        return "SECONDARY"

    if grade == 10:
        return "OL"

    return "UNKNOWN"


def detect_header_row(ws):

    best_row = None
    best_score = -1

    for row_num in range(
        1,
        min(ws.max_row, 25) + 1
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

        if score > best_score:

            best_score = score
            best_row = row_num

    return (
        best_row
        if best_score > 0
        else None
    )


def detect_month_columns(
    ws,
    header_row
):

    result = {}

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

        if value in MONTH_ALIASES:

            result[
                MONTH_ALIASES[value]
            ] = col

    return result


def find_column_containing(
    ws,
    header_row,
    text
):

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

        if text in value:
            return col

    return None


def looks_like_name(value):

    text = clean_text(value)

    if not text:
        return False

    return bool(
        re.search(
            r"[A-Za-z]",
            text
        )
    )


def row_has_summary_label(
    ws,
    row_num,
    month_columns
):

    jan_col = month_columns.get(
        "Jan"
    )

    if jan_col is None:
        return False

    labels = {
        "total",
        "grand total",
        "class total",
        "summary",
        "average",
        "avg",
    }

    for col in range(
        1,
        jan_col
    ):

        value = normalized(
            ws.cell(
                row=row_num,
                column=col
            ).value
        )

        if value in labels:
            return True

        if value.startswith(
            "total "
        ):
            return True

    return False


def row_looks_like_aggregate_total(
    ws,
    row_num,
    month_columns,
    calendar,
    research_months
):

    extreme = 0

    for month in research_months:

        col = month_columns.get(
            month
        )

        info = calendar.get(
            month
        )

        if not col or not info:
            continue

        raw = ws.cell(
            row=row_num,
            column=col
        ).value

        text = clean_text(raw)

        if not text or text == "-":
            continue

        try:
            value = float(text)

        except (
            TypeError,
            ValueError
        ):
            continue

        school_days = info[
            "school_days"
        ]

        if (
            school_days > 0
            and
            value
            >
            school_days
            +
            MAX_REASONABLE_OVERAGE
        ):
            extreme += 1

    return extreme >= 2


def parse_attendance_value(
    raw_value,
    school_days
):

    if school_days == 0:

        return (
            None,
            None,
            None,
            "NOT_APPLICABLE",
            False,
            None,
            None,
        )

    text = clean_text(
        raw_value
    )

    if not text or text == "-":

        return (
            None,
            None,
            None,
            "UNKNOWN",
            False,
            None,
            None,
        )

    try:

        raw_present = float(
            text
        )

    except (
        ValueError,
        TypeError
    ):

        return (
            None,
            None,
            None,
            "INVALID",
            False,
            None,
            f"Unrecognized value: {raw_value}",
        )

    if raw_present < 0:

        return (
            None,
            None,
            None,
            "INVALID",
            False,
            None,
            f"Negative value: {raw_present}",
        )

    if raw_present <= school_days:

        absent = (
            school_days
            -
            raw_present
        )

        percentage = (
            raw_present
            /
            school_days
            *
            100
        )

        return (
            raw_present,
            absent,
            percentage,
            "RECORDED",
            False,
            None,
            None,
        )

    overage = (
        raw_present
        -
        school_days
    )

    if (
        overage
        <=
        MAX_REASONABLE_OVERAGE
    ):

        reason = (
            f"Raw attendance {raw_present} "
            f"exceeded school days {school_days}; "
            f"capped at 100%."
        )

        return (
            school_days,
            0.0,
            100.0,
            "CAPPED_TO_SCHOOL_DAYS",
            True,
            reason,
            None,
        )

    return (
        None,
        None,
        None,
        "INVALID",
        False,
        None,
        (
            f"Attendance {raw_present} "
            f"greatly exceeds {school_days}."
        ),
    )


def extract_calendar(
    workbook,
    file_name,
    year
):

    ws = workbook[
        "School Days Total"
    ]

    header = None

    for row in range(
        1,
        min(
            ws.max_row,
            20
        ) + 1
    ):

        values = [
            normalized(
                ws.cell(
                    row=row,
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

            header = row
            break

    if header is None:

        raise ValueError(
            f"School Days Total header "
            f"not found in {file_name}"
        )

    month_col = None
    days_col = None
    term_col = None

    for col in range(
        1,
        ws.max_column + 1
    ):

        value = normalized(
            ws.cell(
                row=header,
                column=col
            ).value
        )

        if value == "month":
            month_col = col

        elif "school day" in value:
            days_col = col

        elif value in {
            "term",
            "terms"
        }:
            term_col = col

    calendar = {}
    records = []

    for row in range(
        header + 1,
        ws.max_row + 1
    ):

        raw_month = normalized(
            ws.cell(
                row=row,
                column=month_col
            ).value
        )

        if (
            raw_month
            not in
            MONTH_ALIASES
        ):
            continue

        month = (
            MONTH_ALIASES[
                raw_month
            ]
        )

        if (
            month
            not in
            RESEARCH_MONTHS_BY_YEAR[
                year
            ]
        ):
            continue

        days = clean_number(
            ws.cell(
                row=row,
                column=days_col
            ).value
        )

        term = clean_integer(
            ws.cell(
                row=row,
                column=term_col
            ).value
        )

        calendar[
            month
        ] = {
            "school_days":
                days,
            "term":
                term,
        }

        records.append(
            {
                "source_file":
                    file_name,
                "year":
                    year,
                "month":
                    month,
                "term":
                    term,
                "school_days":
                    days,
            }
        )

    return (
        calendar,
        records
    )


monthly_records = []
calendar_records = []
summary_records = []
issues = []
adjustments = []


print(
    "=" * 110
)

print(
    "STUDENT PERFORMANCE AI - "
    "ATTENDANCE PARSER"
)

print(
    "=" * 110
)


for (
    file_name,
    year
) in ATTENDANCE_FILES.items():

    filepath = (
        RAW_DATA_DIR /
        file_name
    )

    workbook = load_workbook(
        filepath,
        data_only=True
    )

    (
        calendar,
        calendar_rows
    ) = extract_calendar(
        workbook,
        file_name,
        year
    )

    calendar_records.extend(
        calendar_rows
    )

    research_months = (
        RESEARCH_MONTHS_BY_YEAR[
            year
        ]
    )

    print(
        f"\n{file_name} | "
        f"Year {year}"
    )

    for ws in workbook.worksheets:

        if not is_target_sheet(
            ws.title
        ):
            continue

        (
            grade,
            class_name
        ) = parse_grade_class(
            ws.title
        )

        if (
            year,
            ws.title
        ) in KNOWN_UNAVAILABLE_ATTENDANCE:

            summary_records.append(
                {
                    "file_name":
                        file_name,
                    "year":
                        year,
                    "sheet_name":
                        ws.title,
                    "grade":
                        grade,
                    "class":
                        class_name,
                    "availability_status":
                        "KNOWN_UNAVAILABLE",
                    "students_parsed":
                        0,
                    "admission_only_students":
                        0,
                    "missing_admission_no":
                        0,
                    "monthly_records_created":
                        0,
                    "invalid_values":
                        0,
                }
            )

            print(
                f"{ws.title:<4} | "
                f"SKIPPED - "
                f"KNOWN_UNAVAILABLE"
            )

            continue

        header_row = (
            detect_header_row(
                ws
            )
        )

        month_columns = (
            detect_month_columns(
                ws,
                header_row
            )
        )

        index_col = (
            find_column_containing(
                ws,
                header_row,
                "index"
            )
        )

        admission_col = (
            find_column_containing(
                ws,
                header_row,
                "admission"
            )
        )

        if admission_col is None:

            issues.append(
                {
                    "file_name":
                        file_name,
                    "sheet_name":
                        ws.title,
                    "excel_row":
                        header_row,
                    "excel_column":
                        None,
                    "issue_type":
                        "ADMISSION_COLUMN_NOT_FOUND",
                    "details":
                        "",
                }
            )

            continue

        student_count = 0
        admission_only_count = 0
        missing_admission_count = 0
        monthly_count = 0

        recorded_count = 0
        capped_count = 0
        unknown_count = 0
        na_count = 0
        invalid_count = 0

        for row_num in range(
            header_row + 1,
            ws.max_row + 1
        ):

            if row_has_summary_label(
                ws,
                row_num,
                month_columns
            ):
                continue

            if row_looks_like_aggregate_total(
                ws,
                row_num,
                month_columns,
                calendar,
                research_months
            ):
                continue

            index_value = (
                clean_identifier(
                    ws.cell(
                        row=row_num,
                        column=index_col
                    ).value
                )
                if index_col
                else ""
            )

            admission_no = (
                clean_identifier(
                    ws.cell(
                        row=row_num,
                        column=admission_col
                    ).value
                )
            )

            name_identifier = ""

            if (
                index_value
                and
                looks_like_name(
                    index_value
                )
            ):

                name_identifier = (
                    index_value
                )

                index_value = ""

            has_attendance = False

            for month in research_months:

                col = month_columns.get(
                    month
                )

                if col is None:
                    continue

                value = ws.cell(
                    row=row_num,
                    column=col
                ).value

                if (
                    value is not None
                    and
                    clean_text(value)
                ):

                    has_attendance = True
                    break

            if not has_attendance:
                continue

            if (
                not index_value
                and
                not admission_no
                and
                not name_identifier
            ):
                continue

            student_count += 1

            if (
                not index_value
                and
                admission_no
            ):
                admission_only_count += 1

            if not admission_no:
                missing_admission_count += 1

            if name_identifier:

                identifier_type = (
                    "NAME"
                )

            elif index_value:

                identifier_type = (
                    "INDEX"
                )

            else:

                identifier_type = (
                    "ADMISSION_ONLY"
                )

            identifier_value = (
                name_identifier
                or
                index_value
                or
                admission_no
            )

            for month in research_months:

                col = month_columns.get(
                    month
                )

                if col is None:
                    continue

                info = calendar.get(
                    month
                )

                if info is None:
                    continue

                raw_value = ws.cell(
                    row=row_num,
                    column=col
                ).value

                (
                    days_present,
                    days_absent,
                    attendance_pct,
                    status,
                    adjusted,
                    adjustment_reason,
                    issue,
                ) = parse_attendance_value(
                    raw_value,
                    info[
                        "school_days"
                    ]
                )

                if status == "RECORDED":

                    recorded_count += 1

                elif (
                    status
                    ==
                    "CAPPED_TO_SCHOOL_DAYS"
                ):

                    capped_count += 1

                elif status == "UNKNOWN":

                    unknown_count += 1

                elif (
                    status
                    ==
                    "NOT_APPLICABLE"
                ):

                    na_count += 1

                elif status == "INVALID":

                    invalid_count += 1

                monthly_records.append(
                    {
                        "source_file":
                            file_name,
                        "source_sheet":
                            ws.title,

                        # This physically distinguishes students even where identifiers are duplicated.
                        "source_excel_row":
                            row_num,

                        "source_excel_column":
                            get_column_letter(
                                col
                            ),
                        "year":
                            year,
                        "term":
                            info["term"],
                        "month":
                            month,
                        "grade":
                            grade,
                        "class":
                            class_name,
                        "academic_stage":
                            get_academic_stage(
                                grade
                            ),
                        "identifier_type":
                            identifier_type,
                        "identifier_value":
                            identifier_value,
                        "index_no":
                            (
                                index_value
                                or None
                            ),
                        "canonical_index_no":
                            (
                                canonical_index(
                                    index_value
                                )
                                if index_value
                                else None
                            ),
                        "admission_no":
                            (
                                admission_no
                                or None
                            ),
                        "student_name_identifier":
                            (
                                name_identifier
                                or None
                            ),
                        "school_days":
                            info[
                                "school_days"
                            ],
                        "raw_attendance":
                            (
                                clean_text(
                                    raw_value
                                )
                                if raw_value
                                is not None
                                else None
                            ),
                        "days_present":
                            days_present,
                        "days_absent":
                            days_absent,
                        "monthly_attendance_pct":
                            attendance_pct,
                        "attendance_status":
                            status,
                        "adjustment_applied":
                            adjusted,
                        "adjustment_reason":
                            adjustment_reason,
                    }
                )

                monthly_count += 1

                if adjusted:

                    adjustments.append(
                        {
                            "file_name":
                                file_name,
                            "year":
                                year,
                            "sheet_name":
                                ws.title,
                            "excel_row":
                                row_num,
                            "identifier_value":
                                identifier_value,
                            "index_no":
                                index_value,
                            "admission_no":
                                admission_no,
                            "month":
                                month,
                            "term":
                                info[
                                    "term"
                                ],
                            "raw_attendance":
                                raw_value,
                            "official_school_days":
                                info[
                                    "school_days"
                                ],
                            "processed_days_present":
                                days_present,
                            "reason":
                                adjustment_reason,
                        }
                    )

                if issue:

                    issues.append(
                        {
                            "file_name":
                                file_name,
                            "sheet_name":
                                ws.title,
                            "excel_row":
                                row_num,
                            "excel_column":
                                get_column_letter(
                                    col
                                ),
                            "issue_type":
                                "INVALID_ATTENDANCE_VALUE",
                            "details":
                                issue,
                        }
                    )

        summary_records.append(
            {
                "file_name":
                    file_name,
                "year":
                    year,
                "sheet_name":
                    ws.title,
                "grade":
                    grade,
                "class":
                    class_name,
                "availability_status":
                    "AVAILABLE",
                "students_parsed":
                    student_count,
                "admission_only_students":
                    admission_only_count,
                "missing_admission_no":
                    missing_admission_count,
                "monthly_records_created":
                    monthly_count,
                "recorded_values":
                    recorded_count,
                "capped_values":
                    capped_count,
                "unknown_values":
                    unknown_count,
                "not_applicable_values":
                    na_count,
                "invalid_values":
                    invalid_count,
            }
        )

        print(
            f"{ws.title:<4} | "
            f"Students: "
            f"{student_count:<3} | "
            f"Admission-only: "
            f"{admission_only_count:<2} | "
            f"Missing Admission: "
            f"{missing_admission_count:<3} | "
            f"Invalid: "
            f"{invalid_count}"
        )

    workbook.close()


monthly_df = pd.DataFrame(
    monthly_records
)

calendar_df = pd.DataFrame(
    calendar_records
)

summary_df = pd.DataFrame(
    summary_records
)

issues_df = pd.DataFrame(
    issues
)

adjustments_df = pd.DataFrame(
    adjustments
)


term_records = []


if not monthly_df.empty:

    # source_excel_row is part of the grouping: distinct students can share an Index/Admission No.
    group_columns = [
        "source_file",
        "source_sheet",
        "source_excel_row",
        "year",
        "term",
        "grade",
        "class",
        "academic_stage",
        "identifier_type",
        "identifier_value",
        "index_no",
        "canonical_index_no",
        "admission_no",
        "student_name_identifier",
    ]

    for (
        values,
        group
    ) in monthly_df.groupby(
        group_columns,
        dropna=False,
        sort=False
    ):

        data = dict(
            zip(
                group_columns,
                values
            )
        )

        expected_days = (
            group[
                "school_days"
            ].sum()
        )

        valid = group[
            group[
                "attendance_status"
            ].isin(
                [
                    "RECORDED",
                    "CAPPED_TO_SCHOOL_DAYS",
                ]
            )
        ]

        observed_days = (
            valid[
                "school_days"
            ].sum()
        )

        days_present = (
            valid[
                "days_present"
            ].sum()
        )

        days_absent = (
            valid[
                "days_absent"
            ].sum()
        )

        attendance_pct = (
            days_present
            /
            observed_days
            *
            100
            if observed_days > 0
            else None
        )

        coverage_pct = (
            observed_days
            /
            expected_days
            *
            100
            if expected_days > 0
            else None
        )

        capped = int(
            (
                group[
                    "attendance_status"
                ]
                ==
                "CAPPED_TO_SCHOOL_DAYS"
            ).sum()
        )

        unknown = int(
            (
                group[
                    "attendance_status"
                ]
                ==
                "UNKNOWN"
            ).sum()
        )

        invalid = int(
            (
                group[
                    "attendance_status"
                ]
                ==
                "INVALID"
            ).sum()
        )

        if invalid > 0:

            quality = (
                "HAS_INVALID_VALUES"
            )

        elif (
            coverage_pct is None
            or
            coverage_pct == 0
        ):

            quality = (
                "NO_OBSERVED_ATTENDANCE"
            )

        elif coverage_pct < 100:

            quality = (
                "PARTIAL"
            )

        elif capped > 0:

            quality = (
                "COMPLETE_WITH_ADJUSTMENT"
            )

        else:

            quality = (
                "COMPLETE"
            )

        term_records.append(
            {
                **data,
                "expected_term_school_days":
                    expected_days,
                "observed_school_days":
                    observed_days,
                "days_present":
                    days_present,
                "days_absent":
                    days_absent,
                "attendance_pct":
                    attendance_pct,
                "attendance_coverage_pct":
                    coverage_pct,
                "months_observed":
                    len(valid),
                "months_capped":
                    capped,
                "months_unknown":
                    unknown,
                "months_invalid":
                    invalid,
                "incomplete_attendance":
                    (
                        coverage_pct is None
                        or
                        coverage_pct < 100
                    ),
                "attendance_data_quality":
                    quality,
            }
        )


term_df = pd.DataFrame(
    term_records
)


if issues_df.empty:

    issues_df = pd.DataFrame(
        columns=[
            "file_name",
            "sheet_name",
            "excel_row",
            "excel_column",
            "issue_type",
            "details",
        ]
    )


if adjustments_df.empty:

    adjustments_df = pd.DataFrame(
        columns=[
            "file_name",
            "year",
            "sheet_name",
            "excel_row",
            "identifier_value",
            "index_no",
            "admission_no",
            "month",
            "term",
            "raw_attendance",
            "official_school_days",
            "processed_days_present",
            "reason",
        ]
    )


monthly_df.to_csv(
    MONTHLY_OUTPUT,
    index=False
)

term_df.to_csv(
    TERM_OUTPUT,
    index=False
)

calendar_df.to_csv(
    CALENDAR_OUTPUT,
    index=False
)

summary_df.to_csv(
    PARSE_SUMMARY_OUTPUT,
    index=False
)

# Diagnostic output is kept only when genuine parsing issues exist.
if not issues_df.empty:
    issues_df.to_csv(
        ISSUES_OUTPUT,
        index=False
    )
elif ISSUES_OUTPUT.exists():
    ISSUES_OUTPUT.unlink()

# Adjustments are kept for auditability.
adjustments_df.to_csv(
    ADJUSTMENTS_OUTPUT,
    index=False
)


print(
    "\n"
    +
    "=" * 110
)

print(
    "ATTENDANCE PARSING COMPLETE"
)

print(
    "=" * 110
)

print(
    f"\nMonthly records: "
    f"{len(monthly_df)}"
)

print(
    f"Term records: "
    f"{len(term_df)}"
)

print(
    f"Adjustments: "
    f"{len(adjustments_df)}"
)

print(
    f"Parsing issues: "
    f"{len(issues_df)}"
)


if not term_df.empty:

    print(
        "\nAttendance data quality:"
    )

    print(
        term_df[
            "attendance_data_quality"
        ]
        .value_counts()
        .to_string()
    )
