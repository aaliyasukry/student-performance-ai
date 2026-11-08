from pathlib import Path
from openpyxl import load_workbook
import pandas as pd
import re


PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
REPORTS_DIR = PROJECT_ROOT / "reports"

REPORTS_DIR.mkdir(parents=True, exist_ok=True)


SPECIAL_VALUES_OUTPUT = (
    REPORTS_DIR / "attendance_special_values_audit.csv"
)

MONTH_SUMMARY_OUTPUT = (
    REPORTS_DIR / "attendance_value_by_month.csv"
)

STUDENT_PATTERN_OUTPUT = (
    REPORTS_DIR / "attendance_student_value_patterns.csv"
)


ATTENDANCE_FILES = {
    "attendance_2025.xlsx": 2025,
    "attendance_2026.xlsx": 2026,
}


RESEARCH_MONTHS = {
    2025: [
        "Jan", "Feb", "Mar", "Apr",
        "May", "Jun", "Jul", "Aug",
        "Sep", "Oct", "Nov", "Dec"
    ],

    2026: [
        "Jan", "Feb", "Mar", "Apr",
        "May", "Jun", "Jul"
    ],
}


# Known unavailable source
KNOWN_UNAVAILABLE = {
    (2026, "9A")
}


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

    return clean_text(value).lower()


def classify_value(value):
    """Raw attendance value classification."""

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
            return "POSITIVE"

        return "NEGATIVE"

    except ValueError:

        return "OTHER"


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

    return (
        int(match.group(1)),
        match.group(2).upper()
    )


def detect_header_row(ws):

    best_row = None
    best_score = -1

    for row_num in range(
        1,
        min(ws.max_row, 20) + 1
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

            if value in MONTH_ALIASES:
                score += 2

        if score > best_score:

            best_score = score
            best_row = row_num

    return best_row


def detect_month_columns(
    ws,
    header_row
):

    result = {}

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

            result[
                MONTH_ALIASES[value]
            ] = col_num

    return result


def find_identifier_column(
    ws,
    header_row,
    month_columns
):

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

    # Fallback: immediately before Jan
    jan_col = month_columns.get("Jan")

    if jan_col and jan_col > 1:
        return jan_col - 1

    return None


def get_calendar_map(
    workbook,
    research_months
):
    """Return the school days and term for each month."""

    calendar = {}

    ws = workbook[
        "School Days Total"
    ]

    header_row = None

    for row_num in range(
        1,
        min(ws.max_row, 20) + 1
    ):

        values = [
            normalized(
                ws.cell(
                    row=row_num,
                    column=col_num
                ).value
            )
            for col_num in range(
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

            header_row = row_num
            break

    month_col = None
    days_col = None
    term_col = None

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

        if value == "month":

            month_col = col_num

        elif "school day" in value:

            days_col = col_num

        elif value in {
            "term",
            "terms"
        }:

            term_col = col_num

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

        month = MONTH_ALIASES[
            raw_month
        ]

        if month not in research_months:
            continue

        raw_days = ws.cell(
            row=row_num,
            column=days_col
        ).value

        raw_term = ws.cell(
            row=row_num,
            column=term_col
        ).value

        try:

            school_days = float(
                raw_days
            )

        except (
            TypeError,
            ValueError
        ):

            school_days = None

        try:

            term = int(
                float(raw_term)
            )

        except (
            TypeError,
            ValueError
        ):

            term = None

        calendar[
            month
        ] = {
            "school_days":
                school_days,

            "term":
                term,
        }

    return calendar


def determine_position(
    month_index,
    positive_indices,
    school_days
):
    """Helps understand what ZERO / DASH / EMPTY means."""

    if school_days == 0:

        return "NO_SCHOOL_MONTH"

    if not positive_indices:

        return "NO_POSITIVE_ATTENDANCE"

    first_positive = min(
        positive_indices
    )

    last_positive = max(
        positive_indices
    )

    if month_index < first_positive:

        return "BEFORE_FIRST_POSITIVE"

    if month_index > last_positive:

        return "AFTER_LAST_POSITIVE"

    return "BETWEEN_POSITIVES"


special_records = []
monthly_records = []
student_pattern_records = []


print(
    "=" * 100
)

print(
    "STUDENT PERFORMANCE AI - "
    "ATTENDANCE VALUE SEMANTICS AUDIT"
)

print(
    "=" * 100
)


for file_name, year in (
    ATTENDANCE_FILES.items()
):

    filepath = (
        RAW_DATA_DIR
        / file_name
    )

    workbook = load_workbook(
        filepath,
        read_only=False,
        data_only=True
    )

    research_months = (
        RESEARCH_MONTHS[
            year
        ]
    )

    calendar = get_calendar_map(
        workbook,
        research_months
    )

    print(
        f"\n{file_name} | {year}"
    )

    for ws in workbook.worksheets:

        if not is_target_sheet(
            ws.title
        ):
            continue

        # Known unavailable attendance

        if (
            year,
            ws.title
        ) in KNOWN_UNAVAILABLE:

            print(
                f"{ws.title:<4} | "
                f"SKIPPED - known unavailable"
            )

            continue

        grade, class_name = (
            parse_grade_class(
                ws.title
            )
        )

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

        identifier_col = (
            find_identifier_column(
                ws,
                header_row,
                month_columns
            )
        )

        class_student_count = 0

        # Student rows

        for row_num in range(
            header_row + 1,
            ws.max_row + 1
        ):

            identifier = clean_text(
                ws.cell(
                    row=row_num,
                    column=identifier_col
                ).value
            )

            if not identifier:
                continue

            values = []

            for month in research_months:

                col_num = (
                    month_columns.get(
                        month
                    )
                )

                value = (
                    ws.cell(
                        row=row_num,
                        column=col_num
                    ).value
                    if col_num
                    else None
                )

                values.append(
                    value
                )

            # Skip rows with no attendance information
            if not any(
                clean_text(value)
                for value in values
            ):
                continue

            class_student_count += 1

            value_types = [
                classify_value(value)
                for value in values
            ]

            positive_indices = [
                i
                for i, value_type
                in enumerate(
                    value_types
                )
                if value_type
                == "POSITIVE"
            ]

            # Per-student pattern

            student_pattern_records.append(
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

                    "excel_row":
                        row_num,

                    "identifier":
                        identifier,

                    "positive_months":
                        value_types.count(
                            "POSITIVE"
                        ),

                    "zero_months":
                        value_types.count(
                            "ZERO"
                        ),

                    "dash_months":
                        value_types.count(
                            "DASH"
                        ),

                    "empty_months":
                        value_types.count(
                            "EMPTY"
                        ),
                }
            )

            # Exact special-value cells

            for i, month in enumerate(
                research_months
            ):

                value_type = (
                    value_types[i]
                )

                if value_type not in {
                    "ZERO",
                    "DASH",
                    "EMPTY",
                }:
                    continue

                calendar_info = (
                    calendar.get(
                        month,
                        {}
                    )
                )

                school_days = (
                    calendar_info.get(
                        "school_days"
                    )
                )

                term = (
                    calendar_info.get(
                        "term"
                    )
                )

                position = (
                    determine_position(
                        i,
                        positive_indices,
                        school_days
                    )
                )

                special_records.append(
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

                        "excel_row":
                            row_num,

                        "identifier":
                            identifier,

                        "month":
                            month,

                        "term":
                            term,

                        "school_days":
                            school_days,

                        "raw_value":
                            values[i],

                        "value_type":
                            value_type,

                        "position":
                            position,
                    }
                )

        print(
            f"{ws.title:<4} | "
            f"Students analysed: "
            f"{class_student_count}"
        )

    workbook.close()


special_df = pd.DataFrame(
    special_records
)

student_pattern_df = pd.DataFrame(
    student_pattern_records
)


if not special_df.empty:

    monthly_df = (
        special_df
        .groupby(
            [
                "year",
                "month",
                "term",
                "school_days",
                "value_type",
                "position",
            ],
            dropna=False
        )
        .size()
        .reset_index(
            name="count"
        )
    )

else:

    monthly_df = pd.DataFrame(
        columns=[
            "year",
            "month",
            "term",
            "school_days",
            "value_type",
            "position",
            "count",
        ]
    )


special_df.to_csv(
    SPECIAL_VALUES_OUTPUT,
    index=False
)

monthly_df.to_csv(
    MONTH_SUMMARY_OUTPUT,
    index=False
)

student_pattern_df.to_csv(
    STUDENT_PATTERN_OUTPUT,
    index=False
)


print(
    "\n"
    + "=" * 100
)

print(
    "ATTENDANCE VALUE SEMANTICS AUDIT COMPLETE"
)

print(
    "=" * 100
)


if not special_df.empty:

    print(
        "\nSpecial values:"
    )

    print(
        special_df[
            "value_type"
        ]
        .value_counts()
        .to_string()
    )

    print(
        "\nSpecial values by context:"
    )

    print(
        special_df
        .groupby(
            [
                "value_type",
                "position"
            ]
        )
        .size()
        .to_string()
    )

    print(
        "\nSpecial values in months "
        "with zero school days:"
    )

    zero_school_days = (
        special_df[
            special_df[
                "school_days"
            ] == 0
        ]
    )

    if zero_school_days.empty:

        print(
            "None"
        )

    else:

        print(
            zero_school_days[
                "value_type"
            ]
            .value_counts()
            .to_string()
        )


print(
    "\nOutputs:"
)

print(
    f"  {SPECIAL_VALUES_OUTPUT}"
)

print(
    f"  {MONTH_SUMMARY_OUTPUT}"
)

print(
    f"  {STUDENT_PATTERN_OUTPUT}"
)
