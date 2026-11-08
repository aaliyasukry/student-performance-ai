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

REPORTS_DIR.mkdir(exist_ok=True)

OUTPUT_CSV = DATA_QUALITY_DIR / "marks_structure_audit.csv"
LEGACY_MISSING_ID_CSV = REPORTS_DIR / "students_missing_index.csv"


MARK_FILES = {
    "marks_2025_t1.xlsx": (2025, 1),
    "marks_2025_t2.xlsx": (2025, 2),
    "marks_2025_t3.xlsx": (2025, 3),
    "marks_2026_t1.xlsx": (2026, 1),
    "marks_2026_t2.xlsx": (2026, 2),
}


# After student rows begin, stop scanning after this many consecutive empty identity rows.
MAX_CONSECUTIVE_EMPTY_ROWS = 5


# Obvious rows that should never be treated as students
NON_STUDENT_KEYWORDS = {
    "total",
    "average",
    "rank",
    "summary",
    "subject",
    "student no",
    "student number",
    "index",
    "index no",
    "index #",
    "name",
    "student name",
    "class average",
    "grade average",
    "pass",
    "fail",
    "remarks",
    "remark",
    "teacher",
    "class teacher",
    "principal",
    "signature",
    "date",
}


# Ignore these header words when checking whether a no-ID row has marks.
NON_MARK_COLUMN_KEYWORDS = {
    "student",
    "index",
    "name",
    "total",
    "average",
    "avg",
    "rank",
    "position",
    "grade",
    "remarks",
    "remark",
    "signature",
    "teacher",
    "principal",
    "date",
    "count",
    "no. of",
    "number of",
}


# Exam-status values that mark legitimate examination rows.
EXAM_STATUS_VALUES = {
    "ab",
    "abs",
    "absent",
}


def clean_text(value):
    """Convert a value to stripped text safely."""

    if value is None:
        return ""

    return str(value).strip()


def normalized(value):
    """Normalize text for comparisons."""

    text = clean_text(value).lower()

    text = re.sub(r"[\n\r\t]+", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def is_target_class_sheet(sheet_name):
    """Only Grades 6-10 sheets are in scope."""

    return bool(
        re.fullmatch(
            r"(6|7|8|9|10)[A-Za-z]+",
            sheet_name.strip()
        )
    )


def parse_grade_class(sheet_name):
    """Split a sheet name such as 10B into grade 10 and class B."""

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
    """Current research grouping."""

    if grade in [6, 7, 8, 9]:
        return "SECONDARY"

    if grade == 10:
        return "OL"

    return "UNKNOWN"


def detect_header_row(ws, search_rows=25):
    """Find the row most likely to contain student identity headers such as Index and Name."""

    best_row = None
    best_score = -1

    limit = min(
        search_rows,
        ws.max_row
    )

    for row_num in range(1, limit + 1):

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

        score = 0

        for value in values:

            if "index" in value:
                score += 5

            if (
                value == "name"
                or "student name" in value
            ):
                score += 5

            if "student no" in value:
                score += 3

            if "student number" in value:
                score += 3

            if "admission" in value:
                score += 2

        if score > best_score:
            best_score = score
            best_row = row_num

    if best_score <= 0:
        return None

    return best_row


def find_identity_columns(
    ws,
    header_row
):
    """Detect the Student No., Index No. and Student Name columns."""

    student_no_col = None
    index_col = None
    name_col = None

    if header_row is None:
        return (
            student_no_col,
            index_col,
            name_col
        )

    start_row = max(
        1,
        header_row - 2
    )

    end_row = min(
        ws.max_row,
        header_row + 2
    )

    for row_num in range(
        start_row,
        end_row + 1
    ):

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

            if student_no_col is None:

                if (
                    "student no" in value
                    or
                    "student number" in value
                ):
                    student_no_col = col_num

            if index_col is None:

                if "index" in value:
                    index_col = col_num

            if name_col is None:

                if (
                    value == "name"
                    or
                    value == "student name"
                    or
                    "student name" in value
                ):
                    name_col = col_num

    return (
        student_no_col,
        index_col,
        name_col
    )


def get_column_header_text(
    ws,
    col_num,
    header_row
):
    """Combine nearby header cells for a column."""

    start_row = max(
        1,
        header_row - 2
    )

    end_row = min(
        ws.max_row,
        header_row + 2
    )

    parts = []

    for row_num in range(
        start_row,
        end_row + 1
    ):

        value = clean_text(
            ws.cell(
                row=row_num,
                column=col_num
            ).value
        )

        if value:
            parts.append(value)

    return " ".join(parts)


def detect_possible_mark_columns(
    ws,
    header_row,
    student_no_col,
    index_col,
    name_col
):
    """Find columns that could contain subject marks."""

    identity_columns = {
        col
        for col in [
            student_no_col,
            index_col,
            name_col
        ]
        if col is not None
    }

    possible_columns = []

    for col_num in range(
        1,
        ws.max_column + 1
    ):

        if col_num in identity_columns:
            continue

        header_text = normalized(
            get_column_header_text(
                ws,
                col_num,
                header_row
            )
        )

        # Completely empty header columns are not useful
        if not header_text:
            continue

        # Reject obvious calculated/admin columns
        skip = False

        for keyword in NON_MARK_COLUMN_KEYWORDS:

            if keyword in header_text:
                skip = True
                break

        if skip:
            continue

        possible_columns.append(
            col_num
        )

    return possible_columns


def contains_non_student_keyword(text):
    """Detect obvious footer/header/signature text."""

    text = normalized(text)

    if not text:
        return False

    for keyword in NON_STUDENT_KEYWORDS:

        if text == keyword:
            return True

        if text.startswith(
            keyword + " "
        ):
            return True

        if text.startswith(
            keyword + ":"
        ):
            return True

    return False


def looks_like_student_name(value):
    """Basic student-name validation."""

    text = clean_text(value)

    if not text:
        return False

    if contains_non_student_keyword(
        text
    ):
        return False

    # Reject a pure number
    try:
        float(text)
        return False
    except ValueError:
        pass

    # Must contain at least one alphabetic character
    if not re.search(
        r"[A-Za-z]",
        text
    ):
        return False

    return True


def get_student_identity(
    ws,
    row_num,
    student_no_col,
    index_col,
    name_col
):
    """Read student identity cells."""

    student_no = (
        clean_text(
            ws.cell(
                row=row_num,
                column=student_no_col
            ).value
        )
        if student_no_col
        else ""
    )

    index_no = (
        clean_text(
            ws.cell(
                row=row_num,
                column=index_col
            ).value
        )
        if index_col
        else ""
    )

    name = (
        clean_text(
            ws.cell(
                row=row_num,
                column=name_col
            ).value
        )
        if name_col
        else ""
    )

    return (
        student_no,
        index_no,
        name
    )


def is_valid_mark_value(value):
    """Determine whether a cell looks like real examination data."""

    if value is None:
        return False

    # Actual numeric cell
    if isinstance(
        value,
        (int, float)
    ):

        try:
            number = float(value)

            if 0 <= number <= 100:
                return True

        except Exception:
            return False

    text = normalized(value)

    if not text:
        return False

    # Exam absence/status
    if text in EXAM_STATUS_VALUES:
        return True

    # Numeric stored as text
    try:
        number = float(text)

        if 0 <= number <= 100:
            return True

    except ValueError:
        pass

    return False


def row_has_academic_data(
    ws,
    row_num,
    mark_columns
):
    """Check whether the row contains at least one legitimate mark/exam-status value."""

    for col_num in mark_columns:

        value = ws.cell(
            row=row_num,
            column=col_num
        ).value

        if is_valid_mark_value(
            value
        ):
            return True

    return False


def classify_student_row(
    ws,
    row_num,
    student_no_col,
    index_col,
    name_col,
    mark_columns
):
    """Classify a row as valid_student_with_id, valid_student_without_id, empty or non_student."""

    (
        student_no,
        index_no,
        name
    ) = get_student_identity(
        ws,
        row_num,
        student_no_col,
        index_col,
        name_col
    )

    # Completely empty identity row

    if (
        not student_no
        and not index_no
        and not name
    ):

        return (
            "empty",
            student_no,
            index_no,
            name
        )

    # Reject obvious footer/header text

    combined_identity = " ".join(
        [
            student_no,
            index_no,
            name
        ]
    )

    if contains_non_student_keyword(
        combined_identity
    ):

        return (
            "non_student",
            student_no,
            index_no,
            name
        )

    # Name must look like a person/student name

    valid_name = looks_like_student_name(
        name
    )

    if not valid_name:

        return (
            "non_student",
            student_no,
            index_no,
            name
        )

    # Check if row contains academic data

    has_academic_data = (
        row_has_academic_data(
            ws,
            row_num,
            mark_columns
        )
    )

    # Student with ID

    if (
        student_no
        or index_no
    ):

        return (
            "valid_student_with_id",
            student_no,
            index_no,
            name
        )

    # Student without ID: needs a name and real examination data.

    if has_academic_data:

        return (
            "valid_student_without_id",
            student_no,
            index_no,
            name
        )

    # Name but no ID or marks: likely a teacher, principal or footer row.

    return (
        "non_student",
        student_no,
        index_no,
        name
    )


def detect_student_region(
    ws,
    header_row,
    student_no_col,
    index_col,
    name_col,
    mark_columns
):
    """Detect genuine student records."""

    if header_row is None:

        return {
            "with_id_count": 0,
            "without_id_count": 0,
            "total_student_count": 0,
            "first_student_row": None,
            "last_student_row": None,
            "missing_id_students": []
        }

    students_with_id = []
    students_without_id = []

    started = False
    consecutive_empty = 0

    for row_num in range(
        header_row + 1,
        ws.max_row + 1
    ):

        (
            classification,
            student_no,
            index_no,
            name
        ) = classify_student_row(
            ws,
            row_num,
            student_no_col,
            index_col,
            name_col,
            mark_columns
        )

        # Valid student with identifier

        if (
            classification
            == "valid_student_with_id"
        ):

            started = True
            consecutive_empty = 0

            students_with_id.append(
                row_num
            )

        # Valid student without identifier

        elif (
            classification
            == "valid_student_without_id"
        ):

            started = True
            consecutive_empty = 0

            students_without_id.append(
                {
                    "excel_row": row_num,
                    "student_no": student_no,
                    "index_no": index_no,
                    "name": name
                }
            )

        # Empty row

        elif classification == "empty":

            if started:

                consecutive_empty += 1

                if (
                    consecutive_empty
                    >= MAX_CONSECUTIVE_EMPTY_ROWS
                ):
                    break

        # Footer / teacher / principal etc.

        else:

            if started:
                consecutive_empty = 0

    all_student_rows = (
        students_with_id
        +
        [
            item["excel_row"]
            for item in students_without_id
        ]
    )

    if all_student_rows:

        first_student_row = min(
            all_student_rows
        )

        last_student_row = max(
            all_student_rows
        )

    else:

        first_student_row = None
        last_student_row = None

    return {
        "with_id_count": len(
            students_with_id
        ),

        "without_id_count": len(
            students_without_id
        ),

        "total_student_count": (
            len(students_with_id)
            +
            len(students_without_id)
        ),

        "first_student_row": (
            first_student_row
        ),

        "last_student_row": (
            last_student_row
        ),

        "missing_id_students": (
            students_without_id
        )
    }


def get_header_snapshot(
    ws,
    header_row
):
    """Capture nearby headers for debugging."""

    if header_row is None:
        return ""

    start_row = max(
        1,
        header_row - 2
    )

    end_row = min(
        ws.max_row,
        header_row + 2
    )

    items = []

    for row_num in range(
        start_row,
        end_row + 1
    ):

        for col_num in range(
            1,
            ws.max_column + 1
        ):

            value = clean_text(
                ws.cell(
                    row=row_num,
                    column=col_num
                ).value
            )

            if value:

                items.append(
                    f"{get_column_letter(col_num)}"
                    f"{row_num}={value}"
                )

    return " | ".join(items)


results = []
missing_id_students = []


print("=" * 90)
print(
    "STUDENT PERFORMANCE AI - "
    "MARKS STRUCTURE AUDIT"
)
print("=" * 90)


for file_name, (
    year,
    term
) in MARK_FILES.items():

    filepath = (
        RAW_DATA_DIR
        / file_name
    )

    if not filepath.exists():

        print(
            f"\nMISSING FILE: "
            f"{file_name}"
        )

        continue

    print(
        f"\n{'-' * 90}"
    )

    print(
        f"{file_name}  |  "
        f"Year {year}  |  "
        f"Term {term}"
    )

    print(
        f"{'-' * 90}"
    )

    workbook = load_workbook(
        filepath,
        read_only=False,
        data_only=False
    )

    target_sheets = [
        ws
        for ws in workbook.worksheets
        if is_target_class_sheet(
            ws.title
        )
    ]

    for ws in target_sheets:

        grade, class_name = (
            parse_grade_class(
                ws.title
            )
        )

        academic_stage = (
            get_academic_stage(
                grade
            )
        )

        # Detect structure

        header_row = (
            detect_header_row(
                ws
            )
        )

        (
            student_no_col,
            index_col,
            name_col
        ) = find_identity_columns(
            ws,
            header_row
        )

        mark_columns = (
            detect_possible_mark_columns(
                ws,
                header_row,
                student_no_col,
                index_col,
                name_col
            )
        )

        # Detect students

        region = (
            detect_student_region(
                ws,
                header_row,
                student_no_col,
                index_col,
                name_col,
                mark_columns
            )
        )

        header_snapshot = (
            get_header_snapshot(
                ws,
                header_row
            )
        )

        # Main audit record

        result = {

            "file_name":
                file_name,

            "year":
                year,

            "term":
                term,

            "sheet_name":
                ws.title,

            "grade":
                grade,

            "class":
                class_name,

            "academic_stage":
                academic_stage,

            "excel_max_row":
                ws.max_row,

            "excel_max_column":
                ws.max_column,

            "detected_header_row":
                header_row,

            "student_no_column":
                (
                    get_column_letter(
                        student_no_col
                    )
                    if student_no_col
                    else None
                ),

            "index_column":
                (
                    get_column_letter(
                        index_col
                    )
                    if index_col
                    else None
                ),

            "name_column":
                (
                    get_column_letter(
                        name_col
                    )
                    if name_col
                    else None
                ),

            "possible_mark_columns":
                len(
                    mark_columns
                ),

            "first_student_row":
                region[
                    "first_student_row"
                ],

            "last_student_row":
                region[
                    "last_student_row"
                ],

            "students_with_id":
                region[
                    "with_id_count"
                ],

            "students_without_id":
                region[
                    "without_id_count"
                ],

            "detected_student_rows":
                region[
                    "total_student_count"
                ],

            "header_snapshot":
                header_snapshot
        }

        results.append(
            result
        )

        # Missing-ID student records

        for student in region[
            "missing_id_students"
        ]:

            missing_id_students.append(
                {
                    "file_name":
                        file_name,

                    "year":
                        year,

                    "term":
                        term,

                    "sheet_name":
                        ws.title,

                    "grade":
                        grade,

                    "class":
                        class_name,

                    "academic_stage":
                        academic_stage,

                    "excel_row":
                        student[
                            "excel_row"
                        ],

                    "student_no":
                        student[
                            "student_no"
                        ],

                    "index_no":
                        student[
                            "index_no"
                        ],

                    "name":
                        student[
                            "name"
                        ],
                }
            )

        # Console output

        print(
            f"{ws.title:<4} | "
            f"Header: "
            f"{str(header_row):<3} | "
            f"Index: "
            f"{str(result['index_column']):<3} | "
            f"Name: "
            f"{str(result['name_column']):<3} | "
            f"Students: "
            f"{region['total_student_count']:<3} | "
            f"Missing ID: "
            f"{region['without_id_count']}"
        )

    workbook.close()


audit_df = pd.DataFrame(
    results
)

audit_df.to_csv(
    OUTPUT_CSV,
    index=False
)


# The old student-level CSV contained PII and is no longer written.
if LEGACY_MISSING_ID_CSV.exists():
    LEGACY_MISSING_ID_CSV.unlink()


print(
    "\n"
    + "=" * 90
)

print(
    "MARKS STRUCTURE AUDIT COMPLETE"
)

print(
    "=" * 90
)


print(
    f"\nClass sheets audited: "
    f"{len(audit_df)}"
)


if not audit_df.empty:

    print(
        "\nSheets per examination:"
    )

    print(
        audit_df
        .groupby(
            [
                "year",
                "term"
            ]
        )
        .size()
        .to_string()
    )

    print(
        "\nHeader rows detected:"
    )

    print(
        audit_df[
            "detected_header_row"
        ]
        .value_counts(
            dropna=False
        )
        .sort_index()
        .to_string()
    )

    print(
        "\nStudent records detected "
        "(all sheets/terms):"
    )

    print(
        audit_df[
            "detected_student_rows"
        ].sum()
    )

    print(
        "\nStudent records with "
        "identifier:"
    )

    print(
        audit_df[
            "students_with_id"
        ].sum()
    )

    print(
        "\nStudent records without "
        "identifier but with academic data:"
    )

    print(
        audit_df[
            "students_without_id"
        ].sum()
    )

    print(
        "\nRecords by academic stage:"
    )

    print(
        audit_df
        .groupby(
            "academic_stage"
        )[
            "detected_student_rows"
        ]
        .sum()
        .to_string()
    )


print(
    "\nMain audit saved to:"
)

print(
    OUTPUT_CSV
)
