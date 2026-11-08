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


MARKS_LONG_OUTPUT = CLEANED_DATA_DIR / "marks_long.csv"

TERM_SUMMARY_OUTPUT = (
    CLEANED_DATA_DIR / "marks_student_term_summary.csv"
)

PARSE_SUMMARY_OUTPUT = DATA_QUALITY_DIR / "marks_parse_summary.csv"

SUBJECT_AUDIT_OUTPUT = DATA_QUALITY_DIR / "marks_subjects_detected.csv"

ISSUES_OUTPUT = REPORTS_DIR / "marks_parse_issues.csv"

CORRECTIONS_OUTPUT = DATA_QUALITY_DIR / "marks_corrections_applied.csv"


MARK_FILES = {
    "marks_2025_t1.xlsx": {
        "year": 2025,
        "term": 1,
    },
    "marks_2025_t2.xlsx": {
        "year": 2025,
        "term": 2,
    },
    "marks_2025_t3.xlsx": {
        "year": 2025,
        "term": 3,
    },
    "marks_2026_t1.xlsx": {
        "year": 2026,
        "term": 1,
    },
    "marks_2026_t2.xlsx": {
        "year": 2026,
        "term": 2,
    },
}


# Explicit corrections keyed by (file, sheet, row, column) for an audit trail.

DATA_CORRECTIONS = {
    (
        "marks_2026_t1.xlsx",
        "8A",
        25,
        "H",
    ): {
        "corrected_value": 81,
        "reason": (
            "Raw source value was 814. "
            "Confirmed during manual data audit "
            "to be treated as 81."
        ),
    }
}


# Manually audited labels that are worksheet summaries, not students.

NON_STUDENT_ROW_LABELS = {
    "s",
    "a+b+c",
    "a + b + c",
    "percentage",
    "above 40",
}


SUBJECT_ALIASES = {
    # Languages
    "tamil": "Tamil",
    "english": "English",
    "sinhala": "Sinhala",

    # Core
    "maths": "Mathematics",
    "math": "Mathematics",
    "mathematics": "Mathematics",
    "science": "Science",
    "religion": "Religion",
    "riligion": "Religion",
    "islam": "Religion",
    "saivanery": "Religion",
    "history": "History",
    "geography": "Geography",

    # Secondary subjects
    "art": "Art",
    "helth & phy.ed": "Health & Physical Education",
    "health & phy.ed": "Health & Physical Education",
    "health & physical education": "Health & Physical Education",
    "health and physical education": "Health & Physical Education",
    "ce": "CE",
    "ict": "ICT",
    "pts": "PTS",

    # O/L basket subjects
    "commerc": "Commerce",
    "commerce": "Commerce",
    "tamil lit": "Tamil Literature",
    "tamil literature": "Tamil Literature",
    "arab. lit": "Arabic Literature",
    "arab lit": "Arabic Literature",
    "arabic lit": "Arabic Literature",
    "arabic literature": "Arabic Literature",
    "eng. lit": "English Literature",
    "eng lit": "English Literature",
    "english lit": "English Literature",
    "english literature": "English Literature",
    "home sc": "Home Science",
    "home science": "Home Science",
    "agriculture": "Agriculture",
}


OL_CORE_SUBJECTS = {
    "Tamil",
    "English",
    "Mathematics",
    "Science",
    "Religion",
    "History",
}

OL_GROUP_1 = {
    "Sinhala",
    "Commerce",
}

OL_GROUP_2 = {
    "Tamil Literature",
    "Arabic Literature",
    "English Literature",
    "Art",
}

OL_GROUP_3 = {
    "Home Science",
    "ICT",
    "Agriculture",
}


ABSENT_VALUES = {
    "ab",
    "abs",
    "absent",
}


def clean_text(value):
    """Convert any cell value to stripped text."""

    if value is None:
        return ""

    return str(value).strip()


def normalized(value):
    """Normalize text for comparisons."""

    text = clean_text(value).lower()

    text = re.sub(
        r"[\n\r\t]+",
        " ",
        text,
    )

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


def clean_identifier(value):
    """Preserve identifiers as text."""

    if value is None:
        return ""

    if isinstance(value, int):
        return str(value)

    if isinstance(value, float):
        if value.is_integer():
            return str(int(value))

        return str(value)

    return clean_text(value)


def is_target_sheet(sheet_name):
    """Only Grades 6-10 sheets are in scope."""

    return bool(
        re.fullmatch(
            r"(6|7|8|9|10)[A-Za-z]+",
            sheet_name.strip(),
        )
    )


def parse_grade_class(sheet_name):
    """Split a sheet name such as 10B into grade 10 and class B."""

    match = re.fullmatch(
        r"(6|7|8|9|10)([A-Za-z]+)",
        sheet_name.strip(),
    )

    if not match:
        return None, None

    return (
        int(match.group(1)),
        match.group(2).upper(),
    )


def get_academic_stage(grade):
    """Research grouping."""

    if grade in {
        6,
        7,
        8,
        9,
    }:
        return "SECONDARY"

    if grade == 10:
        return "OL"

    return "UNKNOWN"


def detect_header_row(
    ws,
    search_rows=25,
):
    """Locate the student identity header row."""

    best_row = None
    best_score = -1

    max_search_row = min(
        search_rows,
        ws.max_row,
    )

    for row_num in range(
        1,
        max_search_row + 1,
    ):
        score = 0

        for col_num in range(
            1,
            ws.max_column + 1,
        ):
            value = normalized(
                ws.cell(
                    row=row_num,
                    column=col_num,
                ).value
            )

            if "index" in value:
                score += 5

            if (
                value == "name"
                or value == "student name"
            ):
                score += 5

            if (
                "student no" in value
                or "student number" in value
            ):
                score += 3

        if score > best_score:
            best_score = score
            best_row = row_num

    if best_score <= 0:
        return None

    return best_row


def find_identity_columns(
    ws,
    header_row,
):
    """Find the Student No, Index No and Name columns."""

    student_no_col = None
    index_col = None
    name_col = None

    if header_row is None:
        return (
            None,
            None,
            None,
        )

    start_row = max(
        1,
        header_row - 1,
    )

    end_row = min(
        ws.max_row,
        header_row + 1,
    )

    for row_num in range(
        start_row,
        end_row + 1,
    ):
        for col_num in range(
            1,
            ws.max_column + 1,
        ):
            value = normalized(
                ws.cell(
                    row=row_num,
                    column=col_num,
                ).value
            )

            if (
                index_col is None
                and "index" in value
            ):
                index_col = col_num

            if name_col is None:
                if (
                    value == "name"
                    or value == "student name"
                ):
                    name_col = col_num

            if student_no_col is None:
                if (
                    "student no" in value
                    or "student number" in value
                    or value
                    in {
                        "s#",
                        "s.no",
                        "s no",
                        "#",
                    }
                ):
                    student_no_col = col_num

    # Fallback: the column before Index is often Student No.
    if (
        student_no_col is None
        and index_col is not None
        and index_col > 1
    ):
        candidate_col = index_col - 1

        candidate_text = normalized(
            ws.cell(
                row=header_row,
                column=candidate_col,
            ).value
        )

        if candidate_text in {
            "",
            "\\",
            "/",
            "#",
            "s#",
        }:
            student_no_col = candidate_col

    return (
        student_no_col,
        index_col,
        name_col,
    )


def find_summary_columns(
    ws,
    header_row,
):
    """Detect optional Total / Average / Rank columns."""

    summary_columns = {
        "total": None,
        "average": None,
        "rank": None,
    }

    start_row = max(
        1,
        header_row - 1,
    )

    end_row = min(
        ws.max_row,
        header_row + 2,
    )

    for col_num in range(
        1,
        ws.max_column + 1,
    ):
        for row_num in range(
            start_row,
            end_row + 1,
        ):
            value = normalized(
                ws.cell(
                    row=row_num,
                    column=col_num,
                ).value
            )

            if (
                value == "total"
                and summary_columns["total"] is None
            ):
                summary_columns["total"] = col_num

            elif (
                value == "average"
                and summary_columns["average"] is None
            ):
                summary_columns["average"] = col_num

            elif (
                value == "rank"
                and summary_columns["rank"] is None
            ):
                summary_columns["rank"] = col_num

    return summary_columns


def standardize_subject(value):
    """Convert raw school subject header into standard subject name."""

    key = normalized(value)

    if not key:
        return None

    return SUBJECT_ALIASES.get(key)


def detect_subject_columns(
    ws,
    header_row,
    name_col,
    summary_columns,
):
    """Detect actual subject-mark columns."""

    subjects = []

    if name_col is None:
        return subjects

    first_subject_col = name_col + 1

    # Find earliest available summary boundary

    boundaries = [
        col
        for col in [
            summary_columns.get("total"),
            summary_columns.get("average"),
            summary_columns.get("rank"),
        ]
        if col is not None
    ]

    if boundaries:
        last_subject_col = min(boundaries) - 1
    else:
        last_subject_col = ws.max_column

    # Subject headings may span multiple rows

    candidate_rows = range(
        max(
            1,
            header_row - 1,
        ),
        min(
            ws.max_row,
            header_row + 2,
        )
        + 1,
    )

    for col_num in range(
        first_subject_col,
        last_subject_col + 1,
    ):
        detected_subject = None
        raw_subject = None

        for row_num in candidate_rows:
            raw_value = ws.cell(
                row=row_num,
                column=col_num,
            ).value

            subject = standardize_subject(
                raw_value
            )

            if subject:
                detected_subject = subject
                raw_subject = clean_text(
                    raw_value
                )
                break

        if detected_subject:
            subjects.append(
                {
                    "column": col_num,
                    "column_letter": get_column_letter(
                        col_num
                    ),
                    "raw_subject": raw_subject,
                    "subject": detected_subject,
                }
            )

    return subjects


def group_subject_columns(subject_columns):
    """Group source columns that map to the same standardized subject."""

    grouped = {}

    for subject_info in subject_columns:
        subject = subject_info["subject"]
        grouped.setdefault(subject, []).append(subject_info)

    return [
        {
            "subject": subject,
            "columns": columns,
        }
        for subject, columns in grouped.items()
    ]


def get_subject_group(
    grade,
    subject,
):
    """Return the subject group and whether the subject is optional."""

    if grade in {
        6,
        7,
        8,
        9,
    }:
        return (
            "STANDARD",
            False,
        )

    if grade == 10:
        if subject in OL_CORE_SUBJECTS:
            return (
                "CORE",
                False,
            )

        if subject in OL_GROUP_1:
            return (
                "GROUP_1",
                True,
            )

        if subject in OL_GROUP_2:
            return (
                "GROUP_2",
                True,
            )

        if subject in OL_GROUP_3:
            return (
                "GROUP_3",
                True,
            )

    return (
        "UNKNOWN",
        False,
    )


def is_confirmed_non_student_summary_row(
    student_name,
):
    """Identify summary rows manually confirmed not to be students."""

    return (
        normalized(student_name)
        in NON_STUDENT_ROW_LABELS
    )


def looks_like_student_name(value):
    """Basic check that row contains a student name rather than footer/signature text."""

    text = clean_text(value)

    if not text:
        return False

    lower = normalized(text)

    forbidden = {
        "class teacher",
        "teacher",
        "principal",
        "signature",
        "date",
        "total",
        "average",
        "rank",
        "name",
    }

    for keyword in forbidden:
        if (
            lower == keyword
            or lower.startswith(
                keyword + " "
            )
            or lower.startswith(
                keyword + ":"
            )
        ):
            return False

    try:
        float(text)
        return False

    except ValueError:
        pass

    if not re.search(
        r"[A-Za-z]",
        text,
    ):
        return False

    return True


def has_any_academic_data(
    ws,
    row_num,
    subject_columns,
):
    """Used to confirm students with no Index No."""

    for subject_info in subject_columns:
        value = ws.cell(
            row=row_num,
            column=subject_info["column"],
        ).value

        if value is None:
            continue

        text = normalized(value)

        if text in ABSENT_VALUES:
            return True

        if isinstance(
            value,
            (int, float),
        ):
            if (
                not isinstance(
                    value,
                    bool,
                )
                and -1000
                <= float(value)
                <= 1000
            ):
                return True

        try:
            float(text)
            return True

        except (
            ValueError,
            TypeError,
        ):
            pass

    return False


def apply_documented_correction(
    file_name,
    sheet_name,
    row_num,
    col_num,
    raw_value,
):
    """Apply explicit documented correction."""

    column_letter = get_column_letter(
        col_num
    )

    correction_key = (
        file_name,
        sheet_name,
        row_num,
        column_letter,
    )

    correction = DATA_CORRECTIONS.get(
        correction_key
    )

    if correction is None:
        return (
            raw_value,
            False,
            None,
        )

    return (
        correction["corrected_value"],
        True,
        correction["reason"],
    )


def parse_mark_value(
    value,
    is_optional,
):
    """Convert a cell into (mark, exam_status, issue); statuses: PRESENT, ABSENT, NOT_TAKEN, MISSING, INVALID."""

    # Empty

    if value is None:
        if is_optional:
            return (
                None,
                "NOT_TAKEN",
                None,
            )

        return (
            None,
            "MISSING",
            None,
        )

    text = clean_text(value)
    norm = normalized(value)

    if not text:
        if is_optional:
            return (
                None,
                "NOT_TAKEN",
                None,
            )

        return (
            None,
            "MISSING",
            None,
        )

    # Absent

    if norm in ABSENT_VALUES:
        return (
            None,
            "ABSENT",
            None,
        )

    # Numeric cell

    if (
        isinstance(
            value,
            (int, float),
        )
        and not isinstance(
            value,
            bool,
        )
    ):
        mark = float(value)

        if 0 <= mark <= 100:
            return (
                mark,
                "PRESENT",
                None,
            )

        return (
            None,
            "INVALID",
            (
                f"Numeric mark outside "
                f"0-100: {value}"
            ),
        )

    # Numeric stored as text

    try:
        mark = float(norm)

        if 0 <= mark <= 100:
            return (
                mark,
                "PRESENT",
                None,
            )

        return (
            None,
            "INVALID",
            (
                f"Text numeric mark outside "
                f"0-100: {value}"
            ),
        )

    except (
        ValueError,
        TypeError,
    ):
        pass

    # Unexpected value

    return (
        None,
        "INVALID",
        (
            f"Unrecognized mark value: "
            f"{value}"
        ),
    )


def read_summary_value(
    ws,
    row_num,
    col_num,
):
    """Read cached workbook summary value."""

    if col_num is None:
        return None

    return ws.cell(
        row=row_num,
        column=col_num,
    ).value


marks_records = []
term_summary_records = []
parse_summary_records = []
subject_audit_records = []
issue_records = []
correction_records = []


print("=" * 90)
print("STUDENT PERFORMANCE AI - MARKS PARSER")
print("=" * 90)


for file_name, metadata in MARK_FILES.items():

    year = metadata["year"]
    term = metadata["term"]

    filepath = RAW_DATA_DIR / file_name

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
        f"{file_name} | "
        f"Year {year} | "
        f"Term {term}"
    )

    print(
        f"{'-' * 90}"
    )

    workbook = load_workbook(
        filepath,
        read_only=False,
        data_only=True,
    )

    target_sheets = [
        ws
        for ws in workbook.worksheets
        if is_target_sheet(
            ws.title
        )
    ]

    for ws in target_sheets:

        grade, class_name = parse_grade_class(
            ws.title
        )

        academic_stage = get_academic_stage(
            grade
        )


        header_row = detect_header_row(
            ws
        )

        if header_row is None:

            issue_records.append(
                {
                    "file_name": file_name,
                    "sheet_name": ws.title,
                    "excel_row": None,
                    "excel_column": None,
                    "issue_type": "HEADER_NOT_FOUND",
                    "details": (
                        "Could not locate "
                        "student header row."
                    ),
                }
            )

            continue

        (
            student_no_col,
            index_col,
            name_col,
        ) = find_identity_columns(
            ws,
            header_row,
        )

        if (
            index_col is None
            or name_col is None
        ):

            issue_records.append(
                {
                    "file_name": file_name,
                    "sheet_name": ws.title,
                    "excel_row": header_row,
                    "excel_column": None,
                    "issue_type": (
                        "IDENTITY_COLUMNS_NOT_FOUND"
                    ),
                    "details": (
                        f"Index={index_col}, "
                        f"Name={name_col}"
                    ),
                }
            )

            continue

        # Optional summary columns

        summary_columns = find_summary_columns(
            ws,
            header_row,
        )

        # Subject columns

        subject_columns = detect_subject_columns(
            ws,
            header_row,
            name_col,
            summary_columns,
        )

        subject_column_groups = group_subject_columns(
            subject_columns
        )

        if not subject_columns:

            issue_records.append(
                {
                    "file_name": file_name,
                    "sheet_name": ws.title,
                    "excel_row": header_row,
                    "excel_column": None,
                    "issue_type": (
                        "NO_SUBJECTS_DETECTED"
                    ),
                    "details": (
                        "No recognized subject "
                        "columns were found."
                    ),
                }
            )

            continue


        for subject_info in subject_columns:

            (
                subject_group,
                is_optional,
            ) = get_subject_group(
                grade,
                subject_info["subject"],
            )

            subject_audit_records.append(
                {
                    "file_name": file_name,
                    "year": year,
                    "term": term,
                    "sheet_name": ws.title,
                    "grade": grade,
                    "class": class_name,
                    "academic_stage": academic_stage,
                    "column_letter": (
                        subject_info[
                            "column_letter"
                        ]
                    ),
                    "raw_subject": (
                        subject_info[
                            "raw_subject"
                        ]
                    ),
                    "standard_subject": (
                        subject_info[
                            "subject"
                        ]
                    ),
                    "subject_group": subject_group,
                    "is_optional": is_optional,
                }
            )


        student_count = 0
        missing_id_count = 0
        mark_record_count = 0
        invalid_count = 0
        corrections_count = 0
        excluded_non_student_count = 0

        started = False
        consecutive_empty_rows = 0

        for row_num in range(
            header_row + 1,
            ws.max_row + 1,
        ):

            raw_student_no = (
                clean_identifier(
                    ws.cell(
                        row=row_num,
                        column=student_no_col,
                    ).value
                )
                if student_no_col
                else ""
            )

            raw_index_no = clean_identifier(
                ws.cell(
                    row=row_num,
                    column=index_col,
                ).value
            )

            raw_name = clean_text(
                ws.cell(
                    row=row_num,
                    column=name_col,
                ).value
            )

            # Empty identity row

            if (
                not raw_student_no
                and not raw_index_no
                and not raw_name
            ):

                if started:

                    consecutive_empty_rows += 1

                    if (
                        consecutive_empty_rows
                        >= 5
                    ):
                        break

                continue

            # Runs before student validation so these rows never reach the output tables.

            if is_confirmed_non_student_summary_row(
                raw_name
            ):

                excluded_non_student_count += 1
                continue

            # Footer / teacher / principal row

            if not looks_like_student_name(
                raw_name
            ):
                continue

            # Missing Index student must contain marks

            if not raw_index_no:

                if not has_any_academic_data(
                    ws,
                    row_num,
                    subject_columns,
                ):
                    continue

                missing_id_count += 1

            started = True
            consecutive_empty_rows = 0
            student_count += 1


            student_numeric_marks = []

            absent_subjects = 0
            missing_subjects = 0
            not_taken_subjects = 0
            invalid_subjects = 0

            for subject_group_info in subject_column_groups:

                subject = subject_group_info[
                    "subject"
                ]

                source_columns = subject_group_info[
                    "columns"
                ]

                (
                    subject_group,
                    is_optional,
                ) = get_subject_group(
                    grade,
                    subject,
                )

                source_candidates = []

                for source_info in source_columns:

                    col_num = source_info[
                        "column"
                    ]

                    column_letter = source_info[
                        "column_letter"
                    ]

                    original_raw_value = ws.cell(
                        row=row_num,
                        column=col_num,
                    ).value

                    (
                        value_for_parsing,
                        correction_applied,
                        correction_reason,
                    ) = apply_documented_correction(
                        file_name,
                        ws.title,
                        row_num,
                        col_num,
                        original_raw_value,
                    )

                    if correction_applied:

                        corrections_count += 1

                        correction_records.append(
                            {
                                "file_name": file_name,
                                "sheet_name": ws.title,
                                "excel_row": row_num,
                                "excel_column": column_letter,
                                "raw_value": original_raw_value,
                                "corrected_value": value_for_parsing,
                                "reason": correction_reason,
                            }
                        )

                    source_candidates.append(
                        {
                            "column": col_num,
                            "column_letter": column_letter,
                            "raw_subject": source_info[
                                "raw_subject"
                            ],
                            "original_raw_value": original_raw_value,
                            "value_for_parsing": value_for_parsing,
                            "correction_applied": correction_applied,
                            "correction_reason": correction_reason,
                        }
                    )

                non_empty_candidates = [
                    candidate
                    for candidate in source_candidates
                    if clean_text(
                        candidate[
                            "value_for_parsing"
                        ]
                    )
                ]

                # Alternative source columns (e.g. Islam/Saivanery -> Religion); marks are never summed.

                if len(non_empty_candidates) > 1:

                    candidate_description = ", ".join(
                        f"{candidate['raw_subject']}="
                        f"{candidate['value_for_parsing']}"
                        for candidate
                        in non_empty_candidates
                    )

                    issue_records.append(
                        {
                            "file_name": file_name,
                            "sheet_name": ws.title,
                            "excel_row": row_num,
                            "excel_column": ",".join(
                                candidate[
                                    "column_letter"
                                ]
                                for candidate
                                in non_empty_candidates
                            ),
                            "issue_type": (
                                "MULTIPLE_SOURCE_VALUES_FOR_"
                                "STANDARD_SUBJECT"
                            ),
                            "details": (
                                f"Standardized subject {subject} "
                                f"has multiple populated source "
                                f"columns: {candidate_description}"
                            ),
                        }
                    )

                    selected_candidate = (
                        non_empty_candidates[0]
                    )

                    mark = None
                    exam_status = "INVALID"
                    issue = (
                        f"Multiple populated source columns "
                        f"for standardized subject {subject}."
                    )

                else:

                    if non_empty_candidates:
                        selected_candidate = (
                            non_empty_candidates[0]
                        )
                    else:
                        selected_candidate = (
                            source_candidates[0]
                        )

                    (
                        mark,
                        exam_status,
                        issue,
                    ) = parse_mark_value(
                        selected_candidate[
                            "value_for_parsing"
                        ],
                        is_optional,
                    )

                if mark is not None:
                    student_numeric_marks.append(
                        mark
                    )

                if exam_status == "ABSENT":
                    absent_subjects += 1

                elif exam_status == "MISSING":
                    missing_subjects += 1

                elif exam_status == "NOT_TAKEN":
                    not_taken_subjects += 1

                elif exam_status == "INVALID":
                    invalid_subjects += 1
                    invalid_count += 1

                marks_records.append(
                    {
                        "source_file": file_name,
                        "source_sheet": ws.title,
                        "source_excel_row": row_num,
                        "source_excel_column": (
                            selected_candidate[
                                "column_letter"
                            ]
                        ),
                        "source_subject_header": (
                            selected_candidate[
                                "raw_subject"
                            ]
                        ),
                        "year": year,
                        "term": term,
                        "grade": grade,
                        "class": class_name,
                        "academic_stage": (
                            academic_stage
                        ),
                        "student_no": (
                            raw_student_no
                            if raw_student_no
                            else None
                        ),
                        "index_no": (
                            raw_index_no
                            if raw_index_no
                            else None
                        ),
                        "student_name": raw_name,
                        "missing_index": (
                            not bool(
                                raw_index_no
                            )
                        ),
                        "subject": subject,
                        "subject_group": (
                            subject_group
                        ),
                        "is_optional_subject": (
                            is_optional
                        ),
                        "raw_mark": (
                            clean_text(
                                selected_candidate[
                                    "original_raw_value"
                                ]
                            )
                            if selected_candidate[
                                "original_raw_value"
                            ] is not None
                            else None
                        ),
                        "processed_raw_mark": (
                            clean_text(
                                selected_candidate[
                                    "value_for_parsing"
                                ]
                            )
                            if selected_candidate[
                                "value_for_parsing"
                            ] is not None
                            else None
                        ),
                        "mark": mark,
                        "exam_status": exam_status,
                        "correction_applied": (
                            selected_candidate[
                                "correction_applied"
                            ]
                        ),
                        "correction_reason": (
                            selected_candidate[
                                "correction_reason"
                            ]
                        ),
                    }
                )

                mark_record_count += 1

                if issue and exam_status != "INVALID":

                    issue_records.append(
                        {
                            "file_name": file_name,
                            "sheet_name": ws.title,
                            "excel_row": row_num,
                            "excel_column": (
                                selected_candidate[
                                    "column_letter"
                                ]
                            ),
                            "issue_type": (
                                "INVALID_MARK_VALUE"
                            ),
                            "details": issue,
                        }
                    )

                elif (
                    issue
                    and exam_status == "INVALID"
                    and len(non_empty_candidates) <= 1
                ):

                    issue_records.append(
                        {
                            "file_name": file_name,
                            "sheet_name": ws.title,
                            "excel_row": row_num,
                            "excel_column": (
                                selected_candidate[
                                    "column_letter"
                                ]
                            ),
                            "issue_type": (
                                "INVALID_MARK_VALUE"
                            ),
                            "details": issue,
                        }
                    )


            if student_numeric_marks:

                calculated_total = sum(
                    student_numeric_marks
                )

                calculated_average = (
                    calculated_total
                    / len(
                        student_numeric_marks
                    )
                )

            else:
                calculated_total = None
                calculated_average = None

            source_total = read_summary_value(
                ws,
                row_num,
                summary_columns["total"],
            )

            source_average = read_summary_value(
                ws,
                row_num,
                summary_columns["average"],
            )

            source_rank = read_summary_value(
                ws,
                row_num,
                summary_columns["rank"],
            )

            term_summary_records.append(
                {
                    "source_file": file_name,
                    "source_sheet": ws.title,
                    "source_excel_row": row_num,
                    "year": year,
                    "term": term,
                    "grade": grade,
                    "class": class_name,
                    "academic_stage": (
                        academic_stage
                    ),
                    "student_no": (
                        raw_student_no
                        if raw_student_no
                        else None
                    ),
                    "index_no": (
                        raw_index_no
                        if raw_index_no
                        else None
                    ),
                    "student_name": raw_name,
                    "missing_index": (
                        not bool(
                            raw_index_no
                        )
                    ),
                    "numeric_subject_count": len(
                        student_numeric_marks
                    ),
                    "absent_subject_count": (
                        absent_subjects
                    ),
                    "missing_subject_count": (
                        missing_subjects
                    ),
                    "not_taken_subject_count": (
                        not_taken_subjects
                    ),
                    "invalid_subject_count": (
                        invalid_subjects
                    ),
                    "calculated_total": (
                        calculated_total
                    ),
                    "calculated_average": (
                        calculated_average
                    ),
                    "source_total": source_total,
                    "source_average": (
                        source_average
                    ),
                    "source_rank": source_rank,
                }
            )


        parse_summary_records.append(
            {
                "file_name": file_name,
                "year": year,
                "term": term,
                "sheet_name": ws.title,
                "grade": grade,
                "class": class_name,
                "academic_stage": academic_stage,
                "header_row": header_row,
                "student_no_column": (
                    get_column_letter(
                        student_no_col
                    )
                    if student_no_col
                    else None
                ),
                "index_column": get_column_letter(
                    index_col
                ),
                "name_column": get_column_letter(
                    name_col
                ),
                "total_column": (
                    get_column_letter(
                        summary_columns[
                            "total"
                        ]
                    )
                    if summary_columns[
                        "total"
                    ]
                    else None
                ),
                "average_column": (
                    get_column_letter(
                        summary_columns[
                            "average"
                        ]
                    )
                    if summary_columns[
                        "average"
                    ]
                    else None
                ),
                "rank_column": (
                    get_column_letter(
                        summary_columns[
                            "rank"
                        ]
                    )
                    if summary_columns[
                        "rank"
                    ]
                    else None
                ),
                "subjects_detected": len(
                    subject_column_groups
                ),
                "source_subject_columns_detected": len(
                    subject_columns
                ),
                "students_detected": (
                    student_count
                ),
                "students_missing_index": (
                    missing_id_count
                ),
                "excluded_non_student_rows": (
                    excluded_non_student_count
                ),
                "mark_records_created": (
                    mark_record_count
                ),
                "invalid_mark_values": (
                    invalid_count
                ),
                "corrections_applied": (
                    corrections_count
                ),
            }
        )

        print(
            f"{ws.title:<4} | "
            f"Stage: "
            f"{academic_stage:<9} | "
            f"Students: "
            f"{student_count:<3} | "
            f"Subjects: "
            f"{len(subject_column_groups):<2} | "
            f"Missing ID: "
            f"{missing_id_count:<2} | "
            f"Excluded: "
            f"{excluded_non_student_count:<2} | "
            f"Invalid: "
            f"{invalid_count:<2} | "
            f"Corrected: "
            f"{corrections_count}"
        )

    workbook.close()


marks_df = pd.DataFrame(
    marks_records
)

term_summary_df = pd.DataFrame(
    term_summary_records
)

parse_summary_df = pd.DataFrame(
    parse_summary_records
)

subject_audit_df = pd.DataFrame(
    subject_audit_records
)

issues_df = pd.DataFrame(
    issue_records
)

corrections_df = pd.DataFrame(
    correction_records
)


if not marks_df.empty:

    marks_df = (
        marks_df
        .sort_values(
            by=[
                "year",
                "term",
                "grade",
                "class",
                "source_excel_row",
                "subject",
            ]
        )
    )


if not term_summary_df.empty:

    term_summary_df = (
        term_summary_df
        .sort_values(
            by=[
                "year",
                "term",
                "grade",
                "class",
                "source_excel_row",
            ]
        )
    )


if not parse_summary_df.empty:

    parse_summary_df = (
        parse_summary_df
        .sort_values(
            by=[
                "year",
                "term",
                "grade",
                "class",
            ]
        )
    )


if not subject_audit_df.empty:

    subject_audit_df = (
        subject_audit_df
        .sort_values(
            by=[
                "year",
                "term",
                "grade",
                "class",
                "column_letter",
            ]
        )
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


if corrections_df.empty:

    corrections_df = pd.DataFrame(
        columns=[
            "file_name",
            "sheet_name",
            "excel_row",
            "excel_column",
            "raw_value",
            "corrected_value",
            "reason",
        ]
    )


marks_df.to_csv(
    MARKS_LONG_OUTPUT,
    index=False,
)

term_summary_df.to_csv(
    TERM_SUMMARY_OUTPUT,
    index=False,
)

parse_summary_df.to_csv(
    PARSE_SUMMARY_OUTPUT,
    index=False,
)

subject_audit_df.to_csv(
    SUBJECT_AUDIT_OUTPUT,
    index=False,
)

issues_df.to_csv(
    ISSUES_OUTPUT,
    index=False,
)

corrections_df.to_csv(
    CORRECTIONS_OUTPUT,
    index=False,
)


print(
    "\n"
    + "=" * 90
)

print(
    "MARKS PARSING COMPLETE"
)

print(
    "=" * 90
)


print(
    f"\nClass sheets successfully parsed: "
    f"{len(parse_summary_df)}"
)


print(
    f"\nStudent-subject records created: "
    f"{len(marks_df)}"
)


print(
    f"Student-term records created: "
    f"{len(term_summary_df)}"
)


if not parse_summary_df.empty:

    excluded_non_student_total = int(
        parse_summary_df[
            "excluded_non_student_rows"
        ].sum()
    )

else:

    excluded_non_student_total = 0


print(
    f"Confirmed non-student summary rows excluded: "
    f"{excluded_non_student_total}"
)


if not marks_df.empty:

    print(
        "\nRecords by academic stage:"
    )

    print(
        marks_df
        .groupby(
            "academic_stage"
        )
        .size()
        .to_string()
    )


    print(
        "\nExam status distribution:"
    )

    print(
        marks_df[
            "exam_status"
        ]
        .value_counts(
            dropna=False
        )
        .to_string()
    )


    print(
        "\nStandardized subjects detected:"
    )

    subjects = sorted(
        marks_df[
            "subject"
        ]
        .dropna()
        .unique()
    )

    for subject in subjects:

        print(
            f"  - {subject}"
        )


if not term_summary_df.empty:

    missing_index_count = int(
        term_summary_df[
            "missing_index"
        ].sum()
    )

    print(
        f"\nStudent-term records "
        f"without Index No.: "
        f"{missing_index_count}"
    )


print(
    f"\nDocumented corrections applied: "
    f"{len(corrections_df)}"
)


print(
    f"Remaining parsing issues: "
    f"{len(issues_df)}"
)


if not issues_df.empty:

    print(
        "\nIssue types:"
    )

    print(
        issues_df[
            "issue_type"
        ]
        .value_counts()
        .to_string()
    )


print(
    "\nOutputs:"
)


print(
    "\n  Marks long dataset:"
)

print(
    f"    {MARKS_LONG_OUTPUT}"
)


print(
    "\n  Student-term summary:"
)

print(
    f"    {TERM_SUMMARY_OUTPUT}"
)


print(
    "\n  Parser summary:"
)

print(
    f"    {PARSE_SUMMARY_OUTPUT}"
)


print(
    "\n  Subject audit:"
)

print(
    f"    {SUBJECT_AUDIT_OUTPUT}"
)


print(
    "\n  Parsing issues:"
)

print(
    f"    {ISSUES_OUTPUT}"
)


print(
    "\n  Documented corrections:"
)

print(
    f"    {CORRECTIONS_OUTPUT}"
)
