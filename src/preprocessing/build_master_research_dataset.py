from pathlib import Path
import pandas as pd
import re


PROJECT_ROOT = Path(__file__).resolve().parents[2]

ANONYMIZED_DIR = (
    PROJECT_ROOT /
    "data" /
    "anonymized"
)

STUDENT_TERM_FILE = (
    ANONYMIZED_DIR /
    "student_term_identity.csv"
)

MARKS_LONG_FILE = (
    ANONYMIZED_DIR /
    "marks_long_anonymized.csv"
)

MARKS_TERM_FILE = (
    ANONYMIZED_DIR /
    "marks_student_term_summary_anonymized.csv"
)

ATTENDANCE_FILE = (
    ANONYMIZED_DIR /
    "attendance_term_anonymized.csv"
)

OUTPUT_FILE = (
    ANONYMIZED_DIR /
    "master_research_dataset.csv"
)


def clean_text(value):
    """Safely convert a value to stripped text."""

    if pd.isna(value):
        return ""

    return str(value).strip()


def to_numeric(series):
    """Convert values to numeric while preserving genuine missing values as NaN."""

    return pd.to_numeric(
        series,
        errors="coerce"
    )


def subject_slug(subject):
    """Convert subject names into stable column names."""

    text = clean_text(
        subject
    ).lower()

    text = text.replace(
        "&",
        " and "
    )

    text = re.sub(
        r"[^a-z0-9]+",
        "_",
        text
    )

    text = re.sub(
        r"_+",
        "_",
        text
    )

    return text.strip("_")


def require_columns(
    df,
    required,
    label
):
    """Validate required columns."""

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:

        raise ValueError(
            f"\nMissing columns in {label}:\n"
            f"{missing}\n\n"
            f"Available columns:\n"
            f"{list(df.columns)}"
        )


def assert_unique_student_term(
    df,
    label
):
    """Every student must have at most one row per academic term in student-term tables."""

    duplicate_mask = df.duplicated(
        subset=[
            "student_id",
            "year",
            "term",
        ],
        keep=False
    )

    if duplicate_mask.any():

        duplicates = df.loc[
            duplicate_mask,
            [
                "student_id",
                "year",
                "term",
                "grade",
                "class",
            ]
        ]

        raise RuntimeError(
            f"\nDuplicate student-term rows in "
            f"{label}:\n\n"
            f"{duplicates.to_string(index=False)}"
        )


for path in [
    STUDENT_TERM_FILE,
    MARKS_LONG_FILE,
    MARKS_TERM_FILE,
    ATTENDANCE_FILE,
]:

    if not path.exists():

        raise FileNotFoundError(
            f"Required master-dataset input not found:\n"
            f"{path}"
        )


student_term_df = pd.read_csv(
    STUDENT_TERM_FILE,
    dtype=str
)

marks_long_df = pd.read_csv(
    MARKS_LONG_FILE,
    dtype=str
)

marks_term_df = pd.read_csv(
    MARKS_TERM_FILE,
    dtype=str
)

attendance_df = pd.read_csv(
    ATTENDANCE_FILE,
    dtype=str
)


print(
    "=" * 110
)

print(
    "STUDENT PERFORMANCE AI - "
    "MASTER RESEARCH DATASET"
)

print(
    "=" * 110
)


require_columns(
    student_term_df,
    [
        "student_id",
        "year",
        "term",
        "grade",
        "class",
        "attendance_linkage_status",
        "attendance_linkage_method",
        "attendance_available",
    ],
    "student_term_identity.csv"
)


require_columns(
    marks_term_df,
    [
        "student_id",
        "year",
        "term",
        "grade",
        "class",
        "academic_stage",
    ],
    "marks_student_term_summary_anonymized.csv"
)


require_columns(
    marks_long_df,
    [
        "student_id",
        "year",
        "term",
        "grade",
        "class",
        "academic_stage",
        "subject",
        "mark",
        "exam_status",
    ],
    "marks_long_anonymized.csv"
)


require_columns(
    attendance_df,
    [
        "student_id",
        "year",
        "term",
        "grade",
        "class",
        "academic_stage",
        "expected_term_school_days",
        "observed_school_days",
        "days_present",
        "days_absent",
        "attendance_pct",
        "attendance_coverage_pct",
        "attendance_data_quality",
    ],
    "attendance_term_anonymized.csv"
)


assert_unique_student_term(
    student_term_df,
    "student_term_identity.csv"
)

assert_unique_student_term(
    marks_term_df,
    "marks_student_term_summary_anonymized.csv"
)

assert_unique_student_term(
    attendance_df,
    "attendance_term_anonymized.csv"
)


key_columns = [
    "year",
    "term",
    "grade",
]


for frame in [
    student_term_df,
    marks_term_df,
    marks_long_df,
    attendance_df,
]:

    for column in key_columns:

        frame[
            column
        ] = pd.to_numeric(
            frame[
                column
            ],
            errors="raise"
        ).astype(int)


for frame in [
    student_term_df,
    marks_term_df,
    marks_long_df,
    attendance_df,
]:

    frame[
        "class"
    ] = (
        frame[
            "class"
        ]
        .astype(str)
        .str.strip()
        .str.upper()
    )


subject_duplicate_mask = (
    marks_long_df.duplicated(
        subset=[
            "student_id",
            "year",
            "term",
            "subject",
        ],
        keep=False
    )
)


if subject_duplicate_mask.any():

    duplicates = marks_long_df.loc[
        subject_duplicate_mask,
        [
            "student_id",
            "year",
            "term",
            "subject",
            "mark",
            "exam_status",
        ]
    ]

    raise RuntimeError(
        "\nDuplicate student-term-subject "
        "records found:\n\n"
        f"{duplicates.to_string(index=False)}"
    )


marks_long_df[
    "subject_slug"
] = marks_long_df[
    "subject"
].apply(
    subject_slug
)


subject_mapping = (
    marks_long_df[
        [
            "subject",
            "subject_slug",
        ]
    ]
    .drop_duplicates()
    .sort_values(
        "subject"
    )
)


slug_duplicates = (
    subject_mapping[
        "subject_slug"
    ]
    .duplicated(
        keep=False
    )
)


if slug_duplicates.any():

    duplicates = subject_mapping.loc[
        slug_duplicates
    ]

    raise RuntimeError(
        "\nTwo subject names produce the same "
        "column slug:\n\n"
        f"{duplicates.to_string(index=False)}"
    )


subjects = (
    subject_mapping[
        "subject"
    ]
    .tolist()
)


print(
    f"\nSubjects detected: "
    f"{len(subjects)}"
)


for subject in subjects:

    print(
        f"  - {subject}"
    )


marks_long_df[
    "mark_numeric"
] = to_numeric(
    marks_long_df[
        "mark"
    ]
)


# A PRESENT exam record should contain a numeric mark.

present_without_mark = (
    marks_long_df[
        "exam_status"
    ].eq(
        "PRESENT"
    )
    &
    marks_long_df[
        "mark_numeric"
    ].isna()
)


if present_without_mark.any():

    bad = marks_long_df.loc[
        present_without_mark,
        [
            "student_id",
            "year",
            "term",
            "subject",
            "mark",
            "exam_status",
        ]
    ]

    raise RuntimeError(
        "\nPRESENT subject records without a "
        "numeric mark were found:\n\n"
        f"{bad.to_string(index=False)}"
    )


mark_pivot = (
    marks_long_df
    .pivot(
        index=[
            "student_id",
            "year",
            "term",
        ],
        columns="subject_slug",
        values="mark_numeric"
    )
)


mark_pivot.columns = [
    f"mark_{column}"
    for column in mark_pivot.columns
]


mark_pivot = (
    mark_pivot
    .reset_index()
)


status_pivot = (
    marks_long_df
    .pivot(
        index=[
            "student_id",
            "year",
            "term",
        ],
        columns="subject_slug",
        values="exam_status"
    )
)


status_pivot.columns = [
    f"status_{column}"
    for column in status_pivot.columns
]


status_pivot = (
    status_pivot
    .reset_index()
)


marks_term_keep = [
    "student_id",
    "year",
    "term",
    "grade",
    "class",
    "academic_stage",
    "numeric_subject_count",
    "absent_subject_count",
    "missing_subject_count",
    "not_taken_subject_count",
    "invalid_subject_count",
    "calculated_total",
    "calculated_average",
    "source_total",
    "source_average",
    "source_rank",
]


marks_term_keep = [
    column
    for column in marks_term_keep
    if column in marks_term_df.columns
]


marks_term_work = (
    marks_term_df[
        marks_term_keep
    ]
    .copy()
)


numeric_summary_columns = [
    "numeric_subject_count",
    "absent_subject_count",
    "missing_subject_count",
    "not_taken_subject_count",
    "invalid_subject_count",
    "calculated_total",
    "calculated_average",
    "source_total",
    "source_average",
    "source_rank",
]


for column in numeric_summary_columns:

    if column in marks_term_work.columns:

        marks_term_work[
            column
        ] = to_numeric(
            marks_term_work[
                column
            ]
        )


attendance_keep = [
    "student_id",
    "year",
    "term",
    "expected_term_school_days",
    "observed_school_days",
    "days_present",
    "days_absent",
    "attendance_pct",
    "attendance_coverage_pct",
    "months_observed",
    "months_capped",
    "months_unknown",
    "months_invalid",
    "incomplete_attendance",
    "attendance_data_quality",
]


attendance_keep = [
    column
    for column in attendance_keep
    if column in attendance_df.columns
]


attendance_work = (
    attendance_df[
        attendance_keep
    ]
    .copy()
)


attendance_numeric_columns = [
    "expected_term_school_days",
    "observed_school_days",
    "days_present",
    "days_absent",
    "attendance_pct",
    "attendance_coverage_pct",
    "months_observed",
    "months_capped",
    "months_unknown",
    "months_invalid",
]


for column in attendance_numeric_columns:

    if column in attendance_work.columns:

        attendance_work[
            column
        ] = to_numeric(
            attendance_work[
                column
            ]
        )


master_df = (
    student_term_df
    .merge(
        marks_term_work,
        on=[
            "student_id",
            "year",
            "term",
            "grade",
            "class",
        ],
        how="left",
        validate="one_to_one"
    )
)


master_df = (
    master_df
    .merge(
        mark_pivot,
        on=[
            "student_id",
            "year",
            "term",
        ],
        how="left",
        validate="one_to_one"
    )
)


master_df = (
    master_df
    .merge(
        status_pivot,
        on=[
            "student_id",
            "year",
            "term",
        ],
        how="left",
        validate="one_to_one"
    )
)


master_df = (
    master_df
    .merge(
        attendance_work,
        on=[
            "student_id",
            "year",
            "term",
        ],
        how="left",
        validate="one_to_one"
    )
)


master_df[
    "academic_period"
] = (
    master_df[
        "year"
    ].astype(str)
    +
    "_T"
    +
    master_df[
        "term"
    ].astype(str)
)


# Numeric chronological index used for sorting only.

master_df[
    "period_order"
] = (
    master_df[
        "year"
    ] * 10
    +
    master_df[
        "term"
    ]
)


expected_rows = len(
    student_term_df
)


if len(master_df) != expected_rows:

    raise RuntimeError(
        f"Master dataset row count changed.\n"
        f"Expected: {expected_rows}\n"
        f"Actual:   {len(master_df)}"
    )


assert_unique_student_term(
    master_df,
    "master_research_dataset"
)


# Every identity row must have marks summary data.

missing_marks_summary = int(
    master_df[
        "academic_stage"
    ].isna().sum()
)


if missing_marks_summary:

    raise RuntimeError(
        f"{missing_marks_summary} student-term rows "
        f"have no marks summary."
    )


attendance_available_count = int(
    master_df[
        "attendance_available"
    ]
    .astype(str)
    .str.lower()
    .eq(
        "true"
    )
    .sum()
)


attendance_present_count = int(
    master_df[
        "attendance_pct"
    ]
    .notna()
    .sum()
)


# Use attendance_data_quality, not attendance_pct, for the joined count.

attendance_joined_count = int(
    master_df[
        "attendance_data_quality"
    ]
    .notna()
    .sum()
)


if (
    attendance_joined_count
    !=
    len(attendance_df)
):

    raise RuntimeError(
        "\nAttendance merge count mismatch.\n"
        f"Expected linked attendance rows: "
        f"{len(attendance_df)}\n"
        f"Joined attendance rows: "
        f"{attendance_joined_count}"
    )


if (
    attendance_available_count
    !=
    attendance_joined_count
):

    raise RuntimeError(
        "\nAttendance availability flag does not "
        "match joined attendance records.\n"
        f"Flagged available: "
        f"{attendance_available_count}\n"
        f"Joined attendance: "
        f"{attendance_joined_count}"
    )


forbidden_columns = {
    "student_name",
    "student_no",
    "index_no",
    "canonical_index_no",
    "admission_no",
    "attendance_name",
    "attendance_index_no",
    "attendance_admission_no",
    "persistent_student_id",
    "marks_raw_identifier",
    "resolved_marks_identifier",
    "source_file",
    "source_sheet",
    "source_excel_row",
}


identifier_leaks = sorted(
    forbidden_columns.intersection(
        master_df.columns
    )
)


if identifier_leaks:

    raise RuntimeError(
        "\nPrivacy failure. Direct identifiers "
        "exist in the master dataset:\n"
        f"{identifier_leaks}"
    )


base_columns = [
    "student_id",
    "year",
    "term",
    "academic_period",
    "period_order",
    "grade",
    "class",
    "academic_stage",
    "attendance_linkage_status",
    "attendance_linkage_method",
    "attendance_available",
]


summary_columns = [
    column
    for column in [
        "numeric_subject_count",
        "absent_subject_count",
        "missing_subject_count",
        "not_taken_subject_count",
        "invalid_subject_count",
        "calculated_total",
        "calculated_average",
        "source_total",
        "source_average",
        "source_rank",
    ]
    if column in master_df.columns
]


attendance_columns = [
    column
    for column in [
        "expected_term_school_days",
        "observed_school_days",
        "days_present",
        "days_absent",
        "attendance_pct",
        "attendance_coverage_pct",
        "months_observed",
        "months_capped",
        "months_unknown",
        "months_invalid",
        "incomplete_attendance",
        "attendance_data_quality",
    ]
    if column in master_df.columns
]


mark_columns = sorted(
    [
        column
        for column in master_df.columns
        if column.startswith(
            "mark_"
        )
    ]
)


status_columns = sorted(
    [
        column
        for column in master_df.columns
        if column.startswith(
            "status_"
        )
    ]
)


ordered_columns = (
    base_columns
    +
    summary_columns
    +
    attendance_columns
    +
    mark_columns
    +
    status_columns
)


master_df = master_df[
    ordered_columns
]


master_df = (
    master_df
    .sort_values(
        by=[
            "period_order",
            "grade",
            "class",
            "student_id",
        ]
    )
    .reset_index(
        drop=True
    )
)


master_df.to_csv(
    OUTPUT_FILE,
    index=False
)


print(
    "\n"
    + "=" * 110
)

print(
    "MASTER RESEARCH DATASET COMPLETE"
)

print(
    "=" * 110
)


print(
    f"\nRows: "
    f"{len(master_df)}"
)


print(
    f"Unique students: "
    f"{master_df['student_id'].nunique()}"
)


print(
    f"Columns: "
    f"{len(master_df.columns)}"
)


print(
    f"Subjects represented: "
    f"{len(mark_columns)}"
)


print(
    f"Subject mark columns: "
    f"{len(mark_columns)}"
)


print(
    f"Subject status columns: "
    f"{len(status_columns)}"
)


print(
    f"\nAttendance linked: "
    f"{attendance_joined_count}"
)


print(
    f"Attendance not linked / unavailable: "
    f"{len(master_df) - attendance_joined_count}"
)


print(
    f"Rows with calculable attendance percentage: "
    f"{attendance_present_count}"
)


print(
    "\nAcademic periods:"
)


period_counts = (
    master_df[
        "academic_period"
    ]
    .value_counts()
    .sort_index()
)


print(
    period_counts.to_string()
)


print(
    "\nPrivacy validation:"
)

print(
    "  Direct student identifiers: 0"
)


print(
    "\nOutput:"
)

print(
    f"  {OUTPUT_FILE}"
)
