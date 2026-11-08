from pathlib import Path
import pandas as pd
import re


PROJECT_ROOT = Path(__file__).resolve().parents[2]

CLEANED_DIR = PROJECT_ROOT / "data" / "cleaned"
ANONYMIZED_DIR = PROJECT_ROOT / "data" / "anonymized"

ANONYMIZED_DIR.mkdir(
    parents=True,
    exist_ok=True
)


IDENTITY_MAP_FILE = (
    CLEANED_DIR /
    "student_identity_map.csv"
)

MARKS_LONG_FILE = (
    CLEANED_DIR /
    "marks_long.csv"
)

MARKS_TERM_FILE = (
    CLEANED_DIR /
    "marks_student_term_summary.csv"
)

ATTENDANCE_TERM_FILE = (
    CLEANED_DIR /
    "attendance_term.csv"
)


STUDENT_TERM_OUTPUT = (
    ANONYMIZED_DIR /
    "student_term_identity.csv"
)

MARKS_LONG_OUTPUT = (
    ANONYMIZED_DIR /
    "marks_long_anonymized.csv"
)

MARKS_TERM_OUTPUT = (
    ANONYMIZED_DIR /
    "marks_student_term_summary_anonymized.csv"
)

ATTENDANCE_TERM_OUTPUT = (
    ANONYMIZED_DIR /
    "attendance_term_anonymized.csv"
)


def clean_text(value):
    """Safely convert a value to stripped text."""

    if pd.isna(value):
        return ""

    return str(value).strip()


def to_int(value):
    """Safely convert numeric CSV values such as '1', '1.0', etc."""

    text = clean_text(value)

    if not text:
        return None

    try:
        return int(float(text))

    except (ValueError, TypeError):
        return None


def to_student_id(value):
    """Convert internal persistent IDs (PID0001) to research IDs (STU0001)."""

    text = clean_text(value).upper()

    match = re.fullmatch(
        r"PID(\d+)",
        text
    )

    if not match:
        raise ValueError(
            f"Invalid persistent student ID: {value}"
        )

    number = int(
        match.group(1)
    )

    return f"STU{number:04d}"


def require_columns(
    df,
    columns,
    label
):
    """Ensure an input file contains all columns needed by the anonymization stage."""

    missing = [
        column
        for column in columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"\nMissing columns in {label}:\n"
            f"{missing}\n\n"
            f"Available columns:\n"
            f"{list(df.columns)}"
        )


def marks_source_key(row):
    """Create a deterministic physical marks-row key."""

    return (
        clean_text(
            row.get("source_file")
        ),
        clean_text(
            row.get("source_sheet")
        ),
        to_int(
            row.get("source_excel_row")
        ),
        to_int(
            row.get("year")
        ),
        to_int(
            row.get("term")
        ),
        to_int(
            row.get("grade")
        ),
        clean_text(
            row.get("class")
        ).upper(),
    )


def validate_privacy(
    df,
    label
):
    """Ensure direct identifiers and source-location identifiers are not exported into anonymized research datasets."""

    forbidden_columns = {
        "student_name",
        "student_no",
        "index_no",
        "canonical_index_no",
        "admission_no",
        "identifier_value",
        "student_name_identifier",
        "attendance_name",
        "attendance_index_no",
        "attendance_canonical_index_no",
        "attendance_admission_no",
        "marks_raw_identifier",
        "resolved_marks_identifier",
        "persistent_student_id",

        # Source traceability stays in the private preprocessing environment, not the anonymized research datasets.
        "source_file",
        "source_sheet",
        "source_excel_row",
        "source_excel_column",
    }

    present = sorted(
        forbidden_columns.intersection(
            df.columns
        )
    )

    if present:
        raise RuntimeError(
            f"Privacy validation failed for "
            f"{label}.\n"
            f"Forbidden columns remain:\n"
            f"{present}"
        )


required_files = [
    IDENTITY_MAP_FILE,
    MARKS_LONG_FILE,
    MARKS_TERM_FILE,
    ATTENDANCE_TERM_FILE,
]


for required_file in required_files:

    if not required_file.exists():

        raise FileNotFoundError(
            f"\nRequired anonymization input file "
            f"not found:\n{required_file}"
        )


identity_df = pd.read_csv(
    IDENTITY_MAP_FILE,
    dtype=str
).astype(object)


marks_long_df = pd.read_csv(
    MARKS_LONG_FILE,
    dtype=str
).astype(object)


marks_term_df = pd.read_csv(
    MARKS_TERM_FILE,
    dtype=str
).astype(object)


attendance_df = pd.read_csv(
    ATTENDANCE_TERM_FILE,
    dtype=str
).astype(object)


print(
    "=" * 110
)

print(
    "STUDENT PERFORMANCE AI - "
    "RESEARCH DATA ANONYMIZATION"
)

print(
    "=" * 110
)


require_columns(
    identity_df,
    [
        "marks_row_id",
        "attendance_row_id",
        "year",
        "term",
        "grade",
        "class",
        "match_status",
        "match_method",
        "persistent_student_id",
    ],
    "student_identity_map.csv"
)


require_columns(
    marks_term_df,
    [
        "source_file",
        "source_sheet",
        "source_excel_row",
        "year",
        "term",
        "grade",
        "class",
        "academic_stage",
    ],
    "marks_student_term_summary.csv"
)


require_columns(
    marks_long_df,
    [
        "source_file",
        "source_sheet",
        "source_excel_row",
        "year",
        "term",
        "grade",
        "class",
        "academic_stage",
        "subject",
        "mark",
        "exam_status",
    ],
    "marks_long.csv"
)


require_columns(
    attendance_df,
    [
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
    "attendance_term.csv"
)


# CREATE ANONYMIZED STUDENT IDs

identity_df[
    "student_id"
] = identity_df[
    "persistent_student_id"
].apply(
    to_student_id
)


persistent_count = (
    identity_df[
        "persistent_student_id"
    ].nunique()
)

student_id_count = (
    identity_df[
        "student_id"
    ].nunique()
)


if persistent_count != student_id_count:

    raise RuntimeError(
        "PID -> STU mapping is not one-to-one."
    )


identity_term_duplicates = (
    identity_df.duplicated(
        subset=[
            "student_id",
            "year",
            "term",
        ],
        keep=False
    )
)


if identity_term_duplicates.any():

    duplicate_rows = identity_df.loc[
        identity_term_duplicates,
        [
            "student_id",
            "year",
            "term",
            "grade",
            "class",
        ]
    ]

    raise RuntimeError(
        "\nDuplicate student/year/term identities "
        "found after identity override processing.\n\n"
        f"{duplicate_rows.to_string(index=False)}"
    )


student_term_df = pd.DataFrame(
    {
        "student_id":
            identity_df[
                "student_id"
            ],

        "year":
            identity_df[
                "year"
            ],

        "term":
            identity_df[
                "term"
            ],

        "grade":
            identity_df[
                "grade"
            ],

        "class":
            identity_df[
                "class"
            ],

        "attendance_linkage_status":
            identity_df[
                "match_status"
            ],

        "attendance_linkage_method":
            identity_df[
                "match_method"
            ],

        "attendance_available":
            identity_df[
                "match_status"
            ].eq(
                "MATCHED"
            ),
    }
)


# Reconstruct the marks_row_id numbering used by the cross-year identity audit.

marks_term_work = marks_term_df.copy()


marks_term_work[
    "_marks_row_id"
] = range(
    1,
    len(marks_term_work) + 1
)


marks_lookup = {}


for _, row in identity_df.iterrows():

    marks_row_id = to_int(
        row["marks_row_id"]
    )

    student_id = clean_text(
        row["student_id"]
    )

    if marks_row_id is None:
        raise RuntimeError(
            "student_identity_map.csv contains "
            "a missing marks_row_id."
        )

    if marks_row_id in marks_lookup:
        raise RuntimeError(
            f"Duplicate marks_row_id found in "
            f"identity map: {marks_row_id}"
        )

    marks_lookup[
        marks_row_id
    ] = student_id


marks_term_work[
    "student_id"
] = marks_term_work[
    "_marks_row_id"
].map(
    marks_lookup
)


missing_marks_ids = int(
    marks_term_work[
        "student_id"
    ].isna().sum()
)


if missing_marks_ids:

    raise RuntimeError(
        f"{missing_marks_ids} marks "
        f"student-term rows have no student_id."
    )


marks_term_work[
    "_source_key"
] = marks_term_work.apply(
    marks_source_key,
    axis=1
)


duplicate_source_keys = (
    marks_term_work[
        "_source_key"
    ].duplicated(
        keep=False
    )
)


if duplicate_source_keys.any():

    duplicates = marks_term_work.loc[
        duplicate_source_keys,
        [
            "source_file",
            "source_sheet",
            "source_excel_row",
            "year",
            "term",
            "grade",
            "class",
        ]
    ]

    raise RuntimeError(
        "\nDuplicate physical marks rows found. "
        "Subject-level anonymization would be "
        "unsafe.\n\n"
        f"{duplicates.to_string(index=False)}"
    )


source_to_student = dict(
    zip(
        marks_term_work[
            "_source_key"
        ],
        marks_term_work[
            "student_id"
        ]
    )
)


marks_long_work = marks_long_df.copy()


marks_long_work[
    "_source_key"
] = marks_long_work.apply(
    marks_source_key,
    axis=1
)


marks_long_work[
    "student_id"
] = marks_long_work[
    "_source_key"
].map(
    source_to_student
)


missing_subject_ids = int(
    marks_long_work[
        "student_id"
    ].isna().sum()
)


if missing_subject_ids:

    raise RuntimeError(
        f"{missing_subject_ids} subject-mark "
        f"records have no student_id."
    )


marks_long_columns = [
    "student_id",
    "year",
    "term",
    "grade",
    "class",
    "academic_stage",
    "subject",
    "subject_group",
    "is_optional_subject",
    "mark",
    "exam_status",
    "correction_applied",
]


marks_long_anon = (
    marks_long_work[
        [
            column
            for column in marks_long_columns
            if column in marks_long_work.columns
        ]
    ]
    .copy()
)


marks_term_columns = [
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


marks_term_anon = (
    marks_term_work[
        [
            column
            for column in marks_term_columns
            if column in marks_term_work.columns
        ]
    ]
    .copy()
)


# Reconstruct the attendance row IDs from the cross-year identity audit.

attendance_work = attendance_df.copy()


attendance_work[
    "_attendance_row_id"
] = range(
    1,
    len(attendance_work) + 1
)


matched_identity = identity_df[
    identity_df[
        "match_status"
    ].eq(
        "MATCHED"
    )
].copy()


matched_identity[
    "_attendance_row_id"
] = matched_identity[
    "attendance_row_id"
].apply(
    to_int
)


missing_attendance_ids = int(
    matched_identity[
        "_attendance_row_id"
    ].isna().sum()
)


if missing_attendance_ids:

    raise RuntimeError(
        f"{missing_attendance_ids} matched "
        f"student-term records have no "
        f"attendance_row_id."
    )


duplicate_attendance_ids = (
    matched_identity[
        "_attendance_row_id"
    ].duplicated(
        keep=False
    )
)


if duplicate_attendance_ids.any():

    duplicates = matched_identity.loc[
        duplicate_attendance_ids,
        [
            "student_id",
            "year",
            "term",
            "grade",
            "class",
            "_attendance_row_id",
        ]
    ]

    raise RuntimeError(
        "\nAn attendance row is linked to "
        "multiple student-term records.\n\n"
        f"{duplicates.to_string(index=False)}"
    )


attendance_lookup = dict(
    zip(
        matched_identity[
            "_attendance_row_id"
        ],
        matched_identity[
            "student_id"
        ]
    )
)


attendance_work[
    "student_id"
] = attendance_work[
    "_attendance_row_id"
].map(
    attendance_lookup
)


# Keep only attendance records linked to the marks-based research population.
attendance_work = attendance_work[
    attendance_work[
        "student_id"
    ].notna()
].copy()


attendance_columns = [
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
    "months_observed",
    "months_capped",
    "months_unknown",
    "months_invalid",
    "incomplete_attendance",
    "attendance_data_quality",
]


attendance_anon = (
    attendance_work[
        [
            column
            for column in attendance_columns
            if column in attendance_work.columns
        ]
    ]
    .copy()
)


# PRIVACY VALIDATION

datasets_to_validate = [
    (
        "student_term_identity",
        student_term_df
    ),
    (
        "marks_long_anonymized",
        marks_long_anon
    ),
    (
        "marks_student_term_summary_anonymized",
        marks_term_anon
    ),
    (
        "attendance_term_anonymized",
        attendance_anon
    ),
]


for label, frame in datasets_to_validate:

    validate_privacy(
        frame,
        label
    )


if len(student_term_df) != len(identity_df):

    raise RuntimeError(
        "Student-term row count changed during "
        "anonymization."
    )


if len(marks_term_anon) != len(identity_df):

    raise RuntimeError(
        "Marks student-term row count changed "
        "during anonymization."
    )


if (
    marks_term_anon[
        "student_id"
    ].nunique()
    !=
    identity_df[
        "student_id"
    ].nunique()
):

    raise RuntimeError(
        "Unique student count changed while "
        "anonymizing marks."
    )


student_term_df = (
    student_term_df
    .sort_values(
        by=[
            "student_id",
            "year",
            "term",
        ]
    )
    .reset_index(
        drop=True
    )
)


marks_long_anon = (
    marks_long_anon
    .sort_values(
        by=[
            "student_id",
            "year",
            "term",
            "subject",
        ]
    )
    .reset_index(
        drop=True
    )
)


marks_term_anon = (
    marks_term_anon
    .sort_values(
        by=[
            "student_id",
            "year",
            "term",
        ]
    )
    .reset_index(
        drop=True
    )
)


attendance_anon = (
    attendance_anon
    .sort_values(
        by=[
            "student_id",
            "year",
            "term",
        ]
    )
    .reset_index(
        drop=True
    )
)


student_term_df.to_csv(
    STUDENT_TERM_OUTPUT,
    index=False
)


marks_long_anon.to_csv(
    MARKS_LONG_OUTPUT,
    index=False
)


marks_term_anon.to_csv(
    MARKS_TERM_OUTPUT,
    index=False
)


attendance_anon.to_csv(
    ATTENDANCE_TERM_OUTPUT,
    index=False
)


print(
    "\n"
    + "=" * 110
)

print(
    "ANONYMIZATION COMPLETE"
)

print(
    "=" * 110
)


print(
    f"\nUnique anonymized students: "
    f"{student_term_df['student_id'].nunique()}"
)


print(
    f"Student-term identity rows: "
    f"{len(student_term_df)}"
)


print(
    f"Anonymized marks subject rows: "
    f"{len(marks_long_anon)}"
)


print(
    f"Anonymized marks student-term rows: "
    f"{len(marks_term_anon)}"
)


print(
    f"Anonymized linked attendance-term rows: "
    f"{len(attendance_anon)}"
)


print(
    "\nPrivacy validation:"
)


print(
    "  Direct student identifiers in "
    "anonymized outputs: 0"
)


print(
    "  Names removed: YES"
)


print(
    "  Admission numbers removed: YES"
)


print(
    "  Index numbers removed: YES"
)


print(
    "  Source Excel locations removed: YES"
)


print(
    "\nOutputs:"
)


print(
    f"  {STUDENT_TERM_OUTPUT}"
)


print(
    f"  {MARKS_LONG_OUTPUT}"
)


print(
    f"  {MARKS_TERM_OUTPUT}"
)


print(
    f"  {ATTENDANCE_TERM_OUTPUT}"
)
