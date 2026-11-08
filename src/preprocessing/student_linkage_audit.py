from pathlib import Path
from collections import defaultdict
import pandas as pd
import re


PROJECT_ROOT = Path(__file__).resolve().parents[2]

CLEANED_DATA_DIR = PROJECT_ROOT / "data" / "cleaned"
REPORTS_DIR = PROJECT_ROOT / "reports"

REPORTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


MARKS_FILE = (
    CLEANED_DATA_DIR
    / "marks_student_term_summary.csv"
)

ATTENDANCE_FILE = (
    CLEANED_DATA_DIR
    / "attendance_term.csv"
)


LINKAGE_OUTPUT = (
    REPORTS_DIR
    / "student_linkage_audit.csv"
)

UNMATCHED_MARKS_OUTPUT = (
    REPORTS_DIR
    / "student_linkage_unmatched_marks.csv"
)

UNMATCHED_ATTENDANCE_OUTPUT = (
    REPORTS_DIR
    / "student_linkage_unmatched_attendance.csv"
)

AMBIGUOUS_OUTPUT = (
    REPORTS_DIR
    / "student_linkage_ambiguous.csv"
)

AMBIGUOUS_CANDIDATES_OUTPUT = (
    REPORTS_DIR
    / "student_linkage_ambiguous_candidates.csv"
)

EXCLUDED_NON_STUDENTS_OUTPUT = (
    REPORTS_DIR
    / "student_linkage_excluded_non_students.csv"
)


LEGACY_DIAGNOSTIC_OUTPUTS = [
    UNMATCHED_MARKS_OUTPUT,
    UNMATCHED_ATTENDANCE_OUTPUT,
    AMBIGUOUS_OUTPUT,
    AMBIGUOUS_CANDIDATES_OUTPUT,
    EXCLUDED_NON_STUDENTS_OUTPUT,
]


KNOWN_UNAVAILABLE_ATTENDANCE = {
    (2026, 9, "A"),
}


KNOWN_NON_STUDENT_NAMES = {
    "a b c",
    "percentage",
    "s",
    "above 40",
}


def clean_text(value):

    if pd.isna(value):
        return ""

    return str(value).strip()


def normalize_name(value):

    text = clean_text(
        value
    ).lower()

    if not text:
        return ""

    text = re.sub(
        r"[^a-z0-9]+",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


def normalize_identifier(value):

    text = clean_text(
        value
    )

    if not text:
        return ""

    if re.fullmatch(
        r"\d+\.0",
        text
    ):
        text = text[:-2]

    text = re.sub(
        r"\s+",
        "",
        text
    )

    if text.isdigit():

        text = (
            text.lstrip("0")
            or "0"
        )

    return text.upper()


def canonical_index(value):
    """Extract the final numeric component of a full Index No."""

    text = normalize_identifier(
        value
    )

    if not text:
        return ""

    parts = re.findall(
        r"\d+",
        text
    )

    if not parts:
        return ""

    result = (
        parts[-1].lstrip("0")
        or "0"
    )

    return result


def normalize_year(value):

    try:

        return int(
            float(value)
        )

    except (
        ValueError,
        TypeError
    ):

        return None


def normalize_term(value):

    text = clean_text(
        value
    )

    match = re.search(
        r"[123]",
        text
    )

    if not match:
        return None

    return int(
        match.group(0)
    )


def normalize_grade(value):

    text = clean_text(
        value
    )

    match = re.search(
        r"\d+",
        text
    )

    if not match:
        return None

    return int(
        match.group(0)
    )


def normalize_class(value):

    text = clean_text(
        value
    ).upper()

    if not text:
        return ""

    match = re.fullmatch(
        r"\d+\s*([A-Z]+)",
        text
    )

    if match:
        return match.group(1)

    return text


def scope_key(
    year,
    term,
    grade,
    class_name,
):

    return (
        year,
        term,
        grade,
        class_name,
    )


if not MARKS_FILE.exists():

    raise FileNotFoundError(
        f"Marks file not found:\n"
        f"{MARKS_FILE}"
    )


if not ATTENDANCE_FILE.exists():

    raise FileNotFoundError(
        f"Attendance file not found:\n"
        f"{ATTENDANCE_FILE}"
    )


marks_df = pd.read_csv(
    MARKS_FILE,
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
    "MARKS ↔ ATTENDANCE LINKAGE AUDIT"
)

print(
    "=" * 110
)


required_marks_columns = [
    "year",
    "term",
    "grade",
    "class",
    "index_no",
    "student_name",
]


required_attendance_columns = [
    "source_excel_row",
    "year",
    "term",
    "grade",
    "class",
    "index_no",
    "canonical_index_no",
    "admission_no",
    "student_name_identifier",
]


for column in required_marks_columns:

    if column not in marks_df.columns:

        raise ValueError(
            f"Missing marks column: "
            f"{column}\n"
            f"Available columns:\n"
            f"{list(marks_df.columns)}"
        )


for column in required_attendance_columns:

    if column not in attendance_df.columns:

        raise ValueError(
            f"Missing attendance column: "
            f"{column}\n"
            f"Available columns:\n"
            f"{list(attendance_df.columns)}"
        )


marks_df = marks_df.copy()


marks_df[
    "_normalized_student_name"
] = (
    marks_df[
        "student_name"
    ]
    .apply(
        normalize_name
    )
)


non_student_mask = (
    marks_df[
        "_normalized_student_name"
    ]
    .isin(
        KNOWN_NON_STUDENT_NAMES
    )
)


excluded_non_students_df = (
    marks_df[
        non_student_mask
    ]
    .copy()
)


marks_df = (
    marks_df[
        ~non_student_mask
    ]
    .copy()
)


print(
    f"\nOriginal marks student-term rows: "
    f"{len(marks_df) + len(excluded_non_students_df)}"
)

print(
    f"Non-student rows excluded: "
    f"{len(excluded_non_students_df)}"
)


if not excluded_non_students_df.empty:

    print(
        "\nExcluded row labels:"
    )

    print(
        excluded_non_students_df[
            "student_name"
        ]
        .value_counts(
            dropna=False
        )
        .to_string()
    )


print(
    f"\nMarks student-term records "
    f"after exclusion: "
    f"{len(marks_df)}"
)

print(
    f"Attendance term records: "
    f"{len(attendance_df)}"
)


marks_df[
    "_marks_row_id"
] = range(
    1,
    len(marks_df) + 1
)


marks_df[
    "_year"
] = (
    marks_df[
        "year"
    ]
    .apply(
        normalize_year
    )
)


marks_df[
    "_term"
] = (
    marks_df[
        "term"
    ]
    .apply(
        normalize_term
    )
)


marks_df[
    "_grade"
] = (
    marks_df[
        "grade"
    ]
    .apply(
        normalize_grade
    )
)


marks_df[
    "_class"
] = (
    marks_df[
        "class"
    ]
    .apply(
        normalize_class
    )
)


marks_df[
    "_name"
] = (
    marks_df[
        "student_name"
    ]
    .apply(
        normalize_name
    )
)


# index_no is either the Admission No. or the trailing part of an older Index No.
marks_df[
    "_marks_identifier"
] = (
    marks_df[
        "index_no"
    ]
    .apply(
        normalize_identifier
    )
)


marks_df[
    "_identifier_source"
] = marks_df[
    "_marks_identifier"
].apply(
    lambda value:
        (
            "ORIGINAL_MARKS_IDENTIFIER"
            if value
            else
            "NO_IDENTIFIER_AVAILABLE"
        )
)


attendance_df = (
    attendance_df.copy()
)


attendance_df[
    "_attendance_row_id"
] = range(
    1,
    len(attendance_df) + 1
)


attendance_df[
    "_year"
] = (
    attendance_df[
        "year"
    ]
    .apply(
        normalize_year
    )
)


attendance_df[
    "_term"
] = (
    attendance_df[
        "term"
    ]
    .apply(
        normalize_term
    )
)


attendance_df[
    "_grade"
] = (
    attendance_df[
        "grade"
    ]
    .apply(
        normalize_grade
    )
)


attendance_df[
    "_class"
] = (
    attendance_df[
        "class"
    ]
    .apply(
        normalize_class
    )
)


attendance_df[
    "_admission"
] = (
    attendance_df[
        "admission_no"
    ]
    .apply(
        normalize_identifier
    )
)


attendance_df[
    "_full_index"
] = (
    attendance_df[
        "index_no"
    ]
    .apply(
        normalize_identifier
    )
)


attendance_df[
    "_canonical_index"
] = (
    attendance_df[
        "index_no"
    ]
    .apply(
        canonical_index
    )
)


attendance_df[
    "_name"
] = (
    attendance_df[
        "student_name_identifier"
    ]
    .apply(
        normalize_name
    )
)


admission_lookup = defaultdict(
    list
)

index_fallback_lookup = defaultdict(
    list
)

name_lookup = defaultdict(
    list
)


for _, row in attendance_df.iterrows():

    base = scope_key(
        row["_year"],
        row["_term"],
        row["_grade"],
        row["_class"],
    )

    attendance_row_id = (
        row[
            "_attendance_row_id"
        ]
    )

    # Primary identifier

    if row["_admission"]:

        admission_lookup[
            base
            +
            (
                row["_admission"],
            )
        ].append(
            attendance_row_id
        )


    # Use canonical Index only when Admission No. is absent.

    if (
        row["_canonical_index"]
        and
        not row["_admission"]
    ):

        index_fallback_lookup[
            base
            +
            (
                row[
                    "_canonical_index"
                ],
            )
        ].append(
            attendance_row_id
        )


    # Only attendance rows that actually contain a name.

    if row["_name"]:

        name_lookup[
            base
            +
            (
                row["_name"],
            )
        ].append(
            attendance_row_id
        )


linkage_records = []


for _, mark in marks_df.iterrows():

    year = mark["_year"]
    term = mark["_term"]
    grade = mark["_grade"]
    class_name = mark["_class"]

    student_name = clean_text(
        mark[
            "student_name"
        ]
    )

    normalized_name = (
        mark[
            "_name"
        ]
    )

    identifier = (
        mark[
            "_marks_identifier"
        ]
    )

    identifier_source = (
        mark[
            "_identifier_source"
        ]
    )

    base = scope_key(
        year,
        term,
        grade,
        class_name,
    )

    attendance_row_id = None

    match_status = None
    match_method = None

    candidate_count = 0

    notes = None


    if (
        year,
        grade,
        class_name
    ) in KNOWN_UNAVAILABLE_ATTENDANCE:

        match_status = (
            "ATTENDANCE_UNAVAILABLE"
        )

        match_method = (
            "NOT_APPLICABLE"
        )

        notes = (
            "Historical attendance records "
            "were unavailable for this class."
        )


    else:


        if identifier:

            admission_candidates = (
                admission_lookup.get(
                    base
                    +
                    (
                        identifier,
                    ),
                    []
                )
            )

            candidate_count = len(
                admission_candidates
            )

            if (
                len(
                    admission_candidates
                ) == 1
            ):

                attendance_row_id = (
                    admission_candidates[
                        0
                    ]
                )

                match_status = (
                    "MATCHED"
                )

                match_method = (
                    "EXACT_ADMISSION_NO"
                )

            elif (
                len(
                    admission_candidates
                ) > 1
            ):

                match_status = (
                    "AMBIGUOUS"
                )

                match_method = (
                    "AMBIGUOUS_ADMISSION_NO"
                )

                notes = (
                    "Multiple attendance rows "
                    "have the same Admission No. "
                    "within this year/term/grade/class."
                )


        # Only attendance rows without Admission No.

        if (
            match_status is None
            and
            identifier
        ):

            index_candidates = (
                index_fallback_lookup.get(
                    base
                    +
                    (
                        canonical_index(
                            identifier
                        ),
                    ),
                    []
                )
            )

            candidate_count = len(
                index_candidates
            )

            if (
                len(
                    index_candidates
                ) == 1
            ):

                attendance_row_id = (
                    index_candidates[
                        0
                    ]
                )

                match_status = (
                    "MATCHED"
                )

                match_method = (
                    "CANONICAL_INDEX_FALLBACK"
                )

            elif (
                len(
                    index_candidates
                ) > 1
            ):

                match_status = (
                    "AMBIGUOUS"
                )

                match_method = (
                    "AMBIGUOUS_CANONICAL_INDEX"
                )

                notes = (
                    "Multiple attendance rows "
                    "without Admission No. share "
                    "the same canonical Index No."
                )


        # Same year, term, grade and class only; no cross-year name inference.

        if (
            match_status is None
            and
            normalized_name
        ):

            name_candidates = (
                name_lookup.get(
                    base
                    +
                    (
                        normalized_name,
                    ),
                    []
                )
            )

            candidate_count = len(
                name_candidates
            )

            if (
                len(
                    name_candidates
                ) == 1
            ):

                attendance_row_id = (
                    name_candidates[
                        0
                    ]
                )

                match_status = (
                    "MATCHED"
                )

                match_method = (
                    "EXACT_NAME_SAME_SCOPE"
                )

            elif (
                len(
                    name_candidates
                ) > 1
            ):

                match_status = (
                    "AMBIGUOUS"
                )

                match_method = (
                    "AMBIGUOUS_NAME"
                )

                notes = (
                    "Multiple attendance rows "
                    "in the same year/term/grade/class "
                    "contain the same normalized name."
                )


        if match_status is None:

            match_status = (
                "UNMATCHED"
            )

            if identifier:

                match_method = (
                    "IDENTIFIER_NOT_FOUND"
                )

                notes = (
                    "Marks identifier did not match "
                    "an Admission No., safe Index fallback, "
                    "or exact attendance name "
                    "within the same scope."
                )

            else:

                match_method = (
                    "NO_IDENTIFIER_OR_NAME_MATCH"
                )

                notes = (
                    "Marks row has no identifier and "
                    "there was no exact attendance name "
                    "in the same year/term/grade/class."
                )


    attendance_source_excel_row = None
    attendance_index = None
    attendance_canonical_index = None
    attendance_admission = None
    attendance_name = None


    if attendance_row_id is not None:

        result = attendance_df[
            attendance_df[
                "_attendance_row_id"
            ]
            ==
            attendance_row_id
        ]

        if len(result) == 1:

            attendance_row = (
                result.iloc[0]
            )

            attendance_source_excel_row = (
                attendance_row[
                    "source_excel_row"
                ]
            )

            attendance_index = (
                attendance_row[
                    "_full_index"
                ]
            )

            attendance_canonical_index = (
                attendance_row[
                    "_canonical_index"
                ]
            )

            attendance_admission = (
                attendance_row[
                    "_admission"
                ]
            )

            attendance_name = clean_text(
                attendance_row[
                    "student_name_identifier"
                ]
            )


    linkage_records.append(
        {
            "marks_row_id":
                mark[
                    "_marks_row_id"
                ],

            "attendance_row_id":
                attendance_row_id,

            "attendance_source_excel_row":
                attendance_source_excel_row,

            "year":
                year,

            "term":
                term,

            "grade":
                grade,

            "class":
                class_name,

            "student_name":
                student_name,

            "marks_raw_identifier":
                clean_text(
                    mark[
                        "index_no"
                    ]
                ),

            "resolved_marks_identifier":
                identifier,

            "identifier_resolution_source":
                identifier_source,

            "attendance_index_no":
                attendance_index,

            "attendance_canonical_index_no":
                attendance_canonical_index,

            "attendance_admission_no":
                attendance_admission,

            "attendance_name":
                attendance_name,

            "match_status":
                match_status,

            "match_method":
                match_method,

            "candidate_count":
                candidate_count,

            "notes":
                notes,
        }
    )


linkage_df = pd.DataFrame(
    linkage_records
)


matched_linkage = (
    linkage_df[
        linkage_df[
            "attendance_row_id"
        ].notna()
        &
        (
            linkage_df[
                "match_status"
            ]
            ==
            "MATCHED"
        )
    ]
)


duplicate_attendance_counts = (
    matched_linkage[
        "attendance_row_id"
    ]
    .value_counts()
)


duplicate_attendance_ids = set(
    duplicate_attendance_counts[
        duplicate_attendance_counts
        > 1
    ].index
)


if duplicate_attendance_ids:

    duplicate_mask = (
        linkage_df[
            "attendance_row_id"
        ]
        .isin(
            duplicate_attendance_ids
        )
    )

    linkage_df.loc[
        duplicate_mask,
        "match_status"
    ] = (
        "AMBIGUOUS"
    )

    linkage_df.loc[
        duplicate_mask,
        "match_method"
    ] = (
        "ATTENDANCE_ROW_MATCHED_MULTIPLE_MARKS"
    )

    linkage_df.loc[
        duplicate_mask,
        "notes"
    ] = (
        "The same physical attendance term row "
        "matched more than one marks student."
    )


ambiguous_candidate_records = []


for _, link in linkage_df[
    linkage_df[
        "match_status"
    ]
    ==
    "AMBIGUOUS"
].iterrows():

    base = scope_key(
        link[
            "year"
        ],
        link[
            "term"
        ],
        link[
            "grade"
        ],
        link[
            "class"
        ],
    )

    identifier = normalize_identifier(
        link[
            "resolved_marks_identifier"
        ]
    )

    student_name_norm = (
        normalize_name(
            link[
                "student_name"
            ]
        )
    )

    candidate_ids = set()


    if (
        link[
            "match_method"
        ]
        ==
        "AMBIGUOUS_ADMISSION_NO"
    ):

        candidate_ids.update(
            admission_lookup.get(
                base
                +
                (
                    identifier,
                ),
                []
            )
        )


    elif (
        link[
            "match_method"
        ]
        ==
        "AMBIGUOUS_CANONICAL_INDEX"
    ):

        candidate_ids.update(
            index_fallback_lookup.get(
                base
                +
                (
                    canonical_index(
                        identifier
                    ),
                ),
                []
            )
        )


    elif (
        link[
            "match_method"
        ]
        ==
        "AMBIGUOUS_NAME"
    ):

        candidate_ids.update(
            name_lookup.get(
                base
                +
                (
                    student_name_norm,
                ),
                []
            )
        )


    elif (
        link[
            "match_method"
        ]
        ==
        "ATTENDANCE_ROW_MATCHED_MULTIPLE_MARKS"
    ):

        if pd.notna(
            link[
                "attendance_row_id"
            ]
        ):

            candidate_ids.add(
                int(
                    float(
                        link[
                            "attendance_row_id"
                        ]
                    )
                )
            )


    for candidate_id in sorted(
        candidate_ids
    ):

        candidate = attendance_df[
            attendance_df[
                "_attendance_row_id"
            ]
            ==
            candidate_id
        ]

        if candidate.empty:
            continue

        candidate = (
            candidate.iloc[0]
        )

        ambiguous_candidate_records.append(
            {
                "marks_row_id":
                    link[
                        "marks_row_id"
                    ],

                "student_name":
                    link[
                        "student_name"
                    ],

                "year":
                    link[
                        "year"
                    ],

                "term":
                    link[
                        "term"
                    ],

                "grade":
                    link[
                        "grade"
                    ],

                "class":
                    link[
                        "class"
                    ],

                "marks_identifier":
                    link[
                        "resolved_marks_identifier"
                    ],

                "ambiguity_type":
                    link[
                        "match_method"
                    ],

                "candidate_attendance_row_id":
                    candidate_id,

                "candidate_source_excel_row":
                    candidate[
                        "source_excel_row"
                    ],

                "candidate_index_no":
                    clean_text(
                        candidate[
                            "index_no"
                        ]
                    ),

                "candidate_admission_no":
                    clean_text(
                        candidate[
                            "admission_no"
                        ]
                    ),

                "candidate_name_identifier":
                    clean_text(
                        candidate[
                            "student_name_identifier"
                        ]
                    ),

                "candidate_attendance_pct":
                    candidate.get(
                        "attendance_pct",
                        None
                    ),

                "candidate_data_quality":
                    candidate.get(
                        "attendance_data_quality",
                        None
                    ),
            }
        )


ambiguous_candidates_df = pd.DataFrame(
    ambiguous_candidate_records
)


if ambiguous_candidates_df.empty:

    ambiguous_candidates_df = pd.DataFrame(
        columns=[
            "marks_row_id",
            "student_name",
            "year",
            "term",
            "grade",
            "class",
            "marks_identifier",
            "ambiguity_type",
            "candidate_attendance_row_id",
            "candidate_source_excel_row",
            "candidate_index_no",
            "candidate_admission_no",
            "candidate_name_identifier",
            "candidate_attendance_pct",
            "candidate_data_quality",
        ]
    )


unmatched_marks_df = (
    linkage_df[
        linkage_df[
            "match_status"
        ]
        ==
        "UNMATCHED"
    ]
    .copy()
)


ambiguous_df = (
    linkage_df[
        linkage_df[
            "match_status"
        ]
        ==
        "AMBIGUOUS"
    ]
    .copy()
)


matched_attendance_ids = set(
    linkage_df.loc[
        linkage_df[
            "match_status"
        ]
        ==
        "MATCHED",
        "attendance_row_id"
    ]
    .dropna()
    .astype(int)
    .tolist()
)


unmatched_attendance_df = (
    attendance_df[
        ~attendance_df[
            "_attendance_row_id"
        ]
        .isin(
            matched_attendance_ids
        )
    ]
    .copy()
)


unmatched_attendance_columns = [
    "_attendance_row_id",
    "source_file",
    "source_sheet",
    "source_excel_row",
    "_year",
    "_term",
    "_grade",
    "_class",
    "index_no",
    "canonical_index_no",
    "admission_no",
    "student_name_identifier",
]


for optional_column in [
    "attendance_pct",
    "attendance_coverage_pct",
    "attendance_data_quality",
]:

    if (
        optional_column
        in unmatched_attendance_df.columns
    ):

        unmatched_attendance_columns.append(
            optional_column
        )


unmatched_attendance_df = (
    unmatched_attendance_df[
        unmatched_attendance_columns
    ]
    .rename(
        columns={
            "_attendance_row_id":
                "attendance_row_id",

            "_year":
                "year",

            "_term":
                "term",

            "_grade":
                "grade",

            "_class":
                "class",
        }
    )
)


linkage_df = linkage_df.sort_values(
    by=[
        "year",
        "term",
        "grade",
        "class",
        "student_name",
    ]
)


if not unmatched_marks_df.empty:

    unmatched_marks_df = (
        unmatched_marks_df.sort_values(
            by=[
                "year",
                "term",
                "grade",
                "class",
                "student_name",
            ]
        )
    )


if not ambiguous_df.empty:

    ambiguous_df = (
        ambiguous_df.sort_values(
            by=[
                "year",
                "term",
                "grade",
                "class",
                "student_name",
            ]
        )
    )


if not unmatched_attendance_df.empty:

    unmatched_attendance_df = (
        unmatched_attendance_df.sort_values(
            by=[
                "year",
                "term",
                "grade",
                "class",
                "source_excel_row",
            ]
        )
    )


if not ambiguous_candidates_df.empty:

    ambiguous_candidates_df = (
        ambiguous_candidates_df.sort_values(
            by=[
                "year",
                "term",
                "grade",
                "class",
                "student_name",
                "candidate_source_excel_row",
            ]
        )
    )


linkage_df.to_csv(
    LINKAGE_OUTPUT,
    index=False
)


# Unmatched/ambiguous tables stay in memory for validation and console output.
for legacy_output in LEGACY_DIAGNOSTIC_OUTPUTS:
    if legacy_output.exists():
        legacy_output.unlink()


print(
    "\n"
    +
    "=" * 110
)

print(
    "STUDENT LINKAGE AUDIT COMPLETE"
)

print(
    "=" * 110
)


print(
    f"\nMarks student records audited: "
    f"{len(linkage_df)}"
)


print(
    "\nLinkage status:"
)

print(
    linkage_df[
        "match_status"
    ]
    .value_counts(
        dropna=False
    )
    .to_string()
)


print(
    "\nMatch methods:"
)

print(
    linkage_df[
        "match_method"
    ]
    .value_counts(
        dropna=False
    )
    .to_string()
)


print(
    "\nIdentifier source:"
)

print(
    linkage_df[
        "identifier_resolution_source"
    ]
    .value_counts(
        dropna=False
    )
    .to_string()
)


matched_count = int(
    (
        linkage_df[
            "match_status"
        ]
        ==
        "MATCHED"
    ).sum()
)


unavailable_count = int(
    (
        linkage_df[
            "match_status"
        ]
        ==
        "ATTENDANCE_UNAVAILABLE"
    ).sum()
)


matchable_count = (
    len(linkage_df)
    -
    unavailable_count
)


match_rate = (
    matched_count
    /
    matchable_count
    *
    100
    if matchable_count
    else 0
)


print(
    f"\nMatched: "
    f"{matched_count}"
)


print(
    f"Attendance unavailable: "
    f"{unavailable_count}"
)


print(
    f"Matchable: "
    f"{matchable_count}"
)


print(
    f"Automatic linkage rate: "
    f"{match_rate:.2f}%"
)


print(
    f"\nUnmatched marks: "
    f"{len(unmatched_marks_df)}"
)


print(
    f"Ambiguous: "
    f"{len(ambiguous_df)}"
)


print(
    f"Unmatched attendance: "
    f"{len(unmatched_attendance_df)}"
)


print(
    f"Ambiguous candidate rows: "
    f"{len(ambiguous_candidates_df)}"
)


print(
    "\nOutputs:"
)


print(
    f"  {LINKAGE_OUTPUT}"
)
