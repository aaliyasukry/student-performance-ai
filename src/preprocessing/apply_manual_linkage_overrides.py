from pathlib import Path
import pandas as pd
import re


PROJECT_ROOT = Path(__file__).resolve().parents[2]

CLEANED_DIR = PROJECT_ROOT / "data" / "cleaned"
REPORTS_DIR = PROJECT_ROOT / "reports"
CONFIG_DIR = PROJECT_ROOT / "config"

REPORTS_DIR.mkdir(parents=True, exist_ok=True)
CONFIG_DIR.mkdir(parents=True, exist_ok=True)


AUTO_LINKAGE_FILE = (
    REPORTS_DIR /
    "student_linkage_audit.csv"
)

ATTENDANCE_FILE = (
    CLEANED_DIR /
    "attendance_term.csv"
)

OVERRIDES_FILE = (
    CONFIG_DIR /
    "student_linkage_manual_overrides.csv"
)


FINAL_OUTPUT = (
    REPORTS_DIR /
    "student_linkage_final.csv"
)

OVERRIDE_LOG_OUTPUT = (
    REPORTS_DIR /
    "student_linkage_manual_override_log.csv"
)

FINAL_UNMATCHED_OUTPUT = (
    REPORTS_DIR /
    "student_linkage_final_unmatched_marks.csv"
)

FINAL_AMBIGUOUS_OUTPUT = (
    REPORTS_DIR /
    "student_linkage_final_ambiguous.csv"
)

FINAL_UNMATCHED_ATTENDANCE_OUTPUT = (
    REPORTS_DIR /
    "student_linkage_final_unmatched_attendance.csv"
)


LEGACY_DIAGNOSTIC_OUTPUTS = [
    FINAL_UNMATCHED_OUTPUT,
    FINAL_AMBIGUOUS_OUTPUT,
    FINAL_UNMATCHED_ATTENDANCE_OUTPUT,
]


def clean_text(value):

    if pd.isna(value):
        return ""

    return str(value).strip()


def normalize_name(value):

    text = clean_text(value).lower()

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

    text = clean_text(value)

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
        return (
            text.lstrip("0")
            or "0"
        )

    return text.upper()


def canonical_index(value):

    text = normalize_identifier(
        value
    )

    if not text:
        return ""

    numbers = re.findall(
        r"\d+",
        text
    )

    if not numbers:
        return ""

    return (
        numbers[-1].lstrip("0")
        or "0"
    )


def to_int(value):

    if pd.isna(value):
        return None

    text = clean_text(value)

    if not text:
        return None

    try:
        return int(float(text))

    except (
        ValueError,
        TypeError
    ):
        return None


def normalize_class(value):

    return clean_text(
        value
    ).upper()


for required_file in [
    AUTO_LINKAGE_FILE,
    ATTENDANCE_FILE,
    OVERRIDES_FILE,
]:

    if not required_file.exists():

        raise FileNotFoundError(
            f"Required file not found:\n"
            f"{required_file}"
        )


linkage_df = pd.read_csv(
    AUTO_LINKAGE_FILE,
    dtype=str
)

attendance_df = pd.read_csv(
    ATTENDANCE_FILE,
    dtype=str
)

overrides_df = pd.read_csv(
    OVERRIDES_FILE,
    dtype=str
)


print("=" * 110)
print(
    "STUDENT PERFORMANCE AI - "
    "MANUAL LINKAGE OVERRIDES"
)
print("=" * 110)


attendance_df = attendance_df.copy()

attendance_df[
    "_attendance_row_id"
] = range(
    1,
    len(attendance_df) + 1
)


attendance_df[
    "_year"
] = attendance_df[
    "year"
].apply(to_int)

attendance_df[
    "_term"
] = attendance_df[
    "term"
].apply(to_int)

attendance_df[
    "_grade"
] = attendance_df[
    "grade"
].apply(to_int)

attendance_df[
    "_class"
] = attendance_df[
    "class"
].apply(normalize_class)

attendance_df[
    "_admission"
] = attendance_df[
    "admission_no"
].apply(normalize_identifier)

attendance_df[
    "_index"
] = attendance_df[
    "index_no"
].apply(normalize_identifier)

attendance_df[
    "_canonical_index"
] = attendance_df[
    "index_no"
].apply(canonical_index)

attendance_df[
    "_name"
] = attendance_df[
    "student_name_identifier"
].apply(normalize_name)

attendance_df[
    "_source_row"
] = attendance_df[
    "source_excel_row"
].apply(to_int)


final_df = linkage_df.copy()

# Loaded with dtype=str; cast so manual overrides can write ints, bools and strings.
final_df = final_df.astype(object)

final_df[
    "pre_override_match_status"
] = final_df[
    "match_status"
]

final_df[
    "pre_override_match_method"
] = final_df[
    "match_method"
]

final_df[
    "manual_override"
] = False

final_df[
    "manual_override_field"
] = ""

final_df[
    "manual_override_value"
] = ""

final_df[
    "manual_override_reason"
] = ""


final_df[
    "_year"
] = final_df[
    "year"
].apply(to_int)

final_df[
    "_term"
] = final_df[
    "term"
].apply(to_int)

final_df[
    "_grade"
] = final_df[
    "grade"
].apply(to_int)

final_df[
    "_class"
] = final_df[
    "class"
].apply(normalize_class)

final_df[
    "_student_name_norm"
] = final_df[
    "student_name"
].apply(normalize_name)


used_attendance_ids = set()

for value in final_df.loc[
    final_df[
        "match_status"
    ] == "MATCHED",
    "attendance_row_id"
]:

    row_id = to_int(value)

    if row_id is not None:
        used_attendance_ids.add(
            row_id
        )


override_logs = []


for override_index, override in (
    overrides_df.iterrows()
):

    year = to_int(
        override.get(
            "year"
        )
    )

    term = to_int(
        override.get(
            "term"
        )
    )

    grade = to_int(
        override.get(
            "grade"
        )
    )

    class_name = normalize_class(
        override.get(
            "class"
        )
    )

    student_name = clean_text(
        override.get(
            "student_name"
        )
    )

    student_name_norm = (
        normalize_name(
            student_name
        )
    )

    action = clean_text(
        override.get(
            "action"
        )
    ).upper()

    match_field = clean_text(
        override.get(
            "match_field"
        )
    ).lower()

    match_value = clean_text(
        override.get(
            "match_value"
        )
    )

    reason = clean_text(
        override.get(
            "reason"
        )
    )


    linkage_mask = (
        (final_df["_year"] == year)
        &
        (final_df["_grade"] == grade)
        &
        (final_df["_class"] == class_name)
        &
        (
            final_df[
                "_student_name_norm"
            ]
            ==
            student_name_norm
        )
    )


    if term is not None:

        linkage_mask = (
            linkage_mask
            &
            (
                final_df[
                    "_term"
                ] == term
            )
        )


    linkage_indices = (
        final_df[
            linkage_mask
        ].index.tolist()
    )


    if not linkage_indices:

        override_logs.append(
            {
                "student_name":
                    student_name,
                "year":
                    year,
                "term":
                    term,
                "grade":
                    grade,
                "class":
                    class_name,
                "action":
                    action,
                "match_field":
                    match_field,
                "match_value":
                    match_value,
                "result":
                    "LINKAGE_ROW_NOT_FOUND",
                "details":
                    reason,
            }
        )

        continue


    for linkage_index in linkage_indices:

        current = final_df.loc[
            linkage_index
        ]

        current_term = to_int(
            current[
                "term"
            ]
        )


        if (
            current[
                "match_status"
            ] == "MATCHED"
        ):

            override_logs.append(
                {
                    "student_name":
                        student_name,
                    "year":
                        year,
                    "term":
                        current_term,
                    "grade":
                        grade,
                    "class":
                        class_name,
                    "action":
                        action,
                    "match_field":
                        match_field,
                    "match_value":
                        match_value,
                    "result":
                        "ALREADY_MATCHED_AUTOMATICALLY",
                    "details":
                        current[
                            "match_method"
                        ],
                }
            )

            continue


        if action == "NO_ATTENDANCE":

            final_df.loc[
                linkage_index,
                "match_status"
            ] = (
                "NO_ATTENDANCE_CONFIRMED"
            )

            final_df.loc[
                linkage_index,
                "match_method"
            ] = (
                "MANUAL_CONFIRMED_NO_ATTENDANCE"
            )

            final_df.loc[
                linkage_index,
                "manual_override"
            ] = True

            final_df.loc[
                linkage_index,
                "manual_override_reason"
            ] = reason

            final_df.loc[
                linkage_index,
                "notes"
            ] = reason


            override_logs.append(
                {
                    "student_name":
                        student_name,
                    "year":
                        year,
                    "term":
                        current_term,
                    "grade":
                        grade,
                    "class":
                        class_name,
                    "action":
                        action,
                    "match_field":
                        "",
                    "match_value":
                        "",
                    "result":
                        "NO_ATTENDANCE_CONFIRMED",
                    "details":
                        reason,
                }
            )

            continue


        attendance_scope = (
            (attendance_df["_year"] == year)
            &
            (
                attendance_df[
                    "_term"
                ] == current_term
            )
            &
            (
                attendance_df[
                    "_grade"
                ] == grade
            )
            &
            (
                attendance_df[
                    "_class"
                ] == class_name
            )
        )


        candidates = attendance_df[
            attendance_scope
        ].copy()


        if match_field == "admission":

            wanted = normalize_identifier(
                match_value
            )

            candidates = candidates[
                candidates[
                    "_admission"
                ] == wanted
            ]


        elif match_field == "index":

            wanted = normalize_identifier(
                match_value
            )

            candidates = candidates[
                candidates[
                    "_index"
                ] == wanted
            ]


        elif match_field == "name":

            wanted = normalize_name(
                match_value
            )

            candidates = candidates[
                candidates[
                    "_name"
                ] == wanted
            ]


        elif match_field == "source_row":

            wanted = to_int(
                match_value
            )

            candidates = candidates[
                candidates[
                    "_source_row"
                ] == wanted
            ]


        else:

            override_logs.append(
                {
                    "student_name":
                        student_name,
                    "year":
                        year,
                    "term":
                        current_term,
                    "grade":
                        grade,
                    "class":
                        class_name,
                    "action":
                        action,
                    "match_field":
                        match_field,
                    "match_value":
                        match_value,
                    "result":
                        "INVALID_MATCH_FIELD",
                    "details":
                        reason,
                }
            )

            continue


        if len(candidates) != 1:

            override_logs.append(
                {
                    "student_name":
                        student_name,
                    "year":
                        year,
                    "term":
                        current_term,
                    "grade":
                        grade,
                    "class":
                        class_name,
                    "action":
                        action,
                    "match_field":
                        match_field,
                    "match_value":
                        match_value,
                    "result":
                        (
                            "NO_ATTENDANCE_CANDIDATE"
                            if len(candidates) == 0
                            else
                            "MULTIPLE_ATTENDANCE_CANDIDATES"
                        ),
                    "details":
                        (
                            f"Candidates found: "
                            f"{len(candidates)}. "
                            f"{reason}"
                        ),
                }
            )

            continue


        candidate = (
            candidates.iloc[0]
        )

        attendance_row_id = int(
            candidate[
                "_attendance_row_id"
            ]
        )


        if (
            attendance_row_id
            in used_attendance_ids
        ):

            override_logs.append(
                {
                    "student_name":
                        student_name,
                    "year":
                        year,
                    "term":
                        current_term,
                    "grade":
                        grade,
                    "class":
                        class_name,
                    "action":
                        action,
                    "match_field":
                        match_field,
                    "match_value":
                        match_value,
                    "result":
                        "ATTENDANCE_ROW_ALREADY_USED",
                    "details":
                        (
                            f"attendance_row_id="
                            f"{attendance_row_id}"
                        ),
                }
            )

            continue


        final_df.loc[
            linkage_index,
            "attendance_row_id"
        ] = attendance_row_id

        final_df.loc[
            linkage_index,
            "attendance_source_excel_row"
        ] = candidate[
            "source_excel_row"
        ]

        final_df.loc[
            linkage_index,
            "attendance_index_no"
        ] = candidate[
            "index_no"
        ]

        final_df.loc[
            linkage_index,
            "attendance_canonical_index_no"
        ] = candidate[
            "canonical_index_no"
        ]

        final_df.loc[
            linkage_index,
            "attendance_admission_no"
        ] = candidate[
            "admission_no"
        ]

        final_df.loc[
            linkage_index,
            "attendance_name"
        ] = candidate[
            "student_name_identifier"
        ]

        final_df.loc[
            linkage_index,
            "match_status"
        ] = "MATCHED"

        final_df.loc[
            linkage_index,
            "match_method"
        ] = (
            "MANUAL_OVERRIDE_"
            +
            match_field.upper()
        )

        final_df.loc[
            linkage_index,
            "candidate_count"
        ] = 1

        final_df.loc[
            linkage_index,
            "manual_override"
        ] = True

        final_df.loc[
            linkage_index,
            "manual_override_field"
        ] = match_field

        final_df.loc[
            linkage_index,
            "manual_override_value"
        ] = match_value

        final_df.loc[
            linkage_index,
            "manual_override_reason"
        ] = reason

        final_df.loc[
            linkage_index,
            "notes"
        ] = reason


        used_attendance_ids.add(
            attendance_row_id
        )


        override_logs.append(
            {
                "student_name":
                    student_name,
                "year":
                    year,
                "term":
                    current_term,
                "grade":
                    grade,
                "class":
                    class_name,
                "action":
                    action,
                "match_field":
                    match_field,
                "match_value":
                    match_value,
                "result":
                    "MATCHED_MANUALLY",
                "attendance_row_id":
                    attendance_row_id,
                "source_excel_row":
                    candidate[
                        "source_excel_row"
                    ],
                "attendance_index_no":
                    candidate[
                        "index_no"
                    ],
                "attendance_admission_no":
                    candidate[
                        "admission_no"
                    ],
                "details":
                    reason,
            }
        )


final_df = final_df.drop(
    columns=[
        "_year",
        "_term",
        "_grade",
        "_class",
        "_student_name_norm",
    ],
    errors="ignore"
)


override_log_df = pd.DataFrame(
    override_logs
)


final_unmatched_df = final_df[
    final_df[
        "match_status"
    ] == "UNMATCHED"
].copy()


final_ambiguous_df = final_df[
    final_df[
        "match_status"
    ] == "AMBIGUOUS"
].copy()


final_used_ids = set()

for value in final_df.loc[
    final_df[
        "match_status"
    ] == "MATCHED",
    "attendance_row_id"
]:

    row_id = to_int(value)

    if row_id is not None:
        final_used_ids.add(
            row_id
        )


final_unmatched_attendance_df = (
    attendance_df[
        ~attendance_df[
            "_attendance_row_id"
        ].isin(
            final_used_ids
        )
    ]
    .copy()
)


final_unmatched_attendance_df = (
    final_unmatched_attendance_df.drop(
        columns=[
            "_year",
            "_term",
            "_grade",
            "_class",
            "_admission",
            "_index",
            "_canonical_index",
            "_name",
            "_source_row",
        ],
        errors="ignore"
    )
)


final_df.to_csv(
    FINAL_OUTPUT,
    index=False
)

override_log_df.to_csv(
    OVERRIDE_LOG_OUTPUT,
    index=False
)

# Final diagnostics stay in memory; remove stale copies from older runs.
for legacy_output in LEGACY_DIAGNOSTIC_OUTPUTS:
    if legacy_output.exists():
        legacy_output.unlink()


print("\n" + "=" * 110)
print("FINAL STUDENT LINKAGE RESULT")
print("=" * 110)


print(
    "\nFinal linkage status:"
)

print(
    final_df[
        "match_status"
    ]
    .value_counts(
        dropna=False
    )
    .to_string()
)


manual_matches = int(
    (
        final_df[
            "match_method"
        ]
        .fillna("")
        .str.startswith(
            "MANUAL_OVERRIDE_"
        )
    ).sum()
)


matched = int(
    (
        final_df[
            "match_status"
        ] == "MATCHED"
    ).sum()
)


unavailable = int(
    (
        final_df[
            "match_status"
        ]
        ==
        "ATTENDANCE_UNAVAILABLE"
    ).sum()
)


confirmed_no_attendance = int(
    (
        final_df[
            "match_status"
        ]
        ==
        "NO_ATTENDANCE_CONFIRMED"
    ).sum()
)


unmatched = int(
    (
        final_df[
            "match_status"
        ] == "UNMATCHED"
    ).sum()
)


ambiguous = int(
    (
        final_df[
            "match_status"
        ] == "AMBIGUOUS"
    ).sum()
)


matchable = (
    len(final_df)
    -
    unavailable
)


match_rate = (
    matched
    /
    matchable
    *
    100
    if matchable
    else 0
)


resolved_available = (
    matched
    +
    confirmed_no_attendance
)


resolution_rate = (
    resolved_available
    /
    matchable
    *
    100
    if matchable
    else 0
)


print(
    f"\nMatched: "
    f"{matched}"
)

print(
    f"Manual matches applied: "
    f"{manual_matches}"
)

print(
    f"Attendance unavailable: "
    f"{unavailable}"
)

print(
    f"Confirmed marks-only / no attendance: "
    f"{confirmed_no_attendance}"
)

print(
    f"Still unmatched: "
    f"{unmatched}"
)

print(
    f"Still ambiguous: "
    f"{ambiguous}"
)

print(
    f"\nAttendance-eligible records: "
    f"{matchable}"
)

print(
    f"Final linkage rate: "
    f"{match_rate:.2f}%"
)

print(
    f"Resolution coverage including "
    f"confirmed marks-only records: "
    f"{resolution_rate:.2f}%"
)


print(
    "\nOverride results:"
)

if not override_log_df.empty:

    print(
        override_log_df[
            "result"
        ]
        .value_counts()
        .to_string()
    )


print(
    "\nOutputs:"
)

for output in [
    FINAL_OUTPUT,
    OVERRIDE_LOG_OUTPUT,
]:

    print(
        f"  {output}"
    )
