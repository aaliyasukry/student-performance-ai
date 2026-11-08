from pathlib import Path
from difflib import SequenceMatcher
import pandas as pd
import re


PROJECT_ROOT = Path(__file__).resolve().parents[2]

REPORTS_DIR = PROJECT_ROOT / "reports"

REPORTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


INPUT_FILE = (
    REPORTS_DIR /
    "student_linkage_final.csv"
)

OUTPUT_FILE = (
    REPORTS_DIR /
    "cross_year_identity_audit.csv"
)


SOURCE_YEAR = 2025
TARGET_YEAR = 2026


# Expected progression: 2025 Grades 6-9 to 2026 Grades 7-10.

EXPECTED_GRADE_PROGRESSION = {
    6: 7,
    7: 8,
    8: 9,
    9: 10,
}


def clean_text(value):
    """Safely convert a value to stripped text."""

    if pd.isna(value):
        return ""

    return str(value).strip()


def normalize_name(value):
    """Normalize names for deterministic comparison."""

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
    """Normalize identifier values while preserving full historical Index formats when present."""

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

        return (
            text.lstrip("0")
            or "0"
        )

    return text.upper()


def canonical_index(value):
    """Extract the final numeric component from an Index No."""

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
    """Convert CSV numeric text to int."""

    if pd.isna(value):
        return None

    text = clean_text(
        value
    )

    if not text:
        return None

    try:

        return int(
            float(text)
        )

    except (
        ValueError,
        TypeError
    ):

        return None


def normalize_class(value):
    """Normalize class labels."""

    return clean_text(
        value
    ).upper()


def values_as_set(series, normalizer):
    """Convert non-empty values in a Series into a normalized unique set."""

    result = set()

    for value in series:

        normalized_value = normalizer(
            value
        )

        if normalized_value:

            result.add(
                normalized_value
            )

    return result


def set_to_text(values):
    """Make set readable in the audit CSV."""

    if not values:
        return ""

    return " | ".join(
        sorted(values)
    )


def overlap(left, right):
    """Return intersection of two sets."""

    return (
        set(left)
        &
        set(right)
    )


def name_similarity(
    name_a,
    name_b
):
    """Similarity is used only as supporting audit evidence."""

    if (
        not name_a
        or not name_b
    ):
        return 0.0

    return SequenceMatcher(
        None,
        name_a,
        name_b
    ).ratio()


if not INPUT_FILE.exists():

    raise FileNotFoundError(
        f"Final linkage file not found:\n"
        f"{INPUT_FILE}"
    )


df = pd.read_csv(
    INPUT_FILE,
    dtype=str
)


required_columns = [
    "year",
    "term",
    "grade",
    "class",
    "student_name",
    "resolved_marks_identifier",
    "attendance_index_no",
    "attendance_admission_no",
    "match_status",
]


for column in required_columns:

    if column not in df.columns:

        raise ValueError(
            f"Required column missing: {column}\n"
            f"Available columns:\n"
            f"{list(df.columns)}"
        )


print(
    "=" * 110
)

print(
    "STUDENT PERFORMANCE AI - "
    "CROSS-YEAR STUDENT IDENTITY AUDIT"
)

print(
    "=" * 110
)


df = df.copy()


df["_year"] = (
    df["year"]
    .apply(to_int)
)


df["_term"] = (
    df["term"]
    .apply(to_int)
)


df["_grade"] = (
    df["grade"]
    .apply(to_int)
)


df["_class"] = (
    df["class"]
    .apply(normalize_class)
)


df["_name"] = (
    df["student_name"]
    .apply(normalize_name)
)


df["_marks_identifier"] = (
    df[
        "resolved_marks_identifier"
    ]
    .apply(
        normalize_identifier
    )
)


df["_attendance_admission"] = (
    df[
        "attendance_admission_no"
    ]
    .apply(
        normalize_identifier
    )
)


df["_attendance_index"] = (
    df[
        "attendance_index_no"
    ]
    .apply(
        normalize_identifier
    )
)


df["_attendance_canonical_index"] = (
    df[
        "attendance_index_no"
    ]
    .apply(
        canonical_index
    )
)


# Group by year + grade + class + normalized name, not Admission No. alone (duplicates exist).

entity_records = []


group_columns = [
    "_year",
    "_grade",
    "_class",
    "_name",
]


for group_key, group in df.groupby(
    group_columns,
    dropna=False
):

    (
        year,
        grade,
        class_name,
        normalized_student_name
    ) = group_key


    original_names = {
        clean_text(value)
        for value in group[
            "student_name"
        ]
        if clean_text(value)
    }


    admissions = values_as_set(
        group[
            "attendance_admission_no"
        ],
        normalize_identifier
    )


    full_indexes = values_as_set(
        group[
            "attendance_index_no"
        ],
        normalize_identifier
    )


    canonical_indexes = {
        canonical_index(value)
        for value in group[
            "attendance_index_no"
        ]
        if canonical_index(value)
    }


    marks_identifiers = values_as_set(
        group[
            "resolved_marks_identifier"
        ],
        normalize_identifier
    )


    terms = {
        to_int(value)
        for value in group[
            "term"
        ]
        if to_int(value) is not None
    }


    # Flag same-name collisions: more than one student row in the same term.

    term_counts = (
        group[
            "_term"
        ]
        .value_counts()
    )


    same_term_collision = bool(
        (
            term_counts > 1
        ).any()
    )


    entity_records.append(
        {
            "year":
                year,

            "grade":
                grade,

            "class":
                class_name,

            "normalized_name":
                normalized_student_name,

            "display_name":
                (
                    sorted(
                        original_names
                    )[0]
                    if original_names
                    else ""
                ),

            "name_variants":
                set_to_text(
                    original_names
                ),

            "terms_present":
                ", ".join(
                    str(term)
                    for term in sorted(
                        terms
                    )
                ),

            "term_record_count":
                len(group),

            "admissions":
                admissions,

            "full_indexes":
                full_indexes,

            "canonical_indexes":
                canonical_indexes,

            "marks_identifiers":
                marks_identifiers,

            "same_term_collision":
                same_term_collision,
        }
    )


entities_df = pd.DataFrame(
    entity_records
)


entities_df = (
    entities_df
    .sort_values(
        by=[
            "year",
            "grade",
            "class",
            "display_name",
        ]
    )
    .reset_index(
        drop=True
    )
)


# Temporary entity IDs, not the final anonymized STU IDs.

entities_df[
    "year_entity_id"
] = [
    f"YENT{number:04d}"
    for number in range(
        1,
        len(entities_df) + 1
    )
]


source_entities = (
    entities_df[
        entities_df[
            "year"
        ] == SOURCE_YEAR
    ]
    .copy()
)


target_entities = (
    entities_df[
        entities_df[
            "year"
        ] == TARGET_YEAR
    ]
    .copy()
)


print(
    f"\n{SOURCE_YEAR} year-level student entities: "
    f"{len(source_entities)}"
)

print(
    f"{TARGET_YEAR} year-level student entities: "
    f"{len(target_entities)}"
)


def score_candidate(
    source,
    target
):
    """Calculate cross-year identity evidence."""

    score = 0
    evidence = []


    admission_overlap = overlap(
        source[
            "admissions"
        ],
        target[
            "admissions"
        ]
    )


    full_index_overlap = overlap(
        source[
            "full_indexes"
        ],
        target[
            "full_indexes"
        ]
    )


    canonical_overlap = overlap(
        source[
            "canonical_indexes"
        ],
        target[
            "canonical_indexes"
        ]
    )


    marks_id_overlap = overlap(
        source[
            "marks_identifiers"
        ],
        target[
            "marks_identifiers"
        ]
    )


    exact_name = (
        bool(
            source[
                "normalized_name"
            ]
        )
        and
        source[
            "normalized_name"
        ]
        ==
        target[
            "normalized_name"
        ]
    )


    similarity = name_similarity(
        source[
            "normalized_name"
        ],
        target[
            "normalized_name"
        ]
    )


    if admission_overlap:

        score += 100

        evidence.append(
            "EXACT_ADMISSION"
        )


    if full_index_overlap:

        score += 80

        evidence.append(
            "EXACT_FULL_INDEX"
        )


    if canonical_overlap:

        score += 60

        evidence.append(
            "CANONICAL_INDEX"
        )


    if marks_id_overlap:

        score += 50

        evidence.append(
            "MARKS_IDENTIFIER"
        )


    if exact_name:

        score += 40

        evidence.append(
            "EXACT_NAME"
        )


    elif similarity >= 0.90:

        # Similarity alone never creates a strong match.
        score += 10

        evidence.append(
            "HIGH_NAME_SIMILARITY"
        )


    return {
        "score":
            score,

        "evidence":
            evidence,

        "admission_overlap":
            admission_overlap,

        "full_index_overlap":
            full_index_overlap,

        "canonical_overlap":
            canonical_overlap,

        "marks_id_overlap":
            marks_id_overlap,

        "exact_name":
            exact_name,

        "name_similarity":
            similarity,
    }


audit_records = []

candidate_records = []


for _, source in (
    source_entities.iterrows()
):

    source_grade = to_int(
        source[
            "grade"
        ]
    )


    source_collision = bool(
        source[
            "same_term_collision"
        ]
    )


    # Grade 10 leaves current research scope

    if (
        source_grade
        not in EXPECTED_GRADE_PROGRESSION
    ):

        audit_records.append(
            {
                "source_entity_id":
                    source[
                        "year_entity_id"
                    ],

                "source_year":
                    SOURCE_YEAR,

                "source_grade":
                    source_grade,

                "source_class":
                    source[
                        "class"
                    ],

                "source_name":
                    source[
                        "display_name"
                    ],

                "source_admissions":
                    set_to_text(
                        source[
                            "admissions"
                        ]
                    ),

                "source_indexes":
                    set_to_text(
                        source[
                            "full_indexes"
                        ]
                    ),

                "target_entity_id":
                    "",

                "target_year":
                    TARGET_YEAR,

                "target_grade":
                    "",

                "target_class":
                    "",

                "target_name":
                    "",

                "target_admissions":
                    "",

                "target_indexes":
                    "",

                "match_score":
                    0,

                "name_similarity":
                    "",

                "evidence":
                    "",

                "identity_status":
                    "EXITED_RESEARCH_SCOPE",

                "review_required":
                    False,

                "notes":
                    (
                        "Expected next grade is outside "
                        "the Grades 6-10 research scope."
                    ),
            }
        )

        continue


    expected_target_grade = (
        EXPECTED_GRADE_PROGRESSION[
            source_grade
        ]
    )


    candidates = target_entities[
        target_entities[
            "grade"
        ] == expected_target_grade
    ]


    source_candidates = []


    for _, target in (
        candidates.iterrows()
    ):

        scored = score_candidate(
            source,
            target
        )


        # Ignore completely unrelated candidates.

        if scored[
            "score"
        ] <= 0:

            continue


        source_candidates.append(
            {
                "target":
                    target,

                **scored,
            }
        )


    source_candidates = sorted(
        source_candidates,
        key=lambda item:
            item[
                "score"
            ],
        reverse=True
    )


    # No candidate

    if not source_candidates:

        audit_records.append(
            {
                "source_entity_id":
                    source[
                        "year_entity_id"
                    ],

                "source_year":
                    SOURCE_YEAR,

                "source_grade":
                    source_grade,

                "source_class":
                    source[
                        "class"
                    ],

                "source_name":
                    source[
                        "display_name"
                    ],

                "source_admissions":
                    set_to_text(
                        source[
                            "admissions"
                        ]
                    ),

                "source_indexes":
                    set_to_text(
                        source[
                            "full_indexes"
                        ]
                    ),

                "target_entity_id":
                    "",

                "target_year":
                    TARGET_YEAR,

                "target_grade":
                    expected_target_grade,

                "target_class":
                    "",

                "target_name":
                    "",

                "target_admissions":
                    "",

                "target_indexes":
                    "",

                "match_score":
                    0,

                "name_similarity":
                    "",

                "evidence":
                    "",

                "identity_status":
                    "NO_2026_MATCH",

                "review_required":
                    True,

                "notes":
                    (
                        "No deterministic candidate found "
                        "in the expected next grade."
                    ),
            }
        )

        continue


    best = source_candidates[0]

    target = best[
        "target"
    ]


    # Check whether the top score is tied.

    best_score = best[
        "score"
    ]


    equally_ranked = [
        item
        for item in source_candidates
        if item[
            "score"
        ] == best_score
    ]


    # Determine confidence / status

    if (
        source_collision
        or bool(
            target[
                "same_term_collision"
            ]
        )
    ):

        identity_status = (
            "AMBIGUOUS_YEAR_ENTITY"
        )

        review_required = True


    elif len(
        equally_ranked
    ) > 1:

        identity_status = (
            "AMBIGUOUS_CROSS_YEAR_MATCH"
        )

        review_required = True


    elif (
        "EXACT_ADMISSION"
        in best[
            "evidence"
        ]
    ):

        identity_status = (
            "MATCHED_STRONG"
        )

        review_required = False


    elif (
        "EXACT_FULL_INDEX"
        in best[
            "evidence"
        ]
        and
        (
            "EXACT_NAME"
            in best[
                "evidence"
            ]
            or
            "MARKS_IDENTIFIER"
            in best[
                "evidence"
            ]
        )
    ):

        identity_status = (
            "MATCHED_STRONG"
        )

        review_required = False


    elif (
        "MARKS_IDENTIFIER"
        in best[
            "evidence"
        ]
        and
        "EXACT_NAME"
        in best[
            "evidence"
        ]
    ):

        identity_status = (
            "MATCHED_STRONG"
        )

        review_required = False


    elif (
        "CANONICAL_INDEX"
        in best[
            "evidence"
        ]
        and
        "EXACT_NAME"
        in best[
            "evidence"
        ]
    ):

        identity_status = (
            "MATCHED_STRONG"
        )

        review_required = False


    elif (
        best[
            "evidence"
        ] == [
            "EXACT_NAME"
        ]
    ):

        identity_status = (
            "REVIEW_NAME_ONLY"
        )

        review_required = True


    else:

        identity_status = (
            "REVIEW_SUPPORTING_EVIDENCE"
        )

        review_required = True


    audit_records.append(
        {
            "source_entity_id":
                source[
                    "year_entity_id"
                ],

            "source_year":
                SOURCE_YEAR,

            "source_grade":
                source_grade,

            "source_class":
                source[
                    "class"
                ],

            "source_name":
                source[
                    "display_name"
                ],

            "source_admissions":
                set_to_text(
                    source[
                        "admissions"
                    ]
                ),

            "source_indexes":
                set_to_text(
                    source[
                        "full_indexes"
                    ]
                ),

            "target_entity_id":
                target[
                    "year_entity_id"
                ],

            "target_year":
                TARGET_YEAR,

            "target_grade":
                target[
                    "grade"
                ],

            "target_class":
                target[
                    "class"
                ],

            "target_name":
                target[
                    "display_name"
                ],

            "target_admissions":
                set_to_text(
                    target[
                        "admissions"
                    ]
                ),

            "target_indexes":
                set_to_text(
                    target[
                        "full_indexes"
                    ]
                ),

            "match_score":
                best_score,

            "name_similarity":
                round(
                    best[
                        "name_similarity"
                    ],
                    4
                ),

            "evidence":
                " + ".join(
                    best[
                        "evidence"
                    ]
                ),

            "identity_status":
                identity_status,

            "review_required":
                review_required,

            "notes":
                (
                    "Candidate restricted to expected "
                    "grade progression."
                ),
        }
    )


audit_df = pd.DataFrame(
    audit_records
).astype(object)


matched_target_mask = (
    audit_df[
        "target_entity_id"
    ].fillna("") != ""
)


target_claim_counts = (
    audit_df.loc[
        matched_target_mask,
        "target_entity_id"
    ]
    .value_counts()
)


duplicated_target_ids = set(
    target_claim_counts[
        target_claim_counts > 1
    ].index
)


if duplicated_target_ids:

    duplicate_mask = (
        audit_df[
            "target_entity_id"
        ].isin(
            duplicated_target_ids
        )
    )


    audit_df.loc[
        duplicate_mask,
        "identity_status"
    ] = (
        "AMBIGUOUS_TARGET_CLAIMED_MULTIPLE_TIMES"
    )


    audit_df.loc[
        duplicate_mask,
        "review_required"
    ] = True


    audit_df.loc[
        duplicate_mask,
        "notes"
    ] = (
        "The same 2026 entity was selected by more "
        "than one 2025 entity. Manual review required."
    )


claimed_target_ids = set(
    audit_df[
        "target_entity_id"
    ]
    .dropna()
    .astype(str)
)


claimed_target_ids.discard(
    ""
)


for _, target in (
    target_entities.iterrows()
):

    target_id = target[
        "year_entity_id"
    ]


    if target_id in claimed_target_ids:
        continue


    target_grade = to_int(
        target[
            "grade"
        ]
    )


    # Grade 6 has no Grade 5 source, so it enters the scope directly.

    if target_grade == 6:

        identity_status = (
            "ENTERED_RESEARCH_SCOPE"
        )

        review_required = False

        notes = (
            "Grade 5 is outside the research dataset; "
            "this student's earlier record is therefore "
            "not expected."
        )

    else:

        identity_status = (
            "NEW_OR_TRANSFERRED_2026"
        )

        review_required = True

        notes = (
            "No 2025 identity was linked. This may represent "
            "a new/transfer student or an unresolved identity."
        )


    audit_df.loc[
        len(audit_df)
    ] = {
        "source_entity_id":
            "",

        "source_year":
            SOURCE_YEAR,

        "source_grade":
            "",

        "source_class":
            "",

        "source_name":
            "",

        "source_admissions":
            "",

        "source_indexes":
            "",

        "target_entity_id":
            target_id,

        "target_year":
            TARGET_YEAR,

        "target_grade":
            target_grade,

        "target_class":
            target[
                "class"
            ],

        "target_name":
            target[
                "display_name"
            ],

        "target_admissions":
            set_to_text(
                target[
                    "admissions"
                ]
            ),

        "target_indexes":
            set_to_text(
                target[
                    "full_indexes"
                ]
            ),

        "match_score":
            0,

        "name_similarity":
            "",

        "evidence":
            "",

        "identity_status":
            identity_status,

        "review_required":
            review_required,

        "notes":
            notes,
    }


audit_df = (
    audit_df
    .sort_values(
        by=[
            "review_required",
            "source_grade",
            "target_grade",
            "source_name",
            "target_name",
        ],
        ascending=[
            False,
            True,
            True,
            True,
            True,
        ],
        na_position="last"
    )
    .reset_index(
        drop=True
    )
)


audit_df.to_csv(
    OUTPUT_FILE,
    index=False
)


print(
    "\n"
    + "=" * 110
)

print(
    "CROSS-YEAR IDENTITY AUDIT COMPLETE"
)

print(
    "=" * 110
)


print(
    f"\nYear-level entities created: "
    f"{len(entities_df)}"
)


print(
    "\nIdentity status:"
)

print(
    audit_df[
        "identity_status"
    ]
    .value_counts(
        dropna=False
    )
    .to_string()
)


review_count = int(
    audit_df[
        "review_required"
    ]
    .fillna(False)
    .astype(bool)
    .sum()
)


strong_matches = int(
    (
        audit_df[
            "identity_status"
        ]
        ==
        "MATCHED_STRONG"
    ).sum()
)


print(
    f"\nStrong automatic cross-year matches: "
    f"{strong_matches}"
)


print(
    f"Records requiring manual review: "
    f"{review_count}"
)


print(
    f"\nAudit saved to:\n"
    f"{OUTPUT_FILE}"
)
