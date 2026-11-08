from pathlib import Path
import pandas as pd
import re


PROJECT_ROOT = Path(__file__).resolve().parents[2]

REPORTS_DIR = PROJECT_ROOT / "reports"
CONFIG_DIR = PROJECT_ROOT / "config"
CLEANED_DIR = PROJECT_ROOT / "data" / "cleaned"

REPORTS_DIR.mkdir(parents=True, exist_ok=True)
CONFIG_DIR.mkdir(parents=True, exist_ok=True)
CLEANED_DIR.mkdir(parents=True, exist_ok=True)

AUDIT_FILE = REPORTS_DIR / "cross_year_identity_audit.csv"
OVERRIDES_FILE = CONFIG_DIR / "cross_year_identity_manual_overrides.csv"
LINKAGE_FILE = REPORTS_DIR / "student_linkage_final.csv"

FINAL_AUDIT_OUTPUT = REPORTS_DIR / "cross_year_identity_final.csv"
IDENTITY_MAP_OUTPUT = CLEANED_DIR / "student_identity_map.csv"


def clean_text(value):
    if pd.isna(value):
        return ""

    return str(value).strip()


def normalize_name(value):
    text = clean_text(value).lower()

    if not text:
        return ""

    text = re.sub(
        r"[^a-z0-9]+",
        " ",
        text,
    )

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


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
        TypeError,
    ):
        return None


def normalize_class(value):
    return clean_text(value).upper()


def entity_key(
    year,
    grade,
    class_name,
    student_name,
):
    year_value = to_int(year)
    grade_value = to_int(grade)
    class_value = normalize_class(class_name)
    name_value = normalize_name(student_name)

    if (
        year_value is None
        or grade_value is None
        or not class_value
        or not name_value
    ):
        return None

    return (
        year_value,
        grade_value,
        class_value,
        name_value,
    )


def key_to_text(key):
    if key is None:
        return ""

    return (
        f"{key[0]}|"
        f"{key[1]}|"
        f"{key[2]}|"
        f"{key[3]}"
    )


for required_file in [
    AUDIT_FILE,
    OVERRIDES_FILE,
    LINKAGE_FILE,
]:
    if not required_file.exists():
        raise FileNotFoundError(
            f"Required file not found:\n"
            f"{required_file}"
        )


audit_df = pd.read_csv(
    AUDIT_FILE,
    dtype=str,
).astype(object)

overrides_df = pd.read_csv(
    OVERRIDES_FILE,
    dtype=str,
).astype(object)

linkage_df = pd.read_csv(
    LINKAGE_FILE,
    dtype=str,
).astype(object)


print("=" * 110)
print(
    "STUDENT PERFORMANCE AI - "
    "CROSS-YEAR IDENTITY OVERRIDES"
)
print("=" * 110)


required_audit_columns = [
    "source_year",
    "source_grade",
    "source_class",
    "source_name",
    "target_year",
    "target_grade",
    "target_class",
    "target_name",
    "identity_status",
    "review_required",
]

required_override_columns = [
    "source_year",
    "source_grade",
    "source_class",
    "source_name",
    "target_year",
    "target_grade",
    "target_class",
    "target_name",
    "action",
    "reason",
]

required_linkage_columns = [
    "year",
    "term",
    "grade",
    "class",
    "student_name",
]

for column in required_audit_columns:
    if column not in audit_df.columns:
        raise ValueError(
            f"Missing audit column: {column}"
        )

for column in required_override_columns:
    if column not in overrides_df.columns:
        raise ValueError(
            f"Missing override column: {column}"
        )

for column in required_linkage_columns:
    if column not in linkage_df.columns:
        raise ValueError(
            f"Missing linkage column: {column}"
        )


final_df = audit_df.copy()

final_df[
    "pre_manual_identity_status"
] = final_df[
    "identity_status"
]

final_df[
    "manual_resolution"
] = False

final_df[
    "manual_action"
] = ""

final_df[
    "manual_reason"
] = ""

final_df[
    "final_identity_status"
] = final_df[
    "identity_status"
]


status_map = {
    "MATCHED_STRONG":
        "MATCHED_AUTOMATICALLY",

    "EXITED_RESEARCH_SCOPE":
        "EXITED_RESEARCH_SCOPE",

    "ENTERED_RESEARCH_SCOPE":
        "ENTERED_RESEARCH_SCOPE",
}

for original, final_status in status_map.items():
    mask = (
        final_df[
            "identity_status"
        ] == original
    )

    final_df.loc[
        mask,
        "final_identity_status"
    ] = final_status


override_results = []

for _, override in overrides_df.iterrows():
    action = clean_text(
        override.get("action")
    ).upper()

    reason = clean_text(
        override.get("reason")
    )

    source_key = entity_key(
        override.get("source_year"),
        override.get("source_grade"),
        override.get("source_class"),
        override.get("source_name"),
    )

    target_key = entity_key(
        override.get("target_year"),
        override.get("target_grade"),
        override.get("target_class"),
        override.get("target_name"),
    )

    mask = pd.Series(
        True,
        index=final_df.index,
    )

    if source_key is not None:
        mask &= (
            final_df.apply(
                lambda row:
                    entity_key(
                        row.get("source_year"),
                        row.get("source_grade"),
                        row.get("source_class"),
                        row.get("source_name"),
                    )
                    == source_key,
                axis=1,
            )
        )

    if target_key is not None:
        mask &= (
            final_df.apply(
                lambda row:
                    entity_key(
                        row.get("target_year"),
                        row.get("target_grade"),
                        row.get("target_class"),
                        row.get("target_name"),
                    )
                    == target_key,
                axis=1,
            )
        )

    matching_rows = final_df[
        mask
    ].index.tolist()

    if len(matching_rows) != 1:
        override_results.append(
            {
                "action": action,
                "source_key": key_to_text(source_key),
                "target_key": key_to_text(target_key),
                "result": (
                    "NOT_FOUND"
                    if len(matching_rows) == 0
                    else "MULTIPLE_ROWS_FOUND"
                ),
                "details": (
                    f"Rows found: "
                    f"{len(matching_rows)}"
                ),
            }
        )
        continue

    row_index = matching_rows[0]

    current_status = clean_text(
        final_df.loc[
            row_index,
            "identity_status"
        ]
    )

    if action == "CONFIRM_LEFT":
        if current_status != "NO_2026_MATCH":
            result = "INVALID_STATUS_FOR_ACTION"
        else:
            final_status = "LEFT_SCHOOL_CONFIRMED"
            result = "APPLIED"

    elif action == "CONFIRM_SAME":
        if current_status not in {
            "REVIEW_NAME_ONLY",
            "REVIEW_SUPPORTING_EVIDENCE",
        }:
            result = "INVALID_STATUS_FOR_ACTION"
        else:
            final_status = "MATCHED_MANUALLY"
            result = "APPLIED"

    elif action == "CONFIRM_NEW_OR_TRANSFERRED":
        if current_status != "NEW_OR_TRANSFERRED_2026":
            result = "INVALID_STATUS_FOR_ACTION"
        else:
            final_status = (
                "NEW_OR_TRANSFERRED_CONFIRMED"
            )
            result = "APPLIED"

    else:
        result = "INVALID_ACTION"

    if result == "APPLIED":
        final_df.loc[
            row_index,
            "manual_resolution"
        ] = True

        final_df.loc[
            row_index,
            "manual_action"
        ] = action

        final_df.loc[
            row_index,
            "manual_reason"
        ] = reason

        final_df.loc[
            row_index,
            "final_identity_status"
        ] = final_status

        final_df.loc[
            row_index,
            "review_required"
        ] = False

    override_results.append(
        {
            "action": action,
            "source_key": key_to_text(source_key),
            "target_key": key_to_text(target_key),
            "result": result,
            "details": (
                final_status
                if result == "APPLIED"
                else current_status
            ),
        }
    )


override_results_df = pd.DataFrame(
    override_results
)


failed_overrides = override_results_df[
    override_results_df[
        "result"
    ] != "APPLIED"
]

if not failed_overrides.empty:
    print(
        "\nERROR: Some manual overrides "
        "could not be applied."
    )

    print(
        failed_overrides.to_string(
            index=False
        )
    )

    raise RuntimeError(
        "Cross-year manual override "
        "application failed."
    )


unresolved_mask = (
    final_df[
        "review_required"
    ]
    .fillna(False)
    .astype(str)
    .str.lower()
    .isin({
        "true",
        "1",
        "yes",
    })
)

unresolved_df = final_df[
    unresolved_mask
]

if not unresolved_df.empty:
    print(
        "\nERROR: Cross-year identity audit "
        "still contains unresolved rows."
    )

    print(
        unresolved_df[
            [
                "identity_status",
                "source_name",
                "target_name",
            ]
        ].to_string(
            index=False
        )
    )

    raise RuntimeError(
        "Identity override processing cannot be finalized while "
        "identity rows remain unresolved."
    )


all_entity_keys = set()
matched_pairs = []

for _, row in final_df.iterrows():
    source_key = entity_key(
        row.get("source_year"),
        row.get("source_grade"),
        row.get("source_class"),
        row.get("source_name"),
    )

    target_key = entity_key(
        row.get("target_year"),
        row.get("target_grade"),
        row.get("target_class"),
        row.get("target_name"),
    )

    if source_key is not None:
        all_entity_keys.add(
            source_key
        )

    if target_key is not None:
        all_entity_keys.add(
            target_key
        )

    final_status = clean_text(
        row.get("final_identity_status")
    )

    if final_status in {
        "MATCHED_AUTOMATICALLY",
        "MATCHED_MANUALLY",
    }:
        if (
            source_key is None
            or target_key is None
        ):
            raise RuntimeError(
                "Matched identity row is missing "
                "a source or target entity."
            )

        matched_pairs.append(
            (
                source_key,
                target_key,
            )
        )


parent = {
    key: key
    for key in all_entity_keys
}


def find(key):
    while parent[key] != key:
        parent[key] = parent[
            parent[key]
        ]
        key = parent[key]

    return key


def union(left, right):
    root_left = find(left)
    root_right = find(right)

    if root_left == root_right:
        return

    if root_left < root_right:
        parent[root_right] = root_left
    else:
        parent[root_left] = root_right


for source_key, target_key in matched_pairs:
    union(
        source_key,
        target_key,
    )


identity_groups = {}

for key in sorted(
    all_entity_keys
):
    root = find(key)

    identity_groups.setdefault(
        root,
        []
    ).append(
        key
    )


# Deterministic PID ordering based on the earliest entity key belonging to the identity.
ordered_groups = sorted(
    identity_groups.values(),
    key=lambda group:
        min(group),
)

entity_to_pid = {}

for number, group in enumerate(
    ordered_groups,
    start=1,
):
    pid = f"PID{number:04d}"

    for key in group:
        entity_to_pid[key] = pid


persistent_ids = []

for _, row in final_df.iterrows():
    source_key = entity_key(
        row.get("source_year"),
        row.get("source_grade"),
        row.get("source_class"),
        row.get("source_name"),
    )

    target_key = entity_key(
        row.get("target_year"),
        row.get("target_grade"),
        row.get("target_class"),
        row.get("target_name"),
    )

    source_pid = (
        entity_to_pid.get(
            source_key,
            "",
        )
        if source_key is not None
        else ""
    )

    target_pid = (
        entity_to_pid.get(
            target_key,
            "",
        )
        if target_key is not None
        else ""
    )

    if (
        source_pid
        and target_pid
        and source_pid != target_pid
        and clean_text(
            row.get(
                "final_identity_status"
            )
        )
        in {
            "MATCHED_AUTOMATICALLY",
            "MATCHED_MANUALLY",
        }
    ):
        raise RuntimeError(
            "Matched cross-year entities were "
            "assigned different persistent IDs."
        )

    persistent_ids.append(
        source_pid
        or target_pid
    )


final_df[
    "persistent_student_id"
] = persistent_ids


identity_map_df = linkage_df.copy()

identity_map_df[
    "_entity_key"
] = identity_map_df.apply(
    lambda row:
        entity_key(
            row.get("year"),
            row.get("grade"),
            row.get("class"),
            row.get("student_name"),
        ),
    axis=1,
)

identity_map_df[
    "persistent_student_id"
] = identity_map_df[
    "_entity_key"
].map(
    entity_to_pid
)


missing_pid_df = identity_map_df[
    identity_map_df[
        "persistent_student_id"
    ].isna()
]

if not missing_pid_df.empty:
    print(
        "\nERROR: Some cross-year identity audit records could "
        "not be assigned a persistent identity."
    )

    print(
        missing_pid_df[
            [
                "year",
                "term",
                "grade",
                "class",
                "student_name",
            ]
        ].to_string(
            index=False
        )
    )

    raise RuntimeError(
        "Persistent identity coverage is incomplete."
    )


identity_map_df = identity_map_df.drop(
    columns=[
        "_entity_key",
    ],
    errors="ignore",
)


final_df.to_csv(
    FINAL_AUDIT_OUTPUT,
    index=False,
)

identity_map_df.to_csv(
    IDENTITY_MAP_OUTPUT,
    index=False,
)


print(
    "\n"
    + "=" * 110
)

print(
    "FINAL CROSS-YEAR IDENTITY RESULT"
)

print(
    "=" * 110
)

print(
    "\nFinal identity status:"
)

print(
    final_df[
        "final_identity_status"
    ]
    .value_counts(
        dropna=False
    )
    .to_string()
)

manual_applied = int(
    final_df[
        "manual_resolution"
    ]
    .fillna(False)
    .astype(bool)
    .sum()
)

persistent_count = (
    identity_map_df[
        "persistent_student_id"
    ]
    .nunique()
)

mapped_term_rows = int(
    identity_map_df[
        "persistent_student_id"
    ]
    .notna()
    .sum()
)

print(
    f"\nManual resolutions applied: "
    f"{manual_applied}"
)

print(
    f"Persistent student identities: "
    f"{persistent_count}"
)

print(
    f"Student-term rows mapped: "
    f"{mapped_term_rows} / "
    f"{len(identity_map_df)}"
)

print(
    "Unresolved cross-year identities: 0"
)

print(
    "\nOutputs:"
)

print(
    f"  {FINAL_AUDIT_OUTPUT}"
)

print(
    f"  {IDENTITY_MAP_OUTPUT}"
)
