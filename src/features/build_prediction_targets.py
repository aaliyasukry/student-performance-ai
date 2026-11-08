from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_QUALITY_DIR = PROJECT_ROOT / "reports" / "data_quality"
DATA_QUALITY_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_FILE = (
    PROJECT_ROOT
    / "data"
    / "model_ready"
    / "historical_features.csv"
)

MASTER_FILE = (
    PROJECT_ROOT
    / "data"
    / "anonymized"
    / "master_research_dataset.csv"
)

MODEL_READY_DIR = (
    PROJECT_ROOT
    / "data"
    / "model_ready"
)

REPORTS_DIR = PROJECT_ROOT / "reports"

MODEL_READY_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

OVERALL_OUTPUT_FILE = (
    MODEL_READY_DIR
    / "overall_prediction_dataset.csv"
)

SUBJECT_OUTPUT_FILE = (
    MODEL_READY_DIR
    / "subject_prediction_dataset.csv"
)

SUMMARY_FILE = (
    DATA_QUALITY_DIR
    / "target_construction_summary.csv"
)

SUBJECT_SUMMARY_FILE = (
    DATA_QUALITY_DIR
    / "subject_target_availability_summary.csv"
)


TRAIN_PERIODS = {
    "2025_T2",
    "2025_T3",
}

VALIDATION_PERIODS = {
    "2026_T1",
}

FINAL_TEST_PERIODS = {
    "2026_T2",
}

ACADEMIC_PERIOD_SEQUENCE = {
    "2025_T1": 1,
    "2025_T2": 2,
    "2025_T3": 3,
    "2026_T1": 4,
    "2026_T2": 5,
}


def to_numeric(series):
    return pd.to_numeric(series, errors="coerce")


def assign_split(period):
    if period in TRAIN_PERIODS:
        return "TRAIN"
    if period in VALIDATION_PERIODS:
        return "VALIDATION"
    if period in FINAL_TEST_PERIODS:
        return "FINAL_TEST"
    return "NOT_MODELLED"


def assert_unique_student_term(df, label):
    duplicate_mask = df.duplicated(
        subset=["student_id", "year", "term"],
        keep=False,
    )

    if duplicate_mask.any():
        duplicate_rows = df.loc[
            duplicate_mask,
            ["student_id", "year", "term", "academic_period"],
        ]

        raise RuntimeError(
            f"\nDuplicate student-term rows detected in {label}:\n\n"
            f"{duplicate_rows.to_string(index=False)}"
        )


if not FEATURE_FILE.exists():
    raise FileNotFoundError(
        f"Historical feature dataset not found:\n{FEATURE_FILE}"
    )

if not MASTER_FILE.exists():
    raise FileNotFoundError(
        f"Master research dataset not found:\n{MASTER_FILE}"
    )

features_df = pd.read_csv(FEATURE_FILE)
master_df = pd.read_csv(MASTER_FILE)

print("=" * 110)
print("STUDENT PERFORMANCE AI - TARGET CONSTRUCTION AND TEMPORAL SPLIT")
print("=" * 110)


for dataframe in [features_df, master_df]:
    for column in ["year", "term", "period_order", "grade"]:
        dataframe[column] = pd.to_numeric(
            dataframe[column],
            errors="raise",
        ).astype(int)

assert_unique_student_term(
    features_df,
    "historical_features",
)
assert_unique_student_term(
    master_df,
    "master_research_dataset",
)


mark_columns = sorted(
    column
    for column in master_df.columns
    if column.startswith("mark_")
)

status_columns = sorted(
    column
    for column in master_df.columns
    if column.startswith("status_")
)

subject_slugs = [
    column.replace("mark_", "", 1)
    for column in mark_columns
]

print(f"\nHistorical feature rows: {len(features_df)}")
print(f"Master dataset rows: {len(master_df)}")
print(f"Subjects detected: {len(subject_slugs)}")


target_columns = [
    "student_id",
    "year",
    "term",
    "academic_period",
    "calculated_average",
]

for subject in subject_slugs:
    mark_column = f"mark_{subject}"
    status_column = f"status_{subject}"

    target_columns.append(mark_column)

    if status_column in master_df.columns:
        target_columns.append(status_column)


target_source_df = master_df[target_columns].copy()

target_source_df["calculated_average"] = to_numeric(
    target_source_df["calculated_average"]
)

for mark_column in mark_columns:
    target_source_df[mark_column] = to_numeric(
        target_source_df[mark_column]
    )


integrity_issues = []

for subject in subject_slugs:
    mark_column = f"mark_{subject}"
    status_column = f"status_{subject}"

    if status_column not in target_source_df.columns:
        continue

    status = (
        target_source_df[status_column]
        .astype("string")
        .str.strip()
        .str.upper()
    )

    mark = target_source_df[mark_column]

    present_missing_mark = int(
        (
            status.eq("PRESENT")
            & mark.isna()
        ).sum()
    )

    numeric_non_present = int(
        (
            mark.notna()
            & ~status.eq("PRESENT")
        ).sum()
    )

    if present_missing_mark or numeric_non_present:
        integrity_issues.append(
            {
                "subject": subject,
                "present_missing_mark": present_missing_mark,
                "numeric_mark_non_present": numeric_non_present,
            }
        )

if integrity_issues:
    issue_df = pd.DataFrame(integrity_issues)
    raise RuntimeError(
        "\nSubject target integrity validation failed.\n\n"
        f"{issue_df.to_string(index=False)}"
    )

print("\nSubject target integrity validation: PASSED")


merged_df = features_df.merge(
    target_source_df,
    on=["student_id", "year", "term", "academic_period"],
    how="left",
    validate="one_to_one",
)

if len(merged_df) != len(features_df):
    raise RuntimeError(
        "\nUnexpected row-count change after target merge."
    )


continuity_df = merged_df[
    ["student_id", "academic_period", "has_prior_history"]
].copy()

continuity_df["academic_sequence"] = (
    continuity_df["academic_period"]
    .map(ACADEMIC_PERIOD_SEQUENCE)
)

unmapped_periods = sorted(
    continuity_df.loc[
        continuity_df["academic_sequence"].isna(),
        "academic_period",
    ]
    .dropna()
    .unique()
    .tolist()
)

if unmapped_periods:
    raise RuntimeError(
        "\nAcademic periods missing from ACADEMIC_PERIOD_SEQUENCE:\n"
        f"{unmapped_periods}"
    )

continuity_df = (
    continuity_df
    .sort_values(["student_id", "academic_sequence"])
    .reset_index(drop=True)
)

continuity_df["previous_academic_period"] = (
    continuity_df
    .groupby("student_id", sort=False)["academic_period"]
    .shift(1)
)

continuity_df["previous_academic_sequence"] = (
    continuity_df
    .groupby("student_id", sort=False)["academic_sequence"]
    .shift(1)
)

continuity_df["academic_gap"] = (
    continuity_df["academic_sequence"]
    - continuity_df["previous_academic_sequence"]
)

history_continuity = continuity_df[
    continuity_df["has_prior_history"] == 1
].copy()

non_consecutive_df = history_continuity[
    history_continuity["academic_gap"] != 1
].copy()

if len(non_consecutive_df):
    issue_text = non_consecutive_df[
        [
            "student_id",
            "previous_academic_period",
            "academic_period",
            "academic_gap",
        ]
    ].to_string(index=False)

    raise RuntimeError(
        "\nNon-consecutive historical prediction records detected.\n\n"
        f"{issue_text}"
    )

print(
    f"\nConsecutive historical transitions validated: "
    f"{len(history_continuity)}"
)
print("Non-consecutive historical transitions: 0")


forecast_df = merged_df[
    merged_df["has_prior_history"] == 1
].copy()

forecast_df["dataset_split"] = (
    forecast_df["academic_period"]
    .map(assign_split)
)

unexpected_periods = sorted(
    forecast_df.loc[
        forecast_df["dataset_split"] == "NOT_MODELLED",
        "academic_period",
    ]
    .dropna()
    .unique()
    .tolist()
)

if unexpected_periods:
    raise RuntimeError(
        "\nUnexpected target periods detected:\n"
        f"{unexpected_periods}"
    )

expected_transition_counts = {
    "TRAIN": 577,
    "VALIDATION": 218,
    "FINAL_TEST": 268,
}

for split, expected_count in expected_transition_counts.items():
    actual_count = int(
        (forecast_df["dataset_split"] == split).sum()
    )

    if actual_count != expected_count:
        raise RuntimeError(
            f"\nUnexpected {split} forecasting row count.\n"
            f"Expected: {expected_count}\n"
            f"Actual: {actual_count}"
        )


overall_df = forecast_df.rename(
    columns={
        "calculated_average": "target_calculated_average",
    }
).copy()

raw_target_columns = []
for subject in subject_slugs:
    raw_target_columns.extend(
        [f"mark_{subject}", f"status_{subject}"]
    )

raw_target_columns = [
    column
    for column in raw_target_columns
    if column in overall_df.columns
]

overall_df = overall_df.drop(
    columns=raw_target_columns
)

overall_df["target_available"] = (
    overall_df["target_calculated_average"]
    .notna()
    .astype(int)
)

overall_model_df = overall_df[
    overall_df["target_available"] == 1
].copy()


# One row = student x target term x target subject.

subject_frames = []

global_history_columns = [
    "history_term_count",
    "has_prior_history",
    "prev_calculated_average",
    "historical_mean_average",
    "historical_std_average",
    "performance_trend",
    "prev_numeric_subject_count",
    "prev_absent_subject_count",
    "prev_not_taken_subject_count",
    "historical_mean_absent_subject_count",
    "historical_mean_not_taken_subject_count",
    "prev_attendance_pct",
    "historical_mean_attendance_pct",
    "historical_std_attendance_pct",
    "attendance_trend",
    "prior_attendance_observation_count",
    "prev_days_absent",
    "historical_mean_days_absent",
    "prev_attendance_coverage_pct",
    "prev_attendance_available",
]

subject_base_columns = [
    "student_id",
    "year",
    "term",
    "academic_period",
    "period_order",
    "grade",
    "class",
    "academic_stage",
    "dataset_split",
]

for subject in subject_slugs:
    target_mark_column = f"mark_{subject}"
    target_status_column = f"status_{subject}"

    if target_status_column not in forecast_df.columns:
        continue

    status = (
        forecast_df[target_status_column]
        .astype("string")
        .str.strip()
        .str.upper()
    )

    mark = to_numeric(
        forecast_df[target_mark_column]
    )

    eligible_mask = (
        status.eq("PRESENT")
        & mark.notna()
    )

    if not eligible_mask.any():
        continue

    subject_df = forecast_df.loc[
        eligible_mask,
        subject_base_columns,
    ].copy()

    for column in global_history_columns:
        subject_df[column] = (
            forecast_df.loc[
                eligible_mask,
                column,
            ].values
        )

    subject_df["target_subject"] = subject
    subject_df["target_mark"] = mark.loc[
        eligible_mask
    ].values

    subject_df["prev_subject_mark"] = (
        forecast_df.loc[
            eligible_mask,
            f"prev_mark_{subject}",
        ].values
    )

    subject_df["historical_mean_subject_mark"] = (
        forecast_df.loc[
            eligible_mask,
            f"historical_mean_mark_{subject}",
        ].values
    )

    subject_df["historical_subject_mark_count"] = (
        forecast_df.loc[
            eligible_mask,
            f"historical_mark_count_{subject}",
        ].values
    )

    subject_df["subject_mark_trend"] = (
        forecast_df.loc[
            eligible_mask,
            f"mark_trend_{subject}",
        ].values
    )

    subject_df["prev_subject_absent"] = (
        forecast_df.loc[
            eligible_mask,
            f"prev_absent_{subject}",
        ].values
    )

    subject_df["prev_subject_not_taken"] = (
        forecast_df.loc[
            eligible_mask,
            f"prev_not_taken_{subject}",
        ].values
    )

    recorded_column = f"prev_recorded_{subject}"

    if recorded_column not in forecast_df.columns:
        raise RuntimeError(
            f"Required historical feature missing: {recorded_column}"
        )

    subject_df["prev_subject_recorded"] = (
        forecast_df.loc[
            eligible_mask,
            recorded_column,
        ].values
    )

    subject_frames.append(subject_df)


if subject_frames:
    subject_model_df = pd.concat(
        subject_frames,
        ignore_index=True,
    )
else:
    subject_model_df = pd.DataFrame()


# PRIVACY VALIDATION

forbidden_direct_identifiers = {
    "student_name",
    "student_no",
    "admission_no",
    "index_no",
    "canonical_index_no",
    "persistent_student_id",
    "source_file",
    "source_sheet",
    "source_excel_row",
}

for label, dataframe in [
    ("overall_prediction_dataset", overall_model_df),
    ("subject_prediction_dataset", subject_model_df),
]:
    leaked = sorted(
        forbidden_direct_identifiers.intersection(
            dataframe.columns
        )
    )

    if leaked:
        raise RuntimeError(
            f"\nPrivacy validation failed for {label}:\n"
            f"{leaked}"
        )


# TARGET LEAKAGE VALIDATION

remaining_raw_target_columns = [
    column
    for column in overall_model_df.columns
    if column in mark_columns or column in status_columns
]

if remaining_raw_target_columns:
    raise RuntimeError(
        "\nRaw current-term subject targets remain inside "
        "overall prediction dataset:\n"
        f"{remaining_raw_target_columns}"
    )


overall_model_df = (
    overall_model_df
    .sort_values(
        ["academic_period", "grade", "class", "student_id"]
    )
    .reset_index(drop=True)
)

if not subject_model_df.empty:
    subject_model_df = (
        subject_model_df
        .sort_values(
            [
                "academic_period",
                "grade",
                "class",
                "student_id",
                "target_subject",
            ]
        )
        .reset_index(drop=True)
    )


overall_model_df.to_csv(
    OVERALL_OUTPUT_FILE,
    index=False,
)

subject_model_df.to_csv(
    SUBJECT_OUTPUT_FILE,
    index=False,
)


# Only availability counts are recorded for the final test target.

if not subject_model_df.empty:
    subject_summary_df = (
        subject_model_df
        .groupby(
            ["dataset_split", "target_subject"],
            dropna=False,
        )
        .size()
        .reset_index(name="target_rows")
    )
else:
    subject_summary_df = pd.DataFrame(
        columns=[
            "dataset_split",
            "target_subject",
            "target_rows",
        ]
    )

subject_summary_df.to_csv(
    SUBJECT_SUMMARY_FILE,
    index=False,
)

split_order = [
    "TRAIN",
    "VALIDATION",
    "FINAL_TEST",
]

summary_records = []

for split in split_order:
    overall_split = overall_model_df[
        overall_model_df["dataset_split"] == split
    ]

    if not subject_model_df.empty:
        subject_split = subject_model_df[
            subject_model_df["dataset_split"] == split
        ]
    else:
        subject_split = pd.DataFrame()

    summary_records.append(
        {
            "dataset_split": split,
            "forecast_transition_rows": int(
                (forecast_df["dataset_split"] == split).sum()
            ),
            "overall_target_rows": len(overall_split),
            "overall_unique_students": (
                overall_split["student_id"].nunique()
                if len(overall_split)
                else 0
            ),
            "subject_target_rows": len(subject_split),
            "subject_unique_students": (
                subject_split["student_id"].nunique()
                if len(subject_split)
                else 0
            ),
            "subjects_with_targets": (
                subject_split["target_subject"].nunique()
                if len(subject_split)
                else 0
            ),
        }
    )

summary_df = pd.DataFrame(summary_records)
summary_df.to_csv(
    SUMMARY_FILE,
    index=False,
)


print("\n" + "=" * 110)
print("TARGET CONSTRUCTION COMPLETE")
print("=" * 110)

print(
    f"\nForecast-eligible consecutive-term rows: "
    f"{len(forecast_df)}"
)

print("\nTemporal split:")
for split in split_order:
    rows = int(
        (forecast_df["dataset_split"] == split).sum()
    )
    print(f"  {split:<12}: {rows}")

print("\nOverall-performance targets:")
for split in split_order:
    rows = int(
        (overall_model_df["dataset_split"] == split).sum()
    )
    print(f"  {split:<12}: {rows}")

print(
    f"\nTotal overall prediction rows: "
    f"{len(overall_model_df)}"
)

print("\nSubject-mark targets:")
for split in split_order:
    if subject_model_df.empty:
        rows = 0
        subjects = 0
    else:
        rows = int(
            (subject_model_df["dataset_split"] == split).sum()
        )
        subjects = int(
            subject_model_df.loc[
                subject_model_df["dataset_split"] == split,
                "target_subject",
            ].nunique()
        )

    print(
        f"  {split:<12}: {rows} rows, {subjects} subjects"
    )

print(
    f"\nTotal subject prediction rows: "
    f"{len(subject_model_df)}"
)

print("\nTarget semantics:")
print("  Overall target: target-term calculated average")
print("  Subject target: target-term numeric mark where status=PRESENT")
print("  ABSENT targets converted to zero: NO")
print("  NOT_TAKEN targets converted to zero: NO")
print("  Missing targets fabricated: NO")

print("\nSubject-history semantics:")
print("  prev_subject_recorded carried into pooled subject dataset: YES")
print("  Previous ABSENT remains separate: YES")
print("  Previous NOT_TAKEN remains separate: YES")

print("\nTemporal integrity:")
print(f"  Consecutive transitions: {len(history_continuity)}")
print("  Non-consecutive history rows: 0")
print("  Train target periods: 2025_T2, 2025_T3")
print("  Validation target period: 2026_T1")
print("  Final test target period: 2026_T2")
print("  Final-test target distributions inspected: NO")

print("\nPrivacy:")
print("  Direct student identifiers: 0")
print("  Pseudonymised student_id retained only as row key")

print("\nOutputs:")
print(f"  {OVERALL_OUTPUT_FILE}")
print(f"  {SUBJECT_OUTPUT_FILE}")

print("\nReports:")
print(f"  {SUMMARY_FILE}")
print(f"  {SUBJECT_SUMMARY_FILE}")
