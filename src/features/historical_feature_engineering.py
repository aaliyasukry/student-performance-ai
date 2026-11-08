from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_QUALITY_DIR = PROJECT_ROOT / "reports" / "data_quality"
DATA_QUALITY_DIR.mkdir(parents=True, exist_ok=True)

INPUT_FILE = (
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

REPORTS_DIR = (
    PROJECT_ROOT
    / "reports"
)

MODEL_READY_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = (
    MODEL_READY_DIR
    / "historical_features.csv"
)

SUMMARY_FILE = (
    DATA_QUALITY_DIR
    / "feature_engineering_summary.csv"
)


HOLDOUT_YEAR = 2026
HOLDOUT_TERM = 2


def to_numeric(series):
    """Safely convert values to numeric; unavailable stays NaN."""
    return pd.to_numeric(series, errors="coerce")


def normalize_boolean(series):
    """Normalize common boolean representations to 1/0."""
    mapping = {
        "true": 1.0,
        "false": 0.0,
        "1": 1.0,
        "0": 0.0,
        "yes": 1.0,
        "no": 0.0,
    }

    return (
        series
        .astype(str)
        .str.strip()
        .str.lower()
        .map(mapping)
    )


def previous_value(df, column):
    """Immediately previous observed value for the same student."""
    return (
        df
        .groupby("student_id", sort=False)[column]
        .shift(1)
    )


def historical_mean(df, column):
    """Mean of all PRIOR observed values for each student."""
    return (
        df
        .groupby("student_id", sort=False)[column]
        .transform(
            lambda series:
                series
                .shift(1)
                .expanding(min_periods=1)
                .mean()
        )
    )


def historical_std(df, column):
    """Standard deviation of PRIOR observations only."""
    return (
        df
        .groupby("student_id", sort=False)[column]
        .transform(
            lambda series:
                series
                .shift(1)
                .expanding(min_periods=2)
                .std()
        )
    )


def historical_count(df, column):
    """Count non-missing PRIOR observations."""
    return (
        df
        .groupby("student_id", sort=False)[column]
        .transform(
            lambda series:
                series
                .notna()
                .shift(1, fill_value=False)
                .cumsum()
        )
        .astype(int)
    )


def previous_status(df, column):
    """Immediately previous subject status."""
    return (
        df
        .groupby("student_id", sort=False)[column]
        .shift(1)
    )


def previous_status_flag(df, column, status):
    """1 when the previous term had the requested status."""
    return (
        previous_status(df, column)
        .eq(status)
        .astype(int)
    )


def previous_subject_recorded(df, column):
    """1 when the previous term has a source record for the subject, else 0."""
    prior = previous_status(df, column)

    normalized = (
        prior
        .astype("string")
        .str.strip()
        .str.upper()
    )

    return (
        prior.notna()
        & normalized.ne("NO_RECORD")
    ).astype(int)


def assert_unique_student_term(df, label):
    """Ensure exactly one row per student/year/term."""
    duplicate_mask = df.duplicated(
        subset=["student_id", "year", "term"],
        keep=False,
    )

    if duplicate_mask.any():
        duplicate_rows = df.loc[
            duplicate_mask,
            ["student_id", "year", "term", "grade", "class"],
        ]

        raise RuntimeError(
            f"\nDuplicate student-term rows detected in {label}:\n\n"
            f"{duplicate_rows.to_string(index=False)}"
        )


if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Master research dataset not found:\n{INPUT_FILE}"
    )


df = pd.read_csv(INPUT_FILE)

print("=" * 110)
print("STUDENT PERFORMANCE AI - HISTORICAL FEATURE ENGINEERING")
print("=" * 110)


required_columns = [
    "student_id",
    "year",
    "term",
    "academic_period",
    "period_order",
    "grade",
    "class",
    "academic_stage",
    "calculated_average",
    "numeric_subject_count",
    "absent_subject_count",
    "not_taken_subject_count",
    "attendance_pct",
    "attendance_coverage_pct",
    "days_absent",
    "attendance_available",
]

missing_required = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_required:
    raise ValueError(
        "\nRequired columns missing from master dataset:\n"
        f"{missing_required}"
    )


for column in ["year", "term", "period_order", "grade"]:
    df[column] = pd.to_numeric(
        df[column],
        errors="raise",
    ).astype(int)

numeric_columns = [
    "calculated_average",
    "numeric_subject_count",
    "absent_subject_count",
    "not_taken_subject_count",
    "attendance_pct",
    "attendance_coverage_pct",
    "days_present",
    "days_absent",
]

for column in numeric_columns:
    if column in df.columns:
        df[column] = to_numeric(df[column])


df["_attendance_available_numeric"] = normalize_boolean(
    df["attendance_available"]
)

assert_unique_student_term(
    df,
    "master_research_dataset",
)


df = (
    df
    .sort_values(["student_id", "period_order"])
    .reset_index(drop=True)
)

period_difference = (
    df
    .groupby("student_id", sort=False)["period_order"]
    .diff()
)

if period_difference.dropna().le(0).any():
    raise RuntimeError(
        "Non-increasing academic periods detected for one or more students."
    )


mark_columns = sorted(
    column
    for column in df.columns
    if column.startswith("mark_")
)

status_columns = sorted(
    column
    for column in df.columns
    if column.startswith("status_")
)

subject_slugs = [
    column.replace("mark_", "", 1)
    for column in mark_columns
]

print(f"\nStudent-term rows: {len(df)}")
print(f"Unique students: {df['student_id'].nunique()}")
print(f"Subjects detected: {len(subject_slugs)}")

for column in mark_columns:
    df[column] = to_numeric(df[column])


metadata_columns = [
    "student_id",
    "year",
    "term",
    "academic_period",
    "period_order",
    "grade",
    "class",
    "academic_stage",
]

metadata_df = df[metadata_columns].copy()


features = {}

# History depth
features["history_term_count"] = (
    df.groupby("student_id", sort=False).cumcount()
)
features["has_prior_history"] = (
    features["history_term_count"].gt(0).astype(int)
)

# Overall performance history
features["prev_calculated_average"] = previous_value(
    df, "calculated_average"
)
features["historical_mean_average"] = historical_mean(
    df, "calculated_average"
)
features["historical_std_average"] = historical_std(
    df, "calculated_average"
)

previous_average_2 = (
    df
    .groupby("student_id", sort=False)["calculated_average"]
    .shift(2)
)

features["performance_trend"] = (
    features["prev_calculated_average"]
    - previous_average_2
)

# Exam participation history
features["prev_numeric_subject_count"] = previous_value(
    df, "numeric_subject_count"
)
features["prev_absent_subject_count"] = previous_value(
    df, "absent_subject_count"
)
features["prev_not_taken_subject_count"] = previous_value(
    df, "not_taken_subject_count"
)
features["historical_mean_absent_subject_count"] = historical_mean(
    df, "absent_subject_count"
)
features["historical_mean_not_taken_subject_count"] = historical_mean(
    df, "not_taken_subject_count"
)

# Attendance history
features["prev_attendance_pct"] = previous_value(
    df, "attendance_pct"
)
features["historical_mean_attendance_pct"] = historical_mean(
    df, "attendance_pct"
)
features["historical_std_attendance_pct"] = historical_std(
    df, "attendance_pct"
)

previous_attendance_2 = (
    df
    .groupby("student_id", sort=False)["attendance_pct"]
    .shift(2)
)

features["attendance_trend"] = (
    features["prev_attendance_pct"]
    - previous_attendance_2
)

features["prior_attendance_observation_count"] = historical_count(
    df, "attendance_pct"
)
features["prev_days_absent"] = previous_value(
    df, "days_absent"
)
features["historical_mean_days_absent"] = historical_mean(
    df, "days_absent"
)
features["prev_attendance_coverage_pct"] = previous_value(
    df, "attendance_coverage_pct"
)
features["prev_attendance_available"] = previous_value(
    df, "_attendance_available_numeric"
)


# Per-subject history: previous mark, mean, count, trend and status flags.

for subject in subject_slugs:
    mark_column = f"mark_{subject}"
    status_column = f"status_{subject}"

    previous_mark = previous_value(
        df,
        mark_column,
    )

    features[f"prev_mark_{subject}"] = previous_mark
    features[f"historical_mean_mark_{subject}"] = historical_mean(
        df,
        mark_column,
    )
    features[f"historical_mark_count_{subject}"] = historical_count(
        df,
        mark_column,
    )

    previous_mark_2 = (
        df
        .groupby("student_id", sort=False)[mark_column]
        .shift(2)
    )

    features[f"mark_trend_{subject}"] = (
        previous_mark
        - previous_mark_2
    )

    if status_column in df.columns:
        features[f"prev_absent_{subject}"] = previous_status_flag(
            df,
            status_column,
            "ABSENT",
        )
        features[f"prev_not_taken_{subject}"] = previous_status_flag(
            df,
            status_column,
            "NOT_TAKEN",
        )
        features[f"prev_recorded_{subject}"] = previous_subject_recorded(
            df,
            status_column,
        )
    else:
        features[f"prev_absent_{subject}"] = pd.Series(
            0,
            index=df.index,
            dtype=int,
        )
        features[f"prev_not_taken_{subject}"] = pd.Series(
            0,
            index=df.index,
            dtype=int,
        )
        features[f"prev_recorded_{subject}"] = pd.Series(
            0,
            index=df.index,
            dtype=int,
        )


historical_feature_block = pd.DataFrame(
    features,
    index=df.index,
)

feature_df = pd.concat(
    [metadata_df, historical_feature_block],
    axis=1,
)


# PRIVACY VALIDATION

forbidden_identifier_columns = {
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

privacy_leaks = sorted(
    forbidden_identifier_columns.intersection(feature_df.columns)
)

if privacy_leaks:
    raise RuntimeError(
        "\nPrivacy validation failed.\n"
        "Direct identifiers detected:\n"
        f"{privacy_leaks}"
    )


# CURRENT-TERM LEAKAGE VALIDATION

forbidden_current_outcomes = {
    "calculated_average",
    "calculated_total",
    "source_average",
    "source_total",
    "source_rank",
    "attendance_pct",
    "days_present",
    "days_absent",
    "attendance_coverage_pct",
}

current_outcome_leaks = sorted(
    forbidden_current_outcomes.intersection(feature_df.columns)
)

if current_outcome_leaks:
    raise RuntimeError(
        "\nPotential target leakage detected.\n"
        "Current-term outcome columns present:\n"
        f"{current_outcome_leaks}"
    )

# Only the ORIGINAL current-term mark/status columns are forbidden.
direct_subject_leaks = sorted(
    set(mark_columns + status_columns).intersection(feature_df.columns)
)

if direct_subject_leaks:
    raise RuntimeError(
        "\nPotential subject-level target leakage detected.\n"
        "Current-term mark/status columns present:\n"
        f"{direct_subject_leaks}"
    )


first_history_mask = (
    feature_df["history_term_count"] == 0
)

first_rows_with_previous_average = int(
    feature_df.loc[
        first_history_mask,
        "prev_calculated_average",
    ]
    .notna()
    .sum()
)

if first_rows_with_previous_average:
    raise RuntimeError(
        "\nHistorical leakage validation failed.\n"
        "A student's first observed term contains a previous calculated average."
    )


feature_columns = [
    column
    for column in feature_df.columns
    if column not in metadata_columns
]

holdout_mask = (
    (feature_df["year"] == HOLDOUT_YEAR)
    & (feature_df["term"] == HOLDOUT_TERM)
)

holdout_feature_df = feature_df[holdout_mask]

holdout_with_history = int(
    holdout_feature_df["has_prior_history"].sum()
)

holdout_without_history = (
    len(holdout_feature_df)
    - holdout_with_history
)


feature_df = (
    feature_df
    .sort_values(
        ["period_order", "grade", "class", "student_id"]
    )
    .reset_index(drop=True)
)

if len(feature_df) != len(df):
    raise RuntimeError(
        "\nHistorical feature row count changed.\n"
        f"Input rows: {len(df)}\n"
        f"Output rows: {len(feature_df)}"
    )

assert_unique_student_term(
    feature_df,
    "historical_features",
)

rows_with_history = int(
    feature_df["has_prior_history"].sum()
)

rows_without_history = (
    len(feature_df)
    - rows_with_history
)

expected_rows_without_history = df["student_id"].nunique()

if rows_without_history != expected_rows_without_history:
    raise RuntimeError(
        "\nUnexpected first-history row count.\n"
        f"Expected: {expected_rows_without_history}\n"
        f"Actual: {rows_without_history}"
    )


development_mask = ~(
    (feature_df["year"] == HOLDOUT_YEAR)
    & (feature_df["term"] == HOLDOUT_TERM)
)

development_feature_df = feature_df[development_mask]


feature_df.to_csv(
    OUTPUT_FILE,
    index=False,
)

summary_df = pd.DataFrame(
    [
        {
            "input_student_term_rows": len(df),
            "unique_students": df["student_id"].nunique(),
            "subjects_processed": len(subject_slugs),
            "historical_feature_columns": len(feature_columns),
            "previous_subject_recorded_features": len(subject_slugs),
            "rows_with_prior_history": rows_with_history,
            "rows_without_prior_history": rows_without_history,
            "development_rows": len(development_feature_df),
            "holdout_2026_t2_rows": len(holdout_feature_df),
            "holdout_rows_with_prior_history": holdout_with_history,
            "holdout_rows_without_prior_history": holdout_without_history,
            "direct_identifier_columns": 0,
            "current_term_outcome_columns": 0,
            "current_term_subject_columns": 0,
        }
    ]
)

summary_df.to_csv(
    SUMMARY_FILE,
    index=False,
)


print("\n" + "=" * 110)
print("HISTORICAL FEATURE ENGINEERING COMPLETE")
print("=" * 110)

print(f"\nInput student-term rows: {len(df)}")
print(f"Unique students: {df['student_id'].nunique()}")
print(f"Subjects processed: {len(subject_slugs)}")
print(f"Historical feature columns: {len(feature_columns)}")
print(
    f"Previous-subject-recorded indicators: "
    f"{len(subject_slugs)}"
)

print(f"\nRows with prior history: {rows_with_history}")
print(f"Rows without prior history: {rows_without_history}")
print(f"\nDevelopment rows: {len(development_feature_df)}")

print("\n2026 T2 holdout:")
print(f"  Rows: {len(holdout_feature_df)}")
print(f"  Rows with prior history: {holdout_with_history}")
print(f"  Rows without prior history: {holdout_without_history}")

print("\nLeakage validation:")
print("  Current-term calculated average: NOT INCLUDED")
print("  Current-term attendance: NOT INCLUDED")
print("  Current-term subject marks: NOT INCLUDED")
print("  Current-term subject statuses: NOT INCLUDED")
print("  Historical mark trends: VALID PRIOR-TERM FEATURES")

print("\nMissingness semantics:")
print("  Previous subject ABSENT: EXPLICIT FLAG")
print("  Previous subject NOT_TAKEN: EXPLICIT FLAG")
print("  Previous subject source record available: EXPLICIT FLAG")
print("  Previous subject NO_RECORD: PRESERVED AS NOT RECORDED")

print("\nPrivacy validation:")
print("  Direct student identifiers: 0")

print("\nOutput:")
print(f"  {OUTPUT_FILE}")

print("\nSummary:")
print(f"  {SUMMARY_FILE}")
