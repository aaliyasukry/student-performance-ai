from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "anonymized"
    / "master_research_dataset.csv"
)

REPORTS_DIR = (
    PROJECT_ROOT
    / "reports"
    / "eda"
)

FIGURES_DIR = (
    PROJECT_ROOT
    / "figures"
    / "eda"
)

REPORTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)

FIGURES_DIR.mkdir(
    parents=True,
    exist_ok=True
)


HOLDOUT_YEAR = 2026
HOLDOUT_TERM = 2


def clean_numeric(series):
    """Convert a pandas Series to numeric while preserving invalid / unavailable values as NaN."""

    return pd.to_numeric(
        series,
        errors="coerce"
    )


def save_figure(filename):
    """Save a dissertation-quality figure."""

    path = (
        FIGURES_DIR
        / filename
    )

    plt.tight_layout()

    plt.savefig(
        path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    return path


def safe_summary(series):
    """Generate descriptive statistics for a numeric variable."""

    values = clean_numeric(
        series
    ).dropna()

    if len(values) == 0:

        return {
            "count": 0,
            "mean": np.nan,
            "std": np.nan,
            "min": np.nan,
            "q1": np.nan,
            "median": np.nan,
            "q3": np.nan,
            "max": np.nan,
        }

    return {
        "count":
            len(values),

        "mean":
            values.mean(),

        "std":
            values.std(),

        "min":
            values.min(),

        "q1":
            values.quantile(
                0.25
            ),

        "median":
            values.median(),

        "q3":
            values.quantile(
                0.75
            ),

        "max":
            values.max(),
    }


if not DATA_FILE.exists():

    raise FileNotFoundError(
        f"Master research dataset not found:\n"
        f"{DATA_FILE}"
    )


df = pd.read_csv(
    DATA_FILE
)


print(
    "=" * 110
)

print(
    "STUDENT PERFORMANCE AI - "
    "EXPLORATORY DATA ANALYSIS"
)

print(
    "=" * 110
)


for column in [
    "year",
    "term",
    "grade",
]:

    df[
        column
    ] = pd.to_numeric(
        df[
            column
        ],
        errors="raise"
    ).astype(int)


# 2026 T2 is deliberately excluded from exploratory model-development analysis.

holdout_mask = (
    (df["year"] == HOLDOUT_YEAR)
    &
    (df["term"] == HOLDOUT_TERM)
)


development_df = (
    df[
        ~holdout_mask
    ]
    .copy()
)


holdout_df = (
    df[
        holdout_mask
    ]
    .copy()
)


print(
    f"\nFull master dataset rows: "
    f"{len(df)}"
)


print(
    f"Development / EDA rows: "
    f"{len(development_df)}"
)


print(
    f"Held-out 2026 T2 rows: "
    f"{len(holdout_df)}"
)


mark_columns = sorted(
    [
        column
        for column
        in development_df.columns
        if column.startswith(
            "mark_"
        )
    ]
)


status_columns = sorted(
    [
        column
        for column
        in development_df.columns
        if column.startswith(
            "status_"
        )
    ]
)


print(
    f"\nSubject mark columns: "
    f"{len(mark_columns)}"
)


print(
    f"Subject status columns: "
    f"{len(status_columns)}"
)


overview_records = [
    {
        "metric":
            "full_dataset_rows",

        "value":
            len(df),
    },
    {
        "metric":
            "full_unique_students",

        "value":
            df[
                "student_id"
            ].nunique(),
    },
    {
        "metric":
            "development_rows",

        "value":
            len(development_df),
    },
    {
        "metric":
            "development_unique_students",

        "value":
            development_df[
                "student_id"
            ].nunique(),
    },
    {
        "metric":
            "holdout_rows",

        "value":
            len(holdout_df),
    },
    {
        "metric":
            "holdout_unique_students",

        "value":
            holdout_df[
                "student_id"
            ].nunique(),
    },
    {
        "metric":
            "subject_count",

        "value":
            len(mark_columns),
    },
]


overview_df = pd.DataFrame(
    overview_records
)


overview_df.to_csv(
    REPORTS_DIR
    / "dataset_overview.csv",
    index=False
)


period_summary = (
    development_df
    .groupby(
        [
            "year",
            "term",
            "academic_period",
        ],
        dropna=False
    )
    .agg(
        student_term_rows=(
            "student_id",
            "size"
        ),

        unique_students=(
            "student_id",
            "nunique"
        ),

        grades=(
            "grade",
            "nunique"
        ),
    )
    .reset_index()
)


period_summary.to_csv(
    REPORTS_DIR
    / "period_summary.csv",
    index=False
)


grade_period_summary = (
    development_df
    .groupby(
        [
            "academic_period",
            "grade",
        ],
        dropna=False
    )
    .agg(
        rows=(
            "student_id",
            "size"
        ),

        unique_students=(
            "student_id",
            "nunique"
        ),

        mean_average=(
            "calculated_average",
            lambda x:
                clean_numeric(
                    x
                ).mean()
        ),

        median_average=(
            "calculated_average",
            lambda x:
                clean_numeric(
                    x
                ).median()
        ),

        mean_attendance_pct=(
            "attendance_pct",
            lambda x:
                clean_numeric(
                    x
                ).mean()
        ),
    )
    .reset_index()
)


grade_period_summary.to_csv(
    REPORTS_DIR
    / "grade_period_summary.csv",
    index=False
)


missing_records = []


for column in development_df.columns:

    missing_count = int(
        development_df[
            column
        ].isna().sum()
    )

    missing_pct = (
        missing_count
        /
        len(
            development_df
        )
        *
        100
    )

    missing_records.append(
        {
            "column":
                column,

            "missing_count":
                missing_count,

            "missing_percentage":
                round(
                    missing_pct,
                    4
                ),
        }
    )


missingness_df = (
    pd.DataFrame(
        missing_records
    )
    .sort_values(
        by=[
            "missing_percentage",
            "column",
        ],
        ascending=[
            False,
            True,
        ]
    )
    .reset_index(
        drop=True
    )
)


missingness_df.to_csv(
    REPORTS_DIR
    / "missingness_analysis.csv",
    index=False
)


performance_fields = [
    "calculated_average",
    "calculated_total",
    "numeric_subject_count",
    "absent_subject_count",
    "missing_subject_count",
    "not_taken_subject_count",
]


performance_records = []


for column in performance_fields:

    if column not in development_df.columns:
        continue

    summary = safe_summary(
        development_df[
            column
        ]
    )

    performance_records.append(
        {
            "variable":
                column,

            **summary,
        }
    )


performance_summary_df = (
    pd.DataFrame(
        performance_records
    )
)


performance_summary_df.to_csv(
    REPORTS_DIR
    / "performance_summary.csv",
    index=False
)


subject_records = []


for mark_column in mark_columns:

    subject = (
        mark_column
        .replace(
            "mark_",
            "",
            1
        )
    )

    values = clean_numeric(
        development_df[
            mark_column
        ]
    )


    numeric_count = int(
        values.notna().sum()
    )


    missing_count = int(
        values.isna().sum()
    )


    if numeric_count:

        mean_value = (
            values.mean()
        )

        median_value = (
            values.median()
        )

        std_value = (
            values.std()
        )

        min_value = (
            values.min()
        )

        max_value = (
            values.max()
        )

        pass_40_count = int(
            (
                values >= 40
            ).sum()
        )

        pass_40_pct = (
            pass_40_count
            /
            numeric_count
            *
            100
        )

    else:

        mean_value = np.nan
        median_value = np.nan
        std_value = np.nan
        min_value = np.nan
        max_value = np.nan
        pass_40_pct = np.nan


    subject_records.append(
        {
            "subject":
                subject,

            "numeric_mark_count":
                numeric_count,

            "missing_mark_count":
                missing_count,

            "mean":
                mean_value,

            "median":
                median_value,

            "std":
                std_value,

            "min":
                min_value,

            "max":
                max_value,

            "mark_ge_40_pct":
                pass_40_pct,
        }
    )


subject_summary_df = (
    pd.DataFrame(
        subject_records
    )
    .sort_values(
        by="mean",
        ascending=False,
        na_position="last"
    )
    .reset_index(
        drop=True
    )
)


subject_summary_df.to_csv(
    REPORTS_DIR
    / "subject_performance_summary.csv",
    index=False
)


status_records = []


for status_column in status_columns:

    subject = (
        status_column
        .replace(
            "status_",
            "",
            1
        )
    )


    counts = (
        development_df[
            status_column
        ]
        .fillna(
            "NO_RECORD"
        )
        .value_counts()
    )


    total = len(
        development_df
    )


    record = {
        "subject":
            subject,

        "total_student_terms":
            total,
    }


    for status in [
        "PRESENT",
        "ABSENT",
        "NOT_TAKEN",
        "MISSING",
        "INVALID",
        "NO_RECORD",
    ]:

        count = int(
            counts.get(
                status,
                0
            )
        )

        record[
            f"{status.lower()}_count"
        ] = count

        record[
            f"{status.lower()}_pct"
        ] = (
            count
            /
            total
            *
            100
        )


    status_records.append(
        record
    )


subject_status_df = (
    pd.DataFrame(
        status_records
    )
)


subject_status_df.to_csv(
    REPORTS_DIR
    / "subject_status_summary.csv",
    index=False
)


attendance_variables = [
    "expected_term_school_days",
    "observed_school_days",
    "days_present",
    "days_absent",
    "attendance_pct",
    "attendance_coverage_pct",
]


attendance_records = []


for column in attendance_variables:

    if column not in development_df.columns:
        continue

    summary = safe_summary(
        development_df[
            column
        ]
    )

    attendance_records.append(
        {
            "variable":
                column,

            **summary,
        }
    )


attendance_summary_df = (
    pd.DataFrame(
        attendance_records
    )
)


attendance_summary_df.to_csv(
    REPORTS_DIR
    / "attendance_summary.csv",
    index=False
)


attendance_quality_df = (
    development_df[
        "attendance_data_quality"
    ]
    .fillna(
        "NO_LINKED_ATTENDANCE"
    )
    .value_counts()
    .rename_axis(
        "attendance_data_quality"
    )
    .reset_index(
        name="count"
    )
)


attendance_quality_df[
    "percentage"
] = (
    attendance_quality_df[
        "count"
    ]
    /
    len(
        development_df
    )
    *
    100
)


attendance_quality_df.to_csv(
    REPORTS_DIR
    / "attendance_quality_summary.csv",
    index=False
)


correlation_candidates = [
    "calculated_average",
    "calculated_total",
    "numeric_subject_count",
    "absent_subject_count",
    "missing_subject_count",
    "not_taken_subject_count",
    "attendance_pct",
    "attendance_coverage_pct",
    "days_present",
    "days_absent",
]


correlation_candidates += (
    mark_columns
)


correlation_candidates = [
    column
    for column
    in correlation_candidates
    if column in development_df.columns
]


correlation_df = (
    development_df[
        correlation_candidates
    ]
    .apply(
        pd.to_numeric,
        errors="coerce"
    )
    .corr(
        method="spearman"
    )
)


correlation_df.to_csv(
    REPORTS_DIR
    / "spearman_correlation.csv"
)


average_values = clean_numeric(
    development_df[
        "calculated_average"
    ]
).dropna()


plt.figure(
    figsize=(
        8,
        5
    )
)


plt.hist(
    average_values,
    bins=20,
    color="#6F42C1",
    edgecolor="#4B2E83"
)


plt.xlabel(
    "Calculated Term Average"
)


plt.ylabel(
    "Student-Term Records"
)


plt.title(
    "Distribution of Student Term Performance"
)


performance_hist_path = save_figure(
    "performance_distribution.png"
)


attendance_values = clean_numeric(
    development_df[
        "attendance_pct"
    ]
).dropna()


plt.figure(
    figsize=(
        8,
        5
    )
)


plt.hist(
    attendance_values,
    bins=20,
    color="#6F42C1",
    edgecolor="#4B2E83"
)


plt.xlabel(
    "Attendance Percentage"
)


plt.ylabel(
    "Student-Term Records"
)


plt.title(
    "Distribution of Student Attendance"
)


attendance_hist_path = save_figure(
    "attendance_distribution.png"
)


grade_values = []

grade_labels = []


for grade in sorted(
    development_df[
        "grade"
    ].dropna().unique()
):

    values = clean_numeric(
        development_df.loc[
            development_df[
                "grade"
            ] == grade,
            "calculated_average"
        ]
    ).dropna()


    if len(values):

        grade_values.append(
            values
        )

        grade_labels.append(
            str(
                grade
            )
        )


plt.figure(
    figsize=(
        8,
        5
    )
)


plt.boxplot(
    grade_values,
    tick_labels=grade_labels,
    patch_artist=True,

    boxprops=dict(
        facecolor="#D8C7F0",
        edgecolor="#4B2E83",
        linewidth=1.5
    ),

    medianprops=dict(
        color="#6F42C1",
        linewidth=2
    ),

    whiskerprops=dict(
        color="#4B2E83",
        linewidth=1.5
    ),

    capprops=dict(
        color="#4B2E83",
        linewidth=1.5
    ),

    flierprops=dict(
        marker="o",
        markerfacecolor="#D8C7F0",
        markeredgecolor="#4B2E83",
        alpha=0.7
    )
)


plt.xlabel(
    "Grade"
)


plt.ylabel(
    "Calculated Term Average"
)


plt.title(
    "Student Performance by Grade"
)


performance_grade_path = save_figure(
    "performance_by_grade.png"
)


# Subjects with no numeric observations are left out of the chart (kept in the CSV reports).

subject_chart_df = (
    subject_summary_df[
        subject_summary_df[
            "numeric_mark_count"
        ] > 0
    ]
    .copy()
)


subject_chart_df = (
    subject_chart_df
    .sort_values(
        by="mean"
    )
)


plt.figure(
    figsize=(
        9,
        7
    )
)


plt.barh(
    subject_chart_df[
        "subject"
    ],
    subject_chart_df[
        "mean"
    ],
    color="#6F42C1",
    edgecolor="#4B2E83"
)


plt.xlabel(
    "Mean Mark"
)


plt.ylabel(
    "Subject"
)


plt.title(
    "Mean Subject Performance"
)


subject_performance_path = save_figure(
    "subject_mean_performance.png"
)


scatter_df = development_df[
    [
        "attendance_pct",
        "calculated_average",
    ]
].copy()


scatter_df[
    "attendance_pct"
] = clean_numeric(
    scatter_df[
        "attendance_pct"
    ]
)


scatter_df[
    "calculated_average"
] = clean_numeric(
    scatter_df[
        "calculated_average"
    ]
)


scatter_df = (
    scatter_df
    .dropna()
)


plt.figure(
    figsize=(
        8,
        5
    )
)


plt.scatter(
    scatter_df[
        "attendance_pct"
    ],
    scatter_df[
        "calculated_average"
    ],
    color="#6F42C1",
    edgecolor="#4B2E83",
    alpha=0.45
)


plt.xlabel(
    "Attendance Percentage"
)


plt.ylabel(
    "Calculated Term Average"
)


plt.title(
    "Attendance and Academic Performance"
)


attendance_performance_path = save_figure(
    "attendance_vs_performance.png"
)


period_chart = (
    development_df[
        "academic_period"
    ]
    .value_counts()
    .sort_index()
)


plt.figure(
    figsize=(
        8,
        5
    )
)


plt.bar(
    period_chart.index,
    period_chart.values,
    color="#6F42C1",
    edgecolor="#4B2E83"
)


plt.xlabel(
    "Academic Period"
)


plt.ylabel(
    "Student-Term Records"
)


plt.title(
    "Development Dataset Records by Academic Period"
)


period_distribution_path = save_figure(
    "records_by_period.png"
)


print(
    "\n"
    + "=" * 110
)

print(
    "EDA COMPLETE"
)

print(
    "=" * 110
)


print(
    f"\nDevelopment rows analysed: "
    f"{len(development_df)}"
)


print(
    f"Development unique students: "
    f"{development_df['student_id'].nunique()}"
)


print(
    f"Held-out test rows untouched: "
    f"{len(holdout_df)}"
)


print(
    f"Subjects analysed: "
    f"{len(mark_columns)}"
)


print(
    "\nPerformance:"
)


print(
    f"  Mean calculated average: "
    f"{average_values.mean():.2f}"
)


print(
    f"  Median calculated average: "
    f"{average_values.median():.2f}"
)


print(
    "\nAttendance:"
)


print(
    f"  Calculable development attendance rows: "
    f"{len(attendance_values)}"
)


print(
    f"  Mean attendance percentage: "
    f"{attendance_values.mean():.2f}"
)


print(
    "\nAggregate reports saved to:"
)

print(
    f"  {REPORTS_DIR}"
)


print(
    "\nFigures saved to:"
)

print(
    f"  {FIGURES_DIR}"
)


print(
    "\nGenerated figures:"
)


for path in [
    performance_hist_path,
    attendance_hist_path,
    performance_grade_path,
    subject_performance_path,
    attendance_performance_path,
    period_distribution_path,
]:

    print(
        f"  {path.name}"
    )


print(
    "\nIMPORTANT:"
)


print(
    "  2026 T2 was excluded from exploratory "
    "analysis and remains the final "
    "out-of-time test set."
)
