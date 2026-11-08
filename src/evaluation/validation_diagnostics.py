from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


RANDOM_STATE = 42
BOOTSTRAP_ITERATIONS = 3000

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PRIVATE_DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "model_ready"
)

REPORT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "evaluation"
)

FIGURE_DIR = (
    PROJECT_ROOT
    / "figures"
    / "evaluation"
    / "validation"
)

MODEL_REPORT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "modeling"
)

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

FIGURE_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


OVERALL_ACADEMIC_PREDICTIONS = (
    PRIVATE_DATA_DIR
    / "academic_overall_academic_validation_predictions.csv"
)

OVERALL_ATTENDANCE_PREDICTIONS = (
    PRIVATE_DATA_DIR
    / "attendance_overall_attendance_validation_predictions.csv"
)

SUBJECT_ACADEMIC_PREDICTIONS = (
    PRIVATE_DATA_DIR
    / "academic_subject_academic_validation_predictions.csv"
)

SUBJECT_ATTENDANCE_PREDICTIONS = (
    PRIVATE_DATA_DIR
    / "attendance_subject_attendance_validation_predictions.csv"
)


OVERALL_ACADEMIC_METRICS = (
    MODEL_REPORT_DIR
    / "overall_academic_validation_metrics.csv"
)

OVERALL_ATTENDANCE_METRICS = (
    MODEL_REPORT_DIR
    / "overall_attendance_validation_metrics.csv"
)

SUBJECT_ACADEMIC_METRICS = (
    MODEL_REPORT_DIR
    / "subject_academic_validation_metrics.csv"
)

SUBJECT_ATTENDANCE_METRICS = (
    MODEL_REPORT_DIR
    / "subject_attendance_validation_metrics.csv"
)


OVERALL_DIAGNOSTICS_OUTPUT = (
    REPORT_DIR
    / "overall_validation_diagnostics.csv"
)

SUBJECT_DIAGNOSTICS_OUTPUT = (
    REPORT_DIR
    / "subject_validation_diagnostics.csv"
)

OVERALL_GRADE_OUTPUT = (
    REPORT_DIR
    / "overall_grade_validation_diagnostics.csv"
)

SUBJECT_GRADE_OUTPUT = (
    REPORT_DIR
    / "subject_grade_validation_diagnostics.csv"
)

SUBJECT_LEVEL_OUTPUT = (
    REPORT_DIR
    / "subject_random_forest_validation_by_subject.csv"
)

BOOTSTRAP_OUTPUT = (
    REPORT_DIR
    / "attendance_paired_bootstrap.csv"
)

GENERALIZATION_OUTPUT = (
    REPORT_DIR
    / "generalization_gap.csv"
)

SUMMARY_OUTPUT = (
    REPORT_DIR
    / "model_selection_evidence.json"
)


MODELS = {
    "Linear Regression":
        "prediction_linear_regression",

    "Random Forest":
        "prediction_random_forest",

    "XGBoost":
        "prediction_xgboost",
}


def regression_diagnostics(
    actual,
    predicted,
):
    """Detailed validation regression diagnostics."""

    actual = np.asarray(
        actual,
        dtype=float,
    )

    predicted = np.asarray(
        predicted,
        dtype=float,
    )

    residual = (
        actual
        - predicted
    )

    absolute_error = np.abs(
        residual
    )

    mae = mean_absolute_error(
        actual,
        predicted,
    )

    rmse = float(
        np.sqrt(
            mean_squared_error(
                actual,
                predicted,
            )
        )
    )

    r2 = r2_score(
        actual,
        predicted,
    )

    return {
        "rows":
            len(actual),

        "mae":
            float(mae),

        "rmse":
            float(rmse),

        "r2":
            float(r2),

        "mean_residual_actual_minus_predicted":
            float(
                residual.mean()
            ),

        "median_absolute_error":
            float(
                np.median(
                    absolute_error
                )
            ),

        "p75_absolute_error":
            float(
                np.percentile(
                    absolute_error,
                    75,
                )
            ),

        "p90_absolute_error":
            float(
                np.percentile(
                    absolute_error,
                    90,
                )
            ),

        "max_absolute_error":
            float(
                absolute_error.max()
            ),

        "within_5_marks_pct":
            float(
                (
                    absolute_error
                    <= 5
                ).mean()
                * 100
            ),

        "within_10_marks_pct":
            float(
                (
                    absolute_error
                    <= 10
                ).mean()
                * 100
            ),

        "within_15_marks_pct":
            float(
                (
                    absolute_error
                    <= 15
                ).mean()
                * 100
            ),

        "predictions_below_0":
            int(
                (
                    predicted
                    < 0
                ).sum()
            ),

        "predictions_above_100":
            int(
                (
                    predicted
                    > 100
                ).sum()
            ),
    }


def bootstrap_paired_mae_difference(
    actual,
    academic_prediction,
    attendance_prediction,
    iterations=BOOTSTRAP_ITERATIONS,
    random_state=RANDOM_STATE,
):
    """Paired bootstrap of the MAE reduction from adding attendance (positive = attendance helped)."""

    actual = np.asarray(
        actual,
        dtype=float,
    )

    academic_prediction = np.asarray(
        academic_prediction,
        dtype=float,
    )

    attendance_prediction = np.asarray(
        attendance_prediction,
        dtype=float,
    )

    academic_error = np.abs(
        actual
        - academic_prediction
    )

    attendance_error = np.abs(
        actual
        - attendance_prediction
    )

    paired_difference = (
        academic_error
        - attendance_error
    )

    observed_difference = float(
        paired_difference.mean()
    )

    n = len(
        paired_difference
    )

    rng = np.random.default_rng(
        random_state
    )

    bootstrap_means = np.empty(
        iterations,
        dtype=float,
    )

    batch_size = 200

    position = 0

    while position < iterations:

        current_batch = min(
            batch_size,
            iterations - position,
        )

        sample_indices = rng.integers(
            low=0,
            high=n,
            size=(
                current_batch,
                n,
            ),
        )

        bootstrap_means[
            position:
            position + current_batch
        ] = (
            paired_difference[
                sample_indices
            ]
            .mean(
                axis=1
            )
        )

        position += current_batch

    lower_ci = float(
        np.percentile(
            bootstrap_means,
            2.5,
        )
    )

    upper_ci = float(
        np.percentile(
            bootstrap_means,
            97.5,
        )
    )

    academic_mae = float(
        academic_error.mean()
    )

    attendance_mae = float(
        attendance_error.mean()
    )

    improvement_pct = (
        observed_difference
        / academic_mae
        * 100
    )

    if lower_ci > 0:

        evidence = (
            "ATTENDANCE_IMPROVEMENT_ROBUST"
        )

    elif upper_ci < 0:

        evidence = (
            "ATTENDANCE_WORSE_ROBUST"
        )

    else:

        evidence = (
            "NO_CLEAR_PAIRED_ADVANTAGE"
        )

    return {
        "rows":
            n,

        "academic_only_mae":
            academic_mae,

        "academic_plus_attendance_mae":
            attendance_mae,

        "observed_mae_improvement":
            observed_difference,

        "observed_mae_improvement_pct":
            float(
                improvement_pct
            ),

        "bootstrap_95_ci_lower":
            lower_ci,

        "bootstrap_95_ci_upper":
            upper_ci,

        "bootstrap_iterations":
            iterations,

        "evidence":
            evidence,
    }


def subject_sample_flag(
    rows,
):
    if rows < 10:
        return "VERY_SMALL_SAMPLE"

    if rows < 20:
        return "SMALL_SAMPLE"

    return (
        "ADEQUATE_FOR_DESCRIPTIVE_EVALUATION"
    )


def validate_required_files(
    files,
):
    missing = [
        str(path)
        for path in files
        if not path.exists()
    ]

    if missing:

        raise FileNotFoundError(
            "\nRequired academic-model validation/16 files "
            "are missing:\n"
            + "\n".join(
                missing
            )
        )


print(
    "=" * 110
)

print(
    "STUDENT PERFORMANCE AI - "
    "VALIDATION DIAGNOSTICS DETAILED VALIDATION DIAGNOSTICS"
)

print(
    "=" * 110
)


required_files = [
    OVERALL_ACADEMIC_PREDICTIONS,
    OVERALL_ATTENDANCE_PREDICTIONS,
    SUBJECT_ACADEMIC_PREDICTIONS,
    SUBJECT_ATTENDANCE_PREDICTIONS,
    OVERALL_ACADEMIC_METRICS,
    OVERALL_ATTENDANCE_METRICS,
    SUBJECT_ACADEMIC_METRICS,
    SUBJECT_ATTENDANCE_METRICS,
]


validate_required_files(
    required_files
)


overall_academic = pd.read_csv(
    OVERALL_ACADEMIC_PREDICTIONS
)

overall_attendance = pd.read_csv(
    OVERALL_ATTENDANCE_PREDICTIONS
)

subject_academic = pd.read_csv(
    SUBJECT_ACADEMIC_PREDICTIONS
)

subject_attendance = pd.read_csv(
    SUBJECT_ATTENDANCE_PREDICTIONS
)


print(
    "\nValidation prediction rows:"
)

print(
    f"  Overall academic-only: "
    f"{len(overall_academic)}"
)

print(
    f"  Overall + attendance: "
    f"{len(overall_attendance)}"
)

print(
    f"  Subject academic-only: "
    f"{len(subject_academic)}"
)

print(
    f"  Subject + attendance: "
    f"{len(subject_attendance)}"
)


overall_keys = [
    "student_id",
    "academic_period",
    "grade",
]


overall = overall_academic.merge(
    overall_attendance,
    on=overall_keys,
    how="inner",
    suffixes=(
        "_academic",
        "_attendance",
    ),
    validate="one_to_one",
)


if len(overall) != 217:

    raise RuntimeError(
        "\nOverall validation alignment failed.\n"
        f"Expected: 217\n"
        f"Actual: {len(overall)}"
    )


if not np.allclose(
    overall[
        "actual_average_academic"
    ],
    overall[
        "actual_average_attendance"
    ],
):

    raise RuntimeError(
        "\nOverall actual target mismatch "
        "between academic-model validation and attendance-feature validation."
    )


overall[
    "actual"
] = (
    overall[
        "actual_average_academic"
    ]
)


subject_keys = [
    "student_id",
    "academic_period",
    "grade",
    "target_subject",
]


subject = subject_academic.merge(
    subject_attendance,
    on=subject_keys,
    how="inner",
    suffixes=(
        "_academic",
        "_attendance",
    ),
    validate="one_to_one",
)


if len(subject) != 2029:

    raise RuntimeError(
        "\nSubject validation alignment failed.\n"
        f"Expected: 2029\n"
        f"Actual: {len(subject)}"
    )


if not np.allclose(
    subject[
        "actual_mark_academic"
    ],
    subject[
        "actual_mark_attendance"
    ],
):

    raise RuntimeError(
        "\nSubject actual target mismatch "
        "between academic-model validation and attendance-feature validation."
    )


subject[
    "actual"
] = (
    subject[
        "actual_mark_academic"
    ]
)


overall_diagnostic_records = []


for model_name, prediction_base in MODELS.items():

    for feature_set, suffix in [
        (
            "ACADEMIC_ONLY",
            "academic",
        ),
        (
            "ACADEMIC_PLUS_ATTENDANCE",
            "attendance",
        ),
    ]:

        prediction_column = (
            f"{prediction_base}_{suffix}"
        )

        metrics = regression_diagnostics(
            overall[
                "actual"
            ],
            overall[
                prediction_column
            ],
        )

        overall_diagnostic_records.append(
            {
                "task":
                    "OVERALL_PERFORMANCE",

                "model":
                    model_name,

                "feature_set":
                    feature_set,

                **metrics,
            }
        )


overall_diagnostics_df = pd.DataFrame(
    overall_diagnostic_records
)


overall_diagnostics_df.to_csv(
    OVERALL_DIAGNOSTICS_OUTPUT,
    index=False,
)


subject_diagnostic_records = []


for model_name, prediction_base in MODELS.items():

    for feature_set, suffix in [
        (
            "ACADEMIC_ONLY",
            "academic",
        ),
        (
            "ACADEMIC_PLUS_ATTENDANCE",
            "attendance",
        ),
    ]:

        prediction_column = (
            f"{prediction_base}_{suffix}"
        )

        metrics = regression_diagnostics(
            subject[
                "actual"
            ],
            subject[
                prediction_column
            ],
        )

        subject_diagnostic_records.append(
            {
                "task":
                    "POOLED_SUBJECT",

                "model":
                    model_name,

                "feature_set":
                    feature_set,

                **metrics,
            }
        )


subject_diagnostics_df = pd.DataFrame(
    subject_diagnostic_records
)


subject_diagnostics_df.to_csv(
    SUBJECT_DIAGNOSTICS_OUTPUT,
    index=False,
)


overall_grade_records = []


for grade, group in overall.groupby(
    "grade",
    sort=True,
):

    for feature_set, suffix in [
        (
            "ACADEMIC_ONLY",
            "academic",
        ),
        (
            "ACADEMIC_PLUS_ATTENDANCE",
            "attendance",
        ),
    ]:

        metrics = regression_diagnostics(
            group[
                "actual"
            ],
            group[
                f"prediction_random_forest_{suffix}"
            ],
        )

        overall_grade_records.append(
            {
                "grade":
                    grade,

                "feature_set":
                    feature_set,

                **metrics,
            }
        )


overall_grade_df = pd.DataFrame(
    overall_grade_records
)


overall_grade_df.to_csv(
    OVERALL_GRADE_OUTPUT,
    index=False,
)


subject_grade_records = []


for grade, group in subject.groupby(
    "grade",
    sort=True,
):

    for feature_set, suffix in [
        (
            "ACADEMIC_ONLY",
            "academic",
        ),
        (
            "ACADEMIC_PLUS_ATTENDANCE",
            "attendance",
        ),
    ]:

        metrics = regression_diagnostics(
            group[
                "actual"
            ],
            group[
                f"prediction_random_forest_{suffix}"
            ],
        )

        subject_grade_records.append(
            {
                "grade":
                    grade,

                "feature_set":
                    feature_set,

                **metrics,
            }
        )


subject_grade_df = pd.DataFrame(
    subject_grade_records
)


subject_grade_df.to_csv(
    SUBJECT_GRADE_OUTPUT,
    index=False,
)


subject_level_records = []


for target_subject, group in subject.groupby(
    "target_subject",
    sort=True,
):

    rows = len(
        group
    )

    academic_metrics = regression_diagnostics(
        group[
            "actual"
        ],
        group[
            "prediction_random_forest_academic"
        ],
    )

    attendance_metrics = regression_diagnostics(
        group[
            "actual"
        ],
        group[
            "prediction_random_forest_attendance"
        ],
    )

    subject_level_records.append(
        {
            "target_subject":
                target_subject,

            "validation_rows":
                rows,

            "sample_flag":
                subject_sample_flag(
                    rows
                ),

            "academic_only_mae":
                academic_metrics[
                    "mae"
                ],

            "academic_plus_attendance_mae":
                attendance_metrics[
                    "mae"
                ],

            "mae_improvement":
                (
                    academic_metrics[
                        "mae"
                    ]
                    - attendance_metrics[
                        "mae"
                    ]
                ),

            "academic_only_rmse":
                academic_metrics[
                    "rmse"
                ],

            "academic_plus_attendance_rmse":
                attendance_metrics[
                    "rmse"
                ],

            "academic_only_r2":
                academic_metrics[
                    "r2"
                ],

            "academic_plus_attendance_r2":
                attendance_metrics[
                    "r2"
                ],

            "academic_only_within_10_pct":
                academic_metrics[
                    "within_10_marks_pct"
                ],

            "attendance_within_10_pct":
                attendance_metrics[
                    "within_10_marks_pct"
                ],
        }
    )


subject_level_df = pd.DataFrame(
    subject_level_records
)


subject_level_df = (
    subject_level_df
    .sort_values(
        by=[
            "validation_rows",
            "target_subject",
        ],
        ascending=[
            False,
            True,
        ],
    )
)


subject_level_df.to_csv(
    SUBJECT_LEVEL_OUTPUT,
    index=False,
)


bootstrap_records = []


for task_name, dataframe in [
    (
        "OVERALL_PERFORMANCE",
        overall,
    ),
    (
        "POOLED_SUBJECT",
        subject,
    ),
]:

    for model_name, prediction_base in MODELS.items():

        result = (
            bootstrap_paired_mae_difference(
                actual=dataframe[
                    "actual"
                ],
                academic_prediction=dataframe[
                    f"{prediction_base}_academic"
                ],
                attendance_prediction=dataframe[
                    f"{prediction_base}_attendance"
                ],
            )
        )

        bootstrap_records.append(
            {
                "task":
                    task_name,

                "model":
                    model_name,

                **result,
            }
        )


bootstrap_df = pd.DataFrame(
    bootstrap_records
)


bootstrap_df.to_csv(
    BOOTSTRAP_OUTPUT,
    index=False,
)


metric_sources = [
    (
        "OVERALL_PERFORMANCE",
        "ACADEMIC_ONLY",
        OVERALL_ACADEMIC_METRICS,
    ),
    (
        "OVERALL_PERFORMANCE",
        "ACADEMIC_PLUS_ATTENDANCE",
        OVERALL_ATTENDANCE_METRICS,
    ),
    (
        "POOLED_SUBJECT",
        "ACADEMIC_ONLY",
        SUBJECT_ACADEMIC_METRICS,
    ),
    (
        "POOLED_SUBJECT",
        "ACADEMIC_PLUS_ATTENDANCE",
        SUBJECT_ATTENDANCE_METRICS,
    ),
]


generalization_records = []


for (
    task,
    feature_set,
    metrics_file,
) in metric_sources:

    metrics_df = pd.read_csv(
        metrics_file
    )

    for model_name in MODELS:

        model_rows = (
            metrics_df[
                metrics_df[
                    "model"
                ] == model_name
            ]
        )

        train_row = (
            model_rows[
                model_rows[
                    "evaluation_set"
                ] == "TRAIN"
            ]
            .iloc[0]
        )

        validation_row = (
            model_rows[
                model_rows[
                    "evaluation_set"
                ] == "VALIDATION"
            ]
            .iloc[0]
        )

        generalization_records.append(
            {
                "task":
                    task,

                "feature_set":
                    feature_set,

                "model":
                    model_name,

                "train_mae":
                    float(
                        train_row[
                            "mae"
                        ]
                    ),

                "validation_mae":
                    float(
                        validation_row[
                            "mae"
                        ]
                    ),

                "mae_generalization_gap":
                    float(
                        validation_row[
                            "mae"
                        ]
                        - train_row[
                            "mae"
                        ]
                    ),

                "train_rmse":
                    float(
                        train_row[
                            "rmse"
                        ]
                    ),

                "validation_rmse":
                    float(
                        validation_row[
                            "rmse"
                        ]
                    ),

                "rmse_generalization_gap":
                    float(
                        validation_row[
                            "rmse"
                        ]
                        - train_row[
                            "rmse"
                        ]
                    ),

                "train_r2":
                    float(
                        train_row[
                            "r2"
                        ]
                    ),

                "validation_r2":
                    float(
                        validation_row[
                            "r2"
                        ]
                    ),

                "r2_drop":
                    float(
                        train_row[
                            "r2"
                        ]
                        - validation_row[
                            "r2"
                        ]
                    ),
            }
        )


generalization_df = pd.DataFrame(
    generalization_records
)


generalization_df.to_csv(
    GENERALIZATION_OUTPUT,
    index=False,
)


overall_rf_bootstrap = (
    bootstrap_df[
        (
            bootstrap_df[
                "task"
            ] == "OVERALL_PERFORMANCE"
        )
        &
        (
            bootstrap_df[
                "model"
            ] == "Random Forest"
        )
    ]
    .iloc[0]
)


subject_rf_bootstrap = (
    bootstrap_df[
        (
            bootstrap_df[
                "task"
            ] == "POOLED_SUBJECT"
        )
        &
        (
            bootstrap_df[
                "model"
            ] == "Random Forest"
        )
    ]
    .iloc[0]
)


selection_summary = {
    "stage":
        "17",

    "validation_period":
        "2026_T1",

    "protected_final_test_period":
        "2026_T2",

    "final_test_used":
        False,

    "overall": {
        "leading_algorithm":
            "Random Forest",

        "academic_only_mae":
            float(
                overall_rf_bootstrap[
                    "academic_only_mae"
                ]
            ),

        "academic_plus_attendance_mae":
            float(
                overall_rf_bootstrap[
                    "academic_plus_attendance_mae"
                ]
            ),

        "attendance_mae_improvement":
            float(
                overall_rf_bootstrap[
                    "observed_mae_improvement"
                ]
            ),

        "attendance_mae_improvement_pct":
            float(
                overall_rf_bootstrap[
                    "observed_mae_improvement_pct"
                ]
            ),

        "bootstrap_95_ci": [
            float(
                overall_rf_bootstrap[
                    "bootstrap_95_ci_lower"
                ]
            ),
            float(
                overall_rf_bootstrap[
                    "bootstrap_95_ci_upper"
                ]
            ),
        ],

        "attendance_evidence":
            overall_rf_bootstrap[
                "evidence"
            ],
    },

    "subject": {
        "leading_algorithm":
            "Random Forest",

        "academic_only_mae":
            float(
                subject_rf_bootstrap[
                    "academic_only_mae"
                ]
            ),

        "academic_plus_attendance_mae":
            float(
                subject_rf_bootstrap[
                    "academic_plus_attendance_mae"
                ]
            ),

        "attendance_mae_improvement":
            float(
                subject_rf_bootstrap[
                    "observed_mae_improvement"
                ]
            ),

        "attendance_mae_improvement_pct":
            float(
                subject_rf_bootstrap[
                    "observed_mae_improvement_pct"
                ]
            ),

        "bootstrap_95_ci": [
            float(
                subject_rf_bootstrap[
                    "bootstrap_95_ci_lower"
                ]
            ),
            float(
                subject_rf_bootstrap[
                    "bootstrap_95_ci_upper"
                ]
            ),
        ],

        "attendance_evidence":
            subject_rf_bootstrap[
                "evidence"
            ],
    },

    "selection_rule_note":
        (
            "Final feature-set selection should consider "
            "validation accuracy, paired bootstrap evidence, "
            "generalisation, subject/grade stability, "
            "and implementation simplicity. "
            "The protected 2026 T2 test set must not be used "
            "to choose between feature configurations."
        ),
}


with open(
    SUMMARY_OUTPUT,
    "w",
    encoding="utf-8",
) as file:

    json.dump(
        selection_summary,
        file,
        indent=4,
    )


fig, ax = plt.subplots(
    figsize=(8, 7)
)


ax.scatter(
    overall[
        "actual"
    ],
    overall[
        "prediction_random_forest_academic"
    ],
    alpha=0.65,
    label="Academic only",
)


minimum = min(
    overall[
        "actual"
    ].min(),
    overall[
        "prediction_random_forest_academic"
    ].min(),
)


maximum = max(
    overall[
        "actual"
    ].max(),
    overall[
        "prediction_random_forest_academic"
    ].max(),
)


ax.plot(
    [
        minimum,
        maximum,
    ],
    [
        minimum,
        maximum,
    ],
    linestyle="--",
)


ax.set_title(
    "Random Forest Overall Performance\n"
    "2026 T1 Validation: Actual vs Predicted",
    fontweight="bold",
)


ax.set_xlabel(
    "Actual Next-Term Average"
)


ax.set_ylabel(
    "Predicted Next-Term Average"
)


ax.grid(
    alpha=0.20
)


fig.tight_layout()


fig.savefig(
    FIGURE_DIR
    / "overall_rf_actual_vs_predicted.png",
    dpi=300,
    bbox_inches="tight",
)


plt.close(
    fig
)


overall_grade_plot = (
    overall_grade_df.pivot(
        index="grade",
        columns="feature_set",
        values="mae",
    )
    .sort_index()
)


fig, ax = plt.subplots(
    figsize=(9, 6)
)


positions = np.arange(
    len(
        overall_grade_plot
    )
)


width = 0.36


bars1 = ax.bar(
    positions - width / 2,
    overall_grade_plot[
        "ACADEMIC_ONLY"
    ],
    width,
    label="Academic only",
)


bars2 = ax.bar(
    positions + width / 2,
    overall_grade_plot[
        "ACADEMIC_PLUS_ATTENDANCE"
    ],
    width,
    label="Academic + attendance",
)


ax.set_xticks(
    positions
)


ax.set_xticklabels(
    overall_grade_plot.index
)


ax.set_xlabel(
    "Grade"
)


ax.set_ylabel(
    "MAE (marks)"
)


ax.set_title(
    "Random Forest Overall Validation MAE by Grade",
    fontweight="bold",
)


ax.legend()


ax.grid(
    axis="y",
    alpha=0.20,
)


fig.tight_layout()


fig.savefig(
    FIGURE_DIR
    / "overall_rf_mae_by_grade.png",
    dpi=300,
    bbox_inches="tight",
)


plt.close(
    fig
)


subject_academic_error = np.abs(
    subject[
        "actual"
    ]
    - subject[
        "prediction_random_forest_academic"
    ]
)


subject_attendance_error = np.abs(
    subject[
        "actual"
    ]
    - subject[
        "prediction_random_forest_attendance"
    ]
)


fig, ax = plt.subplots(
    figsize=(9, 6)
)


ax.hist(
    subject_academic_error,
    bins=30,
    alpha=0.55,
    label="Academic only",
)


ax.hist(
    subject_attendance_error,
    bins=30,
    alpha=0.55,
    label="Academic + attendance",
)


ax.set_title(
    "Random Forest Subject Prediction\n"
    "Validation Absolute Error Distribution",
    fontweight="bold",
)


ax.set_xlabel(
    "Absolute Error (marks)"
)


ax.set_ylabel(
    "Prediction Count"
)


ax.legend()


ax.grid(
    axis="y",
    alpha=0.20,
)


fig.tight_layout()


fig.savefig(
    FIGURE_DIR
    / "subject_rf_absolute_error_distribution.png",
    dpi=300,
    bbox_inches="tight",
)


plt.close(
    fig
)


subject_plot = (
    subject_level_df[
        subject_level_df[
            "validation_rows"
        ] >= 20
    ]
    .copy()
)


subject_plot = (
    subject_plot
    .sort_values(
        "academic_plus_attendance_mae",
        ascending=True,
    )
)


positions = np.arange(
    len(
        subject_plot
    )
)


width = 0.36


fig_height = max(
    7,
    len(
        subject_plot
    )
    * 0.45,
)


fig, ax = plt.subplots(
    figsize=(
        11,
        fig_height,
    )
)


bars1 = ax.barh(
    positions - width / 2,
    subject_plot[
        "academic_only_mae"
    ],
    width,
    label="Academic only",
)


bars2 = ax.barh(
    positions + width / 2,
    subject_plot[
        "academic_plus_attendance_mae"
    ],
    width,
    label="Academic + attendance",
)


ax.set_yticks(
    positions
)


ax.set_yticklabels(
    subject_plot[
        "target_subject"
    ]
)


ax.set_xlabel(
    "MAE (marks)"
)


ax.set_ylabel(
    "Target Subject"
)


ax.set_title(
    "Random Forest Subject-Level Validation MAE",
    fontweight="bold",
)


ax.legend()


ax.grid(
    axis="x",
    alpha=0.20,
)


fig.tight_layout()


fig.savefig(
    FIGURE_DIR
    / "subject_rf_mae_by_subject.png",
    dpi=300,
    bbox_inches="tight",
)


plt.close(
    fig
)


print(
    "\n"
    + "=" * 110
)

print(
    "VALIDATION DIAGNOSTICS VALIDATION DIAGNOSTICS COMPLETE"
)

print(
    "=" * 110
)


print(
    "\nOVERALL PERFORMANCE — RANDOM FOREST"
)


overall_rf_diag = (
    overall_diagnostics_df[
        overall_diagnostics_df[
            "model"
        ] == "Random Forest"
    ]
)


print(
    overall_rf_diag[
        [
            "feature_set",
            "mae",
            "rmse",
            "r2",
            "median_absolute_error",
            "p90_absolute_error",
            "within_5_marks_pct",
            "within_10_marks_pct",
            "mean_residual_actual_minus_predicted",
        ]
    ]
    .to_string(
        index=False,
        float_format=lambda x: (
            f"{x:.4f}"
        ),
    )
)


print(
    "\nSUBJECT PREDICTION — RANDOM FOREST"
)


subject_rf_diag = (
    subject_diagnostics_df[
        subject_diagnostics_df[
            "model"
        ] == "Random Forest"
    ]
)


print(
    subject_rf_diag[
        [
            "feature_set",
            "mae",
            "rmse",
            "r2",
            "median_absolute_error",
            "p90_absolute_error",
            "within_5_marks_pct",
            "within_10_marks_pct",
            "mean_residual_actual_minus_predicted",
        ]
    ]
    .to_string(
        index=False,
        float_format=lambda x: (
            f"{x:.4f}"
        ),
    )
)


print(
    "\nPAIRED BOOTSTRAP — "
    "ACADEMIC ONLY VS ATTENDANCE"
)


print(
    bootstrap_df[
        [
            "task",
            "model",
            "observed_mae_improvement",
            "observed_mae_improvement_pct",
            "bootstrap_95_ci_lower",
            "bootstrap_95_ci_upper",
            "evidence",
        ]
    ]
    .to_string(
        index=False,
        float_format=lambda x: (
            f"{x:.4f}"
        ),
    )
)


print(
    "\nRANDOM FOREST GENERALISATION GAP"
)


print(
    generalization_df[
        generalization_df[
            "model"
        ] == "Random Forest"
    ]
    [
        [
            "task",
            "feature_set",
            "train_mae",
            "validation_mae",
            "mae_generalization_gap",
            "train_r2",
            "validation_r2",
            "r2_drop",
        ]
    ]
    .to_string(
        index=False,
        float_format=lambda x: (
            f"{x:.4f}"
        ),
    )
)


print(
    "\nRANDOM FOREST OVERALL MAE BY GRADE"
)


print(
    overall_grade_df[
        [
            "grade",
            "feature_set",
            "rows",
            "mae",
            "rmse",
            "r2",
            "within_10_marks_pct",
        ]
    ]
    .to_string(
        index=False,
        float_format=lambda x: (
            f"{x:.4f}"
        ),
    )
)


print(
    "\nRANDOM FOREST SUBJECT MAE BY GRADE"
)


print(
    subject_grade_df[
        [
            "grade",
            "feature_set",
            "rows",
            "mae",
            "rmse",
            "r2",
            "within_10_marks_pct",
        ]
    ]
    .to_string(
        index=False,
        float_format=lambda x: (
            f"{x:.4f}"
        ),
    )
)


print(
    "\nSUBJECT-LEVEL RANDOM FOREST "
    "(>=20 VALIDATION RECORDS)"
)


print(
    subject_level_df[
        subject_level_df[
            "validation_rows"
        ] >= 20
    ]
    [
        [
            "target_subject",
            "validation_rows",
            "academic_only_mae",
            "academic_plus_attendance_mae",
            "mae_improvement",
            "academic_only_within_10_pct",
            "attendance_within_10_pct",
        ]
    ]
    .to_string(
        index=False,
        float_format=lambda x: (
            f"{x:.4f}"
        ),
    )
)


print(
    "\nMODEL-SELECTION EVIDENCE"
)


print(
    "Overall Random Forest attendance evidence:"
)


print(
    f"  Improvement: "
    f"{overall_rf_bootstrap['observed_mae_improvement']:.4f}"
)


print(
    f"  Improvement %: "
    f"{overall_rf_bootstrap['observed_mae_improvement_pct']:.4f}%"
)


print(
    f"  95% bootstrap CI: "
    f"[{overall_rf_bootstrap['bootstrap_95_ci_lower']:.4f}, "
    f"{overall_rf_bootstrap['bootstrap_95_ci_upper']:.4f}]"
)


print(
    f"  Evidence: "
    f"{overall_rf_bootstrap['evidence']}"
)


print(
    "\nSubject Random Forest attendance evidence:"
)


print(
    f"  Improvement: "
    f"{subject_rf_bootstrap['observed_mae_improvement']:.4f}"
)


print(
    f"  Improvement %: "
    f"{subject_rf_bootstrap['observed_mae_improvement_pct']:.4f}%"
)


print(
    f"  95% bootstrap CI: "
    f"[{subject_rf_bootstrap['bootstrap_95_ci_lower']:.4f}, "
    f"{subject_rf_bootstrap['bootstrap_95_ci_upper']:.4f}]"
)


print(
    f"  Evidence: "
    f"{subject_rf_bootstrap['evidence']}"
)


print(
    "\nFINAL TEST PROTECTION:"
)


print(
    "  2026 T2 was NOT loaded by this script."
)


print(
    "  validation diagnostics used only saved 2026 T1 "
    "validation predictions and aggregate "
    "training/validation metrics."
)


print(
    "\nOutputs:"
)


for output in [
    OVERALL_DIAGNOSTICS_OUTPUT,
    SUBJECT_DIAGNOSTICS_OUTPUT,
    OVERALL_GRADE_OUTPUT,
    SUBJECT_GRADE_OUTPUT,
    SUBJECT_LEVEL_OUTPUT,
    BOOTSTRAP_OUTPUT,
    GENERALIZATION_OUTPUT,
    SUMMARY_OUTPUT,
]:

    print(
        f"  {output}"
    )


print(
    "\nFigures:"
)


print(
    f"  {FIGURE_DIR}"
)
