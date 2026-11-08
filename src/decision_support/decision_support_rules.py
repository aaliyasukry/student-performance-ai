from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

REPORT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "decision_support"
)

FIGURE_DIR = (
    PROJECT_ROOT
    / "figures"
    / "decision_support"
)

PRIVATE_DIR = (
    PROJECT_ROOT
    / "data"
    / "model_ready"
)

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

FIGURE_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


OVERALL_PREDICTIONS_FILE = (
    PRIVATE_DIR
    / "final_final_overall_predictions.csv"
)

SUBJECT_PREDICTIONS_FILE = (
    PRIVATE_DIR
    / "final_final_subject_predictions.csv"
)

OVERALL_DATA_FILE = (
    PRIVATE_DIR
    / "overall_prediction_dataset.csv"
)

SUBJECT_DATA_FILE = (
    PRIVATE_DIR
    / "subject_prediction_dataset.csv"
)

OVERALL_METRICS_FILE = (
    PROJECT_ROOT
    / "reports"
    / "evaluation"
    / "final_overall_metrics.csv"
)

SUBJECT_LEVEL_METRICS_FILE = (
    PROJECT_ROOT
    / "reports"
    / "evaluation"
    / "final_subject_level_metrics.csv"
)

SHAP_OVERALL_IMPORTANCE_FILE = (
    PROJECT_ROOT
    / "reports"
    / "explainability"
    / "overall_shap_global_importance.csv"
)

SHAP_SUBJECT_IMPORTANCE_FILE = (
    PROJECT_ROOT
    / "reports"
    / "explainability"
    / "subject_shap_grouped_importance.csv"
)


RELIABILITY_OUTPUT = (
    REPORT_DIR
    / "subject_prediction_reliability.csv"
)

RULES_OUTPUT = (
    REPORT_DIR
    / "decision_support_rules.json"
)

FEATURE_LABEL_OUTPUT = (
    REPORT_DIR
    / "feature_label_map.csv"
)

SUMMARY_OUTPUT = (
    REPORT_DIR
    / "decision_support_summary.json"
)

OVERALL_PREVIEW_OUTPUT = (
    PRIVATE_DIR
    / "decision_support_overall_decision_support_preview.csv"
)

SUBJECT_PREVIEW_OUTPUT = (
    PRIVATE_DIR
    / "decision_support_subject_decision_support_preview.csv"
)


# UI presentation thresholds; they do not modify the trained model.

ACADEMIC_BENCHMARK = 40.0
NEEDS_ATTENTION_UPPER = 50.0
HIGH_PERFORMER_THRESHOLD = 75.0

TREND_THRESHOLD_MARKS = 3.0

ATTENDANCE_GOOD_THRESHOLD = 90.0
ATTENDANCE_MONITOR_THRESHOLD = 80.0
ATTENDANCE_MIN_COVERAGE = 80.0


# Reliability comes from held-out final evaluation, not SHAP values.

MIN_RELIABILITY_ROWS = 20

HIGH_RELIABILITY_R2 = 0.70
HIGH_RELIABILITY_MAX_MAE = 10.0

MODERATE_RELIABILITY_R2 = 0.40
MODERATE_RELIABILITY_MAX_MAE = 15.0


def validate_files(paths):
    missing = [
        str(path)
        for path in paths
        if not path.exists()
    ]

    if missing:
        raise FileNotFoundError(
            "\nRequired decision support input files are missing:\n"
            + "\n".join(missing)
        )


def safe_float(value):
    try:
        if pd.isna(value):
            return np.nan
        return float(value)
    except Exception:
        return np.nan


def academic_risk_label(prediction):
    """Prediction-based academic category."""

    prediction = safe_float(
        prediction
    )

    if np.isnan(prediction):
        return "UNAVAILABLE"

    if prediction < ACADEMIC_BENCHMARK:
        return "HIGH_RISK"

    if prediction < NEEDS_ATTENTION_UPPER:
        return "NEEDS_ATTENTION"

    if prediction < HIGH_PERFORMER_THRESHOLD:
        return "ON_TRACK"

    return "HIGH_PERFORMER"


def academic_risk_message(label):
    messages = {
        "HIGH_RISK":
            (
                "Predicted performance is below the "
                "current school benchmark. Review recent "
                "academic history and consider support."
            ),

        "NEEDS_ATTENTION":
            (
                "Predicted performance is above the "
                "benchmark but remains close enough to "
                "warrant monitoring."
            ),

        "ON_TRACK":
            (
                "Predicted performance is currently "
                "above the benchmark."
            ),

        "HIGH_PERFORMER":
            (
                "Predicted performance is within the "
                "current high-performance presentation band."
            ),

        "UNAVAILABLE":
            (
                "No prediction is currently available."
            ),
    }

    return messages[
        label
    ]


def trend_label(value):
    """Transparent operational trend rule."""

    value = safe_float(
        value
    )

    if np.isnan(value):
        return "INSUFFICIENT_HISTORY"

    if value >= TREND_THRESHOLD_MARKS:
        return "IMPROVING"

    if value <= -TREND_THRESHOLD_MARKS:
        return "DECLINING"

    return "STABLE"


def attendance_context(
    attendance_pct,
    coverage_pct,
    available,
):
    """Attendance remains contextual decision-support information and is NOT a final-model predictor."""

    attendance_pct = safe_float(
        attendance_pct
    )

    coverage_pct = safe_float(
        coverage_pct
    )

    available = safe_float(
        available
    )

    if (
        np.isnan(attendance_pct)
        or (
            not np.isnan(available)
            and available <= 0
        )
    ):
        return "UNAVAILABLE"

    if (
        not np.isnan(coverage_pct)
        and coverage_pct
        < ATTENDANCE_MIN_COVERAGE
    ):
        return "PARTIAL_DATA"

    if attendance_pct >= ATTENDANCE_GOOD_THRESHOLD:
        return "GOOD"

    if attendance_pct >= ATTENDANCE_MONITOR_THRESHOLD:
        return "MONITOR"

    return "CONCERN"


def reliability_label(
    rows,
    mae,
    r2,
):
    """Reliability is evidence about the model for this subject, not confidence in one individual prediction."""

    rows = int(
        rows
    )

    mae = safe_float(
        mae
    )

    r2 = safe_float(
        r2
    )

    if rows < MIN_RELIABILITY_ROWS:
        return "INSUFFICIENT_DATA"

    if np.isnan(mae) or np.isnan(r2):
        return "INSUFFICIENT_DATA"

    if (
        r2 >= HIGH_RELIABILITY_R2
        and mae <= HIGH_RELIABILITY_MAX_MAE
    ):
        return "HIGH"

    if (
        r2 >= MODERATE_RELIABILITY_R2
        and mae <= MODERATE_RELIABILITY_MAX_MAE
    ):
        return "MODERATE"

    return "LOW"


def reliability_display_policy(label):
    """Controls whether the UI should show a prediction."""

    policies = {
        "HIGH":
            "SHOW",

        "MODERATE":
            "SHOW",

        "LOW":
            "SHOW_WITH_CAUTION",

        "INSUFFICIENT_DATA":
            "SUPPRESS_BY_DEFAULT",
    }

    return policies[
        label
    ]


def reliability_message(label):
    messages = {
        "HIGH":
            (
                "Held-out evaluation shows comparatively "
                "strong predictive performance for this subject."
            ),

        "MODERATE":
            (
                "The prediction is usable for decision support, "
                "but subject-level error remains material."
            ),

        "LOW":
            (
                "Held-out performance for this subject is weak. "
                "Use the forecast only as supporting information."
            ),

        "INSUFFICIENT_DATA":
            (
                "Too few held-out observations are available "
                "to support a reliable subject-level forecast."
            ),
    }

    return messages[
        label
    ]


def humanise_subject(subject):
    return (
        str(subject)
        .replace(
            "_",
            " "
        )
        .title()
    )


def humanise_feature(feature):
    """Convert implementation feature names into language suitable for reports/API/UI."""

    feature = str(
        feature
    )

    direct_map = {
        "historical_mean_average":
            "Historical overall average",

        "prev_calculated_average":
            "Previous term overall average",

        "performance_trend":
            "Overall performance trend",

        "historical_std_average":
            "Variation in historical overall performance",

        "history_term_count":
            "Number of historical terms",

        "grade":
            "Grade",

        "prev_subject_mark":
            "Previous mark in this subject",

        "historical_mean_subject_mark":
            "Historical mean mark in this subject",

        "historical_subject_mark_count":
            "Number of historical marks in this subject",

        "subject_mark_trend":
            "Subject performance trend",

        "prev_subject_absent":
            "Previous-term absence for this subject",

        "prev_subject_not_taken":
            "Previous-term subject not taken",

        "target_subject":
            "Target subject",

        "prev_numeric_subject_count":
            "Previous-term subjects with numeric marks",

        "prev_absent_subject_count":
            "Previous-term absent-subject count",

        "prev_not_taken_subject_count":
            "Previous-term not-taken subject count",

        "historical_mean_absent_subject_count":
            "Historical mean absent-subject count",

        "historical_mean_not_taken_subject_count":
            "Historical mean not-taken subject count",
    }

    if feature in direct_map:
        return direct_map[
            feature
        ]

    if feature.startswith(
        "historical_mean_mark_"
    ):
        subject = feature.replace(
            "historical_mean_mark_",
            "",
            1,
        )

        return (
            "Historical mean "
            + humanise_subject(
                subject
            )
            + " mark"
        )

    if feature.startswith(
        "prev_mark_"
    ):
        subject = feature.replace(
            "prev_mark_",
            "",
            1,
        )

        return (
            "Previous "
            + humanise_subject(
                subject
            )
            + " mark"
        )

    if feature.startswith(
        "missingindicator_"
    ):
        base = feature.replace(
            "missingindicator_",
            "",
            1,
        )

        return (
            "Missing-history indicator: "
            + humanise_feature(
                base
            )
        )

    return humanise_subject(
        feature
    )


print(
    "=" * 110
)

print(
    "STUDENT PERFORMANCE AI - "
    "DECISION SUPPORT ACADEMIC DECISION SUPPORT"
)

print(
    "=" * 110
)


validate_files(
    [
        OVERALL_PREDICTIONS_FILE,
        SUBJECT_PREDICTIONS_FILE,
        OVERALL_DATA_FILE,
        SUBJECT_DATA_FILE,
        OVERALL_METRICS_FILE,
        SUBJECT_LEVEL_METRICS_FILE,
        SHAP_OVERALL_IMPORTANCE_FILE,
        SHAP_SUBJECT_IMPORTANCE_FILE,
    ]
)


overall_metrics = pd.read_csv(
    OVERALL_METRICS_FILE
)

subject_metrics = pd.read_csv(
    SUBJECT_LEVEL_METRICS_FILE
)


overall_final = (
    overall_metrics[
        overall_metrics[
            "evaluation_set"
        ] == "FINAL_TEST_2026_T2"
    ]
    .iloc[0]
)


overall_reliability = (
    "HIGH"
    if (
        float(
            overall_final[
                "r2"
            ]
        )
        >= HIGH_RELIABILITY_R2
        and float(
            overall_final[
                "mae"
            ]
        )
        <= HIGH_RELIABILITY_MAX_MAE
    )
    else "MODERATE"
)


reliability_records = []


for _, row in subject_metrics.iterrows():

    label = reliability_label(
        rows=row[
            "rows"
        ],
        mae=row[
            "mae"
        ],
        r2=row[
            "r2"
        ],
    )

    reliability_records.append(
        {
            "target_subject":
                row[
                    "target_subject"
                ],

            "display_subject":
                humanise_subject(
                    row[
                        "target_subject"
                    ]
                ),

            "final_test_rows":
                int(
                    row[
                        "rows"
                    ]
                ),

            "final_test_mae":
                float(
                    row[
                        "mae"
                    ]
                ),

            "final_test_rmse":
                float(
                    row[
                        "rmse"
                    ]
                ),

            "final_test_r2":
                (
                    float(
                        row[
                            "r2"
                        ]
                    )
                    if not pd.isna(
                        row[
                            "r2"
                        ]
                    )
                    else np.nan
                ),

            "within_10_marks_pct":
                float(
                    row[
                        "within_10_marks_pct"
                    ]
                ),

            "prediction_reliability":
                label,

            "display_policy":
                reliability_display_policy(
                    label
                ),

            "reliability_message":
                reliability_message(
                    label
                ),
        }
    )


reliability_df = pd.DataFrame(
    reliability_records
)


reliability_order = {
    "HIGH": 1,
    "MODERATE": 2,
    "LOW": 3,
    "INSUFFICIENT_DATA": 4,
}


reliability_df[
    "_order"
] = reliability_df[
    "prediction_reliability"
].map(
    reliability_order
)


reliability_df = (
    reliability_df
    .sort_values(
        by=[
            "_order",
            "final_test_mae",
        ],
        ascending=[
            True,
            True,
        ],
    )
    .drop(
        columns=[
            "_order"
        ]
    )
    .reset_index(
        drop=True
    )
)


reliability_df.to_csv(
    RELIABILITY_OUTPUT,
    index=False,
)


overall_shap = pd.read_csv(
    SHAP_OVERALL_IMPORTANCE_FILE
)

subject_shap = pd.read_csv(
    SHAP_SUBJECT_IMPORTANCE_FILE
)


all_features = sorted(
    set(
        overall_shap[
            "feature"
        ].astype(str)
    )
    |
    set(
        subject_shap[
            "feature"
        ].astype(str)
    )
)


feature_label_df = pd.DataFrame(
    {
        "feature":
            all_features,

        "display_label": [
            humanise_feature(
                feature
            )
            for feature in all_features
        ],
    }
)


feature_label_df.to_csv(
    FEATURE_LABEL_OUTPUT,
    index=False,
)


overall_predictions = pd.read_csv(
    OVERALL_PREDICTIONS_FILE
)

overall_source = (
    pd.read_csv(
        OVERALL_DATA_FILE
    )
)


overall_source = (
    overall_source[
        overall_source[
            "dataset_split"
        ] == "FINAL_TEST"
    ]
    .copy()
)


overall_context_columns = [
    "student_id",
    "academic_period",
    "grade",
]


for optional_column in [
    "performance_trend",
    "prev_attendance_pct",
    "prev_attendance_coverage_pct",
    "prev_attendance_available",
]:

    if optional_column in overall_source.columns:

        overall_context_columns.append(
            optional_column
        )


overall_preview = (
    overall_predictions
    .merge(
        overall_source[
            overall_context_columns
        ],
        on=[
            "student_id",
            "academic_period",
            "grade",
        ],
        how="left",
        validate="one_to_one",
    )
)


overall_preview[
    "academic_risk"
] = (
    overall_preview[
        "predicted_average"
    ]
    .apply(
        academic_risk_label
    )
)


overall_preview[
    "academic_risk_message"
] = (
    overall_preview[
        "academic_risk"
    ]
    .apply(
        academic_risk_message
    )
)


if "performance_trend" in overall_preview.columns:

    overall_preview[
        "performance_trend_label"
    ] = (
        overall_preview[
            "performance_trend"
        ]
        .apply(
            trend_label
        )
    )

else:

    overall_preview[
        "performance_trend_label"
    ] = "UNAVAILABLE"


if all(
    column in overall_preview.columns
    for column in [
        "prev_attendance_pct",
        "prev_attendance_coverage_pct",
        "prev_attendance_available",
    ]
):

    overall_preview[
        "attendance_context"
    ] = overall_preview.apply(
        lambda row: attendance_context(
            row[
                "prev_attendance_pct"
            ],
            row[
                "prev_attendance_coverage_pct"
            ],
            row[
                "prev_attendance_available"
            ],
        ),
        axis=1,
    )

else:

    overall_preview[
        "attendance_context"
    ] = "UNAVAILABLE"


overall_preview[
    "prediction_reliability"
] = overall_reliability


overall_preview[
    "prediction_reliability_message"
] = reliability_message(
    overall_reliability
)


overall_preview.to_csv(
    OVERALL_PREVIEW_OUTPUT,
    index=False,
)


subject_predictions = pd.read_csv(
    SUBJECT_PREDICTIONS_FILE
)

subject_source = pd.read_csv(
    SUBJECT_DATA_FILE
)


subject_source = (
    subject_source[
        subject_source[
            "dataset_split"
        ] == "FINAL_TEST"
    ]
    .copy()
)


subject_context_columns = [
    "student_id",
    "academic_period",
    "grade",
    "target_subject",
]


for optional_column in [
    "subject_mark_trend",
    "performance_trend",
    "prev_attendance_pct",
    "prev_attendance_coverage_pct",
    "prev_attendance_available",
]:

    if optional_column in subject_source.columns:

        subject_context_columns.append(
            optional_column
        )


subject_preview = (
    subject_predictions
    .merge(
        subject_source[
            subject_context_columns
        ],
        on=[
            "student_id",
            "academic_period",
            "grade",
            "target_subject",
        ],
        how="left",
        validate="one_to_one",
    )
)


subject_preview = (
    subject_preview
    .merge(
        reliability_df[
            [
                "target_subject",
                "prediction_reliability",
                "display_policy",
                "reliability_message",
            ]
        ],
        on="target_subject",
        how="left",
        validate="many_to_one",
    )
)


subject_preview[
    "academic_risk"
] = (
    subject_preview[
        "predicted_mark"
    ]
    .apply(
        academic_risk_label
    )
)


subject_preview[
    "academic_risk_message"
] = (
    subject_preview[
        "academic_risk"
    ]
    .apply(
        academic_risk_message
    )
)


if "subject_mark_trend" in subject_preview.columns:

    subject_preview[
        "subject_trend_label"
    ] = (
        subject_preview[
            "subject_mark_trend"
        ]
        .apply(
            trend_label
        )
    )

else:

    subject_preview[
        "subject_trend_label"
    ] = "UNAVAILABLE"


if all(
    column in subject_preview.columns
    for column in [
        "prev_attendance_pct",
        "prev_attendance_coverage_pct",
        "prev_attendance_available",
    ]
):

    subject_preview[
        "attendance_context"
    ] = subject_preview.apply(
        lambda row: attendance_context(
            row[
                "prev_attendance_pct"
            ],
            row[
                "prev_attendance_coverage_pct"
            ],
            row[
                "prev_attendance_available"
            ],
        ),
        axis=1,
    )

else:

    subject_preview[
        "attendance_context"
    ] = "UNAVAILABLE"


subject_preview[
    "ui_prediction_status"
] = np.where(
    subject_preview[
        "display_policy"
    ]
    == "SUPPRESS_BY_DEFAULT",
    "PREDICTION_NOT_SHOWN_BY_DEFAULT",
    np.where(
        subject_preview[
            "display_policy"
        ]
        == "SHOW_WITH_CAUTION",
        "SHOW_WITH_RELIABILITY_WARNING",
        "SHOW",
    )
)


subject_preview.to_csv(
    SUBJECT_PREVIEW_OUTPUT,
    index=False,
)


rules = {
    "stage":
        "20",

    "purpose":
        (
            "Convert frozen model outputs into "
            "teacher-facing decision-support signals."
        ),

    "model_modified":
        False,

    "academic_risk": {
        "benchmark_mark":
            ACADEMIC_BENCHMARK,

        "categories": {
            "HIGH_RISK":
                f"prediction < {ACADEMIC_BENCHMARK}",

            "NEEDS_ATTENTION":
                (
                    f"{ACADEMIC_BENCHMARK} <= prediction "
                    f"< {NEEDS_ATTENTION_UPPER}"
                ),

            "ON_TRACK":
                (
                    f"{NEEDS_ATTENTION_UPPER} <= prediction "
                    f"< {HIGH_PERFORMER_THRESHOLD}"
                ),

            "HIGH_PERFORMER":
                (
                    f"prediction >= "
                    f"{HIGH_PERFORMER_THRESHOLD}"
                ),
        },

        "policy_note":
            (
                "The 40-mark benchmark is grounded in "
                "the supplied school source sheets. "
                "Other presentation bands are V1 operational "
                "defaults and must remain configurable."
            ),
    },

    "trend": {
        "threshold_marks":
            TREND_THRESHOLD_MARKS,

        "IMPROVING":
            (
                f"trend >= +{TREND_THRESHOLD_MARKS}"
            ),

        "STABLE":
            (
                f"-{TREND_THRESHOLD_MARKS} < trend "
                f"< +{TREND_THRESHOLD_MARKS}"
            ),

        "DECLINING":
            (
                f"trend <= -{TREND_THRESHOLD_MARKS}"
            ),

        "INSUFFICIENT_HISTORY":
            "trend unavailable",
    },

    "attendance_context": {
        "used_by_prediction_model":
            False,

        "GOOD":
            (
                f"attendance >= "
                f"{ATTENDANCE_GOOD_THRESHOLD}%"
            ),

        "MONITOR":
            (
                f"{ATTENDANCE_MONITOR_THRESHOLD}% <= "
                f"attendance < "
                f"{ATTENDANCE_GOOD_THRESHOLD}%"
            ),

        "CONCERN":
            (
                f"attendance < "
                f"{ATTENDANCE_MONITOR_THRESHOLD}%"
            ),

        "PARTIAL_DATA":
            (
                f"attendance coverage < "
                f"{ATTENDANCE_MIN_COVERAGE}%"
            ),

        "UNAVAILABLE":
            "attendance unavailable",
    },

    "prediction_reliability": {
        "basis":
            (
                "final evaluation held-out subject performance "
                "and final-test sample support."
            ),

        "HIGH": {
            "minimum_rows":
                MIN_RELIABILITY_ROWS,

            "minimum_r2":
                HIGH_RELIABILITY_R2,

            "maximum_mae":
                HIGH_RELIABILITY_MAX_MAE,

            "display_policy":
                "SHOW",
        },

        "MODERATE": {
            "minimum_rows":
                MIN_RELIABILITY_ROWS,

            "minimum_r2":
                MODERATE_RELIABILITY_R2,

            "maximum_mae":
                MODERATE_RELIABILITY_MAX_MAE,

            "display_policy":
                "SHOW",
        },

        "LOW": {
            "definition":
                (
                    "Sufficient held-out sample but "
                    "performance below moderate thresholds."
                ),

            "display_policy":
                "SHOW_WITH_CAUTION",
        },

        "INSUFFICIENT_DATA": {
            "definition":
                (
                    f"fewer than "
                    f"{MIN_RELIABILITY_ROWS} "
                    "held-out subject observations, "
                    "or unusable evaluation metrics"
                ),

            "display_policy":
                "SUPPRESS_BY_DEFAULT",
        },

        "important_note":
            (
                "Prediction reliability is not an individual "
                "probability and must not be called confidence."
            ),
    },

    "shap_explanation": {
        "purpose":
            (
                "Explain why a frozen model prediction "
                "was pushed higher or lower."
            ),

        "positive":
            "pushes prediction higher",

        "negative":
            "pushes prediction lower",

        "not_used_for":
            [
                "confidence score",
                "prediction reliability",
                "causal claim",
            ],
    },

    "ui_separation": {
        "academic_risk":
            (
                "How concerning the predicted academic "
                "outcome is."
            ),

        "prediction_reliability":
            (
                "How strong the held-out evidence is for "
                "predictions of this type."
            ),

        "attendance_context":
            (
                "Historical attendance information shown "
                "to the teacher but not used by the final "
                "forecasting model."
            ),

        "why_this_prediction":
            (
                "Human-readable SHAP factors."
            ),
    },
}


with open(
    RULES_OUTPUT,
    "w",
    encoding="utf-8",
) as file:

    json.dump(
        rules,
        file,
        indent=4,
    )


reliability_counts = (
    reliability_df[
        "prediction_reliability"
    ]
    .value_counts()
    .to_dict()
)


summary = {
    "stage":
        "20",

    "overall_prediction_reliability":
        overall_reliability,

    "subject_reliability_counts":
        {
            str(key):
                int(value)
            for key, value
            in reliability_counts.items()
        },

    "subject_reliability": (
        reliability_df[
            [
                "target_subject",
                "prediction_reliability",
                "display_policy",
                "final_test_rows",
                "final_test_mae",
                "final_test_r2",
            ]
        ]
        .to_dict(
            orient="records"
        )
    ),

    "principles": [
        (
            "Academic risk and prediction reliability "
            "are displayed separately."
        ),
        (
            "Attendance is contextual information and is "
            "not represented as a final-model predictor."
        ),
        (
            "SHAP explains model behaviour but is not used "
            "as confidence or reliability."
        ),
        (
            "Low-support predictions must be clearly "
            "qualified or suppressed."
        ),
    ],
}


with open(
    SUMMARY_OUTPUT,
    "w",
    encoding="utf-8",
) as file:

    json.dump(
        summary,
        file,
        indent=4,
        allow_nan=True,
    )


plot_df = (
    reliability_df
    .sort_values(
        by=[
            "prediction_reliability",
            "final_test_mae",
        ]
    )
    .copy()
)


fig_height = max(
    7,
    len(plot_df) * 0.42,
)


fig, ax = plt.subplots(
    figsize=(
        11,
        fig_height,
    )
)


bars = ax.barh(
    plot_df[
        "display_subject"
    ],
    plot_df[
        "final_test_mae"
    ],
)


ax.set_title(
    "Subject Prediction Reliability Evidence\n"
    "Final 2026 T2 MAE",
    fontweight="bold",
)


ax.set_xlabel(
    "Final-test MAE (marks)"
)


ax.set_ylabel(
    "Subject"
)


ax.grid(
    axis="x",
    alpha=0.20,
)


for index, bar in enumerate(
    bars
):

    label = (
        plot_df
        .iloc[index][
            "prediction_reliability"
        ]
    )

    value = bar.get_width()

    ax.text(
        value,
        bar.get_y()
        + bar.get_height() / 2,
        f" {value:.2f} — {label}",
        va="center",
        fontsize=8,
    )


fig.tight_layout()


fig.savefig(
    FIGURE_DIR
    / "subject_prediction_reliability.png",
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
    "DECISION SUPPORT ACADEMIC DECISION SUPPORT COMPLETE"
)

print(
    "=" * 110
)


print(
    "\nIMPORTANT CONCEPT:"
)

print(
    "  Academic risk != prediction reliability"
)

print(
    "  A student can have HIGH academic risk "
    "with HIGH or LOW prediction reliability."
)


print(
    "\nOVERALL MODEL"
)

print(
    f"  Final-test MAE: "
    f"{float(overall_final['mae']):.4f}"
)

print(
    f"  Final-test R²: "
    f"{float(overall_final['r2']):.4f}"
)

print(
    f"  Prediction reliability: "
    f"{overall_reliability}"
)


print(
    "\nSUBJECT PREDICTION RELIABILITY"
)


print(
    reliability_df[
        [
            "display_subject",
            "final_test_rows",
            "final_test_mae",
            "final_test_r2",
            "prediction_reliability",
            "display_policy",
        ]
    ]
    .to_string(
        index=False,
        float_format=lambda value: (
            f"{value:.4f}"
        ),
    )
)


print(
    "\nReliability counts:"
)


for label in [
    "HIGH",
    "MODERATE",
    "LOW",
    "INSUFFICIENT_DATA",
]:

    count = int(
        reliability_counts.get(
            label,
            0,
        )
    )

    print(
        f"  {label}: {count}"
    )


print(
    "\nUI DISPLAY RULE:"
)

print(
    "  HIGH / MODERATE -> show prediction"
)

print(
    "  LOW -> show with clear reliability warning"
)

print(
    "  INSUFFICIENT_DATA -> suppress precise prediction "
    "by default"
)


print(
    "\nAGGREGATE OUTPUTS"
)

print(
    f"  {RELIABILITY_OUTPUT}"
)

print(
    f"  {RULES_OUTPUT}"
)

print(
    f"  {FEATURE_LABEL_OUTPUT}"
)

print(
    f"  {SUMMARY_OUTPUT}"
)


print(
    "\nPRIVATE PREVIEW OUTPUTS"
)

print(
    f"  {OVERALL_PREVIEW_OUTPUT}"
)

print(
    f"  {SUBJECT_PREVIEW_OUTPUT}"
)


print(
    "\nFIGURES"
)

print(
    f"  {FIGURE_DIR}"
)


print(
    "\nIMPORTANT:"
)

print(
    "  decision support did not retrain or change the model."
)

print(
    "  Reliability is based on held-out evaluation, "
    "not SHAP magnitude."
)

print(
    "  Operational thresholds remain configurable "
    "for the school."
)
