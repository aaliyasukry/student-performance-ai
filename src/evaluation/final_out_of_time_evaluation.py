from pathlib import Path
import json

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


RANDOM_STATE = 42

PROJECT_ROOT = Path(__file__).resolve().parents[2]

OVERALL_DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "model_ready"
    / "overall_prediction_dataset.csv"
)

SUBJECT_DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "model_ready"
    / "subject_prediction_dataset.csv"
)

OVERALL_FEATURE_AUDIT_FILE = (
    PROJECT_ROOT
    / "reports"
    / "modeling"
    / "overall_academic_feature_audit.csv"
)

SUBJECT_FEATURE_AUDIT_FILE = (
    PROJECT_ROOT
    / "reports"
    / "modeling"
    / "subject_academic_feature_audit.csv"
)

MODEL_SELECTION_FILE = (
    PROJECT_ROOT
    / "reports"
    / "evaluation"
    / "model_selection_evidence.json"
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
    / "final_evaluation"
)

FINAL_MODEL_DIR = (
    PROJECT_ROOT
    / "models"
    / "final"
)

PRIVATE_OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "model_ready"
)

REPORT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)
FINAL_MODEL_DIR.mkdir(parents=True, exist_ok=True)


OVERALL_METRICS_OUTPUT = (
    REPORT_DIR
    / "final_overall_metrics.csv"
)

OVERALL_GRADE_OUTPUT = (
    REPORT_DIR
    / "final_overall_grade_metrics.csv"
)

SUBJECT_METRICS_OUTPUT = (
    REPORT_DIR
    / "final_subject_metrics.csv"
)

SUBJECT_GRADE_OUTPUT = (
    REPORT_DIR
    / "final_subject_grade_metrics.csv"
)

SUBJECT_LEVEL_OUTPUT = (
    REPORT_DIR
    / "final_subject_level_metrics.csv"
)

SUMMARY_OUTPUT = (
    REPORT_DIR
    / "final_evaluation_summary.json"
)

OVERALL_PREDICTIONS_OUTPUT = (
    PRIVATE_OUTPUT_DIR
    / "final_evaluation_final_overall_predictions.csv"
)

SUBJECT_PREDICTIONS_OUTPUT = (
    PRIVATE_OUTPUT_DIR
    / "final_evaluation_final_subject_predictions.csv"
)

OVERALL_MODEL_OUTPUT = (
    FINAL_MODEL_DIR
    / "overall_random_forest_academic_only.joblib"
)

SUBJECT_MODEL_OUTPUT = (
    FINAL_MODEL_DIR
    / "subject_random_forest_academic_only.joblib"
)


# Attendance gave no clear paired gain, so the academic/history-only Random Forest is locked before the final test.

RF_PARAMS = {
    "n_estimators": 500,
    "max_depth": None,
    "min_samples_split": 4,
    "min_samples_leaf": 2,
    "max_features": "sqrt",
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}


def make_one_hot_encoder():
    try:
        return OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False,
        )
    except TypeError:
        return OneHotEncoder(
            handle_unknown="ignore",
            sparse=False,
        )


def regression_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    residual = y_true - y_pred
    absolute_error = np.abs(residual)

    result = {
        "rows": int(len(y_true)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "median_absolute_error": float(np.median(absolute_error)),
        "p75_absolute_error": float(np.percentile(absolute_error, 75)),
        "p90_absolute_error": float(np.percentile(absolute_error, 90)),
        "max_absolute_error": float(absolute_error.max()),
        "within_5_marks_pct": float((absolute_error <= 5).mean() * 100),
        "within_10_marks_pct": float((absolute_error <= 10).mean() * 100),
        "within_15_marks_pct": float((absolute_error <= 15).mean() * 100),
        "mean_residual_actual_minus_predicted": float(residual.mean()),
        "predictions_below_0": int((y_pred < 0).sum()),
        "predictions_above_100": int((y_pred > 100).sum()),
    }

    if len(y_true) >= 2:
        result["r2"] = float(r2_score(y_true, y_pred))
    else:
        result["r2"] = np.nan

    return result


def subject_sample_flag(rows):
    if rows < 10:
        return "VERY_SMALL_SAMPLE"

    if rows < 20:
        return "SMALL_SAMPLE"

    return "ADEQUATE_FOR_DESCRIPTIVE_EVALUATION"


def load_locked_features(
    audit_file,
    categorical_feature=None,
):
    audit = pd.read_csv(audit_file)

    required = {
        "feature",
        "status",
    }

    missing = required.difference(
        audit.columns
    )

    if missing:
        raise RuntimeError(
            "\nFeature audit is missing required columns:\n"
            f"{sorted(missing)}\n"
            f"File: {audit_file}"
        )

    included = (
        audit.loc[
            audit[
                "status"
            ].astype(str).str.upper() == "INCLUDED",
            "feature",
        ]
        .astype(str)
        .tolist()
    )

    if categorical_feature is None:
        return included

    if categorical_feature not in included:
        raise RuntimeError(
            "\nExpected categorical feature "
            f"'{categorical_feature}' was not INCLUDED in:\n"
            f"{audit_file}"
        )

    numeric = [
        feature
        for feature in included
        if feature != categorical_feature
    ]

    return (
        numeric,
        [categorical_feature],
    )


def validate_files(paths):
    missing = [
        str(path)
        for path in paths
        if not path.exists()
    ]

    if missing:
        raise FileNotFoundError(
            "\nRequired files are missing:\n"
            + "\n".join(missing)
        )


print("=" * 110)

print(
    "STUDENT PERFORMANCE AI - "
    "FINAL OUT-OF-TIME EVALUATION FINAL OUT-OF-TIME EVALUATION"
)

print("=" * 110)

print(
    "\nIMPORTANT:"
    "\n  Model and feature selection was locked in validation diagnostics."
    "\n  This stage evaluates that configuration ONCE on "
    "the protected 2026 T2 test period."
    "\n  Results from 2026 T2 must not be used for further "
    "model selection or tuning."
)


validate_files(
    [
        OVERALL_DATA_FILE,
        SUBJECT_DATA_FILE,
        OVERALL_FEATURE_AUDIT_FILE,
        SUBJECT_FEATURE_AUDIT_FILE,
        MODEL_SELECTION_FILE,
    ]
)


with open(
    MODEL_SELECTION_FILE,
    "r",
    encoding="utf-8",
) as file:
    validation_selection = json.load(file)


if validation_selection.get(
    "final_test_used"
) is not False:

    raise RuntimeError(
        "\nvalidation diagnostics selection record does not "
        "confirm final-test protection."
    )


overall_features = load_locked_features(
    OVERALL_FEATURE_AUDIT_FILE
)


subject_numeric_features, subject_categorical_features = (
    load_locked_features(
        SUBJECT_FEATURE_AUDIT_FILE,
        categorical_feature="target_subject",
    )
)


if not overall_features:
    raise RuntimeError(
        "\nNo locked overall features were found."
    )


if not subject_numeric_features:
    raise RuntimeError(
        "\nNo locked subject numeric features were found."
    )


print(
    f"\nLocked overall academic features: "
    f"{len(overall_features)}"
)

print(
    f"Locked subject numeric features: "
    f"{len(subject_numeric_features)}"
)

print(
    f"Locked subject categorical features: "
    f"{len(subject_categorical_features)}"
)


overall_df = pd.read_csv(
    OVERALL_DATA_FILE
)

subject_df = pd.read_csv(
    SUBJECT_DATA_FILE
)


print(
    f"\nOverall prediction rows: "
    f"{len(overall_df)}"
)

print(
    f"Subject prediction rows: "
    f"{len(subject_df)}"
)


missing_overall_features = sorted(
    set(
        overall_features
    ).difference(
        overall_df.columns
    )
)


if missing_overall_features:
    raise RuntimeError(
        "\nLocked overall features missing from dataset:\n"
        f"{missing_overall_features}"
    )


missing_subject_features = sorted(
    set(
        subject_numeric_features
        + subject_categorical_features
    ).difference(
        subject_df.columns
    )
)


if missing_subject_features:
    raise RuntimeError(
        "\nLocked subject features missing from dataset:\n"
        f"{missing_subject_features}"
    )


# Model selection is complete, so final training uses all development targets through 2026 T1.

overall_dev = (
    overall_df[
        overall_df[
            "dataset_split"
        ].isin(
            [
                "TRAIN",
                "VALIDATION",
            ]
        )
    ]
    .copy()
)


overall_test = (
    overall_df[
        overall_df[
            "dataset_split"
        ] == "FINAL_TEST"
    ]
    .copy()
)


subject_dev = (
    subject_df[
        subject_df[
            "dataset_split"
        ].isin(
            [
                "TRAIN",
                "VALIDATION",
            ]
        )
    ]
    .copy()
)


subject_test = (
    subject_df[
        subject_df[
            "dataset_split"
        ] == "FINAL_TEST"
    ]
    .copy()
)


if len(overall_dev) != 790:
    raise RuntimeError(
        "\nUnexpected overall development count."
        f"\nExpected: 790"
        f"\nActual: {len(overall_dev)}"
    )


if len(overall_test) != 268:
    raise RuntimeError(
        "\nUnexpected overall final-test count."
        f"\nExpected: 268"
        f"\nActual: {len(overall_test)}"
    )


if len(subject_dev) != 8416:
    raise RuntimeError(
        "\nUnexpected subject development count."
        f"\nExpected: 8416"
        f"\nActual: {len(subject_dev)}"
    )


if len(subject_test) != 2912:
    raise RuntimeError(
        "\nUnexpected subject final-test count."
        f"\nExpected: 2912"
        f"\nActual: {len(subject_test)}"
    )


print(
    "\nFinal training / test design:"
)

print(
    f"  Overall development rows: "
    f"{len(overall_dev)}"
)

print(
    f"  Overall final-test rows: "
    f"{len(overall_test)}"
)

print(
    f"  Subject development rows: "
    f"{len(subject_dev)}"
)

print(
    f"  Subject final-test rows: "
    f"{len(subject_test)}"
)


for feature in overall_features:

    overall_dev[
        feature
    ] = pd.to_numeric(
        overall_dev[
            feature
        ],
        errors="coerce",
    )

    overall_test[
        feature
    ] = pd.to_numeric(
        overall_test[
            feature
        ],
        errors="coerce",
    )


X_overall_dev = (
    overall_dev[
        overall_features
    ]
    .copy()
)


X_overall_test = (
    overall_test[
        overall_features
    ]
    .copy()
)


y_overall_dev = pd.to_numeric(
    overall_dev[
        "target_calculated_average"
    ],
    errors="raise",
)


y_overall_test = pd.to_numeric(
    overall_test[
        "target_calculated_average"
    ],
    errors="raise",
)


if y_overall_dev.isna().any():
    raise RuntimeError(
        "\nMissing overall development targets."
    )


if y_overall_test.isna().any():
    raise RuntimeError(
        "\nMissing overall final-test targets."
    )


for feature in subject_numeric_features:

    subject_dev[
        feature
    ] = pd.to_numeric(
        subject_dev[
            feature
        ],
        errors="coerce",
    )

    subject_test[
        feature
    ] = pd.to_numeric(
        subject_test[
            feature
        ],
        errors="coerce",
    )


for feature in subject_categorical_features:

    subject_dev[
        feature
    ] = (
        subject_dev[
            feature
        ]
        .astype(str)
    )

    subject_test[
        feature
    ] = (
        subject_test[
            feature
        ]
        .astype(str)
    )


subject_model_features = (
    subject_numeric_features
    + subject_categorical_features
)


X_subject_dev = (
    subject_dev[
        subject_model_features
    ]
    .copy()
)


X_subject_test = (
    subject_test[
        subject_model_features
    ]
    .copy()
)


y_subject_dev = pd.to_numeric(
    subject_dev[
        "target_mark"
    ],
    errors="raise",
)


y_subject_test = pd.to_numeric(
    subject_test[
        "target_mark"
    ],
    errors="raise",
)


if y_subject_dev.isna().any():
    raise RuntimeError(
        "\nMissing subject development targets."
    )


if y_subject_test.isna().any():
    raise RuntimeError(
        "\nMissing subject final-test targets."
    )


overall_model = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="median",
                add_indicator=True,
            ),
        ),
        (
            "model",
            RandomForestRegressor(
                **RF_PARAMS
            ),
        ),
    ]
)


subject_preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            SimpleImputer(
                strategy="median",
                add_indicator=True,
            ),
            subject_numeric_features,
        ),
        (
            "subject",
            make_one_hot_encoder(),
            subject_categorical_features,
        ),
    ],
    remainder="drop",
    sparse_threshold=0,
)


subject_model = Pipeline(
    steps=[
        (
            "preprocessor",
            subject_preprocessor,
        ),
        (
            "model",
            RandomForestRegressor(
                **RF_PARAMS
            ),
        ),
    ]
)


print(
    "\n"
    + "=" * 110
)

print(
    "TRAINING LOCKED FINAL MODELS ON ALL DEVELOPMENT DATA"
)

print(
    "=" * 110
)


print(
    "\nTraining final overall Random Forest..."
)

overall_model.fit(
    X_overall_dev,
    y_overall_dev,
)


print(
    "Training final pooled-subject Random Forest..."
)

subject_model.fit(
    X_subject_dev,
    y_subject_dev,
)


print(
    "\n"
    + "=" * 110
)

print(
    "EVALUATING PROTECTED 2026 T2 FINAL TEST"
)

print(
    "=" * 110
)


overall_dev_pred = overall_model.predict(
    X_overall_dev
)

overall_test_pred = overall_model.predict(
    X_overall_test
)


subject_dev_pred = subject_model.predict(
    X_subject_dev
)

subject_test_pred = subject_model.predict(
    X_subject_test
)


overall_dev_metrics = regression_metrics(
    y_overall_dev,
    overall_dev_pred,
)


overall_test_metrics = regression_metrics(
    y_overall_test,
    overall_test_pred,
)


subject_dev_metrics = regression_metrics(
    y_subject_dev,
    subject_dev_pred,
)


subject_test_metrics = regression_metrics(
    y_subject_test,
    subject_test_pred,
)


overall_metrics_df = pd.DataFrame(
    [
        {
            "evaluation_set":
                "DEVELOPMENT_TRAINING",
            **overall_dev_metrics,
        },
        {
            "evaluation_set":
                "FINAL_TEST_2026_T2",
            **overall_test_metrics,
        },
    ]
)


overall_metrics_df.to_csv(
    OVERALL_METRICS_OUTPUT,
    index=False,
)


subject_metrics_df = pd.DataFrame(
    [
        {
            "evaluation_set":
                "DEVELOPMENT_TRAINING",
            **subject_dev_metrics,
        },
        {
            "evaluation_set":
                "FINAL_TEST_2026_T2",
            **subject_test_metrics,
        },
    ]
)


subject_metrics_df.to_csv(
    SUBJECT_METRICS_OUTPUT,
    index=False,
)


overall_predictions = (
    overall_test[
        [
            "student_id",
            "academic_period",
            "grade",
            "target_calculated_average",
        ]
    ]
    .copy()
)


overall_predictions = (
    overall_predictions
    .rename(
        columns={
            "target_calculated_average":
                "actual_average"
        }
    )
)


overall_predictions[
    "predicted_average"
] = overall_test_pred


overall_predictions[
    "residual_actual_minus_predicted"
] = (
    overall_predictions[
        "actual_average"
    ]
    - overall_predictions[
        "predicted_average"
    ]
)


overall_predictions[
    "absolute_error"
] = np.abs(
    overall_predictions[
        "residual_actual_minus_predicted"
    ]
)


overall_predictions.to_csv(
    OVERALL_PREDICTIONS_OUTPUT,
    index=False,
)


subject_predictions = (
    subject_test[
        [
            "student_id",
            "academic_period",
            "grade",
            "target_subject",
            "target_mark",
        ]
    ]
    .copy()
)


subject_predictions = (
    subject_predictions
    .rename(
        columns={
            "target_mark":
                "actual_mark"
        }
    )
)


subject_predictions[
    "predicted_mark"
] = subject_test_pred


subject_predictions[
    "residual_actual_minus_predicted"
] = (
    subject_predictions[
        "actual_mark"
    ]
    - subject_predictions[
        "predicted_mark"
    ]
)


subject_predictions[
    "absolute_error"
] = np.abs(
    subject_predictions[
        "residual_actual_minus_predicted"
    ]
)


subject_predictions.to_csv(
    SUBJECT_PREDICTIONS_OUTPUT,
    index=False,
)


overall_grade_records = []


for grade, group in overall_predictions.groupby(
    "grade",
    sort=True,
):

    metrics = regression_metrics(
        group[
            "actual_average"
        ],
        group[
            "predicted_average"
        ],
    )

    overall_grade_records.append(
        {
            "grade":
                grade,

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


for grade, group in subject_predictions.groupby(
    "grade",
    sort=True,
):

    metrics = regression_metrics(
        group[
            "actual_mark"
        ],
        group[
            "predicted_mark"
        ],
    )

    subject_grade_records.append(
        {
            "grade":
                grade,

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


for target_subject, group in subject_predictions.groupby(
    "target_subject",
    sort=True,
):

    metrics = regression_metrics(
        group[
            "actual_mark"
        ],
        group[
            "predicted_mark"
        ],
    )

    subject_level_records.append(
        {
            "target_subject":
                target_subject,

            "sample_flag":
                subject_sample_flag(
                    len(group)
                ),

            **metrics,
        }
    )


subject_level_df = pd.DataFrame(
    subject_level_records
)


subject_level_df = (
    subject_level_df
    .sort_values(
        by=[
            "rows",
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


joblib.dump(
    {
        "pipeline":
            overall_model,

        "task":
            "overall_next_term_performance",

        "algorithm":
            "RandomForestRegressor",

        "feature_set":
            "ACADEMIC_HISTORY_ONLY",

        "features":
            overall_features,

        "training_periods": [
            "2025_T2",
            "2025_T3",
            "2026_T1",
        ],

        "final_test_period":
            "2026_T2",

        "random_state":
            RANDOM_STATE,
    },
    OVERALL_MODEL_OUTPUT,
)


joblib.dump(
    {
        "pipeline":
            subject_model,

        "task":
            "pooled_next_term_subject_mark",

        "algorithm":
            "RandomForestRegressor",

        "feature_set":
            "ACADEMIC_HISTORY_ONLY",

        "numeric_features":
            subject_numeric_features,

        "categorical_features":
            subject_categorical_features,

        "training_periods": [
            "2025_T2",
            "2025_T3",
            "2026_T1",
        ],

        "final_test_period":
            "2026_T2",

        "random_state":
            RANDOM_STATE,
    },
    SUBJECT_MODEL_OUTPUT,
)


summary = {
    "stage":
        "18",

    "selection_locked_before_final_test":
        True,

    "selected_algorithm":
        "Random Forest",

    "selected_feature_set":
        "ACADEMIC_HISTORY_ONLY",

    "selection_basis": {
        "overall_attendance_bootstrap":
            validation_selection[
                "overall"
            ],

        "subject_attendance_bootstrap":
            validation_selection[
                "subject"
            ],

        "reason":
            (
                "Attendance produced only marginal "
                "Random Forest validation improvements and "
                "the paired bootstrap confidence intervals "
                "crossed zero for both forecasting tasks. "
                "The simpler academic-only configuration "
                "was therefore selected before final-test "
                "evaluation."
            ),
    },

    "development_training_periods": [
        "2025_T2",
        "2025_T3",
        "2026_T1",
    ],

    "final_out_of_time_test_period":
        "2026_T2",

    "overall": {
        "development_rows":
            len(
                overall_dev
            ),

        "final_test_rows":
            len(
                overall_test
            ),

        "feature_count":
            len(
                overall_features
            ),

        "development_metrics":
            overall_dev_metrics,

        "final_test_metrics":
            overall_test_metrics,
    },

    "subject": {
        "development_rows":
            len(
                subject_dev
            ),

        "final_test_rows":
            len(
                subject_test
            ),

        "numeric_feature_count":
            len(
                subject_numeric_features
            ),

        "categorical_features":
            subject_categorical_features,

        "final_test_subject_count":
            int(
                subject_predictions[
                    "target_subject"
                ]
                .nunique()
            ),

        "development_metrics":
            subject_dev_metrics,

        "final_test_metrics":
            subject_test_metrics,
    },

    "post_test_rule":
        (
            "The 2026 T2 final-test results are for final "
            "reporting only and must not be used to retune "
            "or reselect the model."
        ),
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


# Overall actual vs predicted

fig, ax = plt.subplots(
    figsize=(8, 7)
)


ax.scatter(
    overall_predictions[
        "actual_average"
    ],
    overall_predictions[
        "predicted_average"
    ],
    alpha=0.65,
    color="#6F42C1",
)


minimum = min(
    overall_predictions[
        "actual_average"
    ].min(),
    overall_predictions[
        "predicted_average"
    ].min(),
)


maximum = max(
    overall_predictions[
        "actual_average"
    ].max(),
    overall_predictions[
        "predicted_average"
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
    color="#4B2E83",
)


ax.set_title(
    "Final Out-of-Time Overall Forecast\n"
    "2026 T2 Actual vs Predicted",
    fontweight="bold",
)


ax.set_xlabel(
    "Actual Term Average"
)


ax.set_ylabel(
    "Predicted Term Average"
)


ax.grid(
    alpha=0.20
)


fig.tight_layout()


fig.savefig(
    FIGURE_DIR
    / "final_overall_actual_vs_predicted.png",
    dpi=300,
    bbox_inches="tight",
)


plt.close(
    fig
)


# Subject absolute error distribution

fig, ax = plt.subplots(
    figsize=(9, 6)
)


ax.hist(
    subject_predictions[
        "absolute_error"
    ],
    bins=30,
    color="#6F42C1",
    edgecolor="#4B2E83",
)


ax.set_title(
    "Final Out-of-Time Subject Forecast\n"
    "2026 T2 Absolute Error Distribution",
    fontweight="bold",
)


ax.set_xlabel(
    "Absolute Error (marks)"
)


ax.set_ylabel(
    "Prediction Count"
)


ax.grid(
    axis="y",
    alpha=0.20,
)


fig.tight_layout()


fig.savefig(
    FIGURE_DIR
    / "final_subject_absolute_error_distribution.png",
    dpi=300,
    bbox_inches="tight",
)


plt.close(
    fig
)


# Subject MAE with adequate test support only

subject_plot = (
    subject_level_df[
        subject_level_df[
            "rows"
        ] >= 20
    ]
    .sort_values(
        "mae",
        ascending=True,
    )
    .copy()
)


if not subject_plot.empty:

    fig_height = max(
        7,
        len(
            subject_plot
        )
        * 0.45,
    )

    fig, ax = plt.subplots(
        figsize=(
            10,
            fig_height,
        )
    )

    bars = ax.barh(
        subject_plot[
            "target_subject"
        ],
        subject_plot[
            "mae"
        ],
        color="#6F42C1",
        edgecolor="#4B2E83",
    )

    ax.set_title(
        "Final 2026 T2 Subject-Level MAE",
        fontweight="bold",
    )

    ax.set_xlabel(
        "MAE (marks)"
    )

    ax.set_ylabel(
        "Subject"
    )

    ax.grid(
        axis="x",
        alpha=0.20,
    )

    for bar in bars:

        value = (
            bar.get_width()
        )

        ax.text(
            value,
            bar.get_y()
            + bar.get_height() / 2,
            f" {value:.2f}",
            va="center",
            fontsize=9,
        )

    fig.tight_layout()

    fig.savefig(
        FIGURE_DIR
        / "final_subject_mae_by_subject.png",
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
    "FINAL OUT-OF-TIME EVALUATION FINAL OUT-OF-TIME EVALUATION COMPLETE"
)

print(
    "=" * 110
)


print(
    "\nLOCKED CONFIGURATION"
)

print(
    "  Algorithm: Random Forest"
)

print(
    "  Feature set: Academic/history only"
)

print(
    "  Attendance predictors: NOT USED"
)

print(
    "  Development training periods: "
    "2025_T2 + 2025_T3 + 2026_T1"
)

print(
    "  Final out-of-time test: 2026_T2"
)


print(
    "\nOVERALL PERFORMANCE"
)

print(
    f"  Development rows: "
    f"{len(overall_dev)}"
)

print(
    f"  Final-test rows: "
    f"{len(overall_test)}"
)

print(
    f"  Academic features: "
    f"{len(overall_features)}"
)

print(
    "\n  Development:"
)

print(
    f"    MAE  = "
    f"{overall_dev_metrics['mae']:.4f}"
)

print(
    f"    RMSE = "
    f"{overall_dev_metrics['rmse']:.4f}"
)

print(
    f"    R²   = "
    f"{overall_dev_metrics['r2']:.4f}"
)

print(
    "\n  FINAL 2026 T2:"
)

print(
    f"    MAE  = "
    f"{overall_test_metrics['mae']:.4f}"
)

print(
    f"    RMSE = "
    f"{overall_test_metrics['rmse']:.4f}"
)

print(
    f"    R²   = "
    f"{overall_test_metrics['r2']:.4f}"
)

print(
    f"    Within ±5 marks  = "
    f"{overall_test_metrics['within_5_marks_pct']:.2f}%"
)

print(
    f"    Within ±10 marks = "
    f"{overall_test_metrics['within_10_marks_pct']:.2f}%"
)

print(
    f"    Mean residual "
    f"(actual-predicted) = "
    f"{overall_test_metrics['mean_residual_actual_minus_predicted']:.4f}"
)


print(
    "\nPOOLED SUBJECT PERFORMANCE"
)

print(
    f"  Development rows: "
    f"{len(subject_dev)}"
)

print(
    f"  Final-test rows: "
    f"{len(subject_test)}"
)

print(
    f"  Subjects: "
    f"{subject_predictions['target_subject'].nunique()}"
)

print(
    f"  Numeric academic features: "
    f"{len(subject_numeric_features)}"
)

print(
    "\n  Development:"
)

print(
    f"    MAE  = "
    f"{subject_dev_metrics['mae']:.4f}"
)

print(
    f"    RMSE = "
    f"{subject_dev_metrics['rmse']:.4f}"
)

print(
    f"    R²   = "
    f"{subject_dev_metrics['r2']:.4f}"
)

print(
    "\n  FINAL 2026 T2:"
)

print(
    f"    MAE  = "
    f"{subject_test_metrics['mae']:.4f}"
)

print(
    f"    RMSE = "
    f"{subject_test_metrics['rmse']:.4f}"
)

print(
    f"    R²   = "
    f"{subject_test_metrics['r2']:.4f}"
)

print(
    f"    Within ±5 marks  = "
    f"{subject_test_metrics['within_5_marks_pct']:.2f}%"
)

print(
    f"    Within ±10 marks = "
    f"{subject_test_metrics['within_10_marks_pct']:.2f}%"
)

print(
    f"    Mean residual "
    f"(actual-predicted) = "
    f"{subject_test_metrics['mean_residual_actual_minus_predicted']:.4f}"
)


print(
    "\nFINAL OVERALL MAE BY GRADE"
)

print(
    overall_grade_df[
        [
            "grade",
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
    "\nFINAL SUBJECT MAE BY GRADE"
)

print(
    subject_grade_df[
        [
            "grade",
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
    "\nFINAL SUBJECT-LEVEL RESULTS "
    "(>=20 TEST RECORDS)"
)

print(
    subject_level_df[
        subject_level_df[
            "rows"
        ] >= 20
    ]
    [
        [
            "target_subject",
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
    "\nFINAL MODEL ARTIFACTS"
)

print(
    f"  {OVERALL_MODEL_OUTPUT}"
)

print(
    f"  {SUBJECT_MODEL_OUTPUT}"
)


print(
    "\nAGGREGATE REPORTS"
)


for output in [
    OVERALL_METRICS_OUTPUT,
    OVERALL_GRADE_OUTPUT,
    SUBJECT_METRICS_OUTPUT,
    SUBJECT_GRADE_OUTPUT,
    SUBJECT_LEVEL_OUTPUT,
    SUMMARY_OUTPUT,
]:

    print(
        f"  {output}"
    )


print(
    "\nPRIVATE STUDENT-LEVEL PREDICTIONS"
)

print(
    f"  {OVERALL_PREDICTIONS_OUTPUT}"
)

print(
    f"  {SUBJECT_PREDICTIONS_OUTPUT}"
)


print(
    "\nFIGURES"
)

print(
    f"  {FIGURE_DIR}"
)


print(
    "\nCRITICAL RESEARCH RULE:"
)

print(
    "  2026 T2 has now been used as the final "
    "out-of-time test set."
)

print(
    "  Do NOT tune, reselect, or modify the model "
    "based on these final-test results."
)

print(
    "  Any future model changes require a new "
    "independent future evaluation period."
)
