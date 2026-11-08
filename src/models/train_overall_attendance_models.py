from pathlib import Path
import json

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBRegressor
except ImportError as exc:
    raise ImportError(
        "\nXGBoost is not installed.\n"
        "Install it inside the project virtual environment with:\n\n"
        "    pip install xgboost\n"
    ) from exc


RANDOM_STATE = 42

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "model_ready"
    / "overall_prediction_dataset.csv"
)

ACADEMIC_METRICS_FILE = (
    PROJECT_ROOT
    / "reports"
    / "modeling"
    / "overall_academic_validation_metrics.csv"
)

REPORT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "modeling"
)

FIGURE_DIR = (
    PROJECT_ROOT
    / "figures"
    / "modeling"
)

MODEL_DIR = (
    PROJECT_ROOT
    / "models"
    / "candidates"
    / "attendance"
)

PRIVATE_OUTPUT_DIR = (
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

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


METRICS_OUTPUT = (
    REPORT_DIR
    / "overall_attendance_validation_metrics.csv"
)

COMPARISON_OUTPUT = (
    REPORT_DIR
    / "overall_feature_set_comparison.csv"
)

FEATURE_AUDIT_OUTPUT = (
    REPORT_DIR
    / "overall_attendance_feature_audit.csv"
)

CONFIG_OUTPUT = (
    REPORT_DIR
    / "overall_attendance_model_config.json"
)

VALIDATION_PREDICTIONS_OUTPUT = (
    PRIVATE_OUTPUT_DIR
    / "attendance_overall_attendance_validation_predictions.csv"
)


EXCLUDED_METADATA_COLUMNS = {
    "student_id",
    "year",
    "term",
    "academic_period",
    "period_order",
    "class",
    "academic_stage",
    "dataset_split",
    "target_calculated_average",
    "target_available",
}


# Historical attendance features only, never current-term.

EXPECTED_ATTENDANCE_FEATURES = {
    "prev_attendance_pct",
    "historical_mean_attendance_pct",
    "historical_std_attendance_pct",
    "attendance_trend",
    "prior_attendance_observation_count",
    "prev_days_absent",
    "historical_mean_days_absent",
    "prev_attendance_coverage_pct",
    "prev_attendance_available",
}


EXCLUDED_SUBJECT_SLUGS = {
    "home_science",
    "pts",
}


def is_attendance_feature(column):
    """Identify historical attendance-related predictor."""

    if column in EXPECTED_ATTENDANCE_FEATURES:
        return True

    lower = column.lower()

    terms = [
        "attendance",
        "days_absent",
        "days_present",
    ]

    return any(
        term in lower
        for term in terms
    )


def belongs_to_excluded_subject(column):
    """Detect features belonging to Home Science or PTS."""

    lower = column.lower()

    for subject in EXCLUDED_SUBJECT_SLUGS:

        patterns = [
            f"_{subject}",
            f"{subject}_",
        ]

        if any(
            pattern in lower
            for pattern in patterns
        ):
            return True

    return False


def calculate_metrics(
    y_true,
    y_pred,
):
    """Standard regression metrics."""

    mae = mean_absolute_error(
        y_true,
        y_pred,
    )

    mse = mean_squared_error(
        y_true,
        y_pred,
    )

    rmse = float(
        np.sqrt(
            mse
        )
    )

    r2 = r2_score(
        y_true,
        y_pred,
    )

    below_zero = int(
        (
            np.asarray(
                y_pred
            )
            < 0
        ).sum()
    )

    above_hundred = int(
        (
            np.asarray(
                y_pred
            )
            > 100
        ).sum()
    )

    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "r2": float(r2),
        "predictions_below_0":
            below_zero,
        "predictions_above_100":
            above_hundred,
    }


def save_validation_metric_figure(
    metrics_df,
    metric,
    ylabel,
    filename,
):
    """attendance feature comparison validation metric figure."""

    plot_df = (
        metrics_df[
            metrics_df[
                "evaluation_set"
            ] == "VALIDATION"
        ]
        .copy()
    )

    plot_df = (
        plot_df
        .sort_values(
            metric,
            ascending=(
                metric != "r2"
            ),
        )
    )

    fig, ax = plt.subplots(
        figsize=(9, 6)
    )

    bars = ax.bar(
        plot_df["model"],
        plot_df[metric],
        color="#6F42C1",
        edgecolor="#4B2E83",
    )

    ax.set_title(
        f"Academic + Attendance — "
        f"2026 T1 Validation {ylabel}",
        fontsize=13,
        fontweight="bold",
    )

    ax.set_xlabel(
        "Model"
    )

    ax.set_ylabel(
        ylabel
    )

    ax.grid(
        axis="y",
        alpha=0.20,
    )

    for bar in bars:

        value = bar.get_height()

        ax.text(
            bar.get_x()
            + bar.get_width() / 2,
            value,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    fig.tight_layout()

    fig.savefig(
        FIGURE_DIR
        / filename,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )


def save_feature_comparison_figure(
    comparison_df,
):
    """Compare validation MAE of the academic-only and academic + attendance models."""

    models = (
        comparison_df[
            "model"
        ]
        .tolist()
    )

    academic_mae = (
        comparison_df[
            "academic_only_mae"
        ]
        .to_numpy()
    )

    attendance_mae = (
        comparison_df[
            "academic_plus_attendance_mae"
        ]
        .to_numpy()
    )

    positions = np.arange(
        len(models)
    )

    width = 0.36

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    bars1 = ax.bar(
        positions - width / 2,
        academic_mae,
        width,
        label="Academic history only",
        color="#D8C7F0",
        edgecolor="#4B2E83",
    )

    bars2 = ax.bar(
        positions + width / 2,
        attendance_mae,
        width,
        label="Academic history + attendance",
        color="#6F42C1",
        edgecolor="#4B2E83",
    )

    ax.set_title(
        "Effect of Attendance Features on "
        "2026 T1 Validation MAE",
        fontsize=13,
        fontweight="bold",
    )

    ax.set_ylabel(
        "Mean Absolute Error (marks)"
    )

    ax.set_xlabel(
        "Model"
    )

    ax.set_xticks(
        positions
    )

    ax.set_xticklabels(
        models
    )

    ax.legend()

    ax.grid(
        axis="y",
        alpha=0.20,
    )

    for bars in [
        bars1,
        bars2,
    ]:

        for bar in bars:

            value = bar.get_height()

            ax.text(
                bar.get_x()
                + bar.get_width() / 2,
                value,
                f"{value:.2f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    fig.tight_layout()

    fig.savefig(
        FIGURE_DIR
        / "overall_academic_vs_attendance_mae.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )


print(
    "=" * 110
)

print(
    "STUDENT PERFORMANCE AI - "
    "ATTENDANCE FEATURE COMPARISONA OVERALL ACADEMIC + ATTENDANCE COMPARISON"
)

print(
    "=" * 110
)


if not INPUT_FILE.exists():

    raise FileNotFoundError(
        "\nOverall prediction dataset not found:\n"
        f"{INPUT_FILE}"
    )


if not ACADEMIC_METRICS_FILE.exists():

    raise FileNotFoundError(
        "\nacademic baseline academic-only metrics not found:\n"
        f"{ACADEMIC_METRICS_FILE}\n\n"
        "Run academic baselineA first."
    )


df = pd.read_csv(
    INPUT_FILE
)


academic_metrics = pd.read_csv(
    ACADEMIC_METRICS_FILE
)


print(
    f"\nInput rows: "
    f"{len(df)}"
)


print(
    f"Columns available: "
    f"{len(df.columns)}"
)


train_df = (
    df[
        df[
            "dataset_split"
        ] == "TRAIN"
    ]
    .copy()
)


validation_df = (
    df[
        df[
            "dataset_split"
        ] == "VALIDATION"
    ]
    .copy()
)


final_test_df = (
    df[
        df[
            "dataset_split"
        ] == "FINAL_TEST"
    ]
)


if len(train_df) != 573:

    raise RuntimeError(
        "\nUnexpected TRAIN count.\n"
        f"Expected: 573\n"
        f"Actual: {len(train_df)}"
    )


if len(validation_df) != 217:

    raise RuntimeError(
        "\nUnexpected VALIDATION count.\n"
        f"Expected: 217\n"
        f"Actual: {len(validation_df)}"
    )


if len(final_test_df) != 268:

    raise RuntimeError(
        "\nUnexpected FINAL_TEST count.\n"
        f"Expected: 268\n"
        f"Actual: {len(final_test_df)}"
    )


print(
    "\nTemporal dataset split:"
)


print(
    f"  TRAIN: "
    f"{len(train_df)}"
)


print(
    f"  VALIDATION: "
    f"{len(validation_df)}"
)


print(
    f"  FINAL_TEST: "
    f"{len(final_test_df)} "
    "(protected — not used)"
)


y_train = pd.to_numeric(
    train_df[
        "target_calculated_average"
    ],
    errors="raise",
)


y_validation = pd.to_numeric(
    validation_df[
        "target_calculated_average"
    ],
    errors="raise",
)


if y_train.isna().any():

    raise RuntimeError(
        "Missing target in TRAIN."
    )


if y_validation.isna().any():

    raise RuntimeError(
        "Missing target in VALIDATION."
    )


feature_audit_records = []

candidate_features = []


for column in df.columns:

    if column in EXCLUDED_METADATA_COLUMNS:

        feature_audit_records.append(
            {
                "feature":
                    column,

                "status":
                    "EXCLUDED",

                "reason":
                    "metadata_or_target",

                "feature_group":
                    "METADATA",
            }
        )

        continue


    if belongs_to_excluded_subject(
        column
    ):

        feature_audit_records.append(
            {
                "feature":
                    column,

                "status":
                    "EXCLUDED",

                "reason":
                    (
                        "subject_not_suitable_for_"
                        "longitudinal_modelling"
                    ),

                "feature_group":
                    "ACADEMIC",
            }
        )

        continue


    candidate_features.append(
        column
    )


usable_features = []

attendance_features_used = []

academic_features_used = []


for column in candidate_features:

    train_numeric = pd.to_numeric(
        train_df[
            column
        ],
        errors="coerce",
    )

    validation_numeric = pd.to_numeric(
        validation_df[
            column
        ],
        errors="coerce",
    )


    if train_numeric.notna().sum() == 0:

        feature_audit_records.append(
            {
                "feature":
                    column,

                "status":
                    "EXCLUDED",

                "reason":
                    "all_missing_in_training",

                "feature_group":
                    (
                        "ATTENDANCE"
                        if is_attendance_feature(
                            column
                        )
                        else "ACADEMIC"
                    ),
            }
        )

        continue


    if (
        train_numeric
        .dropna()
        .nunique()
        <= 1
    ):

        feature_audit_records.append(
            {
                "feature":
                    column,

                "status":
                    "EXCLUDED",

                "reason":
                    "constant_in_training",

                "feature_group":
                    (
                        "ATTENDANCE"
                        if is_attendance_feature(
                            column
                        )
                        else "ACADEMIC"
                    ),
            }
        )

        continue


    train_df[
        column
    ] = train_numeric


    validation_df[
        column
    ] = validation_numeric


    usable_features.append(
        column
    )


    if is_attendance_feature(
        column
    ):

        attendance_features_used.append(
            column
        )

        feature_group = (
            "ATTENDANCE"
        )

    else:

        academic_features_used.append(
            column
        )

        feature_group = (
            "ACADEMIC"
        )


    feature_audit_records.append(
        {
            "feature":
                column,

            "status":
                "INCLUDED",

            "reason":
                "historical_predictor",

            "feature_group":
                feature_group,
        }
    )


if not attendance_features_used:

    raise RuntimeError(
        "\nNo attendance features were selected. "
        "attendance feature comparison comparison cannot proceed."
    )


feature_audit_df = pd.DataFrame(
    feature_audit_records
)


train_missing = {}

validation_missing = {}


for column in df.columns:

    if column in train_df.columns:

        train_missing[
            column
        ] = round(
            float(
                train_df[
                    column
                ]
                .isna()
                .mean()
                * 100
            ),
            4,
        )


    if column in validation_df.columns:

        validation_missing[
            column
        ] = round(
            float(
                validation_df[
                    column
                ]
                .isna()
                .mean()
                * 100
            ),
            4,
        )


feature_audit_df[
    "train_missing_pct"
] = (
    feature_audit_df[
        "feature"
    ]
    .map(
        train_missing
    )
)


feature_audit_df[
    "validation_missing_pct"
] = (
    feature_audit_df[
        "feature"
    ]
    .map(
        validation_missing
    )
)


feature_audit_df = (
    feature_audit_df
    .sort_values(
        by=[
            "status",
            "feature_group",
            "feature",
        ]
    )
    .reset_index(
        drop=True
    )
)


feature_audit_df.to_csv(
    FEATURE_AUDIT_OUTPUT,
    index=False,
)


X_train = (
    train_df[
        usable_features
    ]
    .copy()
)


X_validation = (
    validation_df[
        usable_features
    ]
    .copy()
)


# LEAKAGE VALIDATION

forbidden_features = {
    "student_id",
    "year",
    "term",
    "academic_period",
    "period_order",
    "class",
    "academic_stage",
    "dataset_split",
    "target_calculated_average",
    "target_available",
}


leaked_features = sorted(
    forbidden_features.intersection(
        X_train.columns
    )
)


if leaked_features:

    raise RuntimeError(
        "\nLeakage detected:\n"
        f"{leaked_features}"
    )


available_expected_attendance = sorted(
    EXPECTED_ATTENDANCE_FEATURES.intersection(
        X_train.columns
    )
)


print(
    "\nAttendance features detected:"
)


for feature in sorted(
    attendance_features_used
):

    print(
        f"  - {feature}"
    )


print(
    f"\nAttendance features used: "
    f"{len(attendance_features_used)}"
)


print(
    f"Academic/history features used: "
    f"{len(academic_features_used)}"
)


print(
    f"Total model features used: "
    f"{len(usable_features)}"
)


# Same settings as the academic baseline for a fair comparison.

linear_model = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="median",
                add_indicator=True,
            ),
        ),
        (
            "scaler",
            StandardScaler(),
        ),
        (
            "model",
            LinearRegression(),
        ),
    ]
)


random_forest_model = Pipeline(
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
                n_estimators=500,
                max_depth=None,
                min_samples_split=4,
                min_samples_leaf=2,
                max_features="sqrt",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
        ),
    ]
)


xgboost_model = Pipeline(
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
            XGBRegressor(
                objective="reg:squarederror",
                n_estimators=500,
                learning_rate=0.03,
                max_depth=4,
                min_child_weight=3,
                subsample=0.85,
                colsample_bytree=0.85,
                reg_alpha=0.0,
                reg_lambda=1.0,
                random_state=RANDOM_STATE,
                n_jobs=-1,
                verbosity=0,
            ),
        ),
    ]
)


models = {
    "Linear Regression":
        linear_model,

    "Random Forest":
        random_forest_model,

    "XGBoost":
        xgboost_model,
}


print(
    "\n"
    + "=" * 110
)

print(
    "MODEL TRAINING — ACADEMIC HISTORY + ATTENDANCE"
)

print(
    "=" * 110
)


metric_records = []


validation_predictions = (
    validation_df[
        [
            "student_id",
            "academic_period",
            "grade",
            "target_calculated_average",
        ]
    ]
    .copy()
)


validation_predictions = (
    validation_predictions
    .rename(
        columns={
            "target_calculated_average":
                "actual_average"
        }
    )
)


for model_name, pipeline in models.items():

    print(
        f"\nTraining: "
        f"{model_name}"
    )


    pipeline.fit(
        X_train,
        y_train,
    )


    train_pred = pipeline.predict(
        X_train
    )


    train_metrics = calculate_metrics(
        y_train,
        train_pred,
    )


    metric_records.append(
        {
            "model":
                model_name,

            "feature_set":
                "ACADEMIC_HISTORY_PLUS_ATTENDANCE",

            "evaluation_set":
                "TRAIN",

            "rows":
                len(
                    y_train
                ),

            **train_metrics,
        }
    )


    validation_pred = pipeline.predict(
        X_validation
    )


    validation_metrics = calculate_metrics(
        y_validation,
        validation_pred,
    )


    metric_records.append(
        {
            "model":
                model_name,

            "feature_set":
                "ACADEMIC_HISTORY_PLUS_ATTENDANCE",

            "evaluation_set":
                "VALIDATION",

            "rows":
                len(
                    y_validation
                ),

            **validation_metrics,
        }
    )


    prediction_column = (
        "prediction_"
        + model_name
        .lower()
        .replace(
            " ",
            "_"
        )
    )


    validation_predictions[
        prediction_column
    ] = validation_pred


    model_filename = (
        model_name
        .lower()
        .replace(
            " ",
            "_"
        )
        + "_academic_plus_attendance.joblib"
    )


    joblib.dump(
        {
            "pipeline":
                pipeline,

            "features":
                usable_features,

            "academic_features":
                academic_features_used,

            "attendance_features":
                attendance_features_used,

            "feature_set":
                "ACADEMIC_HISTORY_PLUS_ATTENDANCE",

            "training_target_periods": [
                "2025_T2",
                "2025_T3",
            ],

            "validation_target_period":
                "2026_T1",

            "final_test_used":
                False,
        },
        MODEL_DIR
        / model_filename,
    )


    print(
        f"  TRAIN      "
        f"MAE={train_metrics['mae']:.4f} | "
        f"RMSE={train_metrics['rmse']:.4f} | "
        f"R²={train_metrics['r2']:.4f}"
    )


    print(
        f"  VALIDATION "
        f"MAE={validation_metrics['mae']:.4f} | "
        f"RMSE={validation_metrics['rmse']:.4f} | "
        f"R²={validation_metrics['r2']:.4f}"
    )


metrics_df = pd.DataFrame(
    metric_records
)


validation_metrics_df = (
    metrics_df[
        metrics_df[
            "evaluation_set"
        ] == "VALIDATION"
    ]
    .copy()
)


validation_metrics_df = (
    validation_metrics_df
    .sort_values(
        by=[
            "mae",
            "rmse",
        ],
        ascending=[
            True,
            True,
        ],
    )
    .reset_index(
        drop=True
    )
)


validation_metrics_df[
    "validation_rank"
] = np.arange(
    1,
    len(
        validation_metrics_df
    ) + 1,
)


metrics_df = metrics_df.merge(
    validation_metrics_df[
        [
            "model",
            "validation_rank",
        ]
    ],
    on="model",
    how="left",
)


metrics_df.to_csv(
    METRICS_OUTPUT,
    index=False,
)


academic_validation = (
    academic_metrics[
        academic_metrics[
            "evaluation_set"
        ] == "VALIDATION"
    ]
    [
        [
            "model",
            "mae",
            "rmse",
            "r2",
        ]
    ]
    .copy()
)


academic_validation = (
    academic_validation
    .rename(
        columns={
            "mae":
                "academic_only_mae",

            "rmse":
                "academic_only_rmse",

            "r2":
                "academic_only_r2",
        }
    )
)


attendance_validation = (
    validation_metrics_df[
        [
            "model",
            "mae",
            "rmse",
            "r2",
        ]
    ]
    .copy()
)


attendance_validation = (
    attendance_validation
    .rename(
        columns={
            "mae":
                "academic_plus_attendance_mae",

            "rmse":
                "academic_plus_attendance_rmse",

            "r2":
                "academic_plus_attendance_r2",
        }
    )
)


comparison_df = academic_validation.merge(
    attendance_validation,
    on="model",
    how="inner",
)


# Positive improvement = attendance improved result.

comparison_df[
    "mae_improvement"
] = (
    comparison_df[
        "academic_only_mae"
    ]
    - comparison_df[
        "academic_plus_attendance_mae"
    ]
)


comparison_df[
    "rmse_improvement"
] = (
    comparison_df[
        "academic_only_rmse"
    ]
    - comparison_df[
        "academic_plus_attendance_rmse"
    ]
)


comparison_df[
    "r2_improvement"
] = (
    comparison_df[
        "academic_plus_attendance_r2"
    ]
    - comparison_df[
        "academic_only_r2"
    ]
)


comparison_df[
    "mae_improvement_pct"
] = (
    comparison_df[
        "mae_improvement"
    ]
    / comparison_df[
        "academic_only_mae"
    ]
    * 100
)


comparison_df[
    "attendance_improved_mae"
] = (
    comparison_df[
        "mae_improvement"
    ]
    > 0
)


comparison_df = (
    comparison_df
    .sort_values(
        by="academic_plus_attendance_mae",
        ascending=True,
    )
    .reset_index(
        drop=True
    )
)


comparison_df.to_csv(
    COMPARISON_OUTPUT,
    index=False,
)


validation_predictions.to_csv(
    VALIDATION_PREDICTIONS_OUTPUT,
    index=False,
)


model_config = {
    "stage":
        "16A",

    "task":
        "overall_next_term_performance_regression",

    "experiment":
        "academic_history_plus_attendance",

    "comparison_baseline":
        "academic_academic_history_only",

    "random_state":
        RANDOM_STATE,

    "training_target_periods": [
        "2025_T2",
        "2025_T3",
    ],

    "validation_target_period":
        "2026_T1",

    "protected_final_test_period":
        "2026_T2",

    "train_rows":
        len(
            train_df
        ),

    "validation_rows":
        len(
            validation_df
        ),

    "protected_final_test_rows":
        len(
            final_test_df
        ),

    "academic_feature_count":
        len(
            academic_features_used
        ),

    "attendance_feature_count":
        len(
            attendance_features_used
        ),

    "total_feature_count":
        len(
            usable_features
        ),

    "attendance_features":
        sorted(
            attendance_features_used
        ),

    "final_test_used":
        False,

    "term_difficulty_context": {
        "known_school_assumption":
            (
                "Later terms may involve harder examinations."
            ),

        "included_as_predictor":
            False,

        "reason":
            (
                "Available longitudinal data does not contain "
                "enough repeated years to estimate a stable "
                "T1/T2/T3 difficulty effect without imposing "
                "an unsupported manual adjustment."
            ),
    },

    "models": {
        "Linear Regression": {
            "same_configuration_as_academic":
                True,
        },

        "Random Forest": {
            "n_estimators":
                500,

            "min_samples_split":
                4,

            "min_samples_leaf":
                2,

            "max_features":
                "sqrt",
        },

        "XGBoost": {
            "n_estimators":
                500,

            "learning_rate":
                0.03,

            "max_depth":
                4,

            "min_child_weight":
                3,

            "subsample":
                0.85,

            "colsample_bytree":
                0.85,
        },
    },
}


with open(
    CONFIG_OUTPUT,
    "w",
    encoding="utf-8",
) as file:

    json.dump(
        model_config,
        file,
        indent=4,
    )


save_validation_metric_figure(
    metrics_df,
    metric="mae",
    ylabel="Mean Absolute Error (MAE)",
    filename=(
        "overall_attendance_validation_mae.png"
    ),
)


save_validation_metric_figure(
    metrics_df,
    metric="rmse",
    ylabel="Root Mean Squared Error (RMSE)",
    filename=(
        "overall_attendance_validation_rmse.png"
    ),
)


save_validation_metric_figure(
    metrics_df,
    metric="r2",
    ylabel="R²",
    filename=(
        "overall_attendance_validation_r2.png"
    ),
)


save_feature_comparison_figure(
    comparison_df
)


print(
    "\n"
    + "=" * 110
)

print(
    "ATTENDANCE FEATURE COMPARISONA OVERALL FEATURE COMPARISON COMPLETE"
)

print(
    "=" * 110
)


print(
    "\nExperiment:"
)


print(
    "  Academic/history-only baseline "
    "vs academic/history + attendance"
)


print(
    "\nTemporal design:"
)


print(
    "  TRAIN: 2025_T2 + 2025_T3"
)


print(
    "  VALIDATION: 2026_T1"
)


print(
    "  FINAL_TEST: 2026_T2 — PROTECTED"
)


print(
    "\nFeatures:"
)


print(
    f"  Academic/history: "
    f"{len(academic_features_used)}"
)


print(
    f"  Attendance: "
    f"{len(attendance_features_used)}"
)


print(
    f"  Total: "
    f"{len(usable_features)}"
)


print(
    "\nattendance feature comparison validation ranking:"
)


print(
    validation_metrics_df[
        [
            "model",
            "mae",
            "rmse",
            "r2",
            "predictions_below_0",
            "predictions_above_100",
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
    "\nAcademic-only vs attendance comparison:"
)


print(
    comparison_df[
        [
            "model",
            "academic_only_mae",
            "academic_plus_attendance_mae",
            "mae_improvement",
            "mae_improvement_pct",
            "academic_only_r2",
            "academic_plus_attendance_r2",
            "r2_improvement",
        ]
    ]
    .to_string(
        index=False,
        float_format=lambda x: (
            f"{x:.4f}"
        ),
    )
)


best_model = (
    validation_metrics_df
    .iloc[0][
        "model"
    ]
)


print(
    f"\nLowest attendance feature comparison validation MAE: "
    f"{best_model}"
)


improved_models = (
    comparison_df[
        comparison_df[
            "attendance_improved_mae"
        ]
    ][
        "model"
    ]
    .tolist()
)


print(
    "\nModels where attendance improved validation MAE:"
)


if improved_models:

    for model in improved_models:

        print(
            f"  - {model}"
        )

else:

    print(
        "  None"
    )


print(
    "\nTerm difficulty note:"
)


print(
    "  School expectation that later terms may be "
    "harder has been documented."
)


print(
    "  No manual mark penalty or forced term "
    "adjustment was applied."
)


print(
    "\nIMPORTANT:"
)


print(
    "  Final model has NOT yet been selected."
)


print(
    "  2026 T2 was not fitted, predicted, "
    "scored, or inspected."
)


print(
    "\nOutputs:"
)


print(
    f"  {METRICS_OUTPUT}"
)


print(
    f"  {COMPARISON_OUTPUT}"
)


print(
    f"  {FEATURE_AUDIT_OUTPUT}"
)


print(
    f"  {CONFIG_OUTPUT}"
)


print(
    f"  {FIGURE_DIR}"
)


print(
    "\nPrivate validation predictions:"
)


print(
    f"  {VALIDATION_PREDICTIONS_OUTPUT}"
)
