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
    / "academic"
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
    / "overall_academic_validation_metrics.csv"
)

FEATURE_AUDIT_OUTPUT = (
    REPORT_DIR
    / "overall_academic_feature_audit.csv"
)

CONFIG_OUTPUT = (
    REPORT_DIR
    / "overall_academic_model_config.json"
)

VALIDATION_PREDICTIONS_OUTPUT = (
    PRIVATE_OUTPUT_DIR
    / "academic_overall_academic_validation_predictions.csv"
)


# Metadata and target fields excluded from the feature matrix.

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


# Attendance columns are excluded here and compared later.

ATTENDANCE_FEATURES = {
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


# Subjects without consistent numeric marks stay in cleaned data but are not model features.

EXCLUDED_SUBJECT_SLUGS = {
    "home_science",
    "pts",
}


def is_attendance_feature(column):
    """Return True if column belongs to attendance history."""

    if column in ATTENDANCE_FEATURES:
        return True

    lower = column.lower()

    attendance_terms = [
        "attendance",
        "days_absent",
        "days_present",
    ]

    return any(
        term in lower
        for term in attendance_terms
    )


def belongs_to_excluded_subject(column):
    """Detect historical features belonging to subjects that are intentionally excluded from modelling."""

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
    """Calculate regression metrics."""

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
        "predictions_below_0": below_zero,
        "predictions_above_100": above_hundred,
    }


def save_metric_figure(
    metrics_df,
    metric,
    ylabel,
    filename,
):
    """Save one validation metric comparison chart."""

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
        figsize=(
            9,
            6,
        )
    )

    bars = ax.bar(
        plot_df[
            "model"
        ],
        plot_df[
            metric
        ],
        color="#6F42C1",
        edgecolor="#4B2E83",
    )

    ax.set_title(
        f"2026 T1 Validation — {ylabel}",
        fontsize=14,
        fontweight="bold",
    )

    ax.set_ylabel(
        ylabel
    )

    ax.set_xlabel(
        "Model"
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


print(
    "=" * 110
)

print(
    "STUDENT PERFORMANCE AI - "
    "ACADEMIC MODEL TRAININGA OVERALL ACADEMIC MODEL DEVELOPMENT"
)

print(
    "=" * 110
)


if not INPUT_FILE.exists():

    raise FileNotFoundError(
        "\nOverall prediction dataset not found:\n"
        f"{INPUT_FILE}\n\n"
        "Build the model-ready prediction targets first."
    )


df = pd.read_csv(
    INPUT_FILE
)


required_columns = {
    "student_id",
    "academic_period",
    "dataset_split",
    "grade",
    "target_calculated_average",
}


missing_required = sorted(
    required_columns.difference(
        df.columns
    )
)


if missing_required:

    raise RuntimeError(
        "\nRequired columns missing from "
        "overall prediction dataset:\n"
        f"{missing_required}"
    )


print(
    f"\nInput rows: "
    f"{len(df)}"
)


print(
    f"Columns available: "
    f"{len(df.columns)}"
)


split_counts = (
    df[
        "dataset_split"
    ]
    .value_counts()
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


if len(
    train_df
) != 573:

    raise RuntimeError(
        "\nUnexpected TRAIN row count.\n"
        f"Expected: 573\n"
        f"Actual: {len(train_df)}"
    )


if len(
    validation_df
) != 217:

    raise RuntimeError(
        "\nUnexpected VALIDATION row count.\n"
        f"Expected: 217\n"
        f"Actual: {len(validation_df)}"
    )


if len(
    final_test_df
) != 268:

    raise RuntimeError(
        "\nUnexpected FINAL_TEST row count.\n"
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
    f"(protected — not used)"
)


if train_df[
    "target_calculated_average"
].isna().any():

    raise RuntimeError(
        "\nTRAIN contains missing target values."
    )


if validation_df[
    "target_calculated_average"
].isna().any():

    raise RuntimeError(
        "\nVALIDATION contains missing target values."
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


feature_audit_records = []

candidate_features = []


for column in df.columns:

    if column in EXCLUDED_METADATA_COLUMNS:

        feature_audit_records.append(
            {
                "feature": column,
                "status": "EXCLUDED",
                "reason": "metadata_or_target",
            }
        )

        continue


    if is_attendance_feature(
        column
    ):

        feature_audit_records.append(
            {
                "feature": column,
                "status": "EXCLUDED",
                "reason": (
                    "attendance_feature_removed_"
                    "for_academic_only_experiment"
                ),
            }
        )

        continue


    if belongs_to_excluded_subject(
        column
    ):

        feature_audit_records.append(
            {
                "feature": column,
                "status": "EXCLUDED",
                "reason": (
                    "subject_not_suitable_"
                    "for_longitudinal_modelling"
                ),
            }
        )

        continue


    candidate_features.append(
        column
    )


# Candidate model predictors should be numeric.

usable_features = []


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


    # Entirely missing in training

    if train_numeric.notna().sum() == 0:

        feature_audit_records.append(
            {
                "feature": column,
                "status": "EXCLUDED",
                "reason": (
                    "all_missing_in_training"
                ),
            }
        )

        continue


    # Constant in training

    if (
        train_numeric
        .dropna()
        .nunique()
        <= 1
    ):

        feature_audit_records.append(
            {
                "feature": column,
                "status": "EXCLUDED",
                "reason": (
                    "constant_in_training"
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

    feature_audit_records.append(
        {
            "feature": column,
            "status": "INCLUDED",
            "reason": (
                "academic_historical_predictor"
            ),
        }
    )


feature_audit_df = pd.DataFrame(
    feature_audit_records
)


included_features = sorted(
    usable_features
)


if not included_features:

    raise RuntimeError(
        "\nNo usable academic features were selected."
    )


train_missing_pct = {}

validation_missing_pct = {}


for column in df.columns:

    if column in train_df.columns:

        train_missing_pct[
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

        validation_missing_pct[
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
        train_missing_pct
    )
)


feature_audit_df[
    "validation_missing_pct"
] = (
    feature_audit_df[
        "feature"
    ]
    .map(
        validation_missing_pct
    )
)


feature_audit_df = (
    feature_audit_df
    .sort_values(
        by=[
            "status",
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
        included_features
    ]
    .copy()
)


X_validation = (
    validation_df[
        included_features
    ]
    .copy()
)


# LEAKAGE VALIDATION

FORBIDDEN_MODEL_FEATURES = {
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
    FORBIDDEN_MODEL_FEATURES.intersection(
        X_train.columns
    )
)


if leaked_features:

    raise RuntimeError(
        "\nTarget/metadata leakage detected:\n"
        f"{leaked_features}"
    )


attendance_leakage = [
    column
    for column in X_train.columns
    if is_attendance_feature(
        column
    )
]


if attendance_leakage:

    raise RuntimeError(
        "\nAttendance features unexpectedly entered "
        "academic-only model:\n"
        f"{attendance_leakage}"
    )


# Imputation is fitted on training data only (inside the Pipeline); add_indicator keeps the missingness signal.

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


print(
    "\n"
    + "=" * 110
)

print(
    "MODEL TRAINING"
)

print(
    "=" * 110
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
            "model": model_name,
            "feature_set": (
                "ACADEMIC_HISTORY_ONLY"
            ),
            "evaluation_set": "TRAIN",
            "rows": len(
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
            "model": model_name,
            "feature_set": (
                "ACADEMIC_HISTORY_ONLY"
            ),
            "evaluation_set": (
                "VALIDATION"
            ),
            "rows": len(
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


    # Development-stage models; final models are retrained after selection.

    model_filename = (
        model_name
        .lower()
        .replace(
            " ",
            "_"
        )
        + "_academic_only.joblib"
    )


    joblib.dump(
        {
            "pipeline":
                pipeline,

            "features":
                included_features,

            "feature_set":
                "ACADEMIC_HISTORY_ONLY",

            "training_target_periods":
                [
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


metrics_df.to_csv(
    METRICS_OUTPUT,
    index=False,
)


# Pseudonymised student-level predictions; keep private.

validation_predictions.to_csv(
    VALIDATION_PREDICTIONS_OUTPUT,
    index=False,
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
] = (
    np.arange(
        1,
        len(
            validation_metrics_df
        ) + 1,
    )
)


# Merge rank back for reporting only.

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


model_config = {
    "stage":
        "15A",

    "task":
        "overall_next_term_performance_regression",

    "feature_set":
        "academic_history_only",

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

    "model_feature_count":
        len(
            included_features
        ),

    "attendance_features_used":
        False,

    "final_test_used":
        False,

    "excluded_subjects": sorted(
        EXCLUDED_SUBJECT_SLUGS
    ),

    "models": {
        "Linear Regression": {
            "type":
                "baseline_linear_regression",
        },

        "Random Forest": {
            "n_estimators": 500,
            "min_samples_split": 4,
            "min_samples_leaf": 2,
            "max_features": "sqrt",
        },

        "XGBoost": {
            "n_estimators": 500,
            "learning_rate": 0.03,
            "max_depth": 4,
            "min_child_weight": 3,
            "subsample": 0.85,
            "colsample_bytree": 0.85,
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


save_metric_figure(
    metrics_df,
    metric="mae",
    ylabel="Mean Absolute Error (MAE)",
    filename="overall_academic_validation_mae.png",
)


save_metric_figure(
    metrics_df,
    metric="rmse",
    ylabel="Root Mean Squared Error (RMSE)",
    filename="overall_academic_validation_rmse.png",
)


save_metric_figure(
    metrics_df,
    metric="r2",
    ylabel="R²",
    filename="overall_academic_validation_r2.png",
)


print(
    "\n"
    + "=" * 110
)

print(
    "ACADEMIC MODEL TRAININGA MODEL DEVELOPMENT COMPLETE"
)

print(
    "=" * 110
)


print(
    "\nExperiment:"
)


print(
    "  Task: next-term overall academic "
    "performance prediction"
)


print(
    "  Feature set: academic/history only"
)


print(
    "  Attendance included: NO"
)


print(
    "\nTemporal design:"
)


print(
    "  Training targets: "
    "2025_T2 + 2025_T3"
)


print(
    "  Validation target: "
    "2026_T1"
)


print(
    "  Final test target: "
    "2026_T2 — PROTECTED"
)


print(
    "\nRows:"
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
    f"  FINAL_TEST protected: "
    f"{len(final_test_df)}"
)


print(
    f"\nAcademic model features used: "
    f"{len(included_features)}"
)


print(
    "\nValidation ranking:"
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


best_model = (
    validation_metrics_df
    .iloc[0][
        "model"
    ]
)


print(
    f"\nLowest validation MAE: "
    f"{best_model}"
)


print(
    "\nIMPORTANT:"
)


print(
    "  This is NOT yet the final model selection."
)


print(
    "  Attendance features have not yet been compared."
)


print(
    "  The 2026 T2 final test set was not fitted, "
    "predicted, scored, or inspected."
)


print(
    "\nAggregate reports:"
)


print(
    f"  {METRICS_OUTPUT}"
)


print(
    f"  {FEATURE_AUDIT_OUTPUT}"
)


print(
    f"  {CONFIG_OUTPUT}"
)


print(
    "\nFigures:"
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
