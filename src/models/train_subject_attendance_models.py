from pathlib import Path
import json

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
)

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
    / "subject_prediction_dataset.csv"
)

ACADEMIC_METRICS_FILE = (
    PROJECT_ROOT
    / "reports"
    / "modeling"
    / "subject_academic_validation_metrics.csv"
)

ACADEMIC_PER_SUBJECT_FILE = (
    PROJECT_ROOT
    / "reports"
    / "modeling"
    / "subject_academic_per_subject_metrics.csv"
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
    / "subject_attendance_validation_metrics.csv"
)

PER_SUBJECT_METRICS_OUTPUT = (
    REPORT_DIR
    / "subject_attendance_per_subject_metrics.csv"
)

COMPARISON_OUTPUT = (
    REPORT_DIR
    / "subject_feature_set_comparison.csv"
)

PER_SUBJECT_COMPARISON_OUTPUT = (
    REPORT_DIR
    / "subject_per_subject_feature_comparison.csv"
)

FEATURE_AUDIT_OUTPUT = (
    REPORT_DIR
    / "subject_attendance_feature_audit.csv"
)

CONFIG_OUTPUT = (
    REPORT_DIR
    / "subject_attendance_model_config.json"
)

VALIDATION_PREDICTIONS_OUTPUT = (
    PRIVATE_OUTPUT_DIR
    / "attendance_subject_attendance_validation_predictions.csv"
)


ACADEMIC_NUMERIC_FEATURES = [
    "grade",
    "history_term_count",

    "prev_calculated_average",
    "historical_mean_average",
    "historical_std_average",
    "performance_trend",

    "prev_numeric_subject_count",
    "prev_absent_subject_count",
    "prev_not_taken_subject_count",
    "historical_mean_absent_subject_count",
    "historical_mean_not_taken_subject_count",

    "prev_subject_mark",
    "historical_mean_subject_mark",
    "historical_subject_mark_count",
    "subject_mark_trend",

    "prev_subject_absent",
    "prev_subject_not_taken",
    "prev_subject_recorded",
]


# Historical attendance only; target-term attendance is never used.

ATTENDANCE_NUMERIC_FEATURES = [
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


CATEGORICAL_FEATURES = [
    "target_subject",
]


FORBIDDEN_COLUMNS = {
    "student_id",
    "year",
    "term",
    "academic_period",
    "period_order",
    "class",
    "academic_stage",
    "dataset_split",
    "target_mark",
}


def make_one_hot_encoder():
    """Compatible with newer and older scikit-learn versions."""

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


def calculate_metrics(
    y_true,
    y_pred,
):
    """Calculate standard regression metrics."""

    mae = mean_absolute_error(
        y_true,
        y_pred,
    )

    rmse = float(
        np.sqrt(
            mean_squared_error(
                y_true,
                y_pred,
            )
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


def get_sample_flag(rows):
    """Interpretation flag for subject-level metrics."""

    if rows < 10:
        return "VERY_SMALL_SAMPLE"

    if rows < 20:
        return "SMALL_SAMPLE"

    return "ADEQUATE_FOR_DESCRIPTIVE_EVALUATION"


def calculate_per_subject_metrics(
    predictions_df,
    prediction_column,
    model_name,
):
    """Calculate validation metrics separately for each target subject."""

    records = []

    for subject, group in predictions_df.groupby(
        "target_subject",
        sort=True,
    ):

        y_true = group[
            "actual_mark"
        ].to_numpy()

        y_pred = group[
            prediction_column
        ].to_numpy()

        rows = len(
            group
        )

        mae = mean_absolute_error(
            y_true,
            y_pred,
        )

        rmse = float(
            np.sqrt(
                mean_squared_error(
                    y_true,
                    y_pred,
                )
            )
        )

        if rows >= 2:

            r2 = r2_score(
                y_true,
                y_pred,
            )

        else:

            r2 = np.nan

        records.append(
            {
                "model":
                    model_name,

                "target_subject":
                    subject,

                "validation_rows":
                    rows,

                "mae":
                    float(mae),

                "rmse":
                    float(rmse),

                "r2":
                    (
                        float(r2)
                        if not np.isnan(r2)
                        else np.nan
                    ),

                "interpretation_flag":
                    get_sample_flag(
                        rows
                    ),
            }
        )

    return records


def save_overall_comparison_chart(
    comparison_df,
):
    """Compare academic-only vs academic + attendance validation MAE."""

    plot_df = comparison_df.copy()

    positions = np.arange(
        len(
            plot_df
        )
    )

    width = 0.36

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    academic_bars = ax.bar(
        positions - width / 2,
        plot_df[
            "academic_only_mae"
        ],
        width,
        label="Academic history only",
        color="#D8C7F0",
        edgecolor="#4B2E83",
    )

    attendance_bars = ax.bar(
        positions + width / 2,
        plot_df[
            "academic_plus_attendance_mae"
        ],
        width,
        label="Academic history + attendance",
        color="#6F42C1",
        edgecolor="#4B2E83",
    )

    ax.set_xticks(
        positions
    )

    ax.set_xticklabels(
        plot_df[
            "model"
        ]
    )

    ax.set_ylabel(
        "Mean Absolute Error (marks)"
    )

    ax.set_xlabel(
        "Model"
    )

    ax.set_title(
        "Pooled Subject Prediction — "
        "Effect of Attendance on Validation MAE",
        fontsize=13,
        fontweight="bold",
    )

    ax.legend()

    ax.grid(
        axis="y",
        alpha=0.20,
    )

    for bars in [
        academic_bars,
        attendance_bars,
    ]:

        for bar in bars:

            value = (
                bar.get_height()
            )

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
        / "subject_academic_vs_attendance_mae.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )


def save_best_model_subject_improvement_chart(
    subject_comparison_df,
    best_model,
):
    """Show attendance MAE improvement by subject for the best attendance feature comparison model."""

    plot_df = (
        subject_comparison_df[
            (
                subject_comparison_df[
                    "model"
                ] == best_model
            )
            &
            (
                subject_comparison_df[
                    "validation_rows"
                ] >= 20
            )
        ]
        .copy()
    )

    if plot_df.empty:
        return

    plot_df = (
        plot_df
        .sort_values(
            "mae_improvement",
            ascending=True,
        )
    )

    fig_height = max(
        6,
        len(
            plot_df
        ) * 0.45,
    )

    fig, ax = plt.subplots(
        figsize=(
            10,
            fig_height,
        )
    )

    bars = ax.barh(
        plot_df[
            "target_subject"
        ],
        plot_df[
            "mae_improvement"
        ],
        color="#6F42C1",
        edgecolor="#4B2E83",
    )

    ax.axvline(
        0,
        linewidth=1,
    )

    ax.set_title(
        f"{best_model} — "
        "Attendance Effect by Subject",
        fontsize=13,
        fontweight="bold",
    )

    ax.set_xlabel(
        "MAE Improvement "
        "(positive = attendance helped)"
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
        / "subject_attendance_improvement_by_subject.png",
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
    "ATTENDANCE FEATURE COMPARISONB POOLED SUBJECT ACADEMIC + ATTENDANCE COMPARISON"
)

print(
    "=" * 110
)


if not INPUT_FILE.exists():

    raise FileNotFoundError(
        "\nSubject prediction dataset not found:\n"
        f"{INPUT_FILE}"
    )


if not ACADEMIC_METRICS_FILE.exists():

    raise FileNotFoundError(
        "\nacademic baseline subject metrics missing:\n"
        f"{ACADEMIC_METRICS_FILE}"
    )


if not ACADEMIC_PER_SUBJECT_FILE.exists():

    raise FileNotFoundError(
        "\nacademic baseline per-subject metrics missing:\n"
        f"{ACADEMIC_PER_SUBJECT_FILE}"
    )


df = pd.read_csv(
    INPUT_FILE
)


academic_metrics = pd.read_csv(
    ACADEMIC_METRICS_FILE
)


academic_per_subject = pd.read_csv(
    ACADEMIC_PER_SUBJECT_FILE
)


print(
    f"\nInput rows: "
    f"{len(df)}"
)


print(
    f"Columns available: "
    f"{len(df.columns)}"
)


required_columns = set(
    ACADEMIC_NUMERIC_FEATURES
    + ATTENDANCE_NUMERIC_FEATURES
    + CATEGORICAL_FEATURES
)


required_columns.update(
    {
        "student_id",
        "academic_period",
        "dataset_split",
        "target_mark",
    }
)


missing_columns = sorted(
    required_columns.difference(
        df.columns
    )
)


if missing_columns:

    raise RuntimeError(
        "\nRequired columns missing:\n"
        f"{missing_columns}"
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


if len(train_df) != 6387:

    raise RuntimeError(
        "\nUnexpected TRAIN count.\n"
        f"Expected: 6387\n"
        f"Actual: {len(train_df)}"
    )


if len(validation_df) != 2029:

    raise RuntimeError(
        "\nUnexpected VALIDATION count.\n"
        f"Expected: 2029\n"
        f"Actual: {len(validation_df)}"
    )


if len(final_test_df) != 2912:

    raise RuntimeError(
        "\nUnexpected FINAL_TEST count.\n"
        f"Expected: 2912\n"
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
        "target_mark"
    ],
    errors="raise",
)


y_validation = pd.to_numeric(
    validation_df[
        "target_mark"
    ],
    errors="raise",
)


if y_train.isna().any():

    raise RuntimeError(
        "\nTRAIN contains missing targets."
    )


if y_validation.isna().any():

    raise RuntimeError(
        "\nVALIDATION contains missing targets."
    )


all_numeric_features = (
    ACADEMIC_NUMERIC_FEATURES
    + ATTENDANCE_NUMERIC_FEATURES
)


usable_numeric_features = []

usable_academic_features = []

usable_attendance_features = []

feature_audit_records = []


for column in all_numeric_features:

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


    feature_group = (
        "ATTENDANCE"
        if column
        in ATTENDANCE_NUMERIC_FEATURES
        else "ACADEMIC"
    )


    if (
        train_numeric
        .notna()
        .sum()
        == 0
    ):

        feature_audit_records.append(
            {
                "feature":
                    column,

                "feature_group":
                    feature_group,

                "status":
                    "EXCLUDED",

                "reason":
                    "all_missing_in_training",

                "train_missing_pct":
                    100.0,

                "validation_missing_pct":
                    round(
                        float(
                            validation_numeric
                            .isna()
                            .mean()
                            * 100
                        ),
                        4,
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

                "feature_group":
                    feature_group,

                "status":
                    "EXCLUDED",

                "reason":
                    "constant_in_training",

                "train_missing_pct":
                    round(
                        float(
                            train_numeric
                            .isna()
                            .mean()
                            * 100
                        ),
                        4,
                    ),

                "validation_missing_pct":
                    round(
                        float(
                            validation_numeric
                            .isna()
                            .mean()
                            * 100
                        ),
                        4,
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


    usable_numeric_features.append(
        column
    )


    if feature_group == "ATTENDANCE":

        usable_attendance_features.append(
            column
        )

    else:

        usable_academic_features.append(
            column
        )


    feature_audit_records.append(
        {
            "feature":
                column,

            "feature_group":
                feature_group,

            "status":
                "INCLUDED",

            "reason":
                "historical_predictor",

            "train_missing_pct":
                round(
                    float(
                        train_numeric
                        .isna()
                        .mean()
                        * 100
                    ),
                    4,
                ),

            "validation_missing_pct":
                round(
                    float(
                        validation_numeric
                        .isna()
                        .mean()
                        * 100
                    ),
                    4,
                ),
        }
    )


train_df[
    "target_subject"
] = (
    train_df[
        "target_subject"
    ]
    .astype(str)
)


validation_df[
    "target_subject"
] = (
    validation_df[
        "target_subject"
    ]
    .astype(str)
)


feature_audit_records.append(
    {
        "feature":
            "target_subject",

        "feature_group":
            "SUBJECT_CONTEXT",

        "status":
            "INCLUDED",

        "reason":
            "pooled_subject_identity",

        "train_missing_pct":
            0.0,

        "validation_missing_pct":
            0.0,
    }
)


feature_audit_df = pd.DataFrame(
    feature_audit_records
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


if not usable_attendance_features:

    raise RuntimeError(
        "\nNo usable attendance features found."
    )


print(
    "\nAcademic features used:"
)


for feature in usable_academic_features:

    print(
        f"  - {feature}"
    )


print(
    "\nAttendance features used:"
)


for feature in usable_attendance_features:

    print(
        f"  - {feature}"
    )


print(
    f"\nAcademic numeric features: "
    f"{len(usable_academic_features)}"
)


print(
    f"Attendance numeric features: "
    f"{len(usable_attendance_features)}"
)


print(
    f"Total numeric features: "
    f"{len(usable_numeric_features)}"
)


print(
    f"Categorical features: "
    f"{len(CATEGORICAL_FEATURES)}"
)


model_features = (
    usable_numeric_features
    + CATEGORICAL_FEATURES
)


X_train = (
    train_df[
        model_features
    ]
    .copy()
)


X_validation = (
    validation_df[
        model_features
    ]
    .copy()
)


# LEAKAGE VALIDATION

leaked_columns = sorted(
    FORBIDDEN_COLUMNS.intersection(
        X_train.columns
    )
)


if leaked_columns:

    raise RuntimeError(
        "\nForbidden columns entered model:\n"
        f"{leaked_columns}"
    )


linear_numeric_pipeline = Pipeline(
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
    ]
)


tree_numeric_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="median",
                add_indicator=True,
            ),
        ),
    ]
)


linear_preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            linear_numeric_pipeline,
            usable_numeric_features,
        ),
        (
            "subject",
            make_one_hot_encoder(),
            CATEGORICAL_FEATURES,
        ),
    ],
    remainder="drop",
    sparse_threshold=0,
)


tree_preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            tree_numeric_pipeline,
            usable_numeric_features,
        ),
        (
            "subject",
            make_one_hot_encoder(),
            CATEGORICAL_FEATURES,
        ),
    ],
    remainder="drop",
    sparse_threshold=0,
)


# Same hyperparameters as the academic baseline.

linear_model = Pipeline(
    steps=[
        (
            "preprocessor",
            linear_preprocessor,
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
            "preprocessor",
            tree_preprocessor,
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
            "preprocessor",
            tree_preprocessor,
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
    "POOLED SUBJECT MODEL TRAINING — "
    "ACADEMIC HISTORY + ATTENDANCE"
)

print(
    "=" * 110
)


metric_records = []

per_subject_metric_records = []


validation_predictions = (
    validation_df[
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


validation_predictions = (
    validation_predictions
    .rename(
        columns={
            "target_mark":
                "actual_mark"
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

            "task":
                "POOLED_SUBJECT_REGRESSION",

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

            "task":
                "POOLED_SUBJECT_REGRESSION",

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


    validation_predictions[
        "absolute_error_"
        + model_name
        .lower()
        .replace(
            " ",
            "_"
        )
    ] = np.abs(
        validation_predictions[
            "actual_mark"
        ]
        - validation_pred
    )


    per_subject_metric_records.extend(
        calculate_per_subject_metrics(
            validation_predictions,
            prediction_column,
            model_name,
        )
    )


    model_filename = (
        model_name
        .lower()
        .replace(
            " ",
            "_"
        )
        + "_subject_academic_plus_attendance.joblib"
    )


    joblib.dump(
        {
            "pipeline":
                pipeline,

            "academic_features":
                usable_academic_features,

            "attendance_features":
                usable_attendance_features,

            "categorical_features":
                CATEGORICAL_FEATURES,

            "feature_set":
                "ACADEMIC_HISTORY_PLUS_ATTENDANCE",

            "task":
                "POOLED_SUBJECT_REGRESSION",

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


per_subject_df = pd.DataFrame(
    per_subject_metric_records
)


per_subject_df = (
    per_subject_df
    .sort_values(
        by=[
            "model",
            "validation_rows",
            "target_subject",
        ],
        ascending=[
            True,
            False,
            True,
        ],
    )
    .reset_index(
        drop=True
    )
)


per_subject_df.to_csv(
    PER_SUBJECT_METRICS_OUTPUT,
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


comparison_df = (
    academic_validation
    .merge(
        attendance_validation,
        on="model",
        how="inner",
    )
)


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


academic_subject = (
    academic_per_subject[
        [
            "model",
            "target_subject",
            "validation_rows",
            "mae",
            "rmse",
            "r2",
        ]
    ]
    .copy()
)


academic_subject = (
    academic_subject
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


attendance_subject = (
    per_subject_df[
        [
            "model",
            "target_subject",
            "validation_rows",
            "mae",
            "rmse",
            "r2",
        ]
    ]
    .copy()
)


attendance_subject = (
    attendance_subject
    .rename(
        columns={
            "validation_rows":
                "attendance_validation_rows",

            "mae":
                "academic_plus_attendance_mae",

            "rmse":
                "academic_plus_attendance_rmse",

            "r2":
                "academic_plus_attendance_r2",
        }
    )
)


subject_comparison_df = (
    academic_subject
    .merge(
        attendance_subject,
        on=[
            "model",
            "target_subject",
        ],
        how="inner",
    )
)


if not (
    subject_comparison_df[
        "validation_rows"
    ]
    == subject_comparison_df[
        "attendance_validation_rows"
    ]
).all():

    raise RuntimeError(
        "\nacademic baseline and attendance feature comparison subject sample counts differ."
    )


subject_comparison_df = (
    subject_comparison_df
    .drop(
        columns=[
            "attendance_validation_rows"
        ]
    )
)


subject_comparison_df[
    "mae_improvement"
] = (
    subject_comparison_df[
        "academic_only_mae"
    ]
    - subject_comparison_df[
        "academic_plus_attendance_mae"
    ]
)


subject_comparison_df[
    "mae_improvement_pct"
] = (
    subject_comparison_df[
        "mae_improvement"
    ]
    / subject_comparison_df[
        "academic_only_mae"
    ]
    * 100
)


subject_comparison_df[
    "r2_improvement"
] = (
    subject_comparison_df[
        "academic_plus_attendance_r2"
    ]
    - subject_comparison_df[
        "academic_only_r2"
    ]
)


subject_comparison_df[
    "attendance_improved_mae"
] = (
    subject_comparison_df[
        "mae_improvement"
    ]
    > 0
)


subject_comparison_df[
    "interpretation_flag"
] = (
    subject_comparison_df[
        "validation_rows"
    ]
    .apply(
        get_sample_flag
    )
)


subject_comparison_df = (
    subject_comparison_df
    .sort_values(
        by=[
            "model",
            "validation_rows",
            "target_subject",
        ],
        ascending=[
            True,
            False,
            True,
        ],
    )
    .reset_index(
        drop=True
    )
)


subject_comparison_df.to_csv(
    PER_SUBJECT_COMPARISON_OUTPUT,
    index=False,
)


validation_predictions.to_csv(
    VALIDATION_PREDICTIONS_OUTPUT,
    index=False,
)


best_model = (
    validation_metrics_df
    .iloc[0][
        "model"
    ]
)


config = {
    "stage":
        "16B",

    "task":
        "pooled_next_term_subject_regression",

    "experiment":
        "academic_history_plus_attendance",

    "comparison_baseline":
        "academic_academic_history_only",

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
            usable_academic_features
        ),

    "attendance_feature_count":
        len(
            usable_attendance_features
        ),

    "categorical_features":
        CATEGORICAL_FEATURES,

    "attendance_features":
        usable_attendance_features,

    "final_test_used":
        False,

    "term_difficulty_context": {
        "known_school_assumption":
            (
                "Later terms may contain more difficult examinations."
            ),

        "manual_adjustment_applied":
            False,
    },
}


with open(
    CONFIG_OUTPUT,
    "w",
    encoding="utf-8",
) as file:

    json.dump(
        config,
        file,
        indent=4,
    )


save_overall_comparison_chart(
    comparison_df
)


save_best_model_subject_improvement_chart(
    subject_comparison_df,
    best_model,
)


print(
    "\n"
    + "=" * 110
)

print(
    "ATTENDANCE FEATURE COMPARISONB SUBJECT FEATURE COMPARISON COMPLETE"
)

print(
    "=" * 110
)


print(
    "\nExperiment:"
)


print(
    "  Academic/history only "
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
    "\nFeature counts:"
)


print(
    f"  Academic numeric: "
    f"{len(usable_academic_features)}"
)


print(
    f"  Attendance numeric: "
    f"{len(usable_attendance_features)}"
)


print(
    f"  Categorical: "
    f"{len(CATEGORICAL_FEATURES)}"
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


print(
    f"\nLowest validation MAE: "
    f"{best_model}"
)


print(
    "\nBest-model subject-level attendance effects "
    "(>=20 validation rows):"
)


best_subject_results = (
    subject_comparison_df[
        (
            subject_comparison_df[
                "model"
            ] == best_model
        )
        &
        (
            subject_comparison_df[
                "validation_rows"
            ] >= 20
        )
    ]
    [
        [
            "target_subject",
            "validation_rows",
            "academic_only_mae",
            "academic_plus_attendance_mae",
            "mae_improvement",
            "mae_improvement_pct",
        ]
    ]
    .sort_values(
        "mae_improvement",
        ascending=False,
    )
)


print(
    best_subject_results.to_string(
        index=False,
        float_format=lambda x: (
            f"{x:.4f}"
        ),
    )
)


helped_subjects = (
    best_subject_results[
        best_subject_results[
            "mae_improvement"
        ] > 0
    ]
)


hurt_subjects = (
    best_subject_results[
        best_subject_results[
            "mae_improvement"
        ] < 0
    ]
)


print(
    f"\nSubjects where attendance improved "
    f"{best_model} MAE: "
    f"{len(helped_subjects)}"
)


print(
    f"Subjects where attendance worsened "
    f"{best_model} MAE: "
    f"{len(hurt_subjects)}"
)


print(
    "\nIMPORTANT:"
)


print(
    "  Positive MAE improvement means attendance helped."
)


print(
    "  Negative MAE improvement means attendance hurt."
)


print(
    "  Very small optional-subject samples remain "
    "flagged and should not support broad conclusions."
)


print(
    "  No manual term-difficulty penalty was applied."
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
    f"  {PER_SUBJECT_METRICS_OUTPUT}"
)


print(
    f"  {COMPARISON_OUTPUT}"
)


print(
    f"  {PER_SUBJECT_COMPARISON_OUTPUT}"
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
    "\nPrivate predictions:"
)


print(
    f"  {VALIDATION_PREDICTIONS_OUTPUT}"
)
