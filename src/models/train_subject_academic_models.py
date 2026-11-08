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
        "Install it inside the project environment with:\n\n"
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
    / "subject_academic_validation_metrics.csv"
)

PER_SUBJECT_METRICS_OUTPUT = (
    REPORT_DIR
    / "subject_academic_per_subject_metrics.csv"
)

FEATURE_AUDIT_OUTPUT = (
    REPORT_DIR
    / "subject_academic_feature_audit.csv"
)

CONFIG_OUTPUT = (
    REPORT_DIR
    / "subject_academic_model_config.json"
)

VALIDATION_PREDICTIONS_OUTPUT = (
    PRIVATE_OUTPUT_DIR
    / "academic_subject_academic_validation_predictions.csv"
)


# Academic/history-only pooled subject regression; attendance excluded.

METADATA_COLUMNS = {
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


ATTENDANCE_COLUMNS = {
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


# Predictors known before the target term.

ACADEMIC_NUMERIC_FEATURES = [
    "grade",
    "history_term_count",

    # Overall academic history
    "prev_calculated_average",
    "historical_mean_average",
    "historical_std_average",
    "performance_trend",

    # Historical exam participation
    "prev_numeric_subject_count",
    "prev_absent_subject_count",
    "prev_not_taken_subject_count",
    "historical_mean_absent_subject_count",
    "historical_mean_not_taken_subject_count",

    # Target-subject history
    "prev_subject_mark",
    "historical_mean_subject_mark",
    "historical_subject_mark_count",
    "subject_mark_trend",

    # Explicit missingness / status semantics
    "prev_subject_absent",
    "prev_subject_not_taken",
    "prev_subject_recorded",
]


CATEGORICAL_FEATURES = [
    "target_subject",
]


def make_one_hot_encoder():
    """Support both newer and older scikit-learn versions."""

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

    predictions_below_zero = int(
        (
            np.asarray(
                y_pred
            )
            < 0
        ).sum()
    )

    predictions_above_hundred = int(
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
            predictions_below_zero,
        "predictions_above_100":
            predictions_above_hundred,
    }


def calculate_subject_metrics(
    dataframe,
    prediction_column,
    model_name,
):
    """Calculate validation performance separately for each target subject."""

    records = []

    for subject, group in dataframe.groupby(
        "target_subject",
        sort=True,
    ):

        y_true = (
            group[
                "actual_mark"
            ]
            .to_numpy()
        )

        y_pred = (
            group[
                prediction_column
            ]
            .to_numpy()
        )

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


        if rows < 10:

            interpretation = (
                "VERY_SMALL_SAMPLE"
            )

        elif rows < 20:

            interpretation = (
                "SMALL_SAMPLE"
            )

        else:

            interpretation = (
                "ADEQUATE_FOR_DESCRIPTIVE_EVALUATION"
            )


        records.append(
            {
                "model":
                    model_name,

                "target_subject":
                    subject,

                "validation_rows":
                    rows,

                "mae":
                    float(
                        mae
                    ),

                "rmse":
                    float(
                        rmse
                    ),

                "r2":
                    (
                        float(
                            r2
                        )
                        if not np.isnan(
                            r2
                        )
                        else np.nan
                    ),

                "interpretation_flag":
                    interpretation,
            }
        )

    return records


def save_metric_chart(
    metrics_df,
    metric,
    ylabel,
    filename,
):
    """Save validation model-comparison figure."""

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
            by=metric,
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
        f"Pooled Subject Model — "
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


def save_best_subject_mae_chart(
    per_subject_df,
    best_model,
):
    """Plot subject-level validation MAE for subjects with at least 20 validation observations."""

    plot_df = (
        per_subject_df[
            (
                per_subject_df[
                    "model"
                ] == best_model
            )
            &
            (
                per_subject_df[
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
            "mae",
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
            "mae"
        ],
        color="#6F42C1",
        edgecolor="#4B2E83",
    )


    ax.set_title(
        f"{best_model} — "
        "Subject-Level Validation MAE",
        fontsize=13,
        fontweight="bold",
    )


    ax.set_xlabel(
        "Mean Absolute Error (marks)"
    )


    ax.set_ylabel(
        "Subject"
    )


    ax.grid(
        axis="x",
        alpha=0.20,
    )


    for bar in bars:

        value = bar.get_width()

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
        / "subject_academic_best_model_per_subject_mae.png",
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
    "ACADEMIC MODEL TRAININGB POOLED SUBJECT MODEL DEVELOPMENT"
)

print(
    "=" * 110
)


if not INPUT_FILE.exists():

    raise FileNotFoundError(
        "\nSubject prediction dataset not found:\n"
        f"{INPUT_FILE}\n\n"
        "Build the model-ready prediction targets first."
    )


df = pd.read_csv(
    INPUT_FILE
)


print(
    f"\nInput subject-target rows: "
    f"{len(df)}"
)


print(
    f"Columns available: "
    f"{len(df.columns)}"
)


required_columns = {
    "student_id",
    "academic_period",
    "dataset_split",
    "grade",
    "target_subject",
    "target_mark",
    "prev_subject_recorded",
}


required_columns.update(
    ACADEMIC_NUMERIC_FEATURES
)


missing_required = sorted(
    required_columns.difference(
        df.columns
    )
)


if missing_required:

    raise RuntimeError(
        "\nRequired subject modelling columns "
        "are missing:\n"
        f"{missing_required}"
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
) != 6387:

    raise RuntimeError(
        "\nUnexpected TRAIN subject-target count.\n"
        f"Expected: 6387\n"
        f"Actual: {len(train_df)}"
    )


if len(
    validation_df
) != 2029:

    raise RuntimeError(
        "\nUnexpected VALIDATION subject-target count.\n"
        f"Expected: 2029\n"
        f"Actual: {len(validation_df)}"
    )


if len(
    final_test_df
) != 2912:

    raise RuntimeError(
        "\nUnexpected FINAL_TEST subject-target count.\n"
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
    f"(protected — not used)"
)


print(
    "\nSubjects:"
)


print(
    f"  TRAIN subjects: "
    f"{train_df['target_subject'].nunique()}"
)


print(
    f"  VALIDATION subjects: "
    f"{validation_df['target_subject'].nunique()}"
)


print(
    f"  FINAL_TEST subjects: "
    f"{final_test_df['target_subject'].nunique()} "
    f"(count only)"
)


if train_df[
    "target_mark"
].isna().any():

    raise RuntimeError(
        "\nTRAIN contains missing subject targets."
    )


if validation_df[
    "target_mark"
].isna().any():

    raise RuntimeError(
        "\nVALIDATION contains missing subject targets."
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


feature_audit_records = []

usable_numeric_features = []


for column in ACADEMIC_NUMERIC_FEATURES:

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


    # Entirely unavailable in training

    if train_numeric.notna().sum() == 0:

        feature_audit_records.append(
            {
                "feature":
                    column,

                "feature_type":
                    "NUMERIC",

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


    # Constant values add no predictive information

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

                "feature_type":
                    "NUMERIC",

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


    feature_audit_records.append(
        {
            "feature":
                column,

            "feature_type":
                "NUMERIC",

            "status":
                "INCLUDED",

            "reason":
                "academic_historical_predictor",

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

        "feature_type":
            "CATEGORICAL",

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


for column in sorted(
    ATTENDANCE_COLUMNS
):

    if column in df.columns:

        feature_audit_records.append(
            {
                "feature":
                    column,

                "feature_type":
                    "NUMERIC",

                "status":
                    "EXCLUDED",

                "reason":
                    (
                        "attendance_removed_for_"
                        "academic_only_experiment"
                    ),

                "train_missing_pct":
                    round(
                        float(
                            train_df[
                                column
                            ]
                            .isna()
                            .mean()
                            * 100
                        ),
                        4,
                    ),

                "validation_missing_pct":
                    round(
                        float(
                            validation_df[
                                column
                            ]
                            .isna()
                            .mean()
                            * 100
                        ),
                        4,
                    ),
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
            "feature_type",
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

forbidden_columns = (
    METADATA_COLUMNS
    | ATTENDANCE_COLUMNS
)


leaked_features = sorted(
    forbidden_columns.intersection(
        X_train.columns
    )
)


if leaked_features:

    raise RuntimeError(
        "\nForbidden features entered "
        "subject model:\n"
        f"{leaked_features}"
    )


if "target_mark" in X_train.columns:

    raise RuntimeError(
        "\nTarget leakage detected: "
        "target_mark entered feature matrix."
    )


if "student_id" in X_train.columns:

    raise RuntimeError(
        "\nStudent ID entered feature matrix."
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
    "POOLED SUBJECT MODEL TRAINING"
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


    train_predictions = pipeline.predict(
        X_train
    )


    train_metrics = calculate_metrics(
        y_train,
        train_predictions,
    )


    metric_records.append(
        {
            "model":
                model_name,

            "feature_set":
                "ACADEMIC_HISTORY_ONLY",

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
                "ACADEMIC_HISTORY_ONLY",

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
        calculate_subject_metrics(
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
        + "_subject_academic_only.joblib"
    )


    joblib.dump(
        {
            "pipeline":
                pipeline,

            "numeric_features":
                usable_numeric_features,

            "categorical_features":
                CATEGORICAL_FEATURES,

            "feature_set":
                "ACADEMIC_HISTORY_ONLY",

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


per_subject_metrics_df = pd.DataFrame(
    per_subject_metric_records
)


per_subject_metrics_df = (
    per_subject_metrics_df
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


per_subject_metrics_df.to_csv(
    PER_SUBJECT_METRICS_OUTPUT,
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


model_config = {
    "stage":
        "15B",

    "task":
        "pooled_next_term_subject_regression",

    "feature_set":
        "academic_history_only",

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

    "training_subjects":
        int(
            train_df[
                "target_subject"
            ].nunique()
        ),

    "validation_subjects":
        int(
            validation_df[
                "target_subject"
            ].nunique()
        ),

    "numeric_feature_count":
        len(
            usable_numeric_features
        ),

    "categorical_features":
        CATEGORICAL_FEATURES,

    "attendance_features_used":
        False,

    "final_test_used":
        False,

    "small_subject_samples_retained":
        True,

    "small_sample_evaluation_policy": {
        "very_small_sample":
            "<10 validation rows",

        "small_sample":
            "10-19 validation rows",

        "adequate_for_descriptive_evaluation":
            ">=20 validation rows",
    },

    "models": {
        "Linear Regression": {
            "type":
                "pooled_linear_baseline",
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


save_metric_chart(
    metrics_df,
    metric="mae",
    ylabel="Mean Absolute Error (MAE)",
    filename=(
        "subject_academic_validation_mae.png"
    ),
)


save_metric_chart(
    metrics_df,
    metric="rmse",
    ylabel="Root Mean Squared Error (RMSE)",
    filename=(
        "subject_academic_validation_rmse.png"
    ),
)


save_metric_chart(
    metrics_df,
    metric="r2",
    ylabel="R²",
    filename=(
        "subject_academic_validation_r2.png"
    ),
)


save_best_subject_mae_chart(
    per_subject_metrics_df,
    best_model,
)


print(
    "\n"
    + "=" * 110
)

print(
    "ACADEMIC MODEL TRAININGB POOLED SUBJECT MODEL DEVELOPMENT COMPLETE"
)

print(
    "=" * 110
)


print(
    "\nExperiment:"
)


print(
    "  Task: next-term pooled subject mark prediction"
)


print(
    "  Feature set: academic/history only"
)


print(
    "  Attendance included: NO"
)


print(
    "  Target subject encoded categorically: YES"
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
    "\nSubjects:"
)


print(
    f"  TRAIN: "
    f"{train_df['target_subject'].nunique()}"
)


print(
    f"  VALIDATION: "
    f"{validation_df['target_subject'].nunique()}"
)


print(
    f"\nNumeric academic features used: "
    f"{len(usable_numeric_features)}"
)


print(
    f"Categorical features used: "
    f"{len(CATEGORICAL_FEATURES)}"
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


print(
    f"\nLowest validation MAE: "
    f"{best_model}"
)


print(
    "\nPer-subject validation sample sizes:"
)


subject_counts = (
    validation_df[
        "target_subject"
    ]
    .value_counts()
    .sort_values(
        ascending=False
    )
)


print(
    subject_counts.to_string()
)


print(
    "\nSmall-sample policy:"
)


print(
    "  <10 rows  : VERY_SMALL_SAMPLE"
)


print(
    "  10-19 rows: SMALL_SAMPLE"
)


print(
    "  >=20 rows : descriptive subject-level "
    "evaluation permitted"
)


print(
    "\nIMPORTANT:"
)


print(
    "  Sparse optional subjects remain in the pooled model."
)


print(
    "  Their individual metrics must be interpreted "
    "with their validation sample sizes."
)


print(
    "  Attendance features were not used."
)


print(
    "  2026 T2 was not fitted, predicted, "
    "scored, or inspected."
)


print(
    "\nAggregate outputs:"
)


print(
    f"  {METRICS_OUTPUT}"
)


print(
    f"  {PER_SUBJECT_METRICS_OUTPUT}"
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
