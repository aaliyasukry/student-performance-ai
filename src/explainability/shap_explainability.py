from pathlib import Path
import json

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import shap
except ImportError as exc:
    raise ImportError(
        "\nSHAP is not installed.\n"
        "Run:\n\n"
        "    pip install shap\n"
    ) from exc


RANDOM_STATE = 42

SUBJECT_SHAP_SAMPLE = 1000
TOP_GLOBAL_FEATURES = 20
TOP_LOCAL_FEATURES = 10

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

OVERALL_MODEL_FILE = (
    PROJECT_ROOT
    / "models"
    / "final"
    / "overall_random_forest_academic_only.joblib"
)

SUBJECT_MODEL_FILE = (
    PROJECT_ROOT
    / "models"
    / "final"
    / "subject_random_forest_academic_only.joblib"
)

REPORT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "explainability"
)

FIGURE_DIR = (
    PROJECT_ROOT
    / "figures"
    / "explainability"
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


OVERALL_IMPORTANCE_OUTPUT = (
    REPORT_DIR
    / "overall_shap_global_importance.csv"
)

SUBJECT_IMPORTANCE_OUTPUT = (
    REPORT_DIR
    / "subject_shap_global_importance.csv"
)

SUBJECT_GROUPED_OUTPUT = (
    REPORT_DIR
    / "subject_shap_grouped_importance.csv"
)

SUMMARY_OUTPUT = (
    REPORT_DIR
    / "explainability_summary.json"
)

OVERALL_LOCAL_OUTPUT = (
    PRIVATE_DIR
    / "explainability_overall_local_explanations.csv"
)

SUBJECT_LOCAL_OUTPUT = (
    PRIVATE_DIR
    / "explainability_subject_local_explanations.csv"
)


def validate_files(files):

    missing = [
        str(file)
        for file in files
        if not file.exists()
    ]

    if missing:

        raise FileNotFoundError(
            "\nMissing required files:\n"
            + "\n".join(missing)
        )


def expected_value_to_float(value):

    value = np.asarray(value)

    return float(
        value.reshape(-1)[0]
    )


def normalise_shap_values(values):

    if isinstance(values, list):

        if len(values) != 1:

            raise RuntimeError(
                "Unexpected SHAP multi-output result."
            )

        values = values[0]

    values = np.asarray(values)

    if values.ndim == 3:

        if values.shape[-1] == 1:

            values = values[:, :, 0]

        elif values.shape[0] == 1:

            values = values[0]

    if values.ndim != 2:

        raise RuntimeError(
            f"Unexpected SHAP shape: {values.shape}"
        )

    return values


def simplify_name(name):

    name = str(name)

    for prefix in [
        "numeric__",
        "subject__",
    ]:

        if name.startswith(prefix):

            name = name[
                len(prefix):
            ]

    return name


def global_importance(
    shap_values,
    feature_names,
):

    mean_absolute = np.abs(
        shap_values
    ).mean(
        axis=0
    )

    mean_signed = shap_values.mean(
        axis=0
    )

    result = pd.DataFrame(
        {
            "feature":
                feature_names,

            "mean_absolute_shap":
                mean_absolute,

            "mean_signed_shap":
                mean_signed,
        }
    )

    result = (
        result
        .sort_values(
            "mean_absolute_shap",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )

    result[
        "importance_rank"
    ] = np.arange(
        1,
        len(result) + 1
    )

    total = result[
        "mean_absolute_shap"
    ].sum()

    result[
        "relative_importance_pct"
    ] = (
        result[
            "mean_absolute_shap"
        ]
        / total
        * 100
    )

    return result


def save_importance_chart(
    dataframe,
    output_file,
    title,
):

    plot_df = (
        dataframe
        .head(
            TOP_GLOBAL_FEATURES
        )
        .sort_values(
            "mean_absolute_shap",
            ascending=True,
        )
    )

    height = max(
        7,
        len(plot_df) * 0.4,
    )

    fig, ax = plt.subplots(
        figsize=(
            11,
            height,
        )
    )

    bars = ax.barh(
        plot_df[
            "feature"
        ],
        plot_df[
            "mean_absolute_shap"
        ],
        color="#6F42C1",
        edgecolor="#4B2E83",
    )

    ax.set_title(
        title,
        fontweight="bold",
    )

    ax.set_xlabel(
        "Mean absolute SHAP value"
    )

    ax.set_ylabel(
        "Feature"
    )

    ax.grid(
        axis="x",
        alpha=0.2,
    )

    for bar in bars:

        value = bar.get_width()

        ax.text(
            value,
            bar.get_y()
            + bar.get_height() / 2,
            f" {value:.3f}",
            va="center",
            fontsize=8,
        )

    fig.tight_layout()

    fig.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )


def save_beeswarm(
    shap_values,
    transformed_data,
    feature_names,
    output_file,
    title,
):

    plt.figure(
        figsize=(11, 8)
    )

    shap.summary_plot(
        shap_values,
        transformed_data,
        feature_names=feature_names,
        max_display=15,
        show=False,
    )

    plt.title(
        title,
        fontweight="bold",
    )

    plt.tight_layout()

    plt.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


def save_local_chart(
    shap_row,
    feature_names,
    prediction,
    base_value,
    output_file,
    title,
):

    order = np.argsort(
        np.abs(
            shap_row
        )
    )[::-1][
        :TOP_LOCAL_FEATURES
    ]

    chart = pd.DataFrame(
        {
            "feature": [
                feature_names[index]
                for index in order
            ],

            "shap_value": [
                shap_row[index]
                for index in order
            ],
        }
    )

    chart = (
        chart
        .sort_values(
            "shap_value",
            ascending=True,
        )
    )

    colours = [
        "#6F42C1"
        if value >= 0
        else "#D8C7F0"
        for value in chart[
            "shap_value"
        ]
    ]

    fig, ax = plt.subplots(
        figsize=(10, 7)
    )

    ax.barh(
        chart[
            "feature"
        ],
        chart[
            "shap_value"
        ],
        color=colours,
        edgecolor="#4B2E83",
    )

    ax.axvline(
        0,
        color="#4B2E83",
        linewidth=1,
    )

    ax.set_title(
        (
            f"{title}\n"
            f"Baseline = {base_value:.2f}, "
            f"Prediction = {prediction:.2f}"
        ),
        fontweight="bold",
    )

    ax.set_xlabel(
        "SHAP contribution"
    )

    ax.set_ylabel(
        "Feature"
    )

    ax.grid(
        axis="x",
        alpha=0.2,
    )

    fig.tight_layout()

    fig.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )


def create_local_records(
    shap_values,
    transformed_values,
    feature_names,
    predictions,
    metadata,
    base_value,
    task,
):

    records = []

    for row_index in range(
        len(metadata)
    ):

        order = np.argsort(
            np.abs(
                shap_values[
                    row_index
                ]
            )
        )[::-1][
            :TOP_LOCAL_FEATURES
        ]

        for rank, feature_index in enumerate(
            order,
            start=1,
        ):

            contribution = float(
                shap_values[
                    row_index,
                    feature_index
                ]
            )

            record = {
                "task":
                    task,

                "example_number":
                    row_index + 1,

                "student_id":
                    metadata.iloc[
                        row_index
                    ][
                        "student_id"
                    ],

                "academic_period":
                    metadata.iloc[
                        row_index
                    ][
                        "academic_period"
                    ],

                "grade":
                    metadata.iloc[
                        row_index
                    ][
                        "grade"
                    ],

                "prediction":
                    float(
                        predictions[
                            row_index
                        ]
                    ),

                "base_value":
                    base_value,

                "rank":
                    rank,

                "feature":
                    feature_names[
                        feature_index
                    ],

                "transformed_value":
                    float(
                        transformed_values[
                            row_index,
                            feature_index
                        ]
                    ),

                "shap_contribution":
                    contribution,

                "direction":
                    (
                        "PUSHES_HIGHER"
                        if contribution > 0
                        else (
                            "PUSHES_LOWER"
                            if contribution < 0
                            else "NO_EFFECT"
                        )
                    ),
            }

            if (
                "target_subject"
                in metadata.columns
            ):

                record[
                    "target_subject"
                ] = metadata.iloc[
                    row_index
                ][
                    "target_subject"
                ]

            records.append(
                record
            )

    return records


print(
    "=" * 110
)

print(
    "STUDENT PERFORMANCE AI - "
    "SHAP EXPLAINABILITY SHAP EXPLAINABILITY"
)

print(
    "=" * 110
)


validate_files(
    [
        OVERALL_DATA_FILE,
        SUBJECT_DATA_FILE,
        OVERALL_MODEL_FILE,
        SUBJECT_MODEL_FILE,
    ]
)


overall_artifact = joblib.load(
    OVERALL_MODEL_FILE
)

subject_artifact = joblib.load(
    SUBJECT_MODEL_FILE
)


overall_pipeline = overall_artifact[
    "pipeline"
]

overall_features = overall_artifact[
    "features"
]


subject_pipeline = subject_artifact[
    "pipeline"
]

subject_numeric_features = subject_artifact[
    "numeric_features"
]

subject_categorical_features = subject_artifact[
    "categorical_features"
]


print(
    "\nFinal models loaded."
)

print(
    f"Overall source features: "
    f"{len(overall_features)}"
)

print(
    f"Subject numeric source features: "
    f"{len(subject_numeric_features)}"
)

print(
    f"Subject categorical features: "
    f"{len(subject_categorical_features)}"
)


# 2026 T2 is used only for post-hoc explanation; no tuning or selection.

overall_df = pd.read_csv(
    OVERALL_DATA_FILE
)

subject_df = pd.read_csv(
    SUBJECT_DATA_FILE
)


overall_test = (
    overall_df[
        overall_df[
            "dataset_split"
        ] == "FINAL_TEST"
    ]
    .copy()
    .reset_index(
        drop=True
    )
)


subject_test = (
    subject_df[
        subject_df[
            "dataset_split"
        ] == "FINAL_TEST"
    ]
    .copy()
    .reset_index(
        drop=True
    )
)


if len(overall_test) != 268:

    raise RuntimeError(
        "\nExpected 268 overall final-test rows."
    )


if len(subject_test) != 2912:

    raise RuntimeError(
        "\nExpected 2912 subject final-test rows."
    )


print(
    "\nExplanation records:"
)

print(
    f"  Overall: "
    f"{len(overall_test)}"
)

print(
    f"  Subject: "
    f"{len(subject_test)}"
)

print(
    "  Model modification: NO"
)


for feature in overall_features:

    overall_test[
        feature
    ] = pd.to_numeric(
        overall_test[
            feature
        ],
        errors="coerce",
    )


X_overall = overall_test[
    overall_features
]


overall_imputer = (
    overall_pipeline
    .named_steps[
        "imputer"
    ]
)

overall_rf = (
    overall_pipeline
    .named_steps[
        "model"
    ]
)


overall_transformed = (
    overall_imputer
    .transform(
        X_overall
    )
)


overall_feature_names = (
    overall_imputer
    .get_feature_names_out(
        overall_features
    )
    .tolist()
)


overall_feature_names = [
    simplify_name(name)
    for name in overall_feature_names
]


overall_predictions = (
    overall_rf
    .predict(
        overall_transformed
    )
)


print(
    "\nCalculating overall SHAP values..."
)


overall_explainer = shap.TreeExplainer(
    overall_rf
)


overall_shap = normalise_shap_values(
    overall_explainer.shap_values(
        overall_transformed,
        check_additivity=False,
    )
)


overall_base = expected_value_to_float(
    overall_explainer.expected_value
)


overall_importance = global_importance(
    overall_shap,
    overall_feature_names,
)


overall_importance.to_csv(
    OVERALL_IMPORTANCE_OUTPUT,
    index=False,
)


save_importance_chart(
    overall_importance,
    FIGURE_DIR
    / "overall_shap_global_importance.png",
    "Overall Prediction — Global SHAP Importance",
)


save_beeswarm(
    overall_shap,
    overall_transformed,
    overall_feature_names,
    FIGURE_DIR
    / "overall_shap_beeswarm.png",
    "Overall Prediction — SHAP Effects",
)


# Select low, middle and high predicted examples.

overall_quantiles = [
    0.20,
    0.50,
    0.80,
]


overall_selected_indices = []


for quantile in overall_quantiles:

    target_value = np.quantile(
        overall_predictions,
        quantile,
    )

    index = int(
        np.argmin(
            np.abs(
                overall_predictions
                - target_value
            )
        )
    )

    if index not in overall_selected_indices:

        overall_selected_indices.append(
            index
        )


overall_local_shap = (
    overall_shap[
        overall_selected_indices
    ]
)

overall_local_values = (
    overall_transformed[
        overall_selected_indices
    ]
)

overall_local_predictions = (
    overall_predictions[
        overall_selected_indices
    ]
)

overall_local_metadata = (
    overall_test[
        [
            "student_id",
            "academic_period",
            "grade",
        ]
    ]
    .iloc[
        overall_selected_indices
    ]
    .reset_index(
        drop=True
    )
)


overall_local_records = create_local_records(
    overall_local_shap,
    overall_local_values,
    overall_feature_names,
    overall_local_predictions,
    overall_local_metadata,
    overall_base,
    "OVERALL_PERFORMANCE",
)


pd.DataFrame(
    overall_local_records
).to_csv(
    OVERALL_LOCAL_OUTPUT,
    index=False,
)


for index in range(
    len(
        overall_local_metadata
    )
):

    save_local_chart(
        overall_local_shap[
            index
        ],
        overall_feature_names,
        overall_local_predictions[
            index
        ],
        overall_base,
        FIGURE_DIR
        / (
            "overall_local_explanation_"
            f"{index + 1}.png"
        ),
        (
            "Overall Forecast — "
            f"Representative Example {index + 1}"
        ),
    )


for feature in subject_numeric_features:

    subject_test[
        feature
    ] = pd.to_numeric(
        subject_test[
            feature
        ],
        errors="coerce",
    )


for feature in subject_categorical_features:

    subject_test[
        feature
    ] = (
        subject_test[
            feature
        ]
        .astype(str)
    )


subject_features = (
    subject_numeric_features
    + subject_categorical_features
)


subject_preprocessor = (
    subject_pipeline
    .named_steps[
        "preprocessor"
    ]
)

subject_rf = (
    subject_pipeline
    .named_steps[
        "model"
    ]
)


sample_size = min(
    SUBJECT_SHAP_SAMPLE,
    len(subject_test),
)


subject_sample = (
    subject_test
    .sample(
        n=sample_size,
        random_state=RANDOM_STATE,
    )
    .reset_index(
        drop=True
    )
)


X_subject_sample = subject_sample[
    subject_features
]


subject_transformed = (
    subject_preprocessor
    .transform(
        X_subject_sample
    )
)


if hasattr(
    subject_transformed,
    "toarray",
):

    subject_transformed = (
        subject_transformed
        .toarray()
    )


subject_feature_names = (
    subject_preprocessor
    .get_feature_names_out()
    .tolist()
)


subject_feature_names = [
    simplify_name(name)
    for name in subject_feature_names
]


print(
    "\nCalculating pooled-subject SHAP values..."
)

print(
    f"Global SHAP sample: "
    f"{sample_size}"
)


subject_explainer = shap.TreeExplainer(
    subject_rf
)


subject_shap = normalise_shap_values(
    subject_explainer.shap_values(
        subject_transformed,
        check_additivity=False,
    )
)


subject_base = expected_value_to_float(
    subject_explainer.expected_value
)


subject_importance = global_importance(
    subject_shap,
    subject_feature_names,
)


subject_importance.to_csv(
    SUBJECT_IMPORTANCE_OUTPUT,
    index=False,
)


grouped_rows = []


for _, row in subject_importance.iterrows():

    feature = row[
        "feature"
    ]

    if feature.startswith(
        "target_subject_"
    ):

        conceptual_feature = (
            "target_subject"
        )

    else:

        conceptual_feature = feature

    grouped_rows.append(
        {
            "feature":
                conceptual_feature,

            "mean_absolute_shap":
                row[
                    "mean_absolute_shap"
                ],

            "mean_signed_shap":
                row[
                    "mean_signed_shap"
                ],
        }
    )


subject_grouped = pd.DataFrame(
    grouped_rows
)


subject_grouped = (
    subject_grouped
    .groupby(
        "feature",
        as_index=False,
    )
    .agg(
        {
            "mean_absolute_shap":
                "sum",

            "mean_signed_shap":
                "sum",
        }
    )
)


subject_grouped = (
    subject_grouped
    .sort_values(
        "mean_absolute_shap",
        ascending=False,
    )
    .reset_index(
        drop=True
    )
)


subject_grouped[
    "importance_rank"
] = np.arange(
    1,
    len(subject_grouped) + 1
)


grouped_total = subject_grouped[
    "mean_absolute_shap"
].sum()


subject_grouped[
    "relative_importance_pct"
] = (
    subject_grouped[
        "mean_absolute_shap"
    ]
    / grouped_total
    * 100
)


subject_grouped.to_csv(
    SUBJECT_GROUPED_OUTPUT,
    index=False,
)


save_importance_chart(
    subject_grouped,
    FIGURE_DIR
    / "subject_shap_global_importance.png",
    "Subject Prediction — Global SHAP Importance",
)


save_beeswarm(
    subject_shap,
    subject_transformed,
    subject_feature_names,
    FIGURE_DIR
    / "subject_shap_beeswarm.png",
    "Subject Prediction — SHAP Effects",
)


# Representative Mathematics, English and Science predictions nearest the median.

X_subject_full = subject_test[
    subject_features
]


subject_full_transformed = (
    subject_preprocessor
    .transform(
        X_subject_full
    )
)


if hasattr(
    subject_full_transformed,
    "toarray",
):

    subject_full_transformed = (
        subject_full_transformed
        .toarray()
    )


subject_full_predictions = (
    subject_rf
    .predict(
        subject_full_transformed
    )
)


preferred_subjects = [
    "mathematics",
    "english",
    "science",
]


subject_selected_indices = []


for target_subject in preferred_subjects:

    positions = np.flatnonzero(
        subject_test[
            "target_subject"
        ]
        .astype(str)
        .to_numpy()
        == target_subject
    )

    if len(positions) == 0:

        continue

    predictions = (
        subject_full_predictions[
            positions
        ]
    )

    median_prediction = np.median(
        predictions
    )

    local_index = int(
        np.argmin(
            np.abs(
                predictions
                - median_prediction
            )
        )
    )

    subject_selected_indices.append(
        int(
            positions[
                local_index
            ]
        )
    )


subject_local_values = (
    subject_full_transformed[
        subject_selected_indices
    ]
)


subject_local_predictions = (
    subject_full_predictions[
        subject_selected_indices
    ]
)


subject_local_shap = normalise_shap_values(
    subject_explainer.shap_values(
        subject_local_values,
        check_additivity=False,
    )
)


subject_local_metadata = (
    subject_test[
        [
            "student_id",
            "academic_period",
            "grade",
            "target_subject",
        ]
    ]
    .iloc[
        subject_selected_indices
    ]
    .reset_index(
        drop=True
    )
)


subject_local_records = create_local_records(
    subject_local_shap,
    subject_local_values,
    subject_feature_names,
    subject_local_predictions,
    subject_local_metadata,
    subject_base,
    "POOLED_SUBJECT",
)


pd.DataFrame(
    subject_local_records
).to_csv(
    SUBJECT_LOCAL_OUTPUT,
    index=False,
)


for index in range(
    len(
        subject_local_metadata
    )
):

    subject_name = (
        str(
            subject_local_metadata
            .iloc[index][
                "target_subject"
            ]
        )
        .replace(
            "_",
            " "
        )
        .title()
    )

    save_local_chart(
        subject_local_shap[
            index
        ],
        subject_feature_names,
        subject_local_predictions[
            index
        ],
        subject_base,
        FIGURE_DIR
        / (
            "subject_local_explanation_"
            f"{index + 1}.png"
        ),
        (
            f"{subject_name} Forecast — "
            "Representative Example"
        ),
    )


summary = {
    "stage":
        "19",

    "method":
        "SHAP TreeExplainer",

    "algorithm":
        "Random Forest",

    "feature_set":
        "Academic/history only",

    "explanation_period":
        "2026_T2",

    "purpose":
        "Post-hoc explanation only",

    "model_modified":
        False,

    "overall": {
        "rows_explained":
            len(
                overall_test
            ),

        "baseline_prediction":
            overall_base,

        "top_10_features":
            overall_importance[
                [
                    "feature",
                    "mean_absolute_shap",
                    "relative_importance_pct",
                ]
            ]
            .head(
                10
            )
            .to_dict(
                orient="records"
            ),
    },

    "subject": {
        "global_sample_rows":
            sample_size,

        "baseline_prediction":
            subject_base,

        "top_10_features":
            subject_grouped[
                [
                    "feature",
                    "mean_absolute_shap",
                    "relative_importance_pct",
                ]
            ]
            .head(
                10
            )
            .to_dict(
                orient="records"
            ),
    },

    "interpretation_rules": {
        "positive_shap":
            (
                "Pushes the prediction above "
                "the model baseline."
            ),

        "negative_shap":
            (
                "Pushes the prediction below "
                "the model baseline."
            ),

        "mean_absolute_shap":
            (
                "Measures average feature influence "
                "on the model prediction."
            ),
    },

    "limitations": [
        (
            "SHAP explains model behaviour "
            "but does not establish causality."
        ),
        (
            "SHAP magnitude is not prediction "
            "confidence."
        ),
        (
            "Prediction reliability must be based "
            "on held-out performance and sample size."
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
    )


print(
    "\n"
    + "=" * 110
)

print(
    "SHAP EXPLAINABILITY SHAP EXPLAINABILITY COMPLETE"
)

print(
    "=" * 110
)


print(
    "\nSHAP INTERPRETATION:"
)

print(
    "  Positive SHAP -> pushes prediction higher."
)

print(
    "  Negative SHAP -> pushes prediction lower."
)

print(
    "  Mean |SHAP| -> average magnitude of influence."
)

print(
    "  SHAP does NOT prove causation."
)


print(
    "\nOVERALL MODEL"
)

print(
    f"  Explained rows: "
    f"{len(overall_test)}"
)

print(
    f"  Baseline prediction: "
    f"{overall_base:.4f}"
)

print(
    "\nTop 15 overall features:"
)


print(
    overall_importance[
        [
            "importance_rank",
            "feature",
            "mean_absolute_shap",
            "relative_importance_pct",
        ]
    ]
    .head(
        15
    )
    .to_string(
        index=False,
        float_format=lambda value: (
            f"{value:.4f}"
        ),
    )
)


print(
    "\nPOOLED SUBJECT MODEL"
)

print(
    f"  SHAP sample rows: "
    f"{sample_size}"
)

print(
    f"  Baseline prediction: "
    f"{subject_base:.4f}"
)

print(
    "\nTop 15 subject features:"
)


print(
    subject_grouped[
        [
            "importance_rank",
            "feature",
            "mean_absolute_shap",
            "relative_importance_pct",
        ]
    ]
    .head(
        15
    )
    .to_string(
        index=False,
        float_format=lambda value: (
            f"{value:.4f}"
        ),
    )
)


print(
    "\nRepresentative local subject examples:"
)

print(
    ", ".join(
        subject_local_metadata[
            "target_subject"
        ]
        .astype(str)
        .tolist()
    )
)


print(
    "\nAggregate reports:"
)

print(
    f"  {OVERALL_IMPORTANCE_OUTPUT}"
)

print(
    f"  {SUBJECT_IMPORTANCE_OUTPUT}"
)

print(
    f"  {SUBJECT_GROUPED_OUTPUT}"
)

print(
    f"  {SUMMARY_OUTPUT}"
)


print(
    "\nPrivate student-level explanation files:"
)

print(
    f"  {OVERALL_LOCAL_OUTPUT}"
)

print(
    f"  {SUBJECT_LOCAL_OUTPUT}"
)


print(
    "\nFigures:"
)

print(
    f"  {FIGURE_DIR}"
)


print(
    "\nIMPORTANT:"
)

print(
    "  SHAP explainability only explains the frozen final models."
)

print(
    "  No training, tuning or model selection occurred."
)

print(
    "  SHAP importance is NOT a confidence score."
)
