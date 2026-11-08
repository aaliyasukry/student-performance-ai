from pathlib import Path
import json
import re

import joblib
import numpy as np
import pandas as pd
import pymysql

from dotenv import dotenv_values
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

ENV_FILE = PROJECT_ROOT / ".env.ai"

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

PRIVATE_OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "private"
    / "integration"
)

REPORT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "integration"
)

PRIVATE_OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


TARGET_YEAR = 2026
TARGET_TERM = 2

ALL_SUBJECTS = [
    "agriculture",
    "arabic_literature",
    "art",
    "ce",
    "commerce",
    "english",
    "english_literature",
    "geography",
    "health_and_physical_education",
    "history",
    "home_science",
    "ict",
    "mathematics",
    "pts",
    "religion",
    "science",
    "sinhala",
    "tamil",
    "tamil_literature",
]

PREDICTIVE_SUBJECTS = [
    "agriculture",
    "arabic_literature",
    "art",
    "ce",
    "commerce",
    "english",
    "english_literature",
    "geography",
    "health_and_physical_education",
    "history",
    "ict",
    "mathematics",
    "religion",
    "science",
    "sinhala",
    "tamil",
    "tamil_literature",
]

STATUS_MAP = {
    "present": "PRESENT",
    "absent": "ABSENT",
    "not_assessed": "NOT_TAKEN",
    "missing": "MISSING",
}

EXPECTED_RESEARCH = {
    "overall_rows": 268,
    "overall_mae": 4.2008,
    "overall_rmse": 5.5205,
    "overall_r2": 0.9148,
    "overall_within_10": 93.28,

    "subject_rows": 2912,
    "subject_mae": 9.9139,
    "subject_rmse": 12.5899,
    "subject_r2": 0.7129,
    "subject_within_10": 59.07,
}


# Exact historical feature academic logic

def previous_value(df, column):
    return (
        df
        .groupby(
            "student_id",
            sort=False,
        )[column]
        .shift(1)
    )


def historical_mean(df, column):
    return (
        df
        .groupby(
            "student_id",
            sort=False,
        )[column]
        .transform(
            lambda series:
                series
                .shift(1)
                .expanding(min_periods=1)
                .mean()
        )
    )


def historical_std(df, column):
    return (
        df
        .groupby(
            "student_id",
            sort=False,
        )[column]
        .transform(
            lambda series:
                series
                .shift(1)
                .expanding(min_periods=2)
                .std()
        )
    )


def historical_count(df, column):
    return (
        df
        .groupby(
            "student_id",
            sort=False,
        )[column]
        .transform(
            lambda series:
                series
                .notna()
                .shift(
                    1,
                    fill_value=False,
                )
                .cumsum()
        )
        .astype(int)
    )


def previous_status(df, column):
    return (
        df
        .groupby(
            "student_id",
            sort=False,
        )[column]
        .shift(1)
    )


def previous_status_flag(
    df,
    column,
    status,
):
    return (
        previous_status(
            df,
            column,
        )
        .eq(status)
        .astype(int)
    )


def previous_subject_recorded(
    df,
    column,
):
    prior = previous_status(
        df,
        column,
    )

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


def get_connection():

    if not ENV_FILE.exists():
        raise FileNotFoundError(
            f".env.ai not found:\n{ENV_FILE}"
        )

    env = dotenv_values(ENV_FILE)

    required = [
        "DB_HOST",
        "DB_NAME",
        "DB_USER",
    ]

    missing = [
        key
        for key in required
        if not env.get(key)
    ]

    if missing:
        raise RuntimeError(
            "Missing DB settings in .env.ai: "
            f"{missing}"
        )

    return pymysql.connect(
        host=env["DB_HOST"],
        port=int(
            env.get(
                "DB_PORT",
                3306,
            )
        ),
        user=env["DB_USER"],
        password=env.get(
            "DB_PASSWORD",
            "",
        ),
        database=env["DB_NAME"],
        charset="utf8mb4",
    )


def load_exam_history(connection):

    sql = """
    SELECT
        e.id AS examination_id,

        ay.name AS academic_year,

        at.id AS academic_term_id,
        at.name AS term_name,
        at.start_date AS term_start_date,

        er.student_id,

        ecs.class_id,
        c.name AS class_name,

        g.id AS grade_id,
        g.name AS grade_name,

        s.id AS subject_id,
        s.name AS portal_subject_name,
        s.code AS portal_subject_code,

        asm.model_subject_key,
        asm.model_subject_name,
        asm.is_predictive,

        er.attendance_status,
        er.mark,
        er.remarks AS result_remarks

    FROM examination_results er

    JOIN examination_class_subjects ecs
        ON ecs.id =
           er.examination_class_subject_id

    JOIN examinations e
        ON e.id =
           ecs.examination_id

    JOIN academic_years ay
        ON ay.id =
           e.academic_year_id

    JOIN academic_terms at
        ON at.id =
           e.academic_term_id

    JOIN classes c
        ON c.id =
           ecs.class_id

    JOIN grades g
        ON g.id =
           c.grade_id

    JOIN subjects s
        ON s.id =
           ecs.subject_id

    JOIN ai_subject_mappings asm
        ON asm.subject_id =
           s.id

    WHERE
        ay.name IN ('2025', '2026')

        AND (
            ay.name = '2025'
            OR at.name IN (
                'First Term',
                'Second Term'
            )
        )

    ORDER BY
        at.start_date,
        er.student_id,
        asm.model_subject_key
    """

    return pd.read_sql_query(
        sql,
        connection,
    )


def grade_number(value):

    match = re.search(
        r"(\d+)",
        str(value),
    )

    if not match:
        raise ValueError(
            f"Could not parse grade: {value}"
        )

    return int(
        match.group(1)
    )


def normalize_status(value, remarks=None):
    remarks_text = str(remarks or "").strip().lower()
    if "source status: missing" in remarks_text or "source value: missing" in remarks_text:
        return "MISSING"
    if value is None:
        return "MISSING"

    normalized = str(value).strip().lower()
    if normalized not in STATUS_MAP:
        raise ValueError(f"Unknown examination status: {value}")
    return STATUS_MAP[normalized]


def build_master_rows(raw):

    if raw.empty:
        raise RuntimeError(
            "No examination history returned."
        )

    raw = raw.copy()

    raw["mark"] = pd.to_numeric(
        raw["mark"],
        errors="coerce",
    )

    # Validate duplicate canonical subjects

    duplicate_mask = raw.duplicated(
        subset=[
            "examination_id",
            "student_id",
            "model_subject_key",
        ],
        keep=False,
    )

    if duplicate_mask.any():

        bad = raw.loc[
            duplicate_mask,
            [
                "examination_id",
                "student_id",
                "model_subject_key",
                "portal_subject_code",
            ],
        ]

        raise RuntimeError(
            "\nDuplicate canonical subjects found:\n"
            f"{bad.to_string(index=False)}"
        )

    # Period ordering

    periods = (
        raw[
            [
                "examination_id",
                "academic_year",
                "academic_term_id",
                "term_name",
                "term_start_date",
            ]
        ]
        .drop_duplicates()
        .sort_values(
            [
                "term_start_date",
                "examination_id",
            ]
        )
        .reset_index(drop=True)
    )

    periods[
        "year"
    ] = pd.to_numeric(
        periods["academic_year"],
        errors="raise",
    ).astype(int)

    periods[
        "term"
    ] = (
        periods
        .groupby(
            "year",
            sort=False,
        )
        .cumcount()
        + 1
    )

    periods[
        "period_order"
    ] = np.arange(
        1,
        len(periods) + 1,
    )

    period_lookup = (
        periods
        .set_index(
            "examination_id"
        )
        .to_dict("index")
    )

    rows = []

    for (
        examination_id,
        student_id,
    ), group in raw.groupby(
        [
            "examination_id",
            "student_id",
        ],
        sort=False,
    ):

        class_ids = (
            group["class_id"]
            .dropna()
            .unique()
        )

        grade_ids = (
            group["grade_id"]
            .dropna()
            .unique()
        )

        if len(class_ids) != 1:
            raise RuntimeError(
                "Student has multiple classes "
                "inside one examination: "
                f"{student_id}, exam "
                f"{examination_id}"
            )

        if len(grade_ids) != 1:
            raise RuntimeError(
                "Student has multiple grades "
                "inside one examination: "
                f"{student_id}, exam "
                f"{examination_id}"
            )

        period = period_lookup[
            examination_id
        ]

        grade = grade_number(
            group.iloc[0]["grade_name"]
        )

        row = {
            "student_id": int(
                student_id
            ),
            "year": int(
                period["year"]
            ),
            "term": int(
                period["term"]
            ),
            "academic_period": (
                f"{int(period['year'])}_T"
                f"{int(period['term'])}"
            ),
            "period_order": int(
                period["period_order"]
            ),
            "grade": grade,
            "class": str(
                group.iloc[0]["class_name"]
            ),
            "academic_stage": (
                "OL"
                if grade == 10
                else "SECONDARY"
            ),
        }

        # Default to genuine source NO_RECORD.
        for subject in ALL_SUBJECTS:

            row[
                f"mark_{subject}"
            ] = np.nan

            row[
                f"status_{subject}"
            ] = "NO_RECORD"

        for _, record in group.iterrows():

            subject = record[
                "model_subject_key"
            ]

            if subject not in ALL_SUBJECTS:
                continue

            status = normalize_status(
                record[
                    "attendance_status"
                ]
            )

            mark = record["mark"]

            if (
                status == "PRESENT"
                and pd.isna(mark)
            ):
                raise RuntimeError(
                    "PRESENT result with NULL mark: "
                    f"student={student_id}, "
                    f"exam={examination_id}, "
                    f"subject={subject}"
                )

            row[
                f"status_{subject}"
            ] = status

            row[
                f"mark_{subject}"
            ] = mark

        numeric_marks = [
            row[f"mark_{subject}"]
            for subject in ALL_SUBJECTS
            if pd.notna(
                row[f"mark_{subject}"]
            )
        ]

        statuses = [
            row[f"status_{subject}"]
            for subject in ALL_SUBJECTS
        ]

        row[
            "numeric_subject_count"
        ] = len(
            numeric_marks
        )

        row[
            "absent_subject_count"
        ] = statuses.count(
            "ABSENT"
        )

        row[
            "not_taken_subject_count"
        ] = statuses.count(
            "NOT_TAKEN"
        )

        row[
            "calculated_average"
        ] = (
            float(
                np.mean(
                    numeric_marks
                )
            )
            if numeric_marks
            else np.nan
        )

        rows.append(row)

    master = pd.DataFrame(
        rows
    )

    master = (
        master
        .sort_values(
            [
                "student_id",
                "period_order",
            ]
        )
        .reset_index(drop=True)
    )

    duplicates = master.duplicated(
        subset=[
            "student_id",
            "year",
            "term",
        ],
        keep=False,
    )

    if duplicates.any():
        raise RuntimeError(
            "Duplicate student-term rows "
            "created."
        )

    return master


def build_historical_features(df):

    df = (
        df
        .sort_values(
            [
                "student_id",
                "period_order",
            ]
        )
        .reset_index(drop=True)
    )

    features = {}

    # History depth

    features[
        "history_term_count"
    ] = (
        df
        .groupby(
            "student_id",
            sort=False,
        )
        .cumcount()
    )

    features[
        "has_prior_history"
    ] = (
        features[
            "history_term_count"
        ]
        .gt(0)
        .astype(int)
    )

    # Overall performance

    features[
        "prev_calculated_average"
    ] = previous_value(
        df,
        "calculated_average",
    )

    features[
        "historical_mean_average"
    ] = historical_mean(
        df,
        "calculated_average",
    )

    features[
        "historical_std_average"
    ] = historical_std(
        df,
        "calculated_average",
    )

    previous_average_2 = (
        df
        .groupby(
            "student_id",
            sort=False,
        )[
            "calculated_average"
        ]
        .shift(2)
    )

    features[
        "performance_trend"
    ] = (
        features[
            "prev_calculated_average"
        ]
        - previous_average_2
    )

    # Participation

    features[
        "prev_numeric_subject_count"
    ] = previous_value(
        df,
        "numeric_subject_count",
    )

    features[
        "prev_absent_subject_count"
    ] = previous_value(
        df,
        "absent_subject_count",
    )

    features[
        "prev_not_taken_subject_count"
    ] = previous_value(
        df,
        "not_taken_subject_count",
    )

    features[
        "historical_mean_absent_subject_count"
    ] = historical_mean(
        df,
        "absent_subject_count",
    )

    features[
        "historical_mean_not_taken_subject_count"
    ] = historical_mean(
        df,
        "not_taken_subject_count",
    )

    # Subject history

    for subject in ALL_SUBJECTS:

        mark_column = (
            f"mark_{subject}"
        )

        status_column = (
            f"status_{subject}"
        )

        previous_mark = previous_value(
            df,
            mark_column,
        )

        features[
            f"prev_mark_{subject}"
        ] = previous_mark

        features[
            f"historical_mean_mark_{subject}"
        ] = historical_mean(
            df,
            mark_column,
        )

        features[
            f"historical_mark_count_{subject}"
        ] = historical_count(
            df,
            mark_column,
        )

        previous_mark_2 = (
            df
            .groupby(
                "student_id",
                sort=False,
            )[mark_column]
            .shift(2)
        )

        features[
            f"mark_trend_{subject}"
        ] = (
            previous_mark
            - previous_mark_2
        )

        features[
            f"prev_absent_{subject}"
        ] = previous_status_flag(
            df,
            status_column,
            "ABSENT",
        )

        features[
            f"prev_not_taken_{subject}"
        ] = previous_status_flag(
            df,
            status_column,
            "NOT_TAKEN",
        )

        features[
            f"prev_recorded_{subject}"
        ] = previous_subject_recorded(
            df,
            status_column,
        )

    feature_block = pd.DataFrame(
        features,
        index=df.index,
    )

    metadata = df[
        [
            "student_id",
            "year",
            "term",
            "academic_period",
            "period_order",
            "grade",
            "class",
            "academic_stage",
        ]
    ].copy()

    return pd.concat(
        [
            metadata,
            feature_block,
        ],
        axis=1,
    )


def calculate_metrics(
    actual,
    predicted,
):

    actual = np.asarray(
        actual,
        dtype=float,
    )

    predicted = np.asarray(
        predicted,
        dtype=float,
    )

    errors = (
        predicted
        - actual
    )

    return {
        "rows": int(
            len(actual)
        ),

        "mae": float(
            mean_absolute_error(
                actual,
                predicted,
            )
        ),

        "rmse": float(
            np.sqrt(
                mean_squared_error(
                    actual,
                    predicted,
                )
            )
        ),

        "r2": float(
            r2_score(
                actual,
                predicted,
            )
        ),

        "within_10_pct": float(
            np.mean(
                np.abs(errors) <= 10
            )
            * 100
        ),

        "mean_residual": float(
            np.mean(
                errors
            )
        ),
    }


def normalize_label(value):

    return re.sub(
        r"[^a-z0-9]+",
        "",
        str(value).lower(),
    )


def main():

    print("=" * 90)
    print("PORTAL MODEL PARITY - LIVE FEATURE PARITY")
    print("=" * 90)

    # DB

    connection = get_connection()

    try:
        raw = load_exam_history(
            connection
        )
    finally:
        connection.close()

    print(
        f"\nPortal result rows loaded: "
        f"{len(raw)}"
    )

    # Master student-term reconstruction

    master = build_master_rows(
        raw
    )

    print(
        f"Student-term rows: "
        f"{len(master)}"
    )

    print(
        f"Unique students: "
        f"{master['student_id'].nunique()}"
    )

    print("\nPeriods:")

    print(
        master[
            [
                "academic_period",
                "period_order",
            ]
        ]
        .drop_duplicates()
        .sort_values(
            "period_order"
        )
        .to_string(
            index=False
        )
    )

    # Feature engineering

    features = (
        build_historical_features(
            master
        )
    )

    # Load frozen models

    overall_artifact = joblib.load(
        OVERALL_MODEL_FILE
    )

    subject_artifact = joblib.load(
        SUBJECT_MODEL_FILE
    )

    overall_pipeline = (
        overall_artifact[
            "pipeline"
        ]
    )

    subject_pipeline = (
        subject_artifact[
            "pipeline"
        ]
    )

    overall_features = (
        overall_artifact[
            "features"
        ]
    )

    subject_numeric = (
        subject_artifact[
            "numeric_features"
        ]
    )

    # Validate overall contract

    missing_overall = [
        column
        for column in overall_features
        if column not in features.columns
    ]

    if missing_overall:
        raise RuntimeError(
            "Missing overall model features:\n"
            f"{missing_overall}"
        )

    print(
        "\nOverall model feature contract: "
        f"{len(overall_features)} / "
        f"{len(overall_features)} available"
    )

    # Target 2026 T2

    target_feature_mask = (
        (
            features["year"]
            == TARGET_YEAR
        )
        &
        (
            features["term"]
            == TARGET_TERM
        )
    )

    target_features = (
        features[
            target_feature_mask
        ]
        .copy()
    )

    target_master = (
        master[
            (
                master["year"]
                == TARGET_YEAR
            )
            &
            (
                master["term"]
                == TARGET_TERM
            )
        ]
        .copy()
    )

    target = target_features.merge(
        target_master,
        on=[
            "student_id",
            "year",
            "term",
            "academic_period",
            "period_order",
            "grade",
            "class",
            "academic_stage",
        ],
        how="inner",
        suffixes=(
            "",
            "_actual",
        ),
    )


    overall_rows = target[
        (
            target[
                "history_term_count"
            ] > 0
        )
        &
        (
            target[
                "calculated_average"
            ].notna()
        )
    ].copy()

    X_overall = overall_rows[
        overall_features
    ].copy()

    overall_predictions = (
        overall_pipeline.predict(
            X_overall
        )
    )

    overall_rows[
        "predicted_average"
    ] = overall_predictions

    overall_rows[
        "actual_average"
    ] = overall_rows[
        "calculated_average"
    ]

    overall_rows[
        "error"
    ] = (
        overall_rows[
            "predicted_average"
        ]
        - overall_rows[
            "actual_average"
        ]
    )

    overall_metrics = calculate_metrics(
        overall_rows[
            "actual_average"
        ],
        overall_rows[
            "predicted_average"
        ],
    )


    preprocessor = (
        subject_pipeline
        .named_steps[
            "preprocessor"
        ]
    )

    subject_encoder = (
        preprocessor
        .named_transformers_[
            "subject"
        ]
    )

    model_categories = list(
        subject_encoder.categories_[0]
    )

    category_lookup = {
        normalize_label(category):
        category
        for category
        in model_categories
    }

    canonical_names = (
        raw[
            [
                "model_subject_key",
                "model_subject_name",
            ]
        ]
        .drop_duplicates()
    )

    subject_name_lookup = dict(
        zip(
            canonical_names[
                "model_subject_key"
            ],
            canonical_names[
                "model_subject_name"
            ],
        )
    )

    model_subject_lookup = {}

    for subject in PREDICTIVE_SUBJECTS:

        candidates = [
            subject,
            subject_name_lookup.get(
                subject,
                subject,
            ),
        ]

        matched = None

        for candidate in candidates:

            key = normalize_label(
                candidate
            )

            if key in category_lookup:
                matched = (
                    category_lookup[
                        key
                    ]
                )
                break

        if matched is None:
            raise RuntimeError(
                "Could not map subject "
                f"'{subject}' to trained "
                "model category.\n"
                f"Model categories: "
                f"{model_categories}"
            )

        model_subject_lookup[
            subject
        ] = matched

    print(
        "Subject categories resolved: "
        f"{len(model_subject_lookup)}"
    )


    subject_rows = []

    for _, row in target.iterrows():

        if (
            row[
                "history_term_count"
            ]
            <= 0
        ):
            continue

        for subject in PREDICTIVE_SUBJECTS:

            actual_mark = row[
                f"mark_{subject}"
            ]

            if pd.isna(
                actual_mark
            ):
                continue

            subject_row = {
                "student_id":
                    row["student_id"],

                "year":
                    row["year"],

                "term":
                    row["term"],

                "grade":
                    row["grade"],

                "target_subject":
                    model_subject_lookup[
                        subject
                    ],

                "actual_mark":
                    float(
                        actual_mark
                    ),

                "historical_mean_absent_subject_count":
                    row[
                        "historical_mean_absent_subject_count"
                    ],

                "historical_mean_average":
                    row[
                        "historical_mean_average"
                    ],

                "historical_mean_not_taken_subject_count":
                    row[
                        "historical_mean_not_taken_subject_count"
                    ],

                "historical_mean_subject_mark":
                    row[
                        f"historical_mean_mark_{subject}"
                    ],

                "historical_std_average":
                    row[
                        "historical_std_average"
                    ],

                "historical_subject_mark_count":
                    row[
                        f"historical_mark_count_{subject}"
                    ],

                "history_term_count":
                    row[
                        "history_term_count"
                    ],

                "performance_trend":
                    row[
                        "performance_trend"
                    ],

                "prev_absent_subject_count":
                    row[
                        "prev_absent_subject_count"
                    ],

                "prev_calculated_average":
                    row[
                        "prev_calculated_average"
                    ],

                "prev_not_taken_subject_count":
                    row[
                        "prev_not_taken_subject_count"
                    ],

                "prev_numeric_subject_count":
                    row[
                        "prev_numeric_subject_count"
                    ],

                "prev_subject_absent":
                    row[
                        f"prev_absent_{subject}"
                    ],

                "prev_subject_mark":
                    row[
                        f"prev_mark_{subject}"
                    ],

                "prev_subject_not_taken":
                    row[
                        f"prev_not_taken_{subject}"
                    ],

                "subject_mark_trend":
                    row[
                        f"mark_trend_{subject}"
                    ],
            }

            subject_rows.append(
                subject_row
            )

    subject_df = pd.DataFrame(
        subject_rows
    )

    subject_contract = (
        subject_numeric
        + ["target_subject"]
    )

    missing_subject_features = [
        column
        for column
        in subject_contract
        if column
        not in subject_df.columns
    ]

    if missing_subject_features:
        raise RuntimeError(
            "Missing subject model "
            "features:\n"
            f"{missing_subject_features}"
        )

    X_subject = subject_df[
        subject_contract
    ].copy()

    subject_predictions = (
        subject_pipeline.predict(
            X_subject
        )
    )

    subject_df[
        "predicted_mark"
    ] = subject_predictions

    subject_df[
        "error"
    ] = (
        subject_df[
            "predicted_mark"
        ]
        - subject_df[
            "actual_mark"
        ]
    )

    subject_metrics = calculate_metrics(
        subject_df[
            "actual_mark"
        ],
        subject_df[
            "predicted_mark"
        ],
    )


    master.to_csv(
        PRIVATE_OUTPUT_DIR
        / "portal_parity_master_rows.csv",
        index=False,
    )

    features.to_csv(
        PRIVATE_OUTPUT_DIR
        / "portal_parity_historical_features.csv",
        index=False,
    )

    overall_rows[
        [
            "student_id",
            "year",
            "term",
            "grade",
            "class",
            "actual_average",
            "predicted_average",
            "error",
        ]
    ].to_csv(
        PRIVATE_OUTPUT_DIR
        / "portal_parityd_overall_predictions_2026_t2.csv",
        index=False,
    )

    subject_df.to_csv(
        PRIVATE_OUTPUT_DIR
        / "portal_parityd_subject_predictions_2026_t2.csv",
        index=False,
    )


    summary = {
        "portal": {
            "raw_result_rows":
                int(len(raw)),

            "student_term_rows":
                int(len(master)),

            "unique_students":
                int(
                    master[
                        "student_id"
                    ].nunique()
                ),
        },

        "feature_contract": {
            "overall_required_features":
                int(
                    len(
                        overall_features
                    )
                ),

            "overall_missing_features":
                missing_overall,

            "subject_numeric_features":
                int(
                    len(
                        subject_numeric
                    )
                ),

            "subject_categorical_features":
                1,
        },

        "live_2026_t2": {
            "overall":
                overall_metrics,

            "subject":
                subject_metrics,
        },

        "research_2026_t2":
            EXPECTED_RESEARCH,
    }

    summary[
        "comparison"
    ] = {
        "overall_row_difference":
            overall_metrics[
                "rows"
            ]
            - EXPECTED_RESEARCH[
                "overall_rows"
            ],

        "overall_mae_difference":
            overall_metrics[
                "mae"
            ]
            - EXPECTED_RESEARCH[
                "overall_mae"
            ],

        "subject_row_difference":
            subject_metrics[
                "rows"
            ]
            - EXPECTED_RESEARCH[
                "subject_rows"
            ],

        "subject_mae_difference":
            subject_metrics[
                "mae"
            ]
            - EXPECTED_RESEARCH[
                "subject_mae"
            ],
    }

    summary_file = (
        REPORT_DIR
        / "portal_model_parity.json"
    )

    with open(
        summary_file,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            summary,
            file,
            indent=2,
        )


    print("\n" + "=" * 90)
    print("OVERALL MODEL - LIVE PORTAL 2026 T2")
    print("=" * 90)

    print(
        f"Rows:      "
        f"{overall_metrics['rows']}"
    )
    print(
        f"MAE:       "
        f"{overall_metrics['mae']:.4f}"
    )
    print(
        f"RMSE:      "
        f"{overall_metrics['rmse']:.4f}"
    )
    print(
        f"R²:        "
        f"{overall_metrics['r2']:.4f}"
    )
    print(
        f"Within ±10:"
        f" "
        f"{overall_metrics['within_10_pct']:.2f}%"
    )

    print("\nResearch reference:")
    print(
        "Rows:      268"
    )
    print(
        "MAE:       4.2008"
    )
    print(
        "RMSE:      5.5205"
    )
    print(
        "R²:        0.9148"
    )
    print(
        "Within ±10: 93.28%"
    )

    print("\n" + "=" * 90)
    print("SUBJECT MODEL - LIVE PORTAL 2026 T2")
    print("=" * 90)

    print(
        f"Rows:      "
        f"{subject_metrics['rows']}"
    )
    print(
        f"MAE:       "
        f"{subject_metrics['mae']:.4f}"
    )
    print(
        f"RMSE:      "
        f"{subject_metrics['rmse']:.4f}"
    )
    print(
        f"R²:        "
        f"{subject_metrics['r2']:.4f}"
    )
    print(
        f"Within ±10:"
        f" "
        f"{subject_metrics['within_10_pct']:.2f}%"
    )

    print("\nResearch reference:")
    print(
        "Rows:      2912"
    )
    print(
        "MAE:       9.9139"
    )
    print(
        "RMSE:      12.5899"
    )
    print(
        "R²:        0.7129"
    )
    print(
        "Within ±10: 59.07%"
    )

    print("\n" + "=" * 90)
    print("PORTAL MODEL PARITY COMPLETE")
    print("=" * 90)

    print(
        "\nSummary:\n"
        f"{summary_file}"
    )

    print(
        "\nPrivate row-level files:\n"
        f"{PRIVATE_OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()
