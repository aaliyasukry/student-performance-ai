import hashlib
import json
import math
import re
from uuid import uuid4

import numpy as np
import pandas as pd

from app.repositories.ai_repository import AIRepository
from app.repositories.school_repository import SchoolRepository
from app.services.model_service import ModelService


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
    "exempt": "NOT_TAKEN",
    "not_assessed": "NOT_TAKEN",
    # Legacy research-only status kept for older imported rows.
    "missing": "MISSING",
}

FALLBACK_RELIABILITY = {
    "english": ("HIGH", "SHOW"),
    "history": ("HIGH", "SHOW"),
    "science": ("HIGH", "SHOW"),
    "religion": ("HIGH", "SHOW"),
    "health_and_physical_education": ("MODERATE", "SHOW"),
    "tamil": ("MODERATE", "SHOW"),
    "ce": ("MODERATE", "SHOW"),
    "arabic_literature": ("MODERATE", "SHOW"),
    "mathematics": ("MODERATE", "SHOW"),
    "geography": ("MODERATE", "SHOW"),
    "sinhala": ("MODERATE", "SHOW"),
    "ict": ("MODERATE", "SHOW"),
    "art": ("LOW", "SHOW_WITH_CAUTION"),
    "agriculture": ("LOW", "SHOW_WITH_CAUTION"),
    "english_literature": ("INSUFFICIENT_DATA", "SUPPRESS_BY_DEFAULT"),
    "commerce": ("INSUFFICIENT_DATA", "SUPPRESS_BY_DEFAULT"),
    "tamil_literature": ("INSUFFICIENT_DATA", "SUPPRESS_BY_DEFAULT"),
}


# Historical feature helpers


def previous_value(df, column):
    return df.groupby("student_id", sort=False)[column].shift(1)


def historical_mean(df, column):
    return df.groupby("student_id", sort=False)[column].transform(
        lambda series: series.shift(1).expanding(min_periods=1).mean()
    )


def historical_std(df, column):
    return df.groupby("student_id", sort=False)[column].transform(
        lambda series: series.shift(1).expanding(min_periods=2).std()
    )


def historical_count(df, column):
    return (
        df.groupby("student_id", sort=False)[column]
        .transform(
            lambda series: series.notna()
            .shift(1, fill_value=False)
            .cumsum()
        )
        .astype(int)
    )


def previous_status(df, column):
    return df.groupby("student_id", sort=False)[column].shift(1)


def previous_status_flag(df, column, status):
    return previous_status(df, column).eq(status).astype(int)


def previous_subject_recorded(df, column):
    prior = previous_status(df, column)
    normalized = prior.astype("string").str.strip().str.upper()
    return (prior.notna() & normalized.ne("NO_RECORD")).astype(int)


# Utility helpers


def grade_number(value):
    match = re.search(r"(\d+)", str(value))
    if not match:
        raise ValueError(f"Could not parse grade from '{value}'.")
    return int(match.group(1))


def normalize_status(value, remarks=None):
    """Convert School Portal result states into the research/model semantics."""
    normalized_remarks = str(remarks or "").strip().lower()

    source_missing_marker = (
        "source status: missing" in normalized_remarks
        or "source value: missing" in normalized_remarks
    )

    if source_missing_marker:
        return "MISSING"

    if value is None:
        return "MISSING"

    normalized = str(value).strip().lower()
    if normalized not in STATUS_MAP:
        raise ValueError(f"Unknown examination status '{value}'.")

    return STATUS_MAP[normalized]


def normalize_label(value):
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}

    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]

    if isinstance(value, (np.integer,)):
        return int(value)

    if isinstance(value, (np.floating,)):
        value = float(value)

    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value

    if pd.isna(value):
        return None

    return value


def model_input_hash(features: dict):
    payload = json.dumps(
        json_safe(features),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def academic_risk(value: float):
    if value < 40:
        return "HIGH_RISK"
    if value < 50:
        return "ATTENTION"
    if value < 75:
        return "ON_TRACK"
    return "HIGH_PERFORMER"


def trend_label(predicted: float, previous_value_: float | None):
    if previous_value_ is None:
        return "INSUFFICIENT_HISTORY"

    difference = predicted - float(previous_value_)
    if difference >= 3:
        return "IMPROVING"
    if difference <= -3:
        return "DECLINING"
    return "STABLE"



def attendance_level(value: float | None):
    if value is None:
        return "NO_DATA"
    if value >= 90:
        return "HIGH"
    if value >= 80:
        return "GOOD"
    if value >= 70:
        return "MODERATE"
    return "LOW"


def attendance_trend(current: float | None, previous: float | None):
    if current is None or previous is None:
        return "INSUFFICIENT_HISTORY"
    difference = float(current) - float(previous)
    if difference >= 3:
        return "IMPROVING"
    if difference <= -3:
        return "DECLINING"
    return "STABLE"


def subject_standing(predicted: float, predicted_overall: float):
    difference = float(predicted) - float(predicted_overall)
    if difference >= 10:
        return "STRONG"
    if difference <= -10:
        return "WEAK"
    return "CONSISTENT"


class PredictionService:
    def __init__(
        self,
        model_service: ModelService,
        school_repository: SchoolRepository,
        ai_repository: AIRepository,
    ):
        self.model_service = model_service
        self.school_repository = school_repository
        self.ai_repository = ai_repository

    # Build research-equivalent student-term history

    def _build_master(self, history_rows, target_term, target_class):
        if not history_rows:
            raise ValueError(
                "No historical examination results exist before the target term."
            )

        raw = pd.DataFrame(history_rows)
        raw["mark"] = pd.to_numeric(raw["mark"], errors="coerce")

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
            raise ValueError(
                "Duplicate canonical subjects exist in historical data:\n"
                + bad.to_string(index=False)
            )

        # One term examination per academic year/term is expected for this model.
        period_exam_counts = (
            raw[["academic_year_name", "academic_term_id", "examination_id"]]
            .drop_duplicates()
            .groupby(["academic_year_name", "academic_term_id"])["examination_id"]
            .nunique()
        )

        if (period_exam_counts > 1).any():
            raise ValueError(
                "More than one published examination exists for the same student/term. "
                "The current model expects one term-level result set per academic period."
            )

        rows = []

        for examination_id, group in raw.groupby("examination_id", sort=False):
            first = group.iloc[0]
            grade = grade_number(first["grade_name"])

            row = {
                "student_id": int(first["student_id"]),
                "year": int(first["academic_year_name"]),
                "term": int(first["term_number"]),
                "academic_period": (
                    f"{int(first['academic_year_name'])}_T{int(first['term_number'])}"
                ),
                "period_order": (
                    int(first["academic_year_name"]) * 10
                    + int(first["term_number"])
                ),
                "grade": grade,
                "class": str(first["class_name"]),
                "academic_stage": "OL" if grade == 10 else "SECONDARY",
            }

            for subject in ALL_SUBJECTS:
                row[f"mark_{subject}"] = np.nan
                row[f"status_{subject}"] = "NO_RECORD"

            for _, record in group.iterrows():
                subject = record["model_subject_key"]
                if subject not in ALL_SUBJECTS:
                    continue

                status = normalize_status(
                    record["attendance_status"],
                    record.get("result_remarks"),
                )
                mark = record["mark"]

                if status == "PRESENT" and pd.isna(mark):
                    raise ValueError(
                        "PRESENT historical result has a NULL mark: "
                        f"exam={examination_id}, subject={subject}."
                    )

                row[f"status_{subject}"] = status
                row[f"mark_{subject}"] = mark

            numeric_marks = [
                row[f"mark_{subject}"]
                for subject in ALL_SUBJECTS
                if pd.notna(row[f"mark_{subject}"])
            ]

            statuses = [row[f"status_{subject}"] for subject in ALL_SUBJECTS]

            row["numeric_subject_count"] = len(numeric_marks)
            row["absent_subject_count"] = statuses.count("ABSENT")
            row["not_taken_subject_count"] = statuses.count("NOT_TAKEN")
            row["calculated_average"] = (
                float(np.mean(numeric_marks)) if numeric_marks else np.nan
            )

            rows.append(row)

        target_grade = grade_number(target_class["grade_name"])
        if target_grade < 6 or target_grade > 10:
            raise ValueError(
                f"The deployed models support Grades 6-10 only; target grade is {target_grade}."
            )

        target_row = {
            "student_id": int(history_rows[0]["student_id"]),
            "year": int(target_term["academic_year_name"]),
            "term": int(target_term["term_number"]),
            "academic_period": target_term["target_period"],
            "period_order": (
                int(target_term["academic_year_name"]) * 10
                + int(target_term["term_number"])
            ),
            "grade": target_grade,
            "class": str(target_class["class_name"]),
            "academic_stage": "OL" if target_grade == 10 else "SECONDARY",
            "numeric_subject_count": np.nan,
            "absent_subject_count": np.nan,
            "not_taken_subject_count": np.nan,
            "calculated_average": np.nan,
        }

        for subject in ALL_SUBJECTS:
            target_row[f"mark_{subject}"] = np.nan
            target_row[f"status_{subject}"] = "NO_RECORD"

        rows.append(target_row)

        master = pd.DataFrame(rows)
        master = master.sort_values(
            ["student_id", "period_order"]
        ).reset_index(drop=True)

        if master.duplicated(
            subset=["student_id", "year", "term"],
            keep=False,
        ).any():
            raise ValueError("Duplicate student-term rows were generated.")

        return master

    # Historical academic feature construction

    def _build_historical_features(self, df):
        df = df.sort_values(
            ["student_id", "period_order"]
        ).reset_index(drop=True)

        features = {}

        features["history_term_count"] = df.groupby(
            "student_id", sort=False
        ).cumcount()

        features["has_prior_history"] = (
            features["history_term_count"].gt(0).astype(int)
        )

        features["prev_calculated_average"] = previous_value(
            df, "calculated_average"
        )
        features["historical_mean_average"] = historical_mean(
            df, "calculated_average"
        )
        features["historical_std_average"] = historical_std(
            df, "calculated_average"
        )

        previous_average_2 = df.groupby(
            "student_id", sort=False
        )["calculated_average"].shift(2)

        features["performance_trend"] = (
            features["prev_calculated_average"] - previous_average_2
        )

        features["prev_numeric_subject_count"] = previous_value(
            df, "numeric_subject_count"
        )
        features["prev_absent_subject_count"] = previous_value(
            df, "absent_subject_count"
        )
        features["prev_not_taken_subject_count"] = previous_value(
            df, "not_taken_subject_count"
        )
        features["historical_mean_absent_subject_count"] = historical_mean(
            df, "absent_subject_count"
        )
        features["historical_mean_not_taken_subject_count"] = historical_mean(
            df, "not_taken_subject_count"
        )

        for subject in ALL_SUBJECTS:
            mark_column = f"mark_{subject}"
            status_column = f"status_{subject}"

            previous_mark = previous_value(df, mark_column)

            features[f"prev_mark_{subject}"] = previous_mark
            features[f"historical_mean_mark_{subject}"] = historical_mean(
                df, mark_column
            )
            features[f"historical_mark_count_{subject}"] = historical_count(
                df, mark_column
            )

            previous_mark_2 = df.groupby(
                "student_id", sort=False
            )[mark_column].shift(2)

            features[f"mark_trend_{subject}"] = (
                previous_mark - previous_mark_2
            )

            features[f"prev_absent_{subject}"] = previous_status_flag(
                df, status_column, "ABSENT"
            )
            features[f"prev_not_taken_{subject}"] = previous_status_flag(
                df, status_column, "NOT_TAKEN"
            )
            features[f"prev_recorded_{subject}"] = previous_subject_recorded(
                df, status_column
            )

        feature_block = pd.DataFrame(features, index=df.index)

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

        return pd.concat([metadata, feature_block], axis=1)

    def _subject_category_lookup(self, history_rows):
        categories = self.model_service.subject_categories
        category_lookup = {
            normalize_label(category): category
            for category in categories
        }

        canonical_names = {}
        for row in history_rows:
            canonical_names[row["model_subject_key"]] = row["model_subject_name"]

        result = {}

        for subject in PREDICTIVE_SUBJECTS:
            candidates = [subject, canonical_names.get(subject, subject)]
            matched = None

            for candidate in candidates:
                key = normalize_label(candidate)
                if key in category_lookup:
                    matched = category_lookup[key]
                    break

            if matched is None:
                raise ValueError(
                    f"Could not map canonical subject '{subject}' to trained "
                    f"model categories: {categories}"
                )

            result[subject] = matched

        return result

    def _actuals(self, actual_rows):
        if not actual_rows:
            return {
                "overall": None,
                "subjects": {},
            }

        seen = set()
        numeric_marks = []
        subjects = {}

        for row in actual_rows:
            subject = row["model_subject_key"]
            if subject not in ALL_SUBJECTS:
                continue

            if subject in seen:
                raise ValueError(
                    f"Duplicate canonical target subject '{subject}' exists."
                )
            seen.add(subject)

            mark = row["mark"]
            if mark is not None:
                mark = float(mark)
                numeric_marks.append(mark)

            if subject in PREDICTIVE_SUBJECTS and mark is not None:
                subjects[subject] = {
                    "value": mark,
                    "source_record_id": int(row["examination_result_id"]),
                }

        overall = None
        if numeric_marks:
            overall = {
                "value": float(np.mean(numeric_marks)),
                "source_record_id": None,
            }

        return {
            "overall": overall,
            "subjects": subjects,
        }


    def _academic_history(self, master):
        periods = []
        for _, row in master.iloc[:-1].iterrows():
            periods.append(
                {
                    "year": int(row["year"]),
                    "term": int(row["term"]),
                    "academic_period": row["academic_period"],
                    "grade": int(row["grade"]),
                    "class": str(row["class"]),
                    "average": json_safe(row.get("calculated_average")),
                    "numeric_subject_count": int(row.get("numeric_subject_count", 0) or 0),
                }
            )
        return periods

    def _attendance_period_summary(self, rows):
        if not rows:
            return None

        first = rows[0]
        statuses = [str(row.get("attendance_status") or "").lower() for row in rows]
        present = statuses.count("present")
        late = statuses.count("late")
        absent = statuses.count("absent")
        excused = statuses.count("excused")
        total = len(statuses)
        attended = present + late
        percentage = (attended / total * 100.0) if total else None

        return {
            "year": int(first["academic_year_name"]),
            "term": int(first["term_number"]),
            "academic_period": f"{int(first['academic_year_name'])}_T{int(first['term_number'])}",
            "term_name": first["academic_term_name"],
            "class": first["class_name"],
            "grade": first["grade_name"],
            "total_records": total,
            "attended_count": attended,
            "present_count": present,
            "late_count": late,
            "absent_count": absent,
            "excused_count": excused,
            "attendance_pct": round(percentage, 2) if percentage is not None else None,
            "attendance_level": attendance_level(percentage),
        }

    def _build_attendance_context(self, historical_rows, target_rows=None):
        target_rows = target_rows or []
        grouped = {}

        for row in historical_rows:
            key = (int(row["academic_year_name"]), int(row["term_number"]))
            grouped.setdefault(key, []).append(row)

        historical_periods = [
            self._attendance_period_summary(grouped[key])
            for key in sorted(grouped)
        ]
        historical_periods = [item for item in historical_periods if item is not None]

        target_period = self._attendance_period_summary(target_rows)

        total_records = sum(item["total_records"] for item in historical_periods)
        attended_count = sum(item["attended_count"] for item in historical_periods)
        historical_average = (
            attended_count / total_records * 100.0
            if total_records
            else None
        )

        latest = historical_periods[-1] if historical_periods else None
        previous = historical_periods[-2] if len(historical_periods) >= 2 else None
        trend = attendance_trend(
            latest.get("attendance_pct") if latest else None,
            previous.get("attendance_pct") if previous else None,
        )

        display_period = target_period or latest

        return {
            "available": bool(historical_periods or target_period),
            "used_by_final_predictor": False,
            "calculation_note": (
                "Attendance percentage is attended sessions (present + late) divided "
                "by all recorded completed/locked attendance sessions. Attendance is "
                "contextual only and is not an input to the V1 predictor."
            ),
            "historical_periods": historical_periods,
            "historical_average_pct": (
                round(historical_average, 2)
                if historical_average is not None
                else None
            ),
            "latest_historical_period": latest,
            "historical_trend": trend,
            "target_period": target_period,
            "display_period": display_period,
        }

    def predict_cohort(
        self,
        *,
        target_year: int,
        target_term: int,
        grade_id: int,
        class_id: int | None,
        persist: bool,
        source: str,
        requested_by_user_id: int | None,
        include_actual_if_available: bool,
    ):
        target = self.school_repository.get_target_term(target_year, target_term)
        if not target:
            raise ValueError(
                f"Academic year {target_year} does not contain term {target_term}."
            )

        students = self.school_repository.get_cohort_students(
            target_year=target_year,
            grade_id=grade_id,
            class_id=class_id,
        )
        if not students:
            raise ValueError("No students were found for the selected cohort.")

        results = []
        failures = []
        subject_accumulator = {subject: [] for subject in PREDICTIVE_SUBJECTS}

        for student in students:
            try:
                prediction = self.predict_student(
                    student_id=int(student["id"]),
                    target_year=target_year,
                    target_term=target_term,
                    persist=persist,
                    source=source,
                    requested_by_user_id=requested_by_user_id,
                    include_actual_if_available=include_actual_if_available,
                )
            except Exception as exc:
                failures.append(
                    {
                        "student_id": int(student["id"]),
                        "full_name": student["full_name"],
                        "class": student["class_name"],
                        "reason": str(exc),
                    }
                )
                continue

            display_attendance = prediction["attendance_context"].get("display_period")
            attendance_pct = (
                display_attendance.get("attendance_pct")
                if display_attendance
                else None
            )
            attendance_period = (
                display_attendance.get("academic_period")
                if display_attendance
                else None
            )

            row = {
                "student_id": prediction["student"]["id"],
                "full_name": prediction["student"]["full_name"],
                "student_index": prediction["student"]["student_index"],
                "admission_no": prediction["student"]["admission_no"],
                "grade": prediction["target"]["grade"],
                "class": prediction["target"]["class"],
                "predicted_average": prediction["overall"]["predicted_average"],
                "actual_average": prediction["overall"]["actual_average"],
                "academic_risk": prediction["overall"]["academic_risk"],
                "prediction_reliability": prediction["overall"]["prediction_reliability"],
                "performance_trend": prediction["overall"]["performance_trend"],
                "attendance_pct": attendance_pct,
                "attendance_level": attendance_level(attendance_pct),
                "attendance_period": attendance_period,
                "prediction_run_id": prediction["persistence"]["prediction_run_id"],
            }
            results.append(row)

            for subject in prediction["subjects"]:
                subject_accumulator[subject["subject_key"]].append(subject)

        if not results:
            raise ValueError(
                "No eligible students could be predicted for the selected cohort."
            )

        predicted_values = [float(item["predicted_average"]) for item in results]
        risk_distribution = {
            "HIGH_RISK": 0,
            "ATTENTION": 0,
            "ON_TRACK": 0,
            "HIGH_PERFORMER": 0,
        }
        for item in results:
            risk = item["academic_risk"]
            risk_distribution[risk] = risk_distribution.get(risk, 0) + 1

        attendance_values = [
            float(item["attendance_pct"])
            for item in results
            if item["attendance_pct"] is not None
        ]
        attendance_distribution = {
            "HIGH": 0,
            "GOOD": 0,
            "MODERATE": 0,
            "LOW": 0,
            "NO_DATA": 0,
        }
        for item in results:
            level = item["attendance_level"]
            attendance_distribution[level] = attendance_distribution.get(level, 0) + 1

        low_attendance = sum(1 for item in results if item["attendance_level"] == "LOW")
        high_risk_low_attendance = sum(
            1
            for item in results
            if item["academic_risk"] == "HIGH_RISK"
            and item["attendance_level"] == "LOW"
        )

        subject_summary = []
        for subject, rows in subject_accumulator.items():
            if not rows:
                continue
            predictions = [float(row["predicted_mark"]) for row in rows]
            subject_summary.append(
                {
                    "subject_key": subject,
                    "target_subject": rows[0]["target_subject"],
                    "student_count": len(rows),
                    "predicted_average": round(float(np.mean(predictions)), 2),
                    "high_risk_students": sum(
                        1 for row in rows if row["academic_risk"] == "HIGH_RISK"
                    ),
                    "attention_students": sum(
                        1 for row in rows if row["academic_risk"] == "ATTENTION"
                    ),
                    "on_track_students": sum(
                        1 for row in rows if row["academic_risk"] == "ON_TRACK"
                    ),
                    "high_performer_students": sum(
                        1 for row in rows if row["academic_risk"] == "HIGH_PERFORMER"
                    ),
                    "prediction_reliability": rows[0]["prediction_reliability"],
                    "display_policy": rows[0]["display_policy"],
                }
            )

        subject_summary.sort(key=lambda item: item["predicted_average"])
        results.sort(key=lambda item: float(item["predicted_average"]))

        first = results[0]
        class_label = first["class"] if class_id is not None else "All classes"

        return {
            "target": {
                "year": target_year,
                "term": target_term,
                "period": target["target_period"],
                "term_name": target["academic_term_name"],
                "grade": first["grade"],
                "class": class_label,
            },
            "filters": {
                "grade_id": grade_id,
                "class_id": class_id,
            },
            "summary": {
                "students_found": len(students),
                "students_predicted": len(results),
                "students_failed": len(failures),
                "predicted_class_average": round(float(np.mean(predicted_values)), 2),
                "highest_predicted_average": round(max(predicted_values), 2),
                "lowest_predicted_average": round(min(predicted_values), 2),
                "risk_distribution": risk_distribution,
                "attendance_average_pct": (
                    round(float(np.mean(attendance_values)), 2)
                    if attendance_values
                    else None
                ),
                "attendance_distribution": attendance_distribution,
                "low_attendance_students": low_attendance,
                "high_risk_low_attendance_students": high_risk_low_attendance,
            },
            "students": results,
            "subject_summary": subject_summary,
            "failures": failures,
            "persistence": {
                "requested": persist,
                "mode": (
                    "one prediction run per successfully predicted student"
                    if persist
                    else "not persisted"
                ),
            },
            "attendance_note": (
                "Attendance is displayed as contextual decision-support information "
                "and is not used by the V1 prediction model."
            ),
        }

    def predict_student(
        self,
        *,
        student_id: int,
        target_year: int,
        target_term: int,
        persist: bool,
        source: str,
        requested_by_user_id: int | None,
        include_actual_if_available: bool,
    ):
        student = self.school_repository.get_student(student_id)
        if not student:
            raise ValueError(f"Student {student_id} was not found.")

        target = self.school_repository.get_target_term(
            target_year,
            target_term,
        )
        if not target:
            raise ValueError(
                f"Academic year {target_year} does not contain term {target_term}."
            )

        target_class = self.school_repository.get_student_class_for_year(
            student_id,
            target["academic_year_id"],
        )
        if not target_class:
            raise ValueError(
                "No class-history record exists for this student in the target academic year."
            )

        history_rows = self.school_repository.get_student_exam_history_before(
            student_id,
            target["start_date"],
        )

        master = self._build_master(
            history_rows,
            target,
            target_class,
        )

        features = self._build_historical_features(master)
        target_features = features.iloc[-1].to_dict()

        if int(target_features["history_term_count"]) < 1:
            raise ValueError(
                "Insufficient prior academic history for next-term prediction."
            )

        overall_model = self.ai_repository.get_active_model_by_key(
            "overall_random_forest_academic_only"
        )
        subject_model = self.ai_repository.get_active_model_by_key(
            "subject_random_forest_academic_only"
        )

        if not overall_model or not subject_model:
            raise ValueError(
                "Required active model versions are not registered in ai_model_versions."
            )

        missing_overall = [
            name
            for name in self.model_service.overall_features
            if name not in target_features
        ]
        if missing_overall:
            raise ValueError(
                "Feature contract mismatch. Missing overall features: "
                + ", ".join(missing_overall)
            )

        overall_input = {
            name: json_safe(target_features.get(name))
            for name in self.model_service.overall_features
        }

        overall_prediction = self.model_service.predict_overall(overall_input)

        reliability_row = self.ai_repository.get_overall_reliability()
        overall_reliability = reliability_row.get("reliability", "HIGH")
        overall_display_policy = reliability_row.get("display_policy", "SHOW")

        overall_prev = json_safe(target_features.get("prev_calculated_average"))

        subject_category_lookup = self._subject_category_lookup(history_rows)
        reliability_map = self.ai_repository.get_subject_reliability_map()

        applicable_subject_rows = (
            self.school_repository.get_student_applicable_predictive_subjects(
                student_id=student_id,
                academic_year_id=target["academic_year_id"],
                grade_id=target_class["grade_id"],
                target_start_date=target["start_date"],
            )
        )
        applicable_subject_keys = {
            row["model_subject_key"]
            for row in applicable_subject_rows
            if row.get("model_subject_key") in PREDICTIVE_SUBJECTS
        }
        applicable_subjects = [
            subject
            for subject in PREDICTIVE_SUBJECTS
            if subject in applicable_subject_keys
        ]
        applicable_subject_sources = {
            row["model_subject_key"]: [
                source
                for source in str(row.get("eligibility_sources") or "").split(",")
                if source
            ]
            for row in applicable_subject_rows
            if row.get("model_subject_key") in applicable_subject_keys
        }

        if not applicable_subjects:
            raise ValueError(
                "No predictive subjects applicable to this student could be resolved "
                "from curriculum, subject enrolments, or prior participation."
            )

        actual_rows = []
        if include_actual_if_available:
            actual_rows = self.school_repository.get_target_actual_results(
                student_id,
                target["academic_term_id"],
            )

        actuals = self._actuals(actual_rows)

        attendance_historical_rows = (
            self.school_repository.get_student_attendance_before(
                student_id,
                target["start_date"],
            )
        )
        attendance_target_rows = []
        if include_actual_if_available:
            attendance_target_rows = (
                self.school_repository.get_student_attendance_for_term(
                    student_id,
                    target["academic_term_id"],
                )
            )

        attendance_context = self._build_attendance_context(
            attendance_historical_rows,
            attendance_target_rows,
        )
        academic_history = self._academic_history(master)

        predictions_to_persist = []

        overall_actual = actuals["overall"]
        overall_error = None
        if overall_actual is not None:
            overall_error = overall_prediction - overall_actual["value"]

        overall_result = {
            "prediction_id": None,
            "model_version_id": int(overall_model["id"]),
            "model_version": overall_model["version"],
            "predicted_average": round(overall_prediction, 4),
            "actual_average": (
                round(overall_actual["value"], 4)
                if overall_actual is not None
                else None
            ),
            "error": round(overall_error, 4) if overall_error is not None else None,
            "absolute_error": (
                round(abs(overall_error), 4)
                if overall_error is not None
                else None
            ),
            "academic_risk": academic_risk(overall_prediction),
            "prediction_reliability": overall_reliability,
            "display_policy": overall_display_policy,
            "performance_trend": trend_label(overall_prediction, overall_prev),
            "feature_count": len(overall_input),
        }

        predictions_to_persist.append(
            {
                "model_version_id": int(overall_model["id"]),
                "student_id": student_id,
                "task_type": "OVERALL",
                "target_subject": None,
                "predicted_value": overall_prediction,
                "academic_risk": overall_result["academic_risk"],
                "prediction_reliability": overall_reliability,
                "display_policy": overall_display_policy,
                "performance_trend": overall_result["performance_trend"],
                "attendance_context": None,
                "model_input_hash": model_input_hash(overall_input),
                "features": overall_input,
                "actual": overall_actual,
            }
        )

        subject_results = []

        for subject in applicable_subjects:
            subject_input = {
                "grade": json_safe(target_features["grade"]),
                "historical_mean_absent_subject_count": json_safe(
                    target_features["historical_mean_absent_subject_count"]
                ),
                "historical_mean_average": json_safe(
                    target_features["historical_mean_average"]
                ),
                "historical_mean_not_taken_subject_count": json_safe(
                    target_features["historical_mean_not_taken_subject_count"]
                ),
                "historical_mean_subject_mark": json_safe(
                    target_features[f"historical_mean_mark_{subject}"]
                ),
                "historical_std_average": json_safe(
                    target_features["historical_std_average"]
                ),
                "historical_subject_mark_count": json_safe(
                    target_features[f"historical_mark_count_{subject}"]
                ),
                "history_term_count": json_safe(
                    target_features["history_term_count"]
                ),
                "performance_trend": json_safe(
                    target_features["performance_trend"]
                ),
                "prev_absent_subject_count": json_safe(
                    target_features["prev_absent_subject_count"]
                ),
                "prev_calculated_average": json_safe(
                    target_features["prev_calculated_average"]
                ),
                "prev_not_taken_subject_count": json_safe(
                    target_features["prev_not_taken_subject_count"]
                ),
                "prev_numeric_subject_count": json_safe(
                    target_features["prev_numeric_subject_count"]
                ),
                "prev_subject_absent": json_safe(
                    target_features[f"prev_absent_{subject}"]
                ),
                "prev_subject_mark": json_safe(
                    target_features[f"prev_mark_{subject}"]
                ),
                "prev_subject_not_taken": json_safe(
                    target_features[f"prev_not_taken_{subject}"]
                ),
                "subject_mark_trend": json_safe(
                    target_features[f"mark_trend_{subject}"]
                ),
                "target_subject": subject_category_lookup[subject],
            }

            prediction = self.model_service.predict_subject(subject_input)

            reliability_item = reliability_map.get(subject)
            if reliability_item:
                reliability = reliability_item["reliability"]
                display_policy = reliability_item["display_policy"]
            else:
                reliability, display_policy = FALLBACK_RELIABILITY[subject]

            actual = actuals["subjects"].get(subject)
            error = None
            if actual is not None:
                error = prediction - actual["value"]

            previous_subject_mark = json_safe(
                target_features[f"prev_mark_{subject}"]
            )

            subject_history = []
            for _, history_row in master.iloc[:-1].iterrows():
                subject_history.append(
                    {
                        "year": int(history_row["year"]),
                        "term": int(history_row["term"]),
                        "academic_period": history_row["academic_period"],
                        "mark": json_safe(history_row.get(f"mark_{subject}")),
                        "status": str(history_row.get(f"status_{subject}") or "NO_RECORD"),
                    }
                )

            expected_change = (
                prediction - float(previous_subject_mark)
                if previous_subject_mark is not None
                else None
            )

            result = {
                "prediction_id": None,
                "subject_key": subject,
                "target_subject": subject_category_lookup[subject],
                "predicted_mark": round(prediction, 4),
                "previous_mark": (
                    round(float(previous_subject_mark), 4)
                    if previous_subject_mark is not None
                    else None
                ),
                "expected_change": (
                    round(float(expected_change), 4)
                    if expected_change is not None
                    else None
                ),
                "subject_standing": subject_standing(
                    prediction,
                    overall_prediction,
                ),
                "history": subject_history,
                "actual_mark": (
                    round(actual["value"], 4)
                    if actual is not None
                    else None
                ),
                "error": round(error, 4) if error is not None else None,
                "absolute_error": (
                    round(abs(error), 4) if error is not None else None
                ),
                "academic_risk": academic_risk(prediction),
                "prediction_reliability": reliability,
                "display_policy": display_policy,
                "performance_trend": trend_label(
                    prediction,
                    previous_subject_mark,
                ),
                "feature_count": len(subject_input),
                "eligibility_sources": applicable_subject_sources.get(subject, []),
            }

            subject_results.append(result)

            predictions_to_persist.append(
                {
                    "model_version_id": int(subject_model["id"]),
                    "student_id": student_id,
                    "task_type": "SUBJECT",
                    "target_subject": subject,
                    "predicted_value": prediction,
                    "academic_risk": result["academic_risk"],
                    "prediction_reliability": reliability,
                    "display_policy": display_policy,
                    "performance_trend": result["performance_trend"],
                    "attendance_context": None,
                    "model_input_hash": model_input_hash(subject_input),
                    "features": subject_input,
                    "actual": actual,
                }
            )

        run_uuid = str(uuid4())
        persistence = {
            "persisted": False,
            "prediction_run_id": None,
        }

        history_periods = [
            {
                "year": int(row["year"]),
                "term": int(row["term"]),
                "academic_period": row["academic_period"],
            }
            for _, row in master.iloc[:-1].iterrows()
        ]

        if persist:
            persisted = self.ai_repository.persist_prediction_bundle(
                run_uuid=run_uuid,
                source=source,
                requested_by_user_id=requested_by_user_id,
                target_period=target["target_period"],
                request_context={
                    "student_id": student_id,
                    "target_year": target_year,
                    "target_term": target_term,
                    "target_period": target["target_period"],
                    "history_periods": history_periods,
                    "feature_builder": "HISTORICAL_ACADEMIC_V1",
                    "overall_feature_count": len(overall_input),
                    "subject_feature_count": (
                        len(self.model_service.subject_numeric_features) + 1
                    ),
                    "applicable_subject_count": len(applicable_subjects),
                    "applicable_subjects": applicable_subjects,
                    "actuals_used_as_features": False,
                    "attendance_used_by_predictor": False,
                },
                predictions=predictions_to_persist,
            )

            persistence = {
                "persisted": True,
                "prediction_run_id": persisted["prediction_run_id"],
            }

            overall_result["prediction_id"] = predictions_to_persist[0].get(
                "prediction_id"
            )

            for index, result in enumerate(subject_results, start=1):
                result["prediction_id"] = predictions_to_persist[index].get(
                    "prediction_id"
                )

        actual_subject_count = sum(
            1 for item in subject_results if item["actual_mark"] is not None
        )

        return {
            "run_uuid": run_uuid,
            "persistence": persistence,
            "student": {
                "id": int(student["id"]),
                "full_name": student["full_name"],
                "student_index": student["student_index"],
                "admission_no": student["admission_no"],
                "status": student["status"],
            },
            "target": {
                "year": target_year,
                "term": target_term,
                "period": target["target_period"],
                "term_name": target["academic_term_name"],
                "grade": target_class["grade_name"],
                "class": target_class["class_name"],
            },
            "history": {
                "period_count": int(target_features["history_term_count"]),
                "periods": history_periods,
                "latest_previous_average": overall_prev,
                "academic_history": academic_history,
            },
            "attendance_context": attendance_context,
            "model_generation": {
                "label": "V1",
                "attendance_used_by_predictor": False,
                "future_note": (
                    "Future model versions can be retrained and re-evaluated as the "
                    "School Portal accumulates more longitudinal academic and attendance data."
                ),
            },
            "models": {
                "overall": {
                    "id": int(overall_model["id"]),
                    "model_key": overall_model["model_key"],
                    "version": overall_model["version"],
                },
                "subject": {
                    "id": int(subject_model["id"]),
                    "model_key": subject_model["model_key"],
                    "version": subject_model["version"],
                },
            },
            "overall": overall_result,
            "subject_selection": {
                "policy": (
                    "core grade curriculum + student subject enrolments + "
                    "target-year prior participation, intersected with V1 predictive subjects"
                ),
                "count": len(applicable_subjects),
                "subject_keys": applicable_subjects,
                "sources": applicable_subject_sources,
                "target_term_results_used_for_selection": False,
            },
            "subjects": subject_results,
            "validation": {
                "actual_target_data_available": overall_actual is not None,
                "actual_subject_marks_available": actual_subject_count,
                "actual_values_were_not_used_as_features": True,
            },
        }
