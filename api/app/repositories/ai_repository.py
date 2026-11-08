import json
from typing import Any

from sqlalchemy import text

from app.database import engine


class AIRepository:
    def get_active_models(self):
        sql = text(
            """
            SELECT
                id,
                model_key,
                version,
                task_type,
                algorithm,
                feature_set,
                final_test_rows,
                final_test_mae,
                final_test_rmse,
                final_test_r2,
                final_test_within_10_pct,
                is_active,
                activated_at
            FROM ai_model_versions
            WHERE is_active = 1
            ORDER BY task_type, id
            """
        )

        with engine.connect() as connection:
            rows = connection.execute(sql).mappings().all()

        return [dict(row) for row in rows]

    def get_active_model_by_key(self, model_key: str):
        sql = text(
            """
            SELECT *
            FROM ai_model_versions
            WHERE model_key = :model_key
              AND is_active = 1
            ORDER BY id DESC
            LIMIT 1
            """
        )

        with engine.connect() as connection:
            row = connection.execute(
                sql,
                {"model_key": model_key},
            ).mappings().first()

        return dict(row) if row else None

    def get_subject_reliability(self):
        sql = text(
            """
            SELECT
                m.scope_value AS subject,
                m.row_count,
                m.mae,
                m.rmse,
                m.r2,
                m.within_10_pct,
                m.reliability_label AS reliability,
                m.display_policy
            FROM ai_model_metrics m
            INNER JOIN ai_model_versions mv
                ON mv.id = m.model_version_id
            WHERE
                mv.model_key = 'subject_random_forest_academic_only'
                AND mv.is_active = 1
                AND m.metric_scope = 'SUBJECT'
                AND m.evaluation_period = '2026_T2'
            ORDER BY
                FIELD(
                    m.reliability_label,
                    'HIGH',
                    'MODERATE',
                    'LOW',
                    'INSUFFICIENT_DATA'
                ),
                m.mae
            """
        )

        with engine.connect() as connection:
            rows = connection.execute(sql).mappings().all()

        return [dict(row) for row in rows]

    def get_subject_reliability_map(self):
        return {
            row["subject"]: {
                "reliability": row["reliability"],
                "display_policy": row["display_policy"],
            }
            for row in self.get_subject_reliability()
        }

    def get_overall_reliability(self):
        sql = text(
            """
            SELECT
                mv.model_key,
                mv.version,
                m.row_count,
                m.mae,
                m.rmse,
                m.r2,
                m.within_10_pct,
                m.reliability_label AS reliability,
                m.display_policy
            FROM ai_model_metrics m
            INNER JOIN ai_model_versions mv
                ON mv.id = m.model_version_id
            WHERE
                mv.model_key = 'overall_random_forest_academic_only'
                AND mv.is_active = 1
                AND m.metric_scope = 'OVERALL'
                AND m.evaluation_period = '2026_T2'
            LIMIT 1
            """
        )

        with engine.connect() as connection:
            row = connection.execute(sql).mappings().first()

        return dict(row) if row else {}

    def persist_prediction_bundle(
        self,
        *,
        run_uuid: str,
        source: str,
        requested_by_user_id: int | None,
        target_period: str,
        request_context: dict[str, Any],
        predictions: list[dict[str, Any]],
    ):
        run_insert = text(
            """
            INSERT INTO ai_prediction_runs (
                run_uuid,
                source,
                status,
                requested_by_user_id,
                target_period,
                total_students,
                total_predictions,
                request_context_json,
                started_at,
                completed_at
            ) VALUES (
                :run_uuid,
                :source,
                'COMPLETED',
                :requested_by_user_id,
                :target_period,
                1,
                :total_predictions,
                :request_context_json,
                CURRENT_TIMESTAMP,
                CURRENT_TIMESTAMP
            )
            """
        )

        prediction_insert = text(
            """
            INSERT INTO ai_predictions (
                prediction_run_id,
                model_version_id,
                student_id,
                target_period,
                task_type,
                target_subject,
                predicted_value,
                academic_risk,
                prediction_reliability,
                display_policy,
                performance_trend,
                attendance_context,
                model_input_hash
            ) VALUES (
                :prediction_run_id,
                :model_version_id,
                :student_id,
                :target_period,
                :task_type,
                :target_subject,
                :predicted_value,
                :academic_risk,
                :prediction_reliability,
                :display_policy,
                :performance_trend,
                :attendance_context,
                :model_input_hash
            )
            """
        )

        feature_insert = text(
            """
            INSERT INTO ai_prediction_features (
                prediction_id,
                feature_schema_version,
                features_json
            ) VALUES (
                :prediction_id,
                '1.0',
                :features_json
            )
            """
        )

        outcome_insert = text(
            """
            INSERT INTO ai_actual_outcomes (
                prediction_id,
                actual_value,
                absolute_error,
                squared_error,
                outcome_period,
                source_record_id
            ) VALUES (
                :prediction_id,
                :actual_value,
                :absolute_error,
                :squared_error,
                :outcome_period,
                :source_record_id
            )
            """
        )

        persisted = {
            "prediction_run_id": None,
            "prediction_ids": [],
        }

        with engine.begin() as connection:
            run_result = connection.execute(
                run_insert,
                {
                    "run_uuid": run_uuid,
                    "source": source,
                    "requested_by_user_id": requested_by_user_id,
                    "target_period": target_period,
                    "total_predictions": len(predictions),
                    "request_context_json": json.dumps(
                        request_context,
                        ensure_ascii=False,
                        allow_nan=False,
                    ),
                },
            )

            prediction_run_id = int(run_result.lastrowid)
            persisted["prediction_run_id"] = prediction_run_id

            for item in predictions:
                pred_result = connection.execute(
                    prediction_insert,
                    {
                        "prediction_run_id": prediction_run_id,
                        "model_version_id": item["model_version_id"],
                        "student_id": item["student_id"],
                        "target_period": target_period,
                        "task_type": item["task_type"],
                        "target_subject": item.get("target_subject"),
                        "predicted_value": item["predicted_value"],
                        "academic_risk": item.get("academic_risk"),
                        "prediction_reliability": item.get(
                            "prediction_reliability"
                        ),
                        "display_policy": item.get("display_policy"),
                        "performance_trend": item.get("performance_trend"),
                        "attendance_context": item.get("attendance_context"),
                        "model_input_hash": item.get("model_input_hash"),
                    },
                )

                prediction_id = int(pred_result.lastrowid)
                persisted["prediction_ids"].append(prediction_id)
                item["prediction_id"] = prediction_id

                connection.execute(
                    feature_insert,
                    {
                        "prediction_id": prediction_id,
                        "features_json": json.dumps(
                            item["features"],
                            ensure_ascii=False,
                            allow_nan=False,
                        ),
                    },
                )

                actual = item.get("actual")
                if actual is not None:
                    error = float(item["predicted_value"]) - float(actual["value"])
                    connection.execute(
                        outcome_insert,
                        {
                            "prediction_id": prediction_id,
                            "actual_value": actual["value"],
                            "absolute_error": abs(error),
                            "squared_error": error ** 2,
                            "outcome_period": target_period,
                            "source_record_id": actual.get("source_record_id"),
                        },
                    )

        return persisted

    def get_prediction_summary(self):
        summary_sql = text(
            """
            SELECT
                (SELECT COUNT(*) FROM ai_prediction_runs) AS total_runs,
                (SELECT COUNT(*) FROM ai_prediction_runs WHERE status = 'COMPLETED') AS completed_runs,
                (SELECT COUNT(*) FROM ai_predictions) AS total_predictions,
                (SELECT COUNT(DISTINCT student_id) FROM ai_predictions) AS distinct_students_predicted,
                (SELECT MAX(completed_at) FROM ai_prediction_runs WHERE status = 'COMPLETED') AS latest_completed_at
            """
        )

        risk_sql = text(
            """
            SELECT
                academic_risk,
                COUNT(*) AS count
            FROM ai_predictions
            WHERE task_type = 'OVERALL'
            GROUP BY academic_risk
            """
        )

        recent_sql = text(
            """
            SELECT
                id,
                run_uuid,
                source,
                status,
                target_period,
                total_students,
                total_predictions,
                requested_by_user_id,
                started_at,
                completed_at
            FROM ai_prediction_runs
            ORDER BY id DESC
            LIMIT 8
            """
        )

        with engine.connect() as connection:
            summary = dict(connection.execute(summary_sql).mappings().one())
            risks = connection.execute(risk_sql).mappings().all()
            recent = connection.execute(recent_sql).mappings().all()

        return {
            **summary,
            "overall_risk_distribution": {
                str(row["academic_risk"] or "UNKNOWN"): int(row["count"])
                for row in risks
            },
            "recent_runs": [dict(row) for row in recent],
        }

