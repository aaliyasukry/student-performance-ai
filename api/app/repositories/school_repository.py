from sqlalchemy import text

from app.database import engine


class SchoolRepository:
    def get_integration_readiness(self):
        sql = text(
            """
            SELECT
                (SELECT COUNT(*) FROM students) AS students,
                (SELECT COUNT(*) FROM student_class_histories) AS student_class_histories,
                (SELECT COUNT(*) FROM examination_results) AS examination_results,
                (SELECT COUNT(*) FROM examination_results er
                 WHERE er.student_id IS NOT NULL AND er.mark IS NOT NULL) AS numeric_examination_results,
                (SELECT COUNT(DISTINCT CONCAT(e.academic_year_id, ':', COALESCE(e.academic_term_id, 0)))
                 FROM examination_results er
                 INNER JOIN examination_class_subjects ecs
                    ON ecs.id = er.examination_class_subject_id
                 INNER JOIN examinations e
                    ON e.id = ecs.examination_id
                 WHERE er.student_id IS NOT NULL AND er.mark IS NOT NULL) AS academic_result_periods,
                (SELECT COUNT(*) FROM examinations WHERE status = 'published') AS published_examinations,
                (SELECT COUNT(*) FROM subjects) AS subjects,
                (SELECT COUNT(*) FROM academic_years) AS academic_years,
                (SELECT COUNT(*) FROM academic_terms) AS academic_terms,
                (SELECT COUNT(*) FROM student_attendances) AS student_attendances,
                (SELECT COUNT(*) FROM attendance_sessions) AS attendance_sessions,
                (SELECT COUNT(*) FROM information_schema.VIEWS
                 WHERE TABLE_SCHEMA = DATABASE()
                   AND TABLE_NAME IN (
                        'ai_v_student_exam_history',
                        'ai_v_student_class_history',
                        'ai_v_student_attendance_history'
                   )) AS ai_source_views
            """
        )

        with engine.connect() as connection:
            row = connection.execute(sql).mappings().one()

        result = dict(row)

        result["schema_ready"] = (
            int(result["ai_source_views"]) == 3
            and int(result["subjects"]) > 0
            and int(result["academic_years"]) > 0
            and int(result["academic_terms"]) > 0
        )

        result["prediction_data_ready"] = (
            int(result["students"]) > 0
            and int(result["numeric_examination_results"]) > 0
            and int(result["student_class_histories"]) > 0
            and int(result["academic_result_periods"]) >= 2
        )

        if result["prediction_data_ready"]:
            result["next_action"] = (
                "Live historical data exists. Prediction API can be tested."
            )
        elif result["schema_ready"]:
            result["next_action"] = (
                "Schema integration is ready. Import students and historical "
                "examination results before prediction."
            )
        else:
            result["next_action"] = (
                "Complete required school master-data/schema setup before prediction."
            )

        return result

    def get_source_contract(self):
        return {
            "student_identity": {
                "table": "students",
                "primary_key": "students.id",
                "ai_reference": "ai_predictions.student_id",
                "used_as_model_feature": False,
            },
            "historical_class": {
                "table": "student_class_histories",
                "joins": ["academic_years", "classes", "grades"],
            },
            "academic_results": {
                "tables": [
                    "examination_results",
                    "examination_class_subjects",
                    "examinations",
                    "academic_years",
                    "academic_terms",
                    "subjects",
                    "classes",
                    "grades",
                ],
                "view": "ai_v_student_exam_history",
            },
            "attendance_context": {
                "tables": [
                    "student_attendances",
                    "attendance_sessions",
                    "school_calendar_days",
                    "academic_terms",
                    "academic_years",
                ],
                "view": "ai_v_student_attendance_history",
                "used_by_final_predictor": False,
            },
        }

    def get_student(self, student_id: int):
        sql = text(
            """
            SELECT
                id,
                full_name,
                student_index,
                admission_no,
                status
            FROM students
            WHERE id = :student_id
            LIMIT 1
            """
        )

        with engine.connect() as connection:
            row = connection.execute(
                sql,
                {"student_id": student_id},
            ).mappings().first()

        return dict(row) if row else None

    def get_target_term(self, target_year: int, target_term: int):
        sql = text(
            """
            SELECT
                ay.id AS academic_year_id,
                ay.name AS academic_year_name,
                at.id AS academic_term_id,
                at.name AS academic_term_name,
                at.start_date,
                at.end_date
            FROM academic_years ay
            INNER JOIN academic_terms at
                ON at.academic_year_id = ay.id
            WHERE CAST(ay.name AS UNSIGNED) = :target_year
            ORDER BY at.start_date, at.id
            """
        )

        with engine.connect() as connection:
            rows = connection.execute(
                sql,
                {"target_year": target_year},
            ).mappings().all()

        if target_term < 1 or target_term > len(rows):
            return None

        row = dict(rows[target_term - 1])
        row["term_number"] = target_term
        row["target_period"] = f"{target_year}_T{target_term}"
        return row

    def get_student_class_for_year(self, student_id: int, academic_year_id: int):
        sql = text(
            """
            SELECT
                sch.student_id,
                sch.academic_year_id,
                sch.class_id,
                c.name AS class_name,
                c.grade_id,
                g.name AS grade_name,
                g.code AS grade_code,
                sch.is_current
            FROM student_class_histories sch
            INNER JOIN classes c
                ON c.id = sch.class_id
            INNER JOIN grades g
                ON g.id = c.grade_id
            WHERE
                sch.student_id = :student_id
                AND sch.academic_year_id = :academic_year_id
            ORDER BY sch.is_current DESC, sch.id DESC
            LIMIT 1
            """
        )

        with engine.connect() as connection:
            row = connection.execute(
                sql,
                {
                    "student_id": student_id,
                    "academic_year_id": academic_year_id,
                },
            ).mappings().first()

        return dict(row) if row else None

    def get_student_applicable_predictive_subjects(
        self,
        *,
        student_id: int,
        academic_year_id: int,
        grade_id: int,
        target_start_date,
    ):
        """Return only predictive subjects that are applicable to this student."""
        sql = text(
            """
            WITH curriculum_core AS (
                SELECT
                    asm.model_subject_key,
                    asm.model_subject_name,
                    'CURRICULUM_CORE' AS eligibility_source
                FROM grade_curricula gc
                INNER JOIN grade_curriculum_subjects gcs
                    ON gcs.grade_curriculum_id = gc.id
                LEFT JOIN subject_basket_subjects sbs
                    ON sbs.grade_curriculum_subject_id = gcs.id
                INNER JOIN ai_subject_mappings asm
                    ON asm.subject_id = gcs.subject_id
                WHERE
                    gc.academic_year_id = :academic_year_id
                    AND gc.grade_id = :grade_id
                    AND gcs.is_active = 1
                    AND sbs.grade_curriculum_subject_id IS NULL
                    AND asm.is_predictive = 1
            ),
            selected_subjects AS (
                SELECT
                    asm.model_subject_key,
                    asm.model_subject_name,
                    'STUDENT_ENROLMENT' AS eligibility_source
                FROM student_subject_enrolments sse
                INNER JOIN ai_subject_mappings asm
                    ON asm.subject_id = sse.subject_id
                WHERE
                    sse.student_id = :student_id
                    AND sse.academic_year_id = :academic_year_id
                    AND asm.is_predictive = 1
            ),
            target_year_history AS (
                SELECT DISTINCT
                    asm.model_subject_key,
                    asm.model_subject_name,
                    'TARGET_YEAR_HISTORY' AS eligibility_source
                FROM ai_v_student_exam_history v
                INNER JOIN ai_subject_mappings asm
                    ON asm.subject_id = v.subject_id
                WHERE
                    v.student_id = :student_id
                    AND v.academic_year_id = :academic_year_id
                    AND v.term_start_date < :target_start_date
                    AND v.examination_status = 'published'
                    AND asm.is_predictive = 1
                    AND (
                        v.mark IS NOT NULL
                        OR LOWER(COALESCE(v.attendance_status, '')) IN ('present', 'absent')
                    )
            ),
            eligible AS (
                SELECT * FROM curriculum_core
                UNION ALL
                SELECT * FROM selected_subjects
                UNION ALL
                SELECT * FROM target_year_history
            )
            SELECT
                model_subject_key,
                MAX(model_subject_name) AS model_subject_name,
                GROUP_CONCAT(
                    DISTINCT eligibility_source
                    ORDER BY eligibility_source
                    SEPARATOR ','
                ) AS eligibility_sources
            FROM eligible
            GROUP BY model_subject_key
            ORDER BY model_subject_name
            """
        )

        params = {
            "student_id": student_id,
            "academic_year_id": academic_year_id,
            "grade_id": grade_id,
            "target_start_date": target_start_date,
        }

        with engine.connect() as connection:
            rows = connection.execute(sql, params).mappings().all()

        if rows:
            return [dict(row) for row in rows]

        # Fallback for historical imports: keep only subjects the student took before the target term.
        fallback_sql = text(
            """
            SELECT DISTINCT
                asm.model_subject_key,
                asm.model_subject_name,
                'HISTORICAL_PARTICIPATION_FALLBACK' AS eligibility_sources
            FROM ai_v_student_exam_history v
            INNER JOIN ai_subject_mappings asm
                ON asm.subject_id = v.subject_id
            WHERE
                v.student_id = :student_id
                AND v.term_start_date < :target_start_date
                AND v.examination_status = 'published'
                AND asm.is_predictive = 1
                AND (
                    v.mark IS NOT NULL
                    OR LOWER(COALESCE(v.attendance_status, '')) IN ('present', 'absent')
                )
            ORDER BY asm.model_subject_name
            """
        )

        with engine.connect() as connection:
            rows = connection.execute(
                fallback_sql,
                {
                    "student_id": student_id,
                    "target_start_date": target_start_date,
                },
            ).mappings().all()

        return [dict(row) for row in rows]

    def get_student_exam_history_before(self, student_id: int, target_start_date):
        sql = text(
            """
            SELECT
                v.examination_result_id,
                v.student_id,
                v.examination_id,
                v.academic_year_id,
                v.academic_year_name,
                v.academic_term_id,
                v.academic_term_name,
                v.term_start_date,
                v.term_end_date,
                v.examination_name,
                v.examination_status,
                v.class_id,
                v.class_name,
                v.grade_id,
                v.grade_name,
                v.grade_code,
                v.subject_id,
                v.subject_name AS portal_subject_name,
                v.subject_code AS portal_subject_code,
                asm.model_subject_key,
                asm.model_subject_name,
                asm.is_predictive,
                asm.mapping_role,
                v.attendance_status,
                v.mark,
                v.result_remarks,
                1 + (
                    SELECT COUNT(*)
                    FROM academic_terms at2
                    WHERE
                        at2.academic_year_id = v.academic_year_id
                        AND at2.start_date < v.term_start_date
                ) AS term_number
            FROM ai_v_student_exam_history v
            INNER JOIN ai_subject_mappings asm
                ON asm.subject_id = v.subject_id
            WHERE
                v.student_id = :student_id
                AND v.term_start_date < :target_start_date
                AND v.examination_status = 'published'
            ORDER BY
                v.term_start_date,
                v.examination_id,
                asm.model_subject_key
            """
        )

        with engine.connect() as connection:
            rows = connection.execute(
                sql,
                {
                    "student_id": student_id,
                    "target_start_date": target_start_date,
                },
            ).mappings().all()

        return [dict(row) for row in rows]

    def get_target_actual_results(self, student_id: int, academic_term_id: int):
        sql = text(
            """
            SELECT
                v.examination_result_id,
                v.student_id,
                v.examination_id,
                v.academic_year_id,
                v.academic_year_name,
                v.academic_term_id,
                v.academic_term_name,
                v.class_id,
                v.class_name,
                v.grade_id,
                v.grade_name,
                v.subject_id,
                v.subject_name AS portal_subject_name,
                v.subject_code AS portal_subject_code,
                asm.model_subject_key,
                asm.model_subject_name,
                asm.is_predictive,
                v.attendance_status,
                v.mark,
                v.result_remarks
            FROM ai_v_student_exam_history v
            INNER JOIN ai_subject_mappings asm
                ON asm.subject_id = v.subject_id
            WHERE
                v.student_id = :student_id
                AND v.academic_term_id = :academic_term_id
                AND v.examination_status = 'published'
            ORDER BY
                v.examination_id,
                asm.model_subject_key
            """
        )

        with engine.connect() as connection:
            rows = connection.execute(
                sql,
                {
                    "student_id": student_id,
                    "academic_term_id": academic_term_id,
                },
            ).mappings().all()

        return [dict(row) for row in rows]

    def get_cohort_options(self, target_year: int):
        sql = text(
            """
            WITH latest_history AS (
                SELECT
                    sch.student_id,
                    sch.academic_year_id,
                    sch.class_id,
                    ROW_NUMBER() OVER (
                        PARTITION BY sch.student_id, sch.academic_year_id
                        ORDER BY sch.is_current DESC, sch.id DESC
                    ) AS row_num
                FROM student_class_histories sch
                INNER JOIN academic_years ay
                    ON ay.id = sch.academic_year_id
                WHERE CAST(ay.name AS UNSIGNED) = :target_year
            )
            SELECT
                ay.id AS academic_year_id,
                ay.name AS academic_year_name,
                g.id AS grade_id,
                g.name AS grade_name,
                g.code AS grade_code,
                c.id AS class_id,
                c.name AS class_name,
                c.code AS class_code,
                COUNT(DISTINCT lh.student_id) AS student_count
            FROM latest_history lh
            INNER JOIN academic_years ay ON ay.id = lh.academic_year_id
            INNER JOIN classes c ON c.id = lh.class_id
            INNER JOIN grades g ON g.id = c.grade_id
            WHERE
                lh.row_num = 1
                AND CAST(REGEXP_SUBSTR(g.name, '[0-9]+') AS UNSIGNED) BETWEEN 6 AND 10
            GROUP BY
                ay.id, ay.name,
                g.id, g.name, g.code,
                c.id, c.name, c.code
            ORDER BY
                CAST(REGEXP_SUBSTR(g.name, '[0-9]+') AS UNSIGNED),
                c.name
            """
        )

        with engine.connect() as connection:
            rows = connection.execute(
                sql,
                {"target_year": target_year},
            ).mappings().all()

        grades = {}
        for row in rows:
            item = dict(row)
            grade_id = int(item["grade_id"])
            grade = grades.setdefault(
                grade_id,
                {
                    "grade_id": grade_id,
                    "grade_name": item["grade_name"],
                    "grade_code": item["grade_code"],
                    "student_count": 0,
                    "classes": [],
                },
            )
            count = int(item["student_count"] or 0)
            grade["student_count"] += count
            grade["classes"].append(
                {
                    "class_id": int(item["class_id"]),
                    "class_name": item["class_name"],
                    "class_code": item["class_code"],
                    "student_count": count,
                }
            )

        return {
            "target_year": target_year,
            "grades": list(grades.values()),
        }

    def get_cohort_students(
        self,
        *,
        target_year: int,
        grade_id: int,
        class_id: int | None = None,
    ):
        sql = text(
            """
            WITH latest_history AS (
                SELECT
                    sch.student_id,
                    sch.academic_year_id,
                    sch.class_id,
                    ROW_NUMBER() OVER (
                        PARTITION BY sch.student_id, sch.academic_year_id
                        ORDER BY sch.is_current DESC, sch.id DESC
                    ) AS row_num
                FROM student_class_histories sch
                INNER JOIN academic_years ay
                    ON ay.id = sch.academic_year_id
                WHERE CAST(ay.name AS UNSIGNED) = :target_year
            )
            SELECT
                s.id,
                s.full_name,
                s.student_index,
                s.admission_no,
                s.status,
                g.id AS grade_id,
                g.name AS grade_name,
                c.id AS class_id,
                c.name AS class_name
            FROM latest_history lh
            INNER JOIN students s ON s.id = lh.student_id
            INNER JOIN classes c ON c.id = lh.class_id
            INNER JOIN grades g ON g.id = c.grade_id
            WHERE
                lh.row_num = 1
                AND g.id = :grade_id
                AND (:class_id IS NULL OR c.id = :class_id)
            ORDER BY c.name, s.full_name, s.id
            """
        )

        with engine.connect() as connection:
            rows = connection.execute(
                sql,
                {
                    "target_year": target_year,
                    "grade_id": grade_id,
                    "class_id": class_id,
                },
            ).mappings().all()

        return [dict(row) for row in rows]

    def get_student_attendance_before(self, student_id: int, target_start_date):
        sql = text(
            """
            SELECT
                v.student_attendance_id,
                v.student_id,
                v.attendance_session_id,
                v.attendance_date,
                v.attendance_session_status,
                v.attendance_status,
                v.attendance_remarks,
                v.class_id,
                v.class_name,
                v.grade_id,
                v.grade_name,
                v.academic_term_id,
                v.academic_term_name,
                v.academic_year_id,
                v.academic_year_name,
                v.is_working_day,
                1 + (
                    SELECT COUNT(*)
                    FROM academic_terms at2
                    WHERE
                        at2.academic_year_id = v.academic_year_id
                        AND at2.start_date < current_term.start_date
                ) AS term_number
            FROM ai_v_student_attendance_history v
            INNER JOIN academic_terms current_term
                ON current_term.id = v.academic_term_id
            WHERE
                v.student_id = :student_id
                AND v.attendance_date < :target_start_date
                AND v.academic_term_id IS NOT NULL
                AND v.attendance_session_status IN ('completed', 'locked')
                AND (v.is_working_day = 1 OR v.is_working_day IS NULL)
            ORDER BY v.attendance_date, v.attendance_session_id
            """
        )

        with engine.connect() as connection:
            rows = connection.execute(
                sql,
                {
                    "student_id": student_id,
                    "target_start_date": target_start_date,
                },
            ).mappings().all()

        return [dict(row) for row in rows]

    def get_student_attendance_for_term(
        self,
        student_id: int,
        academic_term_id: int,
    ):
        sql = text(
            """
            SELECT
                v.student_attendance_id,
                v.student_id,
                v.attendance_session_id,
                v.attendance_date,
                v.attendance_session_status,
                v.attendance_status,
                v.attendance_remarks,
                v.class_id,
                v.class_name,
                v.grade_id,
                v.grade_name,
                v.academic_term_id,
                v.academic_term_name,
                v.academic_year_id,
                v.academic_year_name,
                v.is_working_day,
                1 + (
                    SELECT COUNT(*)
                    FROM academic_terms at2
                    WHERE
                        at2.academic_year_id = v.academic_year_id
                        AND at2.start_date < current_term.start_date
                ) AS term_number
            FROM ai_v_student_attendance_history v
            INNER JOIN academic_terms current_term
                ON current_term.id = v.academic_term_id
            WHERE
                v.student_id = :student_id
                AND v.academic_term_id = :academic_term_id
                AND v.attendance_session_status IN ('completed', 'locked')
                AND (v.is_working_day = 1 OR v.is_working_day IS NULL)
            ORDER BY v.attendance_date, v.attendance_session_id
            """
        )

        with engine.connect() as connection:
            rows = connection.execute(
                sql,
                {
                    "student_id": student_id,
                    "academic_term_id": academic_term_id,
                },
            ).mappings().all()

        return [dict(row) for row in rows]

