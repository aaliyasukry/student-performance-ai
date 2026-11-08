-- Read-only views used by the AI service
SET NAMES utf8mb4;

CREATE OR REPLACE VIEW ai_v_student_exam_history AS
SELECT
    er.id AS examination_result_id,
    er.student_id,
    e.id AS examination_id,
    e.academic_year_id,
    ay.name AS academic_year_name,
    e.academic_term_id,
    at.name AS academic_term_name,
    at.start_date AS term_start_date,
    at.end_date AS term_end_date,
    e.name AS examination_name,
    e.type AS examination_type,
    e.status AS examination_status,
    e.start_date AS examination_start_date,
    e.end_date AS examination_end_date,
    e.published_at,
    ecs.id AS examination_class_subject_id,
    ecs.class_id,
    c.name AS class_name,
    c.grade_id,
    g.name AS grade_name,
    g.code AS grade_code,
    ecs.subject_id,
    COALESCE(s.name, ecs.subject_name) AS subject_name,
    COALESCE(s.code, ecs.subject_code) AS subject_code,
    ecs.max_marks,
    ecs.pass_mark,
    ecs.status AS result_sheet_status,
    er.attendance_status,
    er.mark,
    er.grade AS result_grade,
    er.remarks AS result_remarks,
    er.created_at AS result_created_at,
    er.updated_at AS result_updated_at
FROM examination_results er
INNER JOIN examination_class_subjects ecs ON ecs.id = er.examination_class_subject_id
INNER JOIN examinations e ON e.id = ecs.examination_id
INNER JOIN academic_years ay ON ay.id = e.academic_year_id
LEFT JOIN academic_terms at ON at.id = e.academic_term_id
LEFT JOIN subjects s ON s.id = ecs.subject_id
LEFT JOIN classes c ON c.id = ecs.class_id
LEFT JOIN grades g ON g.id = c.grade_id
WHERE er.student_id IS NOT NULL;

CREATE OR REPLACE VIEW ai_v_student_class_history AS
SELECT
    sch.id AS student_class_history_id,
    sch.student_id,
    sch.academic_year_id,
    ay.name AS academic_year_name,
    sch.class_id,
    c.name AS class_name,
    c.code AS class_code,
    c.grade_id,
    g.name AS grade_name,
    g.code AS grade_code,
    sch.is_current,
    sch.created_at,
    sch.updated_at
FROM student_class_histories sch
INNER JOIN academic_years ay ON ay.id = sch.academic_year_id
INNER JOIN classes c ON c.id = sch.class_id
INNER JOIN grades g ON g.id = c.grade_id;

CREATE OR REPLACE VIEW ai_v_student_attendance_history AS
SELECT
    sa.id AS student_attendance_id,
    sa.student_id,
    ats.id AS attendance_session_id,
    ats.attendance_date,
    ats.status AS attendance_session_status,
    sa.status AS attendance_status,
    sa.remarks AS attendance_remarks,
    ats.class_id,
    c.name AS class_name,
    c.grade_id,
    g.name AS grade_name,
    g.code AS grade_code,
    at.id AS academic_term_id,
    at.name AS academic_term_name,
    ay.id AS academic_year_id,
    ay.name AS academic_year_name,
    scd.is_working_day
FROM student_attendances sa
INNER JOIN attendance_sessions ats ON ats.id = sa.attendance_session_id
INNER JOIN classes c ON c.id = ats.class_id
INNER JOIN grades g ON g.id = c.grade_id
LEFT JOIN academic_terms at ON ats.attendance_date BETWEEN at.start_date AND at.end_date
LEFT JOIN academic_years ay ON ay.id = at.academic_year_id
LEFT JOIN school_calendar_days scd ON scd.calendar_date = ats.attendance_date;

SELECT TABLE_NAME
FROM information_schema.VIEWS
WHERE TABLE_SCHEMA = DATABASE()
  AND TABLE_NAME LIKE 'ai_v_%'
ORDER BY TABLE_NAME;