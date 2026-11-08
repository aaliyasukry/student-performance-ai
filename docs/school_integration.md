# School Portal Integration

## Confirmed Operational Sources

- Student identity: `students.id`
- Historical class and grade: `student_class_histories -> classes -> grades`
- Examination results: `examination_results -> examination_class_subjects -> examinations -> academic_years/academic_terms -> subjects`
- Attendance context: `student_attendances -> attendance_sessions`

## AI Source Views

- `ai_v_student_exam_history`
- `ai_v_student_class_history`
- `ai_v_student_attendance_history`
- `ai_v_subject_mapping`

The final deployed predictor uses academic/history features only. Attendance remains available as contextual information and as research evidence from the feature-set comparison.

## Prediction Contract

`POST /api/v1/predictions/student` accepts a student ID, target year and target term. The service queries only academic periods before the target term, constructs an empty target anchor row, generates the model inputs, predicts, and optionally persists the result.

For validation against a completed term, actual target values may be returned after prediction generation. They are never passed into the feature builder.

## Subject Mapping

Portal subjects are normalized through `ai_subject_mappings`. Religion aliases map to the single model subject `religion`; non-predictive Home Science and PTS remain mapped for source completeness but are excluded from the deployed subject model.

## Source-Missing Provenance

The portal should not introduce a custom examination status named `missing`. A historical source value that was genuinely missing is stored as `not_assessed` with `Source status: missing` in remarks. The AI layer recognizes that provenance marker and restores the model semantic `MISSING`.
