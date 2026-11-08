-- Active production model registrations. Safe to rerun.
SET NAMES utf8mb4;

INSERT INTO ai_model_versions (
    model_key, version, task_type, algorithm, feature_set, artifact_path,
    training_start_period, training_end_period, final_test_period,
    final_test_rows, final_test_mae, final_test_rmse, final_test_r2,
    final_test_within_10_pct, is_active, metadata_json, notes, activated_at
) VALUES
(
    'overall_random_forest_academic_only', '1.0.0', 'OVERALL',
    'RandomForestRegressor', 'ACADEMIC_HISTORY_ONLY',
    'models/final/overall_random_forest_academic_only.joblib',
    '2025_T2', '2026_T1', '2026_T2', 268, 4.2008, 5.5205, 0.9148, 93.2800, 1,
    JSON_OBJECT('source_feature_count',108,'attendance_used',FALSE,'shap_enabled',TRUE),
    'Frozen final overall model selected before 2026 T2 evaluation.', CURRENT_TIMESTAMP
),
(
    'subject_random_forest_academic_only', '1.0.0', 'SUBJECT',
    'RandomForestRegressor', 'ACADEMIC_HISTORY_ONLY',
    'models/final/subject_random_forest_academic_only.joblib',
    '2025_T2', '2026_T1', '2026_T2', 2912, 9.9139, 12.5899, 0.7129, 59.0700, 1,
    JSON_OBJECT('numeric_source_feature_count',17,'categorical_features',JSON_ARRAY('target_subject'),'attendance_used',FALSE,'shap_enabled',TRUE),
    'Frozen final pooled-subject model selected before 2026 T2 evaluation.', CURRENT_TIMESTAMP
)
ON DUPLICATE KEY UPDATE
    artifact_path = VALUES(artifact_path),
    is_active = VALUES(is_active);
