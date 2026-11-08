-- ============================================================
-- STUDENT PERFORMANCE AI
-- AI PERSISTENCE SCHEMA
-- MySQL 8.x recommended
--
-- PURPOSE
--   Store AI model lifecycle, prediction audit, feature snapshots,
--   SHAP explanations, actual outcomes and evaluation metrics.
--
-- IMPORTANT
--   Existing school tables remain the source of truth for:
--     - students
--     - marks
--     - attendance
--     - grades/classes/subjects
--
--   This schema DOES NOT duplicate the full operational school DB.
--
--   `student_id` below intentionally has no foreign key yet because
--   the exact production school-table name/key must be confirmed
--   against the operational students table at application level.
-- ============================================================

SET NAMES utf8mb4;
SET FOREIGN_KEY_CHECKS = 0;

-- ------------------------------------------------------------
-- 1. MODEL VERSIONS
-- One row per deployable model version.
-- ------------------------------------------------------------

CREATE TABLE IF NOT EXISTS ai_model_versions (
    id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
    model_key VARCHAR(100) NOT NULL,
    version VARCHAR(30) NOT NULL,
    task_type VARCHAR(30) NOT NULL,
    algorithm VARCHAR(100) NOT NULL,
    feature_set VARCHAR(100) NOT NULL,

    artifact_path VARCHAR(500) NOT NULL,
    artifact_sha256 CHAR(64) NULL,

    training_start_period VARCHAR(20) NULL,
    training_end_period VARCHAR(20) NULL,
    final_test_period VARCHAR(20) NULL,

    final_test_rows INT UNSIGNED NULL,
    final_test_mae DECIMAL(10,4) NULL,
    final_test_rmse DECIMAL(10,4) NULL,
    final_test_r2 DECIMAL(10,4) NULL,
    final_test_within_10_pct DECIMAL(10,4) NULL,

    is_active TINYINT(1) NOT NULL DEFAULT 0,

    metadata_json JSON NULL,
    notes TEXT NULL,

    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    activated_at TIMESTAMP NULL DEFAULT NULL,
    retired_at TIMESTAMP NULL DEFAULT NULL,

    PRIMARY KEY (id),
    UNIQUE KEY uq_ai_model_versions_key_version (model_key, version),
    KEY idx_ai_model_versions_active (task_type, is_active),

    CONSTRAINT chk_ai_model_versions_task
        CHECK (task_type IN ('OVERALL', 'SUBJECT'))
) ENGINE=InnoDB
  DEFAULT CHARSET=utf8mb4
  COLLATE=utf8mb4_unicode_ci;


-- ------------------------------------------------------------
-- 2. TRAINING / RETRAINING RUNS
-- Records how each model version was produced.
-- ------------------------------------------------------------

CREATE TABLE IF NOT EXISTS ai_training_runs (
    id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
    run_uuid CHAR(36) NOT NULL,

    model_version_id BIGINT UNSIGNED NULL,

    run_type VARCHAR(30) NOT NULL DEFAULT 'INITIAL_RESEARCH',
    status VARCHAR(30) NOT NULL DEFAULT 'COMPLETED',

    data_start_period VARCHAR(20) NULL,
    data_end_period VARCHAR(20) NULL,

    training_rows INT UNSIGNED NULL,
    validation_rows INT UNSIGNED NULL,
    final_test_rows INT UNSIGNED NULL,

    configuration_json JSON NULL,
    metrics_json JSON NULL,

    started_at TIMESTAMP NULL DEFAULT NULL,
    completed_at TIMESTAMP NULL DEFAULT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,

    PRIMARY KEY (id),
    UNIQUE KEY uq_ai_training_runs_uuid (run_uuid),
    KEY idx_ai_training_runs_model (model_version_id),

    CONSTRAINT fk_ai_training_runs_model
        FOREIGN KEY (model_version_id)
        REFERENCES ai_model_versions(id)
        ON DELETE SET NULL
) ENGINE=InnoDB
  DEFAULT CHARSET=utf8mb4
  COLLATE=utf8mb4_unicode_ci;


-- ------------------------------------------------------------
-- 3. MODEL METRICS
-- Stores overall, grade-level and subject-level evaluation evidence.
-- This table powers UI prediction-reliability rules.
-- ------------------------------------------------------------

CREATE TABLE IF NOT EXISTS ai_model_metrics (
    id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,

    model_version_id BIGINT UNSIGNED NOT NULL,

    evaluation_period VARCHAR(20) NOT NULL,
    metric_scope VARCHAR(30) NOT NULL,
    scope_value VARCHAR(100) NULL,

    row_count INT UNSIGNED NOT NULL,

    mae DECIMAL(10,4) NULL,
    rmse DECIMAL(10,4) NULL,
    r2 DECIMAL(10,4) NULL,

    within_5_pct DECIMAL(10,4) NULL,
    within_10_pct DECIMAL(10,4) NULL,
    within_15_pct DECIMAL(10,4) NULL,

    reliability_label VARCHAR(30) NULL,
    display_policy VARCHAR(40) NULL,

    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,

    PRIMARY KEY (id),
    KEY idx_ai_model_metrics_model (model_version_id),
    KEY idx_ai_model_metrics_scope (
        model_version_id,
        metric_scope,
        scope_value
    ),

    CONSTRAINT fk_ai_model_metrics_model
        FOREIGN KEY (model_version_id)
        REFERENCES ai_model_versions(id)
        ON DELETE CASCADE,

    CONSTRAINT chk_ai_model_metrics_scope
        CHECK (
            metric_scope IN (
                'OVERALL',
                'GRADE',
                'SUBJECT',
                'SUBJECT_GRADE'
            )
        )
) ENGINE=InnoDB
  DEFAULT CHARSET=utf8mb4
  COLLATE=utf8mb4_unicode_ci;


-- ------------------------------------------------------------
-- 4. PREDICTION RUNS
-- One API/batch request may produce multiple student predictions.
-- ------------------------------------------------------------

CREATE TABLE IF NOT EXISTS ai_prediction_runs (
    id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
    run_uuid CHAR(36) NOT NULL,

    source VARCHAR(30) NOT NULL DEFAULT 'API',
    status VARCHAR(30) NOT NULL DEFAULT 'STARTED',

    requested_by_user_id BIGINT UNSIGNED NULL,

    target_period VARCHAR(20) NOT NULL,

    total_students INT UNSIGNED NOT NULL DEFAULT 0,
    total_predictions INT UNSIGNED NOT NULL DEFAULT 0,

    request_context_json JSON NULL,

    started_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP NULL DEFAULT NULL,

    PRIMARY KEY (id),
    UNIQUE KEY uq_ai_prediction_runs_uuid (run_uuid),
    KEY idx_ai_prediction_runs_period (target_period),
    KEY idx_ai_prediction_runs_status (status),

    CONSTRAINT chk_ai_prediction_runs_source
        CHECK (
            source IN (
                'STANDALONE_UI',
                'LARAVEL',
                'API',
                'BATCH',
                'RESEARCH'
            )
        )
) ENGINE=InnoDB
  DEFAULT CHARSET=utf8mb4
  COLLATE=utf8mb4_unicode_ci;


-- ------------------------------------------------------------
-- 5. PREDICTIONS
-- One row per overall or subject prediction.
-- ------------------------------------------------------------

CREATE TABLE IF NOT EXISTS ai_predictions (
    id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,

    prediction_run_id BIGINT UNSIGNED NOT NULL,
    model_version_id BIGINT UNSIGNED NOT NULL,

    student_id BIGINT UNSIGNED NOT NULL,

    target_period VARCHAR(20) NOT NULL,
    task_type VARCHAR(30) NOT NULL,

    target_subject VARCHAR(100) NULL,

    predicted_value DECIMAL(10,4) NOT NULL,

    academic_risk VARCHAR(30) NULL,
    prediction_reliability VARCHAR(30) NULL,
    display_policy VARCHAR(40) NULL,

    performance_trend VARCHAR(30) NULL,
    attendance_context VARCHAR(30) NULL,

    model_input_hash CHAR(64) NULL,

    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,

    PRIMARY KEY (id),

    KEY idx_ai_predictions_student (
        student_id,
        target_period
    ),

    KEY idx_ai_predictions_run (
        prediction_run_id
    ),

    KEY idx_ai_predictions_model (
        model_version_id
    ),

    KEY idx_ai_predictions_task_subject (
        task_type,
        target_subject
    ),

    CONSTRAINT fk_ai_predictions_run
        FOREIGN KEY (prediction_run_id)
        REFERENCES ai_prediction_runs(id)
        ON DELETE CASCADE,

    CONSTRAINT fk_ai_predictions_model
        FOREIGN KEY (model_version_id)
        REFERENCES ai_model_versions(id)
        ON DELETE RESTRICT,

    CONSTRAINT chk_ai_predictions_task
        CHECK (
            task_type IN (
                'OVERALL',
                'SUBJECT'
            )
        )
) ENGINE=InnoDB
  DEFAULT CHARSET=utf8mb4
  COLLATE=utf8mb4_unicode_ci;


-- ------------------------------------------------------------
-- 6. FEATURE SNAPSHOTS
-- Saves exactly what the frozen model received at prediction time.
--
-- This is intentionally a JSON snapshot rather than duplicating
-- all operational school records into an AI database.
-- ------------------------------------------------------------

CREATE TABLE IF NOT EXISTS ai_prediction_features (
    id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,

    prediction_id BIGINT UNSIGNED NOT NULL,

    feature_schema_version VARCHAR(30) NOT NULL DEFAULT '1.0',
    features_json JSON NOT NULL,

    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,

    PRIMARY KEY (id),
    UNIQUE KEY uq_ai_prediction_features_prediction (prediction_id),

    CONSTRAINT fk_ai_prediction_features_prediction
        FOREIGN KEY (prediction_id)
        REFERENCES ai_predictions(id)
        ON DELETE CASCADE
) ENGINE=InnoDB
  DEFAULT CHARSET=utf8mb4
  COLLATE=utf8mb4_unicode_ci;


-- ------------------------------------------------------------
-- 7. SHAP / EXPLANATION SNAPSHOTS
-- Stores human-readable and structured explanation factors.
-- ------------------------------------------------------------

CREATE TABLE IF NOT EXISTS ai_prediction_explanations (
    id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,

    prediction_id BIGINT UNSIGNED NOT NULL,

    baseline_value DECIMAL(10,4) NULL,

    top_positive_json JSON NULL,
    top_negative_json JSON NULL,
    explanation_json JSON NULL,

    explanation_method VARCHAR(50) NOT NULL DEFAULT 'SHAP',

    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,

    PRIMARY KEY (id),
    UNIQUE KEY uq_ai_prediction_explanations_prediction (prediction_id),

    CONSTRAINT fk_ai_prediction_explanations_prediction
        FOREIGN KEY (prediction_id)
        REFERENCES ai_predictions(id)
        ON DELETE CASCADE
) ENGINE=InnoDB
  DEFAULT CHARSET=utf8mb4
  COLLATE=utf8mb4_unicode_ci;


-- ------------------------------------------------------------
-- 8. ACTUAL OUTCOMES
-- After the future term finishes, attach the real mark/result
-- to the historical prediction so model drift can be monitored.
-- ------------------------------------------------------------

CREATE TABLE IF NOT EXISTS ai_actual_outcomes (
    id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,

    prediction_id BIGINT UNSIGNED NOT NULL,

    actual_value DECIMAL(10,4) NOT NULL,

    absolute_error DECIMAL(10,4) NULL,
    squared_error DECIMAL(14,4) NULL,

    outcome_period VARCHAR(20) NOT NULL,

    source_record_id BIGINT UNSIGNED NULL,

    recorded_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,

    PRIMARY KEY (id),
    UNIQUE KEY uq_ai_actual_outcomes_prediction (prediction_id),

    CONSTRAINT fk_ai_actual_outcomes_prediction
        FOREIGN KEY (prediction_id)
        REFERENCES ai_predictions(id)
        ON DELETE CASCADE
) ENGINE=InnoDB
  DEFAULT CHARSET=utf8mb4
  COLLATE=utf8mb4_unicode_ci;


-- ------------------------------------------------------------
-- 9. OPTIONAL MODEL-LIFECYCLE EVENT LOG
-- Useful for promotion/retirement/retraining audit.
-- ------------------------------------------------------------

CREATE TABLE IF NOT EXISTS ai_model_events (
    id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,

    model_version_id BIGINT UNSIGNED NOT NULL,

    event_type VARCHAR(40) NOT NULL,
    event_details_json JSON NULL,

    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,

    PRIMARY KEY (id),
    KEY idx_ai_model_events_model (
        model_version_id,
        created_at
    ),

    CONSTRAINT fk_ai_model_events_model
        FOREIGN KEY (model_version_id)
        REFERENCES ai_model_versions(id)
        ON DELETE CASCADE
) ENGINE=InnoDB
  DEFAULT CHARSET=utf8mb4
  COLLATE=utf8mb4_unicode_ci;


SET FOREIGN_KEY_CHECKS = 1;

SET FOREIGN_KEY_CHECKS = 1;
