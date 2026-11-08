
SET NAMES utf8mb4;

-- Remove prior previous subject/overall reliability rows for v1.0.0
-- so any previously guessed fields cannot survive.
DELETE m
FROM ai_model_metrics m
INNER JOIN ai_model_versions mv
    ON mv.id = m.model_version_id
WHERE
    mv.version = '1.0.0'
    AND m.evaluation_period = '2026_T2'
    AND (
        (mv.model_key = 'overall_random_forest_academic_only'
         AND m.metric_scope = 'OVERALL')
        OR
        (mv.model_key = 'subject_random_forest_academic_only'
         AND m.metric_scope = 'SUBJECT')
    );

-- Overall model: fully verified final holdout metrics.
INSERT INTO ai_model_metrics (
    model_version_id,
    evaluation_period,
    metric_scope,
    scope_value,
    row_count,
    mae,
    rmse,
    r2,
    within_5_pct,
    within_10_pct,
    reliability_label,
    display_policy
)
SELECT
    mv.id,
    '2026_T2',
    'OVERALL',
    NULL,
    268,
    4.2008,
    5.5205,
    0.9148,
    64.9300,
    93.2800,
    'HIGH',
    'SHOW'
FROM ai_model_versions mv
WHERE
    mv.model_key = 'overall_random_forest_academic_only'
    AND mv.version = '1.0.0';

-- Subject model: exact verified rows / MAE / R2 + final decision-support reliability.
INSERT INTO ai_model_metrics (
    model_version_id,
    evaluation_period,
    metric_scope,
    scope_value,
    row_count,
    mae,
    rmse,
    r2,
    within_10_pct,
    reliability_label,
    display_policy
)
SELECT
    mv.id,
    '2026_T2',
    'SUBJECT',
    seed.scope_value,
    seed.row_count,
    seed.mae,
    NULL,
    seed.r2,
    NULL,
    seed.reliability_label,
    seed.display_policy
FROM ai_model_versions mv
JOIN (
    SELECT 'english' AS scope_value, 260 AS row_count, 8.2692 AS mae, 0.8132 AS r2, 'HIGH' AS reliability_label, 'SHOW' AS display_policy
    UNION ALL SELECT 'history', 256, 8.5175, 0.7283, 'HIGH', 'SHOW'
    UNION ALL SELECT 'science', 256, 8.5865, 0.7188, 'HIGH', 'SHOW'
    UNION ALL SELECT 'religion', 260, 9.5091, 0.7058, 'HIGH', 'SHOW'

    UNION ALL SELECT 'health_and_physical_education', 197, 8.7961, 0.5970, 'MODERATE', 'SHOW'
    UNION ALL SELECT 'tamil', 263, 9.1043, 0.6882, 'MODERATE', 'SHOW'
    UNION ALL SELECT 'ce', 193, 10.1058, 0.6376, 'MODERATE', 'SHOW'
    UNION ALL SELECT 'arabic_literature', 32, 10.2463, 0.4268, 'MODERATE', 'SHOW'
    UNION ALL SELECT 'mathematics', 259, 10.3534, 0.7127, 'MODERATE', 'SHOW'
    UNION ALL SELECT 'geography', 199, 10.5340, 0.6583, 'MODERATE', 'SHOW'
    UNION ALL SELECT 'sinhala', 254, 10.9524, 0.7055, 'MODERATE', 'SHOW'
    UNION ALL SELECT 'ict', 223, 11.2729, 0.5030, 'MODERATE', 'SHOW'

    UNION ALL SELECT 'art', 223, 12.8499, 0.0331, 'LOW', 'SHOW_WITH_CAUTION'
    UNION ALL SELECT 'agriculture', 25, 16.5019, -0.1771, 'LOW', 'SHOW_WITH_CAUTION'

    UNION ALL SELECT 'english_literature', 4, 7.0373, 0.6389, 'INSUFFICIENT_DATA', 'SUPPRESS_BY_DEFAULT'
    UNION ALL SELECT 'commerce', 7, 8.5824, 0.2589, 'INSUFFICIENT_DATA', 'SUPPRESS_BY_DEFAULT'
    UNION ALL SELECT 'tamil_literature', 1, 22.9021, NULL, 'INSUFFICIENT_DATA', 'SUPPRESS_BY_DEFAULT'
) seed
WHERE
    mv.model_key = 'subject_random_forest_academic_only'
    AND mv.version = '1.0.0';

-- Verification
SELECT
    mv.model_key,
    m.metric_scope,
    m.scope_value,
    m.row_count,
    m.mae,
    m.rmse,
    m.r2,
    m.within_10_pct,
    m.reliability_label,
    m.display_policy
FROM ai_model_metrics m
INNER JOIN ai_model_versions mv
    ON mv.id = m.model_version_id
WHERE
    mv.version = '1.0.0'
    AND m.evaluation_period = '2026_T2'
ORDER BY
    mv.task_type,
    m.metric_scope,
    m.scope_value;