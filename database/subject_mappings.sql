-- Canonical mapping between School Portal subjects and model subject keys.
SET NAMES utf8mb4;

CREATE TABLE IF NOT EXISTS ai_subject_mappings (
    id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
    subject_id BIGINT UNSIGNED NOT NULL,
    model_subject_key VARCHAR(100) NOT NULL,
    model_subject_name VARCHAR(150) NOT NULL,
    is_predictive TINYINT(1) NOT NULL DEFAULT 1,
    mapping_role VARCHAR(30) NOT NULL DEFAULT 'PRIMARY',
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    PRIMARY KEY (id),
    UNIQUE KEY uq_ai_subject_mappings_subject (subject_id),
    KEY idx_ai_subject_mappings_model_key (model_subject_key)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

DELETE FROM ai_subject_mappings;

INSERT INTO ai_subject_mappings (subject_id, model_subject_key, model_subject_name, is_predictive, mapping_role)
SELECT s.id, x.model_subject_key, x.model_subject_name, x.is_predictive, x.mapping_role
FROM subjects s
INNER JOIN (
    SELECT 'Agri' code, 'agriculture' model_subject_key, 'Agriculture' model_subject_name, 1 is_predictive, 'PRIMARY' mapping_role
    UNION ALL SELECT 'AL','arabic_literature','Arabic Literature',1,'PRIMARY'
    UNION ALL SELECT 'Art','art','Art',1,'PRIMARY'
    UNION ALL SELECT 'CE','ce','CE',1,'PRIMARY'
    UNION ALL SELECT 'COM','commerce','Commerce',1,'PRIMARY'
    UNION ALL SELECT 'ENG','english','English',1,'PRIMARY'
    UNION ALL SELECT 'E','english','English',1,'ALIAS'
    UNION ALL SELECT 'ELIT','english_literature','English Literature',1,'PRIMARY'
    UNION ALL SELECT 'Geo','geography','Geography',1,'PRIMARY'
    UNION ALL SELECT 'HE','health_and_physical_education','Health And Physical Education',1,'PRIMARY'
    UNION ALL SELECT 'His','history','History',1,'PRIMARY'
    UNION ALL SELECT 'IT','ict','ICT',1,'PRIMARY'
    UNION ALL SELECT 'M','mathematics','Mathematics',1,'PRIMARY'
    UNION ALL SELECT 'REL','religion','Religion',1,'PRIMARY'
    UNION ALL SELECT 'ISLAM','religion','Religion',1,'ALIAS'
    UNION ALL SELECT 'SIVAM','religion','Religion',1,'ALIAS'
    UNION ALL SELECT 'SCI','science','Science',1,'PRIMARY'
    UNION ALL SELECT 'Sin','sinhala','Sinhala',1,'PRIMARY'
    UNION ALL SELECT 'T','tamil','Tamil',1,'PRIMARY'
    UNION ALL SELECT 'TL','tamil_literature','Tamil Literature',1,'PRIMARY'
    UNION ALL SELECT 'HOME-SCI','home_science','Home Science',0,'NON_PREDICTIVE'
    UNION ALL SELECT 'PTS','pts','PTS',0,'NON_PREDICTIVE'
) x ON UPPER(s.code) = UPPER(x.code);

CREATE OR REPLACE VIEW ai_v_subject_mapping AS
SELECT
    asm.id AS mapping_id, asm.subject_id, s.code AS portal_subject_code,
    s.name AS portal_subject_name, asm.model_subject_key, asm.model_subject_name,
    asm.is_predictive, asm.mapping_role
FROM ai_subject_mappings asm
INNER JOIN subjects s ON s.id = asm.subject_id;
