# AI-Powered Student Performance Forecasting and Academic Decision Support System

An undergraduate research project and deployable AI service that forecasts a student's **next-term academic performance** from their historical school records, and presents the result to teachers as explainable decision support.

## What the system does

For a student in Grades 6-10 the system predicts, before the term is examined:

- the **overall average mark** for the next term, and
- the **mark in each subject** for the next term.

Each prediction is accompanied by a risk category, a trend indicator, a reliability label derived from held-out evaluation, and SHAP-based explanations of the factors that drove it. Predictions never use any information from the term being predicted.

## Data

### Source

The data comes from one school's records for 2025 Term 1 to 2026 Term 2 (five terms):

- examination mark workbooks (one sheet per grade/class per term), and
- monthly attendance workbooks (one per year).

### What is and is not in this repository

| Data | In Git | Notes |
|---|---|---|
| Raw Excel workbooks (`data/raw/`) | No | Contain student names and identifiers |
| Cleaned parsed tables (`data/cleaned/`) | No | Still linked to real identities |
| Linkage / identity audit outputs and manual overrides | No | Contain identifying information |
| **Anonymized datasets (`data/anonymized/`)** | **Yes** | Names, admission/index numbers and source locations removed; students are pseudonymized as `STU0001`... |
| **Model-ready datasets (`data/model_ready/`)** | **Yes** | Leakage-safe features and targets built from the anonymized data |
| Trained models (`models/`) | No | Too large for Git; built locally by the training pipeline (see Quick start) |

The anonymized dataset covers 349 students and 1,412 student-term records. It is sufficient to reproduce feature engineering, model training, evaluation and explainability without access to the raw school files.

### Anonymized files

```text
data/anonymized/master_research_dataset.csv                  one row per student per term (marks + attendance)
data/anonymized/marks_long_anonymized.csv                    one row per student, term and subject
data/anonymized/marks_student_term_summary_anonymized.csv    term totals, averages and ranks
data/anonymized/attendance_term_anonymized.csv               term attendance summary
data/anonymized/student_term_identity.csv                    class and linkage status per student-term
```

### Model-ready files

```text
data/model_ready/historical_features.csv             features computed from prior terms only
data/model_ready/overall_prediction_dataset.csv      target = next-term overall average
data/model_ready/subject_prediction_dataset.csv      target = next-term mark per subject
```

## Quick start for reviewers

No database or private data is needed. Requires Python 3.10+. The trained models are not stored in Git; this builds them locally from the committed model-ready datasets.

```powershell
git clone <repository-url>
cd student-performance-ai
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

python src/features/historical_feature_engineering.py
python src/features/build_prediction_targets.py
python src/models/train_overall_academic_models.py
python src/models/train_subject_academic_models.py
python src/models/train_overall_attendance_models.py
python src/models/train_subject_attendance_models.py
python src/evaluation/validation_diagnostics.py
python src/evaluation/final_out_of_time_evaluation.py
python scripts/validation/evaluate_final_models.py
```

`final_out_of_time_evaluation.py` trains the two final Random Forest models and saves them to `models/final/`. The last command loads them, predicts the held-out 2026 Term 2 rows and prints the metrics reported in the Results table below. Optional steps: `src/explainability/shap_explainability.py` and `src/decision_support/decision_support_rules.py` regenerate the SHAP outputs and decision-support rules.

## Method

1. **Parsing and cleaning**: marks and attendance workbooks are parsed with explicit, audited handling of absences, missing values, summary rows and documented manual corrections.
2. **Student linkage and anonymization**: marks and attendance are linked per student, identities are resolved across years, and direct identifiers are replaced with pseudonymous IDs.
3. **Feature engineering**: historical features are built only from terms before the target term (previous mark, historical mean, standard deviation, trend, participation and status flags, optional attendance history). Leakage checks fail the pipeline if any current-term value enters the feature matrix.
4. **Temporal evaluation**: training uses 2025 T2 and 2025 T3 targets, validation uses 2026 T1, and **2026 T2 is held out as a final out-of-time test** that is only used after model selection is locked.
5. **Model comparison**: Linear Regression, Random Forest and XGBoost, each trained with academic/history-only features and with academic + attendance features.
6. **Selection**: attendance gave no clear paired improvement (bootstrap 95% confidence intervals include zero), so the simpler academic/history-only **Random Forest** was chosen for both tasks.
7. **Explainability and decision support**: SHAP global and local explanations, plus rules for risk bands, trend and reliability.

## Results

Final out-of-time test on 2026 Term 2:

| Predictor | Rows | MAE | RMSE | R² | Within ±10 marks |
|---|---:|---:|---:|---:|---:|
| Overall average | 268 | 4.2008 | 5.5205 | 0.9148 | 93.28% |
| Pooled subject mark | 2,912 | 9.9139 | 12.5899 | 0.7129 | 59.07% |

Retraining locally can shift the overall-average metrics slightly in the third or fourth decimal place (for example MAE 4.198 vs 4.201); the subject-mark metrics reproduce exactly.

Figures are in `figures/` and the underlying metrics in `reports/`.

## Repository structure

```text
api/                  FastAPI service that serves predictions
data/anonymized/      anonymized research datasets
data/model_ready/     engineered features and prediction targets
database/             AI schema, portal views, subject mapping and seeds (MySQL)
docs/                 architecture and school portal integration notes
figures/              EDA, modeling, evaluation and explainability figures
reports/              reproducible metrics and evaluation evidence
scripts/validation/   model contract export and portal parity validation
src/                  preprocessing, features, models, evaluation, explainability, decision support
ui/                   React + TypeScript front end for the prediction service
```

## Pipeline

Run from the repository root. Steps 1-4 need the private raw data and are provided for transparency; steps 5 onward run from the committed datasets.

```text
1. src/preprocessing/data_inventory.py
   src/preprocessing/marks_structure_audit.py
   src/preprocessing/attendance_structure_audit.py
2. src/preprocessing/marks_parser.py
   src/preprocessing/attendance_parser.py
   src/preprocessing/attendance_value_semantics_audit.py
3. src/preprocessing/student_linkage_audit.py
   src/preprocessing/apply_manual_linkage_overrides.py
   src/preprocessing/cross_year_identity_audit.py
   src/preprocessing/apply_cross_year_identity_overrides.py
4. src/preprocessing/build_master_research_dataset.py
   src/preprocessing/anonymize_research_data.py
5. src/preprocessing/eda_analysis.py
6. src/features/historical_feature_engineering.py
   src/features/build_prediction_targets.py
7. src/models/train_overall_academic_models.py
   src/models/train_subject_academic_models.py
   src/models/train_overall_attendance_models.py
   src/models/train_subject_attendance_models.py
8. src/evaluation/validation_diagnostics.py
   src/evaluation/final_out_of_time_evaluation.py
9. src/explainability/shap_explainability.py
   src/decision_support/decision_support_rules.py
```

The final models are written to `models/final/` (not committed to Git):

```text
models/final/overall_random_forest_academic_only.joblib
models/final/subject_random_forest_academic_only.joblib
```

## Running the API

The API reads student history from a MySQL School Portal database, so it needs a MySQL server with the portal tables. The `database/` scripts add the AI tables and views on top of an existing portal schema; the portal data itself is private and not included. Install the dependencies as in the quick start, then copy `api/.env.ai.example` to `.env.ai` in the repository root and set the database credentials. Create the database objects by running these scripts in order:

```text
database/schema.sql
database/source_views.sql
database/subject_mappings.sql
database/seed_models.sql
database/seed_model_metrics.sql
```

Start the service:

```powershell
uvicorn app.main:app --app-dir api --reload --host 127.0.0.1 --port 8000
```

Interactive documentation is available at `http://127.0.0.1:8000/docs`. Main endpoints:

```text
GET  /health
GET  /api/v1/models/active
GET  /api/v1/models/subject-reliability
GET  /api/v1/models/overall-reliability
GET  /api/v1/integration/readiness
POST /api/v1/predictions/student
```

A student prediction reads only periods before the requested target term. Actual target values may be returned for validation when they already exist, but are never used as features.

## Running the UI

```powershell
cd ui
npm install
npm run dev
```

The UI expects the API at `http://127.0.0.1:8000`.

## Validation utilities

```text
scripts/validation/evaluate_final_models.py        evaluate the final models on the 2026 T2 test rows (no database needed)
scripts/validation/export_model_contract.py        export the feature contract of the trained models
scripts/validation/validate_portal_model_parity.py reproduce the 2026 T2 evaluation from live portal data
```

## Privacy

Raw school records, identity linkage outputs, database credentials and private prediction extracts are excluded from version control. Only the anonymized and model-ready datasets are published.
