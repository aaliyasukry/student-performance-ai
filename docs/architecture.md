# System Architecture

## Purpose

The AI service provides auditable next-term academic forecasts without duplicating the School Portal's operational data.

```text
School Portal / Standalone UI
            |
          FastAPI
            |
   +--------+---------+
   |                  |
School MySQL      Frozen models
   |
AI lifecycle tables
```

The School Portal database remains the source of truth for students, academic years, terms, classes, subjects, examination results and attendance.

## Runtime Flow

1. A prediction request identifies a student and target academic term.
2. FastAPI retrieves only historical records before the target period.
3. The historical feature builder reconstructs the same feature definitions used during model development.
4. The active overall and pooled-subject Random Forest artifacts generate forecasts.
5. Decision-support rules attach academic-risk, trend, reliability and display-policy labels.
6. The exact model inputs are stored as JSON feature snapshots.
7. When actual results later exist, they are linked to the historical predictions for monitoring.

## AI Lifecycle Tables

- `ai_model_versions` - deployable model versions and final evaluation summary.
- `ai_training_runs` - training/retraining audit trail.
- `ai_model_metrics` - held-out metrics used by reliability rules.
- `ai_prediction_runs` - one API or batch request.
- `ai_predictions` - overall and subject forecasts.
- `ai_prediction_features` - exact model input snapshot used for each prediction.
- `ai_prediction_explanations` - SHAP explanation snapshots.
- `ai_actual_outcomes` - later actual values and prediction error.
- `ai_model_events` - optional model lifecycle events.

## Model Lifecycle

The saved Random Forest models do not automatically learn when new marks are inserted. New outcomes are monitored against historical predictions. Retraining creates a separate candidate model version, which is promoted only after validation. Production model artifacts are never silently overwritten.

## Reliability and Risk

Academic risk describes the predicted outcome. Prediction reliability describes the strength of held-out evidence for that type of prediction. These are intentionally separate concepts. SHAP explains model behaviour and must not be presented as confidence or causality.

## Security and Privacy

Feature snapshots contain model features, not unnecessary personal identifiers. Authorization for prediction requests and history belongs to the School Portal. Local credentials, raw research data and private extracts remain outside version control.
