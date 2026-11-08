from pathlib import Path

import joblib
import pandas as pd

from app.config import settings


class ModelService:
    def __init__(self):
        self.overall_artifact = None
        self.subject_artifact = None
        self.ready = False

    def load_models(self):
        overall_path = Path(settings.overall_model_path)
        subject_path = Path(settings.subject_model_path)

        missing = [
            str(path)
            for path in [overall_path, subject_path]
            if not path.exists()
        ]

        if missing:
            raise FileNotFoundError(
                "Required final model artifacts are missing:\n"
                + "\n".join(missing)
            )

        self.overall_artifact = joblib.load(overall_path)
        self.subject_artifact = joblib.load(subject_path)
        self.ready = True

    @property
    def overall_features(self):
        return list(self.overall_artifact["features"])

    @property
    def subject_numeric_features(self):
        return list(self.subject_artifact["numeric_features"])

    @property
    def subject_categorical_features(self):
        return list(self.subject_artifact["categorical_features"])

    @property
    def subject_categories(self):
        pipeline = self.subject_artifact["pipeline"]
        preprocessor = pipeline.named_steps["preprocessor"]
        encoder = preprocessor.named_transformers_["subject"]
        return list(encoder.categories_[0])

    def predict_overall(self, feature_row: dict):
        frame = pd.DataFrame(
            [{name: feature_row.get(name) for name in self.overall_features}]
        )
        return float(self.overall_artifact["pipeline"].predict(frame)[0])

    def predict_subject(self, feature_row: dict):
        contract = self.subject_numeric_features + self.subject_categorical_features
        frame = pd.DataFrame(
            [{name: feature_row.get(name) for name in contract}]
        )
        return float(self.subject_artifact["pipeline"].predict(frame)[0])

    def health_summary(self):
        return {
            "ready": self.ready,
            "overall": {
                "path": settings.overall_model_path,
                "loaded": self.overall_artifact is not None,
                "source_feature_count": (
                    len(self.overall_artifact.get("features", []))
                    if self.overall_artifact
                    else None
                ),
            },
            "subject": {
                "path": settings.subject_model_path,
                "loaded": self.subject_artifact is not None,
                "numeric_feature_count": (
                    len(self.subject_artifact.get("numeric_features", []))
                    if self.subject_artifact
                    else None
                ),
                "categorical_features": (
                    self.subject_artifact.get("categorical_features")
                    if self.subject_artifact
                    else None
                ),
            },
        }
