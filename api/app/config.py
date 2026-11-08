from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    app_name: str = "Student Performance AI"
    app_version: str = "1.2.0"

    db_host: str = "127.0.0.1"
    db_port: int = 3306
    db_name: str = "schoolportal"
    db_user: str = "root"
    db_password: str = ""

    overall_model_path: str = str(
        PROJECT_ROOT
        / "models"
        / "final"
        / "overall_random_forest_academic_only.joblib"
    )

    subject_model_path: str = str(
        PROJECT_ROOT
        / "models"
        / "final"
        / "subject_random_forest_academic_only.joblib"
    )

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env.ai",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
