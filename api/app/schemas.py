from typing import Literal, Optional

from pydantic import BaseModel, Field


class ActiveModelResponse(BaseModel):
    id: int
    model_key: str
    version: str
    task_type: Literal["OVERALL", "SUBJECT"]
    algorithm: str
    feature_set: str
    final_test_mae: Optional[float] = None
    final_test_rmse: Optional[float] = None
    final_test_r2: Optional[float] = None
    is_active: bool


class SubjectReliabilityResponse(BaseModel):
    subject: str
    row_count: int
    mae: Optional[float] = None
    rmse: Optional[float] = None
    r2: Optional[float] = None
    within_10_pct: Optional[float] = None
    reliability: str
    display_policy: str


class StudentPredictionRequest(BaseModel):
    student_id: int = Field(gt=0)
    target_year: int = Field(ge=2025, le=2100)
    target_term: int = Field(ge=1, le=3)
    persist: bool = True
    source: Literal[
        "STANDALONE_UI",
        "LARAVEL",
        "API",
        "BATCH",
        "RESEARCH",
    ] = "API"
    requested_by_user_id: Optional[int] = Field(default=None, gt=0)
    include_actual_if_available: bool = True


class CohortPredictionRequest(BaseModel):
    target_year: int = Field(ge=2025, le=2100)
    target_term: int = Field(ge=1, le=3)
    grade_id: int = Field(gt=0)
    class_id: Optional[int] = Field(default=None, gt=0)
    persist: bool = False
    source: Literal[
        "STANDALONE_UI",
        "LARAVEL",
        "API",
        "BATCH",
        "RESEARCH",
    ] = "BATCH"
    requested_by_user_id: Optional[int] = Field(default=None, gt=0)
    include_actual_if_available: bool = False

