from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from app.config import settings
from app.database import check_database_connection
from app.repositories.ai_repository import AIRepository
from app.repositories.school_repository import SchoolRepository
from app.schemas import CohortPredictionRequest, StudentPredictionRequest
from app.services.model_service import ModelService
from app.services.prediction_service import PredictionService


model_service = ModelService()
ai_repository = AIRepository()
school_repository = SchoolRepository()
prediction_service = PredictionService(
    model_service=model_service,
    school_repository=school_repository,
    ai_repository=ai_repository,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    model_service.load_models()
    yield


app = FastAPI(
    title="Student Performance AI API",
    version="1.2.0",
    description=(
        "FastAPI service for next-term student performance forecasting, "
        "model lifecycle metadata, feature snapshots and school-system integration."
    ),
    lifespan=lifespan,
)


@app.get("/")
def root():
    return {
        "service": settings.app_name,
        "version": settings.app_version,
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health")
def health():
    db_ok, db_message = check_database_connection()

    return {
        "status": "ok" if db_ok and model_service.ready else "degraded",
        "service": settings.app_name,
        "version": settings.app_version,
        "database": {
            "connected": db_ok,
            "message": db_message,
        },
        "models": model_service.health_summary(),
    }


@app.get("/api/v1/models/active")
def get_active_models():
    return {
        "database_models": ai_repository.get_active_models(),
        "local_artifacts": model_service.health_summary(),
    }


@app.get("/api/v1/models/subject-reliability")
def get_subject_reliability():
    return {
        "items": ai_repository.get_subject_reliability()
    }


@app.get("/api/v1/models/overall-reliability")
def get_overall_reliability():
    return ai_repository.get_overall_reliability()


@app.get("/api/v1/integration/readiness")
def get_integration_readiness():
    return school_repository.get_integration_readiness()


@app.get("/api/v1/integration/source-contract")
def get_source_contract():
    return school_repository.get_source_contract()


@app.get("/api/v1/predictions/summary")
def get_prediction_summary():
    return ai_repository.get_prediction_summary()


@app.get("/api/v1/lookups/cohort-options")
def get_cohort_options(target_year: int):
    return school_repository.get_cohort_options(target_year)


@app.post("/api/v1/predictions/student")
def predict_student(request: StudentPredictionRequest):
    try:
        return prediction_service.predict_student(
            student_id=request.student_id,
            target_year=request.target_year,
            target_term=request.target_term,
            persist=request.persist,
            source=request.source,
            requested_by_user_id=request.requested_by_user_id,
            include_actual_if_available=request.include_actual_if_available,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {exc}",
        ) from exc

@app.post("/api/v1/predictions/cohort")
def predict_cohort(request: CohortPredictionRequest):
    try:
        return prediction_service.predict_cohort(
            target_year=request.target_year,
            target_term=request.target_term,
            grade_id=request.grade_id,
            class_id=request.class_id,
            persist=request.persist,
            source=request.source,
            requested_by_user_id=request.requested_by_user_id,
            include_actual_if_available=request.include_actual_if_available,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Cohort prediction failed: {exc}",
        ) from exc

