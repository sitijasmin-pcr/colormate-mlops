from fastapi import APIRouter
from backend.app.config import settings

router = APIRouter()


@router.get("/")
def root():
    return {
        "message": "Welcome to ColorMate API!",
        "project": "ColorMate MLOps",
        "version": settings.APP_VERSION,
        "status": "running"
    }


@router.get("/health")
def health_check():
    return {
        "status": "healthy",
        "application": settings.APP_NAME
    }