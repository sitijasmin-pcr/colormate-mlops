from fastapi import FastAPI
from backend.app.config import settings


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description=settings.APP_DESCRIPTION,
    debug=settings.DEBUG
)


@app.get("/")
def root():
    return {
        "message": "Welcome to ColorMate API!",
        "project": "ColorMate MLOps",
        "version": settings.APP_VERSION,
        "status": "running"
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "application": settings.APP_NAME
    }