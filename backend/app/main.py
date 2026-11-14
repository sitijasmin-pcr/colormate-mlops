from fastapi import FastAPI
from backend.app.config import settings
from backend.app.api.routes import router

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description=settings.APP_DESCRIPTION,
    debug=settings.DEBUG
)

app.include_router(router)