from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from backend.app.config import settings
from backend.app.services.image_service import validate_image


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


@router.post("/api/v1/analyze/upload")
async def upload_customer_photo(
    file: UploadFile = File(...),
    processing_consent: bool = Form(...),
    storage_consent: bool = Form(False),
    consent_version: str = Form(...)
):
    """
    Receive and validate a customer's photo.
    Images are processed in memory and are not stored.
    """

    try:
        if not processing_consent:
            raise HTTPException(
                status_code=403,
                detail="Photo processing consent is required."
            )

        if not consent_version.strip():
            raise HTTPException(
                status_code=400,
                detail="Consent version is required."
            )

        image_bytes = await file.read(5 * 1024 * 1024 + 1)

        validation_result = validate_image(image_bytes)

        return {
            "status": "success",
            "message": "Photo uploaded and validated successfully.",
            "filename": file.filename,
            "image": validation_result,
            "consent": {
                "processing_consent": processing_consent,
                "storage_consent": storage_consent,
                "consent_version": consent_version
            },
            "storage_status": "not_stored"
        }

    except ValueError as error:
        status_code = (
            413 if "size must not exceed" in str(error) else 400
        )

        raise HTTPException(
            status_code=status_code,
            detail=str(error)
        )

    finally:
        await file.close()