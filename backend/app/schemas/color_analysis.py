from datetime import datetime
from enum import Enum
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


# ==========================================
# 1. PERSONAL COLOR CATEGORY
# ==========================================

class PersonalColorSeason(str, Enum):
    SPRING = "spring"
    SUMMER = "summer"
    AUTUMN = "autumn"
    WINTER = "winter"


# ==========================================
# 2. CUSTOMER PHOTO CONSENT
# ==========================================

class PhotoConsent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    processing_consent: bool = Field(
        description="Permission to process the customer's photo"
    )

    storage_consent: bool = Field(
        default=False,
        description="Separate permission to retain the customer's photo"
    )

    consent_version: str = Field(
        min_length=1,
        description="Version of the consent statement shown to the customer"
    )


# ==========================================
# 3. COLOR RECOMMENDATION
# ==========================================

class ColorRecommendation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    color_name: str = Field(min_length=1)
    hex_code: str = Field(
        pattern=r"^#[0-9A-Fa-f]{6}$",
        description="Color in hexadecimal format"
    )
    description: str | None = None


# ==========================================
# 4. COLOR ANALYSIS RESULT
# ==========================================

class ColorAnalysisResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    analysis_id: UUID
    season: PersonalColorSeason
    confidence_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Model confidence between 0 and 1"
    )
    summary: str
    recommendations: list[ColorRecommendation] = Field(
        default_factory=list
    )
    created_at: datetime


# ==========================================
# 5. PHOTO CONSENT RECORD
# ==========================================

class PhotoConsentRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    customer_id: UUID
    consent: PhotoConsent
    recorded_at: datetime