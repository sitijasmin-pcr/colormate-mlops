from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    APP_NAME: str = "ColorMate API"
    APP_VERSION: str = "1.0.0"
    APP_DESCRIPTION: str = (
        "AI Personal Color Analysis and Makeup Recommendation System"
    )
    DEBUG: bool = True

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=True
    )


settings = Settings()