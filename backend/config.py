from pydantic_settings import BaseSettings
from typing import Optional
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
class Settings(BaseSettings):
    # App
    APP_NAME: str = " Student Emotional Support Platform"
    DEBUG: bool = False

    # Security
    SECRET_KEY: str = "CHANGE_THIS_IN_PRODUCTION_USE_STRONG_RANDOM_KEY_32CHARS"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60

    # Database
    DATABASE_URL: str 

    # Risk Escalation
    RISK_ESCALATION_THRESHOLD: float = 0.65

    # AI Models
    EMOTION_MODEL: str = "distilbert-base-uncased-emotion"
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"

    # Paths
    KNOWLEDGE_BASE_DIR: str =  os.path.join(BASE_DIR, "knowledge_base")
    VECTOR_STORE_PATH: str = os.path.join(BASE_DIR, "vector_store")

    class Config:
        env_file = ".env"


settings = Settings()
