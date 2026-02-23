from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging

from database import init_db
from routers.auth_router import router as auth_router
from routers.chat_router import router as chat_router
from routers.admin_router import router as admin_router
from routers.therapist_router import router as therapist_router

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up: initializing database...")
    await init_db()
    logger.info("Database initialized.")
    yield
    logger.info("Shutting down.")


app = FastAPI(
    title="AI Student Emotional Support Platform",
    description="A secure, ethical AI-powered emotional support system for students.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "http://127.0.0.1:8501"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_router)
app.include_router(chat_router)
app.include_router(admin_router)
app.include_router(therapist_router)


@app.get("/")
async def root():
    return {
        "app": "AI Student Emotional Support Platform",
        "version": "1.0.0",
        "status": "running",
        "disclaimer": "This system is NOT a replacement for professional mental health care."
    }


@app.get("/health")
async def health():
    return {"status": "healthy"}
