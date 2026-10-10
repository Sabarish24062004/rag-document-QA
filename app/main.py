"""FastAPI application entry point.

Run with:  uvicorn app.main:app --reload
Then open http://127.0.0.1:8000 for the web UI (API docs at /docs).
"""

import logging
import mimetypes

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.api.routes import router
from app.auth import AuthMiddleware, router as auth_router
from app.config import settings

logging.basicConfig(
    level=settings.log_level,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title=settings.app_name,
    description="Upload PDFs and ask questions answered with page-level citations.",
    version="1.0.0",
)

app.add_middleware(AuthMiddleware)
app.include_router(auth_router)
app.include_router(router)


@app.get("/health")
def health() -> dict[str, str]:
    """Health check used by the frontend and monitoring."""
    return {"status": "healthy"}


# Not in every OS mimetypes table; browsers need it to install the app.
mimetypes.add_type("application/manifest+json", ".webmanifest")

# Mounted last so /api and /health take precedence; serves index.html at "/".
app.mount("/", StaticFiles(directory=settings.frontend_dir, html=True), name="frontend")
