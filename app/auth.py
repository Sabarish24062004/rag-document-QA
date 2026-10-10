"""Optional single-password protection for a hosted deployment.

When APP_PASSWORD is set, every /api route (except login/session) and the API
docs require a session cookie obtained from POST /api/login. The static
frontend stays public so it can show the login screen. The cookie holds an
HMAC of the password, so changing APP_PASSWORD logs everyone out.
"""

import hashlib
import hmac
import time

from fastapi import APIRouter, HTTPException, Request, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from starlette.middleware.base import BaseHTTPMiddleware

from app import config

COOKIE_NAME = "rag_session"
COOKIE_MAX_AGE = 30 * 24 * 3600  # 30 days
PUBLIC_API_PATHS = {"/api/login", "/api/logout", "/api/session"}
PROTECTED_PREFIXES = ("/api/", "/docs", "/redoc", "/openapi.json")

router = APIRouter(prefix="/api")


def _password() -> str:
    return config.settings.app_password


def _session_token(password: str) -> str:
    return hmac.new(password.encode(), b"rag-document-qa-session-v1", hashlib.sha256).hexdigest()


def is_authenticated(request: Request) -> bool:
    password = _password()
    if not password:
        return True
    cookie = request.cookies.get(COOKIE_NAME, "")
    return hmac.compare_digest(cookie, _session_token(password))


class AuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        path = request.url.path
        if (
            path.startswith(PROTECTED_PREFIXES)
            and path not in PUBLIC_API_PATHS
            and not is_authenticated(request)
        ):
            return JSONResponse({"detail": "Login required."}, status_code=401)
        return await call_next(request)


class LoginRequest(BaseModel):
    password: str = Field(max_length=500)


@router.get("/session")
def session(request: Request) -> dict:
    return {"auth_required": bool(_password()), "authenticated": is_authenticated(request)}


@router.post("/login")
def login(body: LoginRequest, request: Request, response: Response) -> dict:
    password = _password()
    if not password:
        return {"authenticated": True}
    if not hmac.compare_digest(body.password.encode(), password.encode()):
        time.sleep(1)  # slow down password guessing
        raise HTTPException(401, "Wrong password.")
    response.set_cookie(
        COOKIE_NAME,
        _session_token(password),
        max_age=COOKIE_MAX_AGE,
        httponly=True,
        samesite="lax",
        secure=request.url.scheme == "https",
    )
    return {"authenticated": True}


@router.post("/logout")
def logout(response: Response) -> dict:
    response.delete_cookie(COOKIE_NAME)
    return {"authenticated": False}
