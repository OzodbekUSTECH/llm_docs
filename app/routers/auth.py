from fastapi import APIRouter, HTTPException, status, Response, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from datetime import datetime, timedelta
import secrets

router = APIRouter(
    prefix="/auth",
    tags=["authentication"]
)

# Credentials (в production используйте БД и хеширование)
VALID_USERNAME = "admin"
VALID_PASSWORD = "9cdda67ded3f25811728276cefa76b80913b4c54"

# Simple in-memory session store (в production используйте Redis)
sessions = {}


class LoginRequest(BaseModel):
    username: str
    password: str


class LoginResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    message: str


@router.post("/login", response_model=LoginResponse)
async def login(request: LoginRequest, response: Response):
    """Аутентификация пользователя"""
    
    # Validate credentials
    if request.username != VALID_USERNAME or request.password != VALID_PASSWORD:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Неверный логин или пароль"
        )
    
    # Generate session token
    session_token = secrets.token_urlsafe(32)
    
    # Store session (expires in 24 hours)
    sessions[session_token] = {
        "username": request.username,
        "created_at": datetime.utcnow(),
        "expires_at": datetime.utcnow() + timedelta(hours=24)
    }
    
    # Set cookie
    response.set_cookie(
        key="session_token",
        value=session_token,
        httponly=True,
        max_age=86400,  # 24 hours
        samesite="lax",
        secure=False  # Set to True in production with HTTPS
    )
    
    return LoginResponse(
        access_token=session_token,
        message="Успешный вход в систему"
    )


@router.post("/logout")
async def logout(request: Request, response: Response):
    """Выход из системы"""
    
    session_token = request.cookies.get("session_token")
    
    if session_token and session_token in sessions:
        del sessions[session_token]
    
    response = JSONResponse(content={"message": "Успешный выход"})
    response.delete_cookie("session_token")
    
    return response


@router.get("/check")
async def check_auth(request: Request):
    """Проверка авторизации"""
    
    session_token = request.cookies.get("session_token")
    
    if not session_token or session_token not in sessions:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Не авторизован"
        )
    
    session = sessions[session_token]
    
    # Check if session expired
    if datetime.utcnow() > session["expires_at"]:
        del sessions[session_token]
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Сессия истекла"
        )
    
    return {
        "authenticated": True,
        "username": session["username"]
    }


def verify_session(request: Request) -> bool:
    """Helper function to verify session"""
    
    session_token = request.cookies.get("session_token")
    
    if not session_token or session_token not in sessions:
        return False
    
    session = sessions[session_token]
    
    # Check if session expired
    if datetime.utcnow() > session["expires_at"]:
        del sessions[session_token]
        return False
    
    return True

