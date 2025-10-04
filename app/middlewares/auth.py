from fastapi import Request, Response
from fastapi.responses import RedirectResponse
from starlette.middleware.base import BaseHTTPMiddleware
from datetime import datetime


class AuthMiddleware(BaseHTTPMiddleware):
    """Middleware для проверки авторизации"""
    
    # Пути, которые не требуют авторизации
    PUBLIC_PATHS = [
        "/login",
        "/auth/login",
        "/auth/logout",
    ]
    
    # Статические файлы и API пути
    SKIP_PREFIXES = [
        "/storage/",
    ]
    
    def __init__(self, app, sessions_store):
        super().__init__(app)
        self.sessions = sessions_store
    
    async def dispatch(self, request: Request, call_next):
        # Skip authentication for public paths
        if request.url.path in self.PUBLIC_PATHS:
            return await call_next(request)
        
        # Skip authentication for certain prefixes
        for prefix in self.SKIP_PREFIXES:
            if request.url.path.startswith(prefix):
                return await call_next(request)
        
        # Check session token
        session_token = request.cookies.get("session_token")
        
        if not session_token or session_token not in self.sessions:
            # Redirect to login page for HTML pages
            if self._is_html_request(request):
                return RedirectResponse(url="/login", status_code=302)
            # Return 401 for API requests
            else:
                return Response(
                    content='{"detail":"Не авторизован"}',
                    status_code=401,
                    media_type="application/json"
                )
        
        # Check if session expired
        session = self.sessions[session_token]
        if datetime.utcnow() > session["expires_at"]:
            del self.sessions[session_token]
            
            if self._is_html_request(request):
                return RedirectResponse(url="/login?expired=true", status_code=302)
            else:
                return Response(
                    content='{"detail":"Сессия истекла"}',
                    status_code=401,
                    media_type="application/json"
                )
        
        # Continue with the request
        response = await call_next(request)
        return response
    
    def _is_html_request(self, request: Request) -> bool:
        """Check if the request is for an HTML page"""
        accept = request.headers.get("accept", "")
        return "text/html" in accept or request.url.path in ["/", "/documents"]

