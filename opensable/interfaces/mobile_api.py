"""
Open-Sable Mobile API

REST API for mobile applications (iOS/Android).
Provides endpoints for messaging, session management, and real-time updates.
"""

import asyncio
import logging
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
import secrets

try:
    import jwt

    JWT_AVAILABLE = True
except ImportError:
    jwt = None
    JWT_AVAILABLE = False

try:
    from fastapi import FastAPI, HTTPException, Depends, Header, WebSocket, WebSocketDisconnect
    from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel
    import uvicorn

    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    FastAPI = None
    BaseModel = object

from opensable.core.agent import SableAgent
from opensable.core.config import Config
from opensable.core.session_manager import SessionManager
from opensable.core.commands import CommandHandler

logger = logging.getLogger(__name__)

# Security
security = HTTPBearer()


class MessageRequest(BaseModel):
    """Message request from mobile client"""

    message: str
    session_id: Optional[str] = None
    channel: str = "mobile"
    metadata: Optional[Dict[str, Any]] = None


class MessageResponse(BaseModel):
    """Message response to mobile client"""

    response: str
    session_id: str
    timestamp: str
    tokens_used: Optional[int] = None


class SessionInfo(BaseModel):
    """Session information"""

    session_id: str
    channel: str
    user_id: str
    created_at: str
    last_active: str
    message_count: int
    state: str


class AuthRequest(BaseModel):
    """Authentication request"""

    username: str
    password: str
    device_id: Optional[str] = None


class AuthResponse(BaseModel):
    """Authentication response"""

    access_token: str
    refresh_token: str
    expires_in: int
    user_id: str


class MobileAPI:
    """Mobile API server"""

    def __init__(self, config: Config):
        self.config = config
        self.app = FastAPI(title="Open-Sable Mobile API", version="0.2.0")
        self.agent = SableAgent(config)
        self.session_manager = SessionManager()
        self.command_handler = CommandHandler(self.session_manager)

        # JWT settings
        self.jwt_secret = getattr(config, "jwt_secret", secrets.token_hex(32))
        self.jwt_algorithm = "HS256"
        self.access_token_expire = 3600  # 1 hour
        self.refresh_token_expire = 86400 * 30  # 30 days

        # Active WebSocket connections
        self.websocket_connections: Dict[str, WebSocket] = {}

        # User authentication (simple in-memory store)
        self.users = {"demo": "demo123"}  # username: password (hash in production!)

        self._setup_middleware()
        self._setup_routes()

    def _setup_middleware(self):
        """Setup CORS and other middleware"""
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],  # Configure properly in production
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    def create_access_token(self, user_id: str, device_id: Optional[str] = None) -> str:
        """Create JWT access token"""
        expire = datetime.utcnow() + timedelta(seconds=self.access_token_expire)
        payload = {"user_id": user_id, "device_id": device_id, "exp": expire, "type": "access"}
        return jwt.encode(payload, self.jwt_secret, algorithm=self.jwt_algorithm)

    def create_refresh_token(self, user_id: str, device_id: Optional[str] = None) -> str:
        """Create JWT refresh token"""
        expire = datetime.utcnow() + timedelta(seconds=self.refresh_token_expire)
        payload = {"user_id": user_id, "device_id": device_id, "exp": expire, "type": "refresh"}
        return jwt.encode(payload, self.jwt_secret, algorithm=self.jwt_algorithm)

    def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Verify and decode JWT token"""
        try:
            payload = jwt.decode(token, self.jwt_secret, algorithms=[self.jwt_algorithm])
            return payload
        except jwt.ExpiredSignatureError:
            logger.warning("Token expired")
            return None
        except jwt.InvalidTokenError as e:
            logger.warning(f"Invalid token: {e}")
            return None

    async def get_current_user(
        self, credentials: HTTPAuthorizationCredentials = Depends(security)
    ) -> str:
        """Get current user from bearer token"""
        token = credentials.credentials
        payload = self.verify_token(token)

        if not payload or payload.get("type") != "access":
            raise HTTPException(status_code=401, detail="Invalid or expired token")

        return payload.get("user_id")

    def _setup_routes(self):
        """Setup API routes"""

        @self.app.post("/auth/login", response_model=AuthResponse)
        async def login(request: AuthRequest):
            """Authenticate user and get tokens"""
            username = request.username
            password = request.password

            # Validate credentials (simple check - use proper auth in production!)
            if username not in self.users or self.users[username] != password:
                raise HTTPException(status_code=401, detail="Invalid credentials")

            # Create tokens
            access_token = self.create_access_token(username, request.device_id)
            refresh_token = self.create_refresh_token(username, request.device_id)

            logger.info(f"User logged in: {username}")

            return AuthResponse(
                access_token=access_token,
                refresh_token=refresh_token,
                expires_in=self.access_token_expire,
                user_id=username,
            )

        @self.app.post("/auth/refresh", response_model=AuthResponse)
        async def refresh(credentials: HTTPAuthorizationCredentials = Depends(security)):
            """Refresh access token"""
            token = credentials.credentials
            payload = self.verify_token(token)

            if not payload or payload.get("type") != "refresh":
                raise HTTPException(status_code=401, detail="Invalid refresh token")

            user_id = payload.get("user_id")
            device_id = payload.get("device_id")

            # Create new tokens
            access_token = self.create_access_token(user_id, device_id)
            refresh_token = self.create_refresh_token(user_id, device_id)

            return AuthResponse(
                access_token=access_token,
                refresh_token=refresh_token,
                expires_in=self.access_token_expire,
                user_id=user_id,
            )

        @self.app.get("/")
        async def root():
            """API root"""
            return {
                "service": "Open-Sable Mobile API",
                "version": "0.2.0",
                "status": "running",
                "endpoints": {
                    "auth": "/auth/login",
                    "message": "/message",
                    "sessions": "/sessions",
                    "websocket": "/ws",
                    "docs": "/docs",
                },
            }

        @self.app.get("/health")
        async def health():
            """Health check"""
            return {
                "status": "healthy",
                "sessions": len(self.session_manager.sessions),
                "websocket_connections": len(self.websocket_connections),
            }

        @self.app.post("/message", response_model=MessageResponse)
        async def send_message(
            request: MessageRequest, user_id: str = Depends(self.get_current_user)
        ):
            """Send message to agent"""
            try:
                # Get or create session
                session = self.session_manager.get_or_create_session(
                    channel=request.channel, user_id=user_id
                )

                # Check for commands
                if request.message.startswith("/"):
                    result = self.command_handler.handle_command(
                        request.message, session, is_admin=False
                    )

                    if result:
                        return MessageResponse(
                            response=result.message,
                            session_id=session.session_id,
                            timestamp=datetime.utcnow().isoformat(),
                        )

                # Process through agent
                response = await self.agent.run(request.message, session)

                return MessageResponse(
                    response=response,
                    session_id=session.session_id,
                    timestamp=datetime.utcnow().isoformat(),
                )

            except Exception as e:
                logger.error(f"Error processing message: {e}", exc_info=True)
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/sessions", response_model=List[SessionInfo])
        async def list_sessions(user_id: str = Depends(self.get_current_user)):
            """List user's sessions"""
            sessions = self.session_manager.list_sessions(filters={"user_id": user_id})

            return [
                SessionInfo(
                    session_id=s.session_id,
                    channel=s.channel,
                    user_id=s.user_id,
                    created_at=s.created_at.isoformat(),
                    last_active=s.last_active.isoformat(),
                    message_count=len(s.messages),
                    state=s.state,
                )
                for s in sessions
            ]

        @self.app.get("/sessions/{session_id}", response_model=SessionInfo)
        async def get_session(session_id: str, user_id: str = Depends(self.get_current_user)):
            """Get session details"""
            session = self.session_manager.get_session(session_id)

            if not session or session.user_id != user_id:
                raise HTTPException(status_code=404, detail="Session not found")

            return SessionInfo(
                session_id=session.session_id,
                channel=session.channel,
                user_id=session.user_id,
                created_at=session.created_at.isoformat(),
                last_active=session.last_active.isoformat(),
                message_count=len(session.messages),
                state=session.state,
            )

        @self.app.delete("/sessions/{session_id}")
        async def delete_session(session_id: str, user_id: str = Depends(self.get_current_user)):
            """Delete session"""
            session = self.session_manager.get_session(session_id)

            if not session or session.user_id != user_id:
                raise HTTPException(status_code=404, detail="Session not found")

            self.session_manager.delete_session(session_id)

            return {"status": "deleted", "session_id": session_id}

        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket, token: str):
            """WebSocket connection for real-time messaging"""
            # Verify token
            payload = self.verify_token(token)
            if not payload or payload.get("type") != "access":
                await websocket.close(code=1008)
                return

            user_id = payload.get("user_id")

            await websocket.accept()
            self.websocket_connections[user_id] = websocket

            try:
                logger.info(f"WebSocket connected: {user_id}")

                # Send welcome
                await websocket.send_json(
                    {
                        "type": "connected",
                        "user_id": user_id,
                        "timestamp": datetime.utcnow().isoformat(),
                    }
                )

                # Handle messages
                while True:
                    data = await websocket.receive_json()
                    await self._handle_websocket_message(websocket, user_id, data)

            except WebSocketDisconnect:
                logger.info(f"WebSocket disconnected: {user_id}")
            except Exception as e:
                logger.error(f"WebSocket error: {e}", exc_info=True)
            finally:
                if user_id in self.websocket_connections:
                    del self.websocket_connections[user_id]

    async def _handle_websocket_message(self, websocket: WebSocket, user_id: str, data: dict):
        """Handle WebSocket message"""
        msg_type = data.get("type")

        if msg_type == "ping":
            await websocket.send_json({"type": "pong", "timestamp": datetime.utcnow().isoformat()})

        elif msg_type == "message":
            message = data.get("message")
            session_id = data.get("session_id")

            # Get or create session
            session = None
            if session_id:
                session = self.session_manager.get_session(session_id)

            if not session:
                session = self.session_manager.get_or_create_session(
                    channel="mobile", user_id=user_id
                )

            # Process message
            response = await self.agent.run(message, session)

            # Send response
            await websocket.send_json(
                {
                    "type": "response",
                    "response": response,
                    "session_id": session.session_id,
                    "timestamp": datetime.utcnow().isoformat(),
                }
            )

        else:
            await websocket.send_json(
                {"type": "error", "message": f"Unknown message type: {msg_type}"}
            )

    async def start(self, host: str = "0.0.0.0", port: int = 8000):
        """Start mobile API server"""
        logger.info(f"Starting Open-Sable Mobile API on {host}:{port}")
        logger.info(f"API Docs: http://{host}:{port}/docs")

        config = uvicorn.Config(self.app, host=host, port=port, log_level="info")

        server = uvicorn.Server(config)
        await server.serve()


if __name__ == "__main__":
    from opensable.core.config import load_config

    config = load_config()
    api = MobileAPI(config)

    asyncio.run(api.start())
