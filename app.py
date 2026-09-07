"""FastAPI backend with hybrid storage (file-based for anonymous, DB for authenticated)."""

from __future__ import annotations

import json
import uuid
from typing import AsyncGenerator, Generator
from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sqlalchemy.orm import Session
from backend import storage, llm

from backend.db import (
    SessionLocal,
    User,
    get_user_by_username,
    create_user,
    authenticate_user,
    get_session,
    list_user_sessions,
    create_or_update_session,
    delete_session,
    get_session_state,
)
from backend.auth import create_access_token, verify_token

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class Token(BaseModel):
    access_token: str
    token_type: str


class UserCreate(BaseModel):
    username: str
    password: str
    first_name: str 
    last_name: str 

class LoginRequest(BaseModel):
    username: str
    password: str


class ChatRequest(BaseModel):
    message: str
    history: list[dict[str, str]]
    session_id: str | None = None   # optional; if not provided, generate


class SessionUpdate(BaseModel):
    messages: list[dict[str, str]]
    tone_label: str = "neutral"
    tone_confidence: float = 0.0
    title: str = "New Chat"


class TitleRequest(BaseModel):
    history: list[dict[str, str]]

class ToneRequest(BaseModel):
    history: list[dict[str, str]]
    current_message: str = ""

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(title="ChatTalk API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Authentication dependencies
# ---------------------------------------------------------------------------

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login", auto_error=False)


def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_current_user_optional(
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db),
) -> User | None:
    if not token:
        return None
    payload = verify_token(token)
    if not payload:
        return None
    username = payload.get("sub")
    if not username:
        return None
    user = get_user_by_username(db, username)
    return user


def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db),
) -> User:
    user = get_current_user_optional(token, db)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user


# ---------------------------------------------------------------------------
# Authentication endpoints
# ---------------------------------------------------------------------------

@app.post("/signup", response_model=Token)
def signup(user_data: UserCreate, db: Session = Depends(get_db)):
    existing = get_user_by_username(db, user_data.username)
    if existing:
        raise HTTPException(status_code=400, detail="Username already taken")
    user = create_user(
        db,
        username=user_data.username,
        password=user_data.password,
        first_name=user_data.first_name,
        last_name=user_data.last_name
    )
    if not user:
        raise HTTPException(status_code=500, detail="Failed to create user")
    access_token = create_access_token(data={"sub": user.username})
    return {"access_token": access_token, "token_type": "bearer"}


@app.post("/login", response_model=Token)
def login(login_data: LoginRequest, db: Session = Depends(get_db)):
    user = authenticate_user(db, login_data.username, login_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token = create_access_token(data={"sub": user.username})
    return {"access_token": access_token, "token_type": "bearer"}

@app.post("/tone")
def get_tone_endpoint(
    request: ToneRequest,
    current_user: User | None = Depends(get_current_user_optional),
):
    tone = llm.get_last_tone(request.history, request.current_message)
    return {"label": tone.label, "confidence": tone.confidence}


# ---------------------------------------------------------------------------
# Chat endpoints – hybrid storage
# ---------------------------------------------------------------------------

def _get_storage_for_user(user: User | None):
    """Return (storage_module, user_id) for authenticated or None for anonymous."""
    if user:
        return ("db", user.id)
    return ("file", None)


def _load_state(user: User | None, session_id: str, db: Session | None = None):
    if user:
        # DB
        state = get_session_state(db, user.id, session_id)
        return state
    else:
        # File
        state = storage.load_history(session_id)
        return state


def _save_state(user: User | None, session_id: str, state: dict, db: Session | None = None):
    if user:
        create_or_update_session(
            db,
            user.id,
            session_id,
            title=state.get("title", "New Chat"),
            messages=state.get("messages", []),
            tone_label=state.get("tone_label", "neutral"),
            tone_confidence=state.get("tone_confidence", 0.0),
        )
    else:
        storage.save_history(state, session_id)


def _delete_state(user: User | None, session_id: str, db: Session | None = None):
    if user:
        delete_session(db, user.id, session_id)
    else:
        storage.clear_history(session_id)


def _list_sessions(user: User | None, db: Session | None = None) -> list[dict]:
    if user:
        sessions = list_user_sessions(db, user.id)
        result = []
        for s in sessions:
            preview = ""
            if s.messages:
                for msg in s.messages:
                    if msg.get("role") == "user" and msg.get("content"):
                        preview = msg["content"][:48] + ("..." if len(msg["content"]) > 48 else "")
                        break
                if not preview:
                    preview = "(empty)"
            result.append({
                "sid": s.session_id,
                "title": s.title,
                "preview": preview,
                "message_count": len(s.messages),
                "tone_label": s.tone_label,
                "updated_at": s.updated_at.timestamp(),
            })
        return result
    else:
        return storage.list_sessions()


# ---------------------------------------------------------------------------
# API endpoints – now using hybrid storage
# ---------------------------------------------------------------------------

@app.post("/chat/stream")
async def chat_stream(
    request: ChatRequest,
    current_user: User | None = Depends(get_current_user_optional),
    db: Session = Depends(get_db),
):
    # Ensure session_id exists
    session_id = request.session_id or str(uuid.uuid4())

    # The streaming generator
    async def event_generator() -> AsyncGenerator[str, None]:
        provider_info: dict[str, str] = {"provider": "placeholder"}
        try:
            stream = llm.generate_reply_stream(
                user_message=request.message,
                history=request.history,
                result_info=provider_info,
            )
            # Yield chunks
            for chunk in stream:
                yield json.dumps({"chunk": chunk, "provider": provider_info.get("provider", "unknown")}) + "\n"
            yield json.dumps({"done": True}) + "\n"

            # After streaming, update session with new history
            # The client will send updated history via save endpoint.
            # But we can optionally auto-save; we'll rely on client.

        except Exception as e:
            yield json.dumps({"error": str(e)}) + "\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.get("/sessions")
def list_sessions_endpoint(
    current_user: User | None = Depends(get_current_user_optional),
    db: Session = Depends(get_db),
):
    return _list_sessions(current_user, db)


@app.get("/session/{session_id}")
def load_session_endpoint(
    session_id: str,
    current_user: User | None = Depends(get_current_user_optional),
    db: Session = Depends(get_db),
):
    state = _load_state(current_user, session_id, db)
    return state


@app.post("/session/{session_id}")
def save_session_endpoint(
    session_id: str,
    update: SessionUpdate,
    current_user: User | None = Depends(get_current_user_optional),
    db: Session = Depends(get_db),
):
    state = {
        "messages": update.messages,
        "tone_label": update.tone_label,
        "tone_confidence": update.tone_confidence,
        "title": update.title,
    }
    _save_state(current_user, session_id, state, db)
    return {"status": "ok", "session_id": session_id}


@app.delete("/session/{session_id}")
def delete_session_endpoint(
    session_id: str,
    current_user: User | None = Depends(get_current_user_optional),
    db: Session = Depends(get_db),
):
    _delete_state(current_user, session_id, db)
    return {"status": "ok", "session_id": session_id}


@app.post("/title")
def generate_title_endpoint(
    request: TitleRequest,
    current_user: User | None = Depends(get_current_user_optional),
):
    title = llm.generate_title(request.history)
    return {"title": title}


@app.get("/config")
def get_config(current_user: User | None = Depends(get_current_user_optional)):
    return llm.get_config()


@app.get("/me")
def get_me(current_user: User | None = Depends(get_current_user_optional)):
    if current_user:
        return {"authenticated": True, "username": current_user.username}
    return {"authenticated": False}

# Serve static frontend
app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")