"""PostgreSQL storage for authenticated users."""

from __future__ import annotations

import os
from datetime import datetime
from typing import Any

from sqlalchemy import create_engine, Column, Integer, String, DateTime, JSON, Float, ForeignKey, UniqueConstraint
from sqlalchemy.orm import declarative_base, sessionmaker, Session, relationship
from sqlalchemy.exc import IntegrityError
from passlib.context import CryptContext

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql://user:pass@localhost/chattalk")
SECRET_KEY = os.environ.get("SECRET_KEY", "dev-secret-key-change-me")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24 * 7  # 7 days

pwd_context = CryptContext(
    schemes=["bcrypt"],
    bcrypt__rounds=12,
    bcrypt__truncate_error=False,  # let bcrypt handle truncation
    deprecated="auto",
)

# ---------------------------------------------------------------------------
# SQLAlchemy
# ---------------------------------------------------------------------------
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True, nullable=False)
    first_name = Column(String, nullable=False)   # new
    last_name = Column(String, nullable=False)    # new
    hashed_password = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    sessions = relationship("ChatSession", back_populates="user", cascade="all, delete-orphan")

class ChatSession(Base):
    __tablename__ = "chat_sessions"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    session_id = Column(String, nullable=False)
    title = Column(String, default="New Chat")
    messages = Column(JSON, default=list)
    tone_label = Column(String, default="neutral")
    tone_confidence = Column(Float, default=0.0)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    user = relationship("User", back_populates="sessions")

    __table_args__ = (UniqueConstraint("user_id", "session_id"),)


# ---------------------------------------------------------------------------
# Password hashing
# ---------------------------------------------------------------------------
def hash_password(password: str) -> str:
    # Explicitly truncate to 72 bytes (bcrypt limit)
    if len(password.encode('utf-8')) > 72:
        password = password[:72]
    return pwd_context.hash(password)

def verify_password(plain: str, hashed: str) -> bool:
    # Truncate to match hash
    if len(plain.encode('utf-8')) > 72:
        plain = plain[:72]
    return pwd_context.verify(plain, hashed)


# ---------------------------------------------------------------------------
# User CRUD
# ---------------------------------------------------------------------------
def get_user_by_username(db: Session, username: str) -> User | None:
    return db.query(User).filter(User.username == username).first()


def create_user(db: Session, username: str, password: str, first_name: str = None, last_name: str = None) -> User | None:
    hashed = hash_password(password)
    user = User(
        username=username,
        first_name=first_name,
        last_name=last_name,
        hashed_password=hashed
    )
    try:
        db.add(user)
        db.commit()
        db.refresh(user)
        return user
    except IntegrityError:
        db.rollback()
        return None


def authenticate_user(db: Session, username: str, password: str) -> User | None:
    user = get_user_by_username(db, username)
    if not user:
        return None
    if not verify_password(password, user.hashed_password):
        return None
    return user


# ---------------------------------------------------------------------------
# Session CRUD (authenticated)
# ---------------------------------------------------------------------------
def get_session(db: Session, user_id: int, session_id: str) -> ChatSession | None:
    return db.query(ChatSession).filter(
        ChatSession.user_id == user_id,
        ChatSession.session_id == session_id
    ).first()


def list_user_sessions(db: Session, user_id: int) -> list[ChatSession]:
    return db.query(ChatSession).filter(ChatSession.user_id == user_id).order_by(
        ChatSession.updated_at.desc()
    ).all()


def create_or_update_session(
    db: Session,
    user_id: int,
    session_id: str,
    title: str = "New Chat",
    messages: list[dict] = None,
    tone_label: str = "neutral",
    tone_confidence: float = 0.0,
) -> ChatSession:
    session = get_session(db, user_id, session_id)
    if session:
        if messages is not None:
            session.messages = messages
        session.title = title
        session.tone_label = tone_label
        session.tone_confidence = tone_confidence
        session.updated_at = datetime.utcnow()
    else:
        session = ChatSession(
            user_id=user_id,
            session_id=session_id,
            title=title,
            messages=messages or [],
            tone_label=tone_label,
            tone_confidence=tone_confidence,
        )
        db.add(session)
    db.commit()
    db.refresh(session)
    return session


def delete_session(db: Session, user_id: int, session_id: str) -> bool:
    session = get_session(db, user_id, session_id)
    if not session:
        return False
    db.delete(session)
    db.commit()
    return True


def get_session_state(db: Session, user_id: int, session_id: str) -> dict[str, Any]:
    session = get_session(db, user_id, session_id)
    if not session:
        return {"messages": [], "tone_label": "neutral", "tone_confidence": 0.0, "title": "New Chat"}
    return {
        "messages": session.messages or [],
        "tone_label": session.tone_label or "neutral",
        "tone_confidence": session.tone_confidence or 0.0,
        "title": session.title or "New Chat",
    }


# Create tables if they don't exist (optional)
Base.metadata.create_all(bind=engine)