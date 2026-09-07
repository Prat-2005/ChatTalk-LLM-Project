"""On-disk persistence for ChatTalk chat history.

Each session is stored as a JSON file under `CHATTALK_DATA_DIR`.
The session ID is provided by the caller – no assumption about Streamlit.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any


def data_dir() -> Path:
    """Return (and create) the directory used for chat history files."""
    raw = os.environ.get("CHATTALK_DATA_DIR", ".chattalk_data")
    path = Path(raw).expanduser().resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def _session_path(sid: str) -> Path:
    safe = "".join(c for c in sid if c.isalnum() or c in "-_") or "default"
    return data_dir() / f"session_{safe}.json"


def _session_preview(messages: list[dict[str, Any]]) -> str:
    # Priority 1: First user message with content
    for msg in messages:
        if msg.get("role") == "user" and msg.get("content"):
            text = str(msg["content"]).strip().replace("\n", " ")
            return text[:48] + ("..." if len(text) > 48 else "")
    # Priority 2: Any message with content
    for msg in messages:
        if msg.get("content"):
            text = str(msg["content"]).strip().replace("\n", " ")
            return text[:48] + ("..." if len(text) > 48 else "")
    return "Empty session"


def load_history(sid: str) -> dict[str, Any]:
    """Return persisted state for the given session, or a blank state."""
    path = _session_path(sid)
    default_state = {"messages": [], "tone_label": "neutral", "tone_confidence": 0.0, "title": "New Chat"}
    if not path.is_file():
        return default_state
    try:
        text = path.read_text(encoding="utf-8")
        file_data = json.loads(text)
    except (OSError, json.JSONDecodeError):
        return default_state
    if not isinstance(file_data, dict):
        return default_state
    return default_state | file_data  # type: ignore


def save_history(state: dict[str, Any], sid: str) -> None:
    """Atomically write the given state to the session file."""
    path = _session_path(sid)
    payload = {
        "messages": list(state.get("messages", [])),
        "tone_label": state.get("tone_label", "neutral"),
        "tone_confidence": float(state.get("tone_confidence", 0.0)),
        "title": state.get("title", "New Chat"),
    }
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            delete=False,
            dir=str(path.parent),
            prefix=".session_",
            suffix=".tmp",
        ) as tmp:
            json.dump(payload, tmp, ensure_ascii=False, indent=2)
            tmp_path = Path(tmp.name)
        tmp_path.replace(path)
    except OSError:
        # Persistence is best-effort; never crash the chat because of disk IO.
        pass


def list_sessions() -> list[dict[str, Any]]:
    """Return all persisted sessions, newest first."""
    sessions: list[dict[str, Any]] = []
    for path in data_dir().glob("session_*.json"):
        sid = path.stem.removeprefix("session_")
        try:
            loaded = load_history(sid)
            stat = path.stat()
        except OSError:
            continue
        messages = loaded.get("messages", [])
        sessions.append(
            {
                "sid": sid,
                "path": str(path),
                "updated_at": stat.st_mtime,
                "message_count": len(messages),
                "tone_label": loaded.get("tone_label", "neutral"),
                "preview": _session_preview(messages),
                "title": loaded.get("title", "New Chat"),
            }
        )
    sessions.sort(key=lambda item: item["updated_at"], reverse=True)
    return sessions


def clear_history(sid: str) -> None:
    """Delete the persisted state for the given session."""
    path = _session_path(sid)
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass


__all__ = [
    "clear_history",
    "data_dir",
    "list_sessions",
    "load_history",
    "save_history",
]