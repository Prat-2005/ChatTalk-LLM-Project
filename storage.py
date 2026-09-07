"""On-disk persistence for ChatTalk chat history.

Each Streamlit session gets its own JSON file under `CHATTALK_DATA_DIR`
(default `.chattalk_data/`). Sessions are keyed by `st.session_state`'s
session id; the file format is intentionally trivial so it can be diffed,
grepped, or hand-edited.

We only persist:
  * the message list (role + content)
  * the last detected tone label + confidence

We do NOT persist secrets, model state, or anything else.
"""

from __future__ import annotations

import json
import os
import tempfile
import uuid
from pathlib import Path
from typing import Any


def data_dir() -> Path:
    """Return (and create) the directory used for chat history files."""
    raw = os.environ.get("CHATTALK_DATA_DIR", ".chattalk_data")
    path = Path(raw).expanduser().resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def session_id() -> str:
    """Stable id for the current Streamlit session.

    Streamlit exposes `st.runtime.scriptrunner.get_script_run_ctx().session_id`
    at runtime, but importing Streamlit at module import time is heavy and
    pulls in extra deps for unit tests. We fall back to a random id stored
    in the file on first write.
    """
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx  # type: ignore

        ctx = get_script_run_ctx()
        if ctx is not None:
            return ctx.session_id
    except Exception:
        pass
    return os.environ.get("CHATTALK_SESSION_ID") or uuid.uuid4().hex


def _session_path(sid: str) -> Path:
    safe = "".join(c for c in sid if c.isalnum() or c in "-_") or "default"
    return data_dir() / f"session_{safe}.json"


def _session_preview(messages: list[dict[str, Any]]) -> str:
    # Priority 1: Find the first user message with conten
    for msg in messages:
        if msg.get("role") == "user" and msg.get("content"):
            text = str(msg["content"]).strip().replace("\n", " ")
            return text[:48] + ("..." if len(text) > 48 else "")

    # Priority 2: Fallback to the first message with any content
    for msg in messages:
        if msg.get("content"):
            text = str(msg["content"]).strip().replace("\n", " ")
            return text[:48] + ("..." if len(text) > 48 else "")
    return "Empty session"


def load_history(sid: str = session_id()) -> dict[str, Any]:  # args: sid otherwise default to current session 'session_id()'
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


def save_history(state: dict[str, Any], sid: str = session_id()) -> None:
    """Atomically write the given state to the session file.

    Atomic write = write to a temp file in the same directory, then rename.
    This avoids leaving a half-written file if the process is killed.
    """
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
            tmp.write(json.dumps(payload, ensure_ascii=False, indent=2))
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
                "path": path,
                "updated_at": stat.st_mtime,
                "message_count": len(messages),
                "tone_label": loaded.get("tone_label", "neutral"),
                "preview": _session_preview(messages),
                "title": loaded.get("title", "New Chat"),
            }
        )
    sessions.sort(key=lambda item: item["updated_at"], reverse=True)
    return sessions


def clear_history(sid: str = session_id()) -> None:
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
    "session_id",
]
