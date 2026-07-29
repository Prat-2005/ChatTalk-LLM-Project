"""Tests for storage.py — on-disk chat history persistence."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import storage  # noqa: E402


@pytest.fixture()
def tmp_data_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("CHATTALK_DATA_DIR", str(tmp_path))
    return tmp_path


class TestDataDir:
    def test_creates_directory(self, tmp_data_dir):
        path = storage.data_dir()
        assert path.is_dir()
        assert path == tmp_data_dir.resolve()

    def test_nested_directory_created(self, tmp_path, monkeypatch):
        nested = tmp_path / "deep" / "nested"
        monkeypatch.setenv("CHATTALK_DATA_DIR", str(nested))
        path = storage.data_dir()
        assert path.is_dir()


class TestSessionPath:
    def test_safe_filename(self, tmp_data_dir):
        # Non-alphanumeric chars get stripped; the result must be safe
        # to use as a filename on every platform.
        path = storage._session_path("abc/123:weird?")
        name = path.name
        assert name.startswith("session_")
        assert name.endswith(".json")
        assert all(c.isalnum() or c in "._-" for c in name)

    def test_empty_session_id_falls_back(self, tmp_data_dir):
        # No alphanumerics at all → falls back to "default"
        path = storage._session_path("///")
        assert path.name == "session_default.json"


class TestLoadSave:
    def test_load_returns_blank_when_missing(self, tmp_data_dir):
        state = storage.load_history(sid="nope")
        assert state == {
            "messages": [],
            "tone_label": "neutral",
            "tone_confidence": 0.0,
        }

    def test_save_then_load_roundtrip(self, tmp_data_dir):
        state = {
            "messages": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "hello!"},
            ],
            "tone_label": "playful",
            "tone_confidence": 0.7,
        }
        storage.save_history(state, sid="alpha")
        loaded = storage.load_history(sid="alpha")
        assert loaded == state

    def test_save_is_atomic(self, tmp_data_dir):
        # Write a real file, then re-save. There should be no leftover
        # .tmp files in the directory.
        state = {"messages": [{"role": "user", "content": "x"}], "tone_label": "neutral", "tone_confidence": 0.0}
        storage.save_history(state, sid="beta")
        storage.save_history({**state, "messages": state["messages"] + [{"role": "assistant", "content": "y"}]}, sid="beta")
        leftover = list(tmp_data_dir.glob(".session_*.tmp"))
        assert leftover == []
        loaded = storage.load_history(sid="beta")
        assert len(loaded["messages"]) == 2

    def test_load_handles_corrupt_file(self, tmp_data_dir):
        bad = tmp_data_dir / "session_corrupt.json"
        bad.write_text("not valid json {{{", encoding="utf-8")
        loaded = storage.load_history(sid="corrupt")
        assert loaded["messages"] == []

    def test_clear_history(self, tmp_data_dir):
        state = {"messages": [{"role": "user", "content": "x"}], "tone_label": "neutral", "tone_confidence": 0.0}
        storage.save_history(state, sid="gamma")
        assert storage.load_history(sid="gamma")["messages"]
        storage.clear_history(sid="gamma")
        assert storage.load_history(sid="gamma")["messages"] == []

    def test_clear_history_missing_ok(self, tmp_data_dir):
        # Should not raise even when there's no file.
        storage.clear_history(sid="never-existed")

    def test_list_sessions_orders_newest_first(self, tmp_data_dir):
        storage.save_history(
            {
                "messages": [{"role": "user", "content": "first session"}],
                "tone_label": "neutral",
                "tone_confidence": 0.0,
            },
            sid="alpha",
        )
        import time; time.sleep(0.02)
        storage.save_history(
            {
                "messages": [{"role": "user", "content": "second session"}],
                "tone_label": "playful",
                "tone_confidence": 0.5,
            },
            sid="beta",
        )

        sessions = storage.list_sessions()

        assert [item["sid"] for item in sessions] == ["beta", "alpha"]
        assert sessions[0]["preview"].startswith("second session")
        assert sessions[1]["preview"].startswith("first session")


class TestSessionId:
    def test_returns_string(self, tmp_data_dir, monkeypatch):
        monkeypatch.delenv("CHATTALK_SESSION_ID", raising=False)
        sid = storage.session_id()
        assert isinstance(sid, str)
        assert sid
