"""Tests for prompts.py — tone detection, system prompt, history trimming."""

from __future__ import annotations

import sys
from pathlib import Path

# Make the project root importable when running pytest from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from prompts import (  # noqa: E402
    DEFAULT_TONE,
    build_messages,
    build_system_prompt,
    detect_tone,
    trim_history,
)


# ---------------------------------------------------------------------------
# detect_tone
# ---------------------------------------------------------------------------

class TestDetectTone:
    def test_empty_input_returns_default(self):
        tone = detect_tone([])
        assert tone.label == DEFAULT_TONE
        assert tone.confidence == 0.0

    def test_excited_tone(self):
        tone = detect_tone(["OMG this is amazing!! 🎉"])
        assert tone.label == "excited"

    def test_playful_tone(self):
        tone = detect_tone(["lol that's hilarious 😂"])
        assert tone.label == "playful"

    def test_sad_tone(self):
        tone = detect_tone(["I'm feeling really lonely and sad tonight"])
        assert tone.label == "sad"

    def test_angry_tone_caps(self):
        tone = detect_tone(["THIS IS SO ANNOYING I HATE IT"])
        assert tone.label == "angry"

    def test_serious_tone(self):
        tone = detect_tone(["This is important and I'm worried about the deadline"])
        assert tone.label in ("serious", "angry")  # "worried" + "important"

    def test_flirtatious_tone(self):
        tone = detect_tone(["hey cutie, want to grab a date sometime? 😘"])
        assert tone.label == "flirtatious"

    def test_first_message_is_sticky(self):
        """A first-message flirtatious cue should survive a follow-up that
        would otherwise be neutral."""
        history_msgs = [
            "you're such a cutie 😘",
            "ok so anyway, what time is it?",
        ]
        tone = detect_tone(history_msgs)
        assert tone.label == "flirtatious"

    def test_confidence_in_range(self):
        tone = detect_tone(["wow this is amazing!"])
        assert 0.0 <= tone.confidence <= 1.0

    def test_reasons_populated_for_known_tone(self):
        tone = detect_tone(["omg yay 🔥"])
        assert tone.label == "excited"
        assert tone.reasons, "expected at least one reason for an excited match"

    def test_neutral_when_no_signals(self):
        tone = detect_tone(["the weather is mild today"])
        assert tone.label in (DEFAULT_TONE, "calm", "neutral", "serious")

    def test_multiple_emojis_stack(self):
        tone = detect_tone(["😘😍❤️"])
        assert tone.label == "flirtatious"

    def test_recent_message_can_shift_tone(self):
        """A clearly-sad latest message should beat a flirty opening."""
        history = [
            "you're such a cutie 😘",
            "haha anyway how's it going",
            "honestly I've been really sad and lonely lately",
        ]
        tone = detect_tone(history)
        assert tone.label == "sad"

    def test_neutral_when_messages_are_just_punctuation(self):
        tone = detect_tone(["...", "???", "—"])
        # Should not crash; may land on default or some weak signal.
        assert tone.label in (
            "neutral", DEFAULT_TONE, "serious", "angry", "calm",
        )


# ---------------------------------------------------------------------------
# build_system_prompt
# ---------------------------------------------------------------------------

class TestBuildSystemPrompt:
    def test_contains_persona(self):
        prompt = build_system_prompt(detect_tone(["hello"]))
        assert "ChatTalk" in prompt
        assert "friendly" in prompt.lower() or "companion" in prompt.lower()

    def test_includes_tone_label_when_confident(self):
        from prompts import ToneSignal
        tone = ToneSignal(label="playful", confidence=0.9, reasons=["x"])
        prompt = build_system_prompt(tone)
        assert "playful" in prompt.lower()

    def test_no_tone_phrase_when_neutral_low_confidence(self):
        from prompts import ToneSignal
        tone = ToneSignal(label=DEFAULT_TONE, confidence=0.0, reasons=[])
        prompt = build_system_prompt(tone)
        # Should still contain tone guidance, just not a "dominant tone is X" line.
        assert "dominant tone" not in prompt.lower()

    def test_includes_style_mirror_rule(self):
        prompt = build_system_prompt(detect_tone(["hi"]))
        assert "mirror" in prompt.lower()


# ---------------------------------------------------------------------------
# trim_history
# ---------------------------------------------------------------------------

class TestTrimHistory:
    def test_keeps_short_history_unchanged(self):
        history = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello!"},
        ]
        assert trim_history(history, char_budget=1000) == history

    def test_drops_oldest_when_over_budget(self):
        history = [
            {"role": "user", "content": "a" * 500},
            {"role": "assistant", "content": "b" * 500},
            {"role": "user", "content": "c" * 500},
            {"role": "assistant", "content": "d" * 500},
        ]
        kept = trim_history(history, char_budget=600)
        # Total chars in kept must fit in budget.
        assert sum(len(m["content"]) for m in kept) <= 600 + len(kept)
        # Newest messages win.
        assert kept[-1]["content"].startswith("d")

    def test_empty_history(self):
        assert trim_history([]) == []

    def test_handles_missing_content(self):
        history = [
            {"role": "user", "content": ""},
            {"role": "assistant", "content": "hi"},
        ]
        kept = trim_history(history, char_budget=1000)
        assert len(kept) == 2


# ---------------------------------------------------------------------------
# build_messages
# ---------------------------------------------------------------------------

class TestBuildMessages:
    def test_structure(self):
        msgs = build_messages(
            "hello there",
            history=[
                {"role": "user", "content": "earlier"},
                {"role": "assistant", "content": "earlier reply"},
            ],
        )
        assert msgs[0]["role"] == "system"
        assert msgs[-1] == {"role": "user", "content": "hello there"}
        # System prompt mentions tone guidance
        assert "tone" in msgs[0]["content"].lower()

    def test_includes_recent_history(self):
        msgs = build_messages(
            "now",
            history=[
                {"role": "user", "content": "earlier user msg"},
                {"role": "assistant", "content": "earlier assistant msg"},
            ],
        )
        roles = [m["role"] for m in msgs]
        assert "user" in roles and "assistant" in roles
        assert "earlier user msg" in [m["content"] for m in msgs]
        assert "earlier assistant msg" in [m["content"] for m in msgs]

    def test_system_prompt_contains_guard_rails(self):
        msgs = build_messages("hi", history=[])
        sys = msgs[0]["content"]
        assert "hard rules" in sys.lower() or "never break character" in sys.lower()

    def test_casual_user_triggers_slang_mirror_hint(self):
        msgs = build_messages(
            "lol idk rly",
            history=[{"role": "user", "content": "haha tbh idk bro"}],
        )
        sys = msgs[0]["content"]
        # Slang density > 0.2 in recent messages should mention casual writing.
        assert "casual" in sys.lower() or "slang" in sys.lower() or "relax" in sys.lower()

    def test_length_guidance_appears_in_system(self):
        msgs = build_messages("hi", history=[])
        assert "length" in msgs[0]["content"].lower()
