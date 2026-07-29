"""ChatTalk — Streamlit chat UI (Vibrant & Interactive).

A completely rewritten stunning, dynamic, real-time streaming chat interface
with beautiful dark-first styling, glassmorphism, animations, and native Streamlit chat elements.
"""

from __future__ import annotations

import html
import uuid

import streamlit as st

from llm import generate_reply_stream, get_config, get_last_tone
from prompts import DEFAULT_TONE
import storage


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="ChatTalk",
    page_icon="💬",
    layout="centered",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# Styles — animations, vibrant theme, chat bubbles
# ---------------------------------------------------------------------------

_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

/* ============================================================
   Keyframes for Animations
   ============================================================ */
@keyframes messageSlideIn {
    0% { transform: translateY(16px); opacity: 0; }
    100% { transform: translateY(0); opacity: 1; }
}

@keyframes pulseGlow {
    0% { transform: scale(1); opacity: 0.8; box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.4); }
    50% { transform: scale(1.3); opacity: 1; box-shadow: 0 0 0 4px rgba(16, 185, 129, 0); }
    100% { transform: scale(1); opacity: 0.8; box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }
}

@keyframes typingBounce {
    0%, 60%, 100% { transform: translateY(0); }
    30% { transform: translateY(-4px); }
}

@keyframes fadeIn {
    from { opacity: 0; }
    to { opacity: 1; }
}

@keyframes gradientShift {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

@keyframes typingBounce {
    0%, 80%, 100% { transform: scale(0.3); opacity: 0.3; }
    40% { transform: scale(1); opacity: 1; }
}

/* ============================================================
   Dark Mode Default (Vibrant & Interactive)
   ============================================================ */
:root {
    --bg-page:        #0b0d17;
    --surface:        #13152a;
    --surface-elev:   #1a1d35;
    --border:         #252845;
    --text-primary:   #e8e6f0;
    --text-muted:     #8b87a8;
    --text-dim:       #5e5b75;
    --accent:         #6366f1;
    --accent-sec:     #a78bfa;
    --accent-glow:    rgba(99, 102, 241, 0.15);
    
    --user-bubble:    linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
    --user-shadow:    0 4px 14px rgba(99, 102, 241, 0.25);
    
    --assistant-bg:   #1a1d35;
    --assistant-bdr:  #252845;
    
    --status-live:    #10b981;
    --status-fb:      #3b82f6;
    --status-ph:      #f59e0b;
    
    --glass-bg:       rgba(19, 21, 42, 0.65);
    --glass-border:   rgba(255, 255, 255, 0.08);
}

/* ============================================================
   Light Mode Override
   ============================================================ */
@media (prefers-color-scheme: light) {
    :root {
        --bg-page:        #f0eeff;
        --surface:        #ffffff;
        --surface-elev:   #f7f6fc;
        --border:         #e5e1f5;
        --text-primary:   #1a1733;
        --text-muted:     #6b6586;
        --text-dim:       #a09db0;
        --accent:         #6366f1;
        --accent-sec:     #a78bfa;
        --accent-glow:    rgba(99, 102, 241, 0.1);
        
        --user-bubble:    linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
        --user-shadow:    0 4px 12px rgba(99, 102, 241, 0.2);
        
        --assistant-bg:   #ffffff;
        --assistant-bdr:  #e5e1f5;
        
        --status-live:    #059669;
        --status-fb:      #2563eb;
        --status-ph:      #d97706;
        
        --glass-bg:       rgba(255, 255, 255, 0.7);
        --glass-border:   rgba(99, 102, 241, 0.15);
    }
}

/* ============================================================
   Base Page Chrome & Seamless Full-Screen Background
   ============================================================ */
html, body, .stApp, [data-testid="stAppViewContainer"], 
section.main, .main, [data-testid="stMain"], [data-testid="stMainBlockContainer"],
[data-testid="stBottom"], [data-testid="stBottom"] > div,
.stMainBlockContainer {
    background-color: #0b0d17 !important;
    background-image: 
        radial-gradient(circle at 15% 50%, var(--accent-glow) 0%, transparent 50%),
        radial-gradient(circle at 85% 30%, rgba(167, 139, 250, 0.08) 0%, transparent 50%) !important;
    background-attachment: fixed !important;
    color: var(--text-primary);
    font-family: "Inter", sans-serif;
}

[data-testid="stAppViewContainer"] > section.main {
    min-height: 100vh !important;
}

/* Ensure Streamlit Header & Sidebar Toggle Button are ALWAYS Visible & Clickable */
[data-testid="stHeader"] {
    background: transparent !important;
    box-shadow: none !important;
    border: none !important;
    z-index: 99999 !important;
}

[data-testid="stSidebarCollapseButton"],
button[data-testid="stSidebarCollapseButton"],
button[aria-label="Expand sidebar"],
button[aria-label="Collapse sidebar"],
button[data-testid="baseButton-header"],
[data-testid="stHeader"] button {
    visibility: visible !important;
    display: flex !important;
    opacity: 1 !important;
    color: #ffffff !important;
    background: #1a1d35 !important;
    border: 1px solid #252845 !important;
    border-radius: 10px !important;
    z-index: 999999 !important;
    box-shadow: 0 4px 12px rgba(0,0,0,0.3) !important;
    transition: all 0.2s ease !important;
}

[data-testid="stSidebarCollapseButton"]:hover,
button[aria-label="Expand sidebar"]:hover,
button[aria-label="Collapse sidebar"]:hover,
button[data-testid="baseButton-header"]:hover {
    background: #6366f1 !important;
    color: #ffffff !important;
    border-color: #6366f1 !important;
    transform: scale(1.05) !important;
}

.block-container {
    max-width: 820px;
    padding-top: 1.5rem;
    padding-bottom: 7rem;
}

h1, h2, h3, h4, p, span, div, li, label {
    color: var(--text-primary);
}

/* Hide Streamlit default top right menu & footer but keep header buttons */
#MainMenu, footer, [data-testid="stToolbar"] {
    visibility: hidden;
}

/* Scrollbars */
::-webkit-scrollbar {
    width: 6px;
    height: 6px;
}
::-webkit-scrollbar-thumb {
    background: var(--text-dim);
    border-radius: 999px;
}
::-webkit-scrollbar-thumb:hover {
    background: var(--accent);
}
::-webkit-scrollbar-track {
    background: transparent;
}

/* ============================================================
   WhatsApp & Instagram DM Header Bar
   ============================================================ */
.chat-app-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0.85rem 1.4rem;
    border-radius: 20px;
    background: rgba(19, 21, 42, 0.75);
    border: 1px solid rgba(255, 255, 255, 0.08);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.25);
    margin-bottom: 1.5rem;
    animation: fadeIn 0.6s ease-out;
}

.chat-app-user {
    display: flex;
    align-items: center;
    gap: 0.9rem;
}

.avatar-badge-wrap {
    position: relative;
    width: 44px;
    height: 44px;
    border-radius: 50%;
    background: linear-gradient(135deg, #6366f1 0%, #a855f7 100%);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.4rem;
    box-shadow: 0 4px 14px rgba(99, 102, 241, 0.4);
}

.online-dot {
    position: absolute;
    bottom: 1px;
    right: 1px;
    width: 11px;
    height: 11px;
    border-radius: 50%;
    background: #10b981;
    border: 2px solid #0b0d17;
    animation: pulseGlow 2.5s infinite;
}

.chat-app-title {
    font-weight: 800;
    font-size: 1.15rem;
    color: #ffffff;
    display: flex;
    align-items: center;
    gap: 0.35rem;
    margin: 0;
}

.verified-icon {
    color: #6366f1;
    font-size: 0.9rem;
}

.chat-app-status {
    font-size: 0.8rem;
    color: #8b87a8;
    margin-top: 0.1rem;
}

/* ============================================================
   WhatsApp & Instagram DM Bubble Aesthetics
   ============================================================ */
[data-testid="stChatMessage"] {
    background: transparent !important;
    animation: messageSlideIn 0.35s cubic-bezier(0.16, 1, 0.3, 1) forwards;
    padding: 0.4rem 0 !important;
    margin-bottom: 0.4rem !important;
    display: flex !important;
    width: 100% !important;
}

[data-testid="stChatMessage"] [data-testid="chatAvatarIcon-assistant"],
[data-testid="stChatMessage"] [data-testid="chatAvatarIcon-user"],
[data-testid="stChatMessage"] [data-testid*="ChatMessageAvatar"] {
    background: transparent !important;
    box-shadow: none !important;
    font-size: 1.5rem !important;
}

/* User Message (Sent Message - Right Aligned) */
[data-testid="stChatMessage"]:has([data-testid*="user"]),
[data-testid="stChatMessage"]:has([aria-label*="user"]),
[data-testid="stChatMessage"]:has(span:contains("🧑")),
[data-testid="stChatMessage"]:nth-child(odd) {
    flex-direction: row-reverse !important;
    justify-content: flex-start !important;
}

[data-testid="stChatMessage"]:has([data-testid*="user"]) [data-testid="stMarkdownContainer"],
[data-testid="stChatMessage"]:has([aria-label*="user"]) [data-testid="stMarkdownContainer"],
[data-testid="stChatMessage"]:nth-child(odd) [data-testid="stMarkdownContainer"] {
    background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #d946ef 100%) !important;
    border: none !important;
    border-radius: 20px 20px 4px 20px !important;
    padding: 0.75rem 1.15rem !important;
    color: #ffffff !important;
    box-shadow: 0 4px 15px rgba(124, 58, 237, 0.3) !important;
    max-width: 74% !important;
    display: inline-block !important;
    text-align: left !important;
}

[data-testid="stChatMessage"]:has([data-testid*="user"]) [data-testid="stMarkdownContainer"] p,
[data-testid="stChatMessage"]:nth-child(odd) [data-testid="stMarkdownContainer"] p {
    color: #ffffff !important;
    margin: 0 !important;
    line-height: 1.55 !important;
    font-size: 0.95rem !important;
}

/* Assistant Message (Received Message - Left Aligned) */
[data-testid="stChatMessage"]:has([data-testid*="assistant"]),
[data-testid="stChatMessage"]:has([aria-label*="assistant"]),
[data-testid="stChatMessage"]:has(span:contains("💬")),
[data-testid="stChatMessage"]:nth-child(even) {
    flex-direction: row !important;
    justify-content: flex-start !important;
}

[data-testid="stChatMessage"]:has([data-testid*="assistant"]) [data-testid="stMarkdownContainer"],
[data-testid="stChatMessage"]:has([aria-label*="assistant"]) [data-testid="stMarkdownContainer"],
[data-testid="stChatMessage"]:nth-child(even) [data-testid="stMarkdownContainer"] {
    background: #181b30 !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-left: 3px solid #a855f7 !important;
    border-radius: 20px 20px 20px 4px !important;
    padding: 0.85rem 1.25rem !important;
    color: #e8e6f0 !important;
    box-shadow: 0 4px 18px rgba(0, 0, 0, 0.25) !important;
    max-width: 78% !important;
    display: inline-block !important;
}

[data-testid="stChatMessage"]:has([data-testid*="assistant"]) [data-testid="stMarkdownContainer"] p,
[data-testid="stChatMessage"]:nth-child(even) [data-testid="stMarkdownContainer"] p {
    color: #e8e6f0 !important;
    margin: 0 !important;
    line-height: 1.55 !important;
    font-size: 0.95rem !important;
}

/* ============================================================
   Custom UI Elements (Hero, Status, Tones)
   ============================================================ */

.hero-shell {
    padding: 1.5rem;
    border: 1px solid var(--glass-border);
    border-radius: 20px;
    background: var(--glass-bg);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
    margin-bottom: 2rem;
    animation: fadeIn 0.8s ease-out;
}

.hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    padding: 0.35rem 0.8rem;
    border-radius: 999px;
    font-size: 0.75rem;
    font-weight: 700;
    color: var(--accent);
    background: var(--accent-glow);
    border: 1px solid rgba(99, 102, 241, 0.2);
    margin-bottom: 1rem;
}

.hero-title {
    font-size: 1.7rem;
    font-weight: 800;
    margin: 0 0 0.5rem 0;
    letter-spacing: -0.02em;
    background: linear-gradient(to right, var(--accent), var(--accent-sec));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.hero-subtitle {
    color: var(--text-muted);
    font-size: 0.95rem;
    line-height: 1.6;
    margin: 0;
}

/* Empty State */
.empty-state {
    text-align: center;
    padding: 3rem 0;
    animation: fadeIn 1s ease-out;
}
.empty-emoji {
    font-size: 3rem;
    margin-bottom: 1rem;
    display: inline-block;
    animation: messageSlideIn 0.5s ease-out;
}
.empty-title {
    font-size: 1.2rem;
    font-weight: 700;
    margin-bottom: 0.5rem;
}
.empty-hint {
    color: var(--text-muted);
    font-size: 0.9rem;
    max-width: 400px;
    margin: 0 auto 2rem;
}

/* Quick Prompts Grid */
.quick-prompts {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 0.8rem;
    padding: 0 1rem;
}

.quick-prompt-btn {
    background: var(--surface-elev);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1rem;
    color: var(--text-primary);
    font-size: 0.9rem;
    cursor: pointer;
    transition: all 0.2s ease;
    text-align: left;
    box-shadow: 0 2px 5px rgba(0,0,0,0.05);
}

.quick-prompt-btn:hover {
    background: var(--surface);
    border-color: var(--accent);
    transform: translateY(-2px);
    box-shadow: 0 4px 12px var(--accent-glow);
}

/* Status Pill */
.status-pill {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.4rem 0.9rem;
    border-radius: 999px;
    font-size: 0.8rem;
    font-weight: 600;
    background: var(--surface-elev);
    border: 1px solid var(--border);
}
.status-pill .dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    animation: pulseGlow 2s infinite;
}
.status-live .dot { background: var(--status-live); }
.status-fallback .dot { background: var(--status-fb); }
.status-placeholder .dot { background: var(--status-ph); }

/* Tone Chip */
.tone-chip {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    padding: 0.35rem 0.85rem;
    border-radius: 999px;
    font-size: 0.8rem;
    font-weight: 600;
    background: var(--accent-glow);
    color: var(--accent);
    border: 1px solid rgba(99, 102, 241, 0.2);
}

/* Typing Dots Animation */
.typing-dots {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    padding: 0.4rem 0.8rem;
    background: var(--assistant-bg);
    border: 1px solid var(--assistant-bdr);
    border-left: 4px solid var(--accent);
    border-radius: 12px 12px 12px 4px;
}
.typing-dots span {
    width: 7px;
    height: 7px;
    background-color: var(--accent-sec);
    border-radius: 50%;
    display: inline-block;
    animation: typingBounce 1.4s infinite ease-in-out both;
}
.typing-dots span:nth-child(1) { animation-delay: -0.32s; }
.typing-dots span:nth-child(2) { animation-delay: -0.16s; }
.typing-dots span:nth-child(3) { animation-delay: 0.0s; }

/* ============================================================
   Sidebar Overrides
   ============================================================ */
section[data-testid="stSidebar"] {
    background: rgba(19, 21, 42, 0.4) !important;
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border-right: 1px solid var(--border);
}
@media (prefers-color-scheme: light) {
    section[data-testid="stSidebar"] {
        background: rgba(255, 255, 255, 0.5) !important;
    }
}

section[data-testid="stSidebar"] .stButton button {
    width: 100%;
    border-radius: 12px;
    border: 1px solid var(--border);
    background: var(--surface-elev);
    color: var(--text-primary);
    transition: all 0.2s ease;
}
section[data-testid="stSidebar"] .stButton button:hover {
    background: var(--surface);
    border-color: var(--accent);
    color: var(--accent);
    transform: translateY(-1px);
    box-shadow: 0 4px 12px var(--accent-glow);
}

/* ============================================================
   Chat Input Box (Modern Floating Glass Pill)
   ============================================================ */
[data-testid="stBottom"] {
    background: transparent !important;
    padding-bottom: 1.5rem !important;
    border-top: none !important;
}
[data-testid="stBottom"] > div {
    background: transparent !important;
    border: none !important;
}

[data-testid="stChatInput"] {
    background: transparent !important;
    border: none !important;
    padding: 0 !important;
    max-width: 820px;
    margin: 0 auto;
}

/* Floating glass container for input */
[data-testid="stChatInput"] > div {
    background: rgba(26, 29, 53, 0.85) !important;
    border: 1px solid rgba(99, 102, 241, 0.35) !important;
    border-radius: 24px !important;
    backdrop-filter: blur(16px) !important;
    -webkit-backdrop-filter: blur(16px) !important;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.35), 0 0 0 1px rgba(255, 255, 255, 0.06) !important;
    padding: 4px 10px !important;
    transition: all 0.25s cubic-bezier(0.16, 1, 0.3, 1) !important;
}

[data-testid="stChatInput"] > div:focus-within {
    border-color: var(--accent) !important;
    box-shadow: 0 12px 35px rgba(99, 102, 241, 0.3), 0 0 0 2px var(--accent) !important;
}

[data-testid="stChatInput"] textarea {
    background: transparent !important;
    color: var(--text-primary) !important;
    font-family: "Inter", sans-serif !important;
    font-size: 0.96rem !important;
    border: none !important;
    box-shadow: none !important;
    padding-top: 0.4rem !important;
    padding-bottom: 0.4rem !important;
}

[data-testid="stChatInput"] textarea::placeholder {
    color: var(--text-muted) !important;
}

/* Circular send button */
[data-testid="stChatInput"] button {
    background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%) !important;
    color: #ffffff !important;
    border-radius: 50% !important;
    width: 36px !important;
    height: 36px !important;
    border: none !important;
    box-shadow: 0 2px 10px rgba(99, 102, 241, 0.4) !important;
    transition: all 0.2s ease !important;
}

[data-testid="stChatInput"] button:hover {
    transform: scale(1.08) !important;
    box-shadow: 0 4px 16px rgba(99, 102, 241, 0.6) !important;
}

</style>
"""


# ---------------------------------------------------------------------------
# Core Helpers
# ---------------------------------------------------------------------------

def _append_message(role: str, content: str) -> None:
    st.session_state.messages.append({"role": role, "content": content})


def _session_display_name(session: dict[str, object]) -> str:
    preview = str(session.get("preview") or "Empty session")
    count = int(session.get("message_count") or 0)
    return f"{preview} · {count} msg{'s' if count != 1 else ''}"


def _load_session(sid: str) -> None:
    saved = storage.load_history(sid)
    st.session_state.messages = saved.get("messages", [])
    st.session_state.tone_label = saved.get("tone_label", DEFAULT_TONE) or DEFAULT_TONE
    try:
        st.session_state.tone_confidence = float(saved.get("tone_confidence", 0.0))
    except (TypeError, ValueError):
        st.session_state.tone_confidence = 0.0
    st.session_state.chat_session_id = sid
    st.session_state.loaded_session_id = sid


def _new_session() -> str:
    sid = uuid.uuid4().hex
    st.session_state.messages = []
    st.session_state.tone_label = DEFAULT_TONE
    st.session_state.tone_confidence = 0.0
    st.session_state.chat_session_id = sid
    st.session_state.loaded_session_id = sid
    st.session_state.last_provider_label = "primary"
    return sid


def _tone_emoji(label: str) -> str:
    return {
        "flirtatious": "😉",
        "excited": "✨",
        "playful": "😄",
        "sad": "🤍",
        "angry": "🧊",
        "serious": "🎯",
        "calm": "🌿",
        "energetic": "⚡",
        "neutral": "💬",
    }.get(label, "💬")


def _model_status_html(cfg: dict) -> str:
    primary = cfg.get("primary", {})
    fallback = cfg.get("fallback", {})
    provider = (st.session_state.get("last_provider_label") or "primary").lower()

    if provider == "fallback" and fallback.get("provider"):
        cls = "status-fallback"
        text = f"Live · {fallback.get('model') or fallback.get('provider') or 'fallback'}"
    elif provider == "placeholder":
        cls = "status-placeholder"
        text = "Live · placeholder"
    else:
        cls = "status-live"
        if primary.get("provider"):
            text = f"Live · {primary.get('model') or primary.get('provider') or 'primary'}"
        else:
            text = "Live · primary"
    return (
        f'<span class="status-pill {cls}"><span class="dot"></span>{text}</span>'
    )


# ---------------------------------------------------------------------------
# Session Init
# ---------------------------------------------------------------------------

def _init_state() -> None:
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("tone_label", DEFAULT_TONE)
    st.session_state.setdefault("tone_confidence", 0.0)
    st.session_state.setdefault("pending_input", None)
    st.session_state.setdefault("chat_session_id", storage.session_id())
    st.session_state.setdefault("loaded_session_id", None)
    st.session_state.setdefault("last_provider_label", "primary")


_init_state()
if st.session_state.loaded_session_id != st.session_state.chat_session_id:
    _load_session(st.session_state.chat_session_id)


def _persist() -> None:
    storage.save_history(
        {
            "messages": st.session_state.messages,
            "tone_label": st.session_state.tone_label,
            "tone_confidence": st.session_state.tone_confidence,
        },
        sid=st.session_state.chat_session_id,
    )


# Inject CSS
st.markdown(_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Sidebar UI
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown(
        """
        <div style="padding: 0.5rem 0 1.5rem 0;">
            <div style="font-size: 1.4rem; font-weight: 800; background: linear-gradient(135deg, var(--accent) 0%, var(--accent-sec) 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                💬 ChatTalk
            </div>
            <div style="font-size: 0.85rem; color: var(--text-muted); margin-top: 0.25rem;">
                A premium AI companion experience.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    cfg = get_config()
    st.markdown(_model_status_html(cfg), unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    # New chat button prominent at top of sidebar
    if st.button("＋ New Chat", use_container_width=True, type="primary"):
        _new_session()
        st.rerun()

    st.markdown("<br>**Chat History**", unsafe_allow_html=True)
    sessions = storage.list_sessions()
    current_sid = st.session_state.chat_session_id
    session_map = {item["sid"]: item for item in sessions}
    if current_sid not in session_map:
        session_map[current_sid] = {
            "sid": current_sid,
            "preview": "Current session",
            "message_count": len(st.session_state.messages),
            "updated_at": 0.0,
        }
    ordered_sessions = [session_map[item["sid"]] for item in sessions]
    if current_sid not in {item["sid"] for item in ordered_sessions}:
        ordered_sessions.insert(0, session_map[current_sid])

    for item in ordered_sessions:
        sid = item["sid"]
        is_active = (sid == current_sid)
        preview = item.get("preview") or "Empty chat"
        count = int(item.get("message_count") or 0)
        icon = "💬" if is_active else "🗨️"
        label_text = f"{icon} {preview}"
        
        btn_type = "primary" if is_active else "secondary"
        if st.button(label_text, key=f"hist_btn_{sid}", use_container_width=True, type=btn_type):
            if sid != current_sid:
                _load_session(sid)
                st.rerun()

    st.markdown("<br>**Current Tone**", unsafe_allow_html=True)
    label = st.session_state.tone_label
    conf = st.session_state.tone_confidence
    st.markdown(
        f'<span class="tone-chip">{_tone_emoji(label)} {label.title()} · {int(conf * 100)}%</span>',
        unsafe_allow_html=True,
    )
    
    st.markdown("<div style='margin-top: 0.5rem;'>", unsafe_allow_html=True)
    st.progress(min(max(conf, 0.0), 1.0))
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("---")
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("🧹 Clear", use_container_width=True):
            st.session_state.messages = []
            st.session_state.tone_label = DEFAULT_TONE
            st.session_state.tone_confidence = 0.0
            _persist()
            st.rerun()
    with col_b:
        if st.button("↩️ Undo", use_container_width=True):
            if st.session_state.messages:
                st.session_state.messages.pop()
                tone = get_last_tone(
                    [m for m in st.session_state.messages if m["role"] == "user"],
                    current="",
                )
                st.session_state.tone_label = tone.label
                st.session_state.tone_confidence = tone.confidence
                _persist()
            st.rerun()

    if st.session_state.messages:
        transcript = "\n\n".join(
            f"{m['role'].upper()}: {m['content']}" for m in st.session_state.messages
        )
        st.download_button(
            "⬇️ Export",
            data=transcript,
            file_name="chattalk_transcript.txt",
            mime="text/plain",
            use_container_width=True,
        )


# ---------------------------------------------------------------------------
# Main Content UI
# ---------------------------------------------------------------------------

label = st.session_state.tone_label
tone_emoji_char = _tone_emoji(label)

st.markdown(
    f"""
    <div class="chat-app-header">
        <div class="chat-app-user">
            <div class="avatar-badge-wrap">
                💬
                <span class="online-dot"></span>
            </div>
            <div>
                <div class="chat-app-title">ChatTalk AI <span class="verified-icon">✔</span></div>
                <div class="chat-app-status">Active now · Tone: {tone_emoji_char} {label.title()}</div>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Render history
if not st.session_state.messages:
    st.markdown(
        """
        <div class="empty-state">
            <div class="empty-emoji">👋</div>
            <div class="empty-title">Nothing here yet</div>
            <div class="empty-hint">Start a conversation by typing a message below, or try a quick prompt.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    cols = st.columns(3)
    suggestions = [
        "Hey, how's it going?",
        "I'm feeling a little sad today",
        "I have exciting news!",
    ]
    for idx, suggestion in enumerate(suggestions):
        with cols[idx]:
            if st.button(suggestion, use_container_width=True, key=f"starter_{idx}"):
                st.session_state.pending_input = suggestion
                st.rerun()
else:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"], avatar="🧑" if msg["role"] == "user" else "💬"):
            st.markdown(msg["content"])


# Handle input
if st.session_state.get("pending_input"):
    prompt = st.session_state.pending_input
    st.session_state.pending_input = None
else:
    prompt = st.chat_input("Say something to ChatTalk…")

if prompt:
    # Append & Show user message immediately
    _append_message("user", prompt)
    with st.chat_message("user", avatar="🧑"):
        st.markdown(prompt)

    # Tone update
    tone = get_last_tone(st.session_state.messages, current=prompt)
    st.session_state.tone_label = tone.label
    st.session_state.tone_confidence = tone.confidence

    # Stream assistant reply with typing indicator
    with st.chat_message("assistant", avatar="💬"):
        typing_placeholder = st.empty()
        typing_placeholder.markdown(
            '<div class="typing-dots"><span></span><span></span><span></span></div>',
            unsafe_allow_html=True,
        )
        result_info = {"provider": "placeholder"}

        def stream_generator():
            first_chunk = True
            for chunk in generate_reply_stream(prompt, st.session_state.messages[:-1], result_info):
                if first_chunk:
                    typing_placeholder.empty()
                    first_chunk = False
                yield chunk

        response = st.write_stream(stream_generator())
        typing_placeholder.empty()

    _append_message("assistant", response)
    st.session_state.last_provider_label = result_info["provider"]
    _persist()
    st.rerun()


# ---------------------------------------------------------------------------
# Auto-Scroll + Footer
# ---------------------------------------------------------------------------
st.markdown(
    """
    <script>
        const chatContainer = window.parent.document.querySelector('[data-testid="stVerticalBlock"]');
        if (chatContainer) chatContainer.scrollTop = chatContainer.scrollHeight;
        const mainScroll = window.parent.document.querySelector('.main');
        if (mainScroll) mainScroll.scrollTop = mainScroll.scrollHeight;
    </script>
    <div style="text-align: center; color: var(--text-dim); font-size: 0.78rem; margin-top: 2rem; padding-bottom: 0.5rem; letter-spacing: 0.02em;">
        ChatTalk · built with Streamlit · streaming enabled
    </div>
    """,
    unsafe_allow_html=True,
)
