# ChatTalk

ChatTalk is a Streamlit-based AI companion chat app designed to feel like a
natural, human-like conversation partner. It detects your tone, adapts its
style to match the mood of the conversation, and delivers a polished chat
experience with a modern UI.

This project combines Python, Streamlit, and LLM integration to create a
friendly and interactive assistant experience for real-time chatting.

## ✨ Features

- 💬 Modern chat-style interface with polished bubbles and avatars
- 🎭 Tone-aware responses that detect moods such as playful, calm, serious,
  energetic, sad, or angry
- 🔁 Context-aware replies that prioritize recent conversation history
- 🪞 Style mirroring that adjusts the assistant's tone and phrasing
- 🧠 Local-first LLM support with Ollama, plus fallback handling
- 🧰 Sidebar controls for model information, tone, clear/undo/export actions
- 📋 Copy assistance for individual replies
- 🟢 Live status indicators for connection and response mode

## 🖼️ Interface Preview

A glimpse of the chat experience:

![ChatTalk Interface Preview](https://via.placeholder.com/900x500.png?text=ChatTalk+Interface+Preview)

> Replace the image above with a real screenshot of your app once you have one.

## 🧱 Project structure

```
ChatTalk/
├── app.py             # Streamlit UI
├── llm.py             # Local LLM transport (Ollama), .env loader, fallbacks
├── prompts.py         # Tone detection, style guides, system-prompt builder
├── tests/
│   ├── test_prompts.py
│   └── test_llm.py
├── requirements.txt
├── pytest.ini
├── .env.example
├── PROMPT.md
└── README.md
```

## ▶️ Running locally

```powershell
# 1. Activate the virtual environment
.\.venv\Scripts\Activate.ps1

# 2. Install dependencies
pip install -r requirements.txt
pip install pytest        # only needed to run the test suite

# 3. (Optional) configure an LLM — copy and edit
copy .env.example .env

# 4. Run the app
streamlit run app.py
```

The app opens on http://localhost:8501 by default.

## ⚙️ Environment variables

| Var | Default | Purpose |
|---|---|---|
| `LLM_PROVIDER` | `ollama` | Set blank to use placeholder replies |
| `LLM_MODEL` | `llama3.2` | Model name as the local server knows it |
| `LLM_BASE_URL` | `http://localhost:11434` | Local chat server URL |
| `LLM_API_KEY` | _(blank)_ | Only for hosted / non-local providers |
| `LLM_TEMPERATURE` | `0.8` | Sampling temperature |
| `LLM_MAX_TOKENS` | `512` | Max tokens per reply |

The `.env` loader is stdlib-only — no `python-dotenv` required.

## 🧪 Tests

```powershell
.\.venv\Scripts\python.exe -m pytest
```

40 tests cover tone detection (all labels, recency shift, edge cases), system
prompt construction, history trimming, message assembly, and the Ollama
transport (with mocked HTTP).

## 🤖 How the LLM is called

`llm.generate_reply(user_message, history)`:

1. Aggregates all user messages and detects a tone.
2. Builds a tone-aware system prompt (persona + style + length + guard rails).
3. Trims history to fit a 4,000-char budget.
4. POSTs to `${LLM_BASE_URL}/api/chat` (Ollama chat API).
5. Falls back to a placeholder if the model is unconfigured or unreachable.

Adding another provider (e.g. llama.cpp, LM Studio, or a transformers
pipeline) is a single switch in `llm._ollama_chat`'s sibling.
