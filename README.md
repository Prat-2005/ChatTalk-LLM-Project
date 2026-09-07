# 💬 ChatTalk — Tone-Aware AI Companion

ChatTalk is a vibrant, interactive AI chat application designed to run **100% offline** with your local LLMs, or in a hybrid configuration with optional cloud fallbacks. It detects conversation tone in real-time, adapts its persona, and streams responses through a dark glassmorphism interface styled after modern messaging apps.

---

## 🖼️ Interface Preview

![ChatTalk Interface Preview](ss/snap.png)

---

## ✨ Key Features

- 🔌 **100% Offline Capability**: Run completely offline on your local machine using any local LLM server (e.g. Ollama). No internet connection or external API keys required.
- ⚡ **Real-Time Streaming**: Live word-by-word streaming responses via Server-Sent Events with visual typing indicators.
- 🎭 **Tone-Aware Persona Engine**: Detects emotional tone (flirtatious, excited, playful, sad, angry, serious, calm, energetic, neutral) and adapts assistant style and vocabulary automatically.
- 🪞 **Slang & Style Mirroring**: Automatically matches the user's conversation energy, slang density, and language nuances.
- 🧠 **Flexible Dual-Provider Chain**:
  - **Primary**: Connect any OpenAI-compatible local or hosted LLM provider.
  - **Fallback (Optional)**: Automatically switches to an alternate provider if the primary is unreachable.
  - **Placeholder Mode**: Keeps the UI fully functional with diagnostic messages even if no LLM server is active.
- 🔐 **User Authentication**: JWT-based signup/login with bcrypt password hashing. Authenticated users get persistent, cross-device chat history stored in PostgreSQL.
- 👤 **Anonymous Mode**: Use the app without signing up — chat history is stored locally on disk as JSON files.
- 💾 **Hybrid Storage**: File-based persistence for anonymous users, PostgreSQL for authenticated users — with multi-session management, session switching, and deletion.
- 🏷️ **Auto-Generated Titles**: Chat sessions are automatically titled based on conversation content using the LLM.
- 🎨 **Modern Glassmorphic UI**: Instagram/WhatsApp-style DM chat bubbles, floating glass chat input bar, responsive sidebar, dark mode theme, and real-time tone badge.

---

## 🏗️ Architecture

ChatTalk uses a **client-server architecture**:

- **Backend** — A FastAPI application (`app.py`) serving a RESTful + streaming API, with modular backend logic for LLM routing, tone detection, authentication, and storage.
- **Frontend** — A vanilla HTML/CSS/JavaScript single-page app served as static files, communicating with the backend via `fetch` and `EventSource` for streaming.

---

## 📁 Project Structure

```
ChatTalk/
├── app.py                     # FastAPI entry point — API routes, auth, CORS
├── backend/
│   ├── llm.py                 # LLM provider routing, OpenAI SDK streaming, fallback chain
│   ├── prompts.py             # Tone detection, persona engine, prompt construction & style mirroring
│   ├── storage.py             # File-based chat history persistence (anonymous users)
│   ├── db.py                  # SQLAlchemy models, PostgreSQL session/user CRUD (authenticated users)
│   └── auth.py                # JWT token creation & verification
├── frontend/
│   ├── index.html             # App shell — auth modal, sidebar, chat area
│   ├── script.js              # Client logic — streaming, session management, auth flows
│   └── style.css              # Simple theme & responsive layout
├── requirements.txt           # Python dependencies
├── .env.example               # Environment variable template
├── LICENSE                    # MIT License
└── README.md                  # Documentation
```

---

## 🚀 Running Locally

### Prerequisites

- **Python 3.10+**
- **PostgreSQL** (required for authenticated user features; the app still works in anonymous-only mode without it)
- A local LLM server like [Ollama](https://ollama.com/) (optional — the app falls back to placeholder mode)

### 1. Setup Virtual Environment

```powershell
# Create & activate environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment Variables

Copy `.env.example` to create your `.env` configuration file:

```powershell
copy .env.example .env
```

Edit `.env` to configure your settings:

```env
# PostgreSQL connection (for authenticated users)
DATABASE_URL=postgresql://user:password@localhost/chattalk
SECRET_KEY=your-secret-key

# --- Primary LLM provider (Local or Remote) ---
LLM_PROVIDER=YOUR_LOCAL_PROIVDER
LLM_MODEL=YOUR_LOCAL_MODEL
LLM_BASE_URL=YOUR_LOCAL_API_KEY

# --- (Optional) Fallback provider ---
FALLBACK_PROVIDER=YOUR_FALLBACK_PROVIDER
FALLBACK_MODEL=YOUR_FALLBACK_MODEL
FALLBACK_BASE_URL=YOUR_FALLBACK_BASE_URL
FALLBACK_API_KEY=YOUR_FALLBACK_API_KEY

# --- Sampling ---
LLM_TEMPERATURE=0.8
LLM_MAX_TOKENS=512

# --- On-disk storage (anonymous sessions) ---
CHATTALK_DATA_DIR=.chattalk_data
```

> **Note**: ChatTalk is provider-agnostic — any OpenAI-compatible API endpoint works as a provider (Ollama, LM Studio, vLLM, Groq, OpenRouter, etc.).

### 3. Set Up the Database (Optional)

If you want authenticated user support, create the PostgreSQL database:

```sql
CREATE DATABASE chattalk;
```

Tables are auto-created on first run via SQLAlchemy.

### 4. Launch the Application

```powershell
uvicorn app:app --reload --port 8000
```

Open [http://localhost:8000](http://localhost:8000) in your browser.

---

## 📡 API Endpoints

| Method | Endpoint | Auth | Description |
|--------|----------|------|-------------|
| `POST` | `/signup` | — | Create a new user account |
| `POST` | `/login` | — | Authenticate and receive a JWT |
| `GET` | `/me` | Optional | Get current user info |
| `POST` | `/chat/stream` | Optional | Stream a chat reply (SSE) |
| `POST` | `/tone` | Optional | Detect tone from message history |
| `POST` | `/title` | Optional | Auto-generate a session title |
| `GET` | `/sessions` | Optional | List all chat sessions |
| `GET` | `/session/{id}` | Optional | Load a specific session |
| `POST` | `/session/{id}` | Optional | Save/update a session |
| `DELETE` | `/session/{id}` | Optional | Delete a session |
| `GET` | `/config` | Optional | Get LLM provider configuration |

> Endpoints marked **Optional** for auth work for both anonymous and authenticated users. Anonymous users get file-based storage; authenticated users get PostgreSQL storage.

---

## 🎭 Tone Detection

ChatTalk detects **9 emotional tones** from the user's messages using lexicon matching, emoji analysis, caps detection, and recency-weighted scoring:

| Tone | Description |
|------|-------------|
| 😏 Flirtatious | Playful and teasing with affectionate language |
| 🎉 Excited | High energy, enthusiastic, celebratory |
| 😜 Playful | Humorous, witty, lighthearted banter |
| 😢 Sad | Empathetic, gentle, validating |
| 😡 Angry | Calm acknowledgment, non-escalating |
| 😐 Serious | Clear, focused, respectful |
| 😌 Calm | Relaxed pace, warm and quiet |
| 💪 Energetic | Upbeat, brisk, action-oriented |
| 🙂 Neutral | Friendly, conversational default |

---

## 📄 License

[MIT](LICENSE) © 2026 Pratham Singh
