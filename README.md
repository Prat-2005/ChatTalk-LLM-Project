# 💬 ChatTalk — Tone-Aware AI Companion

ChatTalk is a vibrant, interactive AI chat application designed to run **100% offline** with your local LLMs, or in a hybrid configuration with optional cloud fallbacks. It detects conversation tone in real-time, adapts its persona, and streams responses through a dark glassmorphism interface styled after modern messaging apps.

---

## 🖼️ Interface Preview

![ChatTalk Interface Preview](assets/chattalk_preview.png)

---

## ✨ Key Features

- 🔌 **100% Offline Capability**: Run completely offline on your local machine using any local LLM server. No internet connection or external API keys required.
- ⚡ **Real-Time Streaming**: Live word-by-word streaming responses with visual `...` typing indicators.
- 🎭 **Tone-Aware Persona Engine**: Detects emotional tone (excited, calm, serious, playful, sad, angry) and adapts assistant style and vocabulary automatically.
- 🪞 **Slang & Style Mirroring**: Automatically matches the user's conversation energy and language nuances.
- 🧠 **Flexible Dual-Provider Chain**:
  - **Primary**: Connect any local or hosted LLM provider of your choice.
  - **Fallback (Optional)**: Automatically switches to an alternate LLM provider if the primary local model is unreachable.
  - **Placeholder Mode**: Keeps the UI fully functional even if no LLM server is active.
- 💾 **Persistent Local Storage**: Chat history is saved locally on disk with multi-session management, sequential history switching, undo, and transcript export options.
- 🎨 **Modern Glassmorphic UI**: Instagram/WhatsApp style DM chat bubbles, floating glass chat input bar, responsive sidebar, and dark mode theme.

---

## 📁 Project Structure

```
ChatTalk/
├── app.py                 # Streamlit UI & styling
├── llm.py                 # LLM provider routing & streaming logic
├── prompts.py             # Tone detection, prompt building & style mirroring
├── storage.py             # Chat history disk persistence
├── assets/
│   └── chattalk_glimpse.png # Application preview screenshot
├── tests/
│   ├── test_llm.py        # LLM transport & config tests
│   ├── test_prompts.py    # Tone detection & prompt tests
│   └── test_storage.py    # Persistence tests
├── requirements.txt       # Project dependencies
├── pytest.ini             # Test configuration
├── .env.example           # Environment template
└── README.md              # Documentation
```

---

## 🚀 Running Locally

### 1. Setup Virtual Environment
```powershell
# Create & activate environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
pip install pytest
```

### 2. Configure Environment Variables
Copy `.env.example` to create your `.env` configuration file:
```powershell
copy .env.example .env
```

Edit `.env` to configure your preferred LLM model and provider settings (check `.env.example` for details):

```env
# Primary LLM provider (Local or Remote)
LLM_PROVIDER=YOUR_LOCAL_PROVIDER
LLM_MODEL=YOUR_LOCAL_MODEL
LLM_BASE_URL=YOUR_LOCAL_BASE_URL

# (Optional) Secondary Fallback Provider
FALLBACK_PROVIDER=YOUR_FALLBACK_PROVIDER
FALLBACK_MODEL=YOUR_FALLBACK_MODEL
FALLBACK_BASE_URL=YOUR_FALLBACK_BASE_URL
FALLBACK_API_KEY=YOUR_API_KEY

# Sampling Configuration
LLM_TEMPERATURE=0.8
LLM_MAX_TOKENS=512
```

> **Note**: You can use any local or cloud LLM provider freely. ChatTalk is provider-agnostic.

### 3. Launch the Application
```powershell
streamlit run app.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 🧪 Testing

Run the full automated unit test suite:

```powershell
pytest
```
