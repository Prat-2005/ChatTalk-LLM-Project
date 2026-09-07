// ------------------------------------------------------------------
// State
// ------------------------------------------------------------------
let authToken = localStorage.getItem('chattalk_token');
let currentUsername = localStorage.getItem('chattalk_username') || null;
let currentSessionId = localStorage.getItem('chattalk_session') || null;
let currentHistory = [];
let currentTone = { label: 'neutral', confidence: 0 };
let isStreaming = false;

// DOM refs
const msgContainer = document.getElementById('messageContainer');
const msgInput = document.getElementById('messageInput');
const sendBtn = document.getElementById('sendBtn');
const sessionListEl = document.getElementById('sessionList');
const chatTitle = document.getElementById('chatTitle');
const toneBadge = document.getElementById('toneBadge');
const newChatBtn = document.getElementById('newChatBtn');
const deleteChatBtn = document.getElementById('deleteChatBtn');

// Auth modal elements
const authModal = document.getElementById('authModal');
const authError = document.getElementById('authError');
const authToggleBtn = document.getElementById('authToggleBtn');
const userInfo = document.getElementById('userInfo');
const userAvatar = document.getElementById('userAvatar');
const userName = document.getElementById('userName');
const app = document.getElementById('app');

// Login form
const loginUsername = document.getElementById('loginUsername');
const loginPassword = document.getElementById('loginPassword');
const loginBtn = document.getElementById('loginBtn');

// Signup form
const signupFirstName = document.getElementById('signupFirstName');
const signupLastName = document.getElementById('signupLastName');
const signupUsername = document.getElementById('signupUsername');
const signupPassword = document.getElementById('signupPassword');
const signupBtn = document.getElementById('signupBtn');

// Tabs
const loginTab = document.getElementById('loginTab');
const signupTab = document.getElementById('signupTab');
const loginForm = document.getElementById('loginForm');
const signupForm = document.getElementById('signupForm');
const authTitle = document.getElementById('authTitle');
const closeAuthBtn = document.getElementById('closeAuthBtn');

const API_BASE = '';

// ------------------------------------------------------------------
// Helpers
// ------------------------------------------------------------------
function apiUrl(path) {
  return `${API_BASE}${path}`;
}

function getUserInitial() {
  if (currentUsername) {
    return currentUsername.charAt(0).toUpperCase();
  }
  return 'U';
}

// ------------------------------------------------------------------
// Auth helpers
// ------------------------------------------------------------------
function getHeaders() {
  const headers = { 'Content-Type': 'application/json' };
  if (authToken) {
    headers['Authorization'] = `Bearer ${authToken}`;
  }
  return headers;
}

async function apiFetch(path, options = {}) {
  const resp = await fetch(apiUrl(path), {
    ...options,
    headers: { ...getHeaders(), ...(options.headers || {}) },
  });
  if (resp.status === 401) {
    logout();
    throw new Error('Session expired. Please login again.');
  }
  return resp;
}

function setAuth(token, username) {
  authToken = token;
  currentUsername = username;
  localStorage.setItem('chattalk_token', token);
  localStorage.setItem('chattalk_username', username);
  updateAuthUI();
}

function clearAuth() {
  authToken = null;
  currentUsername = null;
  localStorage.removeItem('chattalk_token');
  localStorage.removeItem('chattalk_username');
  updateAuthUI();
}

function logout() {
  clearAuth();
  location.reload();
}

function updateAuthUI() {
  if (currentUsername) {
    const initial = currentUsername.charAt(0).toUpperCase();
    userAvatar.textContent = initial;
    userName.textContent = currentUsername;
    authToggleBtn.textContent = 'Logout';
    authToggleBtn.onclick = logout;
    authModal.style.display = 'none';
    app.style.display = 'flex';
  } else {
    userAvatar.textContent = 'U';
    userName.textContent = 'Anonymous';
    authToggleBtn.textContent = 'Login';
    authToggleBtn.onclick = () => {
      authModal.style.display = 'flex';
    };
    app.style.display = 'flex';
  }
}

// ------------------------------------------------------------------
// Auth forms – tab switching
// ------------------------------------------------------------------
function showLogin() {
  loginTab.classList.add('active');
  signupTab.classList.remove('active');
  loginForm.style.display = 'flex';
  signupForm.style.display = 'none';
  authTitle.textContent = 'Login to ChatTalk';
  authError.textContent = '';
}

function showSignup() {
  signupTab.classList.add('active');
  loginTab.classList.remove('active');
  signupForm.style.display = 'flex';
  loginForm.style.display = 'none';
  authTitle.textContent = 'Create an Account';
  authError.textContent = '';
}

loginTab.addEventListener('click', showLogin);
signupTab.addEventListener('click', showSignup);

closeAuthBtn.addEventListener('click', () => {
  authModal.style.display = 'none';
  app.style.display = 'flex';
});

// ------------------------------------------------------------------
// Login / Signup handlers
// ------------------------------------------------------------------
async function handleLogin() {
  const username = loginUsername.value.trim();
  const password = loginPassword.value.trim();
  if (!username || !password) {
    authError.textContent = 'Please enter username and password.';
    return;
  }
  try {
    const resp = await fetch(apiUrl('/login'), {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ username, password }),
    });
    const data = await resp.json();
    if (!resp.ok) {
      authError.textContent = data.detail || 'Login failed.';
      return;
    }
    setAuth(data.access_token, username);
    authModal.style.display = 'none';
    renderSessionList(await fetchSessions(), currentSessionId);
  } catch (err) {
    authError.textContent = 'Network error.';
  }
}

async function handleSignup() {
  const firstName = signupFirstName.value.trim();
  const lastName = signupLastName.value.trim();
  const username = signupUsername.value.trim();
  const password = signupPassword.value.trim();
  if (!username || !password) {
    authError.textContent = 'Username and password are required.';
    return;
  }
  const payload = { username, password };
  if (firstName) payload.first_name = firstName;
  if (lastName) payload.last_name = lastName;

  try {
    const resp = await fetch(apiUrl('/signup'), {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    const data = await resp.json();
    if (!resp.ok) {
      authError.textContent = data.detail || 'Signup failed.';
      return;
    }
    setAuth(data.access_token, username);
    authModal.style.display = 'none';
    renderSessionList(await fetchSessions(), currentSessionId);
  } catch (err) {
    authError.textContent = 'Network error.';
  }
}

loginBtn.addEventListener('click', handleLogin);
signupBtn.addEventListener('click', handleSignup);
loginPassword.addEventListener('keydown', (e) => {
  if (e.key === 'Enter') handleLogin();
});
signupPassword.addEventListener('keydown', (e) => {
  if (e.key === 'Enter') handleSignup();
});

// ------------------------------------------------------------------
// API calls (sessions, chat)
// ------------------------------------------------------------------
async function loadSession(sid) {
  if (isStreaming) return;
  try {
    const resp = await apiFetch(`/session/${sid}`);
    if (!resp.ok) throw new Error('Failed to load session');
    const state = await resp.json();
    currentSessionId = sid;
    localStorage.setItem('chattalk_session', sid);
    currentHistory = state.messages || [];
    currentTone = {
      label: state.tone_label || 'neutral',
      confidence: state.tone_confidence || 0,
    };
    updateTitle(state.title || 'New Chat');
    updateTone(currentTone);
    renderMessages(currentHistory);
    renderSessionList(await fetchSessions(), sid);
  } catch (err) {
    console.error('Load session error:', err);
  }
}

async function fetchSessions() {
  try {
    const resp = await apiFetch('/sessions');
    if (!resp.ok) return [];
    return await resp.json();
  } catch {
    return [];
  }
}

async function saveCurrentSession() {
  if (!currentSessionId) return;
  const state = {
    messages: currentHistory,
    tone_label: currentTone.label,
    tone_confidence: currentTone.confidence,
    title: chatTitle.textContent,
  };
  try {
    await apiFetch(`/session/${currentSessionId}`, {
      method: 'POST',
      body: JSON.stringify(state),
    });
  } catch (err) {
    console.error('Save session error:', err);
  }
}

async function deleteCurrentSession() {
  if (!currentSessionId) return;
  if (!confirm('Delete this session?')) return;
  try {
    await apiFetch(`/session/${currentSessionId}`, { method: 'DELETE' });
    localStorage.removeItem('chattalk_session');
    currentSessionId = null;
    currentHistory = [];
    currentTone = { label: 'neutral', confidence: 0 };
    updateTitle('New Chat');
    updateTone(currentTone);
    renderMessages([]);
    renderSessionList(await fetchSessions(), null);
  } catch (err) {
    console.error('Delete error:', err);
  }
}

async function createNewSession() {
  if (isStreaming) return;
  const newId = crypto.randomUUID
    ? crypto.randomUUID()
    : Date.now().toString(36) + Math.random().toString(36).slice(2, 7);
  currentSessionId = newId;
  localStorage.setItem('chattalk_session', newId);
  currentHistory = [];
  currentTone = { label: 'neutral', confidence: 0 };
  updateTitle('New Chat');
  updateTone(currentTone);
  renderMessages([]);
  await saveCurrentSession();
  renderSessionList(await fetchSessions(), currentSessionId);
}

// ------------------------------------------------------------------
// Rendering
// ------------------------------------------------------------------
function renderMessages(history) {
  msgContainer.innerHTML = '';
  history.forEach((msg) => {
    const container = document.createElement('div');
    container.className = `message ${msg.role}`;
    const avatar = document.createElement('div');
    avatar.className = 'avatar';
    if (msg.role === 'user') {
      avatar.textContent = getUserInitial();
    } else {
      avatar.textContent = 'AI';
    }
    container.appendChild(avatar);
    const bubble = document.createElement('div');
    bubble.className = 'bubble';
    bubble.textContent = msg.content;
    if (msg.role === 'assistant' && msg.provider) {
      const tag = document.createElement('span');
      tag.className = 'provider-tag';
      tag.textContent = `via ${msg.provider}`;
      bubble.appendChild(tag);
    }
    container.appendChild(bubble);
    msgContainer.appendChild(container);
  });
  msgContainer.scrollTop = msgContainer.scrollHeight;
}

function renderSessionList(sessions, activeId) {
  sessionListEl.innerHTML = '';
  if (!sessions.length) {
    sessionListEl.innerHTML =
      '<div style="padding:20px;opacity:0.6;font-size:0.9rem;">No saved sessions</div>';
    return;
  }
  sessions.forEach((sess) => {
    const div = document.createElement('div');
    div.className = `session-item${sess.sid === activeId ? ' active' : ''}`;
    div.dataset.sid = sess.sid;
    div.innerHTML = `
      <div class="title">${sess.title || 'New Chat'}</div>
      <div class="preview">${sess.preview || 'Empty'}</div>
    `;

    div.addEventListener('click', () => loadSession(sess.sid));
    sessionListEl.appendChild(div);
  });
}

function updateTitle(title) {
  chatTitle.textContent = title || 'New Chat';
}

function updateTone(tone) {
  const emoji = getToneEmoji(tone.label);
  toneBadge.textContent = `${emoji} ${tone.label} (${Math.round(tone.confidence * 100)}%)`;
}

function getToneEmoji(label) {
  const map = {
    flirtatious: '😏',
    excited: '🔥',
    playful: '😜',
    sad: '😢',
    angry: '😡',
    serious: '🧐',
    calm: '😌',
    energetic: '💪',
    neutral: '😐',
  };
  return map[label] || '😐';
}

// ------------------------------------------------------------------
// Chat logic
// ------------------------------------------------------------------
async function sendMessage() {
  const text = msgInput.value.trim();
  if (!text || isStreaming) return;
  msgInput.value = '';
  msgInput.disabled = true;
  sendBtn.disabled = true;

  const userMsg = { role: 'user', content: text };
  currentHistory.push(userMsg);
  renderMessages(currentHistory);

  // Typing indicator (buffering)
  const typingContainer = document.createElement('div');
  typingContainer.className = 'message assistant';
  const avatar = document.createElement('div');
  avatar.className = 'avatar';
  avatar.textContent = 'AI';
  typingContainer.appendChild(avatar);
  const bubble = document.createElement('div');
  bubble.className = 'bubble';
  bubble.innerHTML = `<div class="typing-indicator"><span></span><span></span><span></span></div>`;
  typingContainer.appendChild(bubble);
  msgContainer.appendChild(typingContainer);
  msgContainer.scrollTop = msgContainer.scrollHeight;

  isStreaming = true;

  try {
    const payload = {
      message: text,
      history: currentHistory.slice(0, -1),
      session_id: currentSessionId,
    };
    const resp = await apiFetch('/chat/stream', {
      method: 'POST',
      body: JSON.stringify(payload),
    });
    if (!resp.ok) throw new Error('Stream request failed');
    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let assistantContent = '';
    let provider = 'unknown';
    let assistantContainer = null;
    let bubbleAss = null;

    let buffer = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop() || '';

      for (const line of lines) {
        if (!line.trim()) continue;

        try {
          const data = JSON.parse(line);

          if (data.error) {
            typingContainer.remove();
            throw new Error(data.error);
         }

         if (data.done) break;

         if (data.chunk) {
          if (!assistantContainer) {
            typingContainer.remove();

            assistantContainer = document.createElement('div');
            assistantContainer.className = 'message assistant';

            const avatarAss = document.createElement('div');
            avatarAss.className = 'avatar';
            avatarAss.textContent = 'AI';

            bubbleAss = document.createElement('div');
            bubbleAss.className = 'bubble';

            assistantContainer.appendChild(avatarAss);
            assistantContainer.appendChild(bubbleAss);
            msgContainer.appendChild(assistantContainer);
          }

          assistantContent += data.chunk;
          bubbleAss.textContent = assistantContent;

          if (data.provider) {
            provider = data.provider;
          }

          msgContainer.scrollTop = msgContainer.scrollHeight;
        }
      } catch (error) {
        console.error('Stream parsing error:', error);
      }
    }
  }

    if (assistantContent) {
      currentHistory.push({ role: 'assistant', content: assistantContent, provider });
      await saveCurrentSession();

      // --- Update mood badge ---
      try {
        const toneResp = await apiFetch('/tone', {
          method: 'POST',
          body: JSON.stringify({
            history: currentHistory,
            current_message: '',
          }),
        });

        if (!toneResp.ok) {
          const errorText = await toneResp.text();
          console.error('Tone update failed:', toneResp.status, errorText);
        } else {
          const toneData = await toneResp.json();

          currentTone = {
          label: toneData.label || 'neutral',
          confidence: toneData.confidence || 0,
       };

       updateTone(currentTone);
       await saveCurrentSession();
      }
    } catch (error) {
      console.error('Tone request error:', error);
    }
      // Generate title if first exchange
      if (currentHistory.length === 2) {
        try {
          const titleResp = await apiFetch('/title', {
            method: 'POST',
            body: JSON.stringify({ history: currentHistory }),
          });
          const titleData = await titleResp.json();
          if (titleData.title) {
            updateTitle(titleData.title);
            await saveCurrentSession();
          }
        } catch (e) {
          /* ignore */
        }
      }
      renderMessages(currentHistory);
      renderSessionList(await fetchSessions(), currentSessionId);
    } else {
      if (assistantContainer) {
        assistantContainer.remove();
      } else {
        typingContainer.remove();
      }
    }
  } catch (err) {
    console.error('Stream error:', err);
    typingContainer.remove();
    const errorContainer = document.createElement('div');
    errorContainer.className = 'message assistant';
    const errAvatar = document.createElement('div');
    errAvatar.className = 'avatar';
    errAvatar.textContent = 'AI';
    errorContainer.appendChild(errAvatar);
    const errBubble = document.createElement('div');
    errBubble.className = 'bubble';
    errBubble.textContent = `❌ Error: ${err.message || 'Something went wrong'}`;
    errorContainer.appendChild(errBubble);
    msgContainer.appendChild(errorContainer);
  } finally {
    isStreaming = false;
    msgInput.disabled = false;
    sendBtn.disabled = false;
    msgInput.focus();
  }
}

// ------------------------------------------------------------------
// Initialization
// ------------------------------------------------------------------
async function init() {
  // Check token
  if (authToken) {
    try {
      const resp = await apiFetch('/me');
      if (resp.ok) {
        const data = await resp.json();
        if (data.authenticated) {
          currentUsername = data.username;
          localStorage.setItem('chattalk_username', currentUsername);
        } else {
          clearAuth();
        }
      } else {
        clearAuth();
      }
    } catch {
      clearAuth();
    }
  }
  updateAuthUI();

  // Show modal if not logged in
  if (!currentUsername) {
    authModal.style.display = 'flex';
    showLogin();
  }

  // Load sessions
  const sessions = await fetchSessions();
  renderSessionList(sessions, currentSessionId);
  if (currentSessionId) {
    await loadSession(currentSessionId);
  } else {
    await createNewSession();
  }

  // Event listeners
  sendBtn.addEventListener('click', sendMessage);
  msgInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  });
  newChatBtn.addEventListener('click', createNewSession);
  deleteChatBtn.addEventListener('click', deleteCurrentSession);
  authToggleBtn.addEventListener('click', () => {
    if (currentUsername) logout();
    else {
      authModal.style.display = 'flex';
      showLogin();
    }
  });
}

init();