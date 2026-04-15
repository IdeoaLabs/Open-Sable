import { create } from 'zustand'

// ─── Unique ID helper ──────────────────────────────────────────────────────
const uid = () => `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`

// ─── Strip <think>...</think> blocks from LLM output ──────────────────────
function stripThinkBlocks(text) {
  if (!text) return text
  // Remove complete <think>...</think> blocks (DeepSeek chain-of-thought)
  let result = text.replace(/<think>[\s\S]*?<\/think>/gi, '')
  // Remove orphan closing tag if opening was stripped in a chunk
  result = result.replace(/<\/think>/gi, '')
  // Remove orphan opening tag
  result = result.replace(/<think>/gi, '')
  return result.trim()
}

// ─── Per-session think-block streaming buffer ──────────────────────────────
// Tracks whether we're currently inside a <think> block while streaming
const thinkState = {} // { [sessionId]: { inThink: boolean, buffer: string } }

// ─── Elapsed timer (global interval handle) ───────────────────────────────
let _elapsedInterval = null
function _startElapsedTimer(store) {
  if (_elapsedInterval) return
  _elapsedInterval = setInterval(() => {
    const { streaming, streamingSince } = store.getState()
    if (!streaming || !streamingSince) {
      clearInterval(_elapsedInterval)
      _elapsedInterval = null
      return
    }
    store.setState({ elapsed: Math.floor((Date.now() - streamingSince) / 1000) })
  }, 500)
}

// ─── Store ─────────────────────────────────────────────────────────────────
export const useSableStore = create((set, get) => ({
  // Connection
  ws: null,
  wsStatus: 'disconnected', // 'connecting' | 'connected' | 'disconnected'
  booting: true, // true until first successful WS connection
  config: { wsUrl: 'ws://localhost:8789', token: '' },
  agentModel: '',
  agentVersion: '',

  // Sessions
  sessions: [],
  activeSessionId: null,

  // Messages per session { [sessionId]: Message[] }
  messages: {},

  // Streaming state
  streaming: false,
  streamingSessionId: null,
  // 'thinking' = model is processing, no tokens yet
  // 'responding' = first token received, streaming words
  streamingPhase: 'thinking',
  // Timestamp when sendMessage was called (for elapsed counter)
  streamingSince: null,
  // Current elapsed seconds (updated every 500ms)
  elapsed: 0,

  // Settings modal
  settingsOpen: false,
  toast: null,

  // Agent tools & live progress
  tools: [],
  agentProgress: null,

  // Multi-agent
  agents: [],          // [{ name, running, is_current }]
  activeAgent: '',     // which agent is selected ('' = local/default)

  // Code execution results: { [request_id]: { stdout, stderr, exit_code, running } }
  codeResults: {},

  // Model groups (from gateway models.list)
  modelGroups: [],   // [{provider, name, models: [{name, active}]}]
  activeProvider: 'ollama',

  // Brain data (for Brain panel)
  brainData: null,
  brainLoading: false,

  // Permission confirmation prompt (from gateway RBAC)
  pendingPermission: null, // { requestId, action, tool, arguments, message }

  // ── Actions ──────────────────────────────────────────────────────────────

  setConfig: (config) => set({ config }),

  // Go back to welcome screen without clearing history
  goHome: () => set({ activeSessionId: null }),

  openSettings: () => set({ settingsOpen: true }),
  closeSettings: () => set({ settingsOpen: false }),

  showToast: (msg) => {
    set({ toast: msg })
    setTimeout(() => set({ toast: null }), 4000)
  },

  // Connect / Reconnect
  connect: (overrideConfig) => {
    const cfg = overrideConfig || get().config
    const { ws } = get()

    if (ws) {
      ws.onclose = null
      ws.onerror = null
      ws.close()
    }

    const url = cfg.token
      ? `${cfg.wsUrl}?token=${encodeURIComponent(cfg.token)}`
      : cfg.wsUrl

    set({ wsStatus: 'connecting', config: cfg })
    const socket = new WebSocket(url)

    socket.onopen = () => {
      set({ wsStatus: 'connected', ws: socket, booting: false })
      // Fetch existing sessions, available tools, and agent status (model info)
      socket.send(JSON.stringify({ type: 'sessions.list' }))
      socket.send(JSON.stringify({ type: 'tools.list' }))
      socket.send(JSON.stringify({ type: 'status' }))
      socket.send(JSON.stringify({ type: 'agents.list' }))
      socket.send(JSON.stringify({ type: 'models.list' }))
    }

    socket.onclose = () => {
      set({ wsStatus: 'disconnected', ws: null, streaming: false })
      // Auto-reconnect after 1s (SableCore may still be booting)
      setTimeout(() => {
        const { wsStatus } = get()
        if (wsStatus === 'disconnected') get().connect()
      }, 1000)
    }

    socket.onerror = () => {
      set({ wsStatus: 'disconnected' })
    }

    socket.onmessage = (event) => {
      let msg
      try { msg = JSON.parse(event.data) } catch { return }
      get()._handleMessage(msg)
    }

    set({ ws: socket })
  },

  disconnect: () => {
    const { ws } = get()
    if (ws) {
      ws.onclose = null
      ws.close()
    }
    set({ ws: null, wsStatus: 'disconnected' })
  },

  // Send a chat message
  sendMessage: (text) => {
    const { ws, wsStatus, activeSessionId, activeAgent, agents } = get()
    if (!ws || wsStatus !== 'connected' || !text.trim()) return

    const sessionId = activeSessionId || uid()

    // Ensure session exists in local state
    get()._ensureSession(sessionId)
    set({ activeSessionId: sessionId })

    // Add user message locally
    get()._addMessage(sessionId, { id: uid(), role: 'user', content: text, ts: Date.now() })

    // Show typing indicator + start elapsed timer
    const now = Date.now()
    set({ streaming: true, streamingSessionId: sessionId, streamingPhase: 'thinking', streamingSince: now, elapsed: 0 })
    _startElapsedTimer({ getState: get, setState: set })

    // If chatting with a remote agent, use agents.chat proxy
    const isRemote = activeAgent && !agents.find(a => a.is_current && a.name === activeAgent)
    if (isRemote) {
      ws.send(JSON.stringify({
        type: 'agents.chat',
        profile: activeAgent,
        session_id: sessionId,
        user_id: 'desktop',
        text,
      }))
    } else {
      ws.send(JSON.stringify({
        type: 'message',
        session_id: sessionId,
        user_id: 'desktop',
        text,
      }))
    }
  },

  // Select agent
  selectAgent: (name) => {
    set({ activeAgent: name, activeSessionId: null })
    // Request sessions list from the selected agent if remote
    const { ws, agents } = get()
    const isRemote = name && !agents.find(a => a.is_current && a.name === name)
    if (!isRemote && ws?.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ type: 'sessions.list' }))
    }
  },

  // New chat,  go to welcome screen; session is created lazily on first sendMessage
  newChat: () => {
    set({ activeSessionId: null })
  },

  // Run a code snippet via gateway and store the result by request_id
  runCode: (code, language, requestId, stdin = null) => {
    const { ws, wsStatus } = get()
    if (!ws || wsStatus !== 'connected') return
    set(prev => ({
      codeResults: { ...prev.codeResults, [requestId]: { running: true, stdout: '', stderr: '', exit_code: null } }
    }))
    ws.send(JSON.stringify({ type: 'code.run', request_id: requestId, code, language, stdin }))
  },

  // Attempt an automatic fix for interactive snippets (replace input() with argv consumption)
  autoFixCode: (code, language, requestId) => {
    const { ws, wsStatus } = get()
    if (!ws || wsStatus !== 'connected') return
    set(prev => ({
      codeResults: { ...prev.codeResults, [requestId]: { running: true, stdout: '', stderr: '', exit_code: null } }
    }))
    ws.send(JSON.stringify({ type: 'code.autofix', request_id: requestId, code, language }))
  },

  // Select session
  selectSession: (sessionId) => {
    const { ws } = get()
    set({ activeSessionId: sessionId })
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ type: 'sessions.history', session_id: sessionId }))
    }
  },

  // Delete session
  deleteSession: (sessionId) => {
    const { sessions, messages, activeSessionId } = get()
    const newSessions = sessions.filter(s => s.id !== sessionId)
    const newMessages = { ...messages }
    delete newMessages[sessionId]
    const newActive = activeSessionId === sessionId
      ? (newSessions[0]?.id || null)
      : activeSessionId
    set({ sessions: newSessions, messages: newMessages, activeSessionId: newActive })
  },

  // ── Internal ──────────────────────────────────────────────────────────────

  _ensureSession: (sessionId) => {
    const { sessions } = get()
    if (!sessions.find(s => s.id === sessionId)) {
      set({
        sessions: [{ id: sessionId, title: 'New chat', ts: Date.now() }, ...sessions],
        messages: { ...get().messages, [sessionId]: [] },
      })
    }
  },

  _addMessage: (sessionId, message) => {
    const { messages, sessions } = get()
    const sessionMsgs = messages[sessionId] || []
    const updated = { ...messages, [sessionId]: [...sessionMsgs, message] }
    // Update session title from first user message
    const updatedSessions = sessions.map(s => {
      if (s.id === sessionId && s.title === 'New chat' && message.role === 'user') {
        return { ...s, title: message.content.slice(0, 46) }
      }
      return s
    })
    set({ messages: updated, sessions: updatedSessions })
  },

  _appendChunk: (sessionId, chunk) => {
    // ── Think-block streaming filter ──────────────────────────────────────
    if (!thinkState[sessionId]) thinkState[sessionId] = { inThink: false, pending: '' }
    const ts = thinkState[sessionId]

    // Accumulate and filter out everything inside <think>...</think>
    let visible = ''
    let buf = ts.pending + chunk

    while (buf.length > 0) {
      if (ts.inThink) {
        const endIdx = buf.indexOf('</think>')
        if (endIdx === -1) {
          // Still inside think block, nothing visible yet,  wait for more chunks
          ts.pending = buf
          buf = ''
        } else {
          // Found end of think block,  discard up to and including </think>
          buf = buf.slice(endIdx + '</think>'.length)
          ts.inThink = false
          ts.pending = ''
        }
      } else {
        const startIdx = buf.indexOf('<think>')
        if (startIdx === -1) {
          visible += buf
          buf = ''
        } else {
          visible += buf.slice(0, startIdx)
          buf = buf.slice(startIdx + '<think>'.length)
          ts.inThink = true
        }
      }
    }

    if (!visible) return // entire chunk was think-block content

    const { messages } = get()
    const sessionMsgs = messages[sessionId] || []
    const last = sessionMsgs[sessionMsgs.length - 1]
    if (last && last.role === 'assistant' && last.streaming) {
      const updated = [...sessionMsgs]
      updated[updated.length - 1] = { ...last, content: last.content + visible }
      set({ messages: { ...messages, [sessionId]: updated } })
    } else {
      get()._addMessage(sessionId, {
        id: uid(), role: 'assistant', content: visible, ts: Date.now(), streaming: true,
      })
    }
  },

  _finalizeMessage: (sessionId, finalText) => {
    // Clear think-block state for this session
    delete thinkState[sessionId]

    // Calculate total response duration
    const { streamingSince } = get()
    const responseDurationMs = streamingSince ? Date.now() - streamingSince : null

    // Strip any remaining think blocks from the final text
    const cleanText = finalText ? stripThinkBlocks(finalText) : null

    const { messages } = get()
    const sessionMsgs = messages[sessionId] || []
    const last = sessionMsgs[sessionMsgs.length - 1]
    if (last && last.role === 'assistant' && last.streaming) {
      const updated = [...sessionMsgs]
      // If server sends a complete final text, prefer that (already stripped)
      // Otherwise keep what we accumulated via chunks
      const finalContent = cleanText || last.content
      updated[updated.length - 1] = {
        ...last,
        content: stripThinkBlocks(finalContent),
        streaming: false,
        responseDurationMs,
      }
      set({ messages: { ...messages, [sessionId]: updated }, streaming: false, streamingSessionId: null, streamingSince: null, elapsed: 0, streamingPhase: 'thinking' })
    } else {
      if (cleanText) {
        get()._addMessage(sessionId, {
          id: uid(), role: 'assistant', content: cleanText, ts: Date.now(), streaming: false, responseDurationMs,
        })
      }
      set({ streaming: false, streamingSessionId: null, streamingSince: null, elapsed: 0, streamingPhase: 'thinking' })
    }
  },

  _handleMessage: (msg) => {
    const { _ensureSession, _appendChunk, _finalizeMessage, sessions } = get()

    switch (msg.type) {
      case 'connected':
        break

      case 'message.start':
        if (msg.session_id) {
          _ensureSession(msg.session_id)
          set({ streaming: true, streamingSessionId: msg.session_id, streamingPhase: 'thinking', agentProgress: 'Thinking…' })
        }
        break

      case 'message.chunk':
        if (msg.session_id && msg.text) {
          _ensureSession(msg.session_id)
          // Switch phase on first chunk
          if (get().streamingPhase !== 'responding') {
            set({ streamingPhase: 'responding', agentProgress: 'Responding…' })
          }
          _appendChunk(msg.session_id, msg.text)
        }
        break

      case 'message.done':
        if (msg.session_id) {
          set({ agentProgress: null })
          _finalizeMessage(msg.session_id, msg.text)
        }
        break

      case 'progress':
        if (msg.text) set({ agentProgress: msg.text })
        break

      case 'tools.list.result':
        if (msg.tools) set({ tools: msg.tools })
        break

      case 'sessions.list.result': {
        const remoteSessions = (msg.sessions || []).map(s => ({
          id: s.session_id || s.id,
          title: s.title || s.session_id || 'Chat',
          ts: s.created_at ? new Date(s.created_at).getTime() : Date.now(),
        }))
        // Keep local sessions that have actual messages (in-progress chats not yet on server)
        const { messages: localMsgs } = get()
        const localOnly = get().sessions.filter(
          ls => !remoteSessions.find(rs => rs.id === ls.id) &&
                (localMsgs[ls.id]?.length > 0)
        )
        set({ sessions: [...remoteSessions, ...localOnly] })
        break
      }

      case 'sessions.history.result': {
        if (!msg.session_id) break
        const msgs = (msg.messages || []).map(m => ({
          id: uid(),
          role: m.role === 'user' ? 'user' : 'assistant',
          content: m.content || m.text || '',
          ts: m.ts || Date.now(),
          streaming: false,
        }))
        set({ messages: { ...get().messages, [msg.session_id]: msgs } })
        break
      }

      case 'error':
        get().showToast(msg.text || 'Gateway error')
        set({ streaming: false, agentProgress: null, streamingSince: null, elapsed: 0, streamingPhase: 'thinking' })
        break

      case 'code.result': {
        const rid = msg.request_id
        if (rid) {
          set(prev => ({
            codeResults: {
              ...prev.codeResults,
              [rid]: { running: false, stdout: msg.stdout || '', stderr: msg.stderr || '', exit_code: msg.exit_code ?? 0 }
            }
          }))
        }
        break
      }

      case 'heartbeat':
      case 'pong':
        break
      case 'status': {
        if (msg.model) set({ agentModel: msg.model })
        if (msg.version) set({ agentVersion: msg.version })
        break
      }

      case 'agents.list.result': {
        const agentsList = msg.agents || []
        set({ agents: agentsList })
        // Auto-select current agent if none selected
        const { activeAgent } = get()
        if (!activeAgent && msg.current) {
          set({ activeAgent: msg.current })
        }
        break
      }

      case 'models.list.result':
        set({ modelGroups: msg.groups || [] })
        if (msg.current) set({ agentModel: msg.current })
        if (msg.provider) set({ activeProvider: msg.provider })
        break

      case 'models.set.result':
        if (msg.success) {
          if (msg.model) set({ agentModel: msg.model })
          if (msg.provider) set({ activeProvider: msg.provider })
          get().requestModels()
        }
        break

      case 'models.import.result':
        if (msg.success) {
          get().showToast(`Model "${msg.model}" imported successfully`)
          get().requestModels()
        } else {
          get().showToast(`Import failed: ${msg.error || 'unknown error'}`)
        }
        break

      case 'brain.data.result':
        set({ brainData: msg, brainLoading: false })
        break

      case 'permission.request':
        // Show confirmation dialog for RBAC "ask" permissions
        set({
          pendingPermission: {
            requestId: msg.requestId,
            action: msg.action,
            tool: msg.tool,
            arguments: msg.arguments || {},
            message: msg.message || `Allow ${msg.tool}?`,
          }
        })
        break
    }
  },

  // ── Model management ─────────────────────────────────────────────────────
  requestModels: () => {
    const { ws } = get()
    if (ws?.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ type: 'models.list' }))
    }
  },

  switchModel: (modelName, provider) => {
    const { ws } = get()
    if (ws?.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ type: 'models.set', model: modelName, provider: provider || '' }))
    }
  },

  importGGUF: (filePath, modelName) => {
    const { ws } = get()
    if (ws?.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ type: 'models.import', path: filePath, name: modelName }))
      get().showToast(`Importing "${modelName}"… this may take a moment`)
    }
  },

  // ── Brain data ───────────────────────────────────────────────────────────
  fetchBrain: () => {
    const { ws } = get()
    if (ws?.readyState === WebSocket.OPEN) {
      set({ brainLoading: true })
      ws.send(JSON.stringify({ type: 'brain.data' }))
    }
  },

  // ── Permission response ──────────────────────────────────────────────────
  respondPermission: (requestId, allowed, remember = false) => {
    const { ws, pendingPermission } = get()
    if (ws?.readyState === WebSocket.OPEN && requestId) {
      ws.send(JSON.stringify({
        type: 'permission.response',
        requestId,
        allowed,
        remember,
        action: pendingPermission?.action || '',
        userId: pendingPermission?.userId || 'default',
      }))
    }
    set({ pendingPermission: null })
  },
}))
