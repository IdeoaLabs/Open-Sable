import { useState, useEffect, useRef, useCallback } from 'react';

function getWSUrl() {
  const proto = location.protocol === 'https:' ? 'wss' : 'ws';
  const params = new URLSearchParams(location.search);
  const token = params.get('token');
  return token
    ? `${proto}://${location.host}/?token=${encodeURIComponent(token)}`
    : `${proto}://${location.host}/`;
}

export function useWebSocket(onExternalMessage) {
  const [connected, setConnected] = useState(false);
  const [streaming, setStreaming] = useState(false);
  const [messages, setMessages] = useState([]);
  const [activity, setActivity] = useState([]);
  const [terminal, setTerminal] = useState([]);
  const [stats, setStats] = useState({});
  const [sessions, setSessions] = useState([]);
  const [model, setModel] = useState('');
  const sessionLoadCallbackRef = useRef(null);
  const [thoughts, setThoughts] = useState(null);
  const [brainData, setBrainData] = useState(null);
  const [modelGroups, setModelGroups] = useState([]);
  const [activeProvider, setActiveProvider] = useState('ollama');
  const [pendingPermission, setPendingPermission] = useState(null);
  const [activeSessionId, setActiveSessionId] = useState('webchat_default');

  const wsRef = useRef(null);
  const streamBuf = useRef('');
  const actIdRef = useRef(0);

  const addActivity = useCallback((type, icon, title, detail) => {
    const id = ++actIdRef.current;
    setActivity(prev => [{ id, type, icon, title, detail, ts: Date.now() / 1000 }, ...prev].slice(0, 200));
  }, []);

  const addTerminal = useCallback((text, cls = '') => {
    setTerminal(prev => [...prev, { text, cls }].slice(-500));
  }, []);

  const handleMonitorEvent = useCallback((event, data) => {
    switch (event) {
      case 'tool.start':
        addActivity('tool', '🔧', `Tool: ${data.name}`, data.args ? JSON.stringify(data.args).slice(0, 100) : '');
        addTerminal(`[tool.start] ${data.name}(${data.args ? JSON.stringify(data.args).slice(0, 80) : ''})`, 'cmd');
        break;
      case 'tool.done':
        if (data.success) {
          addActivity('success', '✅', `Tool done: ${data.name}`, `${data.duration_ms}ms`);
        } else {
          addActivity('error', '❌', `Tool failed: ${data.name}`, (data.result || '').slice(0, 80));
        }
        addTerminal(`[tool.done] ${data.name} → ${data.success ? 'OK' : 'FAIL'} (${data.duration_ms}ms)`, data.success ? '' : 'error');
        break;
      case 'thinking':
        addActivity('think', '🧠', 'Thinking', (data.message || '').slice(0, 100));
        addTerminal(`[thinking] Round ${data.round || '?'}: ${(data.message || '').slice(0, 100)}`, 'info');
        break;
      case 'reasoning':
        addActivity('think', '💭', 'Deep Reasoning', `${data.length} chars`);
        addTerminal(`[reasoning] (${data.length} chars)`, 'info');
        break;
      case 'thinking.done':
        addActivity('success', '💡', 'Thinking complete', '');
        break;
      case 'message.received':
        addActivity('info', '📥', 'Message received', (data.text || '').slice(0, 80));
        break;
      case 'response.sent':
        addActivity('success', '📤', 'Response sent', `${data.length} chars`);
        break;
      default:
        addActivity('info', '📌', event, JSON.stringify(data).slice(0, 80));
        addTerminal(`[${event}] ${JSON.stringify(data).slice(0, 120)}`, 'info');
        break;
    }
  }, [addActivity, addTerminal]);

  const handleMessage = useCallback((msg) => {
    // Let external handler (multi-agent) process first
    if (onExternalMessage && onExternalMessage(msg)) return;

    switch (msg.type) {
      case 'connected':
        break;
      case 'message.start':
        streamBuf.current = '';
        setStreaming(true);
        addActivity('info', '💬', 'Processing', 'Agent is thinking…');
        break;
      case 'message.chunk':
        streamBuf.current += msg.text;
        setMessages(prev => {
          const copy = [...prev];
          if (copy.length && copy[copy.length - 1]._streaming) {
            copy[copy.length - 1] = { ...copy[copy.length - 1], content: streamBuf.current };
          } else {
            copy.push({ role: 'assistant', content: streamBuf.current, ts: Date.now(), _streaming: true });
          }
          return copy;
        });
        break;
      case 'message.done':
        setStreaming(false);
        setMessages(prev => {
          const copy = [...prev];
          if (copy.length && copy[copy.length - 1]._streaming) {
            copy[copy.length - 1] = { role: 'assistant', content: msg.text, ts: Date.now() };
          } else {
            copy.push({ role: 'assistant', content: msg.text, ts: Date.now() });
          }
          return copy;
        });
        addActivity('success', '✅', 'Response sent', (msg.text || '').slice(0, 80));
        break;
      case 'progress':
        addActivity('info', '⚡', 'Progress', msg.text || '');
        addTerminal(`[progress] ${msg.text}`, 'info');
        break;
      case 'error':
        setStreaming(false);
        setMessages(prev => [...prev, { role: 'assistant', content: '⚠️ ' + msg.text, ts: Date.now() }]);
        addActivity('error', '❌', 'Error', msg.text);
        addTerminal(`[error] ${msg.text}`, 'error');
        break;
      case 'status':
        setStats(msg);
        if (msg.model) setModel(msg.model);
        break;
      case 'sessions.list.result':
        setSessions(msg.sessions || []);
        // Auto-load the most recent session if we haven't loaded one yet
        if (msg.sessions?.length > 0) {
          setActiveSessionId(prev => {
            if (prev === 'webchat_default') {
              const latest = msg.sessions[0];
              const sid = latest.session_id || latest.id;
              // Request its full history
              if (wsRef.current?.readyState === WebSocket.OPEN) {
                wsRef.current.send(JSON.stringify({ type: 'sessions.history', session_id: sid }));
              }
              return sid;
            }
            return prev;
          });
        }
        break;
      case 'sessions.delete.result':
        if (msg.success && msg.session_id) {
          // Remove the deleted session from state (server also sends sessions.list.result)
          setSessions(prev => prev.filter(s => (s.session_id || s.id) !== msg.session_id));
        }
        break;
      case 'sessions.history.result': {
        if (!msg.session_id || !msg.messages) break;
        const loaded = msg.messages.map(m => ({
          role: m.role === 'user' ? 'user' : 'assistant',
          content: m.content || m.text || '',
          ts: m.timestamp ? new Date(m.timestamp).getTime() : Date.now(),
        }));
        // If a callback was registered (e.g. remote agent load), use it instead
        if (sessionLoadCallbackRef.current) {
          sessionLoadCallbackRef.current(loaded, msg.session_id);
          sessionLoadCallbackRef.current = null;
        } else {
          setMessages(loaded);
        }
        addActivity('info', '📂', 'Session loaded', `${loaded.length} messages`);
        break;
      }
      case 'monitor.subscribed':
        addActivity('info', '📡', 'Monitor', 'Subscribed to real-time events');
        break;
      case 'monitor.event':
        handleMonitorEvent(msg.event, msg.data || {});
        break;
      case 'monitor.snapshot':
        if (msg.model) setModel(msg.model);
        break;
      case 'thoughts.list.result':
        setThoughts(msg);
        break;
      case 'brain.data.result':
        setBrainData(msg);
        break;
      case 'models.list.result':
        // If response is for a remote agent, let external handler process it
        if (msg._profile) {
          if (onExternalMessage) onExternalMessage(msg);
          break;
        }
        setModelGroups(msg.groups || []);
        if (msg.current) setModel(msg.current);
        if (msg.provider) setActiveProvider(msg.provider);
        break;
      case 'models.set.result':
        // If response is for a remote agent, let external handler process it
        if (msg._profile) {
          if (onExternalMessage) onExternalMessage(msg);
          break;
        }
        if (msg.success) {
          if (msg.model) setModel(msg.model);
          if (msg.provider) setActiveProvider(msg.provider);
          // Refresh model list after switch
          if (wsRef.current?.readyState === WebSocket.OPEN) {
            wsRef.current.send(JSON.stringify({ type: 'models.list' }));
          }
        }
        break;
      case 'models.import.result':
        if (msg.success) {
          addActivity('success', '📦', 'Model imported', msg.model || '');
          // Refresh model list
          if (wsRef.current?.readyState === WebSocket.OPEN) {
            wsRef.current.send(JSON.stringify({ type: 'models.list' }));
          }
        } else {
          addActivity('error', '❌', 'Import failed', msg.error || '');
        }
        break;
      case 'permission.request':
        setPendingPermission({
          requestId: msg.requestId,
          action: msg.action,
          tool: msg.tool,
          arguments: msg.arguments || {},
          message: msg.message || `Allow ${msg.tool}?`,
        });
        break;
      default:
        break;
    }
  }, [addActivity, addTerminal, handleMonitorEvent, onExternalMessage]);

  const connect = useCallback(() => {
    const ws = new WebSocket(getWSUrl());
    wsRef.current = ws;

    ws.onopen = () => {
      setConnected(true);
      addActivity('info', '🔗', 'Connected', 'WebSocket connection established');
      ws.send(JSON.stringify({ type: 'status' }));
      ws.send(JSON.stringify({ type: 'sessions.list' }));
      ws.send(JSON.stringify({ type: 'monitor.subscribe' }));
      ws.send(JSON.stringify({ type: 'thoughts.list', limit: 500 }));
      ws.send(JSON.stringify({ type: 'models.list' }));
    };

    ws.onclose = () => {
      setConnected(false);
      addActivity('error', '🔌', 'Disconnected', 'Reconnecting in 1s…');
      setTimeout(connect, 1000);
    };

    ws.onerror = () => {};
    ws.onmessage = (evt) => {
      try { handleMessage(JSON.parse(evt.data)); } catch {}
    };
  }, [addActivity, handleMessage]);

  useEffect(() => {
    connect();
    const iv = setInterval(() => {
      if (wsRef.current?.readyState === WebSocket.OPEN) {
        wsRef.current.send(JSON.stringify({ type: 'status' }));
      }
    }, 5000);
    return () => clearInterval(iv);
  }, [connect]);

  const sendMessage = useCallback((text) => {
    if (!text || !wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) return;
    setMessages(prev => [...prev, { role: 'user', content: text, ts: Date.now() }]);
    const sid = activeSessionId || 'webchat_default';
    if (text.startsWith('/')) {
      wsRef.current.send(JSON.stringify({ type: 'command', text, session_id: sid }));
    } else {
      wsRef.current.send(JSON.stringify({ type: 'message', text, session_id: sid, user_id: 'dashboard_user' }));
    }
  }, [activeSessionId]);

  const loadSession = useCallback((sessionId, onLoaded) => {
    if (!sessionId) {
      // New chat — generate a fresh session ID
      const fresh = 'dash_' + Date.now().toString(36) + '_' + Math.random().toString(36).slice(2, 6);
      setActiveSessionId(fresh);
      if (onLoaded) onLoaded([], null);
      else setMessages([]);
      return;
    }
    if (onLoaded) {
      sessionLoadCallbackRef.current = onLoaded;
    } else {
      setActiveSessionId(sessionId);
      setMessages([]);
    }
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ type: 'sessions.history', session_id: sessionId }));
    }
  }, []);

  const deleteSession = useCallback((sessionId) => {
    if (!sessionId || !wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) return;
    wsRef.current.send(JSON.stringify({ type: 'sessions.delete', session_id: sessionId }));
    // Optimistically remove from the local sessions list and switch to next session
    setSessions(prev => {
      const remaining = prev.filter(s => (s.session_id || s.id) !== sessionId);
      // If we deleted the active session, switch to the next one or start fresh
      setActiveSessionId(currentId => {
        if (currentId !== sessionId) return currentId;
        if (remaining.length > 0) {
          const next = remaining[0];
          const nextId = next.session_id || next.id;
          // Load the next session's history
          setMessages([]);
          if (wsRef.current?.readyState === WebSocket.OPEN) {
            wsRef.current.send(JSON.stringify({ type: 'sessions.history', session_id: nextId }));
          }
          return nextId;
        }
        // No sessions left — new chat
        setMessages([]);
        return 'dash_' + Date.now().toString(36) + '_' + Math.random().toString(36).slice(2, 6);
      });
      return remaining;
    });
  }, []);

  const clearMessages = useCallback(() => setMessages([]), []);
  const clearActivity = useCallback(() => setActivity([]), []);
  const clearTerminal = useCallback(() => setTerminal([]), []);

  const requestModels = useCallback((profile) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ type: 'models.list', ...(profile ? { profile } : {}) }));
    }
  }, []);

  const switchModel = useCallback((modelName, provider, profile) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ type: 'models.set', model: modelName, provider: provider || '', ...(profile ? { profile } : {}) }));
    }
  }, []);

  const importGGUF = useCallback((filePath, modelName) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ type: 'models.import', path: filePath, name: modelName || '' }));
    }
  }, []);

  const respondPermission = useCallback((requestId, allowed, remember = false) => {
    if (wsRef.current?.readyState === WebSocket.OPEN && requestId) {
      wsRef.current.send(JSON.stringify({
        type: 'permission.response',
        requestId,
        allowed,
        remember,
        action: pendingPermission?.action || '',
      }));
    }
    setPendingPermission(null);
  }, [pendingPermission]);

  return {
    connected, streaming, messages, activity, terminal, stats, sessions, model, thoughts, brainData,
    modelGroups, activeProvider, pendingPermission, activeSessionId,
    sendMessage, loadSession, deleteSession, clearMessages, clearActivity, clearTerminal, requestModels, switchModel, importGGUF, respondPermission, wsRef,
  };
}
