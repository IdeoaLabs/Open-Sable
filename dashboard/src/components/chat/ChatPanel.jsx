import { useState, useRef, useEffect } from 'react';
import { Send, Trash2, Plus, MessageSquare, ChevronLeft, ChevronRight } from 'lucide-react';
import { fmtTime } from '../../lib/utils';

/* ── Chat-area styles ──────────────────────────────────────────────────── */
const s = {
  panel: { display: 'flex', flex: 1, overflow: 'hidden' },
  sidebar: (collapsed) => ({
    width: collapsed ? 0 : 220, minWidth: collapsed ? 0 : 220,
    background: 'var(--bg-secondary)', borderRight: '1px solid var(--border)',
    display: 'flex', flexDirection: 'column', overflow: 'hidden',
    transition: 'width .15s, min-width .15s',
  }),
  sidebarHeader: {
    display: 'flex', alignItems: 'center', gap: 6, padding: '10px 12px',
    borderBottom: '1px solid var(--border)', flexShrink: 0,
  },
  newChatBtn: {
    flex: 1, padding: '6px 10px', borderRadius: 'var(--radius-sm)', border: 'none',
    background: 'var(--accent)', color: 'white', fontWeight: 600, fontSize: 11,
    cursor: 'pointer', display: 'flex', alignItems: 'center', gap: 4, justifyContent: 'center',
  },
  sessionList: { flex: 1, overflowY: 'auto', padding: '6px 0' },
  sessionItem: (active) => ({
    padding: '8px 12px', cursor: 'pointer', fontSize: 12,
    background: active ? 'var(--accent-dim)' : 'transparent',
    color: active ? 'var(--accent-light)' : 'var(--text-muted)',
    borderLeft: active ? '3px solid var(--accent)' : '3px solid transparent',
    display: 'flex', alignItems: 'center', gap: 8,
    transition: 'background .1s',
    whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis',
  }),
  sessionTitle: {
    flex: 1, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
    fontWeight: 600,
  },
  sessionTime: { fontSize: 9, color: 'var(--text-muted)', flexShrink: 0 },
  chatCol: { display: 'flex', flexDirection: 'column', flex: 1, overflow: 'hidden' },
  header: {
    display: 'flex', alignItems: 'center', gap: 8, padding: '12px 16px',
    borderBottom: '1px solid var(--border)', minHeight: 44, flexShrink: 0,
  },
  title: { fontSize: 13, fontWeight: 600, color: 'var(--text)' },
  meta: { marginLeft: 'auto', fontSize: 11, color: 'var(--text-muted)' },
  messages: { flex: 1, overflowY: 'auto', padding: 16 },
  empty: {
    display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100%',
    flexDirection: 'column', gap: 12,
  },
  msg: (user) => ({
    marginBottom: 16, display: 'flex', gap: 10,
    flexDirection: user ? 'row-reverse' : 'row',
  }),
  avatar: (user) => ({
    width: 32, height: 32, borderRadius: 'var(--radius-sm)', flexShrink: 0,
    display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: 14,
    background: user ? 'var(--accent-dim)' : 'var(--teal-dim)',
    color: user ? 'var(--accent-light)' : 'var(--teal)',
  }),
  msgBody: (user) => ({
    flex: 1, minWidth: 0, maxWidth: '80%',
    display: 'flex', flexDirection: 'column',
    alignItems: user ? 'flex-end' : 'flex-start',
  }),
  bubble: (user) => ({
    padding: '10px 14px', borderRadius: 'var(--radius)',
    fontSize: 13.5, lineHeight: 1.6, whiteSpace: 'pre-wrap',
    wordBreak: 'break-word', overflowWrap: 'break-word',
    ...(user
      ? { background: 'var(--accent)', color: 'white', borderBottomRightRadius: 4 }
      : { background: 'var(--bg-tertiary)', border: '1px solid var(--border)', borderBottomLeftRadius: 4 }),
  }),
  time: (user) => ({
    fontSize: 10, color: 'var(--text-muted)', marginTop: 4,
    textAlign: user ? 'right' : 'left',
  }),
  inputArea: {
    padding: '12px 16px', borderTop: '1px solid var(--border)',
    background: 'var(--bg-secondary)', flexShrink: 0,
  },
  row: { display: 'flex', gap: 8 },
  input: {
    flex: 1, padding: '10px 14px', borderRadius: 'var(--radius)',
    border: '1px solid var(--border)', background: 'var(--bg-tertiary)',
    color: 'var(--text)', fontFamily: 'var(--sans)', fontSize: 13.5,
    outline: 'none', resize: 'none',
  },
  sendBtn: {
    padding: '0 18px', borderRadius: 'var(--radius)', border: 'none',
    background: 'var(--accent)', color: 'white', fontWeight: 600, fontSize: 13,
    cursor: 'pointer', display: 'flex', alignItems: 'center',
  },
  iconBtn: {
    background: 'none', border: 'none', color: 'var(--text-muted)',
    cursor: 'pointer', padding: 4, display: 'flex', alignItems: 'center',
  },
};

function relativeTime(ts) {
  const now = Date.now();
  const diff = now - (typeof ts === 'string' ? new Date(ts).getTime() : ts);
  if (diff < 0) return 'just now';
  if (diff < 60000) return 'just now';
  if (diff < 3600000) return `${Math.floor(diff / 60000)}m`;
  if (diff < 86400000) return `${Math.floor(diff / 3600000)}h`;
  return new Date(ts).toLocaleDateString();
}

export default function ChatPanel({ messages, streaming, onSend, onClear, sessions, activeSessionId, onLoadSession, onNewChat, onDeleteSession }) {
  const [input, setInput] = useState('');
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [ctxMenu, setCtxMenu] = useState(null); // {x, y, sessionId}
  const endRef = useRef(null);

  // Close context menu on any click
  useEffect(() => {
    if (!ctxMenu) return;
    const close = () => setCtxMenu(null);
    window.addEventListener('click', close);
    return () => window.removeEventListener('click', close);
  }, [ctxMenu]);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const handleSubmit = (e) => {
    e.preventDefault();
    const text = input.trim();
    if (!text) return;
    onSend(text);
    setInput('');
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e);
    }
  };

  const activeTitle = sessions?.find(s => (s.session_id || s.id) === activeSessionId)?.title || 'Chat';

  return (
    <div style={s.panel}>
      {/* ── Session sidebar ──────────────────────────────────────────── */}
      <div style={s.sidebar(!sidebarOpen)}>
        {sidebarOpen && (
          <>
            <div style={s.sidebarHeader}>
              <button style={s.newChatBtn} onClick={onNewChat} title="New chat">
                <Plus size={13} /> New Chat
              </button>
            </div>
            <div style={s.sessionList}>
              {(sessions || []).map(sess => {
                const sid = sess.session_id || sess.id;
                return (
                  <div
                    key={sid}
                    style={s.sessionItem(sid === activeSessionId)}
                    onClick={() => onLoadSession?.(sid)}
                    onContextMenu={(e) => {
                      e.preventDefault();
                      setCtxMenu({ x: e.clientX, y: e.clientY, sessionId: sid });
                    }}
                    title={sess.title || sid}
                  >
                    <MessageSquare size={12} style={{ flexShrink: 0, opacity: 0.5 }} />
                    <span style={s.sessionTitle}>{sess.title || sid}</span>
                    <span style={s.sessionTime}>{relativeTime(sess.updated_at || sess.created_at)}</span>
                  </div>
                );
              })}
              {(!sessions || sessions.length === 0) && (
                <div style={{ padding: '20px 12px', fontSize: 11, color: 'var(--text-muted)', textAlign: 'center' }}>
                  No conversations yet
                </div>
              )}
            </div>
          </>
        )}
      </div>

      {/* ── Right-click context menu ─────────────────────────────────── */}
      {ctxMenu && (
        <div style={{
          position: 'fixed', left: ctxMenu.x, top: ctxMenu.y, zIndex: 9999,
          background: 'var(--bg-secondary)', border: '1px solid var(--border)',
          borderRadius: 'var(--radius-sm)', boxShadow: '0 4px 16px rgba(0,0,0,.3)',
          padding: '4px 0', minWidth: 140,
        }}>
          <button
            style={{
              width: '100%', padding: '8px 14px', border: 'none', background: 'none',
              color: 'var(--text)', fontSize: 12, cursor: 'pointer', textAlign: 'left',
              display: 'flex', alignItems: 'center', gap: 8,
            }}
            onMouseEnter={(e) => e.currentTarget.style.background = 'var(--bg-hover)'}
            onMouseLeave={(e) => e.currentTarget.style.background = 'none'}
            onClick={() => { onLoadSession?.(ctxMenu.sessionId); setCtxMenu(null); }}
          >
            <MessageSquare size={12} /> Open
          </button>
          <div style={{ height: 1, background: 'var(--border)', margin: '2px 0' }} />
          <button
            style={{
              width: '100%', padding: '8px 14px', border: 'none', background: 'none',
              color: '#ef4444', fontSize: 12, cursor: 'pointer', textAlign: 'left',
              display: 'flex', alignItems: 'center', gap: 8,
            }}
            onMouseEnter={(e) => e.currentTarget.style.background = 'var(--bg-hover)'}
            onMouseLeave={(e) => e.currentTarget.style.background = 'none'}
            onClick={() => { onDeleteSession?.(ctxMenu.sessionId); setCtxMenu(null); }}
          >
            <Trash2 size={12} /> Delete
          </button>
        </div>
      )}

      {/* ── Chat area ────────────────────────────────────────────────── */}
      <div style={s.chatCol}>
        <div style={s.header}>
          <button style={s.iconBtn} onClick={() => setSidebarOpen(v => !v)} title={sidebarOpen ? 'Hide sessions' : 'Show sessions'}>
            {sidebarOpen ? <ChevronLeft size={16} /> : <ChevronRight size={16} />}
          </button>
          <span style={{ fontSize: 16 }}>💬</span>
          <span style={s.title}>{activeTitle}</span>
          <span style={s.meta}>{messages.length} messages</span>
          <button onClick={() => { if (activeSessionId) onDeleteSession?.(activeSessionId); }} style={s.iconBtn} title="Delete chat">
            <Trash2 size={14} />
          </button>
        </div>

        <div style={s.messages}>
          {messages.length === 0 ? (
            <div style={s.empty}>
              <div style={{ fontSize: 48, opacity: .3 }}>🤖</div>
              <p style={{ color: 'var(--text-muted)', fontSize: 13 }}>Send a message to start chatting</p>
              <p style={{ color: 'var(--text-muted)', fontSize: 11 }}>Select a conversation from the sidebar or start a new one</p>
            </div>
          ) : (
            messages.map((m, i) => (
              <div key={i} style={s.msg(m.role === 'user')}>
                <div style={s.avatar(m.role === 'user')}>
                  {m.role === 'user' ? '👤' : '🤖'}
                </div>
                <div style={s.msgBody(m.role === 'user')}>
                  <div style={s.bubble(m.role === 'user')}>
                    {m.content}
                    {m._streaming && (
                      <span className="typing-dots" style={{ marginLeft: 6 }}>
                        <span>●</span><span>●</span><span>●</span>
                      </span>
                    )}
                  </div>
                  <div style={s.time(m.role === 'user')}>{fmtTime(m.ts)}</div>
                </div>
              </div>
            ))
          )}
          <div ref={endRef} />
        </div>

        <div style={s.inputArea}>
          <form onSubmit={handleSubmit}>
            <div style={s.row}>
              <textarea
                style={s.input}
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder="Type a message… (/ for commands, Shift+Enter for newline)"
                disabled={streaming}
                rows={1}
              />
              <button
                type="submit"
                style={{ ...s.sendBtn, opacity: streaming || !input.trim() ? 0.5 : 1 }}
                disabled={streaming || !input.trim()}
              >
                <Send size={16} />
              </button>
            </div>
          </form>
        </div>
      </div>
    </div>
  );
}
