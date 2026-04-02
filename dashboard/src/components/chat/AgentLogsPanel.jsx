import { useRef, useEffect, useState, useCallback } from 'react';
import { ChevronRight, ChevronLeft, Trash2, Pause, Play } from 'lucide-react';
import { fmtTime } from '../../lib/utils';

const ICON_MAP = {
  tool: '🔧', think: '🧠', success: '✅', error: '❌', info: '📌',
};

const COLOR_MAP = {
  tool:    'var(--accent-dim)',
  think:   'var(--yellow-dim)',
  success: 'var(--green-dim)',
  error:   'var(--red-dim)',
  info:    'var(--teal-dim)',
};

const TEXT_COLOR = {
  tool:    'var(--accent-light)',
  think:   'var(--yellow)',
  success: 'var(--green)',
  error:   'var(--red)',
  info:    'var(--teal)',
};

const s = {
  panel: {
    width: 320, minWidth: 260, maxWidth: 420,
    display: 'flex', flexDirection: 'column',
    borderLeft: '1px solid var(--border)',
    background: 'var(--bg-primary)',
    overflow: 'hidden', flexShrink: 0,
    transition: 'width .2s ease',
  },
  collapsed: {
    width: 36, minWidth: 36, maxWidth: 36,
    display: 'flex', flexDirection: 'column',
    borderLeft: '1px solid var(--border)',
    background: 'var(--bg-secondary)',
    alignItems: 'center', paddingTop: 8,
    cursor: 'pointer', flexShrink: 0,
  },
  header: {
    display: 'flex', alignItems: 'center', gap: 6, padding: '8px 12px',
    borderBottom: '1px solid var(--border)', minHeight: 40, flexShrink: 0,
  },
  title: { fontSize: 12, fontWeight: 600, flex: 1 },
  btn: {
    background: 'none', border: 'none', color: 'var(--text-muted)',
    cursor: 'pointer', padding: 3, display: 'flex', alignItems: 'center',
    borderRadius: 4,
  },
  body: { flex: 1, overflowY: 'auto', padding: 0 },
  entry: {
    padding: '6px 12px', borderBottom: '1px solid var(--border)',
    fontSize: 11.5, display: 'flex', gap: 8, alignItems: 'flex-start',
  },
  icon: (type) => ({
    width: 22, height: 22, borderRadius: 4, flexShrink: 0,
    display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: 11,
    background: COLOR_MAP[type] || COLOR_MAP.info,
    color: TEXT_COLOR[type] || TEXT_COLOR.info,
  }),
  text: { flex: 1, minWidth: 0 },
  entryTitle: { fontWeight: 600, color: 'var(--text)', fontSize: 11, marginBottom: 1 },
  detail: {
    color: 'var(--text-muted)', fontSize: 10.5, lineHeight: 1.4,
    overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
  },
  time: { fontSize: 9, color: 'var(--text-muted)', whiteSpace: 'nowrap', flexShrink: 0, marginTop: 2 },
  termLine: {
    padding: '2px 12px', fontFamily: 'var(--mono)', fontSize: 10.5,
    lineHeight: 1.5, whiteSpace: 'pre-wrap', wordBreak: 'break-all',
  },
  tabBar: {
    display: 'flex', borderBottom: '1px solid var(--border)', flexShrink: 0,
  },
  tab: (active) => ({
    flex: 1, padding: '6px 0', fontSize: 11, fontWeight: active ? 600 : 400,
    color: active ? 'var(--accent-light)' : 'var(--text-muted)',
    borderBottom: active ? '2px solid var(--accent)' : '2px solid transparent',
    background: 'none', border: 'none', cursor: 'pointer', textAlign: 'center',
  }),
  empty: {
    padding: 20, textAlign: 'center', color: 'var(--text-muted)', fontSize: 11,
  },
  vertLabel: {
    writingMode: 'vertical-rl', textOrientation: 'mixed',
    fontSize: 11, fontWeight: 600, color: 'var(--text-muted)',
    letterSpacing: 1, marginTop: 12,
  },
};

const TERM_COLORS = { error: 'var(--red)', info: 'var(--text-muted)', cmd: 'var(--teal)' };

export default function AgentLogsPanel({ activity, terminal, agentName, ws, agentProfile }) {
  const [open, setOpen] = useState(true);
  const [tab, setTab] = useState('activity');  // activity | terminal | file
  const [paused, setPaused] = useState(false);
  const endRef = useRef(null);
  const [frozen, setFrozen] = useState({ activity: [], terminal: [] });
  const [fileLines, setFileLines] = useState([]);

  // Fetch log file on tab switch / interval
  useEffect(() => {
    if (tab !== 'file' || !ws?.current) return;
    const fetch = () => {
      try {
        ws.current.send(JSON.stringify({
          type: 'logs.tail', profile: agentProfile || '', lines: 200,
        }));
      } catch (_) {}
    };
    fetch();
    const iv = setInterval(fetch, 5000);
    return () => clearInterval(iv);
  }, [tab, ws, agentProfile]);

  // Listen for logs.tail.result
  useEffect(() => {
    if (!ws?.current) return;
    const handler = (e) => {
      try {
        const msg = JSON.parse(e.data);
        if (msg.type === 'logs.tail.result') {
          setFileLines(msg.lines || []);
        }
      } catch (_) {}
    };
    ws.current.addEventListener('message', handler);
    return () => ws.current?.removeEventListener('message', handler);
  }, [ws]);

  // Auto-scroll
  useEffect(() => {
    if (!paused) endRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [activity, terminal, paused]);

  // Freeze data when paused
  useEffect(() => {
    if (paused) {
      setFrozen({ activity: [...(activity || [])], terminal: [...(terminal || [])] });
    }
  }, [paused]);

  const displayActivity = paused ? frozen.activity : (activity || []);
  const displayTerminal = paused ? frozen.terminal : (terminal || []);

  if (!open) {
    return (
      <div style={s.collapsed} onClick={() => setOpen(true)} title="Show agent logs">
        <ChevronLeft size={14} style={{ color: 'var(--text-muted)' }} />
        <div style={s.vertLabel}>LOGS</div>
      </div>
    );
  }

  return (
    <div style={s.panel}>
      <div style={s.header}>
        <span style={{ fontSize: 13 }}>📋</span>
        <span style={s.title}>{agentName ? `${agentName} logs` : 'Agent Logs'}</span>
        <button style={s.btn} onClick={() => setPaused(p => !p)} title={paused ? 'Resume' : 'Pause'}>
          {paused ? <Play size={13} /> : <Pause size={13} />}
        </button>
        <button style={s.btn} onClick={() => setOpen(false)} title="Collapse">
          <ChevronRight size={14} />
        </button>
      </div>

      <div style={s.tabBar}>
        <button style={s.tab(tab === 'activity')} onClick={() => setTab('activity')}>
          Activity ({displayActivity.length})
        </button>
        <button style={s.tab(tab === 'terminal')} onClick={() => setTab('terminal')}>
          Terminal ({displayTerminal.length})
        </button>
        <button style={s.tab(tab === 'file')} onClick={() => setTab('file')}>
          File Log
        </button>
      </div>

      <div style={s.body}>
        {tab === 'activity' ? (
          displayActivity.length === 0 ? (
            <div style={s.empty}>No activity yet. Agent events will appear here in real-time.</div>
          ) : (
            displayActivity.map(a => (
              <div key={a.id} style={s.entry}>
                <div style={s.icon(a.type)}>{a.icon || ICON_MAP[a.type] || '📌'}</div>
                <div style={s.text}>
                  <div style={s.entryTitle}>{a.title}</div>
                  {a.detail && <div style={s.detail} title={a.detail}>{a.detail}</div>}
                </div>
                <div style={s.time}>{fmtTime(a.ts)}</div>
              </div>
            ))
          )
        ) : tab === 'terminal' ? (
          displayTerminal.length === 0 ? (
            <div style={s.empty}>No terminal output yet.</div>
          ) : (
            displayTerminal.map((l, i) => (
              <div key={i} style={{ ...s.termLine, color: TERM_COLORS[l.cls] || 'var(--green)' }}>
                {l.text}
              </div>
            ))
          )
        ) : (
          fileLines.length === 0 ? (
            <div style={s.empty}>No log file data. Logs appear when the agent runs.</div>
          ) : (
            fileLines.map((line, i) => (
              <div key={i} style={{
                ...s.termLine,
                color: /error|fail|exception/i.test(line) ? 'var(--red)'
                     : /warn/i.test(line) ? 'var(--yellow)'
                     : /info/i.test(line) ? 'var(--text-muted)'
                     : 'var(--green)',
                fontSize: 10,
              }}>
                {line}
              </div>
            ))
          )
        )}
        <div ref={endRef} />
      </div>
    </div>
  );
}
