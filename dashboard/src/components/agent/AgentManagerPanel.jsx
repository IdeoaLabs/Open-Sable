import { useState, useEffect, useCallback } from 'react';
import {
  Users, Plus, Trash2, Play, Square, RefreshCw, Loader2,
  FileText, Settings, Check, AlertCircle, X, ChevronDown, ChevronRight,
  Copy, Zap,
} from 'lucide-react';

function apiUrl(path) {
  const token = new URLSearchParams(location.search).get('token');
  return `${location.protocol}//${location.host}${path}${token ? `?token=${encodeURIComponent(token)}` : ''}`;
}

/* ── Create agent dialog ──────────────────────────────────────── */
function CreateDialog({ onCreated, onCancel }) {
  const [name, setName] = useState('');
  const [soul, setSoul] = useState('');
  const [creating, setCreating] = useState(false);
  const [error, setError] = useState('');

  const submit = async () => {
    setError('');
    setCreating(true);
    try {
      const res = await fetch(apiUrl('/api/agents'), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name: name.trim().toLowerCase(), soul }),
      });
      const data = await res.json();
      if (data.ok) {
        onCreated(data.name);
      } else {
        setError(data.error || 'Failed to create');
      }
    } catch {
      setError('Network error');
    } finally {
      setCreating(false);
    }
  };

  const nameValid = /^[a-z][a-z0-9\-]{0,30}$/.test(name.trim().toLowerCase());

  return (
    <div style={{
      background: 'var(--bg-secondary)', border: '1px solid var(--accent-dim)',
      borderRadius: 10, padding: 16, display: 'flex', flexDirection: 'column', gap: 12,
      marginBottom: 16,
    }}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        <span style={{ fontSize: 13, fontWeight: 700, color: 'var(--accent-light)', display: 'flex', alignItems: 'center', gap: 6 }}>
          <Zap size={14} /> Create New Agent
        </span>
        <button onClick={onCancel} style={{
          background: 'transparent', border: 'none', color: 'var(--text-muted)', cursor: 'pointer', padding: 2, display: 'flex',
        }}>
          <X size={14} />
        </button>
      </div>

      <div>
        <label style={{ fontSize: 10, fontWeight: 600, color: 'var(--text-secondary)', display: 'block', marginBottom: 4 }}>
          AGENT NAME
        </label>
        <input value={name} onChange={e => setName(e.target.value)}
          placeholder="my-agent" spellCheck={false}
          style={{
            width: '100%', padding: '8px 10px', borderRadius: 6,
            border: `1px solid ${name && !nameValid ? 'var(--red, #ef4444)' : 'var(--border)'}`,
            background: 'var(--bg-primary)', color: 'var(--text)',
            fontSize: 13, fontFamily: 'var(--mono)', outline: 'none',
            boxSizing: 'border-box',
          }} />
        <span style={{ fontSize: 9, color: 'var(--text-muted)', marginTop: 2, display: 'block' }}>
          Lowercase letters, numbers, hyphens. This becomes the profile folder name.
        </span>
      </div>

      <div>
        <label style={{ fontSize: 10, fontWeight: 600, color: 'var(--text-secondary)', display: 'block', marginBottom: 4 }}>
          SOUL (PERSONALITY)
        </label>
        <textarea value={soul} onChange={e => setSoul(e.target.value)}
          placeholder="Define who this agent is, its personality, values, voice..."
          rows={6} spellCheck={false}
          style={{
            width: '100%', padding: '8px 10px', borderRadius: 6,
            border: '1px solid var(--border)', background: 'var(--bg-primary)',
            color: 'var(--text)', fontSize: 12, fontFamily: 'var(--mono)',
            outline: 'none', resize: 'vertical', lineHeight: 1.6,
            boxSizing: 'border-box',
          }} />
        <span style={{ fontSize: 9, color: 'var(--text-muted)', marginTop: 2, display: 'block' }}>
          Optional. You can edit the soul.md later. Leave blank for template default.
        </span>
      </div>

      {error && (
        <span style={{ fontSize: 11, color: '#f87171', display: 'flex', alignItems: 'center', gap: 4 }}>
          <AlertCircle size={12} /> {error}
        </span>
      )}

      <div style={{ display: 'flex', gap: 8, justifyContent: 'flex-end' }}>
        <button onClick={onCancel} style={{
          padding: '6px 14px', borderRadius: 6, border: '1px solid var(--border)',
          background: 'transparent', color: 'var(--text-secondary)', cursor: 'pointer',
          fontSize: 11,
        }}>
          Cancel
        </button>
        <button onClick={submit} disabled={creating || !nameValid}
          style={{
            padding: '6px 16px', borderRadius: 6, border: 'none',
            background: nameValid ? 'var(--accent)' : 'var(--bg-tertiary)',
            color: nameValid ? '#fff' : 'var(--text-muted)',
            cursor: nameValid ? 'pointer' : 'default',
            fontSize: 11, fontWeight: 600, display: 'flex', alignItems: 'center', gap: 4,
          }}>
          {creating ? <Loader2 size={12} className="spin" /> : <Plus size={12} />}
          Create Agent
        </button>
      </div>
    </div>
  );
}

/* ── Agent card ───────────────────────────────────────────────── */
function AgentCard({ agent, onStartStop, onDelete, onRefresh }) {
  const [expanded, setExpanded] = useState(false);
  const [soul, setSoul] = useState(null);
  const [soulEdit, setSoulEdit] = useState('');
  const [soulSaving, setSoulSaving] = useState(false);
  const [soulStatus, setSoulStatus] = useState(null);
  const [deleting, setDeleting] = useState(false);
  const [confirmDelete, setConfirmDelete] = useState(false);

  const loadSoul = useCallback(async () => {
    try {
      const res = await fetch(apiUrl(`/api/agents/soul?profile=${encodeURIComponent(agent.name)}`));
      const data = await res.json();
      setSoul(data.content || '');
      setSoulEdit(data.content || '');
    } catch {}
  }, [agent.name]);

  useEffect(() => {
    if (expanded && soul === null) loadSoul();
  }, [expanded, soul, loadSoul]);

  const saveSoul = async () => {
    setSoulSaving(true);
    setSoulStatus(null);
    try {
      const res = await fetch(apiUrl('/api/agents/soul'), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ profile: agent.name, content: soulEdit }),
      });
      const data = await res.json();
      if (data.ok) {
        setSoul(soulEdit);
        setSoulStatus({ type: 'ok', msg: 'Saved' });
        setTimeout(() => setSoulStatus(null), 2000);
      } else {
        setSoulStatus({ type: 'error', msg: data.error });
      }
    } catch {
      setSoulStatus({ type: 'error', msg: 'Network error' });
    } finally {
      setSoulSaving(false);
    }
  };

  const handleDelete = async () => {
    setDeleting(true);
    try {
      const res = await fetch(apiUrl('/api/agents'), {
        method: 'DELETE',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name: agent.name }),
      });
      const data = await res.json();
      if (data.ok) {
        onDelete(agent.name);
      }
    } finally {
      setDeleting(false);
      setConfirmDelete(false);
    }
  };

  return (
    <div style={{
      background: 'var(--bg-secondary)',
      border: `1px solid ${agent.is_current ? 'var(--accent-dim)' : 'var(--border)'}`,
      borderRadius: 10, overflow: 'hidden', transition: 'all .15s',
    }}>
      {/* Header */}
      <div style={{
        padding: '12px 14px', display: 'flex', alignItems: 'center', gap: 10,
        cursor: 'pointer',
      }} onClick={() => setExpanded(!expanded)}>
        {expanded
          ? <ChevronDown size={13} style={{ color: 'var(--text-muted)', flexShrink: 0 }} />
          : <ChevronRight size={13} style={{ color: 'var(--text-muted)', flexShrink: 0 }} />}

        <span style={{
          width: 8, height: 8, borderRadius: '50%', flexShrink: 0,
          background: agent.running ? 'var(--green, #22c55e)' : 'var(--text-muted)',
          boxShadow: agent.running ? '0 0 8px rgba(34,197,94,.5)' : 'none',
        }} />

        <span style={{
          fontSize: 14, fontWeight: 700, color: 'var(--text)', flex: 1,
          textTransform: 'capitalize',
        }}>
          {agent.name}
        </span>

        {agent.is_current && (
          <span style={{
            fontSize: 9, padding: '2px 8px', borderRadius: 99,
            background: 'var(--accent-dim)', color: 'var(--accent-light)',
            fontWeight: 700, letterSpacing: '.04em',
          }}>
            CURRENT
          </span>
        )}

        <span style={{
          fontSize: 10, padding: '2px 8px', borderRadius: 99,
          background: agent.running ? 'rgba(34,197,94,.12)' : 'rgba(255,255,255,.06)',
          color: agent.running ? '#4ade80' : 'var(--text-muted)',
          fontWeight: 600,
        }}>
          {agent.running ? 'Online' : 'Offline'}
        </span>

        {!agent.is_current && (
          <button onClick={e => { e.stopPropagation(); onStartStop(agent); }}
            style={{
              background: 'transparent', border: '1px solid var(--border)', borderRadius: 6,
              color: agent.running ? '#f87171' : '#4ade80', cursor: 'pointer',
              padding: '3px 10px', display: 'flex', alignItems: 'center', gap: 4,
              fontSize: 10, fontWeight: 600,
            }}>
            {agent.running ? <><Square size={10} /> Stop</> : <><Play size={10} /> Start</>}
          </button>
        )}
      </div>

      {/* Expanded */}
      {expanded && (
        <div style={{
          padding: '0 14px 14px', display: 'flex', flexDirection: 'column', gap: 10,
          borderTop: '1px solid var(--border)',
        }}>
          {/* Info badges */}
          <div style={{ display: 'flex', gap: 6, marginTop: 10, flexWrap: 'wrap' }}>
            <span style={{
              fontSize: 9, padding: '2px 7px', borderRadius: 99,
              background: agent.has_soul ? 'rgba(34,197,94,.1)' : 'rgba(239,68,68,.1)',
              color: agent.has_soul ? '#4ade80' : '#f87171',
            }}>
              <FileText size={8} style={{ verticalAlign: '-1px', marginRight: 2 }} />
              soul.md
            </span>
            <span style={{
              fontSize: 9, padding: '2px 7px', borderRadius: 99,
              background: agent.has_env ? 'rgba(34,197,94,.1)' : 'rgba(239,68,68,.1)',
              color: agent.has_env ? '#4ade80' : '#f87171',
            }}>
              <Settings size={8} style={{ verticalAlign: '-1px', marginRight: 2 }} />
              profile.env
            </span>
            <span style={{
              fontSize: 9, padding: '2px 7px', borderRadius: 99,
              background: agent.has_tools ? 'rgba(34,197,94,.1)' : 'rgba(239,68,68,.1)',
              color: agent.has_tools ? '#4ade80' : '#f87171',
            }}>
              <Zap size={8} style={{ verticalAlign: '-1px', marginRight: 2 }} />
              tools.json
            </span>
          </div>

          {/* Soul editor */}
          {soul !== null && (
            <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                <span style={{ fontSize: 10, fontWeight: 700, color: 'var(--text-secondary)' }}>SOUL.MD</span>
                {soulStatus && (
                  <span style={{ fontSize: 10, color: soulStatus.type === 'ok' ? '#4ade80' : '#f87171', display: 'flex', alignItems: 'center', gap: 3 }}>
                    {soulStatus.type === 'ok' ? <Check size={10} /> : <AlertCircle size={10} />}
                    {soulStatus.msg}
                  </span>
                )}
              </div>
              <textarea value={soulEdit} onChange={e => setSoulEdit(e.target.value)}
                rows={6} spellCheck={false}
                style={{
                  width: '100%', padding: '8px 10px', borderRadius: 6,
                  border: '1px solid var(--border)', background: 'var(--bg-primary)',
                  color: 'var(--text)', fontSize: 11, fontFamily: 'var(--mono)',
                  outline: 'none', resize: 'vertical', lineHeight: 1.6,
                  boxSizing: 'border-box',
                }} />
              {soulEdit !== soul && (
                <button onClick={saveSoul} disabled={soulSaving}
                  style={{
                    alignSelf: 'flex-end', padding: '4px 12px', borderRadius: 6,
                    border: 'none', background: 'var(--accent)', color: '#fff',
                    cursor: 'pointer', fontSize: 10, fontWeight: 600,
                    display: 'flex', alignItems: 'center', gap: 4,
                  }}>
                  {soulSaving ? <Loader2 size={10} className="spin" /> : <Check size={10} />}
                  Save Soul
                </button>
              )}
            </div>
          )}

          {/* Delete */}
          {!agent.is_current && agent.name !== 'sable' && (
            <div style={{ borderTop: '1px solid var(--border)', paddingTop: 10, marginTop: 4 }}>
              {!confirmDelete ? (
                <button onClick={() => setConfirmDelete(true)}
                  style={{
                    padding: '5px 12px', borderRadius: 6, border: '1px solid rgba(239,68,68,.3)',
                    background: 'transparent', color: '#f87171', cursor: 'pointer',
                    fontSize: 10, fontWeight: 600, display: 'flex', alignItems: 'center', gap: 4,
                  }}>
                  <Trash2 size={10} /> Delete Agent
                </button>
              ) : (
                <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                  <span style={{ fontSize: 11, color: '#f87171' }}>Delete "{agent.name}" permanently?</span>
                  <button onClick={handleDelete} disabled={deleting}
                    style={{
                      padding: '4px 12px', borderRadius: 6, border: 'none',
                      background: '#ef4444', color: '#fff', cursor: 'pointer',
                      fontSize: 10, fontWeight: 600, display: 'flex', alignItems: 'center', gap: 4,
                    }}>
                    {deleting ? <Loader2 size={10} className="spin" /> : <Trash2 size={10} />}
                    Confirm
                  </button>
                  <button onClick={() => setConfirmDelete(false)}
                    style={{
                      padding: '4px 12px', borderRadius: 6, border: '1px solid var(--border)',
                      background: 'transparent', color: 'var(--text-secondary)', cursor: 'pointer',
                      fontSize: 10,
                    }}>
                    Cancel
                  </button>
                </div>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

/* ── Main panel ───────────────────────────────────────────────── */
export default function AgentManagerPanel({ onStartAgent, onStopAgent }) {
  const [agents, setAgents] = useState([]);
  const [loading, setLoading] = useState(true);
  const [showCreate, setShowCreate] = useState(false);

  const load = useCallback(async () => {
    setLoading(true);
    try {
      const res = await fetch(apiUrl('/api/agents'));
      const data = await res.json();
      setAgents(data.agents || []);
    } catch {}
    setLoading(false);
  }, []);

  useEffect(() => { load(); }, [load]);

  const handleCreated = (name) => {
    setShowCreate(false);
    load();
  };

  const handleDelete = (name) => {
    setAgents(prev => prev.filter(a => a.name !== name));
  };

  const handleStartStop = (agent) => {
    if (agent.running) {
      onStopAgent?.(agent.name);
    } else {
      onStartAgent?.(agent.name);
    }
    // Refresh after a short delay to get updated status
    setTimeout(load, 2000);
  };

  return (
    <div style={{
      flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden',
      background: 'var(--bg-primary)',
    }}>
      {/* Toolbar */}
      <div style={{
        padding: '10px 16px', display: 'flex', alignItems: 'center', gap: 10,
        borderBottom: '1px solid var(--border)', background: 'var(--bg-secondary)',
        flexShrink: 0,
      }}>
        <Users size={16} style={{ color: 'var(--accent-light)' }} />
        <span style={{ fontSize: 13, fontWeight: 700, color: 'var(--text)', letterSpacing: '.02em' }}>
          Agent Manager
        </span>
        <span style={{
          fontSize: 10, padding: '2px 8px', borderRadius: 99,
          background: 'var(--accent-dim)', color: 'var(--accent-light)', fontWeight: 600,
        }}>
          {agents.length} agent{agents.length !== 1 ? 's' : ''}
        </span>
        <span style={{
          fontSize: 10, padding: '2px 8px', borderRadius: 99,
          background: 'rgba(34,197,94,.12)', color: '#4ade80', fontWeight: 600,
        }}>
          {agents.filter(a => a.running).length} running
        </span>

        <div style={{ flex: 1 }} />

        <button onClick={load} disabled={loading} title="Refresh"
          style={{
            background: 'transparent', border: '1px solid var(--border)', borderRadius: 6,
            color: 'var(--text-secondary)', cursor: 'pointer', padding: '4px 8px',
            display: 'flex', alignItems: 'center', gap: 4, fontSize: 11,
          }}>
          {loading ? <Loader2 size={12} className="spin" /> : <RefreshCw size={12} />}
        </button>

        <button onClick={() => setShowCreate(true)}
          style={{
            background: 'var(--accent)', border: 'none', borderRadius: 6,
            color: '#fff', cursor: 'pointer', padding: '4px 14px',
            display: 'flex', alignItems: 'center', gap: 4, fontSize: 11, fontWeight: 600,
          }}>
          <Plus size={12} /> New Agent
        </button>
      </div>

      {/* Content */}
      <div style={{ flex: 1, overflow: 'auto', padding: 16 }}>
        {showCreate && <CreateDialog onCreated={handleCreated} onCancel={() => setShowCreate(false)} />}

        {loading ? (
          <div style={{
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            color: 'var(--text-muted)', fontSize: 13, gap: 8, padding: 40,
          }}>
            <Loader2 size={16} className="spin" /> Loading agents...
          </div>
        ) : agents.length === 0 ? (
          <div style={{
            textAlign: 'center', color: 'var(--text-muted)', fontSize: 12, padding: 40,
          }}>
            No agents found. Click "New Agent" to create one.
          </div>
        ) : (
          <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
            {agents.map(a => (
              <AgentCard
                key={a.name}
                agent={a}
                onStartStop={handleStartStop}
                onDelete={handleDelete}
                onRefresh={load}
              />
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
