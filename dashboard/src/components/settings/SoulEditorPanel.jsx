import { useState, useEffect, useCallback } from 'react';
import { FileText, Save, RefreshCw, Check, AlertCircle, Loader2 } from 'lucide-react';

function apiUrl(path) {
  const token = new URLSearchParams(location.search).get('token');
  return `${location.protocol}//${location.host}${path}${token ? `?token=${encodeURIComponent(token)}` : ''}`;
}

export default function SoulEditorPanel() {
  const [content, setContent] = useState('');
  const [saved, setSaved] = useState('');
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [status, setStatus] = useState(null); // { type: 'ok'|'error', msg }
  const [profile, setProfile] = useState('');

  const load = useCallback(async () => {
    setLoading(true);
    setStatus(null);
    try {
      const res = await fetch(apiUrl('/api/soul'));
      const data = await res.json();
      setContent(data.content || '');
      setSaved(data.content || '');
      setProfile(data.profile || '');
    } catch (e) {
      setStatus({ type: 'error', msg: 'Failed to load soul.md' });
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => { load(); }, [load]);

  const save = useCallback(async () => {
    setSaving(true);
    setStatus(null);
    try {
      const res = await fetch(apiUrl('/api/soul'), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ content }),
      });
      const data = await res.json();
      if (data.ok) {
        setSaved(content);
        setStatus({ type: 'ok', msg: 'Saved' });
        setTimeout(() => setStatus(null), 2000);
      } else {
        setStatus({ type: 'error', msg: data.error || 'Save failed' });
      }
    } catch (e) {
      setStatus({ type: 'error', msg: 'Network error' });
    } finally {
      setSaving(false);
    }
  }, [content]);

  const hasChanges = content !== saved;

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
        <FileText size={16} style={{ color: 'var(--accent-light)' }} />
        <span style={{ fontSize: 13, fontWeight: 700, color: 'var(--text)', letterSpacing: '.02em' }}>
          Soul Editor
        </span>
        {profile && (
          <span style={{
            fontSize: 10, padding: '2px 8px', borderRadius: 99,
            background: 'var(--accent-dim)', color: 'var(--accent-light)',
            fontWeight: 600, letterSpacing: '.03em',
          }}>
            {profile}
          </span>
        )}

        <div style={{ flex: 1 }} />

        {hasChanges && (
          <span style={{
            fontSize: 10, padding: '2px 8px', borderRadius: 99,
            background: 'rgba(251,191,36,.15)', color: '#fbbf24', fontWeight: 600,
          }}>
            unsaved
          </span>
        )}

        {status && (
          <span style={{
            fontSize: 11, display: 'flex', alignItems: 'center', gap: 4,
            color: status.type === 'ok' ? '#4ade80' : '#f87171',
          }}>
            {status.type === 'ok' ? <Check size={12} /> : <AlertCircle size={12} />}
            {status.msg}
          </span>
        )}

        <button onClick={load} disabled={loading}
          title="Reload"
          style={{
            background: 'transparent', border: '1px solid var(--border)', borderRadius: 6,
            color: 'var(--text-secondary)', cursor: 'pointer', padding: '4px 8px',
            display: 'flex', alignItems: 'center', gap: 4, fontSize: 11,
          }}>
          {loading ? <Loader2 size={12} className="spin" /> : <RefreshCw size={12} />}
          Reload
        </button>

        <button onClick={save} disabled={saving || !hasChanges}
          style={{
            background: hasChanges ? 'var(--accent)' : 'var(--bg-tertiary)',
            border: 'none', borderRadius: 6,
            color: hasChanges ? '#fff' : 'var(--text-muted)', cursor: hasChanges ? 'pointer' : 'default',
            padding: '4px 12px', display: 'flex', alignItems: 'center', gap: 4,
            fontSize: 11, fontWeight: 600, transition: 'all .15s',
          }}>
          {saving ? <Loader2 size={12} className="spin" /> : <Save size={12} />}
          Save
        </button>
      </div>

      {/* Editor */}
      {loading ? (
        <div style={{
          flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center',
          color: 'var(--text-muted)', fontSize: 13, gap: 8,
        }}>
          <Loader2 size={16} className="spin" /> Loading soul.md...
        </div>
      ) : (
        <textarea
          value={content}
          onChange={e => setContent(e.target.value)}
          spellCheck={false}
          placeholder="Write your agent's soul here... This defines its personality, voice, values, and prime directive."
          style={{
            flex: 1, resize: 'none', padding: '16px 20px',
            background: 'var(--bg-primary)', color: 'var(--text)',
            border: 'none', outline: 'none',
            fontFamily: 'var(--mono)', fontSize: 12, lineHeight: 1.7,
            whiteSpace: 'pre-wrap', wordWrap: 'break-word',
            overflow: 'auto',
          }}
        />
      )}
    </div>
  );
}
