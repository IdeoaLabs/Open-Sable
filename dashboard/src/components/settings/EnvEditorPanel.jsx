import { useState, useEffect, useCallback } from 'react';
import {
  SlidersHorizontal, Save, RefreshCw, Eye, EyeOff,
  Search, Check, AlertCircle, Loader2, Trash2, Plus, RotateCcw,
} from 'lucide-react';

const SENSITIVE_RE = /key|token|secret|pass|hash|pwd|api_id/i;

function apiUrl(path) {
  const token = new URLSearchParams(location.search).get('token');
  return `${location.protocol}//${location.host}${path}${token ? `?token=${encodeURIComponent(token)}` : ''}`;
}

/* ── Section colour palette ─────────────────────────────────── */
const SECTION_COLORS = [
  { border: 'rgba(99,102,241,.35)',  glow: 'rgba(99,102,241,.07)', accent: '#818cf8' },
  { border: 'rgba(34,197,94,.30)',   glow: 'rgba(34,197,94,.06)',  accent: '#4ade80' },
  { border: 'rgba(251,191,36,.30)',  glow: 'rgba(251,191,36,.06)', accent: '#fbbf24' },
  { border: 'rgba(236,72,153,.30)',  glow: 'rgba(236,72,153,.06)', accent: '#f472b6' },
  { border: 'rgba(14,165,233,.30)',  glow: 'rgba(14,165,233,.06)', accent: '#38bdf8' },
  { border: 'rgba(168,85,247,.30)',  glow: 'rgba(168,85,247,.06)', accent: '#c084fc' },
  { border: 'rgba(249,115,22,.30)',  glow: 'rgba(249,115,22,.06)', accent: '#fb923c' },
  { border: 'rgba(20,184,166,.30)',  glow: 'rgba(20,184,166,.06)', accent: '#2dd4bf' },
];

/* ── Field card ─────────────────────────────────────────────── */
function FieldCard({ entry, onChange, onDelete, onRestore, accent }) {
  const isSensitive = SENSITIVE_RE.test(entry.key);
  const [visible, setVisible] = useState(!isSensitive);

  return (
    <div style={{
      background: entry.deleted ? 'rgba(239,68,68,.04)'    :
                  entry.isNew   ? 'rgba(34,197,94,.06)'    :
                  entry.changed ? 'rgba(99,102,241,.07)'   : 'var(--bg-primary)',
      border: `1px solid ${
        entry.deleted ? 'rgba(239,68,68,.3)'   :
        entry.isNew   ? 'rgba(34,197,94,.3)'   :
        entry.changed ? 'rgba(99,102,241,.4)'  : 'var(--border)'}`,
      borderRadius: 8, padding: '10px 12px',
      display: 'flex', flexDirection: 'column', gap: 5,
      transition: 'all .15s', opacity: entry.deleted ? 0.5 : 1,
    }}>
      <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', gap: 4 }}>
        <span style={{
          fontSize: 10, fontWeight: 700, fontFamily: 'var(--mono)',
          color: entry.deleted ? 'var(--red)'  :
                 entry.isNew   ? '#4ade80'      :
                 entry.changed ? accent         : 'var(--text-secondary)',
          letterSpacing: '.03em', wordBreak: 'break-all', lineHeight: 1.4,
          textDecoration: entry.deleted ? 'line-through' : 'none',
        }}>
          {entry.key}
        </span>
        <div style={{ display: 'flex', gap: 3, flexShrink: 0, alignItems: 'center' }}>
          {entry.isNew && (
            <span style={{ fontSize: 9, padding: '1px 6px', borderRadius: 99,
              background: 'rgba(34,197,94,.2)', color: '#4ade80', fontWeight: 700 }}>new</span>
          )}
          {entry.changed && !entry.isNew && (
            <span style={{ fontSize: 9, padding: '1px 6px', borderRadius: 99,
              background: 'rgba(99,102,241,.2)', color: '#818cf8', fontWeight: 700 }}>edited</span>
          )}
          {entry.deleted ? (
            <button onClick={() => onRestore(entry.key)} title="Restore"
              style={{ background: 'transparent', border: 'none', color: 'var(--text-muted)',
                cursor: 'pointer', padding: 2, display: 'flex' }}>
              <RotateCcw size={11} />
            </button>
          ) : (
            <button onClick={() => onDelete(entry.key)} title="Delete key"
              style={{ background: 'transparent', border: 'none', color: 'var(--text-muted)',
                cursor: 'pointer', padding: 2, display: 'flex' }}>
              <Trash2 size={11} />
            </button>
          )}
        </div>
      </div>

      {entry.comment && !entry.deleted && (
        <span style={{ fontSize: 10, color: 'var(--text-muted)', lineHeight: 1.3, fontStyle: 'italic' }}>
          {entry.comment}
        </span>
      )}

      {!entry.deleted && (
        <div style={{ display: 'flex', gap: 4, alignItems: 'center' }}>
          <input
            type={isSensitive && !visible ? 'password' : 'text'}
            value={entry.value}
            onChange={e => onChange(entry.key, e.target.value)}
            placeholder="(empty)"
            spellCheck={false}
            autoComplete="off"
            style={{
              flex: 1, padding: '5px 8px', borderRadius: 6, minWidth: 0,
              border: `1px solid ${entry.changed || entry.isNew ? 'rgba(99,102,241,.5)' : 'var(--border)'}`,
              background: 'var(--bg-secondary)', color: 'var(--text)',
              fontFamily: 'var(--mono)', fontSize: 11, outline: 'none',
            }}
          />
          {isSensitive && (
            <button onClick={() => setVisible(v => !v)} title={visible ? 'Hide' : 'Show'}
              style={{ background: 'transparent', border: 'none', color: 'var(--text-muted)',
                cursor: 'pointer', padding: 4, display: 'flex', alignItems: 'center', flexShrink: 0 }}>
              {visible ? <EyeOff size={13} /> : <Eye size={13} />}
            </button>
          )}
        </div>
      )}
    </div>
  );
}

/* ── Inline add-key form ─────────────────────────────────────── */
function AddKeyForm({ accent, onConfirm, onCancel }) {
  const [newKey, setNewKey] = useState('');
  const [newVal, setNewVal] = useState('');
  const valid = /^[A-Z_][A-Z0-9_]*$/i.test(newKey.trim());

  return (
    <div style={{
      display: 'flex', gap: 6, alignItems: 'center', flexWrap: 'wrap',
      padding: '10px 12px', borderRadius: 8,
      background: 'rgba(0,0,0,.12)', border: `1px dashed ${accent}55`,
    }}>
      <input
        value={newKey}
        onChange={e => setNewKey(e.target.value.toUpperCase())}
        placeholder="KEY_NAME"
        spellCheck={false}
        autoFocus
        style={{
          flex: '0 0 150px', padding: '5px 8px', borderRadius: 6,
          fontFamily: 'var(--mono)', fontSize: 11,
          background: 'var(--bg-secondary)', color: 'var(--text)',
          border: `1px solid ${valid ? accent + '88' : 'var(--border)'}`, outline: 'none',
        }}
      />
      <span style={{ color: 'var(--text-muted)', fontSize: 13 }}>=</span>
      <input
        value={newVal}
        onChange={e => setNewVal(e.target.value)}
        placeholder="value"
        spellCheck={false}
        onKeyDown={e => {
          if (e.key === 'Enter' && valid) onConfirm(newKey.trim(), newVal);
          if (e.key === 'Escape') onCancel();
        }}
        style={{
          flex: 1, minWidth: 120, padding: '5px 8px', borderRadius: 6,
          fontFamily: 'var(--mono)', fontSize: 11,
          background: 'var(--bg-secondary)', color: 'var(--text)',
          border: '1px solid var(--border)', outline: 'none',
        }}
      />
      <button
        onClick={() => valid && onConfirm(newKey.trim(), newVal)}
        disabled={!valid}
        style={{
          padding: '5px 14px', borderRadius: 6, border: 'none',
          cursor: valid ? 'pointer' : 'default', fontWeight: 700, fontSize: 11,
          background: valid ? accent : 'var(--bg-hover)',
          color: valid ? '#000' : 'var(--text-muted)',
        }}>
        Add
      </button>
      <button
        onClick={onCancel}
        style={{
          padding: '5px 10px', borderRadius: 6, fontSize: 11,
          border: '1px solid var(--border)', cursor: 'pointer',
          background: 'transparent', color: 'var(--text-muted)',
        }}>
        Cancel
      </button>
    </div>
  );
}

/* ── Section card ───────────────────────────────────────────── */
function SectionCard({ section, onChange, onDelete, onRestore, onAddEntry, filter, colorIdx }) {
  const col = SECTION_COLORS[colorIdx % SECTION_COLORS.length];
  const [addOpen, setAddOpen] = useState(false);

  const filtered = filter
    ? section.entries.filter(e =>
        e.key.toLowerCase().includes(filter) ||
        e.value.toLowerCase().includes(filter) ||
        (e.comment || '').toLowerCase().includes(filter))
    : section.entries;

  if (!filtered.length && !addOpen) return filter ? null : (
    <div style={{ borderRadius: 10, border: `1px solid ${col.border}`, background: col.glow }}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '8px 14px', background: 'rgba(0,0,0,.14)', borderRadius: 10 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <div style={{ width: 8, height: 8, borderRadius: '50%', background: col.accent, boxShadow: `0 0 6px ${col.accent}` }} />
          <span style={{ fontSize: 11, fontWeight: 700, textTransform: 'uppercase', letterSpacing: '.08em', color: col.accent }}>{section.title || 'General'}</span>
        </div>
        <button onClick={() => setAddOpen(true)} style={{ background: 'transparent', border: 'none', color: col.accent, cursor: 'pointer', display: 'flex', alignItems: 'center', gap: 4, fontSize: 11, padding: '2px 8px' }}>
          <Plus size={12} /> Add key
        </button>
      </div>
      {addOpen && (
        <div style={{ padding: '0 12px 12px' }}>
          <AddKeyForm accent={col.accent} onConfirm={(k, v) => { onAddEntry(k, v); setAddOpen(false); }} onCancel={() => setAddOpen(false)} />
        </div>
      )}
    </div>
  );

  const editedCount = filtered.filter(e => e.changed || e.deleted || e.isNew).length;

  return (
    <div style={{ borderRadius: 10, border: `1px solid ${col.border}`, background: col.glow, overflow: 'hidden' }}>
      {/* Header */}
      <div style={{
        display: 'flex', alignItems: 'center', justifyContent: 'space-between',
        padding: '8px 14px', borderBottom: `1px solid ${col.border}`,
        background: 'rgba(0,0,0,.14)',
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <div style={{ width: 8, height: 8, borderRadius: '50%', background: col.accent, boxShadow: `0 0 6px ${col.accent}` }} />
          <span style={{ fontSize: 11, fontWeight: 700, textTransform: 'uppercase', letterSpacing: '.08em', color: col.accent }}>
            {section.title || 'General'}
          </span>
          {editedCount > 0 && (
            <span style={{ fontSize: 9, padding: '2px 7px', borderRadius: 99, fontWeight: 700, background: 'rgba(99,102,241,.2)', color: '#818cf8' }}>
              {editedCount} edited
            </span>
          )}
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <span style={{ fontSize: 10, padding: '2px 8px', borderRadius: 99, background: 'rgba(255,255,255,.05)', color: 'var(--text-muted)' }}>
            {filtered.filter(e => !e.deleted).length} keys
          </span>
          <button
            onClick={() => setAddOpen(v => !v)}
            style={{
              background: addOpen ? `${col.accent}22` : 'transparent',
              border: `1px solid ${addOpen ? col.accent : 'transparent'}`,
              borderRadius: 6, color: col.accent, cursor: 'pointer',
              display: 'flex', alignItems: 'center', gap: 4, fontSize: 11, padding: '2px 10px',
            }}>
            <Plus size={12} /> Add key
          </button>
        </div>
      </div>

      {/* Grid of field cards */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(210px, 1fr))', gap: 8, padding: '12px 12px 0' }}>
        {filtered.map(entry => (
          <FieldCard
            key={entry.key}
            entry={entry}
            onChange={onChange}
            onDelete={onDelete}
            onRestore={onRestore}
            accent={col.accent}
          />
        ))}
      </div>

      {/* Add-key inline form */}
      {addOpen ? (
        <div style={{ padding: '8px 12px 12px' }}>
          <AddKeyForm
            accent={col.accent}
            onConfirm={(k, v) => { onAddEntry(k, v); setAddOpen(false); }}
            onCancel={() => setAddOpen(false)}
          />
        </div>
      ) : (
        <div style={{ height: 12 }} />
      )}
    </div>
  );
}

/* ── Main panel ─────────────────────────────────────────────── */
export default function EnvEditorPanel() {
  const [sections, setSections] = useState([]);
  const [profile, setProfile] = useState('');
  const [original, setOriginal] = useState({});
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [saved, setSaved] = useState(false);
  const [error, setError] = useState(null);
  const [saveError, setSaveError] = useState(null);
  const [filter, setFilter] = useState('');

  const hasChanges = sections.some(sec => sec.entries.some(e => e.changed || e.deleted || e.isNew));
  const changedCount = sections.reduce((n, sec) =>
    n + sec.entries.filter(e => e.changed || e.deleted || e.isNew).length, 0);

  const load = useCallback(async () => {
    setLoading(true); setError(null);
    try {
      const r = await fetch(apiUrl('/api/env'));
      if (!r.ok) { const d = await r.json().catch(() => ({})); throw new Error(d.error || `HTTP ${r.status}`); }
      const data = await r.json();
      setProfile(data.profile || '');
      const orig = {};
      const secs = (data.sections || []).map(sec => ({
        ...sec,
        entries: (sec.entries || []).map(e => { orig[e.key] = e.value; return { ...e, changed: false }; }),
      }));
      setSections(secs);
      setOriginal(orig);
    } catch (e) { setError(e.message); }
    finally { setLoading(false); }
  }, []);

  useEffect(() => { load(); }, [load]);

  const handleChange = useCallback((key, value) => {
    setSections(prev => prev.map(sec => ({
      ...sec,
      entries: sec.entries.map(e =>
        e.key === key ? { ...e, value, changed: !e.isNew && value !== original[key] } : e),
    })));
  }, [original]);

  const handleDelete = useCallback((key) => {
    setSections(prev => prev.map(sec => ({
      ...sec,
      entries: sec.entries.map(e => e.key === key ? { ...e, deleted: true } : e),
    })));
  }, []);

  const handleRestore = useCallback((key) => {
    setSections(prev => prev.map(sec => ({
      ...sec,
      entries: sec.entries.map(e => e.key === key ? { ...e, deleted: false } : e),
    })));
  }, []);

  const handleAddEntry = useCallback((sectionIdx, key, value) => {
    const exists = sections.some(sec => sec.entries.some(e => e.key === key));
    if (exists) return;
    setSections(prev => prev.map((sec, i) => i !== sectionIdx ? sec : {
      ...sec,
      entries: [...sec.entries, { key, value, comment: null, changed: false, isNew: true, deleted: false }],
    }));
  }, [sections]);

  const handleSave = useCallback(async () => {
    if (!hasChanges || saving) return;
    const updates = {};
    const adds = [];
    const deletes = [];
    sections.forEach(sec => sec.entries.forEach(e => {
      if (e.isNew && !e.deleted) adds.push({ key: e.key, value: e.value });
      else if (e.deleted && !e.isNew) deletes.push(e.key);
      else if (e.changed) updates[e.key] = e.value;
    }));
    setSaving(true); setSaveError(null);
    try {
      const r = await fetch(apiUrl('/api/env'), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ updates, adds, deletes }),
      });
      const data = await r.json();
      if (!r.ok || !data.ok) throw new Error(data.error || `HTTP ${r.status}`);
      setSaved(true);
      setTimeout(() => setSaved(false), 2500);
      load();
    } catch (e) {
      setSaveError(e.message);
      setTimeout(() => setSaveError(null), 4000);
    } finally {
      setSaving(false);
    }
  }, [hasChanges, saving, sections, load]);

  const f = filter.toLowerCase().trim();
  const totalKeys = sections.reduce((n, s) => n + s.entries.filter(e => !e.deleted).length, 0);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', flex: 1, minHeight: 0, overflow: 'hidden' }}>
      {/* Header */}
      <div style={{
        display: 'flex', alignItems: 'center', gap: 10,
        padding: '10px 18px', minHeight: 48, flexShrink: 0,
        borderBottom: '1px solid var(--border)', flexWrap: 'wrap',
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8, fontSize: 14, fontWeight: 700 }}>
          <SlidersHorizontal size={16} color="var(--accent)" />
          Agent Config
          {profile && (
            <span style={{
              padding: '2px 10px', borderRadius: 99, fontSize: 10, fontWeight: 700,
              background: 'var(--accent-dim)', color: 'var(--accent-light)',
              fontFamily: 'var(--mono)', letterSpacing: '.06em',
            }}>{profile}</span>
          )}
          {!loading && (
            <span style={{ fontSize: 10, color: 'var(--text-muted)', fontWeight: 400 }}>
              {totalKeys} keys
            </span>
          )}
        </div>

        <div style={{
          display: 'flex', alignItems: 'center', gap: 6, marginLeft: 'auto',
          background: 'var(--bg-tertiary)', border: '1px solid var(--border)',
          borderRadius: 'var(--radius-sm)', padding: '5px 10px',
        }}>
          <Search size={12} color="var(--text-muted)" />
          <input
            value={filter}
            onChange={e => setFilter(e.target.value)}
            placeholder="Filter…"
            style={{ background: 'transparent', border: 'none', outline: 'none', color: 'var(--text)', fontSize: 12, width: 140 }}
          />
        </div>

        <button onClick={load} disabled={loading} title="Reload"
          style={{
            background: 'transparent', border: '1px solid var(--border)',
            borderRadius: 'var(--radius-sm)', padding: '5px 8px',
            color: 'var(--text-muted)', cursor: 'pointer', display: 'flex', alignItems: 'center',
          }}>
          <RefreshCw size={13} style={loading ? { animation: 'spin 1s linear infinite' } : {}} />
        </button>

        <button onClick={handleSave} disabled={!hasChanges || saving}
          style={{
            display: 'flex', alignItems: 'center', gap: 6,
            padding: '6px 16px', borderRadius: 'var(--radius-sm)', border: 'none',
            cursor: hasChanges && !saving ? 'pointer' : 'default', fontWeight: 700, fontSize: 12,
            background: saved ? 'rgba(34,197,94,.18)' : hasChanges ? 'var(--accent)' : 'var(--bg-hover)',
            color: saved ? 'var(--green)' : hasChanges ? '#fff' : 'var(--text-muted)',
            transition: 'all .2s',
          }}>
          {saving ? <Loader2 size={13} style={{ animation: 'spin 1s linear infinite' }} />
           : saved  ? <Check size={13} /> : <Save size={13} />}
          {saving ? 'Saving…' : saved ? 'Saved!' : hasChanges
            ? `Save ${changedCount} change${changedCount !== 1 ? 's' : ''}` : 'No changes'}
        </button>
      </div>

      {error && (
        <div style={{
          display: 'flex', alignItems: 'center', gap: 8, padding: '10px 16px',
          margin: '12px 18px', background: 'var(--red-dim)', border: '1px solid var(--red)',
          borderRadius: 8, color: 'var(--red)', fontSize: 12,
        }}>
          <AlertCircle size={14} /> {error}
        </div>
      )}

      <div style={{ flex: 1, minHeight: 0, overflowY: 'auto', padding: '14px 18px' }}>
        <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
          {loading ? (
            <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', gap: 12, padding: 60, color: 'var(--text-muted)', fontSize: 13 }}>
              <Loader2 size={28} style={{ animation: 'spin 1s linear infinite', color: 'var(--accent)' }} />
              Loading configuration…
            </div>
          ) : sections.length === 0 ? (
            <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', gap: 10, padding: 60, color: 'var(--text-muted)', fontSize: 13 }}>
              <AlertCircle size={24} /> No configuration found
            </div>
          ) : sections.map((sec, i) => (
            <SectionCard
              key={i}
              section={sec}
              onChange={handleChange}
              onDelete={handleDelete}
              onRestore={handleRestore}
              onAddEntry={(k, v) => handleAddEntry(i, k, v)}
              filter={f}
              colorIdx={i}
            />
          ))}
        </div>
      </div>

      {(saved || saveError) && (
        <div style={{
          position: 'fixed', bottom: 24, right: 24, zIndex: 9999,
          display: 'flex', alignItems: 'center', gap: 8,
          padding: '10px 18px', borderRadius: 8, boxShadow: '0 8px 32px rgba(0,0,0,.4)',
          background: saveError ? 'var(--red-dim)' : 'rgba(34,197,94,.15)',
          border: `1px solid ${saveError ? 'var(--red)' : 'var(--green)'}`,
          color: saveError ? 'var(--red)' : 'var(--green)',
          fontSize: 13, fontWeight: 600, animation: 'fadeInUp .2s ease',
        }}>
          {saveError ? <><AlertCircle size={14} /> {saveError}</> : <><Check size={14} /> Configuration saved</>}
        </div>
      )}

      <style>{`
        @keyframes spin { to { transform: rotate(360deg); } }
        @keyframes fadeInUp { from { opacity: 0; transform: translateY(8px); } to { opacity: 1; transform: translateY(0); } }
      `}</style>
    </div>
  );
}
