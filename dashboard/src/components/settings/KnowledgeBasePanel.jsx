import { useState, useEffect, useCallback } from 'react';
import {
  BookOpen, Plus, Trash2, RefreshCw, Search, Check, AlertCircle, Loader2,
  Tag, ChevronDown, ChevronRight, X, Upload, FileText,
} from 'lucide-react';

function apiUrl(path) {
  const token = new URLSearchParams(location.search).get('token');
  return `${location.protocol}//${location.host}${path}${token ? `?token=${encodeURIComponent(token)}` : ''}`;
}

const CATEGORIES = ['insight', 'skill', 'strategy', 'pattern', 'error_recovery', 'custom'];

const CAT_COLORS = {
  insight:        { bg: 'rgba(99,102,241,.12)',  text: '#818cf8' },
  skill:          { bg: 'rgba(34,197,94,.12)',   text: '#4ade80' },
  strategy:       { bg: 'rgba(251,191,36,.12)',  text: '#fbbf24' },
  pattern:        { bg: 'rgba(14,165,233,.12)',  text: '#38bdf8' },
  error_recovery: { bg: 'rgba(236,72,153,.12)',  text: '#f472b6' },
  custom:         { bg: 'rgba(168,85,247,.12)',  text: '#c084fc' },
};

function catColor(cat) {
  return CAT_COLORS[cat] || CAT_COLORS.custom;
}

/* ── Add entry form ───────────────────────────────────────────── */
function AddForm({ onAdd, onCancel }) {
  const [title, setTitle] = useState('');
  const [content, setContent] = useState('');
  const [category, setCategory] = useState('insight');
  const [tags, setTags] = useState('');
  const [saving, setSaving] = useState(false);

  const submit = async () => {
    if (!title.trim() || !content.trim()) return;
    setSaving(true);
    try {
      const res = await fetch(apiUrl('/api/knowledge-base'), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          title: title.trim(),
          content: content.trim(),
          category,
          tags: tags.split(',').map(t => t.trim()).filter(Boolean),
        }),
      });
      const data = await res.json();
      if (data.ok) {
        onAdd(data.entry);
      }
    } finally {
      setSaving(false);
    }
  };

  return (
    <div style={{
      background: 'var(--bg-secondary)', border: '1px solid var(--accent-dim)',
      borderRadius: 10, padding: 16, display: 'flex', flexDirection: 'column', gap: 10,
      marginBottom: 12,
    }}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        <span style={{ fontSize: 12, fontWeight: 700, color: 'var(--accent-light)' }}>
          Add Knowledge
        </span>
        <button onClick={onCancel} style={{
          background: 'transparent', border: 'none', color: 'var(--text-muted)',
          cursor: 'pointer', padding: 2, display: 'flex',
        }}>
          <X size={14} />
        </button>
      </div>

      <input value={title} onChange={e => setTitle(e.target.value)}
        placeholder="Title" spellCheck={false}
        style={{
          padding: '6px 10px', borderRadius: 6, border: '1px solid var(--border)',
          background: 'var(--bg-primary)', color: 'var(--text)',
          fontSize: 12, fontFamily: 'var(--sans)', outline: 'none',
        }} />

      <textarea value={content} onChange={e => setContent(e.target.value)}
        placeholder="Knowledge content..." rows={4} spellCheck={false}
        style={{
          padding: '6px 10px', borderRadius: 6, border: '1px solid var(--border)',
          background: 'var(--bg-primary)', color: 'var(--text)',
          fontSize: 12, fontFamily: 'var(--sans)', outline: 'none', resize: 'vertical',
          lineHeight: 1.5,
        }} />

      <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
        <select value={category} onChange={e => setCategory(e.target.value)}
          style={{
            padding: '4px 8px', borderRadius: 6, border: '1px solid var(--border)',
            background: 'var(--bg-primary)', color: 'var(--text)', fontSize: 11,
            outline: 'none',
          }}>
          {CATEGORIES.map(c => <option key={c} value={c}>{c}</option>)}
        </select>
        <input value={tags} onChange={e => setTags(e.target.value)}
          placeholder="Tags (comma-separated)" spellCheck={false}
          style={{
            flex: 1, padding: '4px 8px', borderRadius: 6, border: '1px solid var(--border)',
            background: 'var(--bg-primary)', color: 'var(--text)', fontSize: 11,
            fontFamily: 'var(--sans)', outline: 'none',
          }} />
      </div>

      <button onClick={submit} disabled={saving || !title.trim() || !content.trim()}
        style={{
          alignSelf: 'flex-end', padding: '5px 16px', borderRadius: 6,
          border: 'none', cursor: 'pointer', fontSize: 11, fontWeight: 600,
          background: title.trim() && content.trim() ? 'var(--accent)' : 'var(--bg-tertiary)',
          color: title.trim() && content.trim() ? '#fff' : 'var(--text-muted)',
          display: 'flex', alignItems: 'center', gap: 4,
        }}>
        {saving ? <Loader2 size={12} className="spin" /> : <Plus size={12} />}
        Add
      </button>
    </div>
  );
}

/* ── File upload area ──────────────────────────────────────────── */
const ACCEPTED = '.pdf,.docx,.xlsx,.xls,.txt,.md,.csv,.log,.json,.yaml,.yml,.toml';

function UploadArea({ onUploaded, onClose }) {
  const [uploading, setUploading] = useState(false);
  const [results, setResults] = useState(null);
  const [dragOver, setDragOver] = useState(false);
  const fileRef = { current: null };

  const uploadFiles = async (files) => {
    if (!files || files.length === 0) return;
    setUploading(true);
    setResults(null);
    try {
      const form = new FormData();
      for (const f of files) form.append('files', f);
      const res = await fetch(apiUrl('/api/knowledge-base/upload'), {
        method: 'POST',
        body: form,
      });
      const data = await res.json();
      setResults(data.results || []);
      if (data.ok) onUploaded();
    } catch (err) {
      setResults([{ filename: 'upload', ok: false, error: err.message }]);
    } finally {
      setUploading(false);
    }
  };

  const onDrop = (e) => {
    e.preventDefault();
    setDragOver(false);
    uploadFiles(e.dataTransfer.files);
  };

  return (
    <div style={{
      background: 'var(--bg-secondary)', border: '1px solid var(--accent-dim)',
      borderRadius: 10, padding: 16, marginBottom: 12,
      display: 'flex', flexDirection: 'column', gap: 10,
    }}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        <span style={{ fontSize: 12, fontWeight: 700, color: 'var(--accent-light)' }}>
          Upload Files
        </span>
        <button onClick={onClose} style={{
          background: 'transparent', border: 'none', color: 'var(--text-muted)',
          cursor: 'pointer', padding: 2, display: 'flex',
        }}>
          <X size={14} />
        </button>
      </div>

      <div
        onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
        onDragLeave={() => setDragOver(false)}
        onDrop={onDrop}
        onClick={() => fileRef.current?.click()}
        style={{
          border: `2px dashed ${dragOver ? 'var(--accent)' : 'var(--border)'}`,
          borderRadius: 8, padding: '28px 20px', textAlign: 'center',
          cursor: uploading ? 'wait' : 'pointer',
          background: dragOver ? 'rgba(99,102,241,.06)' : 'transparent',
          transition: 'all .15s',
        }}
      >
        <input
          type="file"
          multiple
          accept={ACCEPTED}
          ref={el => fileRef.current = el}
          style={{ display: 'none' }}
          onChange={e => uploadFiles(e.target.files)}
        />
        {uploading ? (
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 8, color: 'var(--accent-light)' }}>
            <Loader2 size={18} className="spin" />
            <span style={{ fontSize: 12 }}>Processing files...</span>
          </div>
        ) : (
          <>
            <Upload size={24} style={{ color: 'var(--text-muted)', marginBottom: 6 }} />
            <div style={{ fontSize: 12, color: 'var(--text-secondary)', marginBottom: 4 }}>
              Drop files here or click to browse
            </div>
            <div style={{ fontSize: 10, color: 'var(--text-muted)' }}>
              PDF, DOCX, Excel, TXT, Markdown, CSV, JSON, YAML (max 10MB each)
            </div>
          </>
        )}
      </div>

      {results && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
          {results.map((r, i) => (
            <div key={i} style={{
              display: 'flex', alignItems: 'center', gap: 6,
              fontSize: 11, padding: '4px 8px', borderRadius: 6,
              background: r.ok ? 'rgba(34,197,94,.08)' : 'rgba(239,68,68,.08)',
            }}>
              {r.ok
                ? <Check size={12} style={{ color: '#4ade80', flexShrink: 0 }} />
                : <AlertCircle size={12} style={{ color: '#f87171', flexShrink: 0 }} />}
              <FileText size={11} style={{ color: 'var(--text-muted)', flexShrink: 0 }} />
              <span style={{ color: 'var(--text)', flex: 1, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                {r.filename}
              </span>
              {!r.ok && (
                <span style={{ color: '#f87171', fontSize: 10 }}>{r.error}</span>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

/* ── Entry card ───────────────────────────────────────────────── */
function EntryCard({ entry, onDelete }) {
  const [expanded, setExpanded] = useState(false);
  const [deleting, setDeleting] = useState(false);
  const cc = catColor(entry.category);

  const handleDelete = async (e) => {
    e.stopPropagation();
    setDeleting(true);
    try {
      const res = await fetch(apiUrl('/api/knowledge-base'), {
        method: 'DELETE',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ learning_id: entry.learning_id }),
      });
      const data = await res.json();
      if (data.ok) onDelete(entry.learning_id);
    } finally {
      setDeleting(false);
    }
  };

  return (
    <div style={{
      background: 'var(--bg-secondary)', border: '1px solid var(--border)',
      borderRadius: 8, overflow: 'hidden', transition: 'all .15s',
    }}>
      <div onClick={() => setExpanded(!expanded)} style={{
        padding: '10px 12px', display: 'flex', alignItems: 'center', gap: 8,
        cursor: 'pointer',
      }}>
        {expanded ? <ChevronDown size={12} style={{ color: 'var(--text-muted)', flexShrink: 0 }} />
                   : <ChevronRight size={12} style={{ color: 'var(--text-muted)', flexShrink: 0 }} />}

        <span style={{
          fontSize: 10, padding: '1px 7px', borderRadius: 99,
          background: cc.bg, color: cc.text, fontWeight: 700, flexShrink: 0,
        }}>
          {entry.category}
        </span>

        <span style={{
          fontSize: 12, fontWeight: 600, color: 'var(--text)',
          overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', flex: 1,
        }}>
          {entry.title}
        </span>

        <span style={{ fontSize: 9, color: 'var(--text-muted)', flexShrink: 0 }}>
          {entry.source_agent}
        </span>

        <button onClick={handleDelete} disabled={deleting} title="Delete"
          style={{
            background: 'transparent', border: 'none', color: 'var(--text-muted)',
            cursor: 'pointer', padding: 2, display: 'flex', flexShrink: 0,
          }}>
          {deleting ? <Loader2 size={11} className="spin" /> : <Trash2 size={11} />}
        </button>
      </div>

      {expanded && (
        <div style={{
          padding: '0 12px 10px 32px', display: 'flex', flexDirection: 'column', gap: 6,
        }}>
          <p style={{
            fontSize: 11, color: 'var(--text-secondary)', lineHeight: 1.5, margin: 0,
            whiteSpace: 'pre-wrap',
          }}>
            {entry.content}
          </p>
          <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap', alignItems: 'center' }}>
            {(entry.tags || []).map(t => (
              <span key={t} style={{
                fontSize: 9, padding: '1px 6px', borderRadius: 99,
                background: 'rgba(255,255,255,.06)', color: 'var(--text-muted)',
              }}>
                <Tag size={8} style={{ marginRight: 2, verticalAlign: '-1px' }} />{t}
              </span>
            ))}
            {entry.confidence != null && (
              <span style={{ fontSize: 9, color: 'var(--text-muted)' }}>
                conf: {(entry.confidence * 100).toFixed(0)}%
              </span>
            )}
            {entry.created_at && (
              <span style={{ fontSize: 9, color: 'var(--text-muted)' }}>
                {new Date(entry.created_at).toLocaleDateString()}
              </span>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

/* ── Main panel ───────────────────────────────────────────────── */
export default function KnowledgeBasePanel() {
  const [entries, setEntries] = useState([]);
  const [loading, setLoading] = useState(true);
  const [showAdd, setShowAdd] = useState(false);
  const [showUpload, setShowUpload] = useState(false);
  const [search, setSearch] = useState('');
  const [filterCat, setFilterCat] = useState('');

  const load = useCallback(async () => {
    setLoading(true);
    try {
      const res = await fetch(apiUrl('/api/knowledge-base'));
      const data = await res.json();
      setEntries(data.entries || []);
    } catch {
      // ignore
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => { load(); }, [load]);

  const handleAdd = (entry) => {
    setEntries(prev => [...prev, entry]);
    setShowAdd(false);
  };

  const handleDelete = (id) => {
    setEntries(prev => prev.filter(e => e.learning_id !== id));
  };

  const filtered = entries.filter(e => {
    if (filterCat && e.category !== filterCat) return false;
    if (search) {
      const q = search.toLowerCase();
      return (
        (e.title || '').toLowerCase().includes(q) ||
        (e.content || '').toLowerCase().includes(q) ||
        (e.tags || []).some(t => t.toLowerCase().includes(q))
      );
    }
    return true;
  });

  // Unique categories from entries
  const cats = [...new Set(entries.map(e => e.category))].sort();

  return (
    <div style={{
      flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden',
      background: 'var(--bg-primary)',
    }}>
      {/* Toolbar */}
      <div style={{
        padding: '10px 16px', display: 'flex', alignItems: 'center', gap: 10,
        borderBottom: '1px solid var(--border)', background: 'var(--bg-secondary)',
        flexShrink: 0, flexWrap: 'wrap',
      }}>
        <BookOpen size={16} style={{ color: 'var(--accent-light)' }} />
        <span style={{ fontSize: 13, fontWeight: 700, color: 'var(--text)', letterSpacing: '.02em' }}>
          Knowledge Base
        </span>
        <span style={{
          fontSize: 10, padding: '2px 8px', borderRadius: 99,
          background: 'var(--accent-dim)', color: 'var(--accent-light)', fontWeight: 600,
        }}>
          {entries.length}
        </span>

        <div style={{ flex: 1 }} />

        {/* Search */}
        <div style={{
          display: 'flex', alignItems: 'center', gap: 4,
          background: 'var(--bg-primary)', borderRadius: 6,
          border: '1px solid var(--border)', padding: '3px 8px',
        }}>
          <Search size={11} style={{ color: 'var(--text-muted)' }} />
          <input value={search} onChange={e => setSearch(e.target.value)}
            placeholder="Search..." spellCheck={false}
            style={{
              background: 'transparent', border: 'none', outline: 'none',
              color: 'var(--text)', fontSize: 11, fontFamily: 'var(--sans)', width: 100,
            }} />
        </div>

        {/* Category filter */}
        <select value={filterCat} onChange={e => setFilterCat(e.target.value)}
          style={{
            padding: '3px 8px', borderRadius: 6, border: '1px solid var(--border)',
            background: 'var(--bg-primary)', color: 'var(--text)', fontSize: 11, outline: 'none',
          }}>
          <option value="">All</option>
          {cats.map(c => <option key={c} value={c}>{c}</option>)}
        </select>

        <button onClick={load} disabled={loading} title="Reload"
          style={{
            background: 'transparent', border: '1px solid var(--border)', borderRadius: 6,
            color: 'var(--text-secondary)', cursor: 'pointer', padding: '4px 8px',
            display: 'flex', alignItems: 'center', gap: 4, fontSize: 11,
          }}>
          {loading ? <Loader2 size={12} className="spin" /> : <RefreshCw size={12} />}
        </button>

        <button onClick={() => setShowUpload(true)}
          style={{
            background: 'var(--bg-tertiary)', border: '1px solid var(--border)', borderRadius: 6,
            color: 'var(--text-secondary)', cursor: 'pointer', padding: '4px 12px',
            display: 'flex', alignItems: 'center', gap: 4, fontSize: 11, fontWeight: 600,
          }}>
          <Upload size={12} /> Upload
        </button>

        <button onClick={() => setShowAdd(true)}
          style={{
            background: 'var(--accent)', border: 'none', borderRadius: 6,
            color: '#fff', cursor: 'pointer', padding: '4px 12px',
            display: 'flex', alignItems: 'center', gap: 4, fontSize: 11, fontWeight: 600,
          }}>
          <Plus size={12} /> Add
        </button>
      </div>

      {/* Content */}
      <div style={{ flex: 1, overflow: 'auto', padding: 16 }}>
        {showAdd && <AddForm onAdd={handleAdd} onCancel={() => setShowAdd(false)} />}
        {showUpload && <UploadArea onUploaded={() => { setShowUpload(false); load(); }} onClose={() => setShowUpload(false)} />}

        {loading ? (
          <div style={{
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            color: 'var(--text-muted)', fontSize: 13, gap: 8, padding: 40,
          }}>
            <Loader2 size={16} className="spin" /> Loading knowledge base...
          </div>
        ) : filtered.length === 0 ? (
          <div style={{
            textAlign: 'center', color: 'var(--text-muted)', fontSize: 12, padding: 40,
          }}>
            {entries.length === 0
              ? 'No knowledge entries yet. Click "Add" to teach your agent.'
              : 'No entries match your search.'}
          </div>
        ) : (
          <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
            {filtered.map(e => (
              <EntryCard key={e.learning_id} entry={e} onDelete={handleDelete} />
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
