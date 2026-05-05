import { useState, useEffect, useRef, useCallback } from 'react';
import { Monitor, RefreshCw, AlertCircle, ChevronLeft, ChevronRight } from 'lucide-react';

function apiUrl(path) {
  const token = new URLSearchParams(location.search).get('token');
  return `${location.protocol}//${location.host}${path}${token ? `?token=${encodeURIComponent(token)}` : ''}`;
}

const PAGE_ICONS = ['◈', '⬡', '≡', '◉', '⬢'];

const s = {
  panel: { display: 'flex', flexDirection: 'column', flex: 1, overflow: 'hidden' },
  header: {
    display: 'flex', alignItems: 'center', gap: 8, padding: '8px 16px',
    borderBottom: '1px solid var(--border)', minHeight: 44, flexShrink: 0, flexWrap: 'wrap',
  },
  title: { fontSize: 13, fontWeight: 600 },
  badge: (live) => ({
    padding: '2px 8px', borderRadius: 99, fontSize: 10, fontWeight: 700,
    background: live ? 'var(--green-dim)' : 'var(--red-dim)',
    color: live ? 'var(--green)' : 'var(--red)',
  }),
  body: {
    flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center',
    justifyContent: 'center', gap: 16, padding: 20, overflowY: 'auto',
  },
  imgWrap: {
    borderRadius: 12, overflow: 'hidden',
    border: '1px solid var(--border)',
    boxShadow: '0 8px 32px rgba(0,0,0,.5)',
    background: '#000', maxWidth: '100%',
  },
  img: { display: 'block', maxWidth: '100%', maxHeight: '55vh', width: 'auto' },
  navRow: { display: 'flex', alignItems: 'center', gap: 6 },
  pageBtn: (active) => ({
    padding: '6px 12px', borderRadius: 'var(--radius-sm)',
    background: active ? 'var(--accent-dim)' : 'var(--bg-tertiary)',
    color: active ? 'var(--accent-light)' : 'var(--text-muted)',
    cursor: 'pointer', fontSize: 11, fontWeight: 700, transition: 'all .15s',
    border: active ? '1px solid var(--accent)' : '1px solid var(--border)',
  }),
  arrowBtn: {
    display: 'flex', alignItems: 'center', justifyContent: 'center',
    width: 32, height: 32, borderRadius: 'var(--radius-sm)',
    border: '1px solid var(--border)', background: 'var(--bg-tertiary)',
    color: 'var(--text)', cursor: 'pointer',
  },
  hint: { fontSize: 11, color: 'var(--text-muted)', textAlign: 'center' },
  modeRow: { display: 'flex', gap: 4, marginLeft: 'auto' },
  modeBtn: (active) => ({
    padding: '4px 10px', borderRadius: 'var(--radius-sm)', border: 'none',
    background: active ? 'var(--accent-dim)' : 'transparent',
    color: active ? 'var(--accent-light)' : 'var(--text-muted)',
    cursor: 'pointer', fontSize: 10, fontWeight: 600,
  }),
  errBox: {
    display: 'flex', alignItems: 'center', gap: 8, padding: '10px 16px',
    background: 'var(--red-dim)', borderRadius: 'var(--radius)',
    border: '1px solid var(--red)', color: 'var(--red)', fontSize: 12,
  },
};

export default function DisplayPanel() {
  const [mode,    setMode]    = useState('stream');
  const [live,    setLive]    = useState(true);
  const [err,     setErr]     = useState(false);
  const [tick,    setTick]    = useState(0);
  const [pages,   setPages]   = useState([]);
  const [curPage, setCurPage] = useState(null);
  const [sending, setSending] = useState(false);
  const timerRef = useRef(null);
  const pollRef  = useRef(null);

  const fetchStatus = useCallback(async () => {
    try {
      const r = await fetch(apiUrl('/api/display/status'));
      if (!r.ok) return;
      const d = await r.json();
      if (d.pages && d.pages.length) setPages(d.pages);
      if (typeof d.page === 'number') setCurPage(d.page);
      setLive(true);
    } catch { setLive(false); }
  }, []);

  useEffect(() => {
    fetchStatus();
    pollRef.current = setInterval(fetchStatus, 2000);
    return () => clearInterval(pollRef.current);
  }, [fetchStatus]);

  useEffect(() => {
    if (mode !== 'poll') return;
    timerRef.current = setInterval(() => setTick(t => t + 1), 2000);
    return () => clearInterval(timerRef.current);
  }, [mode]);

  const goToPage = useCallback(async (idx) => {
    setSending(true);
    try {
      const token = new URLSearchParams(location.search).get('token');
      const url = `${location.protocol}//${location.host}/api/display/page${token ? `?token=${encodeURIComponent(token)}` : ''}`;
      await fetch(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ page: idx }),
      });
      setCurPage(idx);
      setTimeout(fetchStatus, 500);
    } catch {/* ignore */} finally { setSending(false); }
  }, [fetchStatus]);

  const goPrev = () => curPage !== null && pages.length && goToPage((curPage - 1 + pages.length) % pages.length);
  const goNext = () => curPage !== null && pages.length && goToPage((curPage + 1) % pages.length);

  return (
    <div style={s.panel}>
      <div style={s.header}>
        <Monitor size={15} />
        <span style={s.title}>Pi Display</span>
        <span style={s.badge(live && !err)}>{err ? 'Offline' : live ? 'Live' : ', '}</span>
        <div style={s.modeRow}>
          <button style={s.modeBtn(mode === 'stream')} onClick={() => { setMode('stream'); setErr(false); }}>MJPEG</button>
          <button style={s.modeBtn(mode === 'poll')}   onClick={() => { setMode('poll');   setErr(false); }}>Snapshot</button>
        </div>
        {mode === 'poll' && (
          <button style={{ ...s.arrowBtn, marginLeft: 4 }} onClick={() => setTick(t => t + 1)} title="Refresh">
            <RefreshCw size={13} />
          </button>
        )}
      </div>

      <div style={s.body}>
        {pages.length > 0 && (
          <div style={s.navRow}>
            <button style={s.arrowBtn} onClick={goPrev} disabled={sending}><ChevronLeft size={16} /></button>
            {pages.map((name, i) => (
              <button key={name} style={s.pageBtn(curPage === i)} onClick={() => goToPage(i)} disabled={sending}>
                {PAGE_ICONS[i] || ''} {name}
              </button>
            ))}
            <button style={s.arrowBtn} onClick={goNext} disabled={sending}><ChevronRight size={16} /></button>
          </div>
        )}

        {err ? (
          <div style={s.errBox}>
            <AlertCircle size={16} />
            Display offline,  <code style={{ marginLeft: 4 }}>sudo systemctl start sable-display</code>
          </div>
        ) : mode === 'stream' ? (
          <div style={s.imgWrap}>
            <img key="mjpeg" src={apiUrl('/api/display/stream')} alt="Pi HUD" style={s.img}
              onError={() => { setErr(true); setLive(false); }}
              onLoad={() => { setErr(false); setLive(true); }} />
          </div>
        ) : (
          <div style={s.imgWrap}>
            <img key={tick} src={apiUrl('/api/display/frame') + `&_t=${tick}`} alt="Pi HUD snapshot" style={s.img}
              onError={() => { setErr(true); setLive(false); }}
              onLoad={() => { setErr(false); setLive(true); }} />
          </div>
        )}

        <p style={s.hint}>
          {pages.length > 0 && curPage !== null ? `Page: ${pages[curPage]}  ·  ` : ''}
          {mode === 'stream' ? 'MJPEG,  auto refresh' : 'Snapshot,  every 2 s'}
        </p>
      </div>
    </div>
  );
}
