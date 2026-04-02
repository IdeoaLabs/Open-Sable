import { useState, useRef, useEffect } from 'react';
import { ChevronDown, RefreshCw } from 'lucide-react';

const PROVIDER_ICONS = {
  ollama:     '🦙',
  openai:     '🟢',
  anthropic:  '🟠',
  google:     '🔵',
  gemini:     '🔵',
  groq:       '⚡',
  mistral:    '🌊',
  lmstudio:   '🖥️',
  openrouter: '🔀',
  xai:        '✖️',
  deepseek:   '🐋',
  together:   '🤝',
};

const s = {
  wrapper: { position: 'relative', display: 'inline-flex' },
  trigger: {
    fontSize: 11, fontWeight: 600, padding: '4px 10px', borderRadius: 999,
    display: 'flex', alignItems: 'center', gap: 6,
    background: 'var(--bg-tertiary)', border: '1px solid var(--border)',
    color: 'var(--text-secondary)', cursor: 'pointer',
    transition: 'all .15s',
  },
  triggerHover: {
    borderColor: 'var(--accent)',
    background: 'var(--accent-dim)',
  },
  dropdown: {
    position: 'absolute', top: 'calc(100% + 6px)', left: 0,
    minWidth: 280, maxHeight: 400, overflowY: 'auto',
    background: 'var(--bg-secondary)', border: '1px solid var(--border)',
    borderRadius: 10, boxShadow: '0 12px 40px rgba(0,0,0,.4)',
    zIndex: 1000, padding: '6px 0',
  },
  groupHeader: {
    fontSize: 10, fontWeight: 700, color: 'var(--text-muted)',
    textTransform: 'uppercase', letterSpacing: '.06em',
    padding: '10px 14px 4px', display: 'flex', alignItems: 'center', gap: 6,
  },
  item: (active) => ({
    fontSize: 12, padding: '7px 14px 7px 28px', cursor: 'pointer',
    display: 'flex', alignItems: 'center', gap: 8,
    color: active ? 'var(--accent-light)' : 'var(--text)',
    background: active ? 'var(--accent-dim)' : 'transparent',
    fontWeight: active ? 600 : 400,
    transition: 'background .1s',
  }),
  itemHover: {
    background: 'var(--bg-hover)',
  },
  dot: (active) => ({
    width: 6, height: 6, borderRadius: '50%', flexShrink: 0,
    background: active ? 'var(--green)' : 'transparent',
    border: active ? 'none' : '1px solid var(--border)',
  }),
  refreshBtn: {
    background: 'none', border: 'none', cursor: 'pointer',
    color: 'var(--text-muted)', padding: '2px', display: 'flex',
    alignItems: 'center', marginLeft: 'auto',
  },
  footer: {
    borderTop: '1px solid var(--border)', padding: '8px 14px',
    fontSize: 10, color: 'var(--text-muted)', textAlign: 'center',
  },
};

export default function ModelSelector({ model, modelGroups = [], onModelSwitch, onRefreshModels, agentProfile }) {
  const [open, setOpen] = useState(false);
  const [hoveredItem, setHoveredItem] = useState(null);
  const [hoverTrigger, setHoverTrigger] = useState(false);
  const ref = useRef(null);

  // Close on outside click
  useEffect(() => {
    if (!open) return;
    const handler = (e) => {
      if (ref.current && !ref.current.contains(e.target)) setOpen(false);
    };
    document.addEventListener('mousedown', handler);
    return () => document.removeEventListener('mousedown', handler);
  }, [open]);

  // Count total models
  const totalModels = modelGroups.reduce((sum, g) => sum + (g.models?.length || 0), 0);

  // Truncate model name for display
  const displayName = model
    ? (model.length > 24 ? model.slice(0, 22) + '…' : model)
    : 'No model';

  return (
    <div style={s.wrapper} ref={ref}>
      <div
        style={{ ...s.trigger, ...(hoverTrigger ? s.triggerHover : {}) }}
        onClick={() => { setOpen(!open); if (!open && onRefreshModels) onRefreshModels(agentProfile); }}
        onMouseEnter={() => setHoverTrigger(true)}
        onMouseLeave={() => setHoverTrigger(false)}
      >
        <span>🧠</span>
        <span>{displayName}</span>
        <ChevronDown size={12} style={{ opacity: 0.5, transform: open ? 'rotate(180deg)' : 'none', transition: 'transform .2s' }} />
      </div>

      {open && (
        <div style={s.dropdown}>
          {/* Header with refresh */}
          <div style={{ ...s.groupHeader, paddingBottom: 8, borderBottom: '1px solid var(--border)' }}>
            <span>Select Model</span>
            <span style={{ fontWeight: 400, opacity: 0.7 }}>({totalModels})</span>
            {onRefreshModels && (
              <button style={s.refreshBtn} onClick={(e) => { e.stopPropagation(); onRefreshModels(agentProfile); }} title="Refresh models">
                <RefreshCw size={11} />
              </button>
            )}
          </div>

          {modelGroups.length === 0 && (
            <div style={{ padding: '16px 14px', fontSize: 12, color: 'var(--text-muted)', textAlign: 'center' }}>
              No models available.<br />Install Ollama or configure an API key.
            </div>
          )}

          {modelGroups.map((group) => (
            <div key={group.provider}>
              <div style={s.groupHeader}>
                <span>{PROVIDER_ICONS[group.provider] || '🔮'}</span>
                <span>{group.name}</span>
                <span style={{ fontWeight: 400, opacity: 0.6 }}>({group.models?.length || 0})</span>
              </div>
              {(group.models || []).map((m) => {
                const isActive = m.name === model;
                const key = `${group.provider}:${m.name}`;
                return (
                  <div
                    key={key}
                    style={{
                      ...s.item(isActive),
                      ...(hoveredItem === key && !isActive ? s.itemHover : {}),
                    }}
                    onMouseEnter={() => setHoveredItem(key)}
                    onMouseLeave={() => setHoveredItem(null)}
                    onClick={() => {
                      if (!isActive && onModelSwitch) {
                        onModelSwitch(m.name, group.provider, agentProfile);
                      }
                      setOpen(false);
                    }}
                  >
                    <span style={s.dot(isActive)} />
                    <span style={{ flex: 1 }}>{m.name}</span>
                    {isActive && <span style={{ fontSize: 10, color: 'var(--green)' }}>active</span>}
                  </div>
                );
              })}
            </div>
          ))}

          {totalModels > 0 && (
            <div style={s.footer}>
              Click a model to switch. Changes take effect immediately.
            </div>
          )}
        </div>
      )}
    </div>
  );
}
