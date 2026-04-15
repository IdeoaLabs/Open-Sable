import { useState, useEffect } from 'react';

const ACTION_LABELS = {
  browser_navigate: 'Web Browsing',
  file_write: 'Write File',
  file_delete: 'Delete File',
  system_command: 'Run System Command',
  email_send: 'Send Email',
  email_read: 'Read Email',
  calendar_write: 'Modify Calendar',
};

const ACTION_ICONS = {
  browser_navigate: '🌐',
  file_write: '📝',
  file_delete: '🗑️',
  system_command: '⚙️',
  email_send: '📧',
  email_read: '📬',
  calendar_write: '📅',
};

const ACTION_DESCRIPTIONS = {
  browser_navigate: 'The agent wants to open a web page. This allows it to browse, scrape, or interact with websites on your behalf.',
  file_write: 'The agent wants to create or modify a file on your system. Review the path below to make sure it\'s expected.',
  file_delete: 'The agent wants to permanently delete a file. This action cannot be undone — review carefully.',
  system_command: 'The agent wants to run a shell command on your machine. Only allow if you trust the command shown below.',
  email_send: 'The agent wants to send an email from your account. Check the recipient and content before approving.',
  email_read: 'The agent wants to read emails from your inbox. This lets it access message contents and metadata.',
  calendar_write: 'The agent wants to create or modify a calendar event. Review the details below.',
};

const overlayStyle = {
  position: 'fixed', inset: 0, zIndex: 10000,
  background: 'rgba(0,0,0,0.55)', backdropFilter: 'blur(4px)',
  display: 'flex', alignItems: 'center', justifyContent: 'center',
};

const dialogStyle = {
  background: 'var(--bg-primary, #1a1a2e)', border: '1px solid var(--border, #333)',
  borderRadius: 14, width: 400, maxWidth: '90vw',
  padding: 22, boxShadow: '0 16px 48px rgba(0,0,0,0.5)',
};

export default function PermissionDialog({ pending, onRespond }) {
  const [remember, setRemember] = useState(false);
  const [countdown, setCountdown] = useState(60);

  useEffect(() => {
    if (!pending) return;
    setCountdown(60);
    setRemember(false);
    const iv = setInterval(() => {
      setCountdown(prev => {
        if (prev <= 1) {
          onRespond(pending.requestId, false);
          clearInterval(iv);
          return 0;
        }
        return prev - 1;
      });
    }, 1000);
    return () => clearInterval(iv);
  }, [pending?.requestId]);

  if (!pending) return null;

  const label = ACTION_LABELS[pending.action] || pending.action;
  const icon = ACTION_ICONS[pending.action] || '🔐';
  const args = Object.entries(pending.arguments || {}).filter(([, v]) => v != null && v !== '');

  return (
    <div style={overlayStyle}>
      <div style={dialogStyle}>
        {/* Header */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 14 }}>
          <span style={{ fontSize: 22 }}>{icon}</span>
          <span style={{ fontSize: 14, fontWeight: 600, color: 'var(--text-primary, #eee)', flex: 1 }}>
            Permission Required
          </span>
          <span style={{
            fontSize: 11, color: 'var(--text-muted, #888)',
            background: 'var(--bg-secondary, #252540)', padding: '2px 7px',
            borderRadius: 8, fontVariantNumeric: 'tabular-nums',
          }}>
            {countdown}s
          </span>
        </div>

        {/* Body */}
        <p style={{ fontSize: 13, color: 'var(--text-secondary, #bbb)', margin: '0 0 4px', lineHeight: 1.5 }}>
          Sable wants to use <strong style={{ color: 'var(--text-primary, #eee)' }}>{label}</strong>
        </p>
        <p style={{ fontSize: 11, color: 'var(--text-muted, #888)', margin: '0 0 10px', lineHeight: 1.5 }}>
          {ACTION_DESCRIPTIONS[pending.action] || 'The agent is requesting access to a protected action. Review the details below before deciding.'}
        </p>
        <div style={{
          background: 'var(--bg-secondary, #252540)', borderRadius: 6, padding: '6px 10px',
          fontSize: 12, color: 'var(--accent, #7c5cfc)', marginBottom: 8, fontFamily: 'monospace',
        }}>
          {pending.tool}
        </div>
        {args.length > 0 && (
          <div style={{
            background: 'var(--bg-secondary, #252540)', borderRadius: 6, padding: '8px 10px',
            fontSize: 11, maxHeight: 120, overflowY: 'auto', marginBottom: 12,
          }}>
            {args.map(([k, v]) => (
              <div key={k} style={{ display: 'flex', gap: 6, padding: '2px 0' }}>
                <span style={{ color: 'var(--text-muted, #888)', minWidth: 60, flexShrink: 0 }}>{k}:</span>
                <span style={{ color: 'var(--text-secondary, #bbb)', wordBreak: 'break-all' }}>{String(v)}</span>
              </div>
            ))}
          </div>
        )}

        {/* Remember checkbox */}
        <label style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: 11, color: 'var(--text-muted, #888)', marginBottom: 14, cursor: 'pointer' }}>
          <input type="checkbox" checked={remember} onChange={e => setRemember(e.target.checked)} style={{ accentColor: 'var(--accent, #7c5cfc)' }} />
          Always allow this action
        </label>

        {/* Buttons */}
        <div style={{ display: 'flex', gap: 8, justifyContent: 'flex-end' }}>
          <button
            onClick={() => onRespond(pending.requestId, false)}
            style={{
              padding: '7px 18px', borderRadius: 8, fontSize: 12, fontWeight: 500,
              cursor: 'pointer', border: 'none',
              background: 'var(--bg-secondary, #252540)', color: 'var(--text-secondary, #bbb)',
            }}
          >
            Deny
          </button>
          <button
            onClick={() => onRespond(pending.requestId, true, remember)}
            style={{
              padding: '7px 18px', borderRadius: 8, fontSize: 12, fontWeight: 500,
              cursor: 'pointer', border: 'none',
              background: 'var(--accent, #7c5cfc)', color: '#fff',
            }}
          >
            Allow
          </button>
        </div>
      </div>
    </div>
  );
}
