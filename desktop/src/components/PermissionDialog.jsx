import React, { useState, useEffect } from 'react'
import { useSableStore } from '../hooks/useSable'

const ACTION_LABELS = {
  browser_navigate: 'Web Browsing',
  file_write: 'Write File',
  file_delete: 'Delete File',
  system_command: 'Run System Command',
  email_send: 'Send Email',
  email_read: 'Read Email',
  calendar_write: 'Modify Calendar',
}

const ACTION_ICONS = {
  browser_navigate: '🌐',
  file_write: '📝',
  file_delete: '🗑️',
  system_command: '⚙️',
  email_send: '📧',
  email_read: '📬',
  calendar_write: '📅',
}

const ACTION_DESCRIPTIONS = {
  browser_navigate: 'The agent wants to open a web page. This allows it to browse, scrape, or interact with websites on your behalf.',
  file_write: 'The agent wants to create or modify a file on your system. Review the path below to make sure it\'s expected.',
  file_delete: 'The agent wants to permanently delete a file. This action cannot be undone,  review carefully.',
  system_command: 'The agent wants to run a shell command on your machine. Only allow if you trust the command shown below.',
  email_send: 'The agent wants to send an email from your account. Check the recipient and content before approving.',
  email_read: 'The agent wants to read emails from your inbox. This lets it access message contents and metadata.',
  calendar_write: 'The agent wants to create or modify a calendar event. Review the details below.',
}

export default function PermissionDialog() {
  const pending = useSableStore(s => s.pendingPermission)
  const respond = useSableStore(s => s.respondPermission)
  const [remember, setRemember] = useState(false)
  const [countdown, setCountdown] = useState(60)

  useEffect(() => {
    if (!pending) return
    setCountdown(60)
    setRemember(false)
    const interval = setInterval(() => {
      setCountdown(prev => {
        if (prev <= 1) {
          // Auto-deny on timeout
          respond(pending.requestId, false)
          clearInterval(interval)
          return 0
        }
        return prev - 1
      })
    }, 1000)
    return () => clearInterval(interval)
  }, [pending?.requestId])

  if (!pending) return null

  const label = ACTION_LABELS[pending.action] || pending.action
  const icon = ACTION_ICONS[pending.action] || '🔐'
  const toolArgs = pending.arguments || {}
  const argEntries = Object.entries(toolArgs).filter(([, v]) => v !== undefined && v !== null && v !== '')

  return (
    <div className="permission-overlay">
      <div className="permission-dialog">
        <div className="permission-header">
          <span className="permission-icon">{icon}</span>
          <span className="permission-title">Permission Required</span>
          <span className="permission-countdown">{countdown}s</span>
        </div>

        <div className="permission-body">
          <p className="permission-question">
            Sable wants to use <strong>{label}</strong>
          </p>
          <p className="permission-description" style={{ fontSize: 11, opacity: 0.7, margin: '4px 0 10px', lineHeight: 1.5 }}>
            {ACTION_DESCRIPTIONS[pending.action] || 'The agent is requesting access to a protected action. Review the details below before deciding.'}
          </p>
          <div className="permission-tool">
            <code>{pending.tool}</code>
          </div>
          {argEntries.length > 0 && (
            <div className="permission-args">
              {argEntries.map(([k, v]) => (
                <div key={k} className="permission-arg">
                  <span className="permission-arg-key">{k}:</span>
                  <span className="permission-arg-val">{String(v)}</span>
                </div>
              ))}
            </div>
          )}
        </div>

        <label className="permission-remember">
          <input
            type="checkbox"
            checked={remember}
            onChange={e => setRemember(e.target.checked)}
          />
          Always allow this action
        </label>

        <div className="permission-actions">
          <button
            className="permission-btn deny"
            onClick={() => respond(pending.requestId, false)}
          >
            Deny
          </button>
          <button
            className="permission-btn allow"
            onClick={() => respond(pending.requestId, true, remember)}
          >
            Allow
          </button>
        </div>
      </div>
    </div>
  )
}
