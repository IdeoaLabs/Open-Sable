import React, { useState, useRef, useEffect, useCallback } from 'react'

const api = typeof window !== 'undefined' && window.sable ? window.sable : null
const MAX_RETRIES = 20
const RETRY_DELAY = 3000

export default function DashboardPanel({ config, onClose }) {
  const httpBase = (config?.wsUrl || 'ws://localhost:8789')
    .replace(/^ws:\/\//, 'http://')
    .replace(/^wss:\/\//, 'https://')
    .replace(/\/+$/, '')

  const authUrl = (path) => {
    const base = `${httpBase}${path}`
    return config?.token ? `${base}?token=${encodeURIComponent(config.token)}` : base
  }

  const dashUrl = authUrl('/dashboard')
  const [loaded, setLoaded] = useState(false)
  const [error, setError] = useState(false)
  const [retries, setRetries] = useState(0)
  const wvRef = useRef(null)
  const retryTimer = useRef(null)

  const reload = useCallback(() => {
    setError(false)
    setLoaded(false)
    wvRef.current?.reload()
  }, [])

  // Auto-retry on failure (backend may still be starting)
  useEffect(() => {
    if (error && retries < MAX_RETRIES) {
      retryTimer.current = setTimeout(() => {
        setRetries(r => r + 1)
        reload()
      }, RETRY_DELAY)
      return () => clearTimeout(retryTimer.current)
    }
  }, [error, retries, reload])

  // Reset retries on successful load
  useEffect(() => {
    if (loaded) setRetries(0)
  }, [loaded])

  // Attach webview events via ref (React doesn't support webview synthetic events)
  useEffect(() => {
    const wv = wvRef.current
    if (!wv) return
    const onFinish = () => setLoaded(true)
    const onFail = (e) => {
      // error code -3 is aborted navigation (e.g. redirect), not a real error
      if (e.errorCode && e.errorCode !== -3) setError(true)
    }
    wv.addEventListener('did-finish-load', onFinish)
    wv.addEventListener('did-fail-load', onFail)
    return () => {
      wv.removeEventListener('did-finish-load', onFinish)
      wv.removeEventListener('did-fail-load', onFail)
    }
  }, [])

  // Cleanup timer on unmount
  useEffect(() => () => clearTimeout(retryTimer.current), [])

  const openExternal = () => {
    if (api?.openExternal) {
      api.openExternal(dashUrl)
    } else {
      window.open(dashUrl, '_blank')
    }
  }

  const gaveUp = error && retries >= MAX_RETRIES

  return (
    <div className="dashboard-panel">
      <div className="dashboard-topbar">
        <div className="dashboard-topbar-left">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" width="14" height="14">
            <rect x="3" y="3" width="7" height="7"/><rect x="14" y="3" width="7" height="7"/>
            <rect x="3" y="14" width="7" height="7"/><rect x="14" y="14" width="7" height="7"/>
          </svg>
          <span>Agent Dashboard</span>
          <span className="dashboard-url-pill">{dashUrl}</span>
        </div>
        <div className="dashboard-topbar-right">
          <button className="dash-btn" onClick={openExternal} title="Open in browser">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" width="13" height="13">
              <path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6"/>
              <polyline points="15 3 21 3 21 9"/><line x1="10" y1="14" x2="21" y2="3"/>
            </svg>
            Open in browser
          </button>
          <button className="dash-btn dash-btn-close" onClick={onClose} title="Close">
            ✕ Close
          </button>
        </div>
      </div>

      {gaveUp ? (
        <div className="dashboard-error">
          <div className="dashboard-error-icon">⚠</div>
          <div className="dashboard-error-title">Dashboard not reachable</div>
          <div className="dashboard-error-sub">Make sure SableCore is running at <code>{httpBase}</code></div>
          <div style={{ display: 'flex', gap: 8, marginTop: 16, justifyContent: 'center' }}>
            <button className="btn btn-secondary" onClick={() => { setRetries(0); reload(); }}>Retry</button>
            <button className="btn btn-primary" onClick={openExternal}>Open in browser</button>
          </div>
        </div>
      ) : (
        <>
          {!loaded && (
            <div className="dashboard-loading">
              <div className="typing-indicator" style={{ margin: 'auto' }}>
                <div className="typing-dot"/><div className="typing-dot"/><div className="typing-dot"/>
              </div>
              <span>{error ? 'Starting SableCore…' : 'Loading dashboard…'}</span>
            </div>
          )}
          <webview
            ref={wvRef}
            className="dashboard-iframe"
            src={dashUrl}
            style={{ display: loaded ? 'flex' : 'none' }}
          />
        </>
      )}
    </div>
  )
}
