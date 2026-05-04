import React, { useState, useRef, useEffect } from 'react'

const api = typeof window !== 'undefined' && window.sable ? window.sable : null

// Code / IDE icon
const CodeIcon = () => (
  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" width="14" height="14">
    <polyline points="16 18 22 12 16 6" />
    <polyline points="8 6 2 12 8 18" />
  </svg>
)

export default function DevStudioPanel({ onClose }) {
  const devUrl = 'http://localhost:5700'
  const [loaded, setLoaded] = useState(false)
  const [error, setError] = useState(false)
  const wvRef = useRef(null)
  const retryCountRef = useRef(0)
  const maxRetries = 20  // Next.js can take up to 60s to start on Windows cold boot

  useEffect(() => {
    const wv = wvRef.current
    if (!wv) return
    const onFinish = () => {
      retryCountRef.current = 0
      setLoaded(true)
      setError(false)
    }
    const onFail = (e) => {
      // -3 = ERR_ABORTED (normal), -6 = ERR_FILE_NOT_FOUND (HMR), -2 = ERR_FAILED (transient)
      const ignoreCodes = [-3, -6, -2, -501]
      if (e.errorCode && !ignoreCodes.includes(e.errorCode)) {
        // Auto-retry,  Next.js cold start on Windows can take 20-60s
        if (retryCountRef.current < maxRetries) {
          retryCountRef.current++
          const delay = retryCountRef.current <= 5 ? 3000 : 5000
          console.log(`[DevStudio] Load failed (code ${e.errorCode}), retry ${retryCountRef.current}/${maxRetries} in ${delay/1000}s...`)
          setTimeout(() => wv.reload(), delay)
        } else {
          setError(true)
        }
      }
    }
    wv.addEventListener('did-finish-load', onFinish)
    wv.addEventListener('did-fail-load', onFail)
    return () => {
      wv.removeEventListener('did-finish-load', onFinish)
      wv.removeEventListener('did-fail-load', onFail)
    }
  }, [])

  const openExternal = () => {
    if (api?.openExternal) {
      api.openExternal(devUrl)
    } else {
      window.open(devUrl, '_blank')
    }
  }

  const reload = () => {
    setError(false)
    setLoaded(false)
    wvRef.current?.reload()
  }

  return (
    <div className="dashboard-panel">
      <div className="dashboard-topbar">
        <div className="dashboard-topbar-left">
          <CodeIcon />
          <span>Dev Studio</span>
          <span className="dashboard-url-pill">{devUrl}</span>
        </div>
        <div className="dashboard-topbar-right">
          <button className="dash-btn" onClick={openExternal} title="Open in browser">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" width="13" height="13">
              <path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6"/>
              <polyline points="15 3 21 3 21 9"/><line x1="10" y1="14" x2="21" y2="3"/>
            </svg>
            Open in browser
          </button>
          <button className="dash-btn" onClick={reload} title="Reload">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" width="13" height="13">
              <polyline points="23 4 23 10 17 10"/>
              <path d="M20.49 15a9 9 0 1 1-2.12-9.36L23 10"/>
            </svg>
            Reload
          </button>
          <button className="dash-btn dash-btn-close" onClick={onClose} title="Close">
            ✕ Close
          </button>
        </div>
      </div>

      {error ? (
        <div className="dashboard-error">
          <div className="dashboard-error-icon">⚠</div>
          <div className="dashboard-error-title">Dev Studio not reachable</div>
          <div className="dashboard-error-sub">
            Make sure Sable Dev is running at <code>{devUrl}</code>
            <br />
            <code style={{ fontSize: 11, marginTop: 4, display: 'inline-block' }}>
              cd sable_dev && npm run dev
            </code>
          </div>
          <div style={{ display: 'flex', gap: 8, marginTop: 16, justifyContent: 'center' }}>
            <button className="btn btn-secondary" onClick={reload}>Retry</button>
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
              <span>Loading Dev Studio…</span>
            </div>
          )}
          <webview
            ref={wvRef}
            className="dashboard-iframe"
            src={devUrl}
            style={{ display: loaded ? 'flex' : 'none' }}
          />
        </>
      )}
    </div>
  )
}
