import React, { useEffect, useCallback, useState, useRef } from 'react'

// ─── Sun / Moon SVG icons for the theme toggle ────────────────────────────
const SunIcon = () => (
  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <circle cx="12" cy="12" r="4"/>
    <line x1="12" y1="2" x2="12" y2="5"/>
    <line x1="12" y1="19" x2="12" y2="22"/>
    <line x1="4.22" y1="4.22" x2="6.34" y2="6.34"/>
    <line x1="17.66" y1="17.66" x2="19.78" y2="19.78"/>
    <line x1="2" y1="12" x2="5" y2="12"/>
    <line x1="19" y1="12" x2="22" y2="12"/>
    <line x1="4.22" y1="19.78" x2="6.34" y2="17.66"/>
    <line x1="17.66" y1="6.34" x2="19.78" y2="4.22"/>
  </svg>
)
const MoonIcon = () => (
  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
    <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/>
  </svg>
)
import { useSableStore } from './hooks/useSable.js'
import Sidebar from './components/Sidebar.jsx'
import ChatArea from './components/ChatArea.jsx'
import SettingsDialog from './components/SettingsDialog.jsx'
import DashboardPanel from './components/DashboardPanel.jsx'
import DevStudioPanel from './components/DevStudioPanel.jsx'
import ModelSelector from './components/ModelSelector.jsx'
import BrainPanel from './components/BrainPanel.jsx'
import PermissionDialog from './components/PermissionDialog.jsx'
import LoadingOverlay from './components/LoadingOverlay.jsx'

// ─── Window control helpers ────────────────────────────────────────────────
const api = typeof window !== 'undefined' && window.sable ? window.sable : null

export default function App() {
  const connect = useSableStore(s => s.connect)
  const setConfig = useSableStore(s => s.setConfig)
  const config = useSableStore(s => s.config)
  const settingsOpen = useSableStore(s => s.settingsOpen)
  const toast = useSableStore(s => s.toast)
  const newChat = useSableStore(s => s.newChat)
  const wsStatus = useSableStore(s => s.wsStatus)

  // ── Theme: dark (default) / light ─────────────────────────────────────────
  const [isDark, setIsDark] = useState(
    () => localStorage.getItem('sable-theme') !== 'light'
  )
  const [sidebarCollapsed, setSidebarCollapsed] = useState(
    () => localStorage.getItem('sable-sidebar') === 'collapsed'
  )
  const [dashboardOpen, setDashboardOpen] = useState(false)
  const [dashboardMounted, setDashboardMounted] = useState(false)
  const [devStudioOpen, setDevStudioOpen] = useState(false)
  const [brainOpen, setBrainOpen] = useState(false)

  // ── Platform detection (macOS puts controls on the left) ──────────────
  const platform = api?.platform || 'linux'
  const isMac = platform === 'darwin'

  const toggleSidebar = useCallback(() => {
    setSidebarCollapsed(v => {
      const next = !v
      localStorage.setItem('sable-sidebar', next ? 'collapsed' : 'open')
      return next
    })
  }, [])
  useEffect(() => {
    if (isDark) {
      document.documentElement.classList.remove('light-theme')
    } else {
      document.documentElement.classList.add('light-theme')
    }
    localStorage.setItem('sable-theme', isDark ? 'dark' : 'light')
  }, [isDark])

  // ── Init: load config from Electron then connect ──────────────────────────
  useEffect(() => {
    const init = async () => {
      if (api?.getConfig) {
        try {
          const cfg = await api.getConfig()
          setConfig(cfg)
          connect(cfg)
        } catch {
          connect(config)
        }
      } else {
        // Running in browser preview,  connect with defaults
        connect(config)
      }
    }
    init()
  }, []) // eslint-disable-line react-hooks/exhaustive-deps

  // ── Keyboard shortcuts ────────────────────────────────────────────────────
  const handleKeyDown = useCallback((e) => {
    if ((e.ctrlKey || e.metaKey) && e.key === 'n') {
      e.preventDefault()
      newChat()
    }
    if ((e.ctrlKey || e.metaKey) && e.key === 'b') {
      e.preventDefault()
      toggleSidebar()
    }
    // Ctrl+Shift+D → Dev Studio (check first,  more specific)
    if ((e.ctrlKey || e.metaKey) && e.shiftKey && e.key.toLowerCase() === 'd') {
      e.preventDefault()
      setDevStudioOpen(v => !v)
      setDashboardOpen(false)
      setBrainOpen(false)
      return
    }
    // Ctrl+D → Dashboard
    if ((e.ctrlKey || e.metaKey) && !e.shiftKey && e.key === 'd') {
      e.preventDefault()
      setDashboardOpen(v => !v)
      setDashboardMounted(true)
      setDevStudioOpen(false)
      setBrainOpen(false)
    }
  }, [newChat, toggleSidebar])

  useEffect(() => {
    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [handleKeyDown])

  // ── Window focus/blur → color the traffic light buttons correctly ────────
  useEffect(() => {
    const onBlur  = () => document.querySelector('.app')?.classList.add('window-blurred')
    const onFocus = () => document.querySelector('.app')?.classList.remove('window-blurred')
    window.addEventListener('blur',  onBlur)
    window.addEventListener('focus', onFocus)
    return () => {
      window.removeEventListener('blur',  onBlur)
      window.removeEventListener('focus', onFocus)
    }
  }, [])

  // ── Auto-hide sidebar when Dev Studio opens (not part of webview) ─────
  useEffect(() => {
    if (devStudioOpen) {
      setSidebarCollapsed(true)
      localStorage.setItem('sable-sidebar', 'collapsed')
    }
  }, [devStudioOpen])

  return (
    <div className="app" data-platform={platform}>
      {/* ── Titlebar ─────────────────────────────────────────────────────── */}
      <div className="titlebar">
        <div className="titlebar-left">
          {/* macOS traffic lights (close / minimize / maximize on the LEFT) */}
          {isMac && (
            <div className="mac-controls" style={{ WebkitAppRegion: 'no-drag' }}>
              <button className="mac-btn mac-close" onClick={() => api?.close()} title="Close">
                <svg viewBox="0 0 12 12"><line x1="3" y1="3" x2="9" y2="9"/><line x1="9" y1="3" x2="3" y2="9"/></svg>
              </button>
              <button className="mac-btn mac-minimize" onClick={() => api?.minimize()} title="Minimize">
                <svg viewBox="0 0 12 12"><line x1="2" y1="6" x2="10" y2="6"/></svg>
              </button>
              <button className="mac-btn mac-maximize" onClick={() => api?.maximize()} title="Maximize">
                <svg viewBox="0 0 12 12"><polyline points="4 2 10 2 10 8"/><polyline points="8 10 2 10 2 4"/></svg>
              </button>
            </div>
          )}
          {/* Only show hamburger in titlebar when sidebar is collapsed */}
          {sidebarCollapsed && (
            <button
              className="sidebar-toggle-btn"
              onClick={toggleSidebar}
              title="Show sidebar (Ctrl+B)"
              style={{ WebkitAppRegion: 'no-drag' }}
            >
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" width="14" height="14">
                <line x1="3" y1="6" x2="21" y2="6"/><line x1="3" y1="12" x2="21" y2="12"/><line x1="3" y1="18" x2="21" y2="18"/>
              </svg>
            </button>
          )}
          <div className="titlebar-status">
            <div className={`titlebar-status-dot ${wsStatus}`} />
            SableCore
          </div>
          <ModelSelector />
        </div>

        {/* ── Day / Night toggle ──────────────────────────────────────── */}
        <div className="theme-toggle" style={{ WebkitAppRegion: 'no-drag' }}>
          <input
            type="checkbox"
            id="theme-check"
            checked={!isDark}
            onChange={() => setIsDark(d => !d)}
          />
          <label
            htmlFor="theme-check"
            className="theme-toggle-label"
            title={isDark ? 'Switch to light mode' : 'Switch to dark mode'}
          >
            <div className="theme-box">
              <div className="theme-ball" />
              <div className="theme-scenary">
                <span className="theme-icon theme-moon"><MoonIcon /></span>
                <span className="theme-icon theme-sun"><SunIcon /></span>
              </div>
            </div>
          </label>
        </div>

        <div className="titlebar-actions" style={{ WebkitAppRegion: 'no-drag' }}>
          <button
            className={`titlebar-btn ${devStudioOpen ? 'active' : ''}`}
            title="Dev Studio (Ctrl+Shift+D)"
            onClick={() => { setDevStudioOpen(v => !v); setDashboardOpen(false); setBrainOpen(false) }}
          >
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" width="13" height="13">
              <polyline points="16 18 22 12 16 6" />
              <polyline points="8 6 2 12 8 18" />
            </svg>
          </button>
          <button
            className={`titlebar-btn ${brainOpen ? 'active' : ''}`}
            title="Brain Panel"
            onClick={() => { setBrainOpen(v => !v); setDashboardOpen(false); setDevStudioOpen(false) }}
          >
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" width="13" height="13">
              <path d="M12 2a7 7 0 0 0-7 7c0 2.38 1.19 4.47 3 5.74V17a2 2 0 0 0 2 2h4a2 2 0 0 0 2-2v-2.26c1.81-1.27 3-3.36 3-5.74a7 7 0 0 0-7-7z"/>
              <line x1="10" y1="21" x2="14" y2="21"/>
            </svg>
          </button>
          <button
            className={`titlebar-btn ${dashboardOpen ? 'active' : ''}`}
            title="Agent Dashboard (Ctrl+D)"
            onClick={() => { setDashboardOpen(v => !v); setDashboardMounted(true); setDevStudioOpen(false); setBrainOpen(false) }}
          >
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" width="13" height="13">
              <rect x="3" y="3" width="7" height="7"/><rect x="14" y="3" width="7" height="7"/>
              <rect x="3" y="14" width="7" height="7"/><rect x="14" y="14" width="7" height="7"/>
            </svg>
          </button>
          {/* Windows / Linux window controls on the RIGHT */}
          {!isMac && (
            <>
              <button className="titlebar-btn" title="Minimize" onClick={() => api?.minimize()}>, </button>
              <button className="titlebar-btn" title="Maximize" onClick={() => api?.maximize()}>□</button>
              <button className="titlebar-btn close" title="Close" onClick={() => api?.close()}>✕</button>
            </>
          )}
        </div>
      </div>

      {/* ── Main content ─────────────────────────────────────────────────── */}
      <div className="main">
        <Sidebar collapsed={sidebarCollapsed} onToggle={toggleSidebar} onOpenDevStudio={() => { setDevStudioOpen(true); setDashboardOpen(false); setBrainOpen(false) }} />
        {devStudioOpen
          ? <DevStudioPanel onClose={() => setDevStudioOpen(false)} />
          : brainOpen
            ? <BrainPanel onClose={() => setBrainOpen(false)} />
            : !dashboardOpen
              ? <ChatArea />
              : null
        }
        {/* Keep Dashboard webview mounted (hidden) so it reconnects instantly */}
        {dashboardMounted && (
          <div style={{ display: dashboardOpen && !devStudioOpen && !brainOpen ? 'contents' : 'none' }}>
            <DashboardPanel config={config} onClose={() => setDashboardOpen(false)} />
          </div>
        )}
      </div>

      {/* ── Modals / overlays ────────────────────────────────────────────── */}
      {settingsOpen && <SettingsDialog />}
      <PermissionDialog />
      <LoadingOverlay />
      {toast && <div className="toast">{toast}</div>}
    </div>
  )
}
