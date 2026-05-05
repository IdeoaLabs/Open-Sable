import React, { useEffect, useRef } from 'react'
import { useSableStore } from '../hooks/useSable.js'

/**
 * Desktop Brain Panel,  streamlined view of agent cognitive data.
 * Fetches brain.data via the gateway WebSocket (same as dashboard BrainPanel).
 */
export default function BrainPanel({ onClose }) {
  const brainData = useSableStore(s => s.brainData)
  const brainLoading = useSableStore(s => s.brainLoading)
  const fetchBrain = useSableStore(s => s.fetchBrain)
  const wsStatus = useSableStore(s => s.wsStatus)
  const intervalRef = useRef(null)

  useEffect(() => {
    fetchBrain()
    intervalRef.current = setInterval(fetchBrain, 8000)
    return () => clearInterval(intervalRef.current)
  }, [fetchBrain])

  const d = brainData || {}
  const il = d.inner_life || {}
  const auto = d.autonomous || {}
  const identity = d.identity || null
  const goals = d.goals || {}
  const refs = d.reflections || []
  const cogMem = d.cognitive_memory_count ?? 0
  const proposals = d.proactive_proposals || []
  const journal = d.journal || []
  const moodHistory = d.mood_history || []
  const innerMonologue = d.inner_monologue || []
  const autoLive = d.autonomous_live || {}

  const rawEmotion = il.emotion || d.inner_life_stats?.emotion || ', '
  const emotion = typeof rawEmotion === 'object' ? (rawEmotion.primary || JSON.stringify(rawEmotion)) : String(rawEmotion)
  const valence = (typeof rawEmotion === 'object' ? rawEmotion.valence : null) ?? il.valence ?? 0
  const arousal = (typeof rawEmotion === 'object' ? rawEmotion.arousal : null) ?? il.arousal ?? 0
  const tick = autoLive.tick || auto.tick || 0
  const queueSize = autoLive.queue_size ?? auto.pending_tasks ?? 0
  const completedCount = autoLive.completed_count ?? auto.completed_tasks_count ?? 0

  const impulse = typeof il.impulse === 'object' ? (il.impulse.description || il.impulse.action || JSON.stringify(il.impulse)) : il.impulse
  const fantasy = typeof il.fantasy === 'object' ? (il.fantasy.description || il.fantasy.content || JSON.stringify(il.fantasy)) : il.fantasy

  const goalList = [
    ...(goals.long_term || []),
    ...(goals.short_term || []),
    ...(goals.active || []),
  ].slice(0, 8)

  if (wsStatus !== 'connected') {
    return (
      <div className="brain-panel">
        <div className="brain-topbar">
          <span>🧠</span>
          <span className="brain-topbar-title">Cognitive Brain</span>
          {onClose && <button className="brain-topbar-btn" onClick={onClose}>✕</button>}
        </div>
        <div className="brain-empty">
          <div className="brain-empty-icon">🔌</div>
          <div className="brain-empty-text">Disconnected,  waiting for agent…</div>
        </div>
      </div>
    )
  }

  if (!brainData) {
    return (
      <div className="brain-panel">
        <div className="brain-topbar">
          <span>🧠</span>
          <span className="brain-topbar-title">Cognitive Brain</span>
          {onClose && <button className="brain-topbar-btn" onClick={onClose}>✕</button>}
        </div>
        <div className="brain-empty">
          <div className="brain-empty-icon">{brainLoading ? '⏳' : '🧠'}</div>
          <div className="brain-empty-text">{brainLoading ? 'Loading brain data…' : 'No brain data yet'}</div>
        </div>
      </div>
    )
  }

  return (
    <div className="brain-panel">
      <div className="brain-topbar">
        <span>🧠</span>
        <span className="brain-topbar-title">Cognitive Brain</span>
        {identity?.name && (
          <span style={{ fontSize: 11, color: '#2dd4bf', fontWeight: 600, marginLeft: 8 }}>
            {identity.name}
          </span>
        )}
        <span style={{ fontSize: 10, color: 'var(--text-muted)', fontFamily: 'var(--mono, monospace)', marginLeft: 'auto' }}>
          TICK {tick}
        </span>
        <button className="brain-topbar-btn" onClick={fetchBrain} title="Refresh">
          {brainLoading ? '⏳' : '↻'} Refresh
        </button>
        {onClose && <button className="brain-topbar-btn" onClick={onClose}>✕ Close</button>}
      </div>

      <div className="brain-body">
        {/* ── Stats Overview ─────────────────────────────── */}
        <div className="brain-stat-grid">
          <div className="brain-stat">
            <div className="brain-stat-label">Emotion</div>
            <div className="brain-stat-value" style={{ textTransform: 'capitalize', fontSize: 14 }}>{emotion}</div>
          </div>
          <div className="brain-stat">
            <div className="brain-stat-label">Valence</div>
            <div className="brain-stat-value">{typeof valence === 'number' ? valence.toFixed(2) : valence}</div>
          </div>
          <div className="brain-stat">
            <div className="brain-stat-label">Arousal</div>
            <div className="brain-stat-value">{typeof arousal === 'number' ? arousal.toFixed(2) : arousal}</div>
          </div>
          <div className="brain-stat">
            <div className="brain-stat-label">Memories</div>
            <div className="brain-stat-value">{cogMem}</div>
          </div>
          <div className="brain-stat">
            <div className="brain-stat-label">Queue</div>
            <div className="brain-stat-value">{queueSize}</div>
          </div>
          <div className="brain-stat">
            <div className="brain-stat-label">Completed</div>
            <div className="brain-stat-value">{completedCount}</div>
          </div>
        </div>

        {/* ── Identity ───────────────────────────────────── */}
        {identity && (
          <div className="brain-card">
            <div className="brain-card-header"><span className="brain-card-icon">🪪</span> Identity</div>
            <div className="brain-card-body">
              {identity.core_values && <div><strong>Core Values:</strong> {Array.isArray(identity.core_values) ? identity.core_values.join(', ') : identity.core_values}</div>}
              {identity.personality_summary && <div style={{ marginTop: 4 }}>{identity.personality_summary}</div>}
              {identity.personality_traits && typeof identity.personality_traits === 'object' && (
                <div style={{ marginTop: 6, display: 'flex', gap: 8, flexWrap: 'wrap' }}>
                  {Object.entries(identity.personality_traits).map(([k, v]) => (
                    <span key={k} style={{ fontSize: 10, padding: '2px 8px', borderRadius: 99, background: 'rgba(124,58,237,.1)', color: '#a78bfa' }}>
                      {k}: {typeof v === 'number' ? v.toFixed(2) : v}
                    </span>
                  ))}
                </div>
              )}
            </div>
          </div>
        )}

        {/* ── Impulse & Fantasy ───────────────────────────── */}
        {(impulse || fantasy) && (
          <div className="brain-card">
            <div className="brain-card-header"><span className="brain-card-icon">⚡</span> Inner Life</div>
            <div className="brain-card-body">
              {impulse && <div><strong>Impulse:</strong> {impulse}</div>}
              {fantasy && <div style={{ marginTop: 4 }}><strong>Fantasy:</strong> {fantasy}</div>}
            </div>
          </div>
        )}

        {/* ── Goals ──────────────────────────────────────── */}
        {goalList.length > 0 && (
          <div className="brain-card">
            <div className="brain-card-header"><span className="brain-card-icon">🎯</span> Goals ({goalList.length})</div>
            <div className="brain-card-body">
              {goalList.map((g, i) => (
                <div key={i} className="brain-memory-item">
                  {typeof g === 'string' ? g : (g.description || g.goal || g.text || JSON.stringify(g))}
                </div>
              ))}
            </div>
          </div>
        )}

        {/* ── Inner Monologue ────────────────────────────── */}
        {innerMonologue.length > 0 && (
          <div className="brain-card">
            <div className="brain-card-header"><span className="brain-card-icon">💭</span> Inner Monologue</div>
            <div className="brain-card-body">
              {innerMonologue.slice(-5).map((m, i) => (
                <div key={i} className="brain-memory-item" style={{ fontStyle: 'italic' }}>
                  {typeof m === 'string' ? m : (m.text || m.content || JSON.stringify(m))}
                </div>
              ))}
            </div>
          </div>
        )}

        {/* ── Reflections ────────────────────────────────── */}
        {refs.length > 0 && (
          <div className="brain-card">
            <div className="brain-card-header"><span className="brain-card-icon">🔍</span> Reflections ({refs.length})</div>
            <div className="brain-card-body">
              {refs.slice(-5).map((r, i) => (
                <div key={i} className="brain-memory-item">
                  {typeof r === 'string' ? r : (r.text || r.content || r.summary || JSON.stringify(r))}
                </div>
              ))}
            </div>
          </div>
        )}

        {/* ── Proactive Proposals ─────────────────────────── */}
        {proposals.length > 0 && (
          <div className="brain-card">
            <div className="brain-card-header"><span className="brain-card-icon">💡</span> Proposals ({proposals.length})</div>
            <div className="brain-card-body">
              {proposals.slice(-5).map((p, i) => (
                <div key={i} className="brain-memory-item">
                  {typeof p === 'string' ? p : (p.text || p.proposal || p.description || JSON.stringify(p))}
                </div>
              ))}
            </div>
          </div>
        )}

        {/* ── Journal ────────────────────────────────────── */}
        {journal.length > 0 && (
          <div className="brain-card">
            <div className="brain-card-header"><span className="brain-card-icon">📔</span> Journal ({journal.length})</div>
            <div className="brain-card-body">
              {journal.slice(-3).map((j, i) => (
                <div key={i} className="brain-memory-item">
                  {typeof j === 'string' ? j : (j.entry || j.text || j.content || JSON.stringify(j))}
                </div>
              ))}
            </div>
          </div>
        )}

        {/* ── Mood History ───────────────────────────────── */}
        {moodHistory.length > 0 && (
          <div className="brain-card">
            <div className="brain-card-header"><span className="brain-card-icon">📈</span> Mood History</div>
            <div className="brain-card-body">
              <div style={{ display: 'flex', gap: 4, flexWrap: 'wrap' }}>
                {moodHistory.slice(-20).map((m, i) => {
                  const mood = typeof m === 'object' ? (m.emotion || m.mood || '?') : m
                  return (
                    <span key={i} style={{
                      fontSize: 10, padding: '2px 6px', borderRadius: 99,
                      background: 'var(--surface-2, rgba(255,255,255,.05))',
                      color: 'var(--text-dim)',
                    }}>
                      {mood}
                    </span>
                  )
                })}
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
