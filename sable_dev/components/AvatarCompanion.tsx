'use client';

import { useState, useEffect, useCallback } from 'react';

interface AvatarData {
  id: string;
  displayName: string;
  species: string;
  speciesEmoji: string;
  personality: string[];
  stats: Record<string, number>;
  rarity: string;
  shiny: boolean;
  colorPalette: string[];
  level: number;
  xp: number;
  achievements: string[];
  card: string;
}

interface AvatarCompanionProps {
  isOpen: boolean;
  onClose: () => void;
  latestReaction?: string | null;
}

const RARITY_COLORS: Record<string, string> = {
  common: '#9ca3af',
  uncommon: '#22c55e',
  rare: '#3b82f6',
  epic: '#a855f7',
  legendary: '#f59e0b',
};

export default function AvatarCompanion({ isOpen, onClose, latestReaction }: AvatarCompanionProps) {
  const [avatar, setAvatar] = useState<AvatarData | null>(null);
  const [loading, setLoading] = useState(false);
  const [renaming, setRenaming] = useState(false);
  const [newName, setNewName] = useState('');
  const [showReaction, setShowReaction] = useState(false);

  const fetchAvatar = useCallback(async () => {
    setLoading(true);
    try {
      const res = await fetch('/api/avatar');
      const data = await res.json();
      if (data.success) setAvatar(data.avatar);
    } catch (err) {
      console.error('Failed to fetch avatar:', err);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    if (isOpen && !avatar) fetchAvatar();
  }, [isOpen, avatar, fetchAvatar]);

  useEffect(() => {
    if (latestReaction) {
      setShowReaction(true);
      const timer = setTimeout(() => setShowReaction(false), 4000);
      return () => clearTimeout(timer);
    }
  }, [latestReaction]);

  const handleRename = async () => {
    if (!newName.trim()) return;
    try {
      const res = await fetch('/api/avatar', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ action: 'rename', name: newName.trim() }),
      });
      const data = await res.json();
      if (data.success) {
        setAvatar(prev => prev ? { ...prev, displayName: data.name || newName.trim() } : null);
        setRenaming(false);
        setNewName('');
      }
    } catch (err) {
      console.error('Failed to rename avatar:', err);
    }
  };

  if (!isOpen && !showReaction) return null;

  // Floating reaction bubble (shows even when panel is closed)
  if (!isOpen && showReaction && latestReaction) {
    return (
      <div style={{
        position: 'fixed',
        bottom: '80px',
        right: '20px',
        background: '#1e1e2e',
        border: '1px solid #313244',
        borderRadius: '12px',
        padding: '8px 14px',
        color: '#cdd6f4',
        fontSize: '13px',
        zIndex: 9999,
        maxWidth: '280px',
        animation: 'avatarFadeIn 0.3s ease-out',
        boxShadow: '0 4px 12px rgba(0,0,0,0.3)',
      }}>
        {avatar?.speciesEmoji && <span style={{ marginRight: '6px' }}>{avatar.speciesEmoji}</span>}
        {latestReaction}
      </div>
    );
  }

  const rarityColor = avatar ? (RARITY_COLORS[avatar.rarity] || '#9ca3af') : '#9ca3af';
  const xpPercent = avatar ? ((avatar.xp % 100) / 100) * 100 : 0;

  return (
    <div style={{
      position: 'fixed',
      top: 0,
      right: 0,
      width: '340px',
      height: '100vh',
      background: '#1e1e2e',
      borderLeft: '1px solid #313244',
      zIndex: 9998,
      display: 'flex',
      flexDirection: 'column',
      color: '#cdd6f4',
      fontFamily: 'monospace',
      animation: 'avatarSlideIn 0.25s ease-out',
    }}>
      {/* Header */}
      <div style={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        padding: '14px 16px',
        borderBottom: '1px solid #313244',
      }}>
        <span style={{ fontWeight: 600, fontSize: '14px' }}>Avatar Companion</span>
        <button 
          onClick={onClose}
          style={{
            background: 'none',
            border: 'none',
            color: '#6c7086',
            cursor: 'pointer',
            fontSize: '18px',
            padding: '2px 6px',
          }}
        >
          ✕
        </button>
      </div>

      {loading ? (
        <div style={{ padding: '40px', textAlign: 'center', color: '#6c7086' }}>
          Loading avatar...
        </div>
      ) : avatar ? (
        <div style={{ flex: 1, overflowY: 'auto', padding: '16px' }}>
          {/* Avatar Sprite */}
          <div style={{ textAlign: 'center', marginBottom: '16px' }}>
            <div style={{ 
              fontSize: '64px', 
              filter: avatar.shiny ? 'drop-shadow(0 0 12px gold)' : 'none',
            }}>
              {avatar.speciesEmoji}
            </div>
            {avatar.shiny && (
              <span style={{ color: '#f59e0b', fontSize: '11px' }}>✨ SHINY ✨</span>
            )}
          </div>

          {/* Name + Rarity */}
          <div style={{ textAlign: 'center', marginBottom: '12px' }}>
            {renaming ? (
              <div style={{ display: 'flex', gap: '6px', justifyContent: 'center' }}>
                <input
                  value={newName}
                  onChange={e => setNewName(e.target.value)}
                  onKeyDown={e => e.key === 'Enter' && handleRename()}
                  maxLength={30}
                  placeholder="New name..."
                  style={{
                    background: '#181825',
                    border: '1px solid #313244',
                    borderRadius: '6px',
                    color: '#cdd6f4',
                    padding: '4px 8px',
                    fontSize: '13px',
                    width: '140px',
                  }}
                  autoFocus
                />
                <button 
                  onClick={handleRename}
                  style={{
                    background: '#89b4fa',
                    border: 'none',
                    borderRadius: '6px',
                    color: '#1e1e2e',
                    padding: '4px 10px',
                    cursor: 'pointer',
                    fontSize: '12px',
                  }}
                >
                  Save
                </button>
                <button 
                  onClick={() => { setRenaming(false); setNewName(''); }}
                  style={{
                    background: '#313244',
                    border: 'none',
                    borderRadius: '6px',
                    color: '#cdd6f4',
                    padding: '4px 8px',
                    cursor: 'pointer',
                    fontSize: '12px',
                  }}
                >
                  Cancel
                </button>
              </div>
            ) : (
              <div>
                <span 
                  style={{ fontSize: '18px', fontWeight: 700, cursor: 'pointer' }}
                  onClick={() => setRenaming(true)}
                  title="Click to rename"
                >
                  {avatar.displayName}
                </span>
                <div style={{ 
                  color: rarityColor, 
                  fontSize: '11px', 
                  textTransform: 'uppercase',
                  fontWeight: 600,
                  marginTop: '2px',
                }}>
                  {avatar.rarity} {avatar.species}
                </div>
                <div style={{ color: '#6c7086', fontSize: '11px', marginTop: '2px' }}>
                  {avatar.personality.join(', ')}
                </div>
              </div>
            )}
          </div>

          {/* Level + XP Bar */}
          <div style={{ marginBottom: '16px' }}>
            <div style={{ 
              display: 'flex', 
              justifyContent: 'space-between', 
              fontSize: '11px', 
              marginBottom: '4px',
              color: '#a6adc8',
            }}>
              <span>Level {avatar.level}</span>
              <span>{avatar.xp % 100}/100 XP</span>
            </div>
            <div style={{
              height: '6px',
              background: '#313244',
              borderRadius: '3px',
              overflow: 'hidden',
            }}>
              <div style={{
                height: '100%',
                width: `${xpPercent}%`,
                background: `linear-gradient(90deg, ${rarityColor}, #89b4fa)`,
                borderRadius: '3px',
                transition: 'width 0.3s ease',
              }} />
            </div>
          </div>

          {/* Stats */}
          <div style={{ marginBottom: '16px' }}>
            <div style={{ fontSize: '12px', fontWeight: 600, marginBottom: '8px', color: '#a6adc8' }}>
              STATS
            </div>
            {Object.entries(avatar.stats).map(([stat, value]) => (
              <div key={stat} style={{ 
                display: 'flex', 
                alignItems: 'center', 
                gap: '8px', 
                marginBottom: '4px',
                fontSize: '12px',
              }}>
                <span style={{ width: '70px', color: '#a6adc8', textTransform: 'capitalize' }}>
                  {stat}
                </span>
                <div style={{
                  flex: 1,
                  height: '4px',
                  background: '#313244',
                  borderRadius: '2px',
                  overflow: 'hidden',
                }}>
                  <div style={{
                    height: '100%',
                    width: `${value}%`,
                    background: value > 70 ? '#a6e3a1' : value > 40 ? '#f9e2af' : '#f38ba8',
                    borderRadius: '2px',
                  }} />
                </div>
                <span style={{ width: '24px', textAlign: 'right', color: '#6c7086', fontSize: '11px' }}>
                  {value}
                </span>
              </div>
            ))}
          </div>

          {/* Achievements */}
          {avatar.achievements.length > 0 && (
            <div style={{ marginBottom: '16px' }}>
              <div style={{ fontSize: '12px', fontWeight: 600, marginBottom: '8px', color: '#a6adc8' }}>
                ACHIEVEMENTS
              </div>
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: '4px' }}>
                {avatar.achievements.map(achievement => (
                  <span key={achievement} style={{
                    background: '#313244',
                    borderRadius: '4px',
                    padding: '2px 8px',
                    fontSize: '11px',
                    color: '#f9e2af',
                  }}>
                    🏆 {achievement}
                  </span>
                ))}
              </div>
            </div>
          )}

          {/* Reaction */}
          {showReaction && latestReaction && (
            <div style={{
              background: '#181825',
              borderRadius: '8px',
              padding: '10px 14px',
              fontSize: '13px',
              color: '#cdd6f4',
              border: '1px solid #313244',
              marginBottom: '12px',
            }}>
              💬 {latestReaction}
            </div>
          )}
        </div>
      ) : (
        <div style={{ padding: '40px', textAlign: 'center', color: '#6c7086' }}>
          No avatar found
        </div>
      )}

      <style>{`
        @keyframes avatarSlideIn {
          from { transform: translateX(100%); }
          to { transform: translateX(0); }
        }
        @keyframes avatarFadeIn {
          from { opacity: 0; transform: translateY(10px); }
          to { opacity: 1; transform: translateY(0); }
        }
      `}</style>
    </div>
  );
}
