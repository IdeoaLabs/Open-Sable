import { useState } from 'react';
import {
  MessageSquare, Activity, Terminal, BarChart3,
  QrCode, Settings, Cpu, Smartphone,
  TrendingUp, Sparkles, Clock, Brain, Orbit, Monitor, SlidersHorizontal,
} from 'lucide-react';

const tabs = [
  { id: 'chat',     icon: MessageSquare, label: 'Chat' },
  { id: 'activity', icon: Activity,      label: 'Activity' },
  { id: 'terminal', icon: Terminal,      label: 'Terminal' },
  { id: 'status',   icon: BarChart3,     label: 'Status' },
  { id: 'trading',  icon: TrendingUp,    label: 'Trading' },
  { id: 'tasks',    icon: Sparkles,      label: 'Tasks' },
  { id: 'history',  icon: Clock,         label: 'History' },
  { id: 'thoughts', icon: Brain,         label: 'Thoughts' },
  { id: 'brain',    icon: Orbit,         label: 'Brain' },
  { id: 'qr',       icon: QrCode,        label: 'QR Pair' },
  { id: 'agent',    icon: Cpu,           label: 'Agent' },
  { id: 'devices',  icon: Smartphone,    label: 'Devices' },
  { id: 'display',  icon: Monitor,             label: 'Display' },
  { id: 'config',   icon: SlidersHorizontal,   label: 'Config' },
  { id: 'settings', icon: Settings,             label: 'Settings' },
];

const sidebarStyles = {
  sidebar: {
    width: 56, minWidth: 56, background: 'var(--bg-secondary)',
    borderRight: '1px solid var(--border)', display: 'flex', flexDirection: 'column',
    alignItems: 'center', padding: '8px 0', gap: 2, flexShrink: 0,
    overflowY: 'auto',
  },
  btn: {
    width: 40, height: 40, borderRadius: 'var(--radius-sm)', border: 'none',
    background: 'transparent', color: 'rgba(255,255,255,0.7)', cursor: 'pointer',
    display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center',
    gap: 2, fontSize: 9, fontFamily: 'var(--sans)', transition: 'all .15s',
    position: 'relative',
  },
  active: {
    background: 'var(--accent-dim)', color: 'var(--accent-light)',
  },
  sep: {
    width: 28, height: 1, background: 'var(--border)', margin: '6px 0',
  },
};

export default function Sidebar({ tab, onTabChange, streaming }) {
  return (
    <div style={sidebarStyles.sidebar}>
      {tabs.map((t, i) => (
        <div key={t.id}>
          {(i === 4 || i === 9) && <div style={sidebarStyles.sep} />}
          <button
            style={{ ...sidebarStyles.btn, ...(tab === t.id ? sidebarStyles.active : {}),
              color: tab === t.id ? 'var(--accent-light)' : 'rgba(255,255,255,0.7)' }}
            onClick={() => onTabChange(t.id)}
            title={t.label}
          >
            <t.icon size={18} />
            <span>{t.label}</span>
            {t.id === 'chat' && streaming && (
              <span style={{
                position: 'absolute', top: 3, right: 3, width: 7, height: 7,
                borderRadius: '50%', background: 'var(--green)',
                boxShadow: '0 0 6px rgba(34,197,94,.6)',
              }} />
            )}
          </button>
        </div>
      ))}
    </div>
  );
}
