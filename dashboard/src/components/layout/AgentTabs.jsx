import { Users } from 'lucide-react';

const s = {
  bar: {
    display: 'flex', alignItems: 'center', gap: 2,
    padding: '0 4px', height: '100%',
  },
  tab: {
    display: 'flex', alignItems: 'center', gap: 6,
    padding: '4px 12px', borderRadius: 6,
    border: 'none', background: 'transparent',
    color: 'var(--text-secondary)', cursor: 'pointer',
    fontSize: 12, fontWeight: 500, fontFamily: 'var(--sans)',
    transition: 'all .15s', whiteSpace: 'nowrap',
    position: 'relative',
  },
  active: {
    background: 'var(--accent-dim)',
    color: 'var(--accent-light)',
    fontWeight: 600,
  },
  dot: (running, isCurrent) => ({
    width: 7, height: 7, borderRadius: '50%',
    background: running
      ? 'var(--green)'
      : 'var(--text-muted)',
    boxShadow: running ? '0 0 6px rgba(34,197,94,.5)' : 'none',
    flexShrink: 0,
  }),
  name: {
    textTransform: 'capitalize',
  },
  icon: {
    display: 'flex', alignItems: 'center', marginRight: 4,
    color: 'var(--text-muted)', fontSize: 13,
  },
  sep: {
    width: 1, height: 20, background: 'var(--border)',
    margin: '0 6px', flexShrink: 0,
  },
};

export default function AgentTabs({ agents, currentAgent, onSelect }) {
  if (!agents || agents.length <= 1) return null;

  return (
    <>
      <div style={s.sep} />
      <div style={s.icon}>
        <Users size={14} />
      </div>
      <div style={s.bar}>
        {agents.map(agent => (
          <button
            key={agent.name}
            style={{
              ...s.tab,
              ...(currentAgent === agent.name ? s.active : {}),
            }}
            onClick={() => onSelect(agent.name)}
            title={`${agent.name},  ${agent.running ? 'Online' : 'Offline'}`}
          >
            <span style={s.dot(agent.running, agent.is_current)} />
            <span style={s.name}>{agent.name}</span>
          </button>
        ))}
      </div>
    </>
  );
}
