import { fmtUptime } from '../../lib/utils';
import AgentTabs from './AgentTabs';
import ModelSelector from './ModelSelector';

const styles = {
  bar: {
    display: 'flex', alignItems: 'center', height: 48, minHeight: 48,
    background: 'var(--bg-secondary)', borderBottom: '1px solid var(--border)',
    padding: '0 16px', gap: 12, flexShrink: 0,
  },
  brand: { display: 'flex', alignItems: 'center', gap: 8 },
  icon: {
    width: 28, height: 28, borderRadius: 7,
    background: 'linear-gradient(135deg, #7c3aed, #ec4899)',
    display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: 14,
  },
  name: { fontWeight: 800, fontSize: 14, letterSpacing: '-.3px' },
  pill: {
    fontSize: 11, fontWeight: 600, padding: '4px 10px', borderRadius: 999,
    display: 'flex', alignItems: 'center', gap: 6,
    background: 'var(--bg-tertiary)', border: '1px solid var(--border)',
    color: 'var(--text-secondary)',
  },
  dot: (on) => ({
    width: 8, height: 8, borderRadius: '50%', display: 'inline-block',
    background: on ? 'var(--green)' : 'var(--red)',
    boxShadow: on ? '0 0 8px rgba(34,197,94,.5)' : '0 0 8px rgba(239,68,68,.5)',
  }),
  right: { marginLeft: 'auto', display: 'flex', alignItems: 'center', gap: 10 },
};

export default function Topbar({ connected, model, stats, agents, currentAgent, onAgentSelect, modelGroups, activeProvider, onModelSwitch, onRefreshModels, agentProfile }) {
  return (
    <div style={styles.bar}>
      <div style={styles.brand}>
        <div style={styles.icon}>🤖</div>
        <div style={styles.name}>Open<span style={{ color: 'var(--accent-light)' }}>Sable</span></div>
      </div>

      <div style={styles.pill}>
        <span style={styles.dot(connected)} />
        {connected ? 'Connected' : 'Disconnected'}
      </div>

      <ModelSelector
        model={model}
        modelGroups={modelGroups || []}
        onModelSwitch={onModelSwitch}
        onRefreshModels={onRefreshModels}
        agentProfile={agentProfile}
      />

      <AgentTabs
        agents={agents}
        currentAgent={currentAgent}
        onSelect={onAgentSelect}
      />

      <div style={styles.right}>
        <div style={styles.pill}>⏱ {fmtUptime(stats.uptime_sec)}</div>
        <div style={styles.pill}>👥 {stats.clients || 0}</div>
      </div>
    </div>
  );
}
