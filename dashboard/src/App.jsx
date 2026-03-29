import { useState, useCallback, useRef, useEffect } from 'react';
import { useWebSocket } from './hooks/useWebSocket';
import { useMultiAgent } from './hooks/useMultiAgent';
import Sidebar from './components/layout/Sidebar';
import Topbar from './components/layout/Topbar';
import ChatPanel from './components/chat/ChatPanel';
import ActivityPanel from './components/chat/ActivityPanel';
import TerminalPanel from './components/chat/TerminalPanel';
import StatusPanel from './components/chat/StatusPanel';
import TradingPanel from './components/trading/TradingPanel';
import TaskPanel from './components/assistant/TaskPanel';
import HistoryPanel from './components/assistant/HistoryPanel';
import QRPanel from './components/agent/QRPanel';
import AgentPanel from './components/agent/AgentPanel';
import DevicesPanel from './components/agent/DevicesPanel';
import ThoughtsPanel from './components/agent/ThoughtsPanel';
import BrainPanel from './components/agent/BrainPanel';
import DisplayPanel from './components/agent/DisplayPanel';
import SettingsPanel from './components/settings/SettingsPanel';
import EnvEditorPanel from './components/settings/EnvEditorPanel';
import WifiHuntPanel from './components/network/WifiHuntPanel';
import PermissionDialog from './components/PermissionDialog';
import LoadingOverlay from './components/LoadingOverlay';

const panels = {
  chat: ChatPanel,
  activity: ActivityPanel,
  terminal: TerminalPanel,
  status: StatusPanel,
  trading: TradingPanel,
  tasks: TaskPanel,
  history: HistoryPanel,
  thoughts: ThoughtsPanel,
  brain: BrainPanel,
  qr: QRPanel,
  agent: AgentPanel,
  devices: DevicesPanel,
  display: DisplayPanel,
  settings: SettingsPanel,
  config: EnvEditorPanel,
  wifi: WifiHuntPanel,
};

export default function App() {
  const [tab, setTab] = useState('chat');

  // Multi-agent: create external message handler first, pass to useWebSocket
  const multiAgentRef = useRef(null);
  const onExternalMessage = useCallback((msg) => {
    if (!multiAgentRef.current) return false;
    // Route agents.* messages
    if (msg.type?.startsWith('agents.')) {
      return multiAgentRef.current.handleAgentMessage(msg);
    }
    // Route proxied messages (have _profile field)
    if (msg._profile) {
      return multiAgentRef.current.handleProxiedMessage(msg);
    }
    return false;
  }, []);

  const ws = useWebSocket(onExternalMessage);
  const ma = useMultiAgent(ws.wsRef, ws.connected);

  // Load a session from history and switch to chat (must be after ws is created)
  const handleLoadSession = useCallback((sessionId) => {
    ws.loadSession(sessionId);
    setTab('chat');
  }, [ws.loadSession]);
  multiAgentRef.current = ma;

  // Determine if we're viewing a remote agent or the current (local) agent
  const isLocal = !ma.currentAgent || ma.agents.some(a => a.is_current && a.name === ma.currentAgent);
  const remoteState = !isLocal ? (ma.agentStates[ma.currentAgent] || {}) : null;

  // When switching to a remote agent, fetch its model list
  useEffect(() => {
    if (!isLocal && ma.currentAgent && ws.connected) {
      ws.requestModels(ma.currentAgent);
    }
  }, [isLocal, ma.currentAgent, ws.connected]);

  // Build panel props,  for current agent use live ws data, for remote use proxied state
  // (no useMemo: ws is a new object every render so the memo never saves work)

  // Filter sessions by agent profile
  const localSessions = (ws.sessions || []).filter(s => !s.agent_profile);
  const agentSessions = (ws.sessions || []).filter(s => s.agent_profile === ma.currentAgent);

  let panelProps;
  if (isLocal) {
    panelProps = {
      chat:     { messages: ws.messages, streaming: ws.streaming, onSend: ws.sendMessage, onClear: ws.clearMessages, sessions: localSessions, activeSessionId: ws.activeSessionId, onLoadSession: ws.loadSession, onNewChat: () => ws.loadSession(null), onDeleteSession: ws.deleteSession },
      activity: { activity: ws.activity, onClear: ws.clearActivity },
      terminal: { terminal: ws.terminal, onClear: ws.clearTerminal },
      status:   { stats: ws.stats, sessions: localSessions, model: ws.model, activity: ws.activity },
      trading:  { stats: ws.stats, messages: ws.messages, streaming: ws.streaming, sendMessage: ws.sendMessage },
      tasks:    { streaming: ws.streaming, messages: ws.messages, activity: ws.activity, sendMessage: ws.sendMessage },
      history:  { messages: ws.messages, sessions: localSessions, onLoadSession: handleLoadSession },
      thoughts: { ws: ws.wsRef, thoughts: ws.thoughts, connected: ws.connected },
      brain:    { ws: ws.wsRef, brainData: ws.brainData, connected: ws.connected, profile: null, isLocal: true },
      qr:       {},
      agent:    {},
      devices:  {},
      display:  {},
      settings: { modelGroups: ws.modelGroups, switchModel: ws.switchModel, importGGUF: ws.importGGUF, ws: ws.wsRef, connected: ws.connected, agentProfile: null },
    };
  } else {
    // Remote agent,  interactive view via proxy
    const rs = remoteState || {};
    const sendToRemote = (text) => ma.sendToAgent(ma.currentAgent, text);
    const clearRemoteMsgs = () => {};
    const remoteModelGroups = rs.modelGroups || [];
    // Remote agents use the same local ws session functions since sessions are saved locally
    const remoteActiveSessionId = agentSessions.length > 0 ? (agentSessions[0].session_id || agentSessions[0].id) : '';
    const handleRemoteLoadSession = (sid) => {
      ws.loadSession(sid, (msgs) => {
        ma.setAgentMessages(ma.currentAgent, msgs);
      });
    };
    panelProps = {
      chat:     { messages: rs.messages || [], streaming: rs.streaming || false, onSend: sendToRemote, onClear: clearRemoteMsgs, sessions: agentSessions, activeSessionId: remoteActiveSessionId, onLoadSession: handleRemoteLoadSession, onNewChat: () => { ma.setAgentMessages(ma.currentAgent, []); }, onDeleteSession: ws.deleteSession },
      activity: { activity: rs.activity || [], onClear: () => {} },
      terminal: { terminal: rs.terminal || [], onClear: () => {} },
      status:   { stats: rs.stats || {}, sessions: rs.sessions || [], model: rs.model || '', activity: rs.activity || [] },
      trading:  { stats: rs.stats || {}, messages: rs.messages || [], streaming: rs.streaming || false, sendMessage: sendToRemote },
      tasks:    { streaming: rs.streaming || false, messages: rs.messages || [], activity: rs.activity || [], sendMessage: sendToRemote },
      history:  { messages: rs.messages || [], sessions: agentSessions, onLoadSession: handleLoadSession },
      thoughts: { ws: { current: null }, thoughts: rs.thoughts, connected: rs.connected || false },
      brain:    { ws: ws.wsRef, brainData: rs.brainData || null, connected: ws.connected, profile: ma.currentAgent, isLocal: false },
      qr:       {},
      agent:    {},
      devices:  {},
      display:  {},
      settings: { modelGroups: remoteModelGroups, switchModel: ws.switchModel, importGGUF: ws.importGGUF, ws: ws.wsRef, connected: ws.connected, agentProfile: ma.currentAgent },
    };
  }

  const Panel = panels[tab];

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
      <Topbar
        connected={ws.connected}
        model={isLocal ? ws.model : (remoteState?.model || '')}
        stats={isLocal ? ws.stats : (remoteState?.stats || {})}
        agents={ma.agents}
        currentAgent={ma.currentAgent}
        onAgentSelect={ma.setCurrentAgent}
        modelGroups={isLocal ? ws.modelGroups : (remoteState?.modelGroups || [])}
        activeProvider={isLocal ? ws.activeProvider : (remoteState?.activeProvider || '')}
        onModelSwitch={ws.switchModel}
        onRefreshModels={ws.requestModels}
        agentProfile={isLocal ? null : ma.currentAgent}
      />

      {(isLocal ? ws.streaming : remoteState?.streaming) && (
        <div style={{ height: 2, background: 'var(--accent-dim)', overflow: 'hidden' }}>
          <div style={{
            height: '100%', background: 'var(--accent)',
            animation: 'progress-slide 2s ease-in-out infinite',
          }} />
        </div>
      )}

      <div style={{ display: 'flex', flex: 1, minHeight: 0, overflow: 'hidden' }}>
        <Sidebar tab={tab} onTabChange={setTab} streaming={isLocal ? ws.streaming : (remoteState?.streaming || false)} />

        <div style={{ flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
          {!isLocal && ma.currentAgent && (
            <div style={{
              padding: '4px 16px', fontSize: 11, fontWeight: 600,
              background: 'var(--accent-dim)', color: 'var(--accent-light)',
              borderBottom: '1px solid var(--border)', textTransform: 'uppercase',
              letterSpacing: '0.5px', display: 'flex', alignItems: 'center', gap: 6,
            }}>
              <span style={{ width: 6, height: 6, borderRadius: '50%', background: 'var(--green)' }} />
              Viewing: {ma.currentAgent}
            </div>
          )}
          <div style={{ flex: 1, display: 'flex', overflow: 'hidden' }}>
            <Panel key={ma.currentAgent || '_local'} {...(panelProps[tab] || {})} />
          </div>
        </div>
      </div>

      <PermissionDialog pending={ws.pendingPermission} onRespond={ws.respondPermission} />
      <LoadingOverlay connected={ws.connected} />
    </div>
  );
}
