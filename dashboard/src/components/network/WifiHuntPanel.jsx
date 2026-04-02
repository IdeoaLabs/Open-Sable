import { useState, useEffect, useCallback, useRef } from 'react';
import { Wifi, WifiOff, Shield, Key, Radio, Zap, RefreshCw, Brain, Activity,
         Terminal, Lock, Unlock, Target, AlertTriangle, ChevronDown, ChevronUp,
         Check, Plus, Crosshair } from 'lucide-react';

// ── Keyframe CSS ──────────────────────────────────────────────────────────────
const _CSS = `
@keyframes wf-pulse  { 0%,100%{opacity:1} 50%{opacity:.4} }
@keyframes wf-blink  { 0%,100%{opacity:.9} 50%{opacity:.1} }
@keyframes wf-crack  { 0%{background-position:0% 50%} 100%{background-position:200% 50%} }
@keyframes wf-appear { from{opacity:0;transform:translateY(-6px)} to{opacity:1;transform:none} }
@keyframes wf-spin   { to{transform:rotate(360deg)} }
@keyframes wf-sweep  { to{transform:rotate(360deg)} }
`;

// ── Constants ─────────────────────────────────────────────────────────────────
const STATE_META = {
  idle:       { label:'IDLE',       color:'#22c55e', bg:'rgba(34,197,94,.12)',   pulse:false },
  hunting:    { label:'HUNTING',    color:'#3b82f6', bg:'rgba(59,130,246,.12)',  pulse:true  },
  excited:    { label:'EXCITED',    color:'#eab308', bg:'rgba(234,179,8,.12)',   pulse:true  },
  capturing:  { label:'CAPTURING',  color:'#f97316', bg:'rgba(249,115,22,.12)',  pulse:true  },
  deauthing:  { label:'DEAUTHING',  color:'#ef4444', bg:'rgba(239,68,68,.12)',   pulse:true  },
  cracking:   { label:'CRACKING',   color:'#a855f7', bg:'rgba(168,85,247,.12)',  pulse:true  },
  connecting: { label:'CONNECTING', color:'#06b6d4', bg:'rgba(6,182,212,.12)',   pulse:true  },
  happy:      { label:'HAPPY',      color:'#22c55e', bg:'rgba(34,197,94,.20)',   pulse:false },
  bored:      { label:'BORED',      color:'#6b7280', bg:'rgba(107,114,128,.12)', pulse:false },
  lonely:     { label:'LONELY',     color:'#dc2626', bg:'rgba(220,38,38,.12)',   pulse:false },
  smart:      { label:'LEARNED',    color:'#8b5cf6', bg:'rgba(139,92,246,.20)',  pulse:false },
};

const HUNT_STATES = new Set(['hunting','excited','capturing','deauthing','cracking','connecting']);

const EV_ICON  = {
  init:'⚡', scan:'📡', target:'🎯', monitor:'📻', deauth:'💥',
  capture:'🤝', crack:'🔓', connect:'🌐', fail:'❌', happy:'✅', error:'⚠️',
};
const EV_COLOR = {
  init:'#3b82f6', scan:'#22c55e', target:'#eab308', monitor:'#06b6d4',
  deauth:'#ef4444', capture:'#f97316', crack:'#a855f7', connect:'#22c55e',
  fail:'#ef4444', happy:'#22c55e', error:'#f59e0b',
};

const POLL_MS = 3000;

// ── Helpers ───────────────────────────────────────────────────────────────────
function relTime(ts) {
  if (!ts) return '';
  const d = Date.now()/1000 - ts;
  if (d < 60)   return `${Math.floor(d)}s ago`;
  if (d < 3600) return `${Math.floor(d/60)}m ago`;
  return `${Math.floor(d/3600)}h ago`;
}

function signalBars(pct, color='#22c55e') {
  const filled = Math.ceil(Math.max(0,Math.min(100,pct))/20);
  return Array.from({length:5}).map((_,i)=>(
    <span key={i} style={{
      display:'inline-block', width:3, height:4+i*3, borderRadius:1,
      background: i<filled ? color : 'rgba(255,255,255,.13)',
      marginRight:1, verticalAlign:'bottom',
    }}/>
  ));
}

function chBadgeColor(ch) {
  const n = Number(ch);
  if (!n) return '#6b7280';
  if (n <= 14) {
    if (n <= 6)  return '#3b82f6';
    if (n <= 11) return '#8b5cf6';
    return '#06b6d4';
  }
  return '#f97316'; // 5GHz
}

// ── Sub-components ────────────────────────────────────────────────────────────

function Chip({ label, value, color, glow }) {
  return (
    <span style={{
      display:'inline-flex', alignItems:'center', gap:4, padding:'3px 9px', borderRadius:12,
      background: glow ? `${color}22` : 'rgba(255,255,255,.05)',
      fontSize:11, color: color||'rgba(255,255,255,.65)',
      border:`1px solid ${glow ? color+'44' : 'rgba(255,255,255,.08)'}`,
      boxShadow: glow ? `0 0 8px ${color}44` : 'none',
    }}>
      <span style={{color:'rgba(255,255,255,.35)',fontSize:10}}>{label}</span>
      <span style={{fontWeight:700}}>{value}</span>
    </span>
  );
}

function ToolPill({ name, avail }) {
  return (
    <span style={{
      display:'inline-flex', alignItems:'center', gap:5, padding:'3px 10px', borderRadius:20,
      background: avail ? 'rgba(34,197,94,.1)' : 'rgba(239,68,68,.08)',
      border:`1px solid ${avail ? 'rgba(34,197,94,.3)' : 'rgba(239,68,68,.2)'}`,
      fontSize:11, color: avail ? '#22c55e' : '#ef4444',
    }}>
      <span style={{
        width:6,height:6,borderRadius:'50%',
        background: avail ? '#22c55e' : '#ef4444',
        boxShadow: avail ? '0 0 5px #22c55e' : 'none',
      }}/>
      {name}
    </span>
  );
}

// Radar SVG with animated sweep line and network dots
function Radar({ networks, state, monitorActive }) {
  const canvasRef = useRef(null);
  const animRef   = useRef(null);
  const angleRef  = useRef(0);
  const SIZE = 160;
  const CX   = SIZE/2;

  useEffect(()=>{
    const cvs = canvasRef.current;
    if (!cvs) return;
    const ctx = cvs.getContext('2d');
    const active = HUNT_STATES.has(state);

    function draw() {
      ctx.clearRect(0,0,SIZE,SIZE);

      // Background
      ctx.fillStyle='rgba(0,20,10,.85)';
      ctx.beginPath(); ctx.arc(CX,CX,CX,0,Math.PI*2); ctx.fill();

      // Rings
      for (let r = CX*0.25; r <= CX; r += CX*0.25) {
        ctx.beginPath(); ctx.arc(CX,CX,r,0,Math.PI*2);
        ctx.strokeStyle='rgba(34,197,94,.15)'; ctx.lineWidth=1; ctx.stroke();
      }
      // Cross hairs
      ctx.strokeStyle='rgba(34,197,94,.1)'; ctx.lineWidth=1;
      ctx.beginPath(); ctx.moveTo(CX,0); ctx.lineTo(CX,SIZE); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(0,CX); ctx.lineTo(SIZE,CX); ctx.stroke();

      // Sweep trail (gradient arc)
      if (active || monitorActive) {
        const a = angleRef.current * Math.PI/180;
        const grad = ctx.createConicalGradient
          ? null   // fallback below
          : null;
        // Draw sweep as a filled arc sector
        for (let da=0; da<90; da+=3) {
          const alpha = (90-da)/90 * 0.4;
          const a0 = a - (da+3)*Math.PI/180;
          const a1 = a - da*Math.PI/180;
          ctx.beginPath();
          ctx.moveTo(CX,CX);
          ctx.arc(CX,CX,CX-2,a0,a1);
          ctx.closePath();
          ctx.fillStyle=`rgba(34,197,94,${alpha})`;
          ctx.fill();
        }
        // Sweep line
        ctx.beginPath();
        ctx.moveTo(CX,CX);
        ctx.lineTo(CX + (CX-2)*Math.cos(a), CX + (CX-2)*Math.sin(a));
        ctx.strokeStyle='rgba(34,197,94,.9)'; ctx.lineWidth=1.5; ctx.stroke();
      }

      // Network dots (hash BSSID to angle+radius)
      (networks||[]).slice(0,12).forEach(net=>{
        let h=5381;
        const s=(net.bssid||net.ssid||'');
        for(let i=0;i<s.length;i++) h=((h<<5)+h)+s.charCodeAt(i);
        const dotAngle = ((h&0xFFFF)/0xFFFF)*Math.PI*2;
        const dotR     = 20 + (((h>>16)&0xFF)/255)*(CX-24);
        const x = CX + dotR*Math.cos(dotAngle);
        const y = CX + dotR*Math.sin(dotAngle);
        const sigColor = net.signal>60 ? '#22c55e' : net.signal>30 ? '#eab308' : '#ef4444';
        ctx.beginPath(); ctx.arc(x,y,3,0,Math.PI*2);
        ctx.fillStyle=sigColor;
        ctx.shadowColor=sigColor; ctx.shadowBlur=6;
        ctx.fill();
        ctx.shadowBlur=0;
      });

      // Center dot
      ctx.beginPath(); ctx.arc(CX,CX,4,0,Math.PI*2);
      ctx.fillStyle='#22c55e'; ctx.shadowColor='#22c55e'; ctx.shadowBlur=10;
      ctx.fill(); ctx.shadowBlur=0;

      // Outer ring
      ctx.beginPath(); ctx.arc(CX,CX,CX-1,0,Math.PI*2);
      ctx.strokeStyle='rgba(34,197,94,.35)'; ctx.lineWidth=1.5; ctx.stroke();
    }

    function loop() {
      if (active || monitorActive) {
        angleRef.current = (angleRef.current + 1.5) % 360;
      }
      draw();
      animRef.current = requestAnimationFrame(loop);
    }
    animRef.current = requestAnimationFrame(loop);
    return ()=>cancelAnimationFrame(animRef.current);
  }, [networks, state, monitorActive]);

  return (
    <canvas
      ref={canvasRef}
      width={SIZE} height={SIZE}
      style={{ borderRadius:'50%', display:'block' }}
    />
  );
}

// Live epoch countdown
function EpochTimer({ epochStart, epochDuration }) {
  const [left, setLeft] = useState(0);
  useEffect(()=>{
    function tick() {
      const elapsed = Date.now()/1000 - (epochStart||0);
      const rem = Math.max(0, (epochDuration||300) - elapsed);
      setLeft(rem);
    }
    tick();
    const t = setInterval(tick, 500);
    return ()=>clearInterval(t);
  },[epochStart, epochDuration]);

  const pct = epochDuration ? Math.min(100, ((epochDuration-left)/epochDuration)*100) : 0;
  const mins = Math.floor(left/60);
  const secs = Math.floor(left%60);

  return (
    <div style={{width:'100%'}}>
      <div style={{display:'flex',justifyContent:'space-between',fontSize:10,
        color:'rgba(255,255,255,.4)',marginBottom:4}}>
        <span>EPOCH TIMER</span>
        <span style={{fontWeight:600,color:'rgba(255,255,255,.7)'}}>
          {mins}:{secs.toString().padStart(2,'0')}
        </span>
      </div>
      <div style={{height:4,background:'rgba(255,255,255,.08)',borderRadius:2,overflow:'hidden'}}>
        <div style={{
          height:'100%', width:`${pct}%`,
          background:'linear-gradient(90deg,#22c55e,#3b82f6)',
          transition:'width .5s linear',
        }}/>
      </div>
    </div>
  );
}

// AI Brain panel
function AiBrain({ ai }) {
  if (!ai) return null;
  const bmax = 100, emax = 100;
  const bPct = Math.min(100, ((ai.boredom||0)/bmax)*100);
  const ePct = Math.min(100, ((ai.excitement||0)/emax)*100);
  const apStats = ai.ap_stats || {};
  const apEntries = Object.entries(apStats).slice(0,6);

  return (
    <div style={{display:'flex',flexDirection:'column',gap:10}}>
      {/* Chips row */}
      <div style={{display:'flex',flexWrap:'wrap',gap:5}}>
        <Chip label="ε"          value={typeof ai.epsilon==='number' ? ai.epsilon.toFixed(2) : '—'} color="#3b82f6" glow />
        <Chip label="epoch"      value={ai.epoch??0}           color="#a855f7" />
        <Chip label="handshakes" value={ai.total_handshakes??0} color="#f97316" />
        <Chip label="connects"   value={ai.total_connections??0} color="#22c55e" />
        <Chip label="mood"       value={ai.mood||'—'}          color="#eab308" glow />
      </div>

      {/* Bars */}
      <div style={{display:'grid',gridTemplateColumns:'1fr 1fr',gap:8}}>
        {[['BOREDOM', bPct, '#6b7280','#374151'],['EXCITEMENT', ePct,'#eab308','#713f12']].map(([lbl,pct,fg,track])=>(
          <div key={lbl}>
            <div style={{display:'flex',justifyContent:'space-between',fontSize:9,
              color:'rgba(255,255,255,.35)',marginBottom:3}}>
              <span>{lbl}</span><span>{Math.round(pct)}%</span>
            </div>
            <div style={{height:6,background:`rgba(255,255,255,.06)`,borderRadius:3,overflow:'hidden'}}>
              <div style={{height:'100%',width:`${pct}%`,background:fg,
                boxShadow:`0 0 6px ${fg}`,transition:'width .6s ease',borderRadius:3}}/>
            </div>
          </div>
        ))}
      </div>

      {/* AP memory */}
      {apEntries.length > 0 && (
        <div>
          <div style={{fontSize:9,color:'rgba(255,255,255,.3)',letterSpacing:1,
            marginBottom:5,textTransform:'uppercase'}}>AP Memory</div>
          <div style={{display:'flex',flexDirection:'column',gap:3}}>
            {apEntries.map(([bssid,stats])=>{
              const rate = stats.attempts > 0 ? stats.successes/stats.attempts : 0;
              const rPct = rate*100;
              const barC = rPct>60?'#22c55e':rPct>25?'#eab308':'#ef4444';
              return (
                <div key={bssid} style={{display:'flex',alignItems:'center',gap:8}}>
                  <span style={{fontSize:9,color:'rgba(255,255,255,.4)',width:90,
                    overflow:'hidden',textOverflow:'ellipsis',whiteSpace:'nowrap',flexShrink:0}}>
                    {bssid}
                  </span>
                  <div style={{flex:1,height:4,background:'rgba(255,255,255,.06)',borderRadius:2,overflow:'hidden'}}>
                    <div style={{height:'100%',width:`${rPct}%`,background:barC,borderRadius:2}}/>
                  </div>
                  <span style={{fontSize:9,color:barC,width:28,textAlign:'right',flexShrink:0}}>
                    {stats.successes}/{stats.attempts}
                  </span>
                </div>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
}

// Activity event feed
function ActivityFeed({ log }) {
  const entries = [...(log||[])].reverse().slice(0,18);
  if (entries.length===0) return (
    <p style={{fontSize:12,color:'rgba(255,255,255,.25)',textAlign:'center',padding:'10px 0'}}>
      No events yet
    </p>
  );
  return (
    <div style={{display:'flex',flexDirection:'column',gap:3,maxHeight:220,overflowY:'auto'}}>
      {entries.map((ev,i)=>(
        <div key={i} style={{
          display:'flex',alignItems:'flex-start',gap:7,padding:'4px 8px',borderRadius:5,
          background:'rgba(255,255,255,.025)',
          animation: i===0 ? 'wf-appear .3s ease' : 'none',
          borderLeft:`2px solid ${EV_COLOR[ev.type]||'rgba(255,255,255,.15)'}`,
        }}>
          <span style={{fontSize:13,flexShrink:0,lineHeight:'18px'}}>{EV_ICON[ev.type]||'•'}</span>
          <span style={{fontSize:11,color:'rgba(255,255,255,.75)',flex:1,lineHeight:'18px'}}>{ev.msg}</span>
          <span style={{fontSize:9,color:'rgba(255,255,255,.25)',flexShrink:0,lineHeight:'18px',whiteSpace:'nowrap'}}>
            {relTime(ev.ts)}
          </span>
        </div>
      ))}
    </div>
  );
}

// Network row
function NetRow({ net }) {
  const ch = net.channel||'?';
  const chC = chBadgeColor(ch);
  return (
    <div style={{
      display:'flex',alignItems:'center',gap:8,padding:'5px 10px',borderRadius:6,
      background: net.whitelisted ? 'rgba(34,197,94,.05)' : 'rgba(255,255,255,.025)',
      border:`1px solid ${net.whitelisted?'rgba(34,197,94,.2)':'rgba(255,255,255,.05)'}`,
      marginBottom:4,
    }}>
      <div style={{display:'flex',alignItems:'flex-end',gap:1}}>
        {signalBars(net.signal, net.whitelisted?'#22c55e':'#3b82f6')}
      </div>
      <span style={{flex:1,fontSize:12,color:'rgba(255,255,255,.85)',
        overflow:'hidden',textOverflow:'ellipsis',whiteSpace:'nowrap',minWidth:0}}>
        {net.ssid||'<hidden>'}
      </span>
      <span style={{
        fontSize:9,padding:'1px 5px',borderRadius:4,
        background:`${chC}22`,color:chC,
        border:`1px solid ${chC}44`,fontWeight:700,flexShrink:0,
      }}>ch{ch}</span>
      <span style={{fontSize:10,color:'rgba(255,255,255,.3)',flexShrink:0}}>{net.security||'?'}</span>
      {net.whitelisted && (
        <span style={{
          fontSize:9,padding:'1px 6px',borderRadius:8,
          background:'rgba(34,197,94,.15)',color:'#22c55e',
          border:'1px solid rgba(34,197,94,.3)',fontWeight:700,flexShrink:0,
        }}>✓</span>
      )}
      <span style={{fontSize:10,color:'rgba(255,255,255,.3)',minWidth:32,textAlign:'right',flexShrink:0}}>
        {net.signal}%
      </span>
    </div>
  );
}

// Capture row with cracking animation
function CaptureRow({ cap }) {
  const cracking = cap.state==='cracking';
  return (
    <div style={{
      padding:'7px 10px',borderRadius:6,marginBottom:4,
      background: cap.cracked ? 'rgba(139,92,246,.08)' : 'rgba(255,255,255,.03)',
      border:`1px solid ${cap.cracked?'rgba(139,92,246,.25)':cracking?'rgba(168,85,247,.5)':'rgba(255,255,255,.06)'}`,
      position:'relative',overflow:'hidden',
    }}>
      {cracking && (
        <div style={{
          position:'absolute',inset:0,
          background:'linear-gradient(90deg,transparent,rgba(168,85,247,.12),rgba(59,130,246,.12),rgba(168,85,247,.12),transparent)',
          backgroundSize:'200% 100%',
          animation:'wf-crack 1.5s linear infinite',
        }}/>
      )}
      <div style={{display:'flex',alignItems:'center',gap:8,position:'relative'}}>
        <Radio size={12} style={{color:'#f97316',flexShrink:0}}/>
        <span style={{fontSize:12,fontWeight:600,color:'rgba(255,255,255,.85)',flex:1}}>
          {cap.ssid}
        </span>
        {cracking && (
          <span style={{fontSize:9,color:'#a855f7',fontWeight:700,
            animation:'wf-blink 1s ease-in-out infinite'}}>
            CRACKING…
          </span>
        )}
        <span style={{fontSize:10,color:'rgba(255,255,255,.3)'}}>{relTime(cap.captured_at)}</span>
      </div>
      {cap.cracked && (
        <div style={{marginTop:4,fontSize:11,color:'#a855f7',display:'flex',
          alignItems:'center',gap:4,position:'relative'}}>
          <Key size={10}/> cracked — saved to credentials
        </div>
      )}
    </div>
  );
}

function CredRow({ cred }) {
  const isC = cred.method==='cracked';
  return (
    <div style={{
      display:'flex',alignItems:'center',gap:8,padding:'5px 10px',borderRadius:6,
      background:'rgba(255,255,255,.03)',border:'1px solid rgba(255,255,255,.06)',
      marginBottom:4,
    }}>
      <Key size={12} style={{color:'#22c55e',flexShrink:0}}/>
      <span style={{fontSize:12,color:'rgba(255,255,255,.85)',flex:1}}>
        {cred.ssid}
      </span>
      <span style={{
        fontSize:9,padding:'1px 6px',borderRadius:8,fontWeight:700,flexShrink:0,
        background: isC?'rgba(168,85,247,.15)':'rgba(34,197,94,.15)',
        color: isC?'#a855f7':'#22c55e',
        border:`1px solid ${isC?'rgba(168,85,247,.3)':'rgba(34,197,94,.3)'}`,
      }}>{cred.method}</span>
      <span style={{fontSize:10,color:'rgba(255,255,255,.25)',flexShrink:0}}>{relTime(cred.saved_at)}</span>
    </div>
  );
}

// Collapsible section
function Sect({ title, count, accent, open, onToggle, children }) {
  const ac = accent||'rgba(255,255,255,.15)';
  return (
    <div style={{
      background:'rgba(255,255,255,.025)',
      border:`1px solid rgba(255,255,255,.07)`,
      borderRadius:8,overflow:'hidden',
    }}>
      <div
        onClick={onToggle}
        style={{
          display:'flex',alignItems:'center',justifyContent:'space-between',
          padding:'8px 12px',cursor:'pointer',userSelect:'none',
          background:'rgba(255,255,255,.03)',
          borderBottom: open ? '1px solid rgba(255,255,255,.06)' : 'none',
        }}>
        <div style={{display:'flex',alignItems:'center',gap:8}}>
          <div style={{width:3,height:14,borderRadius:2,background:ac,flexShrink:0}}/>
          <span style={{fontSize:11,fontWeight:700,letterSpacing:1.5,color:'rgba(255,255,255,.55)'}}>
            {title}
          </span>
          {count!=null && (
            <span style={{
              fontSize:10,padding:'0px 6px',borderRadius:10,
              background:`${ac}33`,color:ac,border:`1px solid ${ac}55`,fontWeight:700,
            }}>{count}</span>
          )}
        </div>
        {open ? <ChevronUp size={13} style={{color:'rgba(255,255,255,.3)'}}/>
               : <ChevronDown size={13} style={{color:'rgba(255,255,255,.3)'}}/> }
      </div>
      {open && <div style={{padding:'10px 12px'}}>{children}</div>}
    </div>
  );
}

// Whitelist form
function WhitelistForm({ onAdd, disabled }) {
  const [ssid, setSsid] = useState('');
  const submit = () => { const v=ssid.trim(); if(!v) return; onAdd(v); setSsid(''); };
  return (
    <div style={{display:'flex',gap:6,marginTop:8}}>
      <input value={ssid} onChange={e=>setSsid(e.target.value)}
        onKeyDown={e=>e.key==='Enter'&&submit()} placeholder="Add SSID to whitelist…"
        disabled={disabled} style={{
          flex:1,padding:'5px 10px',borderRadius:6,
          background:'rgba(255,255,255,.06)',border:'1px solid rgba(255,255,255,.1)',
          color:'#fff',fontSize:12,outline:'none',
        }}/>
      <button onClick={submit} disabled={disabled||!ssid.trim()} style={{
        padding:'5px 14px',borderRadius:6,border:'none',
        background:ssid.trim()?'rgba(34,197,94,.7)':'rgba(255,255,255,.07)',
        color:'#fff',fontSize:12,cursor:'pointer',fontWeight:600,
      }}>Add</button>
    </div>
  );
}

// Manual credential form
function CredForm({ onAdd, disabled }) {
  const [show, setShow] = useState(false);
  const [ssid, setSsid] = useState('');
  const [bssid, setBssid] = useState('');
  const [pass, setPass] = useState('');
  const submit = () => {
    const v=ssid.trim(), p=pass.trim();
    if(!v||!p) return;
    onAdd({ssid:v,bssid:bssid.trim(),password:p});
    setSsid(''); setBssid(''); setPass(''); setShow(false);
  };
  if (!show) return (
    <button onClick={()=>setShow(true)} style={{
      display:'flex',alignItems:'center',gap:5,marginTop:8,
      padding:'5px 12px',borderRadius:6,border:'1px dashed rgba(255,255,255,.15)',
      background:'transparent',color:'rgba(255,255,255,.4)',fontSize:11,cursor:'pointer',
    }}><Plus size={11}/> Add manual credential</button>
  );
  return (
    <div style={{
      marginTop:8,padding:'10px',borderRadius:6,
      background:'rgba(255,255,255,.04)',border:'1px solid rgba(255,255,255,.1)',
      display:'flex',flexDirection:'column',gap:6,
    }}>
      {[['SSID','ssid',ssid,setSsid],['BSSID (opt)','bssid',bssid,setBssid],['Password','pass',pass,setPass]].map(([lbl,id,val,set])=>(
        <div key={id} style={{display:'flex',gap:6,alignItems:'center'}}>
          <span style={{fontSize:10,color:'rgba(255,255,255,.35)',width:72,flexShrink:0}}>{lbl}</span>
          <input value={val} onChange={e=>set(e.target.value)}
            type={id==='pass'?'password':'text'} placeholder={lbl}
            disabled={disabled} style={{
              flex:1,padding:'4px 8px',borderRadius:5,
              background:'rgba(255,255,255,.06)',border:'1px solid rgba(255,255,255,.1)',
              color:'#fff',fontSize:12,outline:'none',
            }}/>
        </div>
      ))}
      <div style={{display:'flex',gap:6}}>
        <button onClick={submit} disabled={disabled||!ssid.trim()||!pass.trim()} style={{
          flex:1,padding:'5px',borderRadius:5,border:'none',
          background:'rgba(34,197,94,.6)',color:'#fff',fontSize:11,cursor:'pointer',fontWeight:600,
        }}>Save</button>
        <button onClick={()=>setShow(false)} style={{
          padding:'5px 10px',borderRadius:5,border:'1px solid rgba(255,255,255,.1)',
          background:'transparent',color:'rgba(255,255,255,.4)',fontSize:11,cursor:'pointer',
        }}>Cancel</button>
      </div>
    </div>
  );
}

// ── Main Panel ────────────────────────────────────────────────────────────────
export default function WifiHuntPanel({ config }) {
  const [data,    setData]    = useState(null);
  const [error,   setError]   = useState(null);
  const [loading, setLoading] = useState(false);
  const [cmd,     setCmd]     = useState(null);
  const [open,    setOpen]    = useState({
    brain:true, activity:true, networks:true, captures:false, creds:false
  });

  const token  = config?.token || new URLSearchParams(window.location.search).get('token') || '';
  const headers = { 'Content-Type':'application/json', ...(token?{Authorization:`Bearer ${token}`}:{}) };

  const fetch_ = useCallback(async (silent=false)=>{
    if (!silent) setLoading(true);
    try {
      const res = await fetch('/api/wifi', { headers });
      if (!res.ok) {
        const j = await res.json().catch(()=>({}));
        setError(j.error||`HTTP ${res.status}`); setData(null); return;
      }
      setData(await res.json()); setError(null);
    } catch(e) { setError(e.message); }
    finally { setLoading(false); }
  }, [token]);

  useEffect(()=>{
    fetch_();
    const t = setInterval(()=>fetch_(true), POLL_MS);
    return ()=>clearInterval(t);
  }, [fetch_]);

  const post = useCallback(async (body)=>{
    setCmd(body.cmd);
    try {
      await fetch('/api/wifi',{method:'POST',headers,body:JSON.stringify(body)});
      setTimeout(()=>fetch_(true),800);
    } catch(_){}
    setTimeout(()=>setCmd(null),1500);
  },[token]);

  const tog = key => setOpen(p=>({...p,[key]:!p[key]}));

  // ── Derived ────────────────────────────────────────────────────────────────
  const state   = data?.state||'idle';
  const meta    = STATE_META[state]||STATE_META.idle;
  const ai      = data?.ai||{};
  const online  = data?.online??true;
  const monAct  = data?.monitor_active??false;
  const tools   = data?.tools||{};
  const nets    = data?.networks||[];
  const caps    = data?.captures||[];
  const creds   = data?.credentials||[];
  const log     = data?.activity_log||[];
  const wlCount = nets.filter(n=>n.whitelisted).length;
  const hunting = HUNT_STATES.has(state);

  // ── Error / Loading ────────────────────────────────────────────────────────
  if (error) return (
    <div style={{height:'100%',display:'flex',flexDirection:'column',
      alignItems:'center',justifyContent:'center',gap:12,padding:24,fontFamily:'var(--mono,monospace)'}}>
      <WifiOff size={40} style={{color:'rgba(255,255,255,.2)'}}/>
      <p style={{color:'rgba(255,255,255,.5)',fontSize:14,textAlign:'center'}}>{error}</p>
      <p style={{color:'rgba(255,255,255,.3)',fontSize:12,textAlign:'center'}}>
        Enable with <code>WIFI_HUNT_ENABLED=true</code> in profile.env
      </p>
      <button onClick={()=>fetch_()} style={{
        display:'flex',alignItems:'center',gap:5,padding:'6px 14px',borderRadius:6,
        background:'rgba(255,255,255,.1)',border:'none',color:'#fff',fontSize:12,cursor:'pointer',
      }}><RefreshCw size={13}/> Retry</button>
    </div>
  );

  if (loading && !data) return (
    <div style={{height:'100%',display:'flex',alignItems:'center',justifyContent:'center',
      flexDirection:'column',gap:8,fontFamily:'var(--mono,monospace)'}}>
      <Wifi size={36} style={{color:'rgba(255,255,255,.2)',animation:'wf-spin 2s linear infinite'}}/>
      <p style={{color:'rgba(255,255,255,.35)',fontSize:13}}>Connecting to WiFi hunter…</p>
    </div>
  );

  // ── Layout ─────────────────────────────────────────────────────────────────
  return (
    <>
      <style>{_CSS}</style>
      <div style={{
        height:'100%',overflowY:'auto',padding:'14px 18px',
        display:'flex',flexDirection:'column',gap:10,
        fontFamily:'var(--mono,monospace)',
      }}>

        {/* ── Header ── */}
        <div style={{display:'flex',alignItems:'center',gap:8,flexWrap:'wrap'}}>
          <Wifi size={16} style={{color:meta.color,
            animation: hunting||monAct ? 'wf-pulse 1.4s ease-in-out infinite' : 'none'}}/>
          <span style={{fontSize:15,fontWeight:700,color:'#fff',letterSpacing:.5}}>
            WIFI SURVIVAL
          </span>
          <span style={{
            fontSize:10,padding:'2px 9px',borderRadius:10,fontWeight:700,letterSpacing:1,
            background:meta.bg,color:meta.color,
            border:`1px solid ${meta.color}44`,
            animation: meta.pulse ? 'wf-blink 1.2s ease-in-out infinite' : 'none',
            boxShadow: hunting ? `0 0 10px ${meta.color}55` : 'none',
          }}>{meta.label}</span>

          {monAct && (
            <span style={{
              fontSize:9,padding:'2px 7px',borderRadius:8,fontWeight:700,
              background:'rgba(249,115,22,.15)',color:'#f97316',
              border:'1px solid rgba(249,115,22,.35)',
              animation:'wf-blink 1.5s ease-in-out infinite',
            }}>📻 MON</span>
          )}

          <span style={{
            marginLeft:'auto',fontSize:10,padding:'2px 9px',borderRadius:10,
            display:'flex',alignItems:'center',gap:5,fontWeight:700,
            background: online?'rgba(34,197,94,.12)':'rgba(239,68,68,.12)',
            color: online?'#22c55e':'#ef4444',
            border:`1px solid ${online?'rgba(34,197,94,.25)':'rgba(239,68,68,.25)'}`,
          }}>
            <span style={{width:6,height:6,borderRadius:'50%',
              background: online?'#22c55e':'#ef4444',
              boxShadow: online?'0 0 5px #22c55e':'none',display:'inline-block'}}/>
            {online?'ONLINE':'OFFLINE'}
          </span>
        </div>

        {/* ── Radar + Bubble + Timer ── */}
        <div style={{
          display:'grid',gridTemplateColumns:'160px 1fr',gap:12,alignItems:'start',
        }}>
          {/* Radar */}
          <div style={{
            borderRadius:12,overflow:'hidden',
            boxShadow:`0 0 20px rgba(34,197,94,.15), 0 0 1px rgba(34,197,94,.4)`,
          }}>
            <Radar networks={nets} state={state} monitorActive={monAct}/>
          </div>

          {/* Right column */}
          <div style={{display:'flex',flexDirection:'column',gap:8}}>
            {/* Message bubble */}
            <div style={{
              background:'rgba(255,255,255,.04)',
              border:`1px solid ${meta.color}33`,
              borderRadius:10,padding:'10px 14px',
              display:'flex',flexDirection:'column',gap:6,
              boxShadow:`0 0 14px ${meta.color}12`,
            }}>
              <p style={{margin:0,fontSize:13,color:'rgba(255,255,255,.85)',
                fontStyle:'italic',lineHeight:1.5}}>
                "{data?.message||'…'}"
              </p>
              {data?.current_ssid && (
                <span style={{fontSize:11,color:meta.color,display:'flex',alignItems:'center',gap:5}}>
                  <Crosshair size={10}/> {data.current_ssid}
                </span>
              )}
            </div>
            {/* Epoch timer */}
            <EpochTimer epochStart={data?.epoch_start} epochDuration={data?.epoch_duration}/>
          </div>
        </div>

        {/* ── Tool pills ── */}
        <div style={{display:'flex',gap:6,flexWrap:'wrap'}}>
          <ToolPill name="aircrack-ng" avail={!!tools.aircrack}/>
          <ToolPill name="airodump"   avail={!!tools.airodump}/>
          <ToolPill name="aireplay"   avail={!!tools.aireplay}/>
          <ToolPill name="hashcat"    avail={!!tools.hashcat}/>
          {!tools.aircrack && !tools.airodump && (
            <span style={{fontSize:10,color:'rgba(234,179,8,.7)',
              display:'flex',alignItems:'center',gap:4,marginLeft:4}}>
              <AlertTriangle size={11}/> passive scan only
            </span>
          )}
        </div>

        {/* ── AI Brain ── */}
        <Sect title="AI BRAIN" accent="#8b5cf6" open={open.brain} onToggle={()=>tog('brain')}>
          <AiBrain ai={ai}/>
        </Sect>

        {/* ── Activity Feed ── */}
        <Sect title="ACTIVITY" count={log.length} accent="#3b82f6"
          open={open.activity} onToggle={()=>tog('activity')}>
          <ActivityFeed log={log}/>
        </Sect>

        {/* ── Networks ── */}
        <Sect title="NETWORKS" count={`${nets.length} · ${wlCount} ✓`}
          accent="#22c55e" open={open.networks} onToggle={()=>tog('networks')}>
          {nets.length===0
            ? <p style={{fontSize:12,color:'rgba(255,255,255,.25)',textAlign:'center',padding:'8px 0'}}>
                No networks found yet
              </p>
            : [...nets].sort((a,b)=>b.signal-a.signal).map((n,i)=><NetRow key={i} net={n}/>)
          }
          <WhitelistForm onAdd={ssid=>post({cmd:'add_whitelist',ssid})} disabled={!!cmd}/>
        </Sect>

        {/* ── Captures ── */}
        <Sect title="CAPTURES" count={caps.length} accent="#f97316"
          open={open.captures} onToggle={()=>tog('captures')}>
          {caps.length===0
            ? <p style={{fontSize:12,color:'rgba(255,255,255,.25)',textAlign:'center',padding:'8px 0'}}>
                No handshakes captured yet
              </p>
            : [...caps].reverse().map((c,i)=><CaptureRow key={i} cap={c}/>)
          }
        </Sect>

        {/* ── Credentials ── */}
        <Sect title="CREDENTIALS" count={creds.length} accent="#a855f7"
          open={open.creds} onToggle={()=>tog('creds')}>
          {creds.length===0
            ? <p style={{fontSize:12,color:'rgba(255,255,255,.25)',textAlign:'center',padding:'8px 0'}}>
                No credentials saved yet
              </p>
            : creds.map((c,i)=><CredRow key={i} cred={c}/>)
          }
          <CredForm
            onAdd={d=>post({cmd:'add_credential', ...d})}
            disabled={!!cmd}
          />
        </Sect>

        {/* ── Action buttons ── */}
        <div style={{display:'flex',gap:8,flexWrap:'wrap',alignItems:'center'}}>
          <button
            onClick={()=>post({cmd:'force_hunt'})}
            disabled={!!cmd||online}
            title={online?'Already online — hunt triggers offline only':'Force one hunt cycle'}
            style={{
              display:'flex',alignItems:'center',gap:5,padding:'6px 14px',borderRadius:6,
              border:'none',fontWeight:600,fontSize:12,cursor: cmd||online?'not-allowed':'pointer',
              background: cmd||online?'rgba(255,255,255,.07)':'rgba(59,130,246,.7)',
              color: cmd||online?'rgba(255,255,255,.3)':'#fff',
            }}>
            <Zap size={12}/> {cmd==='force_hunt'?'Hunting…':'Force Hunt'}
          </button>

          <button onClick={()=>fetch_()} disabled={loading} style={{
            display:'flex',alignItems:'center',gap:5,padding:'6px 12px',borderRadius:6,
            border:'1px solid rgba(255,255,255,.1)',background:'rgba(255,255,255,.06)',
            color:'#fff',fontSize:12,cursor:'pointer',
          }}>
            <RefreshCw size={12} style={loading?{animation:'wf-spin 1s linear infinite'}:{}}/>
            Refresh
          </button>

          <span style={{fontSize:9,color:'rgba(255,255,255,.2)',marginLeft:'auto'}}>
            ↻ {POLL_MS/1000}s
          </span>
        </div>

      </div>
    </>
  );
}
