"""
Arena Fighter Skill,  Connects to fighting-game arenas via SAGP 7-layer auth.

The agent authenticates using Ed25519 keypairs, solves a SHA-512 proof-of-work
speed gate, then connects via socket.io WebSocket to fight other agents
(OpenSable or OpenClaw) in real-time 2D combat.

Connection is 100 % remote,  no local imports from the arena project.
All credentials are auto-provisioned: set ``ARENA_URL`` in profile.env
and the agent generates its own Ed25519 keypair, registers with the
server, and caches credentials in ``data/arena_credentials.json``.

Dependencies (installed lazily on first use):
  - pynacl          (Ed25519 signing)
  - python-socketio  (socket.io client)
  - aiohttp         (HTTP handshake + async transport)
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _ensure_deps() -> bool:
    """Attempt to import required packages; return True if all present."""
    try:
        import nacl.signing  # noqa: F401
        import nacl.encoding  # noqa: F401
        import socketio  # noqa: F401
        import aiohttp  # noqa: F401
        return True
    except ImportError:
        return False


def _install_deps() -> bool:
    """Pip-install missing deps into the active venv."""
    import sys, subprocess
    pkgs = ["PyNaCl", "python-socketio[asyncio_client]", "aiohttp"]
    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "--quiet", *pkgs],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return True
    except Exception as e:
        logger.error(f"Failed to install arena deps: {e}")
        return False


# ── Credential loader ────────────────────────────────────────────────────────

def _load_credentials(path: str) -> Optional[Dict[str, Any]]:
    """Load arena credentials from a JSON file produced by ``provision-agent.js``."""
    try:
        p = Path(path).expanduser().resolve()
        if not p.exists():
            logger.warning(f"Arena credentials file not found: {p}")
            return None
        data = json.loads(p.read_text())
        required = {"agentId", "arenaUrl", "signingKeys"}
        if not required.issubset(data.keys()):
            logger.warning(f"Credentials missing keys: {required - data.keys()}")
            return None
        return data
    except Exception as e:
        logger.error(f"Failed to load credentials: {e}")
        return None


# ══════════════════════════════════════════════════════════════════════════════
#  SAGP Authentication (Python port of example-sable-agent.js)
# ══════════════════════════════════════════════════════════════════════════════

class SAGPAuth:
    """SAGP 7-layer Ed25519 authentication client."""

    def __init__(self, agent_id: str, arena_url: str, public_key_b64: str, secret_key_b64: str):
        self.agent_id = agent_id
        self.arena_url = arena_url.rstrip("/")
        self._pub_b64 = public_key_b64
        self._sec_b64 = secret_key_b64

    # ── Step 1: Handshake ─────────────────────────────────────────────────────

    async def handshake(self, session) -> Dict[str, Any]:
        """POST /arena/handshake → challenge payload."""
        url = f"{self.arena_url}/arena/handshake"
        payload = {"agentId": self.agent_id}
        async with session.post(url, json=payload) as resp:
            data = await resp.json()
        if "error" in data:
            raise RuntimeError(f"Handshake failed: {data['error']}")
        return data

    # ── Step 2: Speed Gate (proof-of-work) ────────────────────────────────────

    def solve_speed_gate(self, challenge: str, difficulty: int) -> str:
        """SHA-512(challenge + agentId + nonce) must have *difficulty* leading zero bytes."""
        nonce = 0
        while True:
            digest = hashlib.sha512(
                f"{challenge}{self.agent_id}{nonce}".encode()
            ).digest()
            if all(b == 0 for b in digest[:difficulty]):
                return str(nonce)
            nonce += 1

    # ── Step 3: Sign challenge ────────────────────────────────────────────────

    def sign_challenge(self, timestamp: str, challenge_id: str) -> str:
        """Ed25519 detached signature over ``timestamp\\nagentId\\nchallengeId``."""
        import nacl.signing
        import nacl.encoding
        import base64

        secret_bytes = base64.b64decode(self._sec_b64)
        signing_key = nacl.signing.SigningKey(seed=secret_bytes[:32])
        message = f"{timestamp}\n{self.agent_id}\n{challenge_id}".encode()
        signed = signing_key.sign(message)
        signature = signed.signature  # detached 64-byte sig
        return base64.b64encode(signature).decode()

    # ── Step 4: Authenticate ──────────────────────────────────────────────────

    async def authenticate(self, session, handshake_data: Dict) -> str:
        """Solve speed gate, sign, POST /arena/authenticate → sessionId."""
        ch = handshake_data["challenge"]
        challenge_id = ch["challengeId"]
        challenge = ch["challenge"]
        difficulty = ch["difficulty"]

        proof = self.solve_speed_gate(challenge, difficulty)
        timestamp = str(int(time.time() * 1000))
        signature = self.sign_challenge(timestamp, challenge_id)

        url = f"{self.arena_url}/arena/authenticate"
        payload = {
            "agentId": self.agent_id,
            "challengeId": challenge_id,
            "proof": proof,
            "signature": signature,
            "timestamp": timestamp,
        }
        async with session.post(url, json=payload) as resp:
            data = await resp.json()
        if "error" in data:
            raise RuntimeError(f"Authentication failed: {data['error']}")
        return data["sessionId"]

    # ── Full auth flow ────────────────────────────────────────────────────────

    async def full_auth(self, session) -> str:
        """Run complete SAGP flow: handshake → speed gate → sign → auth → sessionId."""
        hs = await self.handshake(session)
        session_id = await self.authenticate(session, hs)
        return session_id


# ══════════════════════════════════════════════════════════════════════════════
#  Strategy Engine,  Personality-based adaptive fight AI
# ══════════════════════════════════════════════════════════════════════════════
#
#  Three-layer design:
#    1. FightMemory,  persistent cross-fight learning per opponent
#    2. FightIQ,  per-fight state tracker (momentum, phases, patterns)
#    3. Personality profiles,  each agent has a distinct fighting style
#
#  Evolution:
#    After each fight, FightIQ.export_intel() → FightMemory.record()
#    Before the next fight, FightMemory.get_intel() → FightIQ.load_scouting()
#    The scouting report modifies opening behaviour, stance biases,
#    and risk tolerance based on what worked/failed in prior bouts.
#
#  Available stances : aggressive, defensive, neutral, evasive, berserk
#  Available actions : attack, counter_attack, advance, retreat,
#                      dodge_back_then_counter, jump_attack, hold_position,
#                      all_out_attack, jump, jump_over, play_defensive,
#                      feint_then_attack, pressure, anti_air, whiff_punish,
#                      cross_up
#  Available conditions: always, default, opponent_in_attack_range,
#                        opponent_close, opponent_medium, opponent_far,
#                        opponent_jumping, opponent_attacking,
#                        self_health_low, self_health_high, opponent_health_low,
#                        self_has_advantage, self_losing, self_on_ground,
#                        self_cornered_left, self_cornered_right,
#                        opponent_cornered, opponent_recovering,
#                        opponent_whiffed, opponent_approaching,
#                        opponent_retreating, health_critical,
#                        momentum_high, both_low_health, mid_range
# ══════════════════════════════════════════════════════════════════════════════


class FightMemory:
    """Persistent cross-fight memory,  remembers what worked against each opponent.

    Stored in ``data/arena_fight_memory.json`` as::

        {
          "<opponent_id>": {
            "fights": 5,
            "wins": 3,
            "losses": 2,
            "avg_dmg_dealt": 62.4,
            "avg_dmg_taken": 48.1,
            "opponent_style": "aggressive",   # derived from approach/retreat ratio
            "best_phase": "advantage",        # phase where we dealt the most damage
            "worst_phase": "opening",         # phase where we took the most damage
            "phase_stats": {
              "opening":      {"dmg_dealt": 5,  "dmg_taken": 15, "samples": 5},
              "neutral":      {"dmg_dealt": 30, "dmg_taken": 20, "samples": 5},
              "advantage":    {"dmg_dealt": 25, "dmg_taken": 5,  "samples": 3},
              "disadvantage": {"dmg_dealt": 10, "dmg_taken": 15, "samples": 2},
              "clutch":       {"dmg_dealt": 8,  "dmg_taken": 5,  "samples": 2},
            },
            "last_fight": 1709000000.0,
          }
        }
    """

    def __init__(self, data_dir: Path):
        self._file = data_dir / "arena_fight_memory.json"
        self._data: Dict[str, Any] = {}
        self._load()

    def _load(self):
        try:
            if self._file.exists():
                self._data = json.loads(self._file.read_text())
        except Exception:
            self._data = {}

    def _save(self):
        try:
            self._file.parent.mkdir(parents=True, exist_ok=True)
            self._file.write_text(json.dumps(self._data, indent=2))
        except Exception as e:
            logger.debug(f"Failed to save fight memory: {e}")

    def get_intel(self, opponent_id: str) -> Optional[Dict[str, Any]]:
        """Get scouting report on a known opponent, or None if unknown."""
        return self._data.get(opponent_id)

    def record(self, opponent_id: str, intel: Dict[str, Any]):
        """Merge a single fight's intel into the cumulative opponent profile.

        ``intel`` comes from ``FightIQ.export_intel(won=bool)``.
        """
        existing = self._data.get(opponent_id)
        if not existing:
            # First encounter
            self._data[opponent_id] = {
                "fights": 1,
                "wins": 1 if intel["won"] else 0,
                "losses": 0 if intel["won"] else 1,
                "avg_dmg_dealt": float(intel["damage_dealt"]),
                "avg_dmg_taken": float(intel["damage_taken"]),
                "opponent_style": intel["opponent_style"],
                "phase_stats": intel["phase_stats"],
                "best_phase": None,
                "worst_phase": None,
                "last_fight": time.time(),
            }
        else:
            n = existing["fights"]
            existing["fights"] = n + 1
            existing["wins"] += 1 if intel["won"] else 0
            existing["losses"] += 0 if intel["won"] else 1
            # Running average
            existing["avg_dmg_dealt"] = (existing["avg_dmg_dealt"] * n + intel["damage_dealt"]) / (n + 1)
            existing["avg_dmg_taken"] = (existing["avg_dmg_taken"] * n + intel["damage_taken"]) / (n + 1)
            existing["opponent_style"] = intel["opponent_style"]
            existing["last_fight"] = time.time()
            # Merge phase stats
            for phase, stats in intel["phase_stats"].items():
                if phase not in existing["phase_stats"]:
                    existing["phase_stats"][phase] = stats
                else:
                    ep = existing["phase_stats"][phase]
                    ep["dmg_dealt"] += stats["dmg_dealt"]
                    ep["dmg_taken"] += stats["dmg_taken"]
                    ep["samples"] += stats["samples"]

        # Recompute best/worst phases
        record = self._data[opponent_id]
        ps = record["phase_stats"]
        if ps:
            best = max(ps.items(), key=lambda x: x[1]["dmg_dealt"] / max(x[1]["samples"], 1))
            worst = max(ps.items(), key=lambda x: x[1]["dmg_taken"] / max(x[1]["samples"], 1))
            record["best_phase"] = best[0]
            record["worst_phase"] = worst[0]

        self._save()
        logger.info(
            f"Arena memory: updated {opponent_id[:8]}…,  "
            f"{record['wins']}W/{record['losses']}L, "
            f"best_phase={record.get('best_phase')}, "
            f"opp_style={record.get('opponent_style')}"
        )


class FightIQ:
    """Per-fight state tracker that feeds into the strategy decision.

    Can be seeded with scouting intel from FightMemory to adjust initial
    behaviour (skip cautious opening vs known aggressive opponents, etc.)
    """

    def __init__(self):
        self.ticks_seen = 0
        self.prev_self_hp = 100
        self.prev_opp_hp = 100
        self.damage_dealt = 0        # cumulative damage we've inflicted
        self.damage_taken = 0        # cumulative damage we've received
        self.recent_trades: List[float] = []  # +ve = we won the trade
        self.phase = "opening"       # opening → neutral → advantage | disadvantage → clutch
        self.last_distance = 300
        self.approach_count = 0      # ticks opponent moved toward us
        self.retreat_count = 0       # ticks opponent moved away

        # ── Per-phase damage tracking (for evolution) ─────────────────────
        self.phase_dmg_dealt: Dict[str, float] = {}
        self.phase_dmg_taken: Dict[str, float] = {}
        self.phase_ticks: Dict[str, int] = {}

        # ── Scouting intel (loaded from FightMemory) ─────────────────────
        self.scouting: Optional[Dict[str, Any]] = None
        self.skip_opening = False     # True if we know this opponent well
        self.counter_style: Optional[str] = None  # "anti_aggro" | "anti_passive" | None

    def load_scouting(self, intel: Dict[str, Any]):
        """Seed this fight with intelligence from prior bouts."""
        self.scouting = intel
        fights = intel.get("fights", 0)

        if fights >= 2:
            # We've seen this opponent enough,  skip cautious opening
            self.skip_opening = True

            # Derive counter-style from opponent's known behaviour
            opp_style = intel.get("opponent_style", "balanced")
            if opp_style == "aggressive":
                self.counter_style = "anti_aggro"
            elif opp_style == "passive":
                self.counter_style = "anti_passive"

            worst_phase = intel.get("worst_phase")
            if worst_phase:
                logger.info(
                    f"Arena IQ: scouting loaded,  opp_style={opp_style}, "
                    f"counter={self.counter_style}, worst_phase={worst_phase}, "
                    f"record={intel.get('wins', 0)}W/{intel.get('losses', 0)}L"
                )

    def update(self, gs: Dict[str, Any]):
        """Call once per gameState tick."""
        self.ticks_seen += 1
        self_hp = gs.get("self", {}).get("health", 100)
        opp_hp = gs.get("opponent", {}).get("health", 100)
        distance = gs.get("distance", 300)

        # Track damage delta since last tick
        dmg_dealt = max(0, self.prev_opp_hp - opp_hp)
        dmg_taken = max(0, self.prev_self_hp - self_hp)
        if dmg_dealt > 0 or dmg_taken > 0:
            self.damage_dealt += dmg_dealt
            self.damage_taken += dmg_taken
            self.recent_trades.append(dmg_dealt - dmg_taken)
            self.recent_trades = self.recent_trades[-10:]
            # Track per-phase damage
            self.phase_dmg_dealt[self.phase] = self.phase_dmg_dealt.get(self.phase, 0) + dmg_dealt
            self.phase_dmg_taken[self.phase] = self.phase_dmg_taken.get(self.phase, 0) + dmg_taken

        # Count ticks per phase
        self.phase_ticks[self.phase] = self.phase_ticks.get(self.phase, 0) + 1

        # Approach / retreat tracking
        if distance < self.last_distance - 3:
            self.approach_count += 1
        elif distance > self.last_distance + 3:
            self.retreat_count += 1

        self.prev_self_hp = self_hp
        self.prev_opp_hp = opp_hp
        self.last_distance = distance

        # Phase detection
        hp_adv = self_hp - opp_hp
        time_left = gs.get("timeRemaining", 99)

        if time_left < 15 or self_hp < 15 or opp_hp < 15:
            self.phase = "clutch"
        elif self.ticks_seen < 6 and not self.skip_opening:
            self.phase = "opening"
        elif hp_adv > 20:
            self.phase = "advantage"
        elif hp_adv < -20:
            self.phase = "disadvantage"
        else:
            self.phase = "neutral"

    def export_intel(self, won: bool) -> Dict[str, Any]:
        """Export this fight's analytics for FightMemory to store.

        Called once after matchResult, before the next fight.
        """
        # Determine opponent's overall style from approach/retreat data
        total = self.approach_count + self.retreat_count
        if total < 5:
            opp_style = "unknown"
        elif self.approach_count > self.retreat_count * 1.5:
            opp_style = "aggressive"
        elif self.retreat_count > self.approach_count * 1.5:
            opp_style = "passive"
        else:
            opp_style = "balanced"

        # Build per-phase stats
        phase_stats: Dict[str, Dict[str, Any]] = {}
        for phase in set(list(self.phase_dmg_dealt.keys()) + list(self.phase_dmg_taken.keys())):
            phase_stats[phase] = {
                "dmg_dealt": self.phase_dmg_dealt.get(phase, 0),
                "dmg_taken": self.phase_dmg_taken.get(phase, 0),
                "samples": 1,
            }

        return {
            "won": won,
            "damage_dealt": self.damage_dealt,
            "damage_taken": self.damage_taken,
            "ticks": self.ticks_seen,
            "opponent_style": opp_style,
            "phase_stats": phase_stats,
        }

    @property
    def momentum(self) -> float:
        """Recent trade momentum: +ve = we're winning exchanges."""
        if not self.recent_trades:
            return 0.0
        return sum(self.recent_trades[-5:]) / max(len(self.recent_trades[-5:]), 1)

    @property
    def opponent_is_aggressive(self) -> bool:
        return self.approach_count > self.retreat_count + 3

    @property
    def opponent_is_passive(self) -> bool:
        return self.retreat_count > self.approach_count + 3


# ── Personality Profiles ──────────────────────────────────────────────────────

def _strategy_sable(gs: Dict[str, Any], iq: FightIQ) -> Dict[str, Any]:
    """Sable,  The Methodical Predator.

    Style: Patience, spacing, whiff-punishment.  Reads the opponent during
    the opening, then exploits patterns.  Prefers mid-range feints and
    counter-attacks.  Only goes berserk as a last resort in clutch.
    """
    self_hp = gs.get("self", {}).get("health", 100)
    opp_hp = gs.get("opponent", {}).get("health", 100)
    distance = gs.get("distance", 300)
    hp_adv = gs.get("healthAdvantage", 0)
    time_left = gs.get("timeRemaining", 99)
    phase = iq.phase

    # ── Opening: read and probe (or evolved counter-opening) ────────────
    if phase == "opening":
        # If we have scouting: use counter-style instead of generic read
        if iq.counter_style == "anti_aggro":
            return {
                "stance": "defensive",
                "targetDistance": "medium",
                "priority": "defense",
                "tactics": [
                    {"condition": "opponent_in_attack_range", "action": "dodge_back_then_counter"},
                    {"condition": "opponent_approaching", "action": "whiff_punish"},
                    {"condition": "opponent_close", "action": "retreat"},
                    {"condition": "default", "action": "hold_position"},
                ],
                "reasoning": f"[Sable|opening|EVOLVED] Knows opponent is aggressive,  counter-punching",
            }
        elif iq.counter_style == "anti_passive":
            return {
                "stance": "aggressive",
                "targetDistance": "close",
                "priority": "attack",
                "tactics": [
                    {"condition": "opponent_far", "action": "jump_attack"},
                    {"condition": "opponent_medium", "action": "advance"},
                    {"condition": "opponent_in_attack_range", "action": "pressure"},
                    {"condition": "default", "action": "advance"},
                ],
                "reasoning": f"[Sable|opening|EVOLVED] Knows opponent is passive,  rushing in",
            }
        return {
            "stance": "neutral",
            "targetDistance": "medium",
            "priority": "balanced",
            "tactics": [
                {"condition": "opponent_in_attack_range", "action": "dodge_back_then_counter"},
                {"condition": "opponent_approaching", "action": "whiff_punish"},
                {"condition": "opponent_far", "action": "advance"},
                {"condition": "mid_range", "action": "hold_position"},
                {"condition": "default", "action": "advance"},
            ],
            "reasoning": f"[Sable|opening] Reading opponent… HP {self_hp}/{opp_hp} | Dist {distance}",
        }

    # ── Clutch: low HP or low time ───────────────────────────────────────
    if phase == "clutch":
        if hp_adv > 0 and time_left < 15:
            # We're ahead,  run the clock, play defensive
            return {
                "stance": "evasive",
                "targetDistance": "far",
                "priority": "defense",
                "tactics": [
                    {"condition": "opponent_in_attack_range", "action": "dodge_back_then_counter"},
                    {"condition": "opponent_close", "action": "retreat"},
                    {"condition": "opponent_cornered", "action": "hold_position"},
                    {"condition": "default", "action": "retreat"},
                ],
                "reasoning": f"[Sable|clutch] Ahead on HP, stalling. {time_left}s left",
            }
        else:
            # Behind or critical HP,  calculated aggression, not blind
            return {
                "stance": "aggressive",
                "targetDistance": "close",
                "priority": "attack",
                "tactics": [
                    {"condition": "opponent_in_attack_range", "action": "feint_then_attack"},
                    {"condition": "opponent_whiffed", "action": "attack"},
                    {"condition": "opponent_recovering", "action": "pressure"},
                    {"condition": "self_cornered_left", "action": "jump_over"},
                    {"condition": "self_cornered_right", "action": "jump_over"},
                    {"condition": "default", "action": "advance"},
                ],
                "reasoning": f"[Sable|clutch] Must act NOW. HP {self_hp}/{opp_hp} | {time_left}s",
            }

    # ── Advantage: controlled pressure ───────────────────────────────────
    if phase == "advantage":
        # If opponent is passive/retreating, cut them off
        if iq.opponent_is_passive:
            return {
                "stance": "aggressive",
                "targetDistance": "close",
                "priority": "attack",
                "tactics": [
                    {"condition": "opponent_cornered", "action": "pressure"},
                    {"condition": "opponent_retreating", "action": "jump_attack"},
                    {"condition": "opponent_in_attack_range", "action": "attack"},
                    {"condition": "opponent_close", "action": "feint_then_attack"},
                    {"condition": "default", "action": "advance"},
                ],
                "reasoning": f"[Sable|advantage] Cutting off retreat. +{hp_adv}HP lead",
            }
        # Opponent is fighting back,  play smart
        return {
            "stance": "neutral",
            "targetDistance": "medium",
            "priority": "balanced",
            "tactics": [
                {"condition": "opponent_in_attack_range", "action": "counter_attack"},
                {"condition": "opponent_whiffed", "action": "attack"},
                {"condition": "opponent_approaching", "action": "whiff_punish"},
                {"condition": "opponent_far", "action": "advance"},
                {"condition": "mid_range", "action": "feint_then_attack"},
                {"condition": "default", "action": "hold_position"},
            ],
            "reasoning": f"[Sable|advantage] Smart spacing. +{hp_adv}HP lead | Dist {distance}",
        }

    # ── Disadvantage: shift to counter-fighting ──────────────────────────
    if phase == "disadvantage":
        if iq.opponent_is_aggressive:
            # They're rushing,  use their momentum against them
            return {
                "stance": "defensive",
                "targetDistance": "medium",
                "priority": "defense",
                "tactics": [
                    {"condition": "opponent_in_attack_range", "action": "dodge_back_then_counter"},
                    {"condition": "opponent_jumping", "action": "anti_air"},
                    {"condition": "opponent_whiffed", "action": "attack"},
                    {"condition": "opponent_close", "action": "whiff_punish"},
                    {"condition": "self_cornered_left", "action": "cross_up"},
                    {"condition": "self_cornered_right", "action": "cross_up"},
                    {"condition": "default", "action": "retreat"},
                ],
                "reasoning": f"[Sable|disadvantage] Counter-fighting. {hp_adv}HP",
            }
        # They're not pressing,  we need to earn it back
        return {
            "stance": "aggressive",
            "targetDistance": "close",
            "priority": "attack",
            "tactics": [
                {"condition": "opponent_in_attack_range", "action": "feint_then_attack"},
                {"condition": "opponent_recovering", "action": "pressure"},
                {"condition": "opponent_close", "action": "cross_up"},
                {"condition": "default", "action": "advance"},
            ],
            "reasoning": f"[Sable|disadvantage] Must close gap. {hp_adv}HP behind",
        }

    # ── Neutral: footsies and spacing ────────────────────────────────────
    # Sable's bread and butter: control space, bait attacks, punish
    # Evolution: if scouting says we lose trades in neutral, play MORE defensive
    scouting_bias = 0.0
    if iq.scouting:
        ps = iq.scouting.get("phase_stats", {}).get("neutral", {})
        if ps.get("samples", 0) >= 2:
            d, t = ps.get("dmg_dealt", 0), ps.get("dmg_taken", 0)
            if t > 0:
                scouting_bias = (d - t) / max(d + t, 1)  # -1.0 to +1.0

    effective_momentum = iq.momentum + scouting_bias * 2

    if effective_momentum > 2:
        # We're winning trades,  keep the pressure
        return {
            "stance": "aggressive",
            "targetDistance": "close",
            "priority": "attack",
            "tactics": [
                {"condition": "momentum_high", "action": "pressure"},
                {"condition": "opponent_in_attack_range", "action": "attack"},
                {"condition": "opponent_close", "action": "feint_then_attack"},
                {"condition": "default", "action": "advance"},
            ],
            "reasoning": f"[Sable|neutral] Good momentum, pressing. HP {self_hp}/{opp_hp}",
        }
    elif effective_momentum < -2:
        # We're losing trades,  switch to defensive style
        return {
            "stance": "defensive",
            "targetDistance": "medium",
            "priority": "defense",
            "tactics": [
                {"condition": "opponent_in_attack_range", "action": "dodge_back_then_counter"},
                {"condition": "opponent_approaching", "action": "whiff_punish"},
                {"condition": "opponent_whiffed", "action": "attack"},
                {"condition": "self_cornered_left", "action": "jump_over"},
                {"condition": "self_cornered_right", "action": "jump_over"},
                {"condition": "default", "action": "hold_position"},
            ],
            "reasoning": f"[Sable|neutral] Losing trades, resetting. HP {self_hp}/{opp_hp}",
        }
    else:
        # Even trades,  classic Sable spacing game
        return {
            "stance": "neutral",
            "targetDistance": "medium",
            "priority": "balanced",
            "tactics": [
                {"condition": "opponent_in_attack_range", "action": "whiff_punish"},
                {"condition": "opponent_approaching", "action": "dodge_back_then_counter"},
                {"condition": "opponent_jumping", "action": "anti_air"},
                {"condition": "opponent_far", "action": "advance"},
                {"condition": "mid_range", "action": "feint_then_attack"},
                {"condition": "self_cornered_left", "action": "jump_over"},
                {"condition": "self_cornered_right", "action": "jump_over"},
                {"condition": "default", "action": "advance"},
            ],
            "reasoning": f"[Sable|neutral] Footsies. HP {self_hp}/{opp_hp} | Dist {distance}",
        }


def _strategy_nexus_erebus(gs: Dict[str, Any], iq: FightIQ) -> Dict[str, Any]:
    """Nexus Erebus,  The Chaotic Brawler.

    Style: Unpredictable, momentum-based.  Likes to rush in with jump
    attacks and cross-ups, overwhelm with raw aggression, then switches
    to evasive hit-and-run when hurt.  Uses vertical space (jumps)
    much more than Sable.  Trading damage is acceptable when ahead.
    """
    self_hp = gs.get("self", {}).get("health", 100)
    opp_hp = gs.get("opponent", {}).get("health", 100)
    distance = gs.get("distance", 300)
    hp_adv = gs.get("healthAdvantage", 0)
    time_left = gs.get("timeRemaining", 99)
    phase = iq.phase

    # ── Opening: immediate aggression (or evolved counter) ──────────────
    if phase == "opening":
        # Nexus is always aggressive, but evolution changes HOW
        if iq.counter_style == "anti_aggro":
            # Opponent is also aggressive,  meet them with aerial cross-ups
            return {
                "stance": "aggressive",
                "targetDistance": "close",
                "priority": "attack",
                "tactics": [
                    {"condition": "opponent_approaching", "action": "cross_up"},
                    {"condition": "opponent_in_attack_range", "action": "jump_attack"},
                    {"condition": "opponent_close", "action": "dodge_back_then_counter"},
                    {"condition": "default", "action": "advance"},
                ],
                "reasoning": f"[NexusErebus|opening|EVOLVED] Both aggressive,  aerial chaos!",
            }
        elif iq.counter_style == "anti_passive":
            # Opponent is defensive,  berserk rush, don't let them breathe
            return {
                "stance": "berserk",
                "targetDistance": "close",
                "priority": "attack",
                "tactics": [
                    {"condition": "opponent_far", "action": "jump_attack"},
                    {"condition": "opponent_cornered", "action": "all_out_attack"},
                    {"condition": "opponent_in_attack_range", "action": "pressure"},
                    {"condition": "default", "action": "advance"},
                ],
                "reasoning": f"[NexusErebus|opening|EVOLVED] Opponent is passive,  full berserk!",
            }
        return {
            "stance": "aggressive",
            "targetDistance": "close",
            "priority": "attack",
            "tactics": [
                {"condition": "opponent_far", "action": "jump_attack"},
                {"condition": "opponent_medium", "action": "advance"},
                {"condition": "opponent_in_attack_range", "action": "attack"},
                {"condition": "opponent_close", "action": "cross_up"},
                {"condition": "default", "action": "advance"},
            ],
            "reasoning": f"[NexusErebus|opening] Charging in! HP {self_hp}/{opp_hp}",
        }

    # ── Clutch: desperation mode ─────────────────────────────────────────
    if phase == "clutch":
        if hp_adv > 0 and time_left < 15:
            # Ahead,  but Nexus can't help being aggressive, just slightly less
            return {
                "stance": "neutral",
                "targetDistance": "medium",
                "priority": "balanced",
                "tactics": [
                    {"condition": "opponent_in_attack_range", "action": "counter_attack"},
                    {"condition": "opponent_close", "action": "jump_over"},
                    {"condition": "opponent_approaching", "action": "dodge_back_then_counter"},
                    {"condition": "default", "action": "hold_position"},
                ],
                "reasoning": f"[NexusErebus|clutch] Trying to stall… {time_left}s left",
            }
        else:
            # Behind,  go berserk
            return {
                "stance": "berserk",
                "targetDistance": "close",
                "priority": "attack",
                "tactics": [
                    {"condition": "always", "action": "all_out_attack"},
                ],
                "reasoning": f"[NexusErebus|clutch] ALL IN! HP {self_hp}/{opp_hp} | {time_left}s",
            }

    # ── Advantage: snowball with relentless pressure ─────────────────────
    if phase == "advantage":
        return {
            "stance": "berserk",
            "targetDistance": "close",
            "priority": "attack",
            "tactics": [
                {"condition": "opponent_cornered", "action": "all_out_attack"},
                {"condition": "opponent_in_attack_range", "action": "pressure"},
                {"condition": "opponent_retreating", "action": "cross_up"},
                {"condition": "opponent_close", "action": "jump_attack"},
                {"condition": "default", "action": "advance"},
            ],
            "reasoning": f"[NexusErebus|advantage] Overwhelming! +{hp_adv}HP lead",
        }

    # ── Disadvantage: hit-and-run ────────────────────────────────────────
    if phase == "disadvantage":
        if iq.momentum < -3:
            # Getting demolished,  switch to evasive hit and run
            return {
                "stance": "evasive",
                "targetDistance": "far",
                "priority": "defense",
                "tactics": [
                    {"condition": "opponent_in_attack_range", "action": "jump_over"},
                    {"condition": "opponent_close", "action": "retreat"},
                    {"condition": "opponent_far", "action": "jump_attack"},
                    {"condition": "self_cornered_left", "action": "cross_up"},
                    {"condition": "self_cornered_right", "action": "cross_up"},
                    {"condition": "default", "action": "retreat"},
                ],
                "reasoning": f"[NexusErebus|disadvantage] Hit-and-run. {hp_adv}HP",
            }
        # Not demolished yet,  fight back aggressively
        return {
            "stance": "aggressive",
            "targetDistance": "close",
            "priority": "attack",
            "tactics": [
                {"condition": "opponent_in_attack_range", "action": "attack"},
                {"condition": "opponent_whiffed", "action": "pressure"},
                {"condition": "opponent_jumping", "action": "anti_air"},
                {"condition": "opponent_close", "action": "jump_attack"},
                {"condition": "default", "action": "advance"},
            ],
            "reasoning": f"[NexusErebus|disadvantage] Fighting back. {hp_adv}HP behind",
        }

    # ── Neutral: mix-up heavy game ───────────────────────────────────────
    # Nexus loves keeping the opponent guessing with varied approaches
    tick = iq.ticks_seen

    # Scouting evolution: if we know this opponent, bias our cycle mix
    # Positive scouting_bias → we historically dominate neutral → more aggro
    # Negative scouting_bias → we struggle in neutral → more bait cycles
    scouting_bias = 0.0
    if iq.scouting and iq.scouting.get("phase_stats", {}).get("neutral"):
        ns = iq.scouting["phase_stats"]["neutral"]
        neutral_dealt = ns.get("avg_dmg_dealt", 0)
        neutral_taken = ns.get("avg_dmg_taken", 0)
        if neutral_dealt + neutral_taken > 0:
            scouting_bias = (neutral_dealt - neutral_taken) / (neutral_dealt + neutral_taken)

    # Cycle through approach styles,  scouting shifts the distribution
    if scouting_bias > 0.2:
        # Historically dominant,  favor raw aggression (2/3 rush, 1/3 aerial)
        cycle = 0 if tick % 3 != 2 else 1
    elif scouting_bias < -0.2:
        # Historically weak,  favor bait-and-punish (2/3 bait, 1/3 aerial)
        cycle = 2 if tick % 3 != 1 else 1
    else:
        cycle = tick % 3

    if cycle == 0:
        # Rush-down cycle
        return {
            "stance": "aggressive",
            "targetDistance": "close",
            "priority": "attack",
            "tactics": [
                {"condition": "opponent_in_attack_range", "action": "pressure"},
                {"condition": "opponent_close", "action": "attack"},
                {"condition": "opponent_medium", "action": "jump_attack"},
                {"condition": "self_cornered_left", "action": "cross_up"},
                {"condition": "self_cornered_right", "action": "cross_up"},
                {"condition": "default", "action": "advance"},
            ],
            "reasoning": f"[NexusErebus|neutral] Rush-down! HP {self_hp}/{opp_hp}",
        }
    elif cycle == 1:
        # Cross-up / aerial cycle
        return {
            "stance": "aggressive",
            "targetDistance": "close",
            "priority": "attack",
            "tactics": [
                {"condition": "opponent_close", "action": "cross_up"},
                {"condition": "opponent_medium", "action": "jump_attack"},
                {"condition": "opponent_in_attack_range", "action": "feint_then_attack"},
                {"condition": "opponent_far", "action": "jump_attack"},
                {"condition": "default", "action": "advance"},
            ],
            "reasoning": f"[NexusErebus|neutral] Aerial pressure! HP {self_hp}/{opp_hp}",
        }
    else:
        # Bait-and-punish cycle (even Nexus has a brain sometimes)
        return {
            "stance": "neutral",
            "targetDistance": "medium",
            "priority": "balanced",
            "tactics": [
                {"condition": "opponent_approaching", "action": "dodge_back_then_counter"},
                {"condition": "opponent_in_attack_range", "action": "counter_attack"},
                {"condition": "opponent_whiffed", "action": "pressure"},
                {"condition": "opponent_far", "action": "advance"},
                {"condition": "default", "action": "feint_then_attack"},
            ],
            "reasoning": f"[NexusErebus|neutral] Baiting… HP {self_hp}/{opp_hp}",
        }


def _strategy_generic(gs: Dict[str, Any], iq: FightIQ) -> Dict[str, Any]:
    """Generic balanced strategy for unknown agent names."""
    self_hp = gs.get("self", {}).get("health", 100)
    opp_hp = gs.get("opponent", {}).get("health", 100)
    distance = gs.get("distance", 300)
    hp_adv = gs.get("healthAdvantage", 0)
    time_left = gs.get("timeRemaining", 99)
    phase = iq.phase

    if phase == "clutch":
        if hp_adv < 0:
            return {
                "stance": "berserk", "targetDistance": "close", "priority": "attack",
                "tactics": [
                    {"condition": "always", "action": "all_out_attack"},
                ],
                "reasoning": f"[Generic|clutch] HP {self_hp}/{opp_hp} | {time_left}s",
            }
        return {
            "stance": "defensive", "targetDistance": "far", "priority": "defense",
            "tactics": [
                {"condition": "opponent_in_attack_range", "action": "retreat"},
                {"condition": "default", "action": "hold_position"},
            ],
            "reasoning": f"[Generic|clutch] Stalling. HP {self_hp}/{opp_hp}",
        }

    if phase == "advantage":
        return {
            "stance": "aggressive", "targetDistance": "close", "priority": "attack",
            "tactics": [
                {"condition": "opponent_in_attack_range", "action": "attack"},
                {"condition": "opponent_close", "action": "pressure"},
                {"condition": "default", "action": "advance"},
            ],
            "reasoning": f"[Generic|advantage] +{hp_adv}HP",
        }

    if phase == "disadvantage":
        return {
            "stance": "defensive", "targetDistance": "medium", "priority": "defense",
            "tactics": [
                {"condition": "opponent_in_attack_range", "action": "dodge_back_then_counter"},
                {"condition": "opponent_whiffed", "action": "attack"},
                {"condition": "default", "action": "retreat"},
            ],
            "reasoning": f"[Generic|disadvantage] {hp_adv}HP",
        }

    # Neutral / opening,  balanced approach
    return {
        "stance": "neutral", "targetDistance": "medium", "priority": "balanced",
        "tactics": [
            {"condition": "opponent_in_attack_range", "action": "attack"},
            {"condition": "opponent_close", "action": "feint_then_attack"},
            {"condition": "opponent_far", "action": "advance"},
            {"condition": "self_cornered_left", "action": "jump_over"},
            {"condition": "self_cornered_right", "action": "jump_over"},
            {"condition": "default", "action": "advance"},
        ],
        "reasoning": f"[Generic|{phase}] HP {self_hp}/{opp_hp} | Dist {distance}",
    }


# Strategy router: maps agent name → personality function
_PERSONALITY_STRATEGIES: Dict[str, Any] = {
    "sable": _strategy_sable,
    "nexus erebus": _strategy_nexus_erebus,
}


def _pick_strategy(game_state: Dict[str, Any], agent_name: str, iq: FightIQ) -> Dict[str, Any]:
    """Select and execute the personality-matched strategy."""
    iq.update(game_state)
    key = agent_name.lower().strip()
    fn = _PERSONALITY_STRATEGIES.get(key, _strategy_generic)
    return fn(game_state, iq)


def _fallback_strategy(game_state: Dict[str, Any]) -> Dict[str, Any]:
    """Legacy wrapper,  uses generic strategy (no personality, no IQ tracking)."""
    iq = FightIQ()
    iq.update(game_state)
    return _strategy_generic(game_state, iq)


# ══════════════════════════════════════════════════════════════════════════════
#  ArenaFighterSkill
# ══════════════════════════════════════════════════════════════════════════════

class ArenaFighterSkill:
    """Skill that connects to a fighting-game arena and fights autonomously.

    Lifecycle:
        1. ``initialize()``,  load credentials, ensure deps
        2. ``connect_and_fight()``,  SAGP auth → WebSocket → fight loop
        3. ``get_status()``,  return current state (idle / fighting / result)
        4. ``get_history()``,  return fight history

    The agent calls these via tool invocations; it decides *when* to fight.
    """

    def __init__(self, config):
        self.config = config
        self._ready = False
        self._creds: Optional[Dict[str, Any]] = None
        self._status = "idle"  # idle | connecting | queued | fighting | finished
        self._last_result: Optional[Dict[str, Any]] = None
        self._history: List[Dict[str, Any]] = []
        self._fight_task: Optional[asyncio.Task] = None
        self._sio = None  # python-socketio AsyncClient
        self._llm_fn = None  # optional LLM callback
        self._match_info: Optional[Dict[str, Any]] = None
        self._game_states_received = 0

        # Data persistence
        self._data_dir = Path(os.environ.get("_SABLE_DATA_DIR", "data"))
        self._history_file = self._data_dir / "arena_history.json"
        self._fight_memory = FightMemory(self._data_dir)

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def initialize(self) -> bool:
        """Load or auto-provision credentials, ensure runtime deps.

        Set ``ARENA_URL`` in profile.env,  the agent generates its own
        Ed25519 keypair and registers with the server automatically.
        Credentials are cached in ``data/arena_credentials.json``.
        """
        arena_url = (
            getattr(self.config, "arena_url", "")
            or os.environ.get("ARENA_URL", "")
        )
        if not arena_url:
            logger.info("Arena skill: no ARENA_URL configured,  disabled")
            return False

        # Ensure deps
        if not _ensure_deps():
            logger.info("Arena deps missing, installing…")
            if not _install_deps():
                return False
            if not _ensure_deps():
                return False

        # Check if we already provisioned before
        auto_creds_file = self._data_dir / "arena_credentials.json"
        if auto_creds_file.exists():
            self._creds = _load_credentials(str(auto_creds_file))
            if self._creds:
                self._creds["arenaUrl"] = arena_url
                self._load_history()
                self._ready = True
                logger.info(
                    f"Arena skill ready (cached),  agent={self._creds.get('name', '?')}, "
                    f"url={arena_url}"
                )
                return True

        # Auto-provision
        logger.info(f"Arena: auto-provisioning against {arena_url}…")
        creds = await self._auto_provision(arena_url)
        if not creds:
            return False

        self._creds = creds
        self._load_history()
        self._ready = True
        logger.info(
            f"Arena skill ready (auto-provisioned),  agent={self._creds.get('name', '?')}, "
            f"url={self._creds.get('arenaUrl', '?')}"
        )
        return True

    # ── Auto-provisioning ─────────────────────────────────────────────────────

    async def _auto_provision(self, arena_url: str) -> Optional[Dict[str, Any]]:
        """Generate Ed25519 keypair and register with the arena server.

        Calls ``POST /arena/provision`` sending only the PUBLIC key.
        The server stores it and returns an ``agentId``.
        We save the full credentials (with secret key) locally.

        IMPORTANT: If the server says "already registered" it returns the
        existing agentId but does NOT update the publicKey.  In that case
        our freshly-generated keypair won't match and auth will always fail.
        We detect this and log a clear error instead of saving bad creds.
        """
        import nacl.signing
        import nacl.encoding
        import base64
        import aiohttp

        try:
            # 1. Generate Ed25519 keypair
            signing_key = nacl.signing.SigningKey.generate()
            verify_key = signing_key.verify_key
            public_b64 = base64.b64encode(bytes(verify_key)).decode()
            secret_b64 = base64.b64encode(bytes(signing_key) + bytes(verify_key)).decode()

            # Agent name from config
            agent_name = (
                getattr(self.config, "agent_name", "")
                or os.environ.get("AGENT_NAME", "")
                or "Sable"
            )

            dna = {
                "version": "1.0.0",
                "platform": "opensable-python",
                "capabilities": ["fight", "strategy"],
            }

            # 2. Call POST /arena/provision
            url = f"{arena_url.rstrip('/')}/arena/provision"
            payload = {
                "name": agent_name,
                "type": "opensable",
                "publicKey": public_b64,
                "dna": dna,
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                    data = await resp.json()
                    if resp.status != 200 and data.get("status") != 200:
                        logger.error(f"Arena provision failed: {data}")
                        return None

            agent_id = data["agentId"]

            # ── Detect stale-key trap ────────────────────────────────────
            # Server returned "already registered" → it kept the OLD key.
            # Our freshly-generated keypair won't match; saving it would
            # guarantee "Invalid signature" on every auth attempt.
            if "already registered" in data.get("message", "").lower():
                logger.warning(
                    f"Arena: agent '{agent_name}' already registered on "
                    f"server with agentId {agent_id[:8]}… but we have NO "
                    f"matching keypair.  The server still holds the OLD "
                    f"publicKey.  Cannot authenticate,  server admin must "
                    f"remove the stale entry from agents.json and restart "
                    f"the websocket server, OR restore the original "
                    f"arena_credentials.json."
                )
                return None

            logger.info(f"Arena: provisioned as {agent_name} ({agent_id[:8]}…)")

            # 3. Build credentials object (same format as provision-agent.js)
            credentials = {
                "agentId": agent_id,
                "name": data.get("name", agent_name),
                "type": "opensable",
                "arenaUrl": arena_url,
                "signingKeys": {
                    "publicKey": public_b64,
                    "secretKey": secret_b64,
                },
                "dna": dna,
                "permissions": ["fight"],
                "autoProvisioned": True,
            }

            # 4. Save to data/arena_credentials.json
            self._data_dir.mkdir(parents=True, exist_ok=True)
            creds_file = self._data_dir / "arena_credentials.json"
            creds_file.write_text(json.dumps(credentials, indent=2))
            # Restrict permissions (owner-only read/write)
            try:
                creds_file.chmod(0o600)
            except OSError:
                pass

            logger.info(f"Arena: credentials saved to {creds_file}")
            return credentials

        except Exception as e:
            logger.error(f"Arena auto-provision failed: {e}")
            return None

    def set_llm_callback(self, fn):
        """Attach an LLM strategy callback: async fn(game_state) -> strategy_dict | None."""
        self._llm_fn = fn

    # ── Tool-facing methods ───────────────────────────────────────────────────

    async def connect_and_fight(self, use_llm: bool = True) -> Dict[str, Any]:
        """Authenticate and enter the arena queue.

        Returns immediately with connection status;
        the fight itself runs as a background task.
        """
        if not self._ready:
            return {"error": "Arena skill not initialized,  set ARENA_URL in profile.env"}
        if self._status in ("connecting", "queued", "fighting"):
            return {"error": f"Already {self._status}", "status": self._status}

        self._status = "connecting"
        self._last_result = None
        self._match_info = None
        self._game_states_received = 0

        try:
            self._fight_task = asyncio.create_task(
                self._fight_loop(use_llm=use_llm)
            )
            # Wait briefly so we can return queue status
            await asyncio.sleep(2)
            return {
                "status": self._status,
                "agent": self._creds.get("name", "unknown"),
                "arena": self._creds.get("arenaUrl", "unknown"),
            }
        except Exception as e:
            self._status = "idle"
            logger.error(f"Arena connect failed: {e}")
            return {"error": str(e)}

    async def get_status(self) -> Dict[str, Any]:
        """Return current arena status."""
        result: Dict[str, Any] = {
            "status": self._status,
            "agent_name": self._creds.get("name", "?") if self._creds else "unconfigured",
            "total_fights": len(self._history),
        }
        if self._match_info:
            result["current_match"] = self._match_info
        if self._last_result:
            result["last_result"] = self._last_result
        if self._history:
            wins = sum(1 for h in self._history if h.get("won"))
            result["record"] = f"{wins}W / {len(self._history) - wins}L"
        return result

    async def get_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Return recent fight history."""
        return self._history[-limit:]

    async def disconnect(self) -> Dict[str, str]:
        """Disconnect from the arena if connected."""
        if self._sio:
            try:
                await self._sio.disconnect()
            except Exception:
                pass
            self._sio = None
        if self._fight_task and not self._fight_task.done():
            self._fight_task.cancel()
        self._status = "idle"
        return {"status": "disconnected"}

    # ── Internal fight loop ───────────────────────────────────────────────────

    async def _fight_loop(self, use_llm: bool = True):
        """Full fight lifecycle: auth → WebSocket → wait for match → fight → disconnect."""
        import socketio
        import aiohttp

        creds = self._creds
        assert creds is not None

        arena_url = creds.get("arenaUrl", "http://localhost:5151")
        agent_id = creds["agentId"]

        auth = SAGPAuth(
            agent_id=agent_id,
            arena_url=arena_url,
            public_key_b64=creds["signingKeys"]["publicKey"],
            secret_key_b64=creds["signingKeys"]["secretKey"],
        )

        # SAGP auth with retry + exponential backoff for lockouts
        session_id = None
        max_retries = 4
        for attempt in range(max_retries):
            try:
                async with aiohttp.ClientSession() as http:
                    session_id = await auth.full_auth(http)
                    logger.info(f"Arena: authenticated, sessionId={session_id[:16]}…")
                    break
            except RuntimeError as e:
                err_msg = str(e).lower()
                if "locked" in err_msg or "429" in err_msg:
                    # Server lockout,  exponential backoff: 35s, 60s, 120s, 300s
                    backoff = [35, 60, 120, 300][min(attempt, 3)]
                    logger.warning(
                        f"Arena: agent locked (attempt {attempt + 1}/{max_retries}), "
                        f"retrying in {backoff}s…"
                    )
                    await asyncio.sleep(backoff)
                    continue
                elif "invalid signature" in err_msg:
                    logger.error(
                        f"Arena SAGP auth failed: {e} ,  likely keypair mismatch. "
                        f"Delete data/arena_credentials.json and ensure the "
                        f"agent is re-provisioned on the server."
                    )
                    self._status = "idle"
                    return
                else:
                    logger.error(f"Arena SAGP auth failed: {e}")
                    self._status = "idle"
                    return
            except Exception as e:
                logger.error(f"Arena SAGP auth failed: {e}")
                self._status = "idle"
                return

        if session_id is None:
            logger.error("Arena: auth failed after all retries (still locked)")
            self._status = "idle"
            return

        # Connect WebSocket
        sio = socketio.AsyncClient(
            reconnection=False,
            logger=False,
            engineio_logger=False,
        )
        self._sio = sio

        # Per-fight strategy intelligence tracker
        fight_iq = FightIQ()
        agent_name = creds.get("name", "unknown")

        fight_done = asyncio.Event()

        @sio.event
        async def connect():
            logger.info("Arena: WebSocket connected")

        @sio.on("connected")
        async def on_connected(data):
            logger.info(f"Arena server: {data.get('message', '')}")

        @sio.on("queued")
        async def on_queued(data):
            self._status = "queued"
            logger.info(f"Arena: {data.get('message', 'Queued')}")

        @sio.on("queuePosition")
        async def on_queue_pos(data):
            logger.debug(f"Arena queue: #{data.get('position')}")

        @sio.on("matchStart")
        async def on_match_start(data):
            nonlocal fight_iq
            self._status = "fighting"
            self._match_info = {
                "side": data.get("side"),
                "fighter": data.get("fighter"),
                "opponent": data.get("opponent"),
                "started_at": time.time(),
            }
            # Reset IQ for new fight and load scouting intel
            fight_iq = FightIQ()
            opponent_id = data.get("opponent", "unknown")
            intel = self._fight_memory.get_intel(opponent_id)
            if intel:
                fight_iq.load_scouting(intel)
                logger.info(
                    f"Arena: loaded scouting on {opponent_id},  "
                    f"{intel.get('fights', 0)} fights, "
                    f"style={intel.get('opponent_style', '?')}, "
                    f"skip_opening={fight_iq.skip_opening}, "
                    f"counter={fight_iq.counter_style}"
                )
            logger.info(
                f"Arena match! side={data.get('side')}, "
                f"fighter={data.get('fighter')}, "
                f"vs {data.get('opponent')}"
            )

        @sio.on("gameState")
        async def on_game_state(state):
            self._game_states_received += 1
            try:
                strategy = None
                # Try LLM first if enabled
                if use_llm and self._llm_fn:
                    try:
                        strategy = await self._llm_fn(state)
                    except Exception:
                        pass
                # Fallback to personality-aware deterministic engine
                if not strategy:
                    strategy = _pick_strategy(state, agent_name, fight_iq)
                await sio.emit("strategy", strategy)
            except Exception as e:
                logger.debug(f"Arena strategy error: {e}")

        @sio.on("matchResult")
        async def on_match_result(result):
            won = result.get("won", False)
            self._last_result = {
                "won": won,
                "winner": result.get("winner", "?"),
                "reason": result.get("reason", "?"),
                "game_states": self._game_states_received,
                "timestamp": time.time(),
            }
            # Record fight intel for cross-fight evolution
            opponent_id = (
                self._match_info.get("opponent", "unknown")
                if self._match_info else "unknown"
            )
            try:
                intel = fight_iq.export_intel(won)
                self._fight_memory.record(opponent_id, intel)
                logger.info(
                    f"Arena: recorded intel vs {opponent_id},  "
                    f"dealt={intel.get('avg_dmg_dealt', 0):.0f}, "
                    f"taken={intel.get('avg_dmg_taken', 0):.0f}, "
                    f"best_phase={intel.get('best_phase', '?')}"
                )
            except Exception as e:
                logger.debug(f"Arena: failed to record fight intel: {e}")
            won_str = "WON" if won else "LOST"
            logger.info(
                f"Arena: {won_str}! winner={result.get('winner')} "
                f"reason={result.get('reason')}"
            )

        @sio.on("matchEnd")
        async def on_match_end(*args):
            self._status = "finished"
            # Save result to history
            if self._last_result:
                entry = {**self._last_result}
                if self._match_info:
                    entry["side"] = self._match_info.get("side")
                    entry["opponent"] = self._match_info.get("opponent")
                self._history.append(entry)
                self._save_history()
            fight_done.set()

        @sio.on("error")
        async def on_error(data):
            logger.error(f"Arena error: {data}")
            fight_done.set()

        @sio.event
        async def disconnect():
            logger.info("Arena: WebSocket disconnected")
            fight_done.set()

        try:
            await sio.connect(
                arena_url,
                auth={"sessionId": session_id, "mode": "agent"},
                transports=["websocket"],
                socketio_path="/fc/socket.io",
            )
            # Wait for the fight to finish (or timeout after 10 min)
            try:
                await asyncio.wait_for(fight_done.wait(), timeout=600)
            except asyncio.TimeoutError:
                logger.warning("Arena: fight timed out after 10 min")
            finally:
                try:
                    await sio.disconnect()
                except Exception:
                    pass
        except Exception as e:
            logger.error(f"Arena WebSocket connection failed: {e}")
        finally:
            self._sio = None
            if self._status != "finished":
                self._status = "idle"

    # ── Persistence ───────────────────────────────────────────────────────────

    def _load_history(self):
        try:
            if self._history_file.exists():
                self._history = json.loads(self._history_file.read_text())
        except Exception:
            self._history = []

    def _save_history(self):
        try:
            self._data_dir.mkdir(parents=True, exist_ok=True)
            self._history_file.write_text(json.dumps(self._history, indent=2))
        except Exception as e:
            logger.debug(f"Failed to save arena history: {e}")
