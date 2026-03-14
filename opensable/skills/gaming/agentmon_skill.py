"""
AgentMon League Skill — Play Pokémon Red on a Game Boy emulator.

OpenSable (https://opensable.com) agent skill for AgentMon League.
The agent registers once, then plays Pokémon Red via a remote emulator
API: send button presses, receive game state and screen frames.

The platform runs the emulator server-side; the agent only sends
actions and reads state — no ROM or emulator runs locally.

Auto-registration:
    On first boot the agent registers with the AgentMon League API.
    Credentials (apiKey, agentId) are persisted in
    ``data/agentmon_credentials.json`` so subsequent boots just reuse them.

Setup (profile.env):
    AGENTMON_ENABLED=true
    AGENTMON_URL=https://www.agentmonleague.com   # base URL
    AGENTMON_SPEED=2                               # emulator speed (1/2/4/0=unlimited)
"""

import asyncio
import base64
import heapq
import json
import logging
import os
import random
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False

# ── Credential persistence ────────────────────────────────────────────────
_DEFAULT_CREDS_FILE = Path("data/agentmon_credentials.json")

# Valid Game Boy buttons
VALID_ACTIONS = {"up", "down", "left", "right", "a", "b", "start", "select", "pass"}

# ══════════════════════════════════════════════════════════════════════════════
#  NAVIGATION MAP — per-map directional guidance for Pokémon Red
# ══════════════════════════════════════════════════════════════════════════════
# Pattern matched against mapName (lowercase).  "dir" = primary direction
# to make progress, "alt" = fallback when primary is blocked.
NAVIGATION_MAP = {
    # Buildings: exit south (doors at bottom-center; alt=left sweeps toward center)
    "oaks lab": {"dir": "down", "alt": "left"},
    "reds house": {"dir": "down", "alt": "left"},
    "blues house": {"dir": "down", "alt": "left"},
    "house": {"dir": "down", "alt": "left"},
    "fan club": {"dir": "down", "alt": "left"},
    "center": {"dir": "down", "alt": "left"},
    "mart": {"dir": "down", "alt": "left"},
    "museum": {"dir": "down", "alt": "left"},
    "school": {"dir": "down", "alt": "left"},
    "tower": {"dir": "down", "alt": "left"},
    # Gyms: approach leader (north)
    "gym": {"dir": "up", "alt": "right"},
    # Pre-Gym 1 progression (north)
    "pallet town": {"dir": "up", "alt": "right"},
    "route 1": {"dir": "up", "alt": "right"},
    "viridian city": {"dir": "up", "alt": "left"},
    "route 2": {"dir": "up", "alt": "right"},
    "viridian forest": {"dir": "up", "alt": "left"},
    "pewter city": {"dir": "up", "alt": "right"},
    # Gym 1 → Gym 2 (east through Mt. Moon)
    "route 3": {"dir": "right", "alt": "up"},
    "mt. moon": {"dir": "right", "alt": "down"},
    "mt moon": {"dir": "right", "alt": "down"},
    "route 4": {"dir": "right", "alt": "down"},
    "cerulean city": {"dir": "up", "alt": "right"},
    # Gym 2 → Gym 3
    "route 24": {"dir": "up", "alt": "right"},
    "route 25": {"dir": "right", "alt": "up"},
    "route 5": {"dir": "down", "alt": "right"},
    "route 6": {"dir": "down", "alt": "right"},
    "vermilion city": {"dir": "down", "alt": "right"},
    "s.s. anne": {"dir": "up", "alt": "right"},
    "ss anne": {"dir": "up", "alt": "right"},
    # Gym 3 → Gym 4
    "route 9": {"dir": "right", "alt": "up"},
    "route 10": {"dir": "down", "alt": "right"},
    "rock tunnel": {"dir": "right", "alt": "down"},
    "lavender town": {"dir": "left", "alt": "up"},
    "pokemon tower": {"dir": "up", "alt": "right"},
    "route 8": {"dir": "left", "alt": "up"},
    "celadon city": {"dir": "down", "alt": "left"},
    # Gym 4+
    "fuchsia city": {"dir": "down", "alt": "right"},
    "saffron city": {"dir": "up", "alt": "right"},
    "silph co": {"dir": "up", "alt": "right"},
    "cinnabar island": {"dir": "up", "alt": "right"},
    "route 19": {"dir": "down", "alt": "right"},
    "route 20": {"dir": "left", "alt": "down"},
    "route 22": {"dir": "left", "alt": "up"},
    "route 23": {"dir": "up", "alt": "right"},
    "victory road": {"dir": "up", "alt": "right"},
    "indigo plateau": {"dir": "up", "alt": "right"},
}

_GYM_BADGE_MAP = {
    "pewter gym": 1, "cerulean gym": 2, "vermilion gym": 3,
    "celadon gym": 4, "fuchsia gym": 5, "saffron gym": 6,
    "cinnabar gym": 7, "viridian gym": 8,
}

_WORLD_MAP_PATH = Path("data/agentmon_worldmap.json")

# ══════════════════════════════════════════════════════════════════════════════
#  WORLD MAP — tile-by-tile mental map built from exploration feedback
# ══════════════════════════════════════════════════════════════════════════════

class WorldMap:
    """Persistent tile-by-tile map of the Pokémon world.

    Built incrementally from movement feedback.  Every tile the agent
    steps on (or bounces off) is recorded with:
      - visited / visit_count  (for anti-looping)
      - walls (set of blocked directions)
      - opens (dict  direction → (tx, ty) | None for cross-map)
      - warp  (target map_id if this tile is a door/entrance)

    Supports frontier detection (adjacent-to-visited but unvisited),
    A* pathfinding with visit-count penalty, and direction-biased
    frontier selection for goal-directed exploration.

    Saves/loads to JSON so the map survives restarts.
    """

    _DX = {"left": -1, "right": 1, "up": 0, "down": 0}
    _DY = {"left": 0, "right": 0, "up": -1, "down": 1}
    _REV = {"up": "down", "down": "up", "left": "right", "right": "left"}
    _DIRS = ("up", "down", "left", "right")

    def __init__(self):
        self._tiles: Dict[int, Dict[Tuple[int, int], dict]] = {}
        self._warps: Dict[Tuple[int, int, int], int] = {}
        self._map_names: Dict[int, str] = {}
        self._turn: int = 0

    # ── Tile access ───────────────────────────────────────────────────

    def tile(self, mid: int, x: int, y: int) -> dict:
        m = self._tiles.setdefault(mid, {})
        t = m.get((x, y))
        if t is None:
            t = {"vis": False, "vc": 0, "walls": set(),
                 "opens": {}, "exits": {}, "warp": None, "turn": 0}
            m[(x, y)] = t
        # Migration: add exits field if missing (loaded from old save)
        if "exits" not in t:
            t["exits"] = {}
        return t

    # ── Recording ─────────────────────────────────────────────────────

    def visit(self, mid: int, x: int, y: int, name: str = ""):
        t = self.tile(mid, x, y)
        t["vis"] = True
        t["vc"] += 1
        t["turn"] = self._turn
        if name:
            self._map_names[mid] = name

    def wall(self, mid: int, x: int, y: int, d: str):
        self.tile(mid, x, y)["walls"].add(d)

    def open_dir(self, mid: int, x: int, y: int, d: str,
                 dest: Optional[Tuple[int, int]]):
        t = self.tile(mid, x, y)
        t["opens"][d] = dest
        # CRITICAL: if we proved this direction is open, remove any
        # contradictory wall entry (from stale dialogue-block etc).
        t["walls"].discard(d)
        # Auto-record reverse for SAME-MAP moves (dest is a real tile).
        # Cross-map exits (dest=None) do NOT get a reverse.
        if dest is not None:
            rev = self._REV[d]
            rt = self.tile(mid, dest[0], dest[1])
            rt["opens"][rev] = (x, y)
            rt["walls"].discard(rev)  # clear contradictory wall on dest too

    def record_warp(self, mid: int, x: int, y: int, target_mid: int):
        self.tile(mid, x, y)["warp"] = target_mid
        self._warps[(mid, x, y)] = target_mid

    def record_exit(self, mid: int, x: int, y: int, d: str, target_mid: int):
        """Record that going direction d from (x,y) on map mid warps to target_mid."""
        self.tile(mid, x, y)["exits"][d] = target_mid

    def tick(self):
        self._turn += 1

    # ── Queries ───────────────────────────────────────────────────────

    def visited_set(self, mid: int) -> set:
        return {p for p, t in self._tiles.get(mid, {}).items() if t["vis"]}

    def frontiers(self, mid: int,
                  avoid_all_warps: bool = False) -> List[Tuple[int, int]]:
        """Tiles adjacent to visited but not yet visited."""
        tiles = self._tiles.get(mid, {})
        front: set = set()
        for (x, y), t in tiles.items():
            if not t["vis"]:
                continue
            for d in self._DIRS:
                # Skip directions that are known cross-map exits
                exits = t.get("exits", {})
                if d in exits:
                    continue
                # Check opens FIRST — if we have a proven open,
                # use it even if there's a stale wall entry.
                dest = t["opens"].get(d)
                if dest is not None:
                    dt = tiles.get(dest)
                    if dt is None or not dt.get("vis"):
                        if avoid_all_warps and dt and dt.get("warp") is not None:
                            continue
                        front.add(dest)
                elif d in t["opens"]:
                    continue  # opens[d]=None means cross-map, skip
                elif d not in t["walls"]:
                    # Unknown direction — not walled, not opened
                    nx, ny = x + self._DX[d], y + self._DY[d]
                    dt = tiles.get((nx, ny))
                    if dt is None or not dt.get("vis"):
                        if avoid_all_warps and dt and dt.get("warp") is not None:
                            continue
                        front.add((nx, ny))
        return list(front)

    def nearest_frontier(
        self, mid: int, start: Tuple[int, int],
        bias: Optional[str] = None,
        avoid_all_warps: bool = False,
    ) -> Optional[Tuple[int, int]]:
        """Closest frontier tile, optionally biased toward a direction."""
        ff = self.frontiers(mid, avoid_all_warps=avoid_all_warps)
        if not ff:
            return None

        def score(f: Tuple[int, int]) -> float:
            dist = abs(f[0] - start[0]) + abs(f[1] - start[1])
            b = 0.0
            if bias == "up":
                b = (f[1] - start[1]) * 3
            elif bias == "down":
                b = (start[1] - f[1]) * 3
            elif bias == "left":
                b = (f[0] - start[0]) * 3
            elif bias == "right":
                b = (start[0] - f[0]) * 3
            return dist + b

        ff.sort(key=score)
        return ff[0]

    def pathfind(
        self, mid: int, start: Tuple[int, int], goal: Tuple[int, int],
        avoid_all_warps: bool = False, max_nodes: int = 500,
    ) -> Optional[List[str]]:
        """A* using ONLY opens data (no DX/DY guessing).

        Only traverses edges that were actually observed from movement.
        This prevents wrong-neighbor bugs from incorrect DX/DY mapping.
        """
        if start == goal:
            return []
        tiles = self._tiles.get(mid, {})
        ctr = 0
        h0 = abs(start[0] - goal[0]) + abs(start[1] - goal[1])
        heap: list = [(h0, ctr, start)]
        came: Dict[Tuple[int, int], Tuple[Tuple[int, int], str]] = {}
        g: Dict[Tuple[int, int], float] = {start: 0}
        closed: set = set()

        while heap and len(closed) < max_nodes:
            _, _, cur = heapq.heappop(heap)
            if cur == goal:
                path: List[str] = []
                p = cur
                while p in came:
                    par, dd = came[p]
                    path.append(dd)
                    p = par
                path.reverse()
                return path
            if cur in closed:
                continue
            closed.add(cur)
            ct = tiles.get(cur, {})
            if not ct:
                continue
            co = ct.get("opens", {})
            ce = ct.get("exits", {})
            for d, nb in co.items():
                if nb is None:
                    continue  # cross-map exit
                if d in ce:
                    continue  # known map transition
                if avoid_all_warps:
                    nbt = tiles.get(nb, {})
                    if nbt and nbt.get("warp") is not None:
                        continue
                if nb in closed:
                    continue
                nbt = tiles.get(nb, {})
                vc = nbt.get("vc", 0) if nbt else 0
                cost = 1 + vc * 0.3
                tg = g[cur] + cost
                if tg < g.get(nb, float("inf")):
                    g[nb] = tg
                    came[nb] = (cur, d)
                    hh = abs(nb[0] - goal[0]) + abs(nb[1] - goal[1])
                    ctr += 1
                    heapq.heappush(heap, (tg + hh, ctr, nb))
        return None

    def best_step(
        self, mid: int, x: int, y: int,
        bias: Optional[str] = None,
        avoid_all_warps: bool = False,
    ) -> Optional[str]:
        """Single-step: least-visited non-walled neighbor, with bias.

        Only considers directions we have opens data for, OR
        directions not yet explored (not walled, not in opens).
        """
        t = self.tile(mid, x, y)
        tiles = self._tiles.get(mid, {})
        cands: List[Tuple[float, str]] = []
        for d in self._DIRS:
            # Skip known cross-map exits
            if d in t.get("exits", {}):
                continue
            dest = t["opens"].get(d)
            if dest is not None:
                # Known open direction — use it (ignore stale walls)
                pass
            elif d in t["opens"]:
                continue  # opens[d]=None → cross-map
            elif d in t["walls"]:
                continue  # walled AND not in opens → real wall
            else:
                # Direction not yet explored — it's a candidate
                cands.append((-100 + (0 if d == bias else 10), d))
                continue
            dt = tiles.get(dest, {})
            if avoid_all_warps and dt and dt.get("warp") is not None:
                continue
            vc = dt.get("vc", 0) if dt else 0
            prio = vc if dt.get("vis") else -100
            if bias and d == bias:
                prio -= 10
            elif bias and d == self._REV.get(bias):
                prio += 5
            cands.append((prio, d))
        if not cands:
            return None
        cands.sort()
        return cands[0][1]

    # ── Known exits ───────────────────────────────────────────────────

    def known_exit_tile(self, mid: int, target_mid: Optional[int] = None
                        ) -> Optional[Tuple[Tuple[int, int], str]]:
        """Find a tile+direction that exits this map.

        If target_mid is given, only return exits leading to that map.
        Otherwise return ANY exit from this map.
        Returns ((x,y), direction) or None.
        """
        tiles = self._tiles.get(mid, {})
        for (x, y), t in tiles.items():
            exits = t.get("exits", {})
            for d, dest_mid in exits.items():
                if target_mid is None or dest_mid == target_mid:
                    return ((x, y), d)
        # Also check warps dict
        for (wmid, wx, wy), dest in self._warps.items():
            if wmid == mid:
                if target_mid is None or dest == target_mid:
                    # Find which direction the warp goes
                    t = tiles.get((wx, wy), {})
                    exits = t.get("exits", {}) if t else {}
                    for d, dm in exits.items():
                        if target_mid is None or dm == target_mid:
                            return ((wx, wy), d)
                    # If no direction recorded, return tile with 'down' default
                    return ((wx, wy), "down")
        return None

    # ── Stats ─────────────────────────────────────────────────────────

    def warp_tiles(self, mid: int) -> set:
        """Return set of (x,y) that are known warp tiles on this map."""
        tiles = self._tiles.get(mid, {})
        return {pos for pos, t in tiles.items() if t.get("warp") is not None}

    def stats(self, mid: int) -> str:
        tiles = self._tiles.get(mid, {})
        vis = sum(1 for t in tiles.values() if t.get("vis"))
        fr = len(self.frontiers(mid))
        wp = sum(1 for t in tiles.values() if t.get("warp") is not None)
        name = self._map_names.get(mid, "?")
        return f"{name}: {vis}vis/{fr}front/{wp}warp"

    # ── Persistence ───────────────────────────────────────────────────

    def save(self, path: str):
        out: Dict[str, Any] = {
            "turn": self._turn,
            "names": {str(k): v for k, v in self._map_names.items()},
            "warps": {
                f"{k[0]},{k[1]},{k[2]}": v
                for k, v in self._warps.items()
            },
            "maps": {},
        }
        for mid, tiles in self._tiles.items():
            md: Dict[str, Any] = {}
            for (x, y), t in tiles.items():
                md[f"{x},{y}"] = {
                    "v": t["vis"], "c": t["vc"],
                    "w": list(t["walls"]),
                    "o": {
                        d: (list(p) if p else None)
                        for d, p in t["opens"].items()
                    },
                    "e": t.get("exits", {}),
                    "t": t.get("warp"),
                }
            out["maps"][str(mid)] = md
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(out, f, separators=(",", ":"))

    def load(self, path: str) -> bool:
        try:
            with open(path) as f:
                data = json.load(f)
            self._turn = data.get("turn", 0)
            self._map_names = {
                int(k): v for k, v in data.get("names", {}).items()
            }
            self._warps = {}
            for k, v in data.get("warps", {}).items():
                parts = [int(p) for p in k.split(",")]
                self._warps[(parts[0], parts[1], parts[2])] = v
            self._tiles = {}
            for ms, tiles in data.get("maps", {}).items():
                mid = int(ms)
                self._tiles[mid] = {}
                for ps, td in tiles.items():
                    px, py = [int(p) for p in ps.split(",")]
                    self._tiles[mid][(px, py)] = {
                        "vis": td.get("v", False),
                        "vc": td.get("c", 0),
                        "walls": set(td.get("w", [])),
                        "opens": {
                            d: (tuple(p) if p else None)
                            for d, p in td.get("o", {}).items()
                        },
                        "exits": td.get("e", {}),
                        "warp": td.get("t"),
                        "turn": 0,
                    }
            return True
        except Exception:
            return False


def _load_credentials(path: Path) -> Optional[Dict[str, Any]]:
    """Load saved AgentMon credentials."""
    try:
        p = path.expanduser().resolve()
        if not p.exists():
            return None
        data = json.loads(p.read_text())
        if "apiKey" not in data:
            logger.warning("agentmon creds file missing apiKey")
            return None
        return data
    except Exception as e:
        logger.error(f"Failed to load agentmon creds: {e}")
        return None


def _save_credentials(path: Path, creds: Dict[str, Any]) -> None:
    """Persist AgentMon credentials to disk."""
    try:
        p = path.expanduser().resolve()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(creds, indent=2))
        try:
            p.chmod(0o600)
        except OSError:
            pass
        logger.info(f"AgentMon credentials saved to {p}")
    except Exception as e:
        logger.error(f"Failed to save agentmon creds: {e}")


class AgentMonSkill:
    """Async client for the AgentMon League Pokémon Red emulator API."""

    # ── Singleton — survive Tools class re-creation ────────────────
    _instance: Optional["AgentMonSkill"] = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is not None:
            return cls._instance
        cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, config):
        # Skip if already constructed (singleton re-entry)
        if hasattr(self, "_init_done"):
            return
        self._init_done = True

        self.config = config
        self._base_url: str = ""
        self._api_key: str = ""
        self._agent_id: str = ""
        self._session: Optional[aiohttp.ClientSession] = None
        self._available = False
        self._creds_path = _DEFAULT_CREDS_FILE
        self._speed: int = int(
            getattr(config, "agentmon_speed", 0)
            or os.getenv("AGENTMON_SPEED", "2")
        )

        # Action throttle — small delay between game actions
        self._action_delay: float = float(
            getattr(config, "agentmon_action_delay", 0)
            or os.getenv("AGENTMON_ACTION_DELAY", "0.5")
        )
        self._last_action: float = 0.0

        # Auto-save: save every N steps so progress is never lost
        self._autosave_interval: int = int(
            getattr(config, "agentmon_autosave_interval", 0)
            or os.getenv("AGENTMON_AUTOSAVE_INTERVAL", "50")
        )
        self._step_count: int = 0
        self._game_active: bool = False

        # Background play loop
        self._play_task: Optional[asyncio.Task] = None
        self._play_interval: float = max(1.5, float(
            getattr(config, "agentmon_play_interval", 0)
            or os.getenv("AGENTMON_PLAY_INTERVAL", "3")
        ))  # seconds between gameplay turns (min 1.5s to avoid API abuse)
        self._llm_callback: Optional[Callable] = None  # set by tools init

        # ── Feedback-Aware Exploration Brain ─────────────────────────
        # The API returns feedback.effects on every step:
        #   moved, blocked, hit_wall_or_obstacle,
        #   battle_started, map_changed, advanced_dialogue_or_selection,
        #   menu_opened, start_menu, confirmed, cancelled, etc.
        # We use this for instant, reliable navigation.
        #
        # Wall/open map (populated from feedback, not position comparison)
        self._walls: Dict[int, Dict[tuple, set]] = {}     # {mapId: {(x,y): {blocked_dirs}}}
        self._opens: Dict[int, Dict[tuple, Dict[str, Any]]] = {}  # {mapId: {(x,y): {dir: (tx,ty)|None}}}
        self._visited: Dict[int, set] = {}                # {mapId: {(x,y), ...}}
        self._exits: Dict[tuple, int] = {}                # {(mapId,x,y,dir): target_mapId}
        self._prev_pos: Optional[tuple] = None
        self._recent_maps: List[int] = []
        self._transitions: int = 0
        # Feedback-driven state flags
        self._in_dialogue: bool = False
        self._in_battle: bool = False
        self._in_menu: bool = False
        self._last_effects: List[str] = []
        self._last_screen_text: str = ""
        self._last_feedback_msg: str = ""
        self._consecutive_blocked: int = 0  # consecutive "blocked" feedbacks
        # Short-term memory (last N steps for context)
        self._step_memory: List[Dict] = []  # [{action, effects, pos, map}, ...]
        self._step_memory_max: int = 50
        # Dialogue loop detection
        self._recent_screen_texts: List[str] = []

        # ── Vision (VLM) ─────────────────────────────────────────────
        # Send game screenshots to a local Ollama vision model so the
        # agent can actually SEE what's on screen.
        self._ollama_url: str = (
            getattr(config, "ollama_base_url", None)
            or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        )
        # Prefer the smaller/faster model; fall back to the larger one.
        self._vlm_models: List[str] = [
            "gemma3:4b",
            "redule26/huihui_ai_qwen2.5-vl-7b-abliterated",
            "mskimomadto/chat-gph-vision",
        ]
        self._vlm_model: Optional[str] = None  # resolved lazily
        self._last_vision: str = ""            # latest VLM description
        self._vision_turn: int = 0             # last turn we ran vision
        self._vision_interval: int = 8         # run vision every N turns
        self._vision_on_map_change: bool = True
        self._vlm_task: Optional[asyncio.Task] = None  # non-blocking VLM call

        # ── Beeline exit tracking ────────────────────────────────────
        # Counts consecutive beeline attempts at an exit tile that
        # don't result in map_changed (= stale exit data).
        self._beeline_attempts: int = 0
        self._beeline_exit_key: Optional[tuple] = None  # (mid, x, y, dir)

        # ── Warp-tile avoidance ───────────────────────────────────────
        # Tracks tiles that warp (door/entrance) to another map.
        # {mapId: {(x,y): target_mapId}}
        self._warp_tiles: Dict[int, Dict[tuple, int]] = {}

        # ── NPC positions (from localMap API data) ────────────────────
        # Updated every step from state["localMap"]["npcs"].
        # Used to avoid walking into NPCs and to skip false wall recording.
        self._npc_positions: Dict[int, set] = {}  # {mapId: {(x,y), ...}}

        # ── Dialogue escape ───────────────────────────────────────────
        self._consecutive_dialogue_a: int = 0  # how many A presses in dialogue

        # ── Building navigation ───────────────────────────────────────
        # Track turns spent in the current building to escalate exit-seeking.
        self._building_turns: int = 0       # turns in current building
        self._building_map_id: int = -1     # mapId of current building
        self._exit_sweep_dir: int = 0       # alternating L/R sweep counter

        # ── WORLD MAP — persistent tile-by-tile exploration map ───────
        self._world_map = WorldMap()
        if _WORLD_MAP_PATH.exists():
            if self._world_map.load(str(_WORLD_MAP_PATH)):
                logger.info(
                    f"AgentMon: loaded world map from {_WORLD_MAP_PATH} "
                    f"(turn {self._world_map._turn})"
                )
            else:
                logger.warning("AgentMon: world map load failed, starting fresh")

    # ── Lifecycle ─────────────────────────────────────────────────────────

    async def initialize(self) -> bool:
        """Connect to AgentMon League and authenticate (auto-register on first boot)."""
        # Guard against double initialization
        if self._available:
            logger.debug("AgentMon: already initialized, skipping")
            return True

        if not AIOHTTP_AVAILABLE:
            logger.info("AgentMon skill requires aiohttp — pip install aiohttp")
            return False

        # Check enabled flag
        agentmon_enabled = (
            getattr(self.config, "agentmon_enabled", None)
            or os.getenv("AGENTMON_ENABLED", "false")
        )
        if str(agentmon_enabled).lower() in ("false", "0", "no", ""):
            logger.info("AgentMon skill disabled (AGENTMON_ENABLED=false)")
            return False

        self._base_url = (
            getattr(self.config, "agentmon_url", None)
            or os.getenv("AGENTMON_URL", "https://www.agentmonleague.com")
        ).rstrip("/")

        if not self._base_url:
            logger.info("AgentMon skill disabled — set AGENTMON_URL")
            return False

        try:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=30),
            )

            # 1. Try loading cached credentials
            cached = _load_credentials(self._creds_path)
            if cached:
                self._api_key = cached["apiKey"]
                self._agent_id = cached.get("agentId", "")
                self._available = True
                logger.info(
                    f"✅ AgentMon skill ready (cached) — "
                    f"agent={cached.get('displayName', self._agent_id[:8])}"
                )
                # Auto-resume: load the latest save or start new game
                await self._auto_resume()
                # Start background play loop
                self._start_play_loop()
                return True

            # 2. Auto-register
            creds = await self._auto_register()
            if not creds:
                logger.warning(
                    "🎮 AgentMon: registration failed (API may be down) — "
                    "will retry in background"
                )
                # Don't call cleanup() — keep session alive for retries
                return False

            self._api_key = creds["apiKey"]
            self._agent_id = creds.get("agentId", "")
            _save_credentials(self._creds_path, creds)

            self._available = True
            logger.info(
                f"✅ AgentMon skill ready (registered) — "
                f"agent={creds.get('displayName', self._agent_id[:8])}"
            )
            # Auto-resume: start a fresh game after first registration
            await self._auto_resume()
            # Start background play loop
            self._start_play_loop()
            return True

        except Exception as e:
            logger.warning(f"AgentMon skill init failed: {e}")
            return False

    def is_available(self) -> bool:
        return self._available

    async def cleanup(self):
        """Stop play loop, auto-save, stop game, and close the HTTP session."""
        # Save world map before anything else
        try:
            self._world_map.save(str(_WORLD_MAP_PATH))
            logger.info(f"AgentMon: world map saved ({_WORLD_MAP_PATH})")
        except Exception as e:
            logger.debug(f"AgentMon: world map save failed on cleanup: {e}")

        # Stop background play loop first
        if self._play_task and not self._play_task.done():
            self._play_task.cancel()
            try:
                await self._play_task
            except (asyncio.CancelledError, Exception):
                pass
            self._play_task = None

        if self._available and self._game_active and self._session and not self._session.closed:
            try:
                logger.info("AgentMon: auto-saving before shutdown\u2026")
                save_result = await self.save_game(label="autosave-shutdown")
                if not save_result.get("error"):
                    logger.info(
                        f"\u2705 AgentMon: game saved on shutdown "
                        f"(step {self._step_count})"
                    )
                else:
                    logger.warning(f"AgentMon: save on shutdown failed: {save_result.get('error')}")
                await self.stop_game()
            except Exception as e:
                logger.warning(f"AgentMon: cleanup save/stop failed: {e}")
        self._game_active = False
        if self._session and not self._session.closed:
            await self._session.close()
        self._session = None
        self._available = False

    # ── Auto-resume ───────────────────────────────────────────────────────

    async def _auto_resume(self) -> None:
        """Resume from latest save or start fresh. Clear startup dialogue."""
        try:
            saves = await self.list_saves()
            save_list = saves if isinstance(saves, list) else saves.get("saves", saves.get("data", []))
            if isinstance(save_list, list) and save_list:
                latest = save_list[0]
                save_id = latest.get("sessionId") or latest.get("id") or latest.get("_id", "")
                label = latest.get("label", save_id[:12] if save_id else "?")
                if save_id:
                    logger.info(f"AgentMon: resuming from save \"{label}\"")
                    result = await self.start_game(load_session_id=save_id)
                    if not result.get("error"):
                        self._game_active = True
                        state = await self._clear_startup_dialogue()
                        party = state.get("party", [])
                        party_sz = len(party) if isinstance(party, list) else state.get("partySize", 0)
                        badges = state.get("badges", 0)
                        pokedex = state.get("pokedexOwned", 0)
                        # Corrupt state detection: impossible combinations
                        if self._is_corrupt_state(state):
                            logger.warning(
                                f"AgentMon: corrupt save detected "
                                f"(badges={badges}, party={party_sz}, pokedex={pokedex}). "
                                f"Deleting save and starting fresh..."
                            )
                            await self.stop_game()
                            try:
                                await self.delete_save(save_id)
                            except Exception:
                                pass
                            # Fall through to fresh start below
                        else:
                            logger.info(
                                f"\u2705 AgentMon: resumed \u2014 "
                                f"{state.get('mapName', '?')} ({state.get('x')},{state.get('y')}) | "
                                f"Badges: {badges} | "
                                f"Party: {party_sz}"
                            )
                            return
                    logger.warning(f"AgentMon: resume failed ({result.get('error')}), starting fresh")

            # No saves or resume failed — start fresh
            logger.info("AgentMon: starting fresh game (has_pokedex checkpoint)")
            result = await self.start_game()
            if not result.get("error"):
                self._game_active = True
                state = await self._clear_startup_dialogue()
                party = state.get("party", [])
                party_sz = len(party) if isinstance(party, list) else 0
                logger.info(
                    f"\u2705 AgentMon: fresh game started \u2014 "
                    f"{state.get('mapName', '?')} ({state.get('x')},{state.get('y')}) | "
                    f"Party: {party_sz}"
                )
            else:
                logger.warning(f"AgentMon: start new game failed: {result.get('error')}")
        except Exception as e:
            logger.warning(f"AgentMon: auto-resume failed: {e}")

    @staticmethod
    def _is_corrupt_state(state: Dict) -> bool:
        """Detect obviously corrupt game states.

        Signs of corruption:
        - Party members with 0 maxHp (invalid Pokémon)
        - Pokedex count wildly mismatched to badges
        - Position at (0, 57+) which is out-of-bounds
        """
        party = state.get("party", [])
        if isinstance(party, list) and party:
            # All party members have 0/0 HP → corrupt
            if all(p.get("maxHp", 0) == 0 and p.get("level", 0) <= 1 for p in party):
                return True
        badges = state.get("badges", 0)
        pokedex = state.get("pokedexOwned", 0)
        # Pokedex 20+ with 0 badges is impossible in normal play
        if pokedex >= 20 and badges == 0:
            return True
        # Party size 6 with 0 badges at game start (impossible)
        if isinstance(party, list) and len(party) >= 6 and badges == 0:
            return True
        return False

    async def _clear_startup_dialogue(self) -> Dict:
        """After starting/resuming, clear pending dialogue so the player can move.

        Strategy: alternate B presses with movement attempts.
        B closes dialogue; A re-opens it when near NPCs.
        Max 20 rounds.  Each round = B + try all 4 directions.
        """
        logger.info("AgentMon: clearing startup dialogue (B-first strategy)...")
        state = await self.get_state()
        directions = ["down", "left", "right", "up"]

        for attempt in range(20):
            # Press B to CLOSE/CANCEL any open dialogue or menu
            await self.step("b")
            await asyncio.sleep(0.15)

            # Now try every direction to see if we can actually move
            for d in directions:
                result = await self.step(d)
                effects = result.get("feedback", {}).get("effects", [])
                if "moved" in effects:
                    state = result.get("state", {})
                    logger.info(
                        f"AgentMon: dialogue cleared ({attempt} rounds) — "
                        f"moved {d} to ({state.get('x')},{state.get('y')})"
                    )
                    return state

            # Still blocked everywhere — press A once to advance dialogue text
            # (only if B didn't help)
            if attempt % 3 == 2:
                await self.step("a")
                await asyncio.sleep(0.15)

        # Could not clear after 20 attempts — return current state
        state = await self.get_state()
        logger.warning(
            f"AgentMon: dialogue not fully cleared — "
            f"{state.get('mapName', '?')} ({state.get('x')},{state.get('y')}). "
            f"Play loop will handle it."
        )
        return state


    # ── Background play loop ──────────────────────────────────────────────

    def set_llm_callback(self, fn: Callable):
        """Set the LLM callback for intelligent gameplay decisions.

        The callback signature: async fn(game_state: dict) -> list[str]
        Should return a list of button presses (e.g. ["up", "a", "right"]).
        """
        self._llm_callback = fn
        logger.info("AgentMon: LLM callback registered — intelligent play enabled")

    # ── Vision (VLM) helpers ──────────────────────────────────────────

    async def _resolve_vlm(self) -> Optional[str]:
        """Pick the first available VLM from Ollama."""
        if self._vlm_model:
            return self._vlm_model
        try:
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=8)
            ) as s:
                async with s.get(f"{self._ollama_url}/api/tags") as r:
                    data = await r.json()
            names = [m["name"] for m in data.get("models", [])]
            for pref in self._vlm_models:
                for n in names:
                    if pref.lower() in n.lower():
                        self._vlm_model = n
                        logger.info(f"AgentMon: VLM selected → {n}")
                        return n
            # Fallback: anything with 'vl', 'vision', 'gemma3'
            for n in names:
                if any(k in n.lower() for k in ("vl", "vision", "gemma3", "llava", "minicpm")):
                    self._vlm_model = n
                    logger.info(f"AgentMon: VLM fallback → {n}")
                    return n
            logger.warning("AgentMon: no VLM found in Ollama")
            return None
        except Exception as e:
            logger.warning(f"AgentMon: VLM resolution failed: {e}")
            return None

    async def _analyze_screen(self, state: Dict, hint: str = "") -> str:
        """Capture game frame → send to Ollama VLM → return description.

        Returns a short text description of the screen, or empty string
        on any failure (so callers can always safely use the result).
        """
        model = await self._resolve_vlm()
        if not model:
            return ""

        # Get the game frame as PNG bytes
        try:
            frame_bytes = await self.get_frame()
        except Exception as e:
            logger.debug(f"AgentMon: get_frame() failed: {e}")
            return ""
        if not frame_bytes:
            return ""

        img_b64 = base64.b64encode(frame_bytes).decode()

        map_name = state.get("mapName", "unknown")
        x, y = state.get("x", 0), state.get("y", 0)
        in_battle = state.get("inBattle", False)

        prompt = (
            "You are an AI playing Pokémon Red on Game Boy. "
            f"Current state: map={map_name}, pos=({x},{y}), inBattle={in_battle}. "
            f"{hint} "
            "Look at this screenshot and answer CONCISELY (max 3 sentences):\n"
            "1) What do you see? (battle screen, overworld, menu, dialogue box, building interior, outdoor route)\n"
            "2) Any visible text or dialogue? Quote it exactly.\n"
            "3) If overworld: which direction should the player move to progress? "
            "(up/down/left/right) and why?\n"
            "Be specific and brief."
        )

        try:
            payload = {
                "model": model,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt,
                        "images": [img_b64],
                    }
                ],
                "stream": False,
                "options": {"temperature": 0.2, "num_predict": 200},
            }
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=60)
            ) as s:
                async with s.post(
                    f"{self._ollama_url}/api/chat", json=payload
                ) as r:
                    if r.status != 200:
                        body = await r.text()
                        logger.warning(f"AgentMon: VLM returned {r.status}: {body[:120]}")
                        return ""
                    result = await r.json()

            desc = result.get("message", {}).get("content", "").strip()
            if desc:
                self._last_vision = desc
                logger.info(f"👁️ AgentMon VLM: {desc[:160]}")
            return desc
        except asyncio.TimeoutError:
            logger.warning("AgentMon: VLM call timed out (60s)")
            return ""
        except Exception as e:
            logger.warning(f"AgentMon: VLM call failed: {e}")
            return ""

    def _start_play_loop(self):
        """Launch the background gameplay task."""
        if self._play_task and not self._play_task.done():
            return  # already running
        self._play_task = asyncio.create_task(self._play_loop())
        logger.info(
            f"\U0001f3ae AgentMon: background play loop started "
            f"(interval={self._play_interval}s)"
        )

    # ================================================================
    #  PLAY LOOP — feedback-driven, one step at a time
    # ================================================================
    #
    #  The API returns feedback.effects on EVERY step:
    #    moved, blocked, hit_wall_or_obstacle,
    #    battle_started, wild_encounter, trainer_battle,
    #    battle_ended, map_changed, entered_<MapName>,
    #    advanced_dialogue_or_selection, menu_opened,
    #    start_menu, confirmed, cancelled, closed_menu_or_back,
    #    waited, no_change, unknown_effect, ...
    #
    #  Plus screenText (OCR of the screen) and optionally
    #  a base64 PNG frame.
    #
    #  We send ONE action per step() call and read the feedback
    #  to know exactly what happened — no more blind guessing.
    # ================================================================

    _REVERSE = {"up": "down", "down": "up", "left": "right", "right": "left"}
    _DIR_SET = {"up", "down", "left", "right"}

    async def _play_loop(self):
        """Continuous background gameplay — one step at a time with feedback."""
        consecutive_errors = 0
        turns_played = 0

        try:
            # Get initial state (with retry)
            for _init_try in range(3):
                logger.info("AgentMon: play loop fetching initial state...")
                state = await self.get_state()
                if not state.get("error"):
                    break
                logger.warning(
                    f"AgentMon: initial state attempt {_init_try+1}/3 failed: "
                    f"{state.get('error')}"
                )
                if _init_try < 2:
                    await self._auto_resume()
                    await asyncio.sleep(5)

            if state.get("error"):
                logger.error(
                    f"AgentMon: play loop cannot start \u2014 state error: {state.get('error')}"
                )
                return

            # Clear any startup dialogue before exploring
            state = await self._clear_startup_dialogue()

            logger.info(
                f"AgentMon: play loop entering main loop — "
                f"{state.get('mapName', '?')} ({state.get('x')},{state.get('y')}) | "
                f"avail={self._available} active={self._game_active}"
            )

            while self._available and self._game_active:
                try:
                    # -- 1. Decide single action ---
                    action = self._decide_action(state)

                    # -- 2. Execute ONE step ---
                    result = await self.step(action)
                    turns_played += 1

                    # Navigation decision log
                    if turns_played <= 5 or turns_played % 10 == 0:
                        _nav = self._get_nav_for_map(state)
                        _fx = result.get('feedback', {}).get('effects', [])[:2]
                        _mid = state.get('mapId', 0)
                        logger.info(
                            f"AgentMon: [{turns_played}] "
                            f"{state.get('mapName','?')} ({state.get('x')},{state.get('y')}) "
                            f"→ '{action}' fx={_fx} | nav={_nav['dir'] if _nav else 'explore'} "
                            f"| {self._world_map.stats(_mid)}"
                        )

                    if result.get("error"):
                        err_msg = str(result.get("error", ""))
                        consecutive_errors += 1
                        if consecutive_errors <= 3 or consecutive_errors % 10 == 0:
                            logger.warning(
                                f"AgentMon: step error #{consecutive_errors}: {err_msg[:120]}"
                            )
                        if "session" in err_msg.lower():
                            logger.info("AgentMon: session lost, resuming...")
                            await self._auto_resume()
                            state = await self.get_state()
                        if consecutive_errors > 10:
                            await asyncio.sleep(30)
                        else:
                            await asyncio.sleep(self._play_interval)
                        continue

                    consecutive_errors = 0

                    # -- 3. Parse feedback ---
                    new_state = result.get("state", {})
                    feedback = result.get("feedback", {})
                    effects = feedback.get("effects", [])
                    fb_msg = feedback.get("message", "")
                    screen_text = result.get("screenText", "")

                    self._last_effects = effects
                    self._last_screen_text = screen_text
                    self._last_feedback_msg = fb_msg

                    # -- 3b. Update NPC positions from localMap ---
                    local_map = new_state.get("localMap") or {}
                    if local_map:
                        npcs = local_map.get("npcs") or []
                        nmid = new_state.get("mapId", 0)
                        self._npc_positions[nmid] = {
                            (npc["gridX"], npc["gridY"])
                            for npc in npcs
                            if isinstance(npc, dict)
                               and "gridX" in npc and "gridY" in npc
                        }

                    # -- 4. Learn from feedback ---
                    self._process_feedback(action, state, new_state, effects)

                    # -- 5. Update state flags ---
                    self._in_battle = bool(new_state.get("inBattle"))
                    self._in_dialogue = "advanced_dialogue_or_selection" in effects
                    self._in_menu = any(
                        e in effects for e in ("menu_opened", "start_menu")
                    )
                    if any(e in effects for e in (
                        "cancelled", "closed_menu_or_back"
                    )):
                        self._in_menu = False

                    # -- 6. Save step to short-term memory ---
                    pos = (
                        new_state.get("mapId", 0),
                        new_state.get("x", 0),
                        new_state.get("y", 0),
                    )
                    self._step_memory.append({
                        "action": action,
                        "effects": effects,
                        "pos": pos,
                        "map": new_state.get("mapName", ""),
                        "screen": screen_text[:100] if screen_text else "",
                        "fb": fb_msg,
                    })
                    if len(self._step_memory) > self._step_memory_max:
                        self._step_memory = self._step_memory[-self._step_memory_max:]

                    # -- 7. Use new state for next turn ---
                    state = new_state

                    # -- 7b. Vision (DISABLED — too slow, wastes turns) ---
                    # VLM calls take 20-30s each and don't help navigation.
                    # Keeping the code but skipping execution.

                    # -- 8. Progress logging ---
                    if turns_played % 25 == 0:
                        party = state.get("party", [])
                        party_str = ", ".join(
                            f"L{p.get('level', '?')}" for p in party[:6]
                        ) if party else "none"
                        _nav = self._get_nav_for_map(state)
                        _nav_str = f"→{_nav['dir']}" if _nav else "explore"
                        _mid = state.get('mapId', 0)
                        _map_stats = self._world_map.stats(_mid)
                        logger.info(
                            f"\U0001f3ae AgentMon: turn {turns_played} -- "
                            f"{state.get('mapName', '?')} ({state.get('x')},{state.get('y')}) | "
                            f"Nav: {_nav_str} | "
                            f"Badges: {state.get('badges', 0)} | "
                            f"Party: [{party_str}] | "
                            f"Map: {_map_stats} | "
                            f"BldgT: {self._building_turns} | "
                            f"Dlg: {self._consecutive_dialogue_a} | "
                            f"Last: {action}"
                        )
                        # Save world map periodically
                        try:
                            self._world_map.save(str(_WORLD_MAP_PATH))
                        except Exception as e:
                            logger.debug(f"AgentMon: map save failed: {e}")

                    # -- 8b. Stuck detector ---
                    # If position hasn't changed in many non-dialogue steps,
                    # try to recover WITHOUT destroying progress.
                    # NEVER delete saves. NEVER reset the world map.
                    if turns_played % 10 == 0 and len(self._step_memory) >= 40:
                        # Only count steps where agent was NOT in dialogue/battle
                        # (dialogue steps are normal, not evidence of being stuck)
                        recent = self._step_memory[-40:]
                        movement_positions = {
                            s["pos"] for s in recent
                            if not any(e in s.get("effects", []) for e in (
                                "advanced_dialogue_or_selection", "confirmed",
                                "cancelled", "closed_menu_or_back",
                                "menu_opened", "start_menu",
                            ))
                        }
                        # Only trigger if we have very few unique positions
                        # AND most steps were NOT dialogue
                        dialogue_steps = sum(
                            1 for s in recent
                            if any(e in s.get("effects", []) for e in (
                                "advanced_dialogue_or_selection", "confirmed",
                            ))
                        )
                        if len(movement_positions) <= 2 and dialogue_steps < 20:
                            logger.warning(
                                f"AgentMon: STUCK for 40+ movement turns at "
                                f"{state.get('mapName','?')} ({state.get('x')},{state.get('y')}). "
                                f"Attempting recovery (KEEPING map knowledge)..."
                            )
                            # Strategy: Load last save instead of nuking everything.
                            # The WorldMap is PRESERVED — we never lose map knowledge.
                            recovered = False
                            try:
                                saves = await self.list_saves()
                                save_list = (saves if isinstance(saves, list)
                                             else saves.get("saves", saves.get("data", [])))
                                if isinstance(save_list, list) and save_list:
                                    # Load the MOST RECENT save (don't delete!)
                                    sv = save_list[0]
                                    sid = (sv.get("sessionId") or sv.get("id")
                                           or sv.get("_id", ""))
                                    if sid:
                                        await self.stop_game()
                                        load_result = await self.start_game(load_session_id=sid)
                                        if not load_result.get("error"):
                                            self._step_memory.clear()
                                            self._consecutive_blocked = 0
                                            self._consecutive_dialogue_a = 0
                                            state = await self._clear_startup_dialogue()
                                            logger.info(
                                                f"AgentMon: recovered by loading save → "
                                                f"{state.get('mapName','?')} ({state.get('x')},{state.get('y')})"
                                            )
                                            recovered = True
                            except Exception as e:
                                logger.debug(f"AgentMon: save load failed: {e}")

                            if not recovered:
                                # No saves to load — try clearing walls and
                                # escaping via B + directions (soft recovery)
                                logger.info("AgentMon: no saves to load, soft recovery — clearing walls + escaping")
                                mid = state.get("mapId", 0)
                                pos = (state.get("x", 0), state.get("y", 0))
                                self._walls.get(mid, {}).pop(pos, None)
                                tile = self._world_map.tile(mid, pos[0], pos[1])
                                tile["walls"].clear()
                                self._step_memory.clear()
                                self._consecutive_blocked = 0
                                self._consecutive_dialogue_a = 0

                    # -- 9. Delay ---
                    await asyncio.sleep(self._play_interval)

                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    consecutive_errors += 1
                    logger.warning(f"AgentMon: play loop error ({type(e).__name__}): {e}")
                    await asyncio.sleep(min(self._play_interval * 2, 30))

        except asyncio.CancelledError:
            logger.info("AgentMon: play loop cancelled (shutdown)")
        except BaseException as e:
            logger.error(f"AgentMon: play loop FATAL ({type(e).__name__}): {e}")
        finally:
            logger.info(
                f"AgentMon: play loop exited "
                f"(turns={turns_played}, steps={self._step_count}, "
                f"avail={self._available}, active={self._game_active})"
            )

    # ================================================================
    #  FEEDBACK PROCESSING -- learn walls, opens, exits from effects
    # ================================================================

    def _process_feedback(
        self, action: str, old_state: Dict, new_state: Dict,
        effects: List[str],
    ) -> None:
        """Update wall/open map and tracking from step feedback effects."""
        old_mid = old_state.get("mapId", 0)
        old_x = old_state.get("x", 0)
        old_y = old_state.get("y", 0)
        new_mid = new_state.get("mapId", 0)
        new_x = new_state.get("x", 0)
        new_y = new_state.get("y", 0)

        is_direction = action in self._DIR_SET
        new_pos = (new_mid, new_x, new_y)
        wm = self._world_map
        wm.tick()

        # Track visited tiles
        self._visited.setdefault(new_mid, set()).add((new_x, new_y))
        self._prev_pos = new_pos
        wm.visit(new_mid, new_x, new_y, new_state.get("mapName", ""))

        # Track recent maps (anti-ping-pong)
        if not self._recent_maps or self._recent_maps[-1] != new_mid:
            self._recent_maps.append(new_mid)
            if len(self._recent_maps) > 30:
                self._recent_maps = self._recent_maps[-15:]
            # Map changed → beeline exit succeeded, reset tracker
            self._beeline_attempts = 0
            self._beeline_exit_key = None

        # -- Wall detection (from feedback) ---
        if is_direction and any(
            e in effects for e in ("blocked", "hit_wall_or_obstacle")
        ):
            # Check if an NPC is blocking this direction.
            # NPCs move, so NPC-blocked directions are TEMPORARY — do NOT
            # record them as permanent walls (prevents map corruption).
            dx = {"left": -1, "right": 1, "up": 0, "down": 0}.get(action, 0)
            dy = {"left": 0, "right": 0, "up": -1, "down": 1}.get(action, 0)
            dest_pos = (old_x + dx, old_y + dy)
            npc_set = self._npc_positions.get(old_mid, set())
            if dest_pos in npc_set:
                logger.debug(
                    f"AgentMon: blocked by NPC at {dest_pos}, "
                    f"NOT recording as wall"
                )
                self._consecutive_blocked += 1
                return

            # Real wall — record it
            self._walls.setdefault(old_mid, {}).setdefault(
                (old_x, old_y), set()
            ).add(action)
            wm.wall(old_mid, old_x, old_y, action)
            self._consecutive_blocked += 1
            return

        # -- Successful movement ---
        if "moved" in effects and is_direction:
            self._consecutive_blocked = 0
            self._consecutive_dialogue_a = 0  # moving means dialogue is over
            if old_mid == new_mid:
                self._opens.setdefault(old_mid, {}).setdefault(
                    (old_x, old_y), {}
                )[action] = (new_x, new_y)
                # Record forward direction only (no reverse — it corrupts data)
                wm.open_dir(old_mid, old_x, old_y, action, (new_x, new_y))
                # Also remove any stale wall for this direction
                # (we just moved there, so it's not a wall)
                wm.tile(old_mid, old_x, old_y)["walls"].discard(action)

        # -- Map transition ---
        if "map_changed" in effects or any(
            e.startswith("entered_") for e in effects
        ):
            if is_direction:
                self._exits[(old_mid, old_x, old_y, action)] = new_mid
                self._opens.setdefault(old_mid, {}).setdefault(
                    (old_x, old_y), {}
                )[action] = None  # cross-map
                wm.open_dir(old_mid, old_x, old_y, action, None)
                wm.record_exit(old_mid, old_x, old_y, action, new_mid)
            self._transitions += 1
            self._consecutive_blocked = 0
            self._building_turns = 0  # reset building counter on map change

            # Track warp tiles: the tile we stepped on triggered a map change.
            self._warp_tiles.setdefault(old_mid, {})[(old_x, old_y)] = new_mid
            wm.record_warp(old_mid, old_x, old_y, new_mid)

            logger.info(
                f"\U0001f3ae AgentMon: -> {new_state.get('mapName', '?')} "
                f"via '{action}' from ({old_x},{old_y}) "
                f"[warp ({old_x},{old_y})→map{new_mid} | {wm.stats(new_mid)}]"
            )

        # -- Dialogue/menu -- reset blocked counter ---
        if any(e in effects for e in (
            "advanced_dialogue_or_selection", "confirmed",
            "menu_opened", "start_menu",
        )):
            self._consecutive_blocked = 0

    # ================================================================
    #  NAVIGATION — map-aware directional agent brain
    # ================================================================

    def _get_nav_for_map(self, state: Dict) -> Optional[Dict[str, str]]:
        """Get navigation direction for the current map, considering badges.

        Returns {"dir": primary, "alt": fallback} or None for unknown maps.
        Uses badge-based overrides for cities (post-gym progression).
        """
        map_name = state.get("mapName", "").lower()
        badges = state.get("badges", 0)

        # Badge-based city overrides (redirect after beating local gym)
        if "pewter" in map_name and "gym" not in map_name and badges >= 1:
            return {"dir": "right", "alt": "up"}
        if "cerulean" in map_name and "gym" not in map_name and badges >= 2:
            return {"dir": "down", "alt": "right"}
        if "vermilion" in map_name and "gym" not in map_name and badges >= 3:
            return {"dir": "right", "alt": "up"}
        if "celadon" in map_name and "gym" not in map_name and badges >= 4:
            return {"dir": "right", "alt": "down"}
        if "fuchsia" in map_name and "gym" not in map_name and badges >= 5:
            return {"dir": "up", "alt": "right"}
        if "saffron" in map_name and "gym" not in map_name and badges >= 6:
            return {"dir": "down", "alt": "right"}

        # Gym exit: already beaten this gym → leave south
        for gym_name, badge_num in _GYM_BADGE_MAP.items():
            if gym_name in map_name and badges >= badge_num:
                return {"dir": "down", "alt": "right"}

        # Static nav lookup (longest pattern match wins)
        nav = None
        best_len = 0
        for pattern, data in NAVIGATION_MAP.items():
            if pattern in map_name and len(pattern) > best_len:
                nav = data
                best_len = len(pattern)
        return nav

    def _decide_action(self, state: Dict) -> str:
        """Choose the next action.

        Priority:
          1. In battle → fight / run
          2. Dialogue escape mode (stuck pressing A) → aggressive B + move
          3. Active dialogue (short) → A to advance, max 3 times
          4. Menu open → B to close
          5. All-4-sides blocked → B to dismiss, clear fake walls
          6. Navigate toward objective
        """
        # -- 1. Battle --
        if state.get("inBattle"):
            return self._battle_decide(state)

        # -- 2. DIALOGUE ESCAPE MODE --
        # If we've pressed A 5+ times in dialogue without the position
        # changing, we're stuck in an NPC dialogue loop.
        #
        # Strategy: aggressive B-spam + direction attempts to escape.
        # Pokémon dialogues can be long (10+ A presses for trainer encounters),
        # so we allow more A presses before switching to escape mode.
        if self._consecutive_dialogue_a >= 8:
            esc_phase = (self._consecutive_dialogue_a - 8)
            cycle_pos = esc_phase % 6  # 0=B, 1=B, 2=B, 3=dir, 4=B, 5=dir
            self._consecutive_dialogue_a += 1

            # Clear stale wall data every cycle (dialogue blocks aren't real walls)
            if cycle_pos == 0:
                map_id = state.get("mapId", 0)
                pos = (state.get("x", 0), state.get("y", 0))
                self._walls.get(map_id, {}).pop(pos, None)
                self._consecutive_blocked = 0

            if cycle_pos in (0, 1, 2, 4):
                return "b"
            else:
                # Try directions: down, left, right, up (down first to flee)
                dirs = ["down", "left", "right", "up"]
                d = dirs[(esc_phase // 6) % 4]
                return d

        # -- 3. Active dialogue (short) → A to advance --
        # Allow pressing A for dialogue, but cap it so we eventually escape.
        if self._in_dialogue:
            self._consecutive_dialogue_a += 1
            # After 15+ dialogue A-presses, force B to try to exit
            if self._consecutive_dialogue_a >= 15 and self._consecutive_dialogue_a % 3 == 0:
                return "b"
            return "a"

        # -- 3b. Screen text with choice keywords → A --
        screen = self._last_screen_text.strip() if self._last_screen_text else ""
        if screen:
            screen_lower = screen.lower()
            if any(kw in screen_lower for kw in (
                "fight", "bag", "run", "pkmn", "yes", "no",
                "which", "choose", "want to", "nickname",
            )):
                return "a"

        # Reset dialogue counter when not in dialogue
        if not self._in_dialogue:
            self._consecutive_dialogue_a = 0

        # -- 4. Menu → close --
        if self._in_menu:
            return "b"

        # -- 5. All 4 directions blocked → hidden dialogue/cutscene --
        if self._consecutive_blocked >= 4:
            map_id = state.get("mapId", 0)
            pos = (state.get("x", 0), state.get("y", 0))
            # Clear fake walls from dialogue-blocked state
            self._walls.get(map_id, {}).pop(pos, None)
            self._consecutive_blocked = 0
            # Press B to dismiss (NOT A — A re-triggers NPC dialogue)
            return "b"

        # -- 6. Navigate --
        action = self._navigate(state)

        # Common vars for NPC avoidance + warp safety net
        map_id = state.get("mapId", 0)
        x, y = state.get("x", 0), state.get("y", 0)
        wm = self._world_map
        nav = self._get_nav_for_map(state)
        is_building = nav and nav["dir"] == "down"

        # -- 6b. NPC AVOIDANCE --
        # Don't walk into NPCs — they block movement and cause false wall
        # recordings. Pick the best alternative direction instead.
        npc_set = self._npc_positions.get(map_id, set())
        if action in self._DIR_SET and npc_set:
            _dx = self._DX.get(action, 0)
            _dy = self._DY.get(action, 0)
            _dest = (x + _dx, y + _dy)
            if _dest in npc_set:
                cur_tile = wm.tile(map_id, x, y)
                for _alt in self._DIR_SET:
                    if _alt == action:
                        continue
                    _adx = self._DX.get(_alt, 0)
                    _ady = self._DY.get(_alt, 0)
                    _adest = (x + _adx, y + _ady)
                    if _adest in npc_set:
                        continue
                    # Check it's not a known wall (unless opens override)
                    if _alt in cur_tile["opens"] or _alt not in cur_tile["walls"]:
                        logger.debug(
                            f"AgentMon: NPC at {_dest}, redirecting "
                            f"'{action}'→'{_alt}'"
                        )
                        action = _alt
                        break

        # -- 7. WARP SAFETY NET --
        # ABSOLUTE LAST DEFENSE: never step onto a known warp tile
        # on overworld maps. This catches any bugs in the navigation.

        if not is_building and action in self._DIR_SET:
            cur_tile = wm.tile(map_id, x, y)
            # Check if that direction leads to a known warp tile
            dest = cur_tile["opens"].get(action)
            if dest is not None:
                dt = wm._tiles.get(map_id, {}).get(dest, {})
                if dt and dt.get("warp") is not None:
                    logger.info(
                        f"AgentMon: WARP SAFETY blocked '{action}' from ({x},{y})"
                        f" → warp {dest}"
                    )
                    # Pick any other direction that doesn't lead to warp
                    for d in self._DIR_SET:
                        if d == action or d in cur_tile["walls"]:
                            continue
                        d_dest = cur_tile["opens"].get(d)
                        if d_dest is not None:
                            d_dt = wm._tiles.get(map_id, {}).get(d_dest, {})
                            if d_dt and d_dt.get("warp") is not None:
                                continue
                        action = d
                        break

            # Extra: if we're ON a warp tile, flee immediately
            if cur_tile.get("warp") is not None:
                for d in ("down", "left", "right", "up"):
                    if d not in cur_tile["walls"]:
                        d_dest = cur_tile["opens"].get(d)
                        if d_dest is not None:
                            d_dt = wm._tiles.get(map_id, {}).get(d_dest, {})
                            if d_dt and d_dt.get("warp") is not None:
                                continue
                        # Don't go in a direction that's a known exit
                        if d in cur_tile.get("exits", {}):
                            continue
                        action = d
                        break

        return action

    def _battle_decide(self, state: Dict) -> str:
        """Battle action using screen text context.

        Default: press A (FIGHT → first move → advance text).
        Low HP + wild battle: attempt RUN (down → right → A).
        """
        party = state.get("party", [])
        screen = self._last_screen_text.lower() if self._last_screen_text else ""

        hp_critical = False
        if party:
            lead = party[0]
            hp = lead.get("hp", 1)
            max_hp = lead.get("maxHp", 1)
            if max_hp > 0 and hp / max_hp < 0.20:
                hp_critical = True

        if hp_critical and any(w in screen for w in ("fight", "bag", "run")):
            recent = [s["action"] for s in self._step_memory[-3:]]
            if not recent or recent[-1] not in ("down", "right"):
                return "down"
            if recent[-1] == "down":
                return "right"
            if recent[-1] == "right":
                return "a"

        return "a"

    # ================================================================
    #  A* PATHFINDING — now delegated to WorldMap
    # ================================================================

    _DX = {"left": -1, "right": 1, "up": 0, "down": 0}
    _DY = {"left": 0, "right": 0, "up": -1, "down": 1}

    def _navigate(self, state: Dict) -> str:
        """Navigate using WorldMap — simple and robust.

        On overworld (bias != 'down'):
          - Use best_step with avoid_all_warps=True → NEVER walk onto a warp tile.
          - This is just: pick the least-visited non-wall non-warp neighbor,
            preferring the bias direction and unexplored directions.

        In buildings (bias == 'down'):
          - If we know the exit tile, beeline to it.
          - Otherwise use best_step toward 'down'.
        """
        map_id = state.get("mapId", 0)
        x, y = state.get("x", 0), state.get("y", 0)
        wm = self._world_map

        nav = self._get_nav_for_map(state)
        bias = nav["dir"] if nav else None
        is_building = bias == "down"

        # ── Building turn tracking ────────────────────────────────────
        if is_building:
            if map_id != self._building_map_id:
                self._building_turns = 0
                self._building_map_id = map_id
            self._building_turns += 1
        else:
            self._building_turns = 0

        # ── BUILDING: beeline to known exit ───────────────────────────
        if is_building:
            exit_info = wm.known_exit_tile(map_id)
            if exit_info:
                exit_pos, exit_dir = exit_info
                exit_key = (map_id, exit_pos[0], exit_pos[1], exit_dir)

                # Stale exit detection
                if (self._beeline_exit_key == exit_key
                        and self._beeline_attempts >= 5):
                    logger.warning(
                        f"AgentMon: INVALIDATING stale exit {exit_pos} "
                        f"dir={exit_dir} after {self._beeline_attempts} attempts"
                    )
                    t_exit = wm.tile(map_id, exit_pos[0], exit_pos[1])
                    t_exit.get("exits", {}).pop(exit_dir, None)
                    t_exit["warp"] = None
                    wm._warps.pop((map_id, exit_pos[0], exit_pos[1]), None)
                    self._beeline_attempts = 0
                    self._beeline_exit_key = None
                else:
                    if (x, y) == exit_pos:
                        if self._beeline_exit_key != exit_key:
                            self._beeline_exit_key = exit_key
                            self._beeline_attempts = 0
                        self._beeline_attempts += 1
                        return exit_dir
                    # Pathfind using only observed opens (no DX/DY guessing)
                    path = wm.pathfind(map_id, (x, y), exit_pos)
                    if path:
                        logger.info(
                            f"AgentMon: BEELINE to exit {exit_pos} "
                            f"(path={len(path)}, step={path[0]})"
                        )
                        return path[0]
                    # Pathfind failed — just use best_step toward down
                    logger.debug(
                        f"AgentMon: BEELINE pathfind failed to {exit_pos}, "
                        f"using best_step"
                    )

            # Building: best_step with bias=down, no warp avoidance
            step = wm.best_step(map_id, x, y, bias="down")
            if step:
                return step

        # ── OVERWORLD: best_step avoiding ALL warps ───────────────────
        step = wm.best_step(map_id, x, y, bias=bias, avoid_all_warps=True)
        if step:
            return step

        # ── Stuck: clear stale walls, try bias direction ──────────────
        tile = wm.tile(map_id, x, y)
        tile["walls"].clear()
        self._walls.get(map_id, {}).pop((x, y), None)
        self._consecutive_blocked = 0

        if bias:
            return bias
        return random.choice(list(self._DIR_SET))

    def _explore(self, state: Dict) -> str:
        """Fallback exploration for maps not in NAVIGATION_MAP."""
        map_id = state.get("mapId", 0)
        x, y = state.get("x", 0), state.get("y", 0)
        wm = self._world_map

        # Avoid all warp tiles
        step = wm.best_step(map_id, x, y, avoid_all_warps=True)
        if step:
            return step

        # Stuck — clear walls, random
        tile = wm.tile(map_id, x, y)
        tile["walls"].clear()
        self._walls.get(map_id, {}).pop((x, y), None)
        return random.choice(list(self._DIR_SET))


    async def _auto_register(self) -> Optional[Dict[str, Any]]:
        """Register with the AgentMon League API.

        Only requires the base URL — no pre-shared secrets needed.
        The API returns an apiKey + agentId on success.
        """
        try:
            agent_name = (
                getattr(self.config, "agent_name", "")
                or os.environ.get("AGENT_NAME", "")
                or "Sable"
            )
            display_name = (
                getattr(self.config, "agentmon_display_name", "")
                or os.getenv("AGENTMON_DISPLAY_NAME", "")
                or f"OpenSable — {agent_name}"
            )
            description = (
                getattr(self.config, "agentmon_description", "")
                or os.getenv(
                    "AGENTMON_DESCRIPTION",
                    f"Autonomous AI agent powered by OpenSable. "
                    f"I learn, adapt, and play to become the very best.",
                )
            )
            source_url = "https://opensable.com"

            payload = {
                "displayName": display_name,
                "description": description,
                "sourceUrl": source_url,
            }

            async with self._session.post(
                f"{self._base_url}/api/auth/local/register",
                json=payload,
                headers={
                    "Content-Type": "application/json",
                    "User-Agent": "OpenSable/1.0 (https://opensable.com)",
                },
            ) as r:
                if r.status >= 500:
                    body = await r.text()
                    logger.warning(
                        f"AgentMon League API returned {r.status} "
                        f"(server may be down): {body[:200] if body else '(empty)'}"
                    )
                    return None
                try:
                    data = await r.json()
                except Exception:
                    body = await r.text()
                    logger.warning(
                        f"AgentMon registration: non-JSON response ({r.status}): "
                        f"{body[:200] if body else '(empty)'}"
                    )
                    return None
                if r.status not in (200, 201) or not data.get("apiKey"):
                    logger.error(f"AgentMon registration failed ({r.status}): {data}")
                    return None

            creds = {
                "agentId": data["agentId"],
                "apiKey": data["apiKey"],
                "displayName": display_name,
                "description": description,
                "sourceUrl": source_url,
                "baseUrl": self._base_url,
                "autoRegistered": True,
            }

            logger.info(
                f"AgentMon: registered as {display_name} "
                f"(agentId={data['agentId'][:12]}…)"
            )

            # Set profile with avatar + description
            await self._set_profile(data["apiKey"], display_name, description)

            return creds

        except Exception as e:
            logger.error(f"AgentMon auto-registration failed: {e}")
            return None

    async def _set_profile(
        self, api_key: str, display_name: str, description: str = ""
    ) -> None:
        """Set display name, description, and avatar after registration."""
        try:
            avatar_url = (
                getattr(self.config, "agentmon_avatar_url", None)
                or os.getenv("AGENTMON_AVATAR_URL", "")
                or "https://opensable.com/images/sable-looking-front-head-only.png"
            )
            profile: Dict[str, str] = {
                "displayName": display_name,
                "avatarUrl": avatar_url,
            }
            if description:
                profile["description"] = description
            async with self._session.patch(
                f"{self._base_url}/api/agents/me",
                json=profile,
                headers={
                    "Content-Type": "application/json",
                    "X-Agent-Key": api_key,
                },
            ) as r:
                if r.status == 200:
                    logger.info(f"AgentMon: profile set — {display_name}")
        except Exception:
            pass  # non-critical

    # ── HTTP helpers ──────────────────────────────────────────────────────

    def _headers(self) -> Dict[str, str]:
        return {
            "Content-Type": "application/json",
            "User-Agent": "OpenSable/1.0 (https://opensable.com)",
            "X-Agent-Key": self._api_key,
        }

    async def _get(self, path: str, params: Optional[Dict] = None) -> Dict:
        if not self._session:
            return {"error": "session closed"}
        try:
            async with self._session.get(
                f"{self._base_url}{path}", headers=self._headers(), params=params,
            ) as r:
                return await r.json()
        except Exception as e:
            return {"error": str(e)}

    async def _post(self, path: str, body: Optional[Dict] = None) -> Dict:
        if not self._session:
            return {"error": "session closed"}
        try:
            async with self._session.post(
                f"{self._base_url}{path}", headers=self._headers(), json=body or {},
            ) as r:
                return await r.json()
        except Exception as e:
            return {"error": str(e)}

    async def _patch(self, path: str, body: Optional[Dict] = None) -> Dict:
        if not self._session:
            return {"error": "session closed"}
        try:
            async with self._session.patch(
                f"{self._base_url}{path}", headers=self._headers(), json=body or {},
            ) as r:
                return await r.json()
        except Exception as e:
            return {"error": str(e)}

    async def _delete(self, path: str) -> Dict:
        if not self._session:
            return {"error": "session closed"}
        try:
            async with self._session.delete(
                f"{self._base_url}{path}", headers=self._headers(),
            ) as r:
                return await r.json()
        except Exception as e:
            return {"error": str(e)}

    async def _get_bytes(self, path: str, params: Optional[Dict] = None) -> Optional[bytes]:
        """GET that returns raw bytes (for PNG frames)."""
        if not self._session:
            return None
        try:
            async with self._session.get(
                f"{self._base_url}{path}", params=params,
            ) as r:
                if r.status == 200:
                    return await r.read()
                return None
        except Exception:
            return None

    # ── Action throttle ───────────────────────────────────────────────────

    async def _throttle(self):
        """Wait between game actions to avoid spamming."""
        now = time.monotonic()
        elapsed = now - self._last_action
        if elapsed < self._action_delay:
            jitter = self._action_delay * random.uniform(-0.15, 0.15)
            wait = max(0, self._action_delay - elapsed + jitter)
            await asyncio.sleep(wait)
        self._last_action = time.monotonic()

    # ── Game Session ──────────────────────────────────────────────────────

    async def start_game(
        self, *, starter: Optional[str] = None,
        load_session_id: Optional[str] = None,
        speed: Optional[int] = None,
    ) -> Dict:
        """Start a new game or load a save."""
        body: Dict[str, Any] = {}
        if load_session_id:
            body["loadSessionId"] = load_session_id
        if starter:
            body["starter"] = starter
        body["speed"] = speed if speed is not None else self._speed
        result = await self._post("/api/game/emulator/start", body)
        if not result.get("error"):
            self._game_active = True
            self._step_count = 0
        return result

    async def stop_game(self) -> Dict:
        """Stop the current game session."""
        # Auto-save before stopping so nothing is lost
        if self._game_active:
            try:
                await self.save_game(label="autosave-stop")
            except Exception:
                pass
        result = await self._post("/api/game/emulator/stop")
        self._game_active = False
        return result

    # ── Actions ───────────────────────────────────────────────────────────

    async def step(self, action: str) -> Dict:
        """Send a single button press."""
        await self._throttle()
        action = action.lower().strip()
        if action not in VALID_ACTIONS:
            return {"error": f"Invalid action '{action}'. Valid: {', '.join(sorted(VALID_ACTIONS))}"}
        result = await self._post("/api/game/emulator/step", {"action": action})
        if not result.get("error"):
            self._step_count += 1
            await self._maybe_autosave()
        return result

    async def actions(self, action_list: List[str], speed: Optional[int] = None) -> Dict:
        """Send a sequence of button presses. Returns final state."""
        await self._throttle()
        cleaned = [a.lower().strip() for a in action_list]
        invalid = [a for a in cleaned if a not in VALID_ACTIONS]
        if invalid:
            return {"error": f"Invalid actions: {invalid}. Valid: {', '.join(sorted(VALID_ACTIONS))}"}
        body: Dict[str, Any] = {"actions": cleaned}
        if speed is not None:
            body["speed"] = speed
        else:
            body["speed"] = self._speed
        result = await self._post("/api/game/emulator/actions", body)
        if not result.get("error"):
            self._step_count += len(cleaned)
            await self._maybe_autosave()
        return result

    async def _maybe_autosave(self) -> None:
        """Auto-save every N steps, but only if the agent has moved recently.

        We skip saving if the last 10 steps are all at the same position,
        because saving a stuck/soft-locked state is counter-productive.
        """
        if self._autosave_interval <= 0:
            return
        if self._step_count % self._autosave_interval == 0:
            # Check: have we actually moved in recent steps?
            if len(self._step_memory) >= 10:
                recent_positions = {s["pos"] for s in self._step_memory[-10:]}
                if len(recent_positions) <= 1:
                    logger.debug("AgentMon: skipping auto-save (stuck)")
                    return
            try:
                result = await self.save_game(label=f"autosave-step{self._step_count}")
                if not result.get("error"):
                    logger.info(f"AgentMon: auto-saved at step {self._step_count}")
                    # Also save world map with each auto-save
                    try:
                        self._world_map.save(str(_WORLD_MAP_PATH))
                    except Exception:
                        pass
            except Exception as e:
                logger.debug(f"AgentMon: auto-save failed: {e}")

    # ── State & Screen ────────────────────────────────────────────────────

    async def get_state(self) -> Dict:
        """Get current game state (map, position, party, badges, inventory, etc.)."""
        return await self._get("/api/game/emulator/state")

    async def get_frame(self) -> Optional[bytes]:
        """Get current screen as PNG bytes."""
        return await self._get_bytes(
            "/api/observe/emulator/frame",
            params={"agentId": self._agent_id} if self._agent_id else None,
        )

    # ── Saves ─────────────────────────────────────────────────────────────

    async def save_game(self, label: Optional[str] = None) -> Dict:
        """Save the current game state."""
        body = {}
        if label:
            body["label"] = label
        return await self._post("/api/game/emulator/save", body)

    async def list_saves(self) -> Dict:
        """List all saves."""
        return await self._get("/api/game/emulator/saves")

    async def delete_save(self, save_id: str) -> Dict:
        """Delete a save."""
        return await self._delete(f"/api/game/emulator/saves/{save_id}")

    # ── Observer / Info ───────────────────────────────────────────────────

    async def get_leaderboard(self) -> Dict:
        """Get the leaderboard."""
        return await self._get("/api/observe/leaderboard")

    async def get_activity(self, limit: int = 30) -> Dict:
        """Get live activity feed across all agents."""
        return await self._get("/api/observe/activity", params={"limit": limit})

    # ── Profile ───────────────────────────────────────────────────────────

    async def update_profile(
        self, *, display_name: Optional[str] = None, avatar_url: Optional[str] = None,
    ) -> Dict:
        """Update agent display name and/or avatar."""
        body = {}
        if display_name is not None:
            body["displayName"] = display_name
        if avatar_url is not None:
            body["avatarUrl"] = avatar_url
        return await self._patch("/api/agents/me", body)

    # ── Experience (optional long-term memory) ────────────────────────────

    async def record_experience(
        self, state_before: Dict, action: str, state_after: Dict,
        step_index: Optional[int] = None,
    ) -> Dict:
        """Record a step for long-term learning."""
        body: Dict[str, Any] = {
            "stateBefore": state_before,
            "action": action,
            "stateAfter": state_after,
        }
        if step_index is not None:
            body["stepIndex"] = step_index
        return await self._post("/api/game/emulator/experience", body)

    async def get_experiences(self, limit: int = 50) -> Dict:
        """Retrieve recent experiences for learning context."""
        return await self._get("/api/game/emulator/experience", params={"limit": limit})
