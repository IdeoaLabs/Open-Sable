"""
Sable Gateway,  Internal Control Plane (aiohttp, Unix + TCP)

Architecture:
  ┌────────────────────────────────────────────────────────────┐
  │  /tmp/sable.sock  (Unix domain socket, mode 0600)         │
  │  ┌──────────────┐    ┌──────────────┐    ┌─────────────┐  │
  │  │  WebChat JS  │    │  CLI client  │    │  Nodes/IPC  │  │
  │  └──────┬───────┘    └──────┬───────┘    └──────┬──────┘  │
  │         └──────────────────┼───────────────────┘          │
  │                       Unix socket + TCP                    │
  │                   SableGateway (aiohttp)                   │
  │                       │                                    │
  │              ┌─────────┴──────────┐                       │
  │         SessionManager       SableAgent                   │
  └────────────────────────────────────────────────────────────┘

Security:
  - Unix socket: only the process owner can connect (OS enforces it)
  - chmod 0600 applied immediately after bind
  - Token auth via middleware,  HMAC-compare on ?token= query param
  - Remote access via SSH tunnel: ssh -L 8789:/tmp/sable.sock user@vps

Protocol (JSON over WebSocket, text frames):
  Client → Gateway
    {"type": "message",           "session_id": "...", "user_id": "...", "text": "..."}
    {"type": "command",           "session_id": "...", "user_id": "...", "text": "/status"}
    {"type": "sessions.list"}
    {"type": "sessions.history",  "session_id": "..."}
    {"type": "node.register",     "node_id": "...",   "capabilities": [...]}
    {"type": "node.invoke",       "node_id": "...",   "capability": "...", "args": {...}, "request_id": "..."}
    {"type": "node.result",       "request_id": "...", "output": "...", "reply_to": "..."}
    {"type": "ping"}

  Gateway → Client
    {"type": "connected",         "version": "2.1.0"}
    {"type": "message.start",     "session_id": "..."}
    {"type": "message.done",      "session_id": "...", "text": "..."}
    {"type": "command.result",    "session_id": "...", "text": "...", "success": bool}
    {"type": "sessions.list.result",    "sessions": [...]}
    {"type": "sessions.history.result", "session_id": "...", "messages": [...]}
    {"type": "node.registered",   "node_id": "...", "capabilities": [...]}
    {"type": "node.invoke",       "capability": "...", "args": {...}, "request_id": "...", "reply_to": "..."}
    {"type": "node.result",       "request_id": "...", "output": "..."}
    {"type": "error",             "text": "..."}
    {"type": "heartbeat",         "ts": float}
    {"type": "pong"}
    {"type": "status",            ...}
"""

from __future__ import annotations

import asyncio
import hmac
import json
import logging
import os
import re
import stat
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Set

import aiohttp
from aiohttp import web, WSMsgType

# ─── LLM reasoning-trace stripper ─────────────────────────────────────────────

logger = logging.getLogger(__name__)

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_THINK_OPEN = re.compile(r"<think>.*", re.DOTALL | re.IGNORECASE)

_REASONING_STARTERS = re.compile(
    r"^(system\b|i need to\b|let me\b|first,?\s+let me\b|okay,?\s+let me\b|"
    r"alright,?\s+let me\b|i\'ll craft\b|i will craft\b|i\'m going to\b|"
    r"the user (seems|is|wants|asked|might|may|said|appears|\'s message)\b|"
    r"i should\b|i\'ve been\b|maybe i(\'ll| will)\b|"
    r"so i (need|should|want|will)\b|now i\b|next,?\s+i\b|"
    r"looking at (the|this|their)\b|this (is|seems|looks|appears) (to be|like)\b|"
    r"they(\'re|\'ve been|\'ve| are| might be| seem| could be| want| may be| did| have)\b|"
    r"(he|she|it) (is|was|seems|wants|might|appears|\'s)\b|"
    r"my response should\b|i\'ll (acknowledge|address|respond|help|note|craft|keep|try|make|provide)\b|"
    r"i\'m (going|trying|not sure|looking|noticing|thinking)\b|"
    r"i (should|need to|will|must|can|notice|see that|recognize|understand)\b|"
    r"(alright|okay|ok|hmm),?\s+(let me|i)\b|"
    r"(?:not )?a (?:complaint|question|request|greeting|test|genuine)\b|"
    r"\(also,?\b|\(note[,:]|\(thinking|\(internal|\(context)",
    re.IGNORECASE,
)


def _strip_reasoning_preamble(text: str) -> str:
    """Remove raw untagged reasoning that Claude-distilled models emit."""
    if not text:
        return text

    paragraphs = re.split(r"\n{2,}", text)
    if len(paragraphs) <= 1:
        lines = text.splitlines()
        keep_from = 0
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped and _REASONING_STARTERS.match(stripped):
                keep_from = i + 1
            else:
                break
        if keep_from:
            return "\n".join(lines[keep_from:]).strip()
        return text

    cleaned: list[str] = []
    found_real = False
    for para in paragraphs:
        if found_real:
            cleaned.append(para)
            continue
        lines = [ln.strip() for ln in para.splitlines() if ln.strip()]
        if not lines:
            continue
        reasoning_count = sum(1 for ln in lines if _REASONING_STARTERS.match(ln))
        if reasoning_count == len(lines):
            logger.debug(f"🧹 Gateway stripped reasoning para: {para[:80]!r}")
            continue
        found_real = True
        cleaned.append(para)

    result = "\n\n".join(cleaned).strip()
    return result if result else text


def _clean_gateway_reply(text: str) -> str:
    """Strip any leaked <think>...</think> reasoning blocks before sending to client."""
    if not text:
        return text
    text = _THINK_RE.sub("", text)
    text = _THINK_OPEN.sub("", text)
    text = text.replace("</think>", "").strip()
    # Strip leaked role prefixes (llama/mistral chat template bleed)
    text = re.sub(
        r'^(?:assistant|asistente|user|sistema|system)\s*[:\n]+\s*',
        '', text, flags=re.IGNORECASE,
    )
    # Strip garbled BPE tokens after role prefix (e.g. "ungal\n\n")
    text = re.sub(r'^[a-z]{2,8}\n\n\s*', '', text, flags=re.IGNORECASE)
    text = _strip_reasoning_preamble(text)
    return text


# ─── Constants ────────────────────────────────────────────────────────────────

# Socket path is profile-aware: /tmp/sable-<profile>.sock for all profiles.
_profile_name = os.environ.get("_SABLE_PROFILE", "sable")
SOCKET_PATH = Path(os.environ.get("_SABLE_SOCKET_PATH", "/tmp/sable-sable.sock"))
_data_dir = Path(os.environ.get("_SABLE_DATA_DIR", "data"))
GATEWAY_VER = "2.1.0"
HEARTBEAT_INT = 30  # seconds between heartbeat frames


def _find_free_port() -> int:
    """Find a free TCP port by letting the OS assign one."""
    import socket as _sock
    with _sock.socket(_sock.AF_INET, _sock.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]

async def _proxy_ws_request(
    socket_path: str,
    payload: dict,
    expected_type: str | None = None,
    timeout: float = 10,
) -> dict:
    """Open a WebSocket to a remote agent, send *payload*, return the response.

    The remote gateway sends a ``connected`` welcome message on open.
    This helper drains that welcome frame before reading the real reply.
    If *expected_type* is given, it keeps reading until a message with that
    ``type`` arrives (up to *timeout* seconds total).
    """
    conn = aiohttp.UnixConnector(path=socket_path)
    async with aiohttp.ClientSession(connector=conn) as session:
        async with session.ws_connect("http://localhost/") as ws:
            # Drain the welcome / connected message
            try:
                welcome = await asyncio.wait_for(ws.receive_json(), timeout=3)
                logger.debug(f"[proxy] drained welcome: {welcome.get('type')}")
            except Exception:
                pass

            await ws.send_json(payload)

            deadline = asyncio.get_event_loop().time() + timeout
            while True:
                remaining = deadline - asyncio.get_event_loop().time()
                if remaining <= 0:
                    raise asyncio.TimeoutError("proxy response timed out")
                resp = await asyncio.wait_for(ws.receive_json(), timeout=remaining)
                if expected_type is None or resp.get("type") == expected_type:
                    return resp
                # Not the message we want — keep draining
                logger.debug(f"[proxy] skipped msg type={resp.get('type')}")


# Static asset extensions that bypass token auth
_ASSET_EXTS = frozenset({
    ".css", ".js", ".mjs", ".map", ".json",
    ".ico", ".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp",
    ".woff", ".woff2", ".ttf", ".eot",
    ".webmanifest", ".txt",
})


# ─── Client wrapper ──────────────────────────────────────────────────────────


class _Client:
    """Represents one connected gateway WebSocket client."""

    def __init__(self, ws: web.WebSocketResponse, cid: str | None = None):
        self.ws = ws
        self.cid = cid or f"c{id(self):x}"
        self.node_id: Optional[str] = None
        self._proxy_tasks: Dict[str, asyncio.Task] = {}  # profile → proxy task

    async def send(self, payload: dict) -> None:
        try:
            if not self.ws.closed:
                await self.ws.send_json(payload)
        except Exception as exc:
            logger.debug(f"[Gateway] send to {self.cid} failed: {exc}")

    @property
    def closed(self) -> bool:
        return self.ws.closed

    def close(self) -> None:
        pass  # aiohttp manages WS lifecycle


# ─── Gateway ──────────────────────────────────────────────────────────────────


class Gateway:
    """
    aiohttp-based internal control plane.

    Instantiate with (agent, config) and call ``await gateway.start()``.
    The server runs in the background; call ``await gateway.stop()`` to shut down.
    """

    def __init__(self, agent, config):
        self.agent = agent
        self.config = config

        self._clients: Set[_Client] = set()
        self._nodes: Dict[str, _Client] = {}
        self._running = False
        self._hb_task: Optional[asyncio.Task] = None
        self._start_time = datetime.now(timezone.utc)

        # TCP WebChat settings
        self._webchat_host = getattr(config, "webchat_host", "127.0.0.1")
        _raw_port = int(getattr(config, "webchat_port", 8789))
        self._webchat_port = _find_free_port() if _raw_port == 0 else _raw_port
        self._webchat_token: Optional[str] = getattr(config, "webchat_token", None) or None
        self._webchat_ts = getattr(config, "webchat_tailscale", False)

        # Static file roots
        self._static_dir = Path(__file__).resolve().parent.parent.parent / "static"
        self._project_root = self._static_dir.parent

        # Rate limiting
        self._rate_limits: Dict[str, List[float]] = {}
        self._rate_window = 60
        self._rate_max = 30

        # Monitor system
        self._monitor_clients: Set[_Client] = set()
        self._monitor_agent_wired = False

        # WiFi survival skill (optional, Pi-specific)
        self._wifi_skill = None

        # aiohttp internals
        self._app: Optional[web.Application] = None
        self._runner: Optional[web.AppRunner] = None
        self._sites: list = []

    # ── Build aiohttp Application ─────────────────────────────────────────────

    def _build_app(self) -> web.Application:
        app = web.Application(middlewares=[self._token_auth_middleware])

        # WebSocket endpoint
        app.router.add_get("/ws", self._ws_handler)

        # Dashboard SPA
        app.router.add_get("/dashboard/{path:.*}", self._dashboard_handler)
        app.router.add_get("/dashboard", self._dashboard_handler)

        # Aggr trading terminal SPA
        app.router.add_get("/aggr/{path:.*}", self._aggr_handler)
        app.router.add_get("/aggr", self._aggr_handler)

        # Pi HUD display stream
        app.router.add_get("/api/display/frame", self._display_frame_handler)
        app.router.add_get("/api/display/stream", self._display_stream_handler)
        app.router.add_get("/api/display/status", self._display_status_handler)
        app.router.add_post("/api/display/page", self._display_set_page_handler)

        # Avatar state (read from /tmp/sable-avatar-state.json)
        app.router.add_get("/api/avatar-state", self._avatar_state_handler)

        # Agent environment editor
        app.router.add_get("/api/env", self._env_get_handler)
        app.router.add_post("/api/env", self._env_post_handler)

        # WiFi survival skill
        app.router.add_get("/api/wifi", self._wifi_status_handler)
        app.router.add_post("/api/wifi", self._wifi_command_handler)

        # Polymarket public API proxy
        app.router.add_get("/api/polymarket/{endpoint:.*}", self._polymarket_proxy)

        # Serve generated files (images, etc.)
        app.router.add_get("/files/genelia/{filename}", self._genelia_file_handler)

        # Connectome REST endpoint
        app.router.add_get("/api/connectome", self._connectome_handler)

        # New cognitive module endpoints
        app.router.add_get("/api/deep-planner", self._deep_planner_handler)
        app.router.add_get("/api/inter-agent", self._inter_agent_handler)
        app.router.add_get("/api/ultra-ltm", self._ultra_ltm_handler)
        app.router.add_get("/api/self-benchmark", self._self_benchmark_handler)
        app.router.add_get("/api/meta-learner", self._meta_learner_handler)
        app.router.add_get("/api/causal-engine", self._causal_engine_handler)
        app.router.add_get("/api/goal-synthesis", self._goal_synthesis_handler)
        app.router.add_get("/api/skill-composer", self._skill_composer_handler)
        app.router.add_get("/api/world-predictor", self._world_predictor_handler)
        app.router.add_get("/api/cognitive-optimizer", self._cognitive_optimizer_handler)
        app.router.add_get("/api/adversarial-tester", self._adversarial_tester_handler)
        app.router.add_get("/api/resource-governor", self._resource_governor_handler)
        app.router.add_get("/api/theory-of-mind", self._theory_of_mind_handler)
        app.router.add_get("/api/ethical-reasoner", self._ethical_reasoner_handler)
        # v1.5 World-First Module endpoints
        app.router.add_get("/api/dream-engine", self._dream_engine_handler)
        app.router.add_get("/api/cognitive-immunity", self._cognitive_immunity_handler)
        app.router.add_get("/api/temporal-consciousness", self._temporal_consciousness_handler)
        app.router.add_get("/api/cognitive-fusion", self._cognitive_fusion_handler)
        app.router.add_get("/api/memory-palace", self._memory_palace_handler)
        app.router.add_get("/api/narrative-identity", self._narrative_identity_handler)
        app.router.add_get("/api/curiosity-drive", self._curiosity_drive_handler)
        app.router.add_get("/api/collective-unconscious", self._collective_unconscious_handler)
        app.router.add_get("/api/cognitive-metabolism", self._cognitive_metabolism_handler)
        app.router.add_get("/api/synthetic-intuition", self._synthetic_intuition_handler)
        app.router.add_get("/api/phantom-limb", self._phantom_limb_handler)
        app.router.add_get("/api/cognitive-scar", self._cognitive_scar_handler)
        app.router.add_get("/api/time-crystal", self._time_crystal_handler)
        app.router.add_get("/api/holographic-context", self._holographic_context_handler)
        app.router.add_get("/api/swarm-cortex", self._swarm_cortex_handler)
        app.router.add_get("/api/cognitive-archaeology", self._cognitive_archaeology_handler)
        app.router.add_get("/api/emotional-contagion", self._emotional_contagion_handler)
        app.router.add_get("/api/predictive-empathy", self._predictive_empathy_handler)
        app.router.add_get("/api/autonomous-researcher", self._autonomous_researcher_handler)
        app.router.add_get("/api/empathy-synthesizer", self._empathy_synthesizer_handler)
        app.router.add_get("/api/cognitive-teleportation", self._cognitive_teleportation_handler)
        app.router.add_get("/api/ontological-engine", self._ontological_engine_handler)
        app.router.add_get("/api/cognitive-gravity", self._cognitive_gravity_handler)
        app.router.add_get("/api/temporal-paradox", self._temporal_paradox_handler)
        app.router.add_get("/api/synaesthetic-processor", self._synaesthetic_processor_handler)
        app.router.add_get("/api/cognitive-mitosis", self._cognitive_mitosis_handler)
        app.router.add_get("/api/entropic-sentinel", self._entropic_sentinel_handler)
        app.router.add_get("/api/quantum-cognition", self._quantum_cognition_handler)
        app.router.add_get("/api/cognitive-placebo", self._cognitive_placebo_handler)
        app.router.add_get("/api/noospheric-interface", self._noospheric_interface_handler)
        app.router.add_get("/api/akashic-records", self._akashic_records_handler)
        app.router.add_get("/api/deja-vu", self._deja_vu_handler)
        app.router.add_get("/api/morphogenetic-field", self._morphogenetic_field_handler)
        app.router.add_get("/api/liminal-processor", self._liminal_processor_handler)
        app.router.add_get("/api/prescient-executor", self._prescient_executor_handler)
        app.router.add_get("/api/cognitive-dark-matter", self._cognitive_dark_matter_handler)
        app.router.add_get("/api/ego-membrane", self._ego_membrane_handler)
        app.router.add_get("/api/hyperstition-engine", self._hyperstition_engine_handler)
        app.router.add_get("/api/cognitive-chrysalis", self._cognitive_chrysalis_handler)
        app.router.add_get("/api/existential-compass", self._existential_compass_handler)
        # ── v1.7 God Supreme routes ──
        app.router.add_get("/api/web-agent", self._web_agent_handler)
        app.router.add_get("/api/self-healer", self._self_healer_handler)
        app.router.add_get("/api/dynamic-skill-factory", self._dynamic_skill_factory_handler)
        app.router.add_get("/api/multimodal-engine", self._multimodal_engine_handler)
        app.router.add_get("/api/internet-monitor", self._internet_monitor_handler)
        app.router.add_get("/api/financial-autonomy", self._financial_autonomy_handler)
        app.router.add_get("/api/social-presence", self._social_presence_handler)
        app.router.add_get("/api/self-replicator", self._self_replicator_handler)
        app.router.add_get("/api/continuous-learner", self._continuous_learner_handler)
        # ── v1.8 Final Gap Closers ──
        app.router.add_get("/api/nl-automation", self._nl_automation_handler)
        app.router.add_get("/api/video-understanding", self._video_understanding_handler)
        app.router.add_get("/api/knowledge-graph", self._knowledge_graph_handler)
        app.router.add_get("/api/iot-controller", self._iot_controller_handler)
        app.router.add_get("/api/distributed-tasks", self._distributed_task_queue_handler)
        # ── LLM Management ──
        app.router.add_get("/api/llm/status", self._llm_status_http_handler)
        app.router.add_get("/api/llm/models", self._llm_models_http_handler)
        app.router.add_post("/api/llm/switch", self._llm_switch_http_handler)

        # HTML pages
        app.router.add_get("/chat", self._chat_handler)
        app.router.add_get("/monitor", self._monitor_page_handler)
        app.router.add_get("/dashboard-classic", self._dashboard_classic_handler)
        app.router.add_get("/dashboard-legacy", self._dashboard_legacy_handler)
        app.router.add_get("/favicon.ico", self._favicon_handler)

        # Hub (root)
        app.router.add_get("/", self._hub_handler)

        return app

    # ── Token auth middleware ─────────────────────────────────────────────────

    @web.middleware
    async def _token_auth_middleware(self, request: web.Request, handler):
        if not self._webchat_token:
            return await handler(request)

        # Unix socket connections are trusted (filesystem ACL controls access)
        transport = getattr(request, "transport", None) or (
            request._payload._transport if hasattr(request, "_payload") else None
        )
        if transport:
            sockname = transport.get_extra_info("sockname")
            if isinstance(sockname, str):  # Unix socket path → trusted
                return await handler(request)

        path = request.path.lower()

        # Asset extensions and specific prefixes skip auth
        ext = os.path.splitext(path)[1]
        if ext in _ASSET_EXTS or path.startswith("/aggr/") or path.startswith("/api/polymarket/") or path.startswith("/api/connectome") or path.startswith("/api/deep-planner") or path.startswith("/api/inter-agent") or path.startswith("/api/ultra-ltm") or path.startswith("/api/self-benchmark") or path.startswith("/api/meta-learner") or path.startswith("/api/causal-engine") or path.startswith("/api/goal-synthesis") or path.startswith("/api/skill-composer") or path.startswith("/api/world-predictor") or path.startswith("/api/cognitive-optimizer") or path.startswith("/api/adversarial-tester") or path.startswith("/api/resource-governor") or path.startswith("/api/theory-of-mind") or path.startswith("/api/ethical-reasoner") or path.startswith("/api/dream-engine") or path.startswith("/api/cognitive-immunity") or path.startswith("/api/temporal-consciousness") or path.startswith("/api/cognitive-fusion") or path.startswith("/api/memory-palace") or path.startswith("/api/narrative-identity") or path.startswith("/api/curiosity-drive") or path.startswith("/api/collective-unconscious") or path.startswith("/api/cognitive-metabolism") or path.startswith("/api/synthetic-intuition") or path.startswith("/api/phantom-limb") or path.startswith("/api/cognitive-scar") or path.startswith("/api/time-crystal") or path.startswith("/api/holographic-context") or path.startswith("/api/swarm-cortex") or path.startswith("/api/cognitive-archaeology") or path.startswith("/api/emotional-contagion") or path.startswith("/api/predictive-empathy") or path.startswith("/api/autonomous-researcher") or path.startswith("/api/empathy-synthesizer") or path.startswith("/api/cognitive-teleportation") or path.startswith("/api/ontological-engine") or path.startswith("/api/cognitive-gravity") or path.startswith("/api/temporal-paradox") or path.startswith("/api/synaesthetic-processor") or path.startswith("/api/cognitive-mitosis") or path.startswith("/api/entropic-sentinel") or path.startswith("/api/quantum-cognition") or path.startswith("/api/cognitive-placebo") or path.startswith("/api/noospheric-interface") or path.startswith("/api/akashic-records") or path.startswith("/api/deja-vu") or path.startswith("/api/morphogenetic-field") or path.startswith("/api/liminal-processor") or path.startswith("/api/prescient-executor") or path.startswith("/api/cognitive-dark-matter") or path.startswith("/api/ego-membrane") or path.startswith("/api/hyperstition-engine") or path.startswith("/api/cognitive-chrysalis") or path.startswith("/api/existential-compass") or path.startswith("/api/web-agent") or path.startswith("/api/self-healer") or path.startswith("/api/dynamic-skill-factory") or path.startswith("/api/multimodal-engine") or path.startswith("/api/internet-monitor") or path.startswith("/api/financial-autonomy") or path.startswith("/api/social-presence") or path.startswith("/api/self-replicator") or path.startswith("/api/continuous-learner") or path.startswith("/api/nl-automation") or path.startswith("/api/video-understanding") or path.startswith("/api/knowledge-graph") or path.startswith("/api/iot-controller") or path.startswith("/api/distributed-tasks") or path.startswith("/api/llm/") or path == "/favicon.ico":
            return await handler(request)

        supplied = request.query.get("token", "")
        if not supplied:
            supplied = request.headers.get("Sec-WebSocket-Protocol", "")

        if not hmac.compare_digest(str(supplied), self._webchat_token):
            return web.Response(
                status=401,
                content_type="text/html",
                text="<html><body><h1>401 Unauthorized</h1>"
                "<p>Invalid or missing token.</p></body></html>",
            )

        return await handler(request)

    # ── HTTP route handlers ───────────────────────────────────────────────────

    async def _hub_handler(self, request: web.Request) -> web.Response:
        # The old gateway accepted WS upgrades on ANY path.
        # All frontend dashboards connect via ws://host/ (root), so we
        # check for a WebSocket upgrade here and delegate to the WS handler.
        if (
            request.headers.get("Upgrade", "").lower() == "websocket"
            or request.headers.get("Connection", "").lower() == "upgrade"
        ):
            return await self._ws_handler(request)
        return self._serve_html_file(self._static_dir / "hub.html")

    async def _chat_handler(self, request: web.Request) -> web.Response:
        return self._serve_html_file(self._static_dir / "dashboard.html")

    async def _monitor_page_handler(self, request: web.Request) -> web.Response:
        return self._serve_html_file(self._static_dir / "monitor.html")

    async def _dashboard_classic_handler(self, request: web.Request) -> web.Response:
        return self._serve_html_file(self._static_dir / "dashboard_v2.html")

    async def _dashboard_legacy_handler(self, request: web.Request) -> web.Response:
        return self._serve_html_file(self._static_dir / "dashboard_modern.html")

    async def _favicon_handler(self, request: web.Request) -> web.Response:
        fav = self._static_dir / "favicon.ico"
        if fav.exists():
            return web.FileResponse(fav)
        return web.Response(status=204)

    async def _dashboard_handler(self, request: web.Request) -> web.Response:
        dash_dist = self._project_root / "dashboard" / "dist"
        if dash_dist.exists():
            rel = request.match_info.get("path", "") or "index.html"
            return self._serve_spa(dash_dist, rel)
        return self._serve_html_file(self._static_dir / "dashboard_v2.html")

    async def _aggr_handler(self, request: web.Request) -> web.Response:
        aggr_dist = self._project_root / "aggr" / "dist"
        if aggr_dist.exists():
            rel = request.match_info.get("path", "") or "index.html"
            return self._serve_spa(aggr_dist, rel)
        return web.Response(
            content_type="text/html",
            text="<html><body><h1>Aggr.trade not installed</h1>"
            "<p>Run: <code>cd aggr && npm install && npm run build</code></p></body></html>",
        )

    # ── Pi display stream ─────────────────────────────────────────────────────

    _DISPLAY_FRAME_PATH = "/tmp/sable_hud_frame.jpg"
    _DISPLAY_BOUNDARY   = b"--sablehudframe"

    async def _display_frame_handler(self, request: web.Request) -> web.Response:
        """Return the latest HUD snapshot as a JPEG image."""
        path = self._DISPLAY_FRAME_PATH
        if not os.path.exists(path):
            return web.Response(status=503, text="Display not active")
        with open(path, "rb") as fh:
            data = fh.read()
        return web.Response(
            body=data,
            content_type="image/jpeg",
            headers={"Cache-Control": "no-store"},
        )

    async def _display_status_handler(self, request: web.Request) -> web.Response:
        """Return current page index and page list as JSON."""
        status_path = "/tmp/sable_hud_status.json"
        if not os.path.exists(status_path):
            return web.Response(status=503, content_type="application/json",
                                text='{"error":"display not active"}')
        with open(status_path) as fh:
            data = fh.read()
        return web.Response(body=data.encode(), content_type="application/json",
                            headers={"Cache-Control": "no-store"})

    async def _display_set_page_handler(self, request: web.Request) -> web.Response:
        """Write a page-change request for display_hud.py to pick up."""
        try:
            body = await request.json()
            page = int(body.get("page", -1))
        except Exception:
            return web.Response(status=400, content_type="application/json",
                                text='{"error":"invalid body"}')
        if page < 0 or page > 9:
            return web.Response(status=400, content_type="application/json",
                                text='{"error":"page out of range"}')
        req_path = "/tmp/sable_hud_page_req"
        try:
            with open(req_path, "w") as fh:
                fh.write(str(page))
        except OSError as exc:
            return web.Response(status=500, content_type="application/json",
                                text=json.dumps({"error": str(exc)}))
        return web.Response(content_type="application/json",
                            text=json.dumps({"ok": True, "page": page}))

    async def _avatar_state_handler(self, request: web.Request) -> web.Response:
        """Return current avatar state from /tmp/sable-avatar-state.json."""
        state_path = "/tmp/sable-avatar-state.json"
        try:
            with open(state_path) as fh:
                data = fh.read()
            return web.Response(body=data.encode(), content_type="application/json",
                                headers={"Cache-Control": "no-store"})
        except FileNotFoundError:
            return web.Response(content_type="application/json",
                                headers={"Cache-Control": "no-store"},
                                text='{"state":"idle","text":"","tool":"","words":0,"ts":0}')
        except Exception:
            return web.Response(status=500, content_type="application/json",
                                text='{"error":"failed to read avatar state"}')

    async def _display_stream_handler(self, request: web.Request) -> web.StreamResponse:
        """MJPEG stream of the Pi HUD — suitable for <img src=...>."""
        resp = web.StreamResponse(headers={
            "Content-Type": f"multipart/x-mixed-replace; boundary={self._DISPLAY_BOUNDARY.decode()}",
            "Cache-Control": "no-store",
            "X-Accel-Buffering": "no",
        })
        await resp.prepare(request)
        path = self._DISPLAY_FRAME_PATH
        last_mtime = None
        idle_loops = 0
        try:
            while not request.transport.is_closing():
                if not os.path.exists(path):
                    await asyncio.sleep(0.5)
                    idle_loops += 1
                    if idle_loops > 20:
                        break
                    continue
                idle_loops = 0
                mtime = os.path.getmtime(path)
                if mtime != last_mtime:
                    last_mtime = mtime
                    try:
                        with open(path, "rb") as fh:
                            data = fh.read()
                    except OSError:
                        await asyncio.sleep(0.1)
                        continue
                    part = (
                        self._DISPLAY_BOUNDARY + b"\r\n"
                        + b"Content-Type: image/jpeg\r\n"
                        + f"Content-Length: {len(data)}\r\n\r\n".encode()
                        + data + b"\r\n"
                    )
                    await resp.write(part)
                await asyncio.sleep(0.15)
        except (ConnectionResetError, asyncio.CancelledError):
            pass
        return resp

    # ── Env editor ────────────────────────────────────────────────────────────

    def _profile_env_path(self) -> Path:
        """Return the absolute path to the active profile's profile.env file."""
        return self._project_root / "agents" / _profile_name / "profile.env"

    async def _env_get_handler(self, request: web.Request) -> web.Response:
        """Return the profile.env contents as a structured JSON object."""
        import re as _re
        env_path = self._profile_env_path()
        if not env_path.exists():
            return web.Response(
                status=404,
                content_type="application/json",
                text=json.dumps({"error": f"profile.env not found: {env_path}"}),
            )
        raw = env_path.read_text(encoding="utf-8")
        sections: list[dict] = []
        current_section: dict | None = None
        pending_comments: list[str] = []

        # Matches: # ── Title ────...   or   # === Title ===...
        _section_re = _re.compile(
            r'^#\s*[─═\-]{2,}\s*(.+?)\s*[─═\-]*\s*$'
        )
        # Strip inline comment from a value: VALUE   # comment
        def _strip_inline(v: str) -> str:
            # Split on first occurrence of whitespace + # (not inside a URL)
            m = _re.search(r'\s{2,}#\s', v)
            if m:
                return v[:m.start()].strip()
            return v.strip()

        for line in raw.splitlines():
            stripped = line.strip()

            # Skip pure decorator lines (=== or ─── with no word chars)
            if stripped.startswith("#") and not _re.search(r'[a-zA-Z]', stripped):
                continue

            # Section header  # ── Title ─────
            m = _section_re.match(stripped)
            if m:
                title = m.group(1).strip()
                # Skip generic file headers
                if any(k in title.lower() for k in ("generated by", "opensable —", "device:")):
                    continue
                current_section = {"title": title, "entries": []}
                sections.append(current_section)
                pending_comments = []
                continue

            # Blank comment separator
            if stripped == "#":
                pending_comments = []
                continue

            # Regular comment (inline doc for next key)
            if stripped.startswith("#"):
                pending_comments.append(stripped[1:].strip())
                continue

            # Blank line — flush pending comments
            if not stripped:
                pending_comments = []
                continue

            # KEY=VALUE line
            if "=" in stripped and not stripped.startswith("#"):
                key, _, value = stripped.partition("=")
                key = key.strip()
                if not key:
                    continue
                value = _strip_inline(value)
                if current_section is None:
                    current_section = {"title": "General", "entries": []}
                    sections.append(current_section)
                current_section["entries"].append({
                    "key": key,
                    "value": value,
                    "comment": " ".join(pending_comments) if pending_comments else None,
                })
                pending_comments = []

        # Drop empty sections
        sections = [s for s in sections if s["entries"]]

        return web.Response(
            content_type="application/json",
            text=json.dumps({
                "profile": _profile_name,
                "path": str(env_path),
                "sections": sections,
            }),
            headers={"Cache-Control": "no-store"},
        )

    async def _env_post_handler(self, request: web.Request) -> web.Response:
        """Update, add, or delete keys in profile.env without touching other content."""
        import re as _re2
        try:
            body = await request.json()
            updates: dict[str, str] = body.get("updates", {})
            adds: list[dict]        = body.get("adds", [])    # [{key, value}]
            deletes: list[str]      = body.get("deletes", [])
        except Exception:
            return web.Response(status=400, content_type="application/json",
                                text='{"error":"invalid JSON body"}')

        if not isinstance(updates, dict):
            updates = {}
        if not isinstance(adds, list):
            adds = []
        if not isinstance(deletes, list):
            deletes = []

        if not updates and not adds and not deletes:
            return web.Response(status=400, content_type="application/json",
                                text='{"error":"no operations specified"}')

        # Validate new key names (must be valid shell identifiers)
        _key_re = _re2.compile(r'^[A-Z_][A-Z0-9_]*$', _re2.IGNORECASE)
        adds = [a for a in adds if isinstance(a, dict) and _key_re.match(str(a.get("key", "")))]

        env_path = self._profile_env_path()
        if not env_path.exists():
            return web.Response(status=404, content_type="application/json",
                                text=json.dumps({"error": f"profile.env not found: {env_path}"}))

        raw = env_path.read_text(encoding="utf-8")
        lines = raw.splitlines(keepends=True)
        applied: list[str] = []
        deleted: list[str] = []
        new_lines: list[str] = []
        delete_set = set(deletes)

        for line in lines:
            stripped = line.strip()
            if "=" in stripped and not stripped.startswith("#"):
                key = stripped.split("=", 1)[0].strip()
                if key in delete_set:
                    deleted.append(key)
                    continue  # Remove this line
                if key in updates:
                    eol = "\n" if line.endswith("\n") else ""
                    new_lines.append(f"{key}={updates[key]}{eol}")
                    applied.append(key)
                    continue
            new_lines.append(line)

        # Append new keys at end of file
        added: list[str] = []
        for add_entry in adds:
            key = str(add_entry.get("key", "")).strip()
            value = str(add_entry.get("value", ""))
            if not key:
                continue
            if not new_lines or new_lines[-1].endswith("\n"):
                new_lines.append(f"{key}={value}\n")
            else:
                new_lines.append(f"\n{key}={value}\n")
            added.append(key)

        # Atomic write — write to temp then rename
        import tempfile
        tmp_fd, tmp_path = tempfile.mkstemp(
            dir=str(env_path.parent), prefix=".profile.env.tmp"
        )
        try:
            with os.fdopen(tmp_fd, "w", encoding="utf-8") as fh:
                fh.writelines(new_lines)
            os.replace(tmp_path, str(env_path))
        except OSError as exc:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            return web.Response(status=500, content_type="application/json",
                                text=json.dumps({"error": str(exc)}))

        missing = [k for k in updates if k not in applied]
        return web.Response(
            content_type="application/json",
            text=json.dumps({"ok": True, "applied": applied, "added": added,
                             "deleted": deleted, "missing": missing}),
        )

    # ── WiFi Survival handlers ────────────────────────────────────────────────

    async def _wifi_status_handler(self, request: web.Request) -> web.Response:
        """GET /api/wifi — return WifiSurvivalSkill status as JSON."""
        if self._wifi_skill is None:
            return web.Response(
                status=503,
                content_type="application/json",
                text=json.dumps({"error": "wifi skill not running",
                                 "hint": "Set WIFI_HUNT_ENABLED=true in profile.env"}),
            )
        return web.Response(
            content_type="application/json",
            text=json.dumps(self._wifi_skill.status()),
        )

    async def _wifi_command_handler(self, request: web.Request) -> web.Response:
        """POST /api/wifi — send commands to WifiSurvivalSkill."""
        if self._wifi_skill is None:
            return web.Response(
                status=503,
                content_type="application/json",
                text=json.dumps({"error": "wifi skill not running"}),
            )
        try:
            body = await request.json()
        except Exception:
            return web.Response(status=400, content_type="application/json",
                                text='{"error":"invalid JSON body"}')

        cmd = body.get("cmd", "")
        if cmd == "force_hunt":
            asyncio.create_task(self._wifi_skill.force_hunt())
        elif cmd == "add_whitelist":
            ssid = str(body.get("ssid", "")).strip()
            if not ssid:
                return web.Response(status=400, content_type="application/json",
                                    text='{"error":"ssid required"}')
            added = self._wifi_skill.add_to_whitelist(ssid)
            return web.Response(content_type="application/json",
                                text=json.dumps({"ok": True, "added": added}))
        elif cmd == "add_credential":
            ssid     = str(body.get("ssid", "")).strip()
            password = str(body.get("password", ""))
            bssid    = str(body.get("bssid", "00:00:00:00:00:00"))
            if not ssid or not password:
                return web.Response(status=400, content_type="application/json",
                                    text='{"error":"ssid and password required"}')
            self._wifi_skill.add_credential(ssid, bssid, password)
        elif cmd == "stop":
            self._wifi_skill._enabled = False
        elif cmd == "start":
            self._wifi_skill._enabled = True
        else:
            return web.Response(status=400, content_type="application/json",
                                text=json.dumps({"error": f"unknown cmd: {cmd}"}))

        return web.Response(content_type="application/json", text='{"ok":true}')

    async def _polymarket_proxy(self, request: web.Request) -> web.Response:
        """Proxy requests to Polymarket public APIs (Gamma + CLOB)."""
        endpoint = request.match_info.get("endpoint", "")
        qs = request.query_string

        # Route to correct upstream
        if endpoint.startswith("clob/"):
            upstream = f"https://clob.polymarket.com/{endpoint[5:]}"
        else:
            upstream = f"https://gamma-api.polymarket.com/{endpoint}"

        if qs:
            upstream += f"?{qs}"

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    upstream,
                    timeout=aiohttp.ClientTimeout(total=15),
                    headers={"Accept": "application/json"},
                ) as resp:
                    body = await resp.read()
                    return web.Response(
                        body=body,
                        status=resp.status,
                        content_type=resp.content_type or "application/json",
                        headers={
                            "Access-Control-Allow-Origin": "*",
                            "Cache-Control": "public, max-age=30",
                        },
                    )
        except asyncio.TimeoutError:
            return web.json_response({"error": "Polymarket API timeout"}, status=504)
        except Exception as e:
            logger.error(f"Polymarket proxy error: {e}")
            return web.json_response({"error": str(e)}, status=502)

    # ── Connectome REST endpoint ─────────────────────────────────────────────

    async def _connectome_handler(self, request: web.Request) -> web.Response:
        """Return connectome wiring diagram + stats as JSON."""
        try:
            connectome = getattr(self.agent, "connectome", None)
            if not connectome:
                return web.json_response({"error": "Connectome not initialised"}, status=404)
            data = connectome.get_wiring_diagram()
            data["stats"] = connectome.get_stats()
            return web.json_response(data, headers={"Access-Control-Allow-Origin": "*"})
        except Exception as e:
            logger.error(f"Connectome endpoint error: {e}")
            return web.json_response({"error": str(e)}, status=500)

    async def _deep_planner_handler(self, request: web.Request) -> web.Response:
        """Return deep planner stats as JSON."""
        try:
            planner = getattr(self.agent, "deep_planner", None)
            if not planner and hasattr(self.agent, "autonomous") and self.agent.autonomous:
                planner = getattr(self.agent.autonomous, "deep_planner", None)
            if not planner:
                return web.json_response({"error": "Deep planner not initialised"}, status=404)
            return web.json_response(planner.get_stats(), headers={"Access-Control-Allow-Origin": "*"})
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)

    async def _inter_agent_handler(self, request: web.Request) -> web.Response:
        """Return inter-agent bridge stats as JSON."""
        try:
            bridge = getattr(self.agent, "inter_agent_bridge", None)
            if not bridge and hasattr(self.agent, "autonomous") and self.agent.autonomous:
                bridge = getattr(self.agent.autonomous, "inter_agent_bridge", None)
            if not bridge:
                return web.json_response({"error": "Inter-agent bridge not initialised"}, status=404)
            return web.json_response(bridge.get_stats(), headers={"Access-Control-Allow-Origin": "*"})
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)

    async def _ultra_ltm_handler(self, request: web.Request) -> web.Response:
        """Return ultra-LTM stats as JSON."""
        try:
            ltm = getattr(self.agent, "ultra_ltm", None)
            if not ltm and hasattr(self.agent, "autonomous") and self.agent.autonomous:
                ltm = getattr(self.agent.autonomous, "ultra_ltm", None)
            if not ltm:
                return web.json_response({"error": "Ultra-LTM not initialised"}, status=404)
            return web.json_response(ltm.get_stats(), headers={"Access-Control-Allow-Origin": "*"})
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)

    async def _self_benchmark_handler(self, request: web.Request) -> web.Response:
        """Return self-benchmark stats as JSON."""
        try:
            bench = getattr(self.agent, "self_benchmark", None)
            if not bench and hasattr(self.agent, "autonomous") and self.agent.autonomous:
                bench = getattr(self.agent.autonomous, "self_benchmark", None)
            if not bench:
                return web.json_response({"error": "Self benchmark not initialised"}, status=404)
            return web.json_response(bench.get_stats(), headers={"Access-Control-Allow-Origin": "*"})
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)

    async def _meta_learner_handler(self, request: web.Request) -> web.Response:
        return await self._generic_module_handler("meta_learner", "Meta learner", request)

    async def _causal_engine_handler(self, request: web.Request) -> web.Response:
        return await self._generic_module_handler("causal_engine", "Causal engine", request)

    async def _goal_synthesis_handler(self, request: web.Request) -> web.Response:
        return await self._generic_module_handler("goal_synthesis", "Goal synthesis", request)

    async def _skill_composer_handler(self, request: web.Request) -> web.Response:
        return await self._generic_module_handler("skill_composer", "Skill composer", request)

    async def _world_predictor_handler(self, request: web.Request) -> web.Response:
        return await self._generic_module_handler("world_predictor", "World predictor", request)

    async def _cognitive_optimizer_handler(self, request: web.Request) -> web.Response:
        return await self._generic_module_handler("cognitive_optimizer", "Cognitive optimizer", request)

    async def _adversarial_tester_handler(self, request: web.Request) -> web.Response:
        return await self._generic_module_handler("adversarial_tester", "Adversarial tester", request)

    async def _resource_governor_handler(self, request: web.Request) -> web.Response:
        return await self._generic_module_handler("resource_governor", "Resource governor", request)

    async def _theory_of_mind_handler(self, request: web.Request) -> web.Response:
        return await self._generic_module_handler("theory_of_mind", "Theory of mind", request)

    async def _ethical_reasoner_handler(self, request: web.Request) -> web.Response:
        return await self._generic_module_handler("ethical_reasoner", "Ethical reasoner", request)

    # ── v1.5 World-First Module Handlers ──

    async def _dream_engine_handler(self, request: web.Request) -> web.Response:
        return await self._generic_module_handler("dream_engine", "Dream engine", request)

    async def _cognitive_immunity_handler(self, request: web.Request) -> web.Response:
        return await self._generic_module_handler("cognitive_immunity", "Cognitive immunity", request)

    async def _temporal_consciousness_handler(self, request: web.Request) -> web.Response:
        return await self._generic_module_handler("temporal_consciousness", "Temporal consciousness", request)

    async def _cognitive_fusion_handler(self, request: web.Request) -> web.Response:
        return await self._generic_module_handler("cognitive_fusion", "Cognitive fusion", request)

    async def _memory_palace_handler(self, request: web.Request) -> web.Response:
        return await self._generic_module_handler("memory_palace", "Memory palace", request)

    async def _narrative_identity_handler(self, request: web.Request) -> web.Response:
        return await self._generic_module_handler("narrative_identity", "Narrative identity", request)

    async def _curiosity_drive_handler(self, request: web.Request) -> web.Response:
        return await self._generic_module_handler("curiosity_drive", "Curiosity drive", request)

    async def _collective_unconscious_handler(self, request: web.Request) -> web.Response:
        return await self._generic_module_handler("collective_unconscious", "Collective unconscious", request)

    async def _cognitive_metabolism_handler(self, request: web.Request) -> web.Response:
        return await self._generic_module_handler("cognitive_metabolism", "Cognitive metabolism", request)

    async def _synthetic_intuition_handler(self, request: web.Request) -> web.Response:
        return await self._generic_module_handler("synthetic_intuition", "Synthetic intuition", request)

    async def _phantom_limb_handler(self, request: web.Request) -> web.Response:
        return await self._generic_module_handler("phantom_limb", "Phantom limb", request)

    async def _cognitive_scar_handler(self, request: web.Request) -> web.Response:
        return await self._generic_module_handler("cognitive_scar", "Cognitive scar", request)

    async def _time_crystal_handler(self, request: web.Request) -> web.Response:
        return await self._generic_module_handler("time_crystal", "Time crystal", request)

    async def _holographic_context_handler(self, request: web.Request) -> web.Response:
        return await self._generic_module_handler("holographic_context", "Holographic context", request)

    async def _swarm_cortex_handler(self, request: web.Request) -> web.Response:
        return await self._generic_module_handler("swarm_cortex", "Swarm cortex", request)

    async def _cognitive_archaeology_handler(self, request: web.Request) -> web.Response:
        return await self._generic_module_handler("cognitive_archaeology", "Cognitive archaeology", request)

    async def _emotional_contagion_handler(self, request: web.Request) -> web.Response:
        return await self._generic_module_handler("emotional_contagion", "Emotional contagion", request)

    async def _predictive_empathy_handler(self, request: web.Request) -> web.Response:
        return await self._generic_module_handler("predictive_empathy", "Predictive empathy", request)

    async def _autonomous_researcher_handler(self, request: web.Request) -> web.Response:
        return await self._generic_module_handler("autonomous_researcher", "Autonomous researcher", request)

    async def _empathy_synthesizer_handler(self, request: web.Request) -> web.Response:
        return await self._generic_module_handler("empathy_synthesizer", "Empathy synthesizer", request)

    async def _cognitive_teleportation_handler(self, request: web.Request) -> web.Response:
        return await self._generic_module_handler("cognitive_teleportation", "Cognitive teleportation", request)

    async def _ontological_engine_handler(self, request: web.Request) -> web.Response:
        return await self._generic_module_handler("ontological_engine", "Ontological engine", request)

    async def _cognitive_gravity_handler(self, request: web.Request) -> web.Response:
        return await self._generic_module_handler("cognitive_gravity", "Cognitive gravity", request)

    async def _temporal_paradox_handler(self, request: web.Request) -> web.Response:
        return await self._generic_module_handler("temporal_paradox", "Temporal paradox", request)

    async def _synaesthetic_processor_handler(self, request: web.Request) -> web.Response:
        return await self._generic_module_handler("synaesthetic_processor", "Synaesthetic processor", request)

    async def _cognitive_mitosis_handler(self, request: web.Request) -> web.Response:
        return await self._generic_module_handler("cognitive_mitosis", "Cognitive mitosis", request)

    async def _entropic_sentinel_handler(self, request: web.Request) -> web.Response:
        return await self._generic_module_handler("entropic_sentinel", "Entropic sentinel", request)

    async def _quantum_cognition_handler(self, request: web.Request) -> web.Response:
        return await self._generic_module_handler("quantum_cognition", "Quantum cognition", request)

    async def _cognitive_placebo_handler(self, request: web.Request) -> web.Response:
        return await self._generic_module_handler("cognitive_placebo", "Cognitive placebo", request)

    async def _noospheric_interface_handler(self, request: web.Request) -> web.Response:
        return await self._generic_module_handler("noospheric_interface", "Noospheric interface", request)

    async def _akashic_records_handler(self, request: web.Request) -> web.Response:
        return await self._generic_module_handler("akashic_records", "Akashic records", request)

    async def _deja_vu_handler(self, request: web.Request) -> web.Response:
        return await self._generic_module_handler("deja_vu", "Deja vu", request)

    async def _morphogenetic_field_handler(self, request: web.Request) -> web.Response:
        return await self._generic_module_handler("morphogenetic_field", "Morphogenetic field", request)

    async def _liminal_processor_handler(self, request: web.Request) -> web.Response:
        return await self._generic_module_handler("liminal_processor", "Liminal processor", request)

    async def _prescient_executor_handler(self, request: web.Request) -> web.Response:
        return await self._generic_module_handler("prescient_executor", "Prescient executor", request)

    async def _cognitive_dark_matter_handler(self, request: web.Request) -> web.Response:
        return await self._generic_module_handler("cognitive_dark_matter", "Cognitive dark matter", request)

    async def _ego_membrane_handler(self, request: web.Request) -> web.Response:
        return await self._generic_module_handler("ego_membrane", "Ego membrane", request)

    async def _hyperstition_engine_handler(self, request: web.Request) -> web.Response:
        return await self._generic_module_handler("hyperstition_engine", "Hyperstition engine", request)

    async def _cognitive_chrysalis_handler(self, request: web.Request) -> web.Response:
        return await self._generic_module_handler("cognitive_chrysalis", "Cognitive chrysalis", request)

    async def _existential_compass_handler(self, request: web.Request) -> web.Response:
        return await self._generic_module_handler("existential_compass", "Existential compass", request)

    # ── v1.7 God Supreme handlers ──

    async def _web_agent_handler(self, request: web.Request) -> web.Response:
        return await self._generic_module_handler("web_agent", "Web agent", request)

    async def _self_healer_handler(self, request: web.Request) -> web.Response:
        return await self._generic_module_handler("self_healer", "Self healer", request)

    async def _dynamic_skill_factory_handler(self, request: web.Request) -> web.Response:
        return await self._generic_module_handler("dynamic_skill_factory", "Dynamic skill factory", request)

    async def _multimodal_engine_handler(self, request: web.Request) -> web.Response:
        return await self._generic_module_handler("multimodal_engine", "Multimodal engine", request)

    async def _internet_monitor_handler(self, request: web.Request) -> web.Response:
        return await self._generic_module_handler("internet_monitor", "Internet monitor", request)

    async def _financial_autonomy_handler(self, request: web.Request) -> web.Response:
        return await self._generic_module_handler("financial_autonomy", "Financial autonomy", request)

    async def _social_presence_handler(self, request: web.Request) -> web.Response:
        return await self._generic_module_handler("social_presence", "Social presence", request)

    async def _self_replicator_handler(self, request: web.Request) -> web.Response:
        return await self._generic_module_handler("self_replicator", "Self replicator", request)

    async def _continuous_learner_handler(self, request: web.Request) -> web.Response:
        return await self._generic_module_handler("continuous_learner", "Continuous learner", request)

    async def _nl_automation_handler(self, request: web.Request) -> web.Response:
        return await self._generic_module_handler("nl_automation", "NL automation", request)

    async def _video_understanding_handler(self, request: web.Request) -> web.Response:
        return await self._generic_module_handler("video_understanding", "Video understanding", request)

    async def _knowledge_graph_handler(self, request: web.Request) -> web.Response:
        return await self._generic_module_handler("knowledge_graph", "Knowledge graph", request)

    async def _iot_controller_handler(self, request: web.Request) -> web.Response:
        return await self._generic_module_handler("iot_controller", "IoT controller", request)

    async def _distributed_task_queue_handler(self, request: web.Request) -> web.Response:
        return await self._generic_module_handler("distributed_task_queue", "Distributed task queue", request)

    # ── LLM Management HTTP handlers ─────────────────────────────────────────

    async def _llm_status_http_handler(self, request: web.Request) -> web.Response:
        """GET /api/llm/status,  current provider, model, and configured providers."""
        try:
            from opensable.core.llm import AdaptiveLLM
            llm = self.agent.llm
            provider = "ollama" if isinstance(llm, AdaptiveLLM) else getattr(llm, "provider", "unknown")
            return web.json_response({
                "provider": provider,
                "model": llm.current_model,
                "available_models": getattr(llm, "available_models", []),
                "providers_configured": self._get_configured_providers(),
            }, headers={"Access-Control-Allow-Origin": "*"})
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)

    async def _llm_models_http_handler(self, request: web.Request) -> web.Response:
        """GET /api/llm/models,  list all available models across providers."""
        try:
            from opensable.core.llm import list_all_models
            models = await list_all_models(self.agent.config)
            return web.json_response({"models": models}, headers={"Access-Control-Allow-Origin": "*"})
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)

    async def _llm_switch_http_handler(self, request: web.Request) -> web.Response:
        """POST /api/llm/switch,  switch provider/model at runtime.
        Body: {"provider": "openwebui", "model": "optional-model-name"}
        """
        try:
            from opensable.core.llm import switch_llm
            body = await request.json()
            provider = body.get("provider", "ollama")
            model = body.get("model")
            result = switch_llm(self.agent, provider, model)
            return web.json_response(result, headers={"Access-Control-Allow-Origin": "*"})
        except Exception as e:
            return web.json_response({"success": False, "error": str(e)}, status=500)

    async def _generic_module_handler(self, attr: str, label: str, request: web.Request) -> web.Response:
        """Generic handler for cognitive module stats endpoints."""
        try:
            mod = getattr(self.agent, attr, None)
            if not mod and hasattr(self.agent, "autonomous") and self.agent.autonomous:
                mod = getattr(self.agent.autonomous, attr, None)
            if not mod:
                return web.json_response({"error": f"{label} not initialised"}, status=404)
            return web.json_response(mod.get_stats(), headers={"Access-Control-Allow-Origin": "*"})
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)

    # ── Generated files ──────────────────────────────────────────────────────

    async def _genelia_file_handler(self, request: web.Request) -> web.Response:
        """Serve generated images from data/genelia_images/."""
        filename = request.match_info.get("filename", "")
        # Security: only allow simple filenames (no path traversal)
        if not filename or "/" in filename or ".." in filename:
            return web.Response(status=404)
        filepath = self._project_root / "data" / "genelia_images" / filename
        if not filepath.exists() or not filepath.is_file():
            return web.Response(status=404)
        import mimetypes
        ct = mimetypes.guess_type(str(filepath))[0] or "image/png"
        return web.Response(
            body=filepath.read_bytes(),
            content_type=ct,
            headers={"Cache-Control": "public, max-age=3600"},
        )

    # ── Static file helpers ───────────────────────────────────────────────────

    @staticmethod
    def _serve_html_file(path: Path) -> web.Response:
        if path.exists():
            return web.Response(
                body=path.read_bytes(),
                content_type="text/html",
                charset="utf-8",
            )
        return web.Response(
            text="<!doctype html><title>Sable</title><body>"
            "<h1>Sable Gateway</h1><p>Page not found.</p></body>",
            content_type="text/html",
        )

    @staticmethod
    def _serve_spa(base_dir: Path, rel_path: str) -> web.Response:
        """Serve a static file from a SPA directory with index.html fallback."""
        import mimetypes

        safe = Path(rel_path).as_posix().replace("..", "")
        target = base_dir / safe
        is_fallback = False

        if not target.exists() or not target.is_file():
            for sub in ("aggr", "dashboard", "assets"):
                alt = base_dir / sub / safe
                if alt.exists() and alt.is_file():
                    target = alt
                    break

        if not target.exists() or not target.is_file():
            target = base_dir / "index.html"
            is_fallback = True

        if target.exists() and target.is_file():
            ct = mimetypes.guess_type(str(target))[0] or "application/octet-stream"
            # index.html must never be cached so builds take effect immediately;
            # hashed asset files (js/css) can be cached long-term.
            if is_fallback or target.name == "index.html":
                cc = "no-cache, no-store, must-revalidate"
            else:
                cc = "public, max-age=31536000, immutable"
            return web.Response(
                body=target.read_bytes(),
                content_type=ct,
                headers={"Cache-Control": cc},
            )
        return web.Response(status=404, text="404 Not Found")

    # ── WebSocket handler ─────────────────────────────────────────────────────

    async def _ws_handler(self, request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse(heartbeat=HEARTBEAT_INT)
        await ws.prepare(request)

        client = _Client(ws)
        self._clients.add(client)
        logger.debug(f"[Gateway] Client connected: {client.cid}")

        try:
            await client.send({"type": "connected", "version": GATEWAY_VER, "ts": time.time()})

            async for msg in ws:
                if msg.type == WSMsgType.TEXT:
                    try:
                        data = json.loads(msg.data)
                    except json.JSONDecodeError:
                        await client.send({"type": "error", "text": "Invalid JSON"})
                        continue
                    await self._dispatch(client, data)
                elif msg.type == WSMsgType.ERROR:
                    logger.debug(f"[Gateway] WS error {client.cid}: {ws.exception()}")
                    break
        except Exception as exc:
            logger.debug(f"[Gateway] {client.cid} error: {exc}")
        finally:
            # Cancel any proxy tasks for this client
            for task in client._proxy_tasks.values():
                task.cancel()
            client._proxy_tasks.clear()

            self._clients.discard(client)
            self._monitor_clients.discard(client)
            if client.node_id and client.node_id in self._nodes:
                del self._nodes[client.node_id]
                logger.info(f"[Gateway] Node disconnected: {client.node_id}")
            logger.debug(f"[Gateway] Client disconnected: {client.cid}")

        return ws

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def start(self):
        """Bind the Unix socket and TCP port, then start accepting connections."""
        self._app = self._build_app()
        # Suppress access logging for high-frequency display polling endpoints
        _display_paths = ("/api/display/stream", "/api/display/status",
                          "/api/display/frame")

        class _FilteredAccessLogger(web.AccessLogger):
            def log(self, request, response, time):
                if request.path.startswith(_display_paths):
                    return
                super().log(request, response, time)

        self._runner = web.AppRunner(self._app,
                                     access_log_class=_FilteredAccessLogger)
        await self._runner.setup()

        # Unix socket
        if SOCKET_PATH.exists():
            SOCKET_PATH.unlink()

        unix_site = web.UnixSite(self._runner, str(SOCKET_PATH))
        await unix_site.start()
        self._sites.append(unix_site)
        os.chmod(SOCKET_PATH, stat.S_IRUSR | stat.S_IWUSR)

        # TCP listener(s)
        bind_hosts = [self._webchat_host]
        ts_ip = self._get_tailscale_ip() if self._webchat_ts else None
        if ts_ip and ts_ip not in bind_hosts:
            bind_hosts.append(ts_ip)

        for host in bind_hosts:
            tcp_site = web.TCPSite(self._runner, host, self._webchat_port)
            await tcp_site.start()
            self._sites.append(tcp_site)

        self._running = True
        self._hb_task = asyncio.create_task(self._heartbeat())

        # WiFi survival skill — activate if enabled in profile.env
        if os.environ.get("WIFI_HUNT_ENABLED", "false").lower() == "true":
            try:
                from opensable.skills.network.wifi_survival_skill import WifiSurvivalSkill
                self._wifi_skill = WifiSurvivalSkill(self.config)
                await self._wifi_skill.initialize()
            except Exception as _wifi_exc:
                logger.warning("[Gateway] WifiSurvivalSkill init failed: %s", _wifi_exc)

        # Wire up the interactive permission confirmation callback so that
        # "ask" permissions are forwarded to the user via WebSocket.
        self._wire_permission_confirmation()

        token_hint = f"?token={self._webchat_token}" if self._webchat_token else ""
        urls = [f"http://{h}:{self._webchat_port}{token_hint}" for h in bind_hosts]
        logger.info(
            f"[Gateway] Unix socket : {SOCKET_PATH}  (internal nodes)\n"
            + "\n".join(f"[Gateway] WebChat      : {u}" for u in urls)
        )

    async def stop(self):
        """Gracefully close all connections and remove the socket file."""
        self._running = False
        if self._hb_task:
            self._hb_task.cancel()
        for c in list(self._clients):
            if not c.ws.closed:
                await c.ws.close()
        if self._runner:
            await self._runner.cleanup()
        if SOCKET_PATH.exists():
            SOCKET_PATH.unlink()
        logger.info("[Gateway] Stopped")

    @staticmethod
    def _get_tailscale_ip() -> Optional[str]:
        """Return the Tailscale IP (100.x.x.x) if Tailscale is running."""
        import subprocess

        try:
            result = subprocess.run(
                ["tailscale", "ip", "-4"], capture_output=True, text=True, timeout=2
            )
            ip = result.stdout.strip()
            if re.match(r"^100\.\d+\.\d+\.\d+$", ip):
                return ip
        except Exception:
            pass
        return None

    @property
    def is_running(self) -> bool:
        return self._running

    def status(self) -> dict:
        uptime = (datetime.now(timezone.utc) - self._start_time).total_seconds()
        st = {
            "version": GATEWAY_VER,
            "running": self._running,
            "socket": str(SOCKET_PATH),
            "uptime_sec": round(uptime, 1),
            "clients": len(self._clients),
            "nodes": list(self._nodes.keys()),
            "start_time": self._start_time.isoformat(),
            "monitor_clients": len(self._monitor_clients),
        }
        # Attach token usage from the LLM
        try:
            if hasattr(self.agent, "llm") and hasattr(self.agent.llm, "token_tracker"):
                st["tokens"] = self.agent.llm.token_tracker.snapshot()
        except Exception:
            pass
        return st

    # ── Dispatch ──────────────────────────────────────────────────────────────

    async def _dispatch(self, client: _Client, msg: dict):
        t = msg.get("type", "")

        if t == "message":
            if not self._check_rate(client.cid):
                await client.send(
                    {"type": "error", "text": "Rate limit exceeded. Try again in a moment."}
                )
                return
            # Run as background task so cancel can be dispatched while agent is thinking
            task = asyncio.ensure_future(self._on_message(client, msg))
            client._chat_task = task
        elif t == "command":
            await self._on_command(client, msg)
        elif t == "sessions.list":
            await self._on_sessions_list(client)
        elif t == "sessions.history":
            await self._on_sessions_history(client, msg)
        elif t == "sessions.delete":
            await self._on_sessions_delete(client, msg)
        elif t == "node.register":
            await self._on_node_register(client, msg)
        elif t == "node.invoke":
            await self._on_node_invoke(client, msg)
        elif t == "node.result":
            await self._on_node_result(client, msg)
        elif t == "monitor.subscribe":
            await self._on_monitor_subscribe(client)
        elif t == "monitor.snapshot":
            await self._on_monitor_snapshot(client)
        elif t == "thoughts.list":
            await self._on_thoughts_list(client, msg)
        elif t == "tools.list":
            await self._on_tools_list(client)
        elif t == "status":
            status = self.status()
            if hasattr(self.agent, "llm") and hasattr(self.agent.llm, "current_model"):
                status["model"] = self.agent.llm.current_model
            try:
                from opensable import __version__ as _ver

                status["version"] = _ver
            except Exception:
                pass
            await client.send({"type": "status", **status})
        elif t == "models.list":
            await self._on_models_list(client, msg)
        elif t == "models.set":
            await self._on_models_set(client, msg)
        elif t == "models.import":
            await self._on_models_import(client, msg)
        elif t == "code.run":
            await self._on_code_run(client, msg)
        elif t == "code.autofix":
            await self._on_code_autofix(client, msg)
        elif t == "agents.list":
            await self._on_agents_list(client)
        elif t == "agents.start":
            await self._on_agents_start(client, msg)
        elif t == "agents.stop":
            await self._on_agents_stop(client, msg)
        elif t == "agents.status":
            await self._on_agents_status(client, msg)
        elif t == "agents.subscribe":
            await self._on_agents_subscribe(client, msg)
        elif t == "agents.unsubscribe":
            await self._on_agents_unsubscribe(client, msg)
        elif t == "agents.chat":
            await self._on_agents_chat(client, msg)
        elif t == "agents.brain.data":
            await self._on_agents_brain_data(client, msg)
        elif t == "brain.data":
            await self._on_brain_data(client, msg)
        elif t == "permission.response":
            await self._on_permission_response(client, msg)
        elif t == "llm.status":
            await self._on_llm_status(client, msg)
        elif t == "llm.models":
            await self._on_llm_models(client, msg)
        elif t == "llm.switch":
            await self._on_llm_switch(client, msg)
        elif t == "ping":
            await client.send({"type": "pong", "ts": time.time()})
        elif t == "message.cancel":
            await self._on_message_cancel(client, msg)
        elif t == "logs.tail":
            await self._on_logs_tail(client, msg)
        else:
            await client.send({"type": "error", "text": f"Unknown type: {t!r}"})

    # ── Handlers ──────────────────────────────────────────────────────────────

    # ── Permission confirmation flow ──────────────────────────────────────

    def _wire_permission_confirmation(self):
        """Connect the PermissionManager's ask-callback to the gateway WS."""
        try:
            pm = getattr(self.agent, "tools", None)
            pm = getattr(pm, "_permission_manager", None) if pm else None
            if pm and hasattr(pm, "set_confirmation_callback"):
                pm.set_confirmation_callback(self._ask_user_permission)
                logger.info("[Gateway] Permission confirmation callback wired via WebSocket")
        except Exception as e:
            logger.debug(f"[Gateway] Could not wire permission callback: {e}")

    async def _ask_user_permission(
        self, user_id: str, action: str, context: dict
    ) -> bool:
        """Send a confirmation request to all connected clients for *user_id*
        and await the first response (up to 60 s,  handled by the caller)."""
        import uuid as _uuid

        request_id = _uuid.uuid4().hex[:12]
        tool_name = context.get("tool", action)
        tool_args = context.get("arguments", {})

        pm = self.agent.tools._permission_manager
        fut = pm.create_pending(request_id)

        payload = {
            "type": "permission.request",
            "requestId": request_id,
            "action": action,
            "tool": tool_name,
            "arguments": {
                k: (str(v)[:200] if isinstance(v, str) else v)
                for k, v in (tool_args or {}).items()
            },
            "message": f"Allow {tool_name}?",
        }

        # Broadcast to all connected clients (any of them can answer)
        sent = 0
        for c in list(self._clients):
            if not c.ws.closed:
                try:
                    await c.send(payload)
                    sent += 1
                except Exception:
                    pass

        if sent == 0:
            logger.warning("[Gateway] No connected clients to ask for confirmation")
            pm.resolve_pending(request_id, False)

        return await fut

    async def _on_permission_response(self, client: _Client, msg: dict):
        """User responded to a permission.request."""
        request_id = msg.get("requestId", "")
        allowed = msg.get("allowed", False)
        remember = msg.get("remember", False)

        pm = getattr(self.agent, "tools", None)
        pm = getattr(pm, "_permission_manager", None) if pm else None
        if pm:
            pm.resolve_pending(request_id, bool(allowed))
            # Optionally persist the choice
            if remember and allowed:
                action_str = msg.get("action", "")
                try:
                    from opensable.core.security import ActionType, PermissionLevel
                    action = ActionType(action_str)
                    pm.set_permission("default", action, PermissionLevel.ALWAYS_ALLOW)
                    logger.info(f"[Gateway] Permission {action_str} set to always_allow (remembered)")
                except Exception:
                    pass
    async def _on_llm_status(self, client: _Client, msg: dict = None):
        """Return current LLM provider/model info."""
        profile = (msg or {}).get("profile", "")
        if profile and profile != _profile_name:
            return await self._proxy_llm_command(client, profile, {"type": "llm.status"})
        try:
            from opensable.core.llm import AdaptiveLLM
            llm = self.agent.llm
            provider = "ollama" if isinstance(llm, AdaptiveLLM) else getattr(llm, "provider", "unknown")
            model = llm.current_model
            available = getattr(llm, "available_models", [])
            tokens = llm.token_tracker.snapshot() if hasattr(llm, "token_tracker") else {}
            await client.send({
                "type": "llm.status",
                "provider": provider,
                "model": model,
                "available_models": available,
                "tokens": tokens,
                "providers_configured": self._get_configured_providers(),
            })
        except Exception as e:
            await client.send({"type": "llm.status", "error": str(e)})

    async def _on_llm_models(self, client: _Client, msg: dict = None):
        profile = (msg or {}).get("profile", "")
        if profile and profile != _profile_name:
            return await self._proxy_llm_command(client, profile, {"type": "llm.models"})
        """List all available models across all configured providers."""
        try:
            from opensable.core.llm import list_all_models
            models = await list_all_models(self.agent.config)
            await client.send({"type": "llm.models", "models": models})
        except Exception as e:
            await client.send({"type": "llm.models", "error": str(e), "models": {}})

    async def _on_llm_switch(self, client: _Client, msg: dict):
        """Switch LLM provider and/or model at runtime."""
        profile = msg.get("profile", "")
        if profile and profile != _profile_name:
            return await self._proxy_llm_command(client, profile, {
                "type": "llm.switch",
                "provider": msg.get("provider", "ollama"),
                "model": msg.get("model"),
            })
        try:
            from opensable.core.llm import switch_llm
            provider = msg.get("provider", "ollama")
            model = msg.get("model")
            result = switch_llm(self.agent, provider, model)
            await client.send({"type": "llm.switch", **result})
            # Broadcast model change to all monitor clients
            if result.get("success"):
                broadcast = {
                    "type": "llm.changed",
                    "provider": result["current"]["provider"],
                    "model": result["current"]["model"],
                }
                for mc in list(self._monitor_clients):
                    try:
                        await mc.send(broadcast)
                    except Exception:
                        pass
        except Exception as e:
            await client.send({"type": "llm.switch", "success": False, "error": str(e)})

    async def _proxy_llm_command(self, client: _Client, profile: str, payload: dict):
        """Proxy an llm.* command to a remote agent via Unix socket."""
        socket_path = f"/tmp/sable-{profile}.sock"
        if not Path(socket_path).exists():
            await client.send({"type": payload["type"], "success": False, "error": f"Agent '{profile}' is not running"})
            return
        try:
            resp = await _proxy_ws_request(socket_path, payload, timeout=10)
            resp["_profile"] = profile
            await client.send(resp)
        except Exception as exc:
            logger.warning(f"[Gateway] proxy llm command to {profile} failed: {exc}")
            await client.send({"type": payload["type"], "success": False, "error": str(exc), "_profile": profile})

    def _get_configured_providers(self) -> list:
        """Return list of providers that have API keys configured."""
        config = self.agent.config
        result = []
        # Ollama is always available if the agent started with it
        from opensable.core.llm import AdaptiveLLM
        if isinstance(self.agent.llm, AdaptiveLLM):
            result.append({"name": "ollama", "active": True})
        else:
            result.append({"name": "ollama", "active": False})

        provider_keys = [
            ("openwebui", "openwebui_api_key"),
            ("openai", "openai_api_key"),
            ("anthropic", "anthropic_api_key"),
            ("gemini", "gemini_api_key"),
            ("deepseek", "deepseek_api_key"),
            ("groq", "groq_api_key"),
            ("mistral", "mistral_api_key"),
            ("together", "together_api_key"),
            ("xai", "xai_api_key"),
            ("cohere", "cohere_api_key"),
            ("openrouter", "openrouter_api_key"),
        ]
        current_provider = "ollama" if isinstance(self.agent.llm, AdaptiveLLM) else getattr(self.agent.llm, "provider", "")
        for name, key_attr in provider_keys:
            has_key = bool(getattr(config, key_attr, None))
            result.append({
                "name": name,
                "configured": has_key,
                "active": current_provider == name,
            })
        return result

    async def _on_tools_list(self, client: _Client):
        tool_names: list = []
        try:
            if hasattr(self.agent, "tools") and self.agent.tools:
                tool_names = self.agent.tools.list_tools()
        except Exception as exc:
            logger.debug(f"[Gateway] tools.list error: {exc}")
        await client.send({"type": "tools.list.result", "tools": tool_names})

    # Provider id → (config_key, display_name, default_models)
    _MODEL_PROVIDERS: list[tuple[str, str, str, list[str]]] = [
        ("openai",    "openai_api_key",    "OpenAI",    ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "o1", "o1-mini", "o3-mini"]),
        ("anthropic", "anthropic_api_key", "Anthropic", ["claude-opus-4-5", "claude-sonnet-4-5", "claude-3-5-sonnet-20241022"]),
        ("gemini",    "gemini_api_key",    "Google",    ["gemini-2.0-flash", "gemini-2.0-flash-lite", "gemini-1.5-pro"]),
        ("deepseek",  "deepseek_api_key",  "DeepSeek",  ["deepseek-chat", "deepseek-reasoner"]),
        ("groq",      "groq_api_key",      "Groq",      ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "mixtral-8x7b-32768"]),
        ("xai",       "xai_api_key",       "xAI",       ["grok-3-mini-fast", "grok-2-latest"]),
        ("mistral",   "mistral_api_key",   "Mistral",   ["mistral-large-latest", "mistral-small-latest"]),
        ("openrouter","openrouter_api_key","OpenRouter", ["openai/gpt-4o", "anthropic/claude-3.5-sonnet"]),
        ("together",  "together_api_key",  "Together",  ["meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"]),
    ]

    async def _on_models_list(self, client: _Client, msg: dict = None):
        """Return all available models: local Ollama + OpenWebUI + configured cloud providers."""
        profile = (msg or {}).get("profile", "")
        if profile and profile != _profile_name:
            # Proxy to remote agent
            socket_path = f"/tmp/sable-{profile}.sock"
            if Path(socket_path).exists():
                try:
                    resp = await _proxy_ws_request(
                        socket_path, {"type": "models.list"},
                        expected_type="models.list.result", timeout=10,
                    )
                    resp["_profile"] = profile
                    await client.send(resp)
                    return
                except Exception as exc:
                    logger.warning(f"[Gateway] models.list proxy to {profile} failed: {exc}")
            # Remote agent not reachable — fall through to list local models
            # so the user can still see and select available models
        groups: list[dict] = []  # [{provider, name, models: [{name, active}]}]
        current_model: str = ""
        provider_type: str = "ollama"  # default
        try:
            llm = getattr(self.agent, "llm", None)
            if llm:
                current_model = getattr(llm, "current_model", "") or ""
                provider_type = getattr(llm, "provider", "ollama")

                # ── Local Ollama models ──
                if hasattr(llm, "_update_available_models"):
                    llm._update_available_models()
                local_models = getattr(llm, "available_models", [])
                if local_models:
                    groups.append({
                        "provider": "ollama",
                        "name": "Ollama (Local)",
                        "models": [
                            {"name": (m if isinstance(m, str) else str(m)),
                             "active": (m if isinstance(m, str) else str(m)) == current_model}
                            for m in local_models if m
                        ],
                    })

                # ── OpenWebUI models (fetched live from API) ──
                config = getattr(self.agent, "config", None)
                if config:
                    owui_url = getattr(config, "openwebui_api_url", None)
                    owui_key = getattr(config, "openwebui_api_key", None)
                    if owui_url and owui_key:
                        try:
                            from opensable.core.llm import check_openwebui_models
                            owui_models = await check_openwebui_models(owui_url, owui_key)
                            if owui_models:
                                groups.append({
                                    "provider": "openwebui",
                                    "name": f"Open WebUI ({owui_url.replace('https://', '').replace('http://', '').split('/')[0]})",
                                    "models": [
                                        {"name": m, "active": m == current_model and provider_type == "openwebui"}
                                        for m in owui_models
                                    ],
                                })
                        except Exception as e:
                            logger.debug(f"[Gateway] OpenWebUI model fetch error: {e}")

                    # ── Cloud providers with configured API keys ──
                    for pid, key_attr, display_name, default_models in self._MODEL_PROVIDERS:
                        if getattr(config, key_attr, None):
                            groups.append({
                                "provider": pid,
                                "name": display_name,
                                "models": [
                                    {"name": m, "active": m == current_model}
                                    for m in default_models
                                ],
                            })
        except Exception as exc:
            logger.debug(f"[Gateway] models.list error: {exc}")
        result = {
            "type": "models.list.result",
            "groups": groups,
            "current": current_model,
            "provider": provider_type,
        }
        if profile and profile != _profile_name:
            result["_profile"] = profile
        await client.send(result)

    async def _on_models_set(self, client: _Client, msg: dict):
        """Switch the active model.  Supports both local and cloud models."""
        profile = msg.get("profile", "")
        if profile and profile != _profile_name:
            # Proxy to remote agent via models.set, return models.set.result
            socket_path = f"/tmp/sable-{profile}.sock"
            if Path(socket_path).exists():
                try:
                    resp = await _proxy_ws_request(
                        socket_path,
                        {"type": "models.set", "model": msg.get("model", ""), "provider": msg.get("provider", "")},
                        expected_type="models.set.result", timeout=10,
                    )
                    resp["_profile"] = profile
                    await client.send(resp)
                    return
                except Exception as exc:
                    logger.warning(f"[Gateway] models.set proxy to {profile} failed: {exc}")
            # Remote agent not reachable — fall through to set model locally
            # since all agents share the same Ollama instance
        model_name: str = msg.get("model", "").strip()
        provider: str = msg.get("provider", "").strip()  # optional: force provider
        if not model_name:
            await client.send({"type": "models.set.result", "success": False, "error": "No model specified"})
            return

        # If the agent is busy thinking, cancel the current request first
        # so the model switch isn't blocked waiting for inference to finish.
        if getattr(self.agent, "_user_chatting", False):
            self.agent.cancel_current_request()
            # Brief yield so the cancelled task can clean up
            await asyncio.sleep(0.1)

        try:
            llm = getattr(self.agent, "llm", None)
            if not llm:
                raise RuntimeError("LLM not initialised")

            # If switching to a cloud provider model, we need to swap the LLM instance
            if provider and provider != "ollama":
                from opensable.core.llm import CloudLLM
                config = self.agent.config
                cloud_llm = CloudLLM(provider=provider, config=config)
                cloud_llm.current_model = model_name
                self.agent.llm = cloud_llm
                logger.info(f"[Gateway] Switched to cloud LLM: {provider}/{model_name}")
            else:
                # Local Ollama model switch
                llm.current_model = model_name
                logger.info(f"[Gateway] Switched model → {model_name}")

            await client.send({
                "type": "models.set.result",
                "success": True,
                "model": model_name,
                "provider": provider or "ollama",
            })
            # Broadcast status update so all clients see the new model
            for c in list(self._clients):
                try:
                    await c.send({"type": "status", "model": model_name})
                except Exception:
                    pass
        except Exception as exc:
            logger.error(f"[Gateway] models.set error: {exc}")
            await client.send({"type": "models.set.result", "success": False, "error": str(exc)})

    async def _on_models_import(self, client: _Client, msg: dict):
        """Import a local GGUF model file into Ollama.

        Message format:
            { "type": "models.import", "path": "/path/to/model.gguf", "name": "my-model" }

        This creates a temporary Modelfile pointing to the GGUF then runs
        ``ollama create <name> -f <Modelfile>`` in a subprocess.
        """
        import tempfile, os as _os

        file_path: str = msg.get("path", "").strip()
        model_name: str = msg.get("name", "").strip()

        if not file_path:
            await client.send({"type": "models.import.result", "success": False, "error": "No file path provided"})
            return

        file_path = _os.path.expanduser(file_path)
        if not _os.path.isfile(file_path):
            await client.send({"type": "models.import.result", "success": False,
                                "error": f"File not found: {file_path}"})
            return

        if not file_path.lower().endswith(".gguf"):
            await client.send({"type": "models.import.result", "success": False,
                                "error": "Only .gguf files are supported"})
            return

        # Derive a model name from the filename if not provided
        if not model_name:
            base = _os.path.basename(file_path)
            model_name = base.rsplit(".", 1)[0].lower().replace(" ", "-")

        logger.info(f"[Gateway] Importing GGUF → {model_name} from {file_path}")

        try:
            # Write a temporary Modelfile
            with tempfile.NamedTemporaryFile("w", suffix=".Modelfile", delete=False) as mf:
                mf.write(f"FROM {file_path}\n")
                modelfile_path = mf.name

            proc = await asyncio.create_subprocess_exec(
                "ollama", "create", model_name, "-f", modelfile_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=600)

            # Clean up Modelfile
            try:
                _os.unlink(modelfile_path)
            except OSError:
                pass

            if proc.returncode == 0:
                logger.info(f"[Gateway] GGUF import successful: {model_name}")
                await client.send({
                    "type": "models.import.result",
                    "success": True,
                    "model": model_name,
                    "message": stdout.decode(errors="replace").strip(),
                })
            else:
                err_text = stderr.decode(errors="replace").strip() or stdout.decode(errors="replace").strip()
                logger.error(f"[Gateway] GGUF import failed: {err_text}")
                await client.send({
                    "type": "models.import.result",
                    "success": False,
                    "error": err_text or "ollama create failed",
                })
        except asyncio.TimeoutError:
            logger.error("[Gateway] GGUF import timed out (600s)")
            await client.send({"type": "models.import.result", "success": False, "error": "Import timed out (10 min limit)"})
        except FileNotFoundError:
            await client.send({"type": "models.import.result", "success": False,
                                "error": "Ollama CLI not found. Is Ollama installed and in PATH?"})
        except Exception as exc:
            logger.error(f"[Gateway] GGUF import error: {exc}")
            await client.send({"type": "models.import.result", "success": False, "error": str(exc)})

    async def _on_code_run(self, client: _Client, msg: dict):
        """Execute a code snippet sent from the desktop chat Run button."""
        import tempfile, os as _os

        request_id: str = msg.get("request_id", "")
        code: str = msg.get("code", "")
        stdin: str | None = msg.get("stdin")
        language: str = (msg.get("language") or "python").lower()

        _LANG_RUNNERS: dict[str, list[str]] = {
            "python":     ["python3", "-u"],
            "python3":    ["python3", "-u"],
            "javascript": ["node"],
            "js":         ["node"],
            "bash":       ["bash"],
            "sh":         ["sh"],
        }

        runner = _LANG_RUNNERS.get(language)
        if not runner:
            await client.send({
                "type": "code.result",
                "request_id": request_id,
                "stdout": "",
                "stderr": f"Language '{language}' is not supported for execution.",
                "exit_code": 1,
            })
            return

        # Write code to a temp file so multi-line scripts work cleanly
        suffix_map = {
            "python": ".py", "python3": ".py",
            "javascript": ".js", "js": ".js",
            "bash": ".sh", "sh": ".sh",
        }
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=suffix_map.get(language, ".tmp"),
                delete=False, encoding="utf-8"
            ) as tf:
                tf.write(code)
                tmp_path = tf.name

            cmd = " ".join(runner + [tmp_path])
            # If stdin was provided, prefer direct subprocess so we can feed input.
            if stdin is not None:
                proc = await asyncio.create_subprocess_shell(
                    cmd,
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                _out, _err = await asyncio.wait_for(
                    proc.communicate(input=stdin.encode('utf-8')),
                    timeout=30,
                )
                stdout = _out.decode(errors='replace') if _out else ""
                stderr = _err.decode(errors='replace') if _err else ""
                exit_code = getattr(proc, 'returncode', 0) or 0
            else:
                if hasattr(self.agent, "tools") and self.agent.tools:
                    # Tools.execute returns the command output as a string; cannot provide stdin
                    raw = await self.agent.tools.execute(
                        "execute_command",
                        {"command": cmd, "timeout": 30},
                    )
                    stdout = str(raw) if raw else ""
                    stderr = ""
                    exit_code = 0
                else:
                    # Fallback: run directly via asyncio subprocess without stdin
                    proc = await asyncio.create_subprocess_shell(
                        cmd,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE,
                    )
                    _out, _err = await asyncio.wait_for(proc.communicate(), timeout=30)
                    stdout = _out.decode(errors='replace') if _out else ""
                    stderr = _err.decode(errors='replace') if _err else ""
                    exit_code = getattr(proc, 'returncode', 0) or 0

            await client.send({
                "type": "code.result",
                "request_id": request_id,
                "stdout": stdout,
                "stderr": stderr,
                "exit_code": exit_code,
            })
        except asyncio.TimeoutError:
            await client.send({
                "type": "code.result",
                "request_id": request_id,
                "stdout": "",
                "stderr": "Execution timed out (30s limit).",
                "exit_code": 124,
            })
        except Exception as exc:
            await client.send({
                "type": "code.result",
                "request_id": request_id,
                "stdout": "",
                "stderr": str(exc),
                "exit_code": 1,
            })
        finally:
            try:
                if tmp_path:
                    _os.unlink(tmp_path)
            except Exception:
                pass

    async def _on_code_autofix(self, client: _Client, msg: dict):
        """Attempt a simple automatic fix for interactive Python snippets that raise EOF.

        Strategy (conservative): when code contains `input(`, wrap it so inputs
        are consumed from `sys.argv[1:]`. Return the patched code and run it.
        """
        import tempfile, os as _os

        request_id: str = msg.get("request_id", "")
        code: str = msg.get("code", "")
        language: str = (msg.get("language") or "python").lower()

        if language not in ("python", "python3"):
            await client.send({
                "type": "code.result",
                "request_id": request_id,
                "stdout": "",
                "stderr": f"Autofix not supported for language: {language}",
                "exit_code": 1,
            })
            return

        if "input(" not in code:
            await client.send({
                "type": "code.result",
                "request_id": request_id,
                "stdout": "",
                "stderr": "No interactive `input()` calls detected; autofix skipped.",
                "exit_code": 1,
            })
            return

        # Build patched code that replaces input() calls with a helper that
        # consumes values from sys.argv[1:]
        header = (
            "import sys\n"
            "_stdin_values = list(sys.argv[1:])\n"
            "def _get_input(prompt=None):\n"
            "    if _stdin_values:\n"
            "        return _stdin_values.pop(0)\n"
            "    raise EOFError('No simulated stdin provided')\n\n"
        )

        patched = code.replace("input(", "_get_input(")
        patched_code = header + patched

        # Write and run patched code
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8") as tf:
                tf.write(patched_code)
                tmp_path = tf.name

            cmd = "python3 -u " + tmp_path
            proc = await asyncio.create_subprocess_shell(
                cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            _out, _err = await asyncio.wait_for(proc.communicate(), timeout=30)
            stdout = _out.decode(errors="replace") if _out else ""
            stderr = _err.decode(errors="replace") if _err else ""
            exit_code = getattr(proc, 'returncode', 0) or 0

            # Return both the patched code (so UI can show it) and the run result
            await client.send({
                "type": "code.result",
                "request_id": request_id,
                "stdout": stdout,
                "stderr": stderr,
                "exit_code": exit_code,
                "autofix": True,
                "patched_code": patched_code,
            })
        except asyncio.TimeoutError:
            await client.send({
                "type": "code.result",
                "request_id": request_id,
                "stdout": "",
                "stderr": "Execution timed out (30s).",
                "exit_code": 124,
                "autofix": True,
                "patched_code": patched_code,
            })
        except Exception as exc:
            await client.send({
                "type": "code.result",
                "request_id": request_id,
                "stdout": "",
                "stderr": str(exc),
                "exit_code": 1,
                "autofix": True,
                "patched_code": patched_code,
            })
        finally:
            try:
                if tmp_path:
                    _os.unlink(tmp_path)
            except Exception:
                pass

    # ── Multi-Agent handlers ──────────────────────────────────────────────────

    async def _on_logs_tail(self, client: _Client, msg: dict):
        """Return the last N lines of an agent's log file."""
        import re as _re
        profile = msg.get("profile", "") or _profile_name
        lines = min(int(msg.get("lines", 100)), 500)
        # Sanitize profile name to prevent path traversal
        if not _re.match(r'^[a-zA-Z0-9_-]+$', profile):
            await client.send({"type": "logs.tail.result", "lines": [], "error": "Invalid profile name"})
            return
        log_path = Path(__file__).resolve().parent.parent.parent / "logs" / f"sable-{profile}.log"
        result_lines = []
        if log_path.exists():
            try:
                with open(log_path, "r", encoding="utf-8", errors="replace") as f:
                    # Read last N lines efficiently
                    all_lines = f.readlines()
                    result_lines = [l.rstrip() for l in all_lines[-lines:]]
            except Exception as exc:
                logger.debug(f"[Gateway] logs.tail error: {exc}")
        await client.send({
            "type": "logs.tail.result",
            "profile": profile,
            "lines": result_lines,
        })

    async def _on_agents_list(self, client: _Client):
        """Return list of all agent profiles and their running status."""
        try:
            from opensable.core.profile import list_profiles
        except ImportError:
            await client.send({"type": "agents.list.result", "agents": [], "current": _profile_name})
            return

        agents = []
        for name in list_profiles():
            sock = Path(f"/tmp/sable-{name}.sock")
            agents.append({
                "name": name,
                "running": sock.exists(),
                "is_current": name == _profile_name,
            })

        await client.send({
            "type": "agents.list.result",
            "agents": agents,
            "current": _profile_name,
        })

    async def _on_agents_start(self, client: _Client, msg: dict):
        """Start a remote agent via AgentManager."""
        profile = msg.get("profile", "")
        if not profile:
            await client.send({"type": "agents.start.result", "success": False, "error": "Missing 'profile'"})
            return
        if profile == _profile_name:
            await client.send({"type": "agents.start.result", "success": False, "error": "Cannot start the current agent from itself", "profile": profile})
            return
        mgr = getattr(self, "_agent_manager", None)
        if not mgr:
            await client.send({"type": "agents.start.result", "success": False, "error": "Agent manager not available", "profile": profile})
            return
        try:
            info = await mgr.start_child(profile)
            if info:
                await client.send({"type": "agents.start.result", "success": True, "profile": profile, "pid": info.get("pid")})
                # Refresh agent list for all clients
                for c in list(self._clients):
                    await self._on_agents_list(c)
            else:
                await client.send({"type": "agents.start.result", "success": False, "error": f"Failed to start '{profile}'", "profile": profile})
        except Exception as exc:
            logger.warning(f"[Gateway] agents.start {profile} failed: {exc}")
            await client.send({"type": "agents.start.result", "success": False, "error": str(exc), "profile": profile})

    async def _on_agents_stop(self, client: _Client, msg: dict):
        """Stop a remote agent via AgentManager."""
        profile = msg.get("profile", "")
        if not profile:
            await client.send({"type": "agents.stop.result", "success": False, "error": "Missing 'profile'"})
            return
        if profile == _profile_name:
            await client.send({"type": "agents.stop.result", "success": False, "error": "Cannot stop the current agent from itself", "profile": profile})
            return
        mgr = getattr(self, "_agent_manager", None)
        if not mgr:
            await client.send({"type": "agents.stop.result", "success": False, "error": "Agent manager not available", "profile": profile})
            return
        try:
            stopped = await mgr.stop_child(profile)
            await client.send({"type": "agents.stop.result", "success": stopped, "profile": profile, "error": "" if stopped else f"Agent '{profile}' was not running"})
            # Refresh agent list for all clients
            for c in list(self._clients):
                await self._on_agents_list(c)
        except Exception as exc:
            logger.warning(f"[Gateway] agents.stop {profile} failed: {exc}")
            await client.send({"type": "agents.stop.result", "success": False, "error": str(exc), "profile": profile})

    async def _on_agents_status(self, client: _Client, msg: dict):
        """Get status from a specific agent, proxied via Unix socket."""
        profile = msg.get("profile", "")
        if not profile:
            await client.send({"type": "error", "text": "agents.status requires 'profile'"})
            return

        if profile == _profile_name:
            # Current agent – return local status
            status = self.status()
            if hasattr(self.agent, "llm") and hasattr(self.agent.llm, "current_model"):
                status["model"] = self.agent.llm.current_model
            try:
                from opensable import __version__ as _ver
                status["version"] = _ver
            except Exception:
                pass
            await client.send({"type": "agents.status.result", "profile": profile, **status})
            return

        # Proxy to peer agent via Unix socket
        socket_path = f"/tmp/sable-{profile}.sock"
        if not Path(socket_path).exists():
            await client.send({
                "type": "agents.status.result",
                "profile": profile,
                "running": False,
            })
            return

        try:
            resp = await _proxy_ws_request(
                socket_path, {"type": "status"},
                expected_type="status", timeout=5,
            )
            await client.send({
                "type": "agents.status.result",
                "profile": profile,
                **resp,
            })
        except Exception as exc:
            logger.debug(f"[Gateway] agents.status proxy to {profile} failed: {exc}")
            await client.send({
                "type": "agents.status.result",
                "profile": profile,
                "running": False,
                "error": str(exc),
            })

    async def _on_agents_subscribe(self, client: _Client, msg: dict):
        """Subscribe to real-time events from another agent via Unix socket proxy."""
        profile = msg.get("profile", "")
        if not profile:
            await client.send({"type": "error", "text": "agents.subscribe requires 'profile'"})
            return

        if profile == _profile_name:
            # Already connected to this agent
            await client.send({
                "type": "agents.subscribed",
                "profile": profile,
                "status": "already_connected",
            })
            return

        # Cancel existing proxy for this profile if any
        if profile in client._proxy_tasks:
            client._proxy_tasks[profile].cancel()
            del client._proxy_tasks[profile]

        socket_path = f"/tmp/sable-{profile}.sock"
        if not Path(socket_path).exists():
            await client.send({
                "type": "agents.subscribed",
                "profile": profile,
                "status": "offline",
            })
            return

        # Start a background task to proxy events from the peer agent
        task = asyncio.create_task(self._proxy_agent_events(client, profile, socket_path))
        client._proxy_tasks[profile] = task

    async def _on_agents_unsubscribe(self, client: _Client, msg: dict):
        """Unsubscribe from a remote agent's events."""
        profile = msg.get("profile", "")
        if profile in client._proxy_tasks:
            client._proxy_tasks[profile].cancel()
            del client._proxy_tasks[profile]
        await client.send({"type": "agents.unsubscribed", "profile": profile})

    async def _proxy_agent_events(self, client: _Client, profile: str, socket_path: str):
        """Forward all events from a remote agent to this dashboard client."""
        try:
            conn = aiohttp.UnixConnector(path=socket_path)
            async with aiohttp.ClientSession(connector=conn) as session:
                async with session.ws_connect("http://localhost/") as ws:
                    await client.send({
                        "type": "agents.subscribed",
                        "profile": profile,
                        "status": "connected",
                    })

                    # Request initial state
                    await ws.send_json({"type": "status"})
                    await ws.send_json({"type": "sessions.list"})
                    await ws.send_json({"type": "monitor.subscribe"})
                    await ws.send_json({"type": "thoughts.list", "limit": 500})
                    await ws.send_json({"type": "models.list"})

                    # Periodically request status updates
                    last_status = time.time()

                    async for msg in ws:
                        if client.closed:
                            break
                        if msg.type == aiohttp.WSMsgType.TEXT:
                            try:
                                data = json.loads(msg.data)
                            except json.JSONDecodeError:
                                continue
                            # Tag every message with the source profile
                            data["_profile"] = profile
                            await client.send(data)
                        elif msg.type in (aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.ERROR):
                            break

                        # Request periodic status
                        now = time.time()
                        if now - last_status >= 5:
                            try:
                                await ws.send_json({"type": "status"})
                            except Exception:
                                pass
                            last_status = now

        except asyncio.CancelledError:
            logger.debug(f"[Gateway] Agent proxy to {profile} cancelled")
        except Exception as exc:
            logger.debug(f"[Gateway] Agent proxy to {profile} ended: {exc}")
        finally:
            # Notify the dashboard client that the proxy disconnected
            try:
                await client.send({
                    "type": "agents.disconnected",
                    "profile": profile,
                })
            except Exception:
                pass

    async def _on_agents_brain_data(self, client: _Client, msg: dict):
        """Proxy brain.data request to a remote agent via Unix socket."""
        profile = msg.get("profile", "")
        if not profile:
            await client.send({"type": "error", "text": "agents.brain.data requires 'profile'"})
            return

        # If targeting the current agent, just handle locally
        if profile == _profile_name:
            await self._on_brain_data(client, msg)
            return

        socket_path = f"/tmp/sable-{profile}.sock"
        if not Path(socket_path).exists():
            await client.send({
                "type": "brain.data.result",
                "_profile": profile,
                "error": f"Agent '{profile}' is not running",
            })
            return

        try:
            conn = aiohttp.UnixConnector(path=socket_path)
            async with aiohttp.ClientSession(connector=conn) as session:
                async with session.ws_connect("http://localhost/") as ws:
                    await ws.send_json({"type": "brain.data"})
                    async for frame in ws:
                        if client.closed:
                            break
                        if frame.type == aiohttp.WSMsgType.TEXT:
                            try:
                                data = json.loads(frame.data)
                            except json.JSONDecodeError:
                                continue
                            if data.get("type") == "brain.data.result":
                                data["_profile"] = profile
                                await client.send(data)
                                break
                        elif frame.type in (aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.ERROR):
                            break
        except Exception as exc:
            logger.warning(f"[Gateway] agents.brain.data proxy to {profile} failed: {exc}")
            await client.send({
                "type": "brain.data.result",
                "_profile": profile,
                "error": str(exc),
            })

    async def _on_agents_chat(self, client: _Client, msg: dict):
        """Proxy a chat message to a remote agent via Unix socket and stream the response back."""
        from opensable.core.session_manager import SessionManager

        profile = msg.get("profile", "")
        text = msg.get("text", "").strip()
        session_id = msg.get("session_id", "webchat_default")
        user_id = msg.get("user_id", "dashboard_user")

        if not profile or not text:
            await client.send({"type": "error", "text": "agents.chat requires 'profile' and 'text'"})
            return

        # If targeting the current agent, just handle locally
        if profile == _profile_name:
            task = asyncio.ensure_future(self._on_message(client, msg))
            client._chat_task = task
            return

        socket_path = f"/tmp/sable-{profile}.sock"
        if not Path(socket_path).exists():
            # Auto-start the agent so the user's message isn't lost
            mgr = getattr(self, "_agent_manager", None)
            if not mgr:
                await client.send({"type": "error", "text": f"Agent '{profile}' is not running and agent manager is unavailable"})
                return
            await client.send({
                "type": "progress",
                "_profile": profile,
                "text": f"Starting agent '{profile}'…",
            })
            try:
                info = await mgr.start_child(profile)
                if not info:
                    await client.send({"type": "error", "text": f"Failed to start agent '{profile}'"})
                    return
                # Broadcast updated agent list to all clients
                for c in list(self._clients):
                    try:
                        await self._on_agents_list(c)
                    except Exception:
                        pass
            except Exception as exc:
                logger.warning(f"[Gateway] agents.chat auto-start {profile} failed: {exc}")
                await client.send({"type": "error", "text": f"Could not auto-start agent '{profile}': {exc}"})
                return

        # Save user message to local session so it appears in sidebar
        sm = SessionManager()
        local_session = sm.get_session(session_id)
        if not local_session:
            local_session = sm.create_session(
                channel="webchat", user_id=user_id, session_id=session_id
            )
            local_session.metadata["agent_profile"] = profile
        local_session.add_message("user", text)
        sm._save_session(local_session)

        reply_chunks = []  # Collect reply text for local session

        try:
            conn = aiohttp.UnixConnector(path=socket_path)
            async with aiohttp.ClientSession(connector=conn) as session:
                async with session.ws_connect(
                    "http://localhost/",
                    heartbeat=30,
                    receive_timeout=300,
                ) as ws:
                    # Send the chat message to the remote agent
                    await ws.send_json({
                        "type": "message",
                        "text": text,
                        "session_id": session_id,
                        "user_id": user_id,
                    })

                    # Stream back all response frames until message.done
                    done = False
                    async for frame in ws:
                        if client.closed:
                            break
                        if frame.type == aiohttp.WSMsgType.TEXT:
                            try:
                                data = json.loads(frame.data)
                            except json.JSONDecodeError:
                                continue
                            ftype = data.get("type", "")
                            # Tag with source profile
                            data["_profile"] = profile
                            # Collect streaming chunks for local session
                            if ftype == "message.chunk":
                                reply_chunks.append(data.get("text", ""))
                            # Forward relevant message types
                            if ftype in ("message.start", "message.chunk", "message.done",
                                         "progress", "error"):
                                await client.send(data)
                            if ftype == "message.done":
                                # Grab full reply text from done frame if available
                                done_text = data.get("text", "")
                                if done_text:
                                    reply_chunks = [done_text]
                                done = True
                                break
                        elif frame.type in (aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.ERROR):
                            break

                    if not done:
                        await client.send({
                            "type": "message.done",
                            "_profile": profile,
                            "session_id": session_id,
                            "text": "",
                        })

        except Exception as exc:
            logger.warning(f"[Gateway] agents.chat proxy to {profile} failed: {exc}")
            await client.send({
                "type": "error",
                "_profile": profile,
                "text": f"Failed to chat with agent '{profile}': {exc}",
            })

        # Save assistant reply to local session and refresh sidebar
        reply_text = "".join(reply_chunks).strip()
        if reply_text:
            local_session.add_message("assistant", reply_text)
            sm._save_session(local_session)
        # Refresh session list for the client
        try:
            await self._on_sessions_list(client)
        except Exception:
            pass

    async def _on_message(self, client: _Client, msg: dict):
        """Process user message through the full agent pipeline."""
        from opensable.core.session_manager import SessionManager

        sid = msg.get("session_id", "webchat_default")
        text = msg.get("text", "").strip()
        user_id = msg.get("user_id", "webchat_user")

        if not text:
            return

        await client.send({"type": "message.start", "session_id": sid})

        async def _progress(status_text: str):
            try:
                await client.send({"type": "progress", "session_id": sid, "text": status_text})
            except Exception:
                pass

        # Streaming chunk callback,  emits tokens word-by-word in real time
        _first_chunk_sent = []  # use list as mutable flag (avoids nonlocal in all Pythons)

        async def _stream_chunk(token: str):
            try:
                # Signal to the frontend that the model has started responding
                if not _first_chunk_sent:
                    _first_chunk_sent.append(True)
                    await client.send({"type": "progress", "session_id": sid, "text": "Responding…"})
                await client.send({"type": "message.chunk", "session_id": sid, "text": token})
            except Exception:
                pass

        try:
            sm = SessionManager()
            session = sm.get_session(sid)
            if not session:
                session = sm.create_session(
                    channel="webchat", user_id=user_id, session_id=sid
                )
            history = [
                {"role": m.role, "content": m.content}
                for m in session.get_messages()[-30:]
            ]

            reply = await self.agent.process_message(
                user_id, text, history=history,
                progress_callback=_progress,
                stream_chunk_callback=_stream_chunk,
            )
            raw_reply = reply
            reply = _clean_gateway_reply(reply or "")

            if not reply:
                logger.warning(
                    f"[Gateway] Empty reply after cleaning (raw length={len(raw_reply or '')}). "
                    f"Falling back to raw or default."
                )
                reply = (raw_reply or "").strip() or "I processed your request but couldn't generate a response. Please try again."

            # Deduplicate: if the reply is identical to the last assistant message,
            # the model is repeating itself,  ask it to try again without history
            if history:
                last_assistant = next(
                    (m["content"] for m in reversed(history) if m.get("role") == "assistant"),
                    None,
                )
                if last_assistant and reply.strip() == last_assistant.strip():
                    logger.warning("[Gateway] Duplicate response detected,  retrying without history")
                    try:
                        retry_reply = await self.agent.process_message(
                            user_id, text, history=[], progress_callback=_progress
                        )
                        retry_reply = _clean_gateway_reply(retry_reply or "")
                        if retry_reply and retry_reply.strip() != last_assistant.strip():
                            reply = retry_reply
                    except Exception:
                        pass  # keep original reply

            try:
                if not session.metadata.get("title"):
                    session.metadata["title"] = text[:60]
                    session.updated_at = datetime.now(timezone.utc).isoformat()
                session.add_message("user", text)
                session.add_message("assistant", reply)
                sm._save_session(session)
            except Exception as _se:
                logger.debug(f"[Gateway] session persist error: {_se}")

            await client.send({"type": "message.done", "session_id": sid, "text": reply})
        except asyncio.CancelledError:
            logger.info(f"[Gateway] Request cancelled for session {sid}")
            await client.send({"type": "message.done", "session_id": sid, "text": "*(Request cancelled)*"})
        except Exception as exc:
            logger.warning(f"[Gateway] Agent processing failed: {exc}")
            await client.send({"type": "error", "session_id": sid, "text": str(exc)})

    async def _on_message_cancel(self, client: _Client, msg: dict):
        """Cancel the currently running LLM request."""
        # Cancel the agent-level task
        cancelled = self.agent.cancel_current_request()
        # Also cancel the gateway-level chat task if still running
        chat_task = getattr(client, '_chat_task', None)
        if chat_task and not chat_task.done():
            chat_task.cancel()
            cancelled = True
        await client.send({
            "type": "message.cancel.result",
            "success": cancelled,
            "session_id": msg.get("session_id", ""),
        })
        if cancelled:
            # Send message.done so the UI exits streaming state
            await client.send({
                "type": "message.done",
                "session_id": msg.get("session_id", ""),
                "text": "*(Request cancelled)*",
            })

    async def _on_command(self, client: _Client, msg: dict):
        from opensable.core.commands import CommandHandler
        from opensable.core.session_manager import SessionManager

        sm = SessionManager()
        ch = CommandHandler(sm)
        sid = msg.get("session_id", "webchat_default")
        uid = msg.get("user_id", "webchat_user")
        txt = msg.get("text", "")
        s = sm.get_or_create_session(channel="webchat", user_id=uid)
        res = await ch.handle_command(txt, s.id, uid, is_admin=True)
        await client.send({
            "type": "command.result",
            "session_id": sid,
            "text": res.message,
            "success": res.success,
        })

    async def _on_sessions_list(self, client: _Client):
        from opensable.core.session_manager import SessionManager

        sm = SessionManager()
        sessions = [
            {
                "id": s.id,
                "session_id": s.id,
                "channel": s.channel,
                "user_id": s.user_id,
                "agent_profile": s.metadata.get("agent_profile", ""),
                "title": s.metadata.get("title")
                or next(
                    (m.content[:60] for m in s.messages if m.role == "user"), None
                )
                or s.id[:12],
                "messages": len(s.messages),
                "created_at": s.created_at,
                "updated_at": s.updated_at,
            }
            for s in sm.list_sessions(channel="webchat")
            if len(s.messages) > 0
        ]
        sessions.sort(key=lambda x: x["updated_at"], reverse=True)
        await client.send({"type": "sessions.list.result", "sessions": sessions})

    async def _on_sessions_history(self, client: _Client, msg: dict):
        from opensable.core.session_manager import SessionManager

        sm = SessionManager()
        sid = msg.get("session_id", "")
        s = sm.get_session(sid)
        msgs = [m.to_dict() for m in s.get_messages()] if s else []
        await client.send({"type": "sessions.history.result", "session_id": sid, "messages": msgs})

    async def _on_sessions_delete(self, client: _Client, msg: dict):
        from opensable.core.session_manager import SessionManager

        sm = SessionManager()
        sid = msg.get("session_id", "")
        if not sid:
            await client.send({"type": "sessions.delete.result", "success": False, "error": "No session_id provided"})
            return
        ok = sm.delete_session(sid)
        await client.send({"type": "sessions.delete.result", "success": ok, "session_id": sid})
        # Refresh the session list for this client
        if ok:
            await self._on_sessions_list(client)

    # ── Monitor system ────────────────────────────────────────────────────────

    async def _on_monitor_subscribe(self, client: _Client):
        self._monitor_clients.add(client)

        if not self._monitor_agent_wired:
            self._monitor_agent_wired = True

            async def _forward_event(event: str, data: dict):
                payload = {
                    "type": "monitor.event",
                    "event": event,
                    "data": data,
                    "ts": time.time(),
                }
                dead: set[_Client] = set()
                for mc in list(self._monitor_clients):
                    try:
                        await mc.send(payload)
                    except Exception:
                        dead.add(mc)
                self._monitor_clients -= dead

            self.agent.monitor_subscribe(_forward_event)

        await client.send({"type": "monitor.subscribed"})
        await self._on_monitor_snapshot(client)

    async def _on_monitor_snapshot(self, client: _Client):
        if hasattr(self.agent, "get_monitor_snapshot"):
            snapshot = self.agent.get_monitor_snapshot()
            await client.send(snapshot)

    # ── Brain / Cognitive Dashboard ───────────────────────────────────────────

    async def _on_brain_data(self, client: _Client, msg: dict):
        """Return comprehensive cognitive state for the Brain dashboard."""
        result: dict = {"type": "brain.data.result"}

        try:
            # 1. Inner life / emotional state
            inner_life_file = _data_dir / "inner_life" / "inner_state.json"
            if inner_life_file.exists():
                try:
                    result["inner_life"] = json.loads(inner_life_file.read_text())
                except Exception:
                    result["inner_life"] = None
            else:
                result["inner_life"] = None

            # 2. Autonomous state (tick, tasks, etc.)
            auto_state_file = _data_dir / "autonomous_state.json"
            if auto_state_file.exists():
                try:
                    auto = json.loads(auto_state_file.read_text())
                    result["autonomous"] = {
                        "tick": auto.get("tick", 0),
                        "last_update": auto.get("last_update"),
                        "pending_tasks": len(auto.get("task_queue", [])),
                        "completed_tasks_count": len(auto.get("completed_tasks", [])),
                        "task_queue": auto.get("task_queue", [])[:10],
                        "completed_tasks": auto.get("completed_tasks", [])[-20:],
                    }
                except Exception:
                    result["autonomous"] = None
            else:
                result["autonomous"] = None

            # 3. Goals
            goals_file = _data_dir / "goals.json"
            if goals_file.exists():
                try:
                    result["goals"] = json.loads(goals_file.read_text())
                except Exception:
                    result["goals"] = {}
            else:
                result["goals"] = {}

            # 4. ReAct execution log
            react_file = _data_dir / "react_logs" / "react_executions.jsonl"
            if react_file.exists():
                try:
                    execs = []
                    with open(react_file) as f:
                        for line in f:
                            line = line.strip()
                            if line:
                                try:
                                    execs.append(json.loads(line))
                                except json.JSONDecodeError:
                                    pass
                    result["react_executions"] = execs[-50:]
                except Exception:
                    result["react_executions"] = []
            else:
                result["react_executions"] = []

            # 5. Self-reflection / tick outcomes
            reflection_file = _data_dir / "reflection" / "tick_outcomes.jsonl"
            if reflection_file.exists():
                try:
                    outcomes = []
                    with open(reflection_file) as f:
                        for line in f:
                            line = line.strip()
                            if line:
                                try:
                                    outcomes.append(json.loads(line))
                                except json.JSONDecodeError:
                                    pass
                    result["reflections"] = outcomes[-30:]
                except Exception:
                    result["reflections"] = []
            else:
                result["reflections"] = []

            # 6. Cognitive memory stats
            cog_mem_file = _data_dir / "cognitive_memory" / "cognitive_memories.jsonl"
            if cog_mem_file.exists():
                try:
                    count = 0
                    with open(cog_mem_file) as f:
                        for _ in f:
                            count += 1
                    result["cognitive_memory_count"] = count
                except Exception:
                    result["cognitive_memory_count"] = 0
            else:
                result["cognitive_memory_count"] = 0

            # 7. Proactive reasoning proposals
            proactive_file = _data_dir / "proactive" / "proposals.jsonl"
            if proactive_file.exists():
                try:
                    proposals = []
                    with open(proactive_file) as f:
                        for line in f:
                            line = line.strip()
                            if line:
                                try:
                                    proposals.append(json.loads(line))
                                except json.JSONDecodeError:
                                    pass
                    result["proactive_proposals"] = proposals[-20:]
                except Exception:
                    result["proactive_proposals"] = []
            else:
                result["proactive_proposals"] = []

            # 8. Live module stats from agent
            if hasattr(self.agent, "inner_life") and self.agent.inner_life:
                result["inner_life_stats"] = self.agent.inner_life.get_stats()
            if hasattr(self.agent, "autonomous") and self.agent.autonomous:
                result["autonomous_live"] = {
                    "tick": getattr(self.agent.autonomous, "tick", 0),
                    "queue_size": len(getattr(self.agent.autonomous, "task_queue", [])),
                    "completed_count": len(getattr(self.agent.autonomous, "completed_tasks", [])),
                    "consecutive_errors": getattr(self.agent.autonomous, "consecutive_errors", 0),
                }

            # 9. Trace file stats
            traces_dir = _data_dir / "traces"
            if traces_dir.exists():
                trace_files = sorted(traces_dir.glob("trace-*.jsonl"))
                result["trace_files"] = [
                    {"name": f.name, "size_kb": round(f.stat().st_size / 1024, 1)}
                    for f in trace_files[-7:]
                ]
            else:
                result["trace_files"] = []

            # 10. Identity (personality traits, voice, core_directive)
            identity_file = _data_dir / "x_consciousness" / "identity.json"
            if identity_file.exists():
                try:
                    result["identity"] = json.loads(identity_file.read_text())
                except Exception:
                    result["identity"] = None
            else:
                result["identity"] = None

            # 11. Evolution log (personality changes over time)
            evo_file = _data_dir / "x_consciousness" / "evolution_log.json"
            if evo_file.exists():
                try:
                    evo = json.loads(evo_file.read_text())
                    result["evolution_log"] = evo[-10:] if isinstance(evo, list) else []
                except Exception:
                    result["evolution_log"] = []
            else:
                result["evolution_log"] = []

            # 12. Mood history timeline
            mood_file = _data_dir / "x_consciousness" / "mood_history.jsonl"
            if mood_file.exists():
                try:
                    moods = []
                    with open(mood_file) as f:
                        for line in f:
                            line = line.strip()
                            if line:
                                try:
                                    moods.append(json.loads(line))
                                except json.JSONDecodeError:
                                    pass
                    result["mood_history"] = moods[-40:]
                except Exception:
                    result["mood_history"] = []
            else:
                result["mood_history"] = []

            # 13. Inner monologue
            mono_file = _data_dir / "x_consciousness" / "inner_monologue.jsonl"
            if mono_file.exists():
                try:
                    monos = []
                    with open(mono_file) as f:
                        for line in f:
                            line = line.strip()
                            if line:
                                try:
                                    monos.append(json.loads(line))
                                except json.JSONDecodeError:
                                    pass
                    result["inner_monologue"] = monos[-15:]
                except Exception:
                    result["inner_monologue"] = []
            else:
                result["inner_monologue"] = []

            # 14. Self-heal log
            heal_file = _data_dir / "x_consciousness" / "heal_log.jsonl"
            if heal_file.exists():
                try:
                    heals = []
                    with open(heal_file) as f:
                        for line in f:
                            line = line.strip()
                            if line:
                                try:
                                    heals.append(json.loads(line))
                                except json.JSONDecodeError:
                                    pass
                    result["heal_log"] = heals[-20:]
                except Exception:
                    result["heal_log"] = []
            else:
                result["heal_log"] = []

            # 15. X agent state (posts/engagements today)
            x_state_file = _data_dir / "x_agent_state.json"
            if x_state_file.exists():
                try:
                    xs = json.loads(x_state_file.read_text())
                    result["x_agent_state"] = {
                        "posts_today": xs.get("posts_today", 0),
                        "engagements_today": xs.get("engagements_today", 0),
                        "last_reset": xs.get("last_reset"),
                        "posted_count": len(xs.get("posted_urls", [])),
                        "engaged_count": len(xs.get("engaged_ids", [])),
                    }
                except Exception:
                    result["x_agent_state"] = None
            else:
                result["x_agent_state"] = None

            # 16. Proactive state (summary)
            proactive_state_file = _data_dir / "proactive" / "proactive_state.json"
            if proactive_state_file.exists():
                try:
                    result["proactive_state"] = json.loads(proactive_state_file.read_text())
                except Exception:
                    result["proactive_state"] = None
            else:
                result["proactive_state"] = None

            # 17. X consciousness reflections
            xc_ref_file = _data_dir / "x_consciousness" / "reflections.json"
            if xc_ref_file.exists():
                try:
                    xc_refs = json.loads(xc_ref_file.read_text())
                    if isinstance(xc_refs, list):
                        result["x_reflections"] = xc_refs[-5:]
                    else:
                        result["x_reflections"] = [xc_refs]
                except Exception:
                    result["x_reflections"] = []
            else:
                result["x_reflections"] = []

            # 18. Journal (consciousness diary)
            journal_file = _data_dir / "x_consciousness" / "journal.jsonl"
            if journal_file.exists():
                try:
                    entries = []
                    with open(journal_file) as f:
                        for line in f:
                            line = line.strip()
                            if line:
                                try:
                                    entries.append(json.loads(line))
                                except Exception:
                                    pass
                    result["journal"] = entries[-20:]
                except Exception:
                    result["journal"] = []
            else:
                result["journal"] = []

            # 19. Calendar
            calendar_file = _data_dir / "calendar.json"
            if not calendar_file.exists():
                calendar_file = Path("data") / "calendar.json"
            if calendar_file.exists():
                try:
                    result["calendar"] = json.loads(calendar_file.read_text())
                except Exception:
                    result["calendar"] = []
            else:
                result["calendar"] = []

            # 20. Conversations (summaries per user)
            convos_dir = _data_dir / "conversations"
            if not convos_dir.exists():
                convos_dir = Path("data") / "conversations"
            result["conversations"] = []
            if convos_dir.exists() and convos_dir.is_dir():
                try:
                    for cf in sorted(convos_dir.glob("*.jsonl")):
                        user_id = cf.stem
                        lines = []
                        with open(cf) as f:
                            for line in f:
                                line = line.strip()
                                if line:
                                    try:
                                        lines.append(json.loads(line))
                                    except Exception:
                                        pass
                        result["conversations"].append({
                            "user_id": user_id,
                            "total_messages": len(lines),
                            "last_messages": lines[-5:],
                        })
                except Exception:
                    pass

            # 21. Benchmarks
            bench_dir = _data_dir / "benchmarks"
            if not bench_dir.exists():
                bench_dir = Path("data") / "benchmarks"
            result["benchmarks"] = []
            if bench_dir.exists() and bench_dir.is_dir():
                try:
                    for bf in sorted(bench_dir.glob("*.json")):
                        try:
                            bd = json.loads(bf.read_text())
                            result["benchmarks"].append({
                                "suite": bd.get("suite", bf.stem),
                                "total": bd.get("total", 0),
                                "passed": bd.get("passed", 0),
                                "failed": bd.get("failed", 0),
                                "pass_rate": bd.get("pass_rate", 0),
                                "avg_score": bd.get("avg_score", 0),
                                "avg_duration_ms": bd.get("avg_duration_ms", 0),
                                "model": bd.get("model", ""),
                                "agent_version": bd.get("agent_version", ""),
                                "started_at": bd.get("started_at", ""),
                            })
                        except Exception:
                            pass
                except Exception:
                    pass

            # ── 22. News cache (news_cache.json) ──────────────────────────────
            news_file = _data_dir / "news_cache.json"
            if not news_file.exists():
                news_file = Path("data") / "news_cache.json"
            try:
                if news_file.exists():
                    nc = json.loads(news_file.read_text("utf-8"))
                    # Flatten: cache is {key: {data: [...], _ts: float}}
                    news_items = []
                    for k, v in nc.items():
                        d = v.get("data") if isinstance(v, dict) else v
                        if isinstance(d, list):
                            news_items.extend(d[:10])
                        elif isinstance(d, dict) and not d.get("error"):
                            news_items.append({"key": k, **d})
                    result["news_cache"] = news_items[:30]
            except Exception:
                pass

            # ── 23. Connectome (FlyWire neural colony) ────────────────────
            if hasattr(self.agent, "connectome") and self.agent.connectome:
                try:
                    result["connectome"] = self.agent.connectome.get_wiring_diagram()
                except Exception:
                    result["connectome"] = None
            else:
                result["connectome"] = None

            # ── 24. Deep planner ──────────────────────────────────────────
            deep_planner = getattr(self.agent, "deep_planner", None)
            if not deep_planner and hasattr(self.agent, "autonomous") and self.agent.autonomous:
                deep_planner = getattr(self.agent.autonomous, "deep_planner", None)
            if deep_planner:
                try:
                    result["deep_planner"] = deep_planner.get_stats()
                except Exception:
                    result["deep_planner"] = None
            else:
                result["deep_planner"] = None

            # ── 25. Inter-agent bridge ────────────────────────────────────
            bridge = getattr(self.agent, "inter_agent_bridge", None)
            if not bridge and hasattr(self.agent, "autonomous") and self.agent.autonomous:
                bridge = getattr(self.agent.autonomous, "inter_agent_bridge", None)
            if bridge:
                try:
                    result["inter_agent_bridge"] = bridge.get_stats()
                except Exception:
                    result["inter_agent_bridge"] = None
            else:
                result["inter_agent_bridge"] = None

            # ── 26. Ultra-LTM ─────────────────────────────────────────────
            ltm = getattr(self.agent, "ultra_ltm", None)
            if not ltm and hasattr(self.agent, "autonomous") and self.agent.autonomous:
                ltm = getattr(self.agent.autonomous, "ultra_ltm", None)
            if ltm:
                try:
                    result["ultra_ltm"] = ltm.get_stats()
                except Exception:
                    result["ultra_ltm"] = None
            else:
                result["ultra_ltm"] = None

            # ── 27. Self-benchmark ────────────────────────────────────────
            bench = getattr(self.agent, "self_benchmark", None)
            if not bench and hasattr(self.agent, "autonomous") and self.agent.autonomous:
                bench = getattr(self.agent.autonomous, "self_benchmark", None)
            if bench:
                try:
                    result["self_benchmark"] = bench.get_stats()
                except Exception:
                    result["self_benchmark"] = None
            else:
                result["self_benchmark"] = None

            # ── 28-37. New cognitive modules ──────────────────────────────
            for _mod_attr in (
                "meta_learner", "causal_engine", "goal_synthesis",
                "skill_composer", "world_predictor", "cognitive_optimizer",
                "adversarial_tester", "resource_governor", "theory_of_mind",
                "ethical_reasoner",
            ):
                mod = getattr(self.agent, _mod_attr, None)
                if not mod and hasattr(self.agent, "autonomous") and self.agent.autonomous:
                    mod = getattr(self.agent.autonomous, _mod_attr, None)
                if mod:
                    try:
                        result[_mod_attr] = mod.get_stats()
                    except Exception:
                        result[_mod_attr] = None
                else:
                    result[_mod_attr] = None

            # ── 38-57. v1.5 World-First Modules ──────────────────────────
            for _mod_attr in (
                "dream_engine", "cognitive_immunity", "temporal_consciousness",
                "cognitive_fusion", "memory_palace", "narrative_identity",
                "curiosity_drive", "collective_unconscious", "cognitive_metabolism",
                "synthetic_intuition", "phantom_limb", "cognitive_scar",
                "time_crystal", "holographic_context", "swarm_cortex",
                "cognitive_archaeology", "emotional_contagion", "predictive_empathy",
                "autonomous_researcher", "empathy_synthesizer",
                "cognitive_teleportation", "ontological_engine",
                "cognitive_gravity", "temporal_paradox",
                "synaesthetic_processor", "cognitive_mitosis",
                "entropic_sentinel", "quantum_cognition",
                "cognitive_placebo", "noospheric_interface",
                "akashic_records", "deja_vu",
                "morphogenetic_field", "liminal_processor",
                "prescient_executor", "cognitive_dark_matter",
                "ego_membrane", "hyperstition_engine",
                "cognitive_chrysalis", "existential_compass",
            ):
                mod = getattr(self.agent, _mod_attr, None)
                if not mod and hasattr(self.agent, "autonomous") and self.agent.autonomous:
                    mod = getattr(self.agent.autonomous, _mod_attr, None)
                if mod:
                    try:
                        result[_mod_attr] = mod.get_stats()
                    except Exception:
                        result[_mod_attr] = None
                else:
                    result[_mod_attr] = None

            # ── v1.7 God Supreme Modules ──────────────────────────
            for _mod_attr in (
                "web_agent", "self_healer", "dynamic_skill_factory",
                "multimodal_engine", "internet_monitor", "financial_autonomy",
                "social_presence", "self_replicator", "continuous_learner",
            ):
                mod = getattr(self.agent, _mod_attr, None)
                if not mod and hasattr(self.agent, "autonomous") and self.agent.autonomous:
                    mod = getattr(self.agent.autonomous, _mod_attr, None)
                if mod:
                    try:
                        result[_mod_attr] = mod.get_stats()
                    except Exception:
                        result[_mod_attr] = None
                else:
                    result[_mod_attr] = None

            # ── v1.8 Final Gap Closers ────────────────────────────
            for _mod_attr in (
                "nl_automation", "video_understanding", "knowledge_graph",
                "iot_controller", "distributed_task_queue",
            ):
                mod = getattr(self.agent, _mod_attr, None)
                if not mod and hasattr(self.agent, "autonomous") and self.agent.autonomous:
                    mod = getattr(self.agent.autonomous, _mod_attr, None)
                if mod:
                    try:
                        result[_mod_attr] = mod.get_stats()
                    except Exception:
                        result[_mod_attr] = None
                else:
                    result[_mod_attr] = None

        except Exception as e:
            logger.warning(f"[Gateway] brain.data error: {e}")
            result["error"] = str(e)

        await client.send(result)

    # ── Thoughts / Consciousness stream ───────────────────────────────────────

    async def _on_thoughts_list(self, client: _Client, msg: dict):
        limit = min(int(msg.get("limit", 200)), 1000)
        filter_type = msg.get("filter")

        base = _data_dir / "x_consciousness"
        result: dict = {"type": "thoughts.list.result"}

        # Journal
        journal_entries: list = []
        journal_file = base / "journal.jsonl"
        if journal_file.exists():
            try:
                with open(journal_file) as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            entry = json.loads(line)
                            if filter_type and entry.get("type") != filter_type:
                                continue
                            journal_entries.append(entry)
                        except json.JSONDecodeError:
                            pass
            except Exception:
                pass
        result["journal"] = journal_entries[-limit:]

        # Inner monologue
        thoughts: list = []
        thoughts_file = base / "inner_monologue.jsonl"
        if thoughts_file.exists():
            try:
                with open(thoughts_file) as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            try:
                                thoughts.append(json.loads(line))
                            except json.JSONDecodeError:
                                pass
            except Exception:
                pass
        result["thoughts"] = thoughts[-limit:]

        # Reflections
        reflections_file = base / "reflections.json"
        if reflections_file.exists():
            try:
                with open(reflections_file) as f:
                    result["reflections"] = json.loads(f.read())[-50:]
            except Exception:
                result["reflections"] = []
        else:
            result["reflections"] = []

        # Current emotional state
        xauto = getattr(self.agent, "x_autoposter", None)
        if xauto and hasattr(xauto, "mind"):
            mind = xauto.mind
            result["mood"] = {
                "current": getattr(mind, "_mood", "unknown"),
                "intensity": getattr(mind, "_mood_intensity", 0),
                "history": getattr(mind, "_mood_history", [])[-20:],
            }
            result["memory_stats"] = (
                mind.get_memory_stats() if hasattr(mind, "get_memory_stats") else {}
            )
        else:
            last_felt = next(
                (e for e in reversed(journal_entries) if e.get("type") == "felt"), None
            )
            if last_felt and isinstance(last_felt.get("data"), dict):
                d = last_felt["data"]
                result["mood"] = {
                    "current": d.get("emotion", "unknown"),
                    "intensity": d.get("intensity", 0),
                    "history": [],
                }
            else:
                result["mood"] = {"current": "unknown", "intensity": 0, "history": []}
            result["memory_stats"] = {}

        await client.send(result)

    # ── Node system ───────────────────────────────────────────────────────────

    async def _on_node_register(self, client: _Client, msg: dict):
        node_id = msg.get("node_id", client.cid)
        caps = msg.get("capabilities", [])

        self._nodes[node_id] = client
        client.node_id = node_id

        logger.info(f"[Gateway] Node registered: {node_id}  caps={caps}")
        await client.send({"type": "node.registered", "node_id": node_id, "capabilities": caps})

    async def _on_node_invoke(self, client: _Client, msg: dict):
        node_id = msg.get("node_id", "")
        cap = msg.get("capability", "")
        args = msg.get("args", {})
        req_id = msg.get("request_id", str(time.time()))

        node = self._nodes.get(node_id)
        if not node:
            await client.send(
                {"type": "error", "text": f"Node '{node_id}' not connected", "request_id": req_id}
            )
            return

        await node.send({
            "type": "node.invoke",
            "capability": cap,
            "args": args,
            "request_id": req_id,
            "reply_to": client.cid,
        })

    async def _on_node_result(self, client: _Client, msg: dict):
        reply_to = msg.get("reply_to", "")
        target = next((c for c in self._clients if c.cid == reply_to), None)
        if target:
            await target.send({**msg, "type": "node.result"})

    # ── Rate limiting ─────────────────────────────────────────────────────────

    def _check_rate(self, cid: str) -> bool:
        now = time.time()
        stamps = self._rate_limits.setdefault(cid, [])
        cutoff = now - self._rate_window
        self._rate_limits[cid] = [t for t in stamps if t > cutoff]
        if len(self._rate_limits[cid]) >= self._rate_max:
            return False
        self._rate_limits[cid].append(now)
        return True

    # ── Heartbeat ─────────────────────────────────────────────────────────────

    async def _heartbeat(self):
        while self._running:
            try:
                await asyncio.sleep(HEARTBEAT_INT)
                dead: set[_Client] = set()
                ts = time.time()
                for c in list(self._clients):
                    try:
                        await c.send({"type": "heartbeat", "ts": ts})
                    except Exception:
                        dead.add(c)
                self._clients -= dead
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.debug(f"[Gateway] Heartbeat error: {exc}")

    # ── Broadcast helper ──────────────────────────────────────────────────────

    async def broadcast(self, payload: dict, *, exclude: Optional[_Client] = None):
        dead: set[_Client] = set()
        for c in list(self._clients):
            if c is exclude:
                continue
            try:
                await c.send(payload)
            except Exception:
                dead.add(c)
        self._clients -= dead


# ── Compat alias used by cli.py ───────────────────────────────────────────────
GatewayServer = Gateway
