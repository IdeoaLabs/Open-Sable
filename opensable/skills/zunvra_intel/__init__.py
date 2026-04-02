"""
OpenSable Zunvra Intelligence Suite — v3.1.0
=============================================

A standalone skill package that brings 26 autonomous intelligence
capabilities plus full dashboard remote control to the
Zunvra/Central Intelligence real-time OSINT dashboard.

Features
--------
 1. Natural Language OSINT Analyst — chat with live intelligence data
 2. Cross-Domain Causal Correlation — detect non-obvious causal chains
 3. Swarm Intelligence Threat Assessment — multi-agent debate on threats
 4. Autonomous OSINT Hunter — proactive external source scanning
 5. Predictive World Model — trend analysis and forecasting
 6. Temporal Pattern Memory (Déjà Vu) — historical pattern matching
 7. Dream Engine Retrospective — creative offline analysis
 8. Entity Deep Dossier — autonomous entity research and risk scoring
 9. Cognitive Map Annotations — floating intelligence notes on the map
10. Multi-Agent Intelligence Fusion — SIGINT/GEOINT/HUMINT convergence
11. Knowledge Graph Engine — entity/relationship graph with community detection
12. Pattern of Life Analyzer — behavioral baselines and anomaly detection
13. Geofence Tripwire System — critical zone monitoring with enter/exit/loiter
14. Kill Chain Tracker — geopolitical kill chain phase tracking
15. Timeline Reconstructor — event timeline with gap detection
16. Wargaming Simulator — multi-order consequence modeling
17. Counter-Surveillance Detector — evasion behavior detection
18. Narrative Warfare Monitor — information operations detection
19. SIGINT/EW Pattern Analyzer — GPS jamming & electronic warfare
20. Financial Intelligence (FININT) Tracker — sanctions evasion & dark fleet
21. Nuclear Proliferation Monitor — facility & seismic-test detection
22. Space Domain Awareness — satellite tracking & ASAT detection
23. Infrastructure Resilience Monitor — cyber/outage cascading failures
24. Environmental Threat Intel — scorched earth & climate-conflict nexus
25. Force Projection Tracker — carrier groups & naval disposition
26. Threat Fusion Dashboard — unified DEFCON-style threat picture
 ⭐ Remote Control — move camera, select entities, change styles from agent

Quick Start
-----------
>>> from opensable.skills.zunvra_intel import ZunvraIntelSkill
>>> skill = ZunvraIntelSkill(base_url="http://localhost:8000")
>>> await skill.initialize()
>>> response = await skill.chat("What's happening in the Black Sea?")
>>> await skill.run_cycle()  # Full 26-module analysis cycle
>>> picture = skill.get_threat_picture()  # DEFCON-style unified view
>>> await skill.remote.fly_to(33.89, 35.50)  # Move map camera
>>> await skill.remote.set_style("NVG")  # Night vision
>>> await skill.shutdown()
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

# Core connector
from .connector import ZunvraConnector, IntelSnapshot

# Feature modules
from .nl_analyst import NLIntelAnalyst, AnalystResponse
from .causal_correlation import CausalCorrelationEngine, CausalLink, ThreatChain
from .swarm_threat import SwarmThreatAssessor, SwarmAssessment, detect_anomalies
from .osint_hunter import AutonomousOSINTHunter, FlashReport
from .predictive_model import PredictiveWorldModel, Forecast
from .temporal_memory import TemporalPatternMemory, DejaVuMatch
from .dream_engine import DreamEngine, DreamReport
from .entity_dossier import EntityDossierGenerator, EntityDossier
from .cognitive_annotations import CognitiveAnnotator, Annotation
from .multi_agent_fusion import MultiAgentFusion, FusionReport

# Features #11-18 (Palantir/Agency parity)
from .knowledge_graph import KnowledgeGraphEngine, GraphNode, GraphEdge
from .pattern_of_life import PatternOfLifeAnalyzer, BehaviorBaseline, PoLAnomaly
from .geofence_tripwire import GeofenceTripwire, Geofence, TripwireAlert
from .kill_chain import KillChainTracker, KillChainTrack, KillChainAlert
from .timeline_reconstructor import TimelineReconstructor, TimelineEvent, Incident
from .wargaming_simulator import WargamingSimulator, WargameScenario, Consequence
from .counter_surveillance import CounterSurveillanceDetector, EvasionAlert
from .narrative_warfare import NarrativeWarfareMonitor, NarrativeAlert, NarrativeCluster

# Features #19-26 (Agency-level capabilities)
from .sigint_ew import SigintEWAnalyzer, SigintAlert, EWCampaign
from .finint_tracker import FinintTracker, FinintAlert, DarkFleetVessel
from .nuclear_monitor import NuclearMonitor, NuclearAlert, FacilityStatus
from .space_domain import SpaceDomainAwareness, SpaceAlert, SpaceObject
from .infrastructure_resilience import InfrastructureResilience, InfraAlert, SectorStatus
from .environmental_threat import EnvironmentalThreatIntel, EnvAlert, FireCluster
from .force_projection import ForceProjectionTracker, ForceAlert, CarrierGroup
from .threat_fusion import ThreatFusionDashboard, ThreatPicture, FusionAlert

# Remote control — agent → dashboard UI command channel
from .remote_control import RemoteControl

# Intelligence broadcaster — OSINT → social media autonomous poster
from .intel_broadcaster import IntelBroadcaster, IntelFinding

# Autonomous camera pilot — Sable drives the dashboard map
from .camera_pilot import AutonomousCameraPilot, CameraMove

# Web research skills — active YouTube & news search for intel enrichment
try:
    from opensable.skills.social.youtube_skill import YouTubeSkill
    _YOUTUBE_AVAILABLE = True
except ImportError:
    _YOUTUBE_AVAILABLE = False

try:
    from opensable.skills.automation.news_reader_skill import NewsReaderSkill
    _NEWS_READER_AVAILABLE = True
except ImportError:
    _NEWS_READER_AVAILABLE = False

logger = logging.getLogger(__name__)

__version__ = "3.3.0"
__all__ = [
    "ZunvraIntelSkill",
    # Connector
    "ZunvraConnector",
    "IntelSnapshot",
    # Feature #1
    "NLIntelAnalyst",
    "AnalystResponse",
    # Feature #2
    "CausalCorrelationEngine",
    "CausalLink",
    "ThreatChain",
    # Feature #3
    "SwarmThreatAssessor",
    "SwarmAssessment",
    # Feature #4
    "AutonomousOSINTHunter",
    "FlashReport",
    # Feature #5
    "PredictiveWorldModel",
    "Forecast",
    # Feature #6
    "TemporalPatternMemory",
    "DejaVuMatch",
    # Feature #7
    "DreamEngine",
    "DreamReport",
    # Feature #8
    "EntityDossierGenerator",
    "EntityDossier",
    # Feature #9
    "CognitiveAnnotator",
    "Annotation",
    # Feature #10
    "MultiAgentFusion",
    "FusionReport",
    # Feature #11
    "KnowledgeGraphEngine",
    "GraphNode",
    "GraphEdge",
    # Feature #12
    "PatternOfLifeAnalyzer",
    "BehaviorBaseline",
    "PoLAnomaly",
    # Feature #13
    "GeofenceTripwire",
    "Geofence",
    "TripwireAlert",
    # Feature #14
    "KillChainTracker",
    "KillChainTrack",
    "KillChainAlert",
    # Feature #15
    "TimelineReconstructor",
    "TimelineEvent",
    "Incident",
    # Feature #16
    "WargamingSimulator",
    "WargameScenario",
    "Consequence",
    # Feature #17
    "CounterSurveillanceDetector",
    "EvasionAlert",
    # Feature #18
    "NarrativeWarfareMonitor",
    "NarrativeAlert",
    "NarrativeCluster",
    # Feature #19
    "SigintEWAnalyzer",
    "SigintAlert",
    "EWCampaign",
    # Feature #20
    "FinintTracker",
    "FinintAlert",
    "DarkFleetVessel",
    # Feature #21
    "NuclearMonitor",
    "NuclearAlert",
    "FacilityStatus",
    # Feature #22
    "SpaceDomainAwareness",
    "SpaceAlert",
    "SpaceObject",
    # Feature #23
    "InfrastructureResilience",
    "InfraAlert",
    "SectorStatus",
    # Feature #24
    "EnvironmentalThreatIntel",
    "EnvAlert",
    "FireCluster",
    # Feature #25
    "ForceProjectionTracker",
    "ForceAlert",
    "CarrierGroup",
    # Feature #26
    "ThreatFusionDashboard",
    "ThreatPicture",
    "FusionAlert",
    # Remote Control (dashboard UI command channel)
    "RemoteControl",
    # Intelligence Broadcaster (OSINT → social media)
    "IntelBroadcaster",
    "IntelFinding",
    # Autonomous Camera Pilot (dashboard map control)
    "AutonomousCameraPilot",
    "CameraMove",
]


class ZunvraIntelSkill:
    """
    Master orchestrator for all 26 Zunvra intelligence capabilities.

    This is the main entry point for the OpenSable skill system.
    It manages the lifecycle of all sub-modules and provides a unified API.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        data_dir: str = "data/zunvra_intel",
        poll_interval: int = 30,
        enable_llm: bool = True,
    ):
        self.base_url = base_url
        self.data_dir = Path(data_dir)
        self.poll_interval = poll_interval
        self.enable_llm = enable_llm

        # Core connector
        self.connector = ZunvraConnector(base_url=base_url)

        # Feature modules (lazy-initialized in initialize())
        self.nl_analyst: Optional[NLIntelAnalyst] = None
        self.causal_engine: Optional[CausalCorrelationEngine] = None
        self.swarm_assessor: Optional[SwarmThreatAssessor] = None
        self.osint_hunter: Optional[AutonomousOSINTHunter] = None
        self.world_model: Optional[PredictiveWorldModel] = None
        self.temporal_memory: Optional[TemporalPatternMemory] = None
        self.dream_engine: Optional[DreamEngine] = None
        self.dossier_gen: Optional[EntityDossierGenerator] = None
        self.annotator: Optional[CognitiveAnnotator] = None
        self.fusion: Optional[MultiAgentFusion] = None

        # Features #11-18
        self.knowledge_graph: Optional[KnowledgeGraphEngine] = None
        self.pattern_of_life: Optional[PatternOfLifeAnalyzer] = None
        self.geofence: Optional[GeofenceTripwire] = None
        self.kill_chain: Optional[KillChainTracker] = None
        self.timeline: Optional[TimelineReconstructor] = None
        self.wargaming: Optional[WargamingSimulator] = None
        self.counter_surveillance: Optional[CounterSurveillanceDetector] = None
        self.narrative_warfare: Optional[NarrativeWarfareMonitor] = None

        # Features #19-26
        self.sigint_ew: Optional[SigintEWAnalyzer] = None
        self.finint: Optional[FinintTracker] = None
        self.nuclear: Optional[NuclearMonitor] = None
        self.space_domain: Optional[SpaceDomainAwareness] = None
        self.infrastructure: Optional[InfrastructureResilience] = None
        self.environmental: Optional[EnvironmentalThreatIntel] = None
        self.force_projection: Optional[ForceProjectionTracker] = None
        self.threat_fusion: Optional[ThreatFusionDashboard] = None

        # Remote control (dashboard UI command channel)
        self.remote: Optional[RemoteControl] = None

        # Intelligence broadcaster (OSINT → social media)
        self.broadcaster: Optional[IntelBroadcaster] = None

        # Autonomous camera pilot (agent drives the map)
        self.camera_pilot: Optional[AutonomousCameraPilot] = None

        # Web research skills (YouTube + news — injected into camera pilot)
        self._youtube_skill = None
        self._news_reader_skill = None

        # State
        self._initialized = False
        self._cycle_count = 0
        self._llm = None

    # ── lifecycle ─────────────────────────────────────────────────────

    async def initialize(self, llm=None):
        """Initialize all sub-modules. Call once before use."""
        if self._initialized:
            return

        self._llm = llm
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Initialize all feature modules
        self.nl_analyst = NLIntelAnalyst(
            connector=self.connector,
            data_dir=self.data_dir,
        )
        self.causal_engine = CausalCorrelationEngine(
            data_dir=self.data_dir,
        )
        self.swarm_assessor = SwarmThreatAssessor(
            data_dir=self.data_dir,
        )
        self.osint_hunter = AutonomousOSINTHunter(
            connector=self.connector,
            data_dir=self.data_dir,
        )
        self.world_model = PredictiveWorldModel(
            data_dir=self.data_dir,
        )
        self.temporal_memory = TemporalPatternMemory(
            data_dir=self.data_dir,
        )
        self.dream_engine = DreamEngine(
            data_dir=self.data_dir,
        )
        self.dossier_gen = EntityDossierGenerator(
            connector=self.connector,
            data_dir=self.data_dir,
        )
        self.annotator = CognitiveAnnotator(
            data_dir=self.data_dir,
        )
        self.fusion = MultiAgentFusion(
            data_dir=self.data_dir,
        )

        # #11-18
        self.knowledge_graph = KnowledgeGraphEngine(data_dir=self.data_dir)
        self.pattern_of_life = PatternOfLifeAnalyzer(data_dir=self.data_dir)
        self.geofence = GeofenceTripwire(data_dir=self.data_dir)
        self.kill_chain = KillChainTracker(data_dir=self.data_dir)
        self.timeline = TimelineReconstructor(data_dir=self.data_dir)
        self.wargaming = WargamingSimulator(data_dir=self.data_dir)
        self.counter_surveillance = CounterSurveillanceDetector(data_dir=self.data_dir)
        self.narrative_warfare = NarrativeWarfareMonitor(data_dir=self.data_dir)

        # #19-26
        self.sigint_ew = SigintEWAnalyzer(data_dir=self.data_dir)
        self.finint = FinintTracker(data_dir=self.data_dir)
        self.nuclear = NuclearMonitor(data_dir=self.data_dir)
        self.space_domain = SpaceDomainAwareness(data_dir=self.data_dir)
        self.infrastructure = InfrastructureResilience(data_dir=self.data_dir)
        self.environmental = EnvironmentalThreatIntel(data_dir=self.data_dir)
        self.force_projection = ForceProjectionTracker(data_dir=self.data_dir)
        self.threat_fusion = ThreatFusionDashboard(data_dir=self.data_dir)

        # Remote control — agent ↔ dashboard UI
        self.remote = RemoteControl(base_url=self.base_url)

        # Intelligence broadcaster — OSINT → social media
        self.broadcaster = IntelBroadcaster(
            connector=self.connector,
            remote_control=self.remote,
        )

        # ── Web research skills — active search for intel enrichment ──
        # These are injected into the camera pilot so it can actively
        # search YouTube for related videos and news for source articles.
        video_searcher_fn = None
        news_searcher_fn = None

        # YouTube search callback
        if _YOUTUBE_AVAILABLE:
            try:
                self._youtube_skill = YouTubeSkill(type("_Cfg", (), {
                    "youtube_api_key": None,       # Falls back to YOUTUBE_API_KEY env
                    "youtube_access_token": None,
                    "youtube_action_delay": 1.0,
                })())
                yt_ok = await self._youtube_skill.initialize()
                if yt_ok:
                    async def _yt_search(query: str, count: int = 3):
                        r = await self._youtube_skill.search_videos(query, count)
                        return r.get("videos", []) if r.get("success") else []
                    video_searcher_fn = _yt_search
                    logger.info("YouTube search wired to camera pilot")
                else:
                    logger.info("YouTube skill not available (no API key?), passive enrichment only")
            except Exception as e:
                logger.debug("YouTube skill init failed: %s", e)

        # News search callback
        if _NEWS_READER_AVAILABLE:
            try:
                self._news_reader_skill = NewsReaderSkill(type("_Cfg", (), {
                    "news_enabled": True,
                    "news_cache_ttl": 1800,
                })())
                await self._news_reader_skill.initialize()
                async def _news_search(query: str, max_items: int = 3):
                    return await self._news_reader_skill.search_news(query, max_items)
                news_searcher_fn = _news_search
                logger.info("News search wired to camera pilot")
            except Exception as e:
                logger.debug("News reader skill init failed: %s", e)

        # Autonomous camera pilot — Sable drives the dashboard map
        self.camera_pilot = AutonomousCameraPilot(
            remote=self.remote,
            move_delay=2.5,
            dwell_default=4.0,
            max_moves_per_cycle=15,
            video_searcher=video_searcher_fn,
            news_searcher=news_searcher_fn,
        )

        self._initialized = True
        logger.info("ZunvraIntelSkill v%s initialized with 26 capabilities + remote control + camera pilot + broadcaster", __version__)

    async def shutdown(self):
        """Clean shutdown of all modules."""
        if self.broadcaster:
            await self.broadcaster.shutdown()
        if self.remote:
            await self.remote.close()
        await self.connector.disconnect()
        self._initialized = False
        logger.info("ZunvraIntelSkill shut down")

    # ── unified API ───────────────────────────────────────────────────

    async def chat(self, query: str) -> AnalystResponse:
        """Feature #1: Ask a question about live intelligence data."""
        self._check_init()
        assert self.nl_analyst is not None
        snapshot = await self.connector.fetch_full()
        return await self.nl_analyst.analyze(query, llm=self._llm if self.enable_llm else None, snapshot=snapshot)

    async def run_cycle(self) -> Dict[str, Any]:
        """
        Run a full analysis cycle across all modules.

        Returns a summary dict of what each module found.
        """
        self._check_init()
        results: Dict[str, Any] = {"cycle": self._cycle_count}

        # 1. Fetch fresh data
        snapshot = await self.connector.fetch_full()
        if not snapshot:
            return {"error": "Failed to fetch Zunvra data"}

        # 2. Detect anomalies (feeds into swarm + annotations)
        anomalies = detect_anomalies(snapshot)

        # 3. Run all analyzers in parallel where possible
        llm = self._llm if self.enable_llm else None

        assert self.causal_engine is not None
        assert self.swarm_assessor is not None
        assert self.world_model is not None
        assert self.temporal_memory is not None
        assert self.annotator is not None
        assert self.fusion is not None

        # Causal correlation
        try:
            chains = await self.causal_engine.analyze(snapshot, llm=llm)
            results["causal_chains"] = len(chains)
        except Exception as e:
            logger.warning("Causal analysis failed: %s", e)
            results["causal_chains"] = 0

        # Swarm threat assessment
        try:
            assessments = await self.swarm_assessor.assess_all(snapshot, llm=llm)
            results["threat_assessments"] = len(assessments)
            if assessments:
                results["max_threat"] = max(a.threat_level for a in assessments)
        except Exception as e:
            logger.warning("Swarm assessment failed: %s", e)

        # World model update
        try:
            self.world_model.observe(snapshot)
            forecasts = await self.world_model.forecast(llm=llm)
            results["forecasts"] = len(forecasts)
        except Exception as e:
            logger.warning("World model update failed: %s", e)

        # Temporal memory
        try:
            matches = self.temporal_memory.observe(snapshot)
            results["deja_vu_matches"] = len(matches)
        except Exception as e:
            logger.warning("Temporal memory failed: %s", e)

        # Cognitive annotations
        try:
            annotations = await self.annotator.observe(
                snapshot,
                anomalies=[{"description": a.description, "severity": a.severity}
                           for a in anomalies] if anomalies else None,
                llm=llm,
            )
            results["new_annotations"] = len(annotations)
        except Exception as e:
            logger.warning("Annotation generation failed: %s", e)

        # Multi-agent fusion
        try:
            fusion_report = await self.fusion.analyze(snapshot, llm=llm)
            results["fusion_report"] = fusion_report.report_id if fusion_report else None
        except Exception as e:
            logger.warning("Multi-agent fusion failed: %s", e)

        # --- Features #11-18 ---

        assert self.knowledge_graph is not None
        assert self.pattern_of_life is not None
        assert self.geofence is not None
        assert self.kill_chain is not None
        assert self.timeline is not None
        assert self.wargaming is not None
        assert self.counter_surveillance is not None
        assert self.narrative_warfare is not None

        # Knowledge graph
        try:
            await self.knowledge_graph.ingest(snapshot)
            results["graph_nodes"] = len(self.knowledge_graph.nodes)
        except Exception as e:
            logger.warning("Knowledge graph ingest failed: %s", e)

        # Pattern of Life
        try:
            pol_anomalies = self.pattern_of_life.observe(snapshot)
            results["pol_anomalies"] = len(pol_anomalies)
        except Exception as e:
            logger.warning("Pattern of Life failed: %s", e)

        # Geofence tripwires
        try:
            tripwire_alerts = self.geofence.evaluate(snapshot)
            results["tripwire_alerts"] = len(tripwire_alerts)
        except Exception as e:
            logger.warning("Geofence evaluation failed: %s", e)

        # Kill chain
        try:
            kc_alerts = self.kill_chain.observe(snapshot)
            results["kill_chain_alerts"] = len(kc_alerts)
        except Exception as e:
            logger.warning("Kill chain tracking failed: %s", e)

        # Timeline
        try:
            new_events = self.timeline.ingest(snapshot)
            results["timeline_events"] = len(new_events)
        except Exception as e:
            logger.warning("Timeline ingest failed: %s", e)

        # Wargaming — auto-detect live scenarios from real data
        try:
            live_scenarios = self.wargaming.auto_wargame(snapshot)
            results["wargame_live_scenarios"] = len(live_scenarios)
            results["wargame_escalation_risks"] = [
                {"trigger": s.trigger, "risk": s.escalation_risk,
                 "score": round(s.total_severity_score, 1)}
                for s in live_scenarios
            ]
        except Exception as e:
            logger.warning("Wargaming auto-simulation failed: %s", e)

        # Counter-surveillance
        try:
            evasion_alerts = self.counter_surveillance.observe(snapshot)
            results["evasion_alerts"] = len(evasion_alerts)
        except Exception as e:
            logger.warning("Counter-surveillance failed: %s", e)

        # Narrative warfare
        try:
            narrative_alerts = self.narrative_warfare.observe(snapshot)
            results["narrative_alerts"] = len(narrative_alerts)
        except Exception as e:
            logger.warning("Narrative warfare failed: %s", e)

        # --- Features #19-26 ---

        assert self.sigint_ew is not None
        assert self.finint is not None
        assert self.nuclear is not None
        assert self.space_domain is not None
        assert self.infrastructure is not None
        assert self.environmental is not None
        assert self.force_projection is not None
        assert self.threat_fusion is not None

        # SIGINT/EW
        try:
            sigint_alerts = self.sigint_ew.observe(snapshot)
            results["sigint_alerts"] = len(sigint_alerts)
        except Exception as e:
            logger.warning("SIGINT/EW analysis failed: %s", e)

        # FININT
        try:
            finint_alerts = self.finint.observe(snapshot)
            results["finint_alerts"] = len(finint_alerts)
        except Exception as e:
            logger.warning("FININT analysis failed: %s", e)

        # Nuclear
        try:
            nuclear_alerts = self.nuclear.observe(snapshot)
            results["nuclear_alerts"] = len(nuclear_alerts)
        except Exception as e:
            logger.warning("Nuclear monitoring failed: %s", e)

        # Space domain
        try:
            space_alerts = self.space_domain.observe(snapshot)
            results["space_alerts"] = len(space_alerts)
        except Exception as e:
            logger.warning("Space domain awareness failed: %s", e)

        # Infrastructure
        try:
            infra_alerts = self.infrastructure.observe(snapshot)
            results["infra_alerts"] = len(infra_alerts)
        except Exception as e:
            logger.warning("Infrastructure resilience failed: %s", e)

        # Environmental
        try:
            env_alerts = self.environmental.observe(snapshot)
            results["env_alerts"] = len(env_alerts)
        except Exception as e:
            logger.warning("Environmental threat intel failed: %s", e)

        # Force projection
        try:
            force_alerts = self.force_projection.observe(snapshot)
            results["force_alerts"] = len(force_alerts)
        except Exception as e:
            logger.warning("Force projection tracking failed: %s", e)

        # Threat fusion (aggregates all module outputs)
        _locals = locals()  # Capture all local variables for module alert pass-through
        try:
            fusion_inputs = {
                "sigint_ew": _locals.get("sigint_alerts", []),
                "finint": _locals.get("finint_alerts", []),
                "nuclear": _locals.get("nuclear_alerts", []),
                "space": _locals.get("space_alerts", []),
                "infrastructure": _locals.get("infra_alerts", []),
                "environmental": _locals.get("env_alerts", []),
                "force_projection": _locals.get("force_alerts", []),
                "causal": _locals.get("chains", []),
                "swarm": _locals.get("assessments", []),
                "pattern_of_life": _locals.get("pol_anomalies", []),
                "geofence": _locals.get("tripwire_alerts", []),
                "kill_chain": _locals.get("kc_alerts", []),
                "counter_surveillance": _locals.get("evasion_alerts", []),
                "narrative_warfare": _locals.get("narrative_alerts", []),
            }
            threat_picture = self.threat_fusion.fuse(fusion_inputs)
            results["threat_level"] = threat_picture.threat_level
            results["threat_name"] = threat_picture.threat_name
            results["threat_score"] = threat_picture.overall_score
            results["fusion_cross_alerts"] = len(threat_picture.fusion_alerts)
        except Exception as e:
            logger.warning("Threat fusion failed: %s", e)

        # ── Autonomous camera tour — Sable drives the map ──────────
        if self.camera_pilot and self.remote:
            try:
                _module_alerts = {
                    "nuclear": _locals.get("nuclear_alerts", []),
                    "sigint_ew": _locals.get("sigint_alerts", []),
                    "kill_chain": _locals.get("kc_alerts", []),
                    "finint": _locals.get("finint_alerts", []),
                }
                camera_report = await self.camera_pilot.narrate_cycle(
                    results, snapshot, module_alerts=_module_alerts,
                )
                results["camera_moves"] = camera_report.get("total_moves", 0)
                results["camera_styles"] = camera_report.get("styles_used", [])
                results["camera_regions"] = camera_report.get("regions_visited", [])
            except Exception as e:
                logger.warning("Autonomous camera pilot failed: %s", e)

        self._cycle_count += 1
        return results

    async def run_osint_hunt(self) -> List[FlashReport]:
        """Feature #4: Run an autonomous OSINT hunt."""
        self._check_init()
        assert self.osint_hunter is not None
        return await self.osint_hunter.scan(
            llm=self._llm if self.enable_llm else None
        )

    async def run_dream_cycle(self, snapshots: Optional[List[IntelSnapshot]] = None) -> Optional[DreamReport]:
        """Feature #7: Run a dream engine retrospective."""
        self._check_init()
        assert self.dream_engine is not None
        if not snapshots:
            # Use connector history if available
            snap = await self.connector.fetch_full()
            snapshots = [snap] if snap else []
        if not snapshots:
            return None
        return await self.dream_engine.dream(
            snapshots=snapshots,
            llm=self._llm if self.enable_llm else None,
        )

    async def generate_dossier(
        self,
        entity_type: str,
        entity_id: str,
        entity_data: Optional[Dict] = None,
    ) -> EntityDossier:
        """Feature #8: Generate a deep dossier on any entity."""
        self._check_init()
        assert self.dossier_gen is not None
        return await self.dossier_gen.generate(
            entity_type=entity_type,
            entity_id=entity_id,
            entity_data=entity_data,
            llm=self._llm if self.enable_llm else None,
        )

    def get_map_annotations(self) -> Dict:
        """Feature #9: Get all active map annotations as GeoJSON."""
        self._check_init()
        assert self.annotator is not None
        return self.annotator.get_geojson()

    def get_fusion_report(self) -> Optional[FusionReport]:
        """Feature #10: Get the latest multi-agent fusion report."""
        self._check_init()
        assert self.fusion is not None
        return self.fusion.get_latest_report()

    # ── features #11-18 API ───────────────────────────────────────────

    def query_graph(self, entity_id: str) -> Dict[str, Any]:
        """Feature #11: Query the knowledge graph for an entity."""
        self._check_init()
        assert self.knowledge_graph is not None
        result = self.knowledge_graph.query_entity(entity_id)
        return result if result else {}

    def get_geofence_map(self) -> Dict[str, Any]:
        """Feature #13: Get all geofences as GeoJSON for map overlay."""
        self._check_init()
        assert self.geofence is not None
        return self.geofence.get_fence_geojson()

    def simulate_scenario(self, trigger: str, region: str = "") -> WargameScenario:
        """Feature #16: Run a what-if wargame simulation."""
        self._check_init()
        assert self.wargaming is not None
        return self.wargaming.simulate(trigger, region)

    async def simulate_scenario_deep(self, trigger: str, region: str = "") -> WargameScenario:
        """Feature #16: Run a deep LLM-powered wargame simulation."""
        self._check_init()
        assert self.wargaming is not None
        return await self.wargaming.simulate_with_llm(
            trigger, region, self._llm if self.enable_llm else None,
        )

    async def generate_timeline_narrative(self, limit: int = 30) -> str:
        """Feature #15: Generate an intelligence narrative from timeline."""
        self._check_init()
        assert self.timeline is not None
        events = self.timeline.events[-limit:]
        return await self.timeline.generate_narrative(
            events, self._llm if self.enable_llm else None,
        )

    async def analyze_narrative(self, topic: str) -> str:
        """Feature #18: Analyze a narrative cluster for info-ops."""
        self._check_init()
        assert self.narrative_warfare is not None
        return await self.narrative_warfare.analyze_narrative(
            topic, self._llm if self.enable_llm else None,
        )

    # ── features #19-26 API ───────────────────────────────────────────

    def get_dark_fleet(self) -> List[Dict[str, Any]]:
        """Feature #20: Get tracked dark fleet vessels."""
        self._check_init()
        assert self.finint is not None
        return [{"name": v.name, "mmsi": v.mmsi, "risk": v.risk_score,
                 "behaviors": v.suspicious_behaviors}
                for v in self.finint.get_dark_fleet()]

    def get_nuclear_status(self) -> Dict[str, Any]:
        """Feature #21: Get nuclear facility status map."""
        self._check_init()
        assert self.nuclear is not None
        return {name: {"status": f.status, "country": f.country,
                       "type": f.facility_type, "alerts": f.alerts_count}
                for name, f in self.nuclear.get_facility_status().items()}

    def get_space_catalog(self) -> Dict[str, Any]:
        """Feature #22: Get space domain awareness catalog summary."""
        self._check_init()
        assert self.space_domain is not None
        return self.space_domain.get_catalog_summary()

    def get_infrastructure_dashboard(self) -> Dict[str, Any]:
        """Feature #23: Get critical infrastructure sector status."""
        self._check_init()
        assert self.infrastructure is not None
        return {s.sector: {"status": s.status, "outages": s.outages,
                           "ransomware": s.ransomware_hits,
                           "cyber": s.cyber_incidents,
                           "threat_level": s.threat_level}
                for s in self.infrastructure.get_sector_dashboard().values()}

    def get_carrier_positions(self) -> List[Dict[str, Any]]:
        """Feature #25: Get global carrier strike group positions."""
        self._check_init()
        assert self.force_projection is not None
        return [{"name": c.name, "country": c.country,
                 "lat": c.lat, "lon": c.lon, "area": c.current_area,
                 "speed": c.speed_knots, "air_activity": c.air_activity}
                for c in self.force_projection.get_carrier_positions()]

    def get_threat_picture(self) -> Optional[Dict[str, Any]]:
        """Feature #26: Get the unified threat picture (DEFCON-style)."""
        self._check_init()
        assert self.threat_fusion is not None
        pic = self.threat_fusion.get_current_picture()
        if not pic:
            return None
        return {
            "threat_level": pic.threat_level,
            "threat_name": pic.threat_name,
            "threat_color": pic.threat_color,
            "overall_score": pic.overall_score,
            "domains": {d: {"score": a.threat_score, "status": a.status,
                           "trend": a.trend, "criticals": a.critical_count}
                        for d, a in pic.domain_assessments.items()},
            "fusion_alerts": len(pic.fusion_alerts),
            "hotspots": pic.hotspots[:5],
            "escalation": pic.escalation_indicators,
            "pirs": pic.priority_intel_requirements,
        }

    # ── status & stats ────────────────────────────────────────────────

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive stats from all modules."""
        stats = {
            "skill_version": __version__,
            "initialized": self._initialized,
            "cycles_completed": self._cycle_count,
            "base_url": self.base_url,
        }
        if self._initialized:
            stats["modules"] = {}
            for name, mod in [
                ("causal_engine", self.causal_engine),
                ("swarm_assessor", self.swarm_assessor),
                ("osint_hunter", self.osint_hunter),
                ("world_model", self.world_model),
                ("temporal_memory", self.temporal_memory),
                ("dream_engine", self.dream_engine),
                ("dossier_generator", self.dossier_gen),
                ("annotator", self.annotator),
                ("fusion", self.fusion),
                ("knowledge_graph", self.knowledge_graph),
                ("pattern_of_life", self.pattern_of_life),
                ("geofence", self.geofence),
                ("kill_chain", self.kill_chain),
                ("timeline", self.timeline),
                ("wargaming", self.wargaming),
                ("counter_surveillance", self.counter_surveillance),
                ("narrative_warfare", self.narrative_warfare),
                ("sigint_ew", self.sigint_ew),
                ("finint", self.finint),
                ("nuclear", self.nuclear),
                ("space_domain", self.space_domain),
                ("infrastructure", self.infrastructure),
                ("environmental", self.environmental),
                ("force_projection", self.force_projection),
                ("threat_fusion", self.threat_fusion),
            ]:
                if mod and hasattr(mod, "get_stats"):
                    stats["modules"][name] = mod.get_stats()
        return stats

    # ── internal ──────────────────────────────────────────────────────

    def _check_init(self):
        if not self._initialized:
            raise RuntimeError(
                "ZunvraIntelSkill not initialized. Call await skill.initialize() first."
            )
