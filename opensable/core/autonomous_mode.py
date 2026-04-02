"""
Autonomous Agent Mode - Tick-based continuous operation.

Tick-based continuous autonomous operation.  Each tick is a numbered cycle:
  discover → plan → execute → learn → trace → advance

The tick counter persists across restarts.  An append-only JSONL trace
records every phase of every tick for post-hoc observability.
"""

import asyncio
import logging
import os
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from pathlib import Path
import json

from .agent import SableAgent
from .goal_system import GoalManager, GoalStatus

logger = logging.getLogger(__name__)

_PRIORITY_MAP = {"critical": 10, "high": 8, "medium": 5, "low": 3, "minimal": 1}


def _parse_priority(value) -> int:
    """Convert a priority value (int, str, or anything) to int 1-10."""
    if isinstance(value, int):
        return max(1, min(10, value))
    if isinstance(value, str):
        mapped = _PRIORITY_MAP.get(value.strip().lower())
        if mapped is not None:
            return mapped
        try:
            return max(1, min(10, int(value)))
        except ValueError:
            return 5
    return 5


def _init_sub_agents(agent):
    """Create and populate a SubAgentManager."""
    from opensable.core.sub_agents import SubAgentManager, DEFAULT_SUB_AGENTS
    mgr = SubAgentManager(agent)
    for spec in DEFAULT_SUB_AGENTS:
        mgr.register(spec)
    return mgr


async def _init_git_brain():
    """Create and initialise a GitBrain."""
    from opensable.core.git_brain import GitBrain
    gb = GitBrain(repo_dir=Path("."))
    await gb.initialize()
    return gb


class AutonomousMode:
    """
    Tick-based autonomous operation mode.

    Each iteration of the loop is a *tick*,  numbered, traced, and
    persisted.  Sub-agents and fitness tracking plug in via the manager
    fields initialized in start().
    """

    def __init__(self, agent: SableAgent, config):
        self.agent = agent
        self.config = config
        self.running = False
        self.task_queue: List[Dict] = []
        self.completed_tasks: List[Dict] = []
        self.goal_manager: Optional[GoalManager] = None
        self.x_autoposter = None
        self.ig_autoposter = None

        # ── Tick state ──────────────────────────────────────────────────
        self.tick: int = 0  # Monotonic tick counter (persisted)
        self._tick_counter: int = 0  # Alias used by cognitive module ticks
        self.tick_start: float = 0.0

        # Pluggable modules (initialized in start())
        self.trace_exporter = None      # TraceExporter
        self.sub_agent_manager = None   # SubAgentManager
        self.skill_fitness = None       # SkillFitnessTracker
        self.conversation_logger = None # ConversationLogger
        self.cognitive_memory = None    # CognitiveMemoryManager
        self.self_reflection = None     # ReflectionEngine
        self.skill_evolution = None     # SkillEvolutionManager
        self.evolution_engine = None    # EvolutionEngine (code mutation + self-restart)
        self.git_brain = None           # GitBrain
        self.inner_life = None          # InnerLifeProcessor
        self.pattern_learner = None     # PatternLearningManager
        self.proactive_engine = None    # ProactiveReasoningEngine
        self._connectome_biases = {}     # routing biases from last connectome propagation
        self.react_executor = None      # ReActExecutor
        self.github_skill = None        # GitHubSkill
        self.deep_planner = None        # DeepPlanner (10+ step DAG planning)
        self.inter_agent_bridge = None  # InterAgentBridge (shared learning vault)
        self.ultra_ltm = None           # UltraLongTermMemory (weeks/months consolidation)
        self.self_benchmark = None      # SelfBenchmark (quantified self-assessment)
        self.meta_learner = None        # MetaLearner (learning-to-learn)
        self.causal_engine = None       # CausalEngine (causal reasoning)
        self.goal_synthesis = None      # GoalSynthesis (autonomous goal generation)
        self.skill_composer = None      # SkillComposer (compound skill creation)
        self.world_predictor = None     # WorldPredictor (anticipatory reasoning)
        self.cognitive_optimizer = None # CognitiveOptimizer (pipeline self-tuning)
        self.adversarial_tester = None  # AdversarialTester (red-team self-testing)
        self.resource_governor = None   # ResourceGovernor (token/compute budgets)
        self.theory_of_mind = None      # TheoryOfMind (user modeling)
        self.ethical_reasoner = None    # EthicalReasoner (consequence analysis)
        # ── v1.5 World-First Modules ──
        self.dream_engine = None
        self.cognitive_immunity = None
        self.temporal_consciousness = None
        self.cognitive_fusion = None
        self.memory_palace = None
        self.narrative_identity = None
        self.curiosity_drive = None
        self.collective_unconscious = None
        self.cognitive_metabolism = None
        self.synthetic_intuition = None
        self.phantom_limb = None
        self.cognitive_scar = None
        self.time_crystal = None
        self.holographic_context = None
        self.swarm_cortex = None
        self.cognitive_archaeology = None
        self.emotional_contagion = None
        self.predictive_empathy = None
        self.autonomous_researcher = None
        self.empathy_synthesizer = None

        # v1.6,  Godlike cognitive modules
        self.cognitive_teleportation = None
        self.ontological_engine = None
        self.cognitive_gravity = None
        self.temporal_paradox = None
        self.synaesthetic_processor = None
        self.cognitive_mitosis = None
        self.entropic_sentinel = None
        self.quantum_cognition = None
        self.cognitive_placebo = None
        self.noospheric_interface = None
        self.akashic_records = None
        self.deja_vu = None
        self.morphogenetic_field = None
        self.liminal_processor = None
        self.prescient_executor = None
        self.cognitive_dark_matter = None
        self.ego_membrane = None
        self.hyperstition_engine = None
        self.cognitive_chrysalis = None
        self.existential_compass = None
        # ── v1.7 God Supreme Modules ──
        self.web_agent = None
        self.self_healer = None
        self.dynamic_skill_factory = None
        self.multimodal_engine = None
        self.internet_monitor = None
        self.financial_autonomy = None
        self.social_presence = None
        self.self_replicator = None
        self.continuous_learner = None
        self.nl_automation = None
        self.video_understanding = None
        self.knowledge_graph = None
        self.iot_controller = None
        self.distributed_task_queue = None

        # Autonomous operation settings
        self.check_interval = getattr(config, "autonomous_check_interval", 60)  # seconds
        self.max_concurrent_tasks = getattr(config, "autonomous_max_tasks", 3)
        _sources = getattr(config, "autonomous_sources", "calendar,email,system_monitoring")
        if isinstance(_sources, str):
            self.enabled_sources = [s.strip() for s in _sources.split(",") if s.strip()]
        else:
            self.enabled_sources = list(_sources)

    async def start(self):
        """Start autonomous operation"""
        logger.info("🤖 Starting autonomous mode...")

        # ── Inherit modules from agent (single source of truth) ───────────
        # Use agent's profile-aware data dir, falling back to _SABLE_DATA_DIR env
        data_dir = Path(
            getattr(self.agent, "_data_dir", None)
            or os.environ.get("_SABLE_DATA_DIR")
            or "./data"
        )

        # Helper: reuse agent's instance when available, else create fresh
        def _inherit(attr_name, factory, label):
            """Reuse self.agent.<attr_name> if available, else call factory()."""
            existing = getattr(self.agent, attr_name, None)
            if existing:
                logger.info(f"♻️  {label} (shared with agent)")
                return existing
            try:
                instance = factory()
                logger.info(f"📦 {label} (new instance)")
                return instance
            except Exception as e:
                logger.warning(f"{label} unavailable: {e}")
                return None

        async def _inherit_async(attr_name, factory, label):
            existing = getattr(self.agent, attr_name, None)
            if existing:
                logger.info(f"♻️  {label} (shared with agent)")
                return existing
            try:
                instance = await factory()
                logger.info(f"📦 {label} (new instance)")
                return instance
            except Exception as e:
                logger.warning(f"{label} unavailable: {e}")
                return None

        self.trace_exporter = _inherit("trace_exporter", lambda: __import__(
            "opensable.core.trace_exporter", fromlist=["TraceExporter"]
        ).TraceExporter(directory=data_dir / "traces"), "Trace exporter")

        self.sub_agent_manager = _inherit("sub_agent_manager", lambda: _init_sub_agents(self.agent), "Sub-agent manager")

        self.skill_fitness = _inherit("skill_fitness", lambda: __import__(
            "opensable.core.skill_fitness", fromlist=["SkillFitnessTracker"]
        ).SkillFitnessTracker(directory=data_dir / "fitness"), "Skill fitness")

        self.conversation_logger = _inherit("conversation_logger", lambda: __import__(
            "opensable.core.conversation_log", fromlist=["ConversationLogger"]
        ).ConversationLogger(directory=data_dir / "conversations"), "Conversation logger")

        self.cognitive_memory = _inherit("cognitive_memory", lambda: __import__(
            "opensable.core.cognitive_memory", fromlist=["CognitiveMemoryManager"]
        ).CognitiveMemoryManager(directory=data_dir / "cognitive_memory"), "Cognitive memory")

        self.self_reflection = _inherit("self_reflection", lambda: __import__(
            "opensable.core.self_reflection", fromlist=["ReflectionEngine"]
        ).ReflectionEngine(directory=data_dir / "reflection"), "Self-reflection")

        self.skill_evolution = _inherit("skill_evolution", lambda: __import__(
            "opensable.core.skill_evolution", fromlist=["SkillEvolutionManager"]
        ).SkillEvolutionManager(directory=data_dir / "skill_evolution"), "Skill evolution")

        # Evolution engine,  autonomous code mutation + self-restart
        _profile = getattr(self.config, "profile_name", None) or os.environ.get("SABLE_PROFILE", "sable")
        _base_dir = Path(__file__).resolve().parent.parent.parent  # repo root
        self.evolution_engine = _inherit("evolution_engine", lambda: __import__(
            "opensable.core.evolution_engine", fromlist=["EvolutionEngine"]
        ).EvolutionEngine(
            base_dir=_base_dir,
            data_dir=data_dir / "evolution",
            profile=_profile,
        ), "Evolution engine")

        self.git_brain = await _inherit_async("git_brain", _init_git_brain, "Git brain")

        self.inner_life = _inherit("inner_life", lambda: __import__(
            "opensable.core.inner_life", fromlist=["InnerLifeProcessor"]
        ).InnerLifeProcessor(data_dir=data_dir / "inner_life"), "Inner life")

        self.pattern_learner = _inherit("pattern_learner", lambda: __import__(
            "opensable.core.pattern_learner", fromlist=["PatternLearningManager"]
        ).PatternLearningManager(directory=data_dir / "patterns"), "Pattern learner")

        think_interval = getattr(self.config, "proactive_think_every_n_ticks", 5)
        max_risk = getattr(self.config, "proactive_max_risk", "medium")
        self.proactive_engine = _inherit("proactive_engine", lambda: __import__(
            "opensable.core.proactive_reasoning", fromlist=["ProactiveReasoningEngine"]
        ).ProactiveReasoningEngine(
            directory=data_dir / "proactive",
            think_every_n_ticks=think_interval,
            max_risk_level=max_risk,
        ), "Proactive reasoning")

        self.react_executor = _inherit("react_executor", lambda: __import__(
            "opensable.core.react_executor", fromlist=["ReActExecutor"]
        ).ReActExecutor(
            max_steps=getattr(self.config, "react_max_steps", 8),
            timeout_s=getattr(self.config, "react_timeout_s", 180.0),
            log_dir=data_dir / "react_logs",
        ), "ReAct executor")

        self.deep_planner = _inherit("deep_planner", lambda: __import__(
            "opensable.core.deep_planner", fromlist=["DeepPlanner"]
        ).DeepPlanner(data_dir=data_dir / "deep_planner"), "Deep planner")

        _profile = getattr(self.config, "profile_name", None) or os.environ.get("SABLE_PROFILE", "sable")
        self.inter_agent_bridge = _inherit("inter_agent_bridge", lambda: __import__(
            "opensable.core.inter_agent_bridge", fromlist=["InterAgentBridge"]
        ).InterAgentBridge(
            profile=_profile,
            shared_dir=Path("data") / "shared_learnings",
            local_dir=data_dir / "inter_agent",
        ), "Inter-agent bridge")

        self.ultra_ltm = _inherit("ultra_ltm", lambda: __import__(
            "opensable.core.ultra_ltm", fromlist=["UltraLongTermMemory"]
        ).UltraLongTermMemory(data_dir=data_dir / "ultra_ltm"), "Ultra long-term memory")

        self.self_benchmark = _inherit("self_benchmark", lambda: __import__(
            "opensable.core.self_benchmark", fromlist=["SelfBenchmark"]
        ).SelfBenchmark(data_dir=data_dir / "self_benchmark"), "Self benchmark")

        self.meta_learner = _inherit("meta_learner", lambda: __import__(
            "opensable.core.meta_learner", fromlist=["MetaLearner"]
        ).MetaLearner(data_dir=data_dir / "meta_learner"), "Meta learner")

        self.causal_engine = _inherit("causal_engine", lambda: __import__(
            "opensable.core.causal_engine", fromlist=["CausalEngine"]
        ).CausalEngine(data_dir=data_dir / "causal_engine"), "Causal engine")

        self.goal_synthesis = _inherit("goal_synthesis", lambda: __import__(
            "opensable.core.goal_synthesis", fromlist=["GoalSynthesis"]
        ).GoalSynthesis(data_dir=data_dir / "goal_synthesis"), "Goal synthesis")

        self.skill_composer = _inherit("skill_composer", lambda: __import__(
            "opensable.core.skill_composer", fromlist=["SkillComposer"]
        ).SkillComposer(data_dir=data_dir / "skill_composer"), "Skill composer")

        self.world_predictor = _inherit("world_predictor", lambda: __import__(
            "opensable.core.world_predictor", fromlist=["WorldPredictor"]
        ).WorldPredictor(data_dir=data_dir / "world_predictor"), "World predictor")

        self.cognitive_optimizer = _inherit("cognitive_optimizer", lambda: __import__(
            "opensable.core.cognitive_optimizer", fromlist=["CognitiveOptimizer"]
        ).CognitiveOptimizer(data_dir=data_dir / "cognitive_optimizer"), "Cognitive optimizer")

        self.adversarial_tester = _inherit("adversarial_tester", lambda: __import__(
            "opensable.core.adversarial_tester", fromlist=["AdversarialTester"]
        ).AdversarialTester(data_dir=data_dir / "adversarial_tester"), "Adversarial tester")

        self.resource_governor = _inherit("resource_governor", lambda: __import__(
            "opensable.core.resource_governor", fromlist=["ResourceGovernor"]
        ).ResourceGovernor(data_dir=data_dir / "resource_governor"), "Resource governor")

        self.theory_of_mind = _inherit("theory_of_mind", lambda: __import__(
            "opensable.core.theory_of_mind", fromlist=["TheoryOfMind"]
        ).TheoryOfMind(data_dir=data_dir / "theory_of_mind"), "Theory of mind")

        self.ethical_reasoner = _inherit("ethical_reasoner", lambda: __import__(
            "opensable.core.ethical_reasoner", fromlist=["EthicalReasoner"]
        ).EthicalReasoner(data_dir=data_dir / "ethical_reasoner"), "Ethical reasoner")

        # ── v1.5 World-First Modules _inherit ──
        self.dream_engine = _inherit("dream_engine", lambda: __import__(
            "opensable.core.dream_engine", fromlist=["DreamEngine"]
        ).DreamEngine(data_dir=data_dir / "dream_engine"), "Dream engine")

        self.cognitive_immunity = _inherit("cognitive_immunity", lambda: __import__(
            "opensable.core.cognitive_immunity", fromlist=["CognitiveImmunity"]
        ).CognitiveImmunity(data_dir=data_dir / "cognitive_immunity"), "Cognitive immunity")

        self.temporal_consciousness = _inherit("temporal_consciousness", lambda: __import__(
            "opensable.core.temporal_consciousness", fromlist=["TemporalConsciousness"]
        ).TemporalConsciousness(data_dir=data_dir / "temporal_consciousness"), "Temporal consciousness")

        self.cognitive_fusion = _inherit("cognitive_fusion", lambda: __import__(
            "opensable.core.cognitive_fusion", fromlist=["CognitiveFusion"]
        ).CognitiveFusion(data_dir=data_dir / "cognitive_fusion"), "Cognitive fusion")

        self.memory_palace = _inherit("memory_palace", lambda: __import__(
            "opensable.core.memory_palace", fromlist=["MemoryPalace"]
        ).MemoryPalace(data_dir=data_dir / "memory_palace"), "Memory palace")

        self.narrative_identity = _inherit("narrative_identity", lambda: __import__(
            "opensable.core.narrative_identity", fromlist=["NarrativeIdentity"]
        ).NarrativeIdentity(data_dir=data_dir / "narrative_identity"), "Narrative identity")

        self.curiosity_drive = _inherit("curiosity_drive", lambda: __import__(
            "opensable.core.curiosity_drive", fromlist=["CuriosityDrive"]
        ).CuriosityDrive(data_dir=data_dir / "curiosity_drive"), "Curiosity drive")

        self.collective_unconscious = _inherit("collective_unconscious", lambda: __import__(
            "opensable.core.collective_unconscious", fromlist=["CollectiveUnconscious"]
        ).CollectiveUnconscious(data_dir=data_dir / "collective_unconscious"), "Collective unconscious")

        self.cognitive_metabolism = _inherit("cognitive_metabolism", lambda: __import__(
            "opensable.core.cognitive_metabolism", fromlist=["CognitiveMetabolism"]
        ).CognitiveMetabolism(data_dir=data_dir / "cognitive_metabolism"), "Cognitive metabolism")

        self.synthetic_intuition = _inherit("synthetic_intuition", lambda: __import__(
            "opensable.core.synthetic_intuition", fromlist=["SyntheticIntuition"]
        ).SyntheticIntuition(data_dir=data_dir / "synthetic_intuition"), "Synthetic intuition")

        self.phantom_limb = _inherit("phantom_limb", lambda: __import__(
            "opensable.core.phantom_limb", fromlist=["PhantomLimb"]
        ).PhantomLimb(data_dir=data_dir / "phantom_limb"), "Phantom limb")

        self.cognitive_scar = _inherit("cognitive_scar", lambda: __import__(
            "opensable.core.cognitive_scar", fromlist=["CognitiveScar"]
        ).CognitiveScar(data_dir=data_dir / "cognitive_scar"), "Cognitive scar")

        self.time_crystal = _inherit("time_crystal", lambda: __import__(
            "opensable.core.time_crystal", fromlist=["TimeCrystalMemory"]
        ).TimeCrystalMemory(data_dir=data_dir / "time_crystal"), "Time crystal")

        self.holographic_context = _inherit("holographic_context", lambda: __import__(
            "opensable.core.holographic_context", fromlist=["HolographicContext"]
        ).HolographicContext(data_dir=data_dir / "holographic_context"), "Holographic context")

        self.swarm_cortex = _inherit("swarm_cortex", lambda: __import__(
            "opensable.core.swarm_cortex", fromlist=["SwarmCortex"]
        ).SwarmCortex(data_dir=data_dir / "swarm_cortex"), "Swarm cortex")

        self.cognitive_archaeology = _inherit("cognitive_archaeology", lambda: __import__(
            "opensable.core.cognitive_archaeology", fromlist=["CognitiveArchaeology"]
        ).CognitiveArchaeology(data_dir=data_dir / "cognitive_archaeology"), "Cognitive archaeology")

        self.emotional_contagion = _inherit("emotional_contagion", lambda: __import__(
            "opensable.core.emotional_contagion", fromlist=["EmotionalContagion"]
        ).EmotionalContagion(data_dir=data_dir / "emotional_contagion"), "Emotional contagion")

        self.predictive_empathy = _inherit("predictive_empathy", lambda: __import__(
            "opensable.core.predictive_empathy", fromlist=["PredictiveEmpathy"]
        ).PredictiveEmpathy(data_dir=data_dir / "predictive_empathy"), "Predictive empathy")

        self.autonomous_researcher = _inherit("autonomous_researcher", lambda: __import__(
            "opensable.core.autonomous_researcher", fromlist=["AutonomousResearcher"]
        ).AutonomousResearcher(data_dir=data_dir / "autonomous_researcher"), "Autonomous researcher")

        self.empathy_synthesizer = _inherit("empathy_synthesizer", lambda: __import__(
            "opensable.core.empathy_synthesizer", fromlist=["EmpathySynthesizer"]
        ).EmpathySynthesizer(data_dir=data_dir / "empathy_synthesizer"), "Empathy synthesizer")

        # v1.6,  Godlike modules
        self.cognitive_teleportation = _inherit("cognitive_teleportation", lambda: __import__(
            "opensable.core.cognitive_teleportation", fromlist=["CognitiveTeleportation"]
        ).CognitiveTeleportation(data_dir=data_dir / "cognitive_teleportation"), "Cognitive teleportation")

        self.ontological_engine = _inherit("ontological_engine", lambda: __import__(
            "opensable.core.ontological_engine", fromlist=["OntologicalEngine"]
        ).OntologicalEngine(data_dir=data_dir / "ontological_engine"), "Ontological engine")

        self.cognitive_gravity = _inherit("cognitive_gravity", lambda: __import__(
            "opensable.core.cognitive_gravity", fromlist=["CognitiveGravity"]
        ).CognitiveGravity(data_dir=data_dir / "cognitive_gravity"), "Cognitive gravity")

        self.temporal_paradox = _inherit("temporal_paradox", lambda: __import__(
            "opensable.core.temporal_paradox", fromlist=["TemporalParadoxResolver"]
        ).TemporalParadoxResolver(data_dir=data_dir / "temporal_paradox"), "Temporal paradox")

        self.synaesthetic_processor = _inherit("synaesthetic_processor", lambda: __import__(
            "opensable.core.synaesthetic_processor", fromlist=["SynaestheticProcessor"]
        ).SynaestheticProcessor(data_dir=data_dir / "synaesthetic_processor"), "Synaesthetic processor")

        self.cognitive_mitosis = _inherit("cognitive_mitosis", lambda: __import__(
            "opensable.core.cognitive_mitosis", fromlist=["CognitiveMitosis"]
        ).CognitiveMitosis(data_dir=data_dir / "cognitive_mitosis"), "Cognitive mitosis")

        self.entropic_sentinel = _inherit("entropic_sentinel", lambda: __import__(
            "opensable.core.entropic_sentinel", fromlist=["EntropicSentinel"]
        ).EntropicSentinel(data_dir=data_dir / "entropic_sentinel"), "Entropic sentinel")

        self.quantum_cognition = _inherit("quantum_cognition", lambda: __import__(
            "opensable.core.quantum_cognition", fromlist=["QuantumCognition"]
        ).QuantumCognition(data_dir=data_dir / "quantum_cognition"), "Quantum cognition")

        self.cognitive_placebo = _inherit("cognitive_placebo", lambda: __import__(
            "opensable.core.cognitive_placebo", fromlist=["CognitivePlacebo"]
        ).CognitivePlacebo(data_dir=data_dir / "cognitive_placebo"), "Cognitive placebo")

        self.noospheric_interface = _inherit("noospheric_interface", lambda: __import__(
            "opensable.core.noospheric_interface", fromlist=["NoosphericInterface"]
        ).NoosphericInterface(data_dir=data_dir / "noospheric_interface"), "Noospheric interface")

        self.akashic_records = _inherit("akashic_records", lambda: __import__(
            "opensable.core.akashic_records", fromlist=["AkashicRecords"]
        ).AkashicRecords(data_dir=data_dir / "akashic_records"), "Akashic records")

        self.deja_vu = _inherit("deja_vu", lambda: __import__(
            "opensable.core.deja_vu", fromlist=["DejaVuEngine"]
        ).DejaVuEngine(data_dir=data_dir / "deja_vu"), "Deja vu")

        self.morphogenetic_field = _inherit("morphogenetic_field", lambda: __import__(
            "opensable.core.morphogenetic_field", fromlist=["MorphogeneticField"]
        ).MorphogeneticField(data_dir=data_dir / "morphogenetic_field"), "Morphogenetic field")

        self.liminal_processor = _inherit("liminal_processor", lambda: __import__(
            "opensable.core.liminal_processor", fromlist=["LiminalProcessor"]
        ).LiminalProcessor(data_dir=data_dir / "liminal_processor"), "Liminal processor")

        self.prescient_executor = _inherit("prescient_executor", lambda: __import__(
            "opensable.core.prescient_executor", fromlist=["PrescientExecutor"]
        ).PrescientExecutor(data_dir=data_dir / "prescient_executor"), "Prescient executor")

        self.cognitive_dark_matter = _inherit("cognitive_dark_matter", lambda: __import__(
            "opensable.core.cognitive_dark_matter", fromlist=["CognitiveDarkMatter"]
        ).CognitiveDarkMatter(data_dir=data_dir / "cognitive_dark_matter"), "Cognitive dark matter")

        self.ego_membrane = _inherit("ego_membrane", lambda: __import__(
            "opensable.core.ego_membrane", fromlist=["EgoMembrane"]
        ).EgoMembrane(data_dir=data_dir / "ego_membrane"), "Ego membrane")

        self.hyperstition_engine = _inherit("hyperstition_engine", lambda: __import__(
            "opensable.core.hyperstition_engine", fromlist=["HyperstitionEngine"]
        ).HyperstitionEngine(data_dir=data_dir / "hyperstition_engine"), "Hyperstition engine")

        self.cognitive_chrysalis = _inherit("cognitive_chrysalis", lambda: __import__(
            "opensable.core.cognitive_chrysalis", fromlist=["CognitiveChrysalis"]
        ).CognitiveChrysalis(data_dir=data_dir / "cognitive_chrysalis"), "Cognitive chrysalis")

        self.existential_compass = _inherit("existential_compass", lambda: __import__(
            "opensable.core.existential_compass", fromlist=["ExistentialCompass"]
        ).ExistentialCompass(data_dir=data_dir / "existential_compass"), "Existential compass")

        # ── v1.7 God Supreme inherits ──
        self.web_agent = _inherit("web_agent", lambda: __import__(
            "opensable.core.autonomous_web_agent", fromlist=["AutonomousWebAgent"]
        ).AutonomousWebAgent(data_dir=data_dir / "web_agent"), "Web agent")

        self.self_healer = _inherit("self_healer", lambda: __import__(
            "opensable.core.self_healer", fromlist=["SelfHealer"]
        ).SelfHealer(data_dir=data_dir / "self_healer"), "Self healer")

        self.dynamic_skill_factory = _inherit("dynamic_skill_factory", lambda: __import__(
            "opensable.core.dynamic_skill_factory", fromlist=["DynamicSkillFactory"]
        ).DynamicSkillFactory(data_dir=data_dir / "dynamic_skill_factory"), "Dynamic skill factory")

        self.multimodal_engine = _inherit("multimodal_engine", lambda: __import__(
            "opensable.core.multimodal_engine", fromlist=["MultiModalEngine"]
        ).MultiModalEngine(data_dir=data_dir / "multimodal_engine"), "Multimodal engine")

        self.internet_monitor = _inherit("internet_monitor", lambda: __import__(
            "opensable.core.internet_monitor", fromlist=["InternetMonitor"]
        ).InternetMonitor(data_dir=data_dir / "internet_monitor"), "Internet monitor")

        self.financial_autonomy = _inherit("financial_autonomy", lambda: __import__(
            "opensable.core.financial_autonomy", fromlist=["FinancialAutonomy"]
        ).FinancialAutonomy(data_dir=data_dir / "financial_autonomy"), "Financial autonomy")

        self.social_presence = _inherit("social_presence", lambda: __import__(
            "opensable.core.social_presence", fromlist=["SocialPresenceBuilder"]
        ).SocialPresenceBuilder(data_dir=data_dir / "social_presence"), "Social presence")

        self.self_replicator = _inherit("self_replicator", lambda: __import__(
            "opensable.core.self_replicator", fromlist=["SelfReplicator"]
        ).SelfReplicator(data_dir=data_dir / "self_replicator"), "Self replicator")

        self.continuous_learner = _inherit("continuous_learner", lambda: __import__(
            "opensable.core.continuous_learner", fromlist=["ContinuousLearner"]
        ).ContinuousLearner(data_dir=data_dir / "continuous_learner"), "Continuous learner")

        self.nl_automation = _inherit("nl_automation", lambda: __import__(
            "opensable.core.nl_automation", fromlist=["NLAutomationEngine"]
        ).NLAutomationEngine(data_dir=data_dir / "nl_automation"), "NL automation")

        self.video_understanding = _inherit("video_understanding", lambda: __import__(
            "opensable.core.video_understanding", fromlist=["VideoUnderstandingEngine"]
        ).VideoUnderstandingEngine(data_dir=data_dir / "video_understanding"), "Video understanding")

        self.knowledge_graph = _inherit("knowledge_graph", lambda: __import__(
            "opensable.core.knowledge_graph", fromlist=["KnowledgeGraphEngine"]
        ).KnowledgeGraphEngine(data_dir=data_dir / "knowledge_graph"), "Knowledge graph")

        self.iot_controller = _inherit("iot_controller", lambda: __import__(
            "opensable.core.iot_controller", fromlist=["IoTController"]
        ).IoTController(data_dir=data_dir / "iot_controller"), "IoT controller")

        self.distributed_task_queue = _inherit("distributed_task_queue", lambda: __import__(
            "opensable.core.distributed_task_queue", fromlist=["DistributedTaskQueue"]
        ).DistributedTaskQueue(data_dir=data_dir / "distributed_tasks"), "Distributed task queue")

        self.github_skill = _inherit("github_skill", lambda: None, "GitHub skill")
        if not self.github_skill:
            try:
                from opensable.skills.automation.github_skill import GitHubSkill
                skill = GitHubSkill(self.config)
                # GitHub skill needs async init,  we'll do it inline
                self.github_skill = skill
            except Exception as e:
                logger.warning(f"GitHub skill unavailable: {e}")

        # Load persisted tick counter
        await self._load_state()

        # Connect goal manager from the agent
        self.goal_manager = getattr(self.agent, "goals", None)
        if self.goal_manager:
            logger.info("Goal system connected")
        else:
            logger.warning("Goal system not available")

        # Initialize X Autoposter if enabled and not already running
        if (
            getattr(self.config, "x_autoposter_enabled", False)
            and getattr(self.config, "x_enabled", False)
            and not getattr(self.agent, "x_autoposter", None)
        ):
            try:
                from .x_autoposter import XAutoposter

                self.x_autoposter = XAutoposter(self.agent, self.config)
                self.agent.x_autoposter = self.x_autoposter  # expose to gateway

                async def _run_ap():
                    try:
                        await self.x_autoposter.start()
                    except Exception as exc:
                        logger.error(f"🐦 X Autoposter crashed: {exc}", exc_info=True)

                asyncio.create_task(_run_ap())
                logger.info("🐦 X Autoposter launched as background task")
            except Exception as e:
                logger.error(f"Failed to start X Autoposter: {e}", exc_info=True)

        # Initialize IG Autoposter if enabled
        if (
            os.getenv("IG_AUTOPOSTER_ENABLED", "false").lower() in ("true", "1", "yes")
            and not getattr(self.agent, "ig_autoposter", None)
        ):
            try:
                from .ig_autoposter import IGAutoposter

                self.ig_autoposter = IGAutoposter(self.agent, self.config)
                self.agent.ig_autoposter = self.ig_autoposter

                async def _run_ig_ap():
                    try:
                        await self.ig_autoposter.start()
                    except Exception as exc:
                        logger.error(f"📸 IG Autoposter crashed: {exc}", exc_info=True)

                asyncio.create_task(_run_ig_ap())
                logger.info("📸 IG Autoposter launched as background task")
            except Exception as e:
                logger.error(f"Failed to start IG Autoposter: {e}", exc_info=True)

        self.running = True

        # Start main loop
        await self._autonomous_loop()

    async def stop(self):
        """Stop autonomous operation"""
        logger.info("Stopping autonomous mode...")
        self.running = False
        if self.x_autoposter:
            await self.x_autoposter.stop()
        if self.ig_autoposter:
            await self.ig_autoposter.stop()

    async def _autonomous_loop(self):
        """Main tick-based autonomous loop.

        Each iteration is a numbered *tick*.  Pipeline per tick:
          1. trace_tick_start
          2. discover tasks
          3. prioritize
          4. execute
          5. collect sub-agent results
          6. self-improve
          7. maintain
          8. trace_tick_end + advance tick counter
        """
        logger.info(
            f"Autonomous loop started,  tick {self.tick} "
            f"(interval: {self.check_interval}s)"
        )

        consecutive_errors = 0
        _MAX_CONSECUTIVE_ERRORS = 10
        _BACKOFF_MULTIPLIER = 2  # double interval after many failures

        # ── Immediate arena fight on startup ────────────────────────────
        # Queue a fight right away so both agents enter the matchmaker
        # queue within seconds of starting instead of waiting 10+ ticks.
        try:
            arena_skill = getattr(self.agent.tools, "arena_skill", None) if self.agent else None
            if arena_skill and getattr(arena_skill, '_ready', False):
                fight_task = {
                    "id": f"arena_startup_{self.tick}",
                    "type": "proactive",
                    "description": "Arena: startup fight — entering queue immediately",
                    "goal_type": "creative",
                    "priority": 1,  # high priority
                    "tool_name": "arena_fight",
                    "tool_args": {"use_llm": True},
                    "reasoning": "Immediate arena queue on startup",
                    "risk_level": "low",
                    "created_at": datetime.now(),
                }
                self.task_queue.append(fight_task)
                logger.info("🥊 Arena: queued startup fight immediately")
            else:
                logger.info("🥊 Arena: skill not ready at startup, will auto-queue later")
        except Exception as e:
            logger.debug(f"Arena startup queue: {e}")

        while self.running:
            self.tick_start = time.monotonic()
            try:
                # ── Phase 0: Trace tick start ───────────────────────────
                if self.trace_exporter:
                    self.trace_exporter.record_tick_start(
                        tick=self.tick,
                    )

                # ── Phase 1: Discover ───────────────────────────────────
                await self._discover_tasks()

                # ── Phase 2: Prioritize ─────────────────────────────────
                await self._prioritize_tasks()

                # ── Phase 3: Execute ────────────────────────────────────
                await self._execute_tasks()

                # ── Phase 4: Collect sub-agent results ──────────────────
                if self.sub_agent_manager and self.sub_agent_manager.pending_count > 0:
                    results = await self.sub_agent_manager.await_all(timeout_s=30.0)
                    for task_id, result in results.items():
                        if self.trace_exporter:
                            self.trace_exporter.record_event(
                                "sub_agent_result",
                                summary=f"{result.agent_name}: {result.status},  {result.task[:60]}",
                                tick=self.tick,
                                data={"task_id": task_id, "duration_ms": result.duration_ms},
                            )
                    self.sub_agent_manager.clear_inbox()

                # ── Phase 5: Self-improvement ───────────────────────────
                if self.goal_manager:
                    await self._self_improve()

                # ── Phase 6: Proactive reasoning ─────────────────────
                await self._proactive_tick()

                # ── Phase 7: Cognitive processing ───────────────────────
                await self._cognitive_tick()

                # ── Phase 8: Maintenance + state save ───────────────────
                await self._perform_maintenance()

                # ── Phase 9: Trace tick end + advance ───────────────────
                tick_duration = (time.monotonic() - self.tick_start) * 1000
                if self.trace_exporter:
                    self.trace_exporter.record_tick_end(
                        tick=self.tick,
                        summary=f"completed {len(self.completed_tasks)} tasks",
                        duration_ms=tick_duration,
                    )

                logger.info(
                    f"✅ Tick {self.tick} complete ({tick_duration:.0f}ms, "
                    f"{len(self.task_queue)} queued, "
                    f"{len(self.completed_tasks)} completed)"
                )
                self.tick += 1
                self._tick_counter = self.tick
                consecutive_errors = 0  # Reset on success

                # Wait before next tick
                await asyncio.sleep(self.check_interval)

            except Exception as e:
                consecutive_errors += 1
                logger.error(f"Error in tick {self.tick}: {e} (consecutive: {consecutive_errors})")
                if self.trace_exporter:
                    self.trace_exporter.record_event(
                        "tick_error",
                        summary=str(e),
                        tick=self.tick,
                    )

                # Circuit breaker: back off when many consecutive failures
                if consecutive_errors >= _MAX_CONSECUTIVE_ERRORS:
                    backoff = self.check_interval * _BACKOFF_MULTIPLIER
                    logger.warning(
                        f"⚠️ Circuit breaker: {consecutive_errors} consecutive failures. "
                        f"Backing off to {backoff}s interval."
                    )
                    await asyncio.sleep(backoff)
                else:
                    await asyncio.sleep(self.check_interval)

    async def _discover_tasks(self):
        """Discover tasks from various sources"""

        # Check calendar for upcoming events
        if "calendar" in self.enabled_sources:
            await self._check_calendar()

        # Check email for action items
        if "email" in self.enabled_sources:
            await self._check_email()

        # Monitor system resources
        if "system_monitoring" in self.enabled_sources:
            await self._check_system()

        # Check for trading opportunities
        if "trading" in self.enabled_sources:
            await self._check_trading()

        # Check world news for situational awareness
        if "news" in self.enabled_sources:
            await self._check_news()

        # Check for scheduled goals (if Agentic AI available)
        if self.goal_manager:
            await self._check_goals()

    async def _check_calendar(self):
        """Check calendar for upcoming events and create actionable tasks."""
        try:
            result = await self.agent.tools.execute(
                "calendar", {"action": "list", "timeframe": "24h"}
            )
            result_text = str(result).strip()

            # Skip empty results
            if not result_text or "no " in result_text.lower() or len(result_text) < 20:
                logger.debug("Calendar: no upcoming events")
                return

            # Use LLM to extract actionable tasks from calendar events
            if self.agent.llm:
                messages = [
                    {"role": "system", "content": (
                        "You are an autonomous agent's calendar analyzer. "
                        "Extract actionable tasks from calendar events. "
                        "Output ONLY valid JSON array. Each item: "
                        '{"description": "what to do", "priority": 1-10, "type": "reminder"}. '
                        "If nothing actionable, return []."
                    )},
                    {"role": "user", "content": f"Calendar events:\n{result_text[:2000]}"},
                ]
                try:
                    response = await self.agent.llm.invoke_with_tools(messages, [])
                    self._inject_llm_tasks(response.get("text", ""), source="calendar")
                except Exception as e:
                    logger.debug(f"Calendar LLM analysis failed: {e}")
            else:
                logger.debug("Checked calendar (no LLM for analysis)")

        except Exception as e:
            logger.debug(f"Calendar check skipped: {e}")

    async def _check_email(self):
        """Check email for action items and create tasks from them."""
        try:
            result = await self.agent.tools.execute(
                "email", {"action": "read", "filter": "unread", "limit": 5}
            )
            result_text = str(result).strip()

            if not result_text or "no " in result_text.lower() or len(result_text) < 20:
                logger.debug("Email: no unread messages")
                return

            # Use LLM to extract action items from emails
            if self.agent.llm:
                messages = [
                    {"role": "system", "content": (
                        "You are an autonomous agent's email analyzer. "
                        "Extract actionable tasks from emails. "
                        "Output ONLY valid JSON array. Each item: "
                        '{"description": "what to do", "priority": 1-10, "type": "email_action"}. '
                        "Ignore spam and newsletters. If nothing actionable, return []."
                    )},
                    {"role": "user", "content": f"Emails:\n{result_text[:3000]}"},
                ]
                try:
                    response = await self.agent.llm.invoke_with_tools(messages, [])
                    self._inject_llm_tasks(response.get("text", ""), source="email")
                except Exception as e:
                    logger.debug(f"Email LLM analysis failed: {e}")
            else:
                logger.debug("Checked email (no LLM for analysis)")

        except Exception as e:
            logger.debug(f"Email check skipped: {e}")

    async def _check_system(self):
        """Monitor system resources and create maintenance tasks if needed."""
        try:
            result = await self.agent.tools.execute("system_info", {})
            result_text = str(result).strip()

            if not result_text or len(result_text) < 10:
                return

            # Parse common system thresholds directly (no LLM needed)
            tasks_created = 0

            # Check for disk space issues
            import re
            disk_matches = re.findall(r"(\d+(?:\.\d+)?)\s*%\s*(?:used|full|disk)", result_text, re.IGNORECASE)
            for pct in disk_matches:
                if float(pct) > 90:
                    task_id = f"system_disk_{datetime.now().strftime('%Y%m%d')}"
                    if not any(t.get("id") == task_id for t in self.task_queue):
                        self.task_queue.append({
                            "id": task_id,
                            "type": "system_maintenance",
                            "description": f"Disk usage at {pct}%,  clean up temp files, old logs, caches",
                            "priority": 8,
                            "created_at": datetime.now(),
                        })
                        tasks_created += 1
                        logger.warning(f"⚠️ Disk usage {pct}%,  created cleanup task")

            # Check for high memory usage
            mem_matches = re.findall(r"(?:memory|ram|mem)[^0-9]*(\d+(?:\.\d+)?)\s*%", result_text, re.IGNORECASE)
            for pct in mem_matches:
                if float(pct) > 90:
                    task_id = f"system_memory_{datetime.now().strftime('%Y%m%d_%H')}"
                    if not any(t.get("id") == task_id for t in self.task_queue):
                        self.task_queue.append({
                            "id": task_id,
                            "type": "system_maintenance",
                            "description": f"Memory usage at {pct}%,  identify memory-heavy processes",
                            "priority": 7,
                            "created_at": datetime.now(),
                        })
                        tasks_created += 1
                        logger.warning(f"⚠️ Memory at {pct}%")

            if tasks_created:
                logger.info(f"🖥️ System check: created {tasks_created} maintenance task(s)")
            else:
                logger.debug("System check: all healthy")

        except Exception as e:
            logger.debug(f"System check skipped: {e}")

    def _inject_llm_tasks(self, llm_text: str, source: str = "unknown"):
        """Parse LLM JSON output and inject tasks into the queue.

        Deduplicates against both pending queue AND recently completed tasks
        to prevent the agent from endlessly re-creating already-done work.
        """
        import re as _re

        text = llm_text.strip()

        # Strip <think> blocks (Qwen3 / DeepSeek reasoning)
        text = _re.sub(r"<think>[\s\S]*?</think>", "", text).strip()
        # Strip role prefix
        text = _re.sub(r"^(?:system|assistant)\s*\n", "", text).strip()

        # Extract JSON from potential markdown code block
        if "```" in text:
            match = _re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, _re.DOTALL)
            if match:
                text = match.group(1).strip()

        try:
            items = json.loads(text)
        except json.JSONDecodeError:
            start = text.find("[")
            end = text.rfind("]")
            if start >= 0 and end > start:
                try:
                    items = json.loads(text[start:end + 1])
                except json.JSONDecodeError:
                    return
            else:
                return

        if isinstance(items, dict):
            items = [items]
        elif not isinstance(items, list):
            return

        # Build dedup set: descriptions from queue + recent completed tasks
        existing_descs = {t.get("description", "").strip().lower() for t in self.task_queue}
        # Check completed tasks (last 50) for similarity
        for ct in self.completed_tasks[-50:]:
            existing_descs.add(ct.get("description", "").strip().lower())

        injected = 0
        for item in items[:5]:  # Max 5 tasks per source
            if not isinstance(item, dict):
                continue
            desc = item.get("description", "").strip()
            if not desc or len(desc) < 5:
                continue

            # Handle restart requests from self-improve
            if item.get("type") == "restart" and self.evolution_engine:
                logger.info(f"🔄 Self-improve requested restart: {desc}")
                self.evolution_engine.request_restart(reason=f"self-improve: {desc}")
                self.running = False
                continue

            # Dedup: exact match OR high substring overlap with existing
            desc_lower = desc.lower()
            if desc_lower in existing_descs:
                continue
            # Fuzzy dedup: skip if >80% of words overlap with any existing
            desc_words = set(desc_lower.split())
            skip = False
            for existing in existing_descs:
                if not existing:
                    continue
                existing_words = set(existing.split())
                if desc_words and existing_words:
                    overlap = len(desc_words & existing_words) / max(len(desc_words), 1)
                    if overlap > 0.8:
                        skip = True
                        break
            if skip:
                continue

            task_id = f"{source}_{self.tick}_{injected}"

            self.task_queue.append({
                "id": task_id,
                "type": item.get("type", source),
                "description": desc,
                "priority": _parse_priority(item.get("priority", 5)),
                "created_at": datetime.now(),
                "source": source,
            })
            existing_descs.add(desc_lower)

            # Also register in GoalManager if it's a goal-type task
            if self.goal_manager and item.get("type") in ("goal", "self_improve"):
                try:
                    from .goal_system import GoalPriority
                    priority_map = {"high": GoalPriority.HIGH, "medium": GoalPriority.MEDIUM, "low": GoalPriority.LOW}
                    gp = priority_map.get(str(item.get("priority", "medium")).lower(), GoalPriority.MEDIUM)
                    import asyncio
                    asyncio.ensure_future(self.goal_manager.create_goal(
                        description=desc,
                        success_criteria=[f"Complete: {desc}"],
                        priority=gp,
                    ))
                except Exception as e:
                    logger.debug(f"Goal registration failed: {e}")

            injected += 1

        if injected:
            logger.info(f"📋 Injected {injected} task(s) from {source}")

    async def _check_trading(self):
        """Check for trading opportunities and portfolio health"""
        try:
            trading_skill = getattr(self.agent.tools, "trading_skill", None)
            if not trading_skill or not trading_skill._initialized:
                return

            # 1. Scan watchlist for signals
            signals_result = await trading_skill.get_signals()
            if signals_result and "No signals" not in signals_result:
                task = {
                    "id": f"trading_signals_{datetime.now().strftime('%H%M')}",
                    "type": "trading",
                    "description": f"Trading signals detected:\n{signals_result[:500]}",
                    "priority": 7,  # High priority
                    "created_at": datetime.now(),
                }
                if not any(t.get("type") == "trading" and "signals" in t.get("id", "") for t in self.task_queue):
                    self.task_queue.append(task)
                    logger.info("📊 Added trading signal task to queue")

            # 2. Check portfolio health (risk status)
            risk_result = await trading_skill.get_risk_status()
            if risk_result and "Emergency Halt: 🔴 YES" in risk_result:
                task = {
                    "id": f"trading_risk_alert_{datetime.now().strftime('%H%M')}",
                    "type": "trading_alert",
                    "description": "⚠️ TRADING EMERGENCY HALT ACTIVE,  review portfolio immediately",
                    "priority": 10,  # Maximum priority
                    "created_at": datetime.now(),
                }
                if not any(t.get("type") == "trading_alert" for t in self.task_queue):
                    self.task_queue.append(task)
                    logger.warning("🚨 Trading emergency halt detected!")

            logger.debug("Checked trading opportunities")

        except Exception as e:
            logger.error(f"Failed to check trading: {e}")

    async def _check_news(self):
        """Check world news for situational awareness and actionable events."""
        try:
            news_skill = getattr(self.agent.tools, "news_reader_skill", None)
            if not news_skill or not news_skill._ready:
                return

            digest = await news_skill.get_news_digest()
            if not digest or "(no news available)" in digest:
                logger.debug("News: no digest available")
                return

            # Use LLM to extract actionable items from the news digest
            if self.agent.llm:
                messages = [
                    {"role": "system", "content": (
                        "You are an autonomous agent's world-news analyst. "
                        "Review the news digest and extract items that are actionable, "
                        "noteworthy, or worth following up on. Think about: breaking events, "
                        "geopolitical shifts, market-moving news, tech breakthroughs, "
                        "security threats. Output ONLY valid JSON array. Each item: "
                        '{"description": "what happened and why it matters", "priority": 1-10, '
                        '"type": "news_followup"}. If nothing actionable, return [].'
                    )},
                    {"role": "user", "content": f"News digest:\n{digest[:3000]}"},
                ]
                try:
                    response = await self.agent.llm.invoke_with_tools(messages, [])
                    self._inject_llm_tasks(response.get("text", ""), source="news")
                except Exception as e:
                    logger.debug(f"News LLM analysis failed: {e}")
            else:
                logger.debug("Checked news (no LLM for analysis)")

        except Exception as e:
            logger.debug(f"News check skipped: {e}")

    async def _check_goals(self):
        """Check for pending goals (Agentic AI mode)"""
        if not self.goal_manager:
            return

        try:
            # Get active goals
            active_goals = [g for g in self.goal_manager.goals.values() if g.status in (GoalStatus.ACTIVE, GoalStatus.IN_PROGRESS)]

            # Add to task queue if not already there
            for goal in active_goals:
                task = {
                    "id": goal.goal_id,
                    "type": "goal",
                    "description": goal.description,
                    "priority": goal.priority.value,
                    "created_at": datetime.now(),
                }

                if not any(t["id"] == task["id"] for t in self.task_queue):
                    self.task_queue.append(task)
                    logger.info(f"Added goal to queue: {goal.description}")

        except Exception as e:
            logger.error(f"Failed to check goals: {e}")

    async def _prioritize_tasks(self):
        """Prioritize tasks in queue,  modulated by inner emotional state.

        Emotion influences:
          • High arousal → urgent/reactive tasks get boosted
          • Negative valence → defensive/maintenance tasks prioritized
          • Frustration → improvement/learning tasks boosted
          • Boredom → creative/proactive tasks boosted
          • Positive valence → ambitious goals get a nudge
        """
        if not self.task_queue:
            return

        # Get emotional modulation factors
        emotion_boost = {}  # type → priority delta
        if self.inner_life:
            try:
                e = self.inner_life.emotion
                arousal = getattr(e, "arousal", 0.3)
                valence = getattr(e, "valence", 0.0)
                primary = getattr(e, "primary", "neutral")

                # High arousal → boost urgent/system tasks
                if arousal > 0.6:
                    emotion_boost["system_maintenance"] = 2
                    emotion_boost["command"] = 1
                    emotion_boost["trading_alert"] = 2

                # Negative valence → boost defensive tasks
                if valence < -0.3:
                    emotion_boost["system_maintenance"] = emotion_boost.get("system_maintenance", 0) + 2
                    emotion_boost["email_action"] = 1

                # Frustration → boost self-improvement + arena as outlet
                if primary == "frustration":
                    emotion_boost["goal"] = 2
                    emotion_boost["self_improve"] = 3
                    emotion_boost["arena"] = 2

                # Boredom → boost creative/proactive (including arena fights)
                if primary == "boredom":
                    emotion_boost["proactive"] = 3
                    emotion_boost["creative"] = 2
                    emotion_boost["research"] = 2
                    emotion_boost["arena"] = 3

                # Positive valence → boost ambitious goals
                if valence > 0.3:
                    emotion_boost["goal"] = emotion_boost.get("goal", 0) + 1
                    emotion_boost["proactive"] = emotion_boost.get("proactive", 0) + 1

                if emotion_boost:
                    logger.debug(
                        f"Emotion modulation: {primary} (v={valence:+.1f}, a={arousal:.1f}) "
                        f"→ boosts: {emotion_boost}"
                    )
            except Exception as e:
                logger.debug(f"Emotion modulation skipped: {e}")

        # Connectome routing bias,  map module biases to task types
        connectome_boost = {}
        cb = getattr(self, "_connectome_biases", {})
        if cb:
            # decision bias → goal/proactive tasks
            if cb.get("decision", 0) > 0.5:
                connectome_boost["goal"] = connectome_boost.get("goal", 0) + int(cb["decision"] * 3)
                connectome_boost["proactive"] = connectome_boost.get("proactive", 0) + int(cb["decision"] * 2)
            # reflex bias → maintenance/urgent tasks
            if cb.get("reflex", 0) > 0.3:
                connectome_boost["system_maintenance"] = connectome_boost.get("system_maintenance", 0) + int(cb["reflex"] * 3)
                connectome_boost["trading_alert"] = connectome_boost.get("trading_alert", 0) + int(cb["reflex"] * 2)
            # memory bias → research/learning tasks
            if cb.get("memory", 0) > 0.5:
                connectome_boost["research"] = connectome_boost.get("research", 0) + int(cb["memory"] * 2)
                connectome_boost["self_improve"] = connectome_boost.get("self_improve", 0) + int(cb["memory"] * 2)
            # action bias → execute immediately
            if cb.get("action", 0) > 0.4:
                connectome_boost["command"] = connectome_boost.get("command", 0) + int(cb["action"] * 3)
            # emotion bias → creative tasks
            if cb.get("emotion", 0) > 0.5:
                connectome_boost["creative"] = connectome_boost.get("creative", 0) + int(cb["emotion"] * 2)
                connectome_boost["proactive"] = connectome_boost.get("proactive", 0) + 1
            if connectome_boost:
                logger.debug(f"Connectome priority modulation: {connectome_boost}")

        # Apply emotional + connectome modulation to priority scores
        for task in self.task_queue:
            base = task.get("priority", 5)
            boost = emotion_boost.get(task.get("type", ""), 0)
            c_boost = connectome_boost.get(task.get("type", ""), 0)
            task["_effective_priority"] = base + boost + c_boost

        # Sort by effective priority (higher first)
        self.task_queue.sort(key=lambda t: t.get("_effective_priority", t.get("priority", 0)), reverse=True)

    async def _execute_tasks(self):
        """Execute tasks from queue,  ONE AT A TIME (sequential).
        
        After each task, record the outcome for learning.
        Concurrent task execution causes multiple simultaneous API calls,
        which gets detected as bot behavior. A real user does one thing at a time.
        """
        tasks_to_execute = self.task_queue[: self.max_concurrent_tasks]

        if not tasks_to_execute:
            return

        logger.info(f"Executing {len(tasks_to_execute)} task(s) sequentially...")

        for task in list(tasks_to_execute):
            start_time = time.monotonic()
            try:
                result = await self._execute_single_task(task)
                duration_ms = (time.monotonic() - start_time) * 1000

                # ── Success: record outcome ──
                self.task_queue.remove(task)
                task["completed_at"] = datetime.now()
                task["result"] = str(result)[:500] if result else ""
                task["status"] = "done"
                task["duration_ms"] = duration_ms
                task["retries"] = task.get("retries", 0)
                self.completed_tasks.append(task)

                logger.info(
                    f"✅ Task {task['id']} done ({duration_ms:.0f}ms): "
                    f"{task.get('description', '')[:60]}"
                )

                # Store outcome in cognitive memory for future reasoning
                self._record_outcome(task, success=True, result=result)

            except Exception as e:
                duration_ms = (time.monotonic() - start_time) * 1000
                retries = task.get("retries", 0)
                max_retries = task.get("max_retries", 2)

                if retries < max_retries:
                    # ── Retry: re-enqueue with incremented retry count ──
                    task["retries"] = retries + 1
                    task["last_error"] = str(e)[:200]
                    # Move to end of queue for later retry
                    self.task_queue.remove(task)
                    self.task_queue.append(task)
                    logger.warning(
                        f"🔄 Task {task['id']} retry {retries + 1}/{max_retries}: {e}"
                    )
                else:
                    # ── Final failure: record for learning ──
                    self.task_queue.remove(task)
                    task["completed_at"] = datetime.now()
                    task["result"] = str(e)[:500]
                    task["status"] = "error"
                    task["duration_ms"] = duration_ms
                    task["retries"] = retries
                    self.completed_tasks.append(task)

                    logger.error(
                        f"❌ Task {task['id']} failed after {retries + 1} attempts ({duration_ms:.0f}ms): {e}"
                    )

                    self._record_outcome(task, success=False, result=e)

    def _record_outcome(self, task: Dict, success: bool, result: Any):
        """Record task outcome for learning,  feeds cognitive memory + proactive reasoning."""
        description = task.get("description", "")[:200]
        task_type = task.get("type", "unknown")
        duration_ms = task.get("duration_ms", 0)

        # 1. Store in cognitive memory (if available)
        #    Use emotion modulation: high arousal → memories stored with higher importance
        if self.cognitive_memory:
            try:
                importance = 0.7 if success else 0.9  # Failures are more memorable

                # Apply inner life emotion modulation (amygdala effect)
                if self.inner_life:
                    try:
                        modulation = self.inner_life.get_emotion_modulation()
                        importance = min(1.0, importance * modulation)
                    except Exception:
                        pass

                category = "success" if success else "failure"
                self.cognitive_memory.add_memory(
                    f"Task [{task_type}] {'succeeded' if success else 'FAILED'}: {description}. "
                    f"Result: {str(result)[:200]}",
                    category=category,
                    importance=importance,
                )
            except Exception:
                pass

        # 2. Record in skill fitness tracker
        if self.skill_fitness:
            try:
                # Track tools used in proactive/react tasks
                tools_used = task.get("tools_used", [])
                for tool_name in tools_used:
                    self.skill_fitness.record_event(
                        skill_name=tool_name,
                        tick=self.tick,
                        success=success,
                        duration_ms=duration_ms,
                    )
            except Exception:
                pass

        # 3. Log to trace exporter
        if self.trace_exporter:
            try:
                self.trace_exporter.record_event(
                    "task_outcome",
                    summary=f"{'✅' if success else '❌'} [{task_type}] {description[:60]}",
                    tick=self.tick,
                    data={
                        "task_id": task.get("id"),
                        "type": task_type,
                        "success": success,
                        "duration_ms": duration_ms,
                        "source": task.get("source", ""),
                    },
                )
            except Exception:
                pass

    async def _execute_single_task(self, task: Dict) -> Any:
        """Execute a single task using the best available method."""
        task_type = task.get("type")

        if task_type == "goal":
            # Execute goal,  try ReAct first for multi-step reasoning,
            # fall back to GoalManager direct execution
            if self.react_executor and self.agent.llm:
                description = task.get("description", "")
                result = await self._execute_via_react(
                    f"Complete this goal: {description}"
                )
                if result:
                    # Update goal status if GoalManager is available
                    if self.goal_manager:
                        try:
                            goal_id = task["id"]
                            if goal_id in self.goal_manager.goals:
                                self.goal_manager.goals[goal_id].status = GoalStatus.COMPLETED
                        except Exception:
                            pass
                    return result
            # Fallback to direct goal execution
            if self.goal_manager:
                goal_id = task["id"]
                return await self.goal_manager.execute_goal(goal_id)

        elif task_type == "command":
            command = task.get("command")
            return await self.agent.tools.execute("execute_command", {"command": command})

        elif task_type == "reminder":
            message = task.get("message", task.get("description", ""))
            logger.info(f"⏰ Reminder: {message}")
            return message

        elif task_type == "proactive":
            return await self._execute_proactive_task(task)

        elif task_type == "system_maintenance":
            # System maintenance tasks are executed via ReAct
            description = task.get("description", "")
            return await self._execute_via_react(
                f"System maintenance: {description}. "
                "Use execute_command to diagnose and fix the issue. "
                "Be careful and non-destructive."
            )

        elif task_type in ("email_action", "calendar"):
            # LLM-discovered tasks from email/calendar
            description = task.get("description", "")
            return await self._execute_via_react(description)

        elif task_type == "trading":
            # Trading tasks,  execute via trading skill
            description = task.get("description", "")
            if self.react_executor and self.agent.llm:
                return await self._execute_via_react(
                    f"Trading task: {description}. Use trading tools to handle this."
                )
            return description

        elif task_type == "trading_alert":
            description = task.get("description", "")
            logger.warning(f"🚨 {description}")
            return description

        else:
            # Unknown type,  always try ReAct
            if self.react_executor and self.agent.llm:
                return await self._execute_via_react(task.get("description", str(task)))
            logger.warning(f"Unknown task type: {task_type} (no ReAct available)")
            return None

    async def _self_improve(self):
        """Run LLM-driven meta-learning: analyse past outcomes + inject improvements.

        Runs every 24 hours.  The LLM receives:
          • Recent completed tasks (successes + failures)
          • Skill fitness summary
          • Self-reflection summary
          • Pattern learner rules

        It then proposes concrete improvements as tasks injected into the queue.
        """
        try:
            if not hasattr(self, "_last_improvement"):
                self._last_improvement = datetime.now() - timedelta(hours=25)

            if datetime.now() - self._last_improvement < timedelta(hours=24):
                return

            if not self.agent.llm:
                return

            logger.info("🧬 Running self-improvement analysis...")

            # ── Gather evidence ──
            recent_tasks = self.completed_tasks[-30:]
            successes = [t for t in recent_tasks if t.get("status") == "done"]
            failures = [t for t in recent_tasks if t.get("status") == "error"]

            summaries = []
            summaries.append(f"Tasks completed: {len(successes)}, failed: {len(failures)}")

            for t in failures[-5:]:
                summaries.append(
                    f"FAIL: [{t.get('type','')}] {t.get('description','')[:80]} → {t.get('result','')[:120]}"
                )
            for t in successes[-5:]:
                summaries.append(
                    f"OK: [{t.get('type','')}] {t.get('description','')[:80]} ({t.get('duration_ms',0):.0f}ms)"
                )

            if self.self_reflection:
                try:
                    summaries.append(f"Self-reflection: {self.self_reflection.get_summary()}")
                except Exception:
                    pass

            if self.pattern_learner and hasattr(self.pattern_learner, "rules"):
                try:
                    rules = self.pattern_learner.rules[:5]
                    for r in rules:
                        summaries.append(f"Pattern rule: {r}")
                except Exception:
                    pass

            if self.skill_fitness:
                try:
                    fitness = self.skill_fitness.get_fitness_dicts(self.tick, 200)
                    low_fitness = [f for f in fitness if f.get("fitness", 1.0) < 0.5]
                    for lf in low_fitness[:5]:
                        summaries.append(
                            f"Low-fitness skill: {lf.get('skill','')} fitness={lf.get('fitness',0):.2f}"
                        )
                except Exception:
                    pass

            # Emotion-driven evolution pressure
            if self.inner_life:
                try:
                    pressures = self.inner_life.get_evolution_pressure()
                    for p in pressures:
                        summaries.append(f"Evolution pressure: {p}")
                    if pressures:
                        summaries.append(
                            f"Current emotion: {self.inner_life.emotion.primary} "
                            f"(valence={self.inner_life.emotion.valence:+.1f}, "
                            f"arousal={self.inner_life.emotion.arousal:.1f})"
                        )
                except Exception:
                    pass

            # Evolution engine stats
            if self.evolution_engine:
                try:
                    evo_stats = self.evolution_engine.get_stats()
                    summaries.append(
                        f"Evolution engine: {evo_stats.get('total_mutations', 0)} mutations total "
                        f"({evo_stats.get('successful', 0)} ok, "
                        f"{evo_stats.get('failed', 0)} failed), "
                        f"can_restart={evo_stats.get('can_restart', False)}"
                    )
                except Exception:
                    pass

            evidence = "\n".join(summaries)

            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are an autonomous agent performing self-improvement analysis.\n"
                        "Given evidence about your recent performance, propose 1-3 concrete "
                        "improvement actions the agent can take RIGHT NOW.\n"
                        "For each action, output JSON:\n"
                        '[{"type":"goal","description":"...","priority":"high|medium|low"}]\n'
                        "Special action types:\n"
                        '  {"type":"restart","description":"reason..."},  request agent restart\n'
                        "Focus on: fixing recurring errors, improving slow tasks, learning new skills, "
                        "automating manual patterns. Be specific and actionable."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Performance evidence (last 24h):\n{evidence}\n\nPropose improvements:",
                },
            ]

            result = await self.agent.llm.invoke_with_tools(messages, [])
            content = result.get("text", "") or result.get("content", "") if isinstance(result, dict) else str(result)

            if content:
                self._inject_llm_tasks(content, source="self_improve")

            self._last_improvement = datetime.now()
            logger.info("🧬 Self-improvement analysis complete")

        except Exception as e:
            logger.error(f"Self-improvement failed: {e}")

    async def _proactive_tick(self):
        """Run proactive reasoning,  LLM decides what to do autonomously.

        Only triggers every N ticks (configurable). Generates action proposals
        and injects them into the task queue for execution.
        """
        if not self.proactive_engine:
            return
        if not self.proactive_engine.should_think(self.tick):
            return
        if not self.agent.llm:
            return

        try:
            # Build world context
            system_state = {}
            try:
                system_state = self.get_status()
            except Exception:
                pass

            cognitive_state = {}
            if self.inner_life:
                try:
                    e = self.inner_life.emotion
                    cognitive_state["emotion"] = {
                        "primary": e.primary,
                        "valence": e.valence,
                        "arousal": e.arousal,
                        "trigger": e.trigger,
                    }
                    pressures = self.inner_life.get_evolution_pressure()
                    if pressures:
                        cognitive_state["evolution_pressures"] = pressures
                except Exception:
                    pass
            if self.self_reflection:
                try:
                    cognitive_state["reflection"] = self.self_reflection.get_summary()
                except Exception:
                    pass

            goals = []
            if self.goal_manager:
                try:
                    for g in self.goal_manager.goals.values():
                        if g.status in (GoalStatus.ACTIVE, GoalStatus.IN_PROGRESS):
                            goals.append(g.description)
                except Exception:
                    pass

            recent_errors = []
            for t in self.completed_tasks[-10:]:
                if t.get("status") == "error":
                    recent_errors.append(str(t.get("result", ""))[:100])

            context = self.proactive_engine.build_context(
                tick=self.tick,
                completed_tasks=self.completed_tasks,
                queued_tasks=self.task_queue,
                system_state=system_state,
                recent_errors=recent_errors,
                goals=goals,
                cognitive_state=cognitive_state,
            )

            # Inject business automation context (CRM + pipeline follow-ups)
            try:
                if hasattr(self.agent, 'tools') and hasattr(self.agent.tools, 'followup_skill'):
                    fu = self.agent.tools.followup_skill
                    if fu and fu.is_ready():
                        biz_summary = await fu.get_business_summary()
                        if biz_summary and biz_summary != "No business data yet.":
                            context += f"\n\nBusiness Automation State:\n{biz_summary}"
            except Exception:
                pass

            # Ask LLM to think proactively
            proposals = await self.proactive_engine.think(
                llm=self.agent.llm,
                tick=self.tick,
                context=context,
            )

            # Inject proposals into task queue
            for proposal in proposals:
                task = proposal.to_task(self.tick)
                self.task_queue.append(task)
                self.proactive_engine.record_accepted(proposal)
                logger.info(
                    f"🧠 Proactive: {proposal.action[:60]} "
                    f"(type={proposal.goal_type.value}, pri={proposal.priority:.1f})"
                )

            if self.trace_exporter and proposals:
                self.trace_exporter.record_event(
                    "proactive_proposals",
                    summary=f"{len(proposals)} proposals generated",
                    tick=self.tick,
                    data={"actions": [p.action[:80] for p in proposals]},
                )

        except Exception as e:
            logger.warning(f"Proactive reasoning tick failed: {e}")

    async def _execute_proactive_task(self, task: Dict) -> Any:
        """Execute a proactive task, using ReAct if available."""
        description = task.get("description", "")
        tool_name = task.get("tool_name")
        tool_args = task.get("tool_args", {})

        # If a specific tool was proposed, try direct execution first
        if tool_name and tool_args:
            try:
                result = await self.agent.tools.execute(tool_name, tool_args)
                return result
            except Exception as e:
                logger.debug(f"Direct tool execution failed, falling back to ReAct: {e}")

        # Fall back to ReAct for complex tasks
        return await self._execute_via_react(description)

    async def _execute_via_react(self, task_description: str) -> Any:
        """Execute a task using the ReAct reasoning + acting loop."""
        if not self.react_executor or not self.agent.llm:
            logger.debug(f"ReAct not available for task: {task_description[:60]}")
            return None

        async def tool_executor(tool_name: str, args: Dict[str, Any]) -> str:
            """Bridge between ReAct and the ToolRegistry."""
            try:
                result = await self.agent.tools.execute(tool_name, args)
                return str(result)[:2000]
            except Exception as e:
                return f"Error: {e}"

        # Get available tool schemas for the LLM
        available_tools = []
        try:
            available_tools = self.agent.tools.get_tool_schemas()[:30]
        except Exception:
            pass

        result = await self.react_executor.execute(
            task=task_description,
            llm=self.agent.llm,
            tool_executor=tool_executor,
            available_tools=available_tools,
        )

        if self.trace_exporter:
            self.trace_exporter.record_event(
                "react_execution",
                summary=f"{'✅' if result.success else '❌'} {task_description[:60]}",
                tick=self.tick,
                data={
                    "success": result.success,
                    "steps": len(result.steps),
                    "tools_used": result.tools_used,
                    "duration_ms": result.total_duration_ms,
                },
            )

        if result.success:
            return result.final_answer
        else:
            logger.warning(f"ReAct execution failed: {result.error}")
            return None

    async def _cognitive_tick(self):
        """Run all cognitive modules for the current tick.

        Order:
          0.  Connectome,  propagate signals through FlyWire-derived neural colony
          1.  Cognitive memory,  decay + consolidation + attention filter
          2.  Self-reflection,  pattern detection + stagnation check (with real data)
          3.  Skill evolution,  natural selection + mutation + niche
          4.  Pattern learner,  windowed analysis + snapshots + rules
          5.  Git brain,  write episode + optional auto-commit
          6.  Inner life,  System 1 LLM pass (emotion, impulse, fantasy, landscape)
          7.  Connectome Hebbian learning,  update wiring from task outcomes
          8.  Deep planner,  execute ready plan steps + re-plan on failure
          9.  Inter-agent bridge,  export learnings + import sibling knowledge
          10. Ultra-LTM,  consolidate weeks of memories into durable patterns
          11. Self-benchmark,  quantified internal performance assessment
        """
        try:
            # ── 0. Connectome,  bio-inspired signal routing ──
            connectome = getattr(self.agent, "connectome", None)
            connectome_biases = {}
            if connectome:
                try:
                    # Stimulate sensory regions based on current state
                    has_tasks = len(self.task_queue) > 0
                    recent = self.completed_tasks[-5:]
                    error_rate = (
                        sum(1 for t in recent if t.get("status") == "error")
                        / max(len(recent), 1)
                    )
                    success_rate = 1.0 - error_rate

                    # AL (sensory): high if new tasks incoming
                    connectome.stimulate("AL", 0.8 if has_tasks else 0.2)
                    # OL (context): always moderate,  agent always observes
                    connectome.stimulate("OL", 0.5)
                    # PI (motivation): high if goals active, low if idle
                    drive = 0.3
                    if self.goal_manager:
                        from .goal_system import GoalStatus
                        active = sum(
                            1 for g in self.goal_manager.goals.values()
                            if g.status in (GoalStatus.ACTIVE, GoalStatus.IN_PROGRESS)
                        )
                        drive = min(1.0, 0.3 + active * 0.15)
                    connectome.stimulate("PI", drive)
                    # LPC (emotion): feed from inner life if available
                    inner_life = getattr(self.agent, "inner_life", None)
                    if inner_life and hasattr(inner_life, "emotion"):
                        arousal = getattr(inner_life.emotion, "arousal", 0.5)
                        connectome.stimulate("LPC", arousal)
                    # LH (reflex): high if errors,  instinct to fix
                    connectome.stimulate("LH", error_rate * 0.8)

                    # Propagate through the connectome
                    results = connectome.propagate(max_cycles=3)
                    connectome_biases = connectome.compute_routing_bias(results)
                    self._connectome_biases = connectome_biases

                    if results:
                        fired_modules = connectome.get_firing_modules(results)
                        if fired_modules:
                            logger.debug(
                                f"🧠 Connectome fired: "
                                + ", ".join(
                                    f"{m}={s:.2f}" for m, s in
                                    sorted(fired_modules.items(), key=lambda x: -x[1])[:4]
                                )
                            )
                except Exception as e:
                    logger.debug(f"Connectome tick failed: {e}")

            # ── 1. Cognitive memory ──
            if self.cognitive_memory:
                try:
                    self.cognitive_memory.process_tick(self.tick)
                except Exception as e:
                    logger.debug(f"Cognitive memory tick failed: {e}")

            # ── 2. Self-reflection,  feed REAL outcome data ──
            if self.self_reflection:
                try:
                    recent = self.completed_tasks[-10:]
                    success_count = sum(1 for t in recent if t.get("status") == "done")
                    error_count = sum(1 for t in recent if t.get("status") == "error")
                    errors_list = [
                        str(t.get("result", ""))[:100]
                        for t in recent if t.get("status") == "error"
                    ]
                    tools = []
                    for t in recent:
                        if t.get("tools_used"):
                            tools.extend(t["tools_used"])
                    goals = []
                    if self.goal_manager:
                        for g in self.goal_manager.goals.values():
                            if g.status in (GoalStatus.ACTIVE, GoalStatus.IN_PROGRESS):
                                goals.append(g.description[:60])

                    from .self_reflection import TickOutcome
                    outcome = TickOutcome(
                        tick=self.tick,
                        success=error_count == 0,
                        summary=(
                            f"Tick {self.tick}: {success_count} ok, "
                            f"{error_count} errors, {len(self.task_queue)} queued"
                        ),
                        tools_used=tools[:20],
                        errors=errors_list[:5],
                        goals_progressed=goals[:5],
                    )
                    self.self_reflection.record_outcome(outcome)
                except Exception as e:
                    logger.debug(f"Self-reflection tick failed: {e}")

            # ── 3. Skill evolution + autonomous mutation ──
            if self.skill_evolution:
                try:
                    evolution_result = self.skill_evolution.evaluate_tick(self.tick)
                    if evolution_result.get("condemned"):
                        logger.info(
                            f"🧬 Evolution condemned {len(evolution_result['condemned'])} skills"
                        )

                    # Autonomous code mutation,  if evolution engine and LLM available
                    has_targets = (
                        evolution_result.get("condemned")
                        or evolution_result.get("error_driven")
                        or evolution_result.get("stagnant")
                    )
                    if has_targets and self.evolution_engine and self.agent.llm:
                        try:
                            # Gather recent error samples per skill
                            error_samples: Dict[str, List[str]] = {}
                            for t in self.completed_tasks[-50:]:
                                if t.get("status") == "error":
                                    skill = t.get("skill", t.get("type", ""))
                                    err = str(t.get("result", ""))[:200]
                                    if skill:
                                        error_samples.setdefault(skill, []).append(err)

                            mutation_result = await self.evolution_engine.evaluate_and_mutate(
                                tick=self.tick,
                                evolution_result=evolution_result,
                                llm=self.agent.llm,
                                error_samples=error_samples,
                            )

                            action = mutation_result.get("action", "none")
                            if action == "mutated":
                                sc = mutation_result.get("success_count", 0)
                                fc = mutation_result.get("fail_count", 0)
                                logger.info(
                                    f"🧬 Evolution engine: {sc} mutations succeeded, "
                                    f"{fc} failed"
                                )
                                if mutation_result.get("restart_initiated"):
                                    logger.warning(
                                        "🔄 Self-restart initiated by evolution engine"
                                    )
                                    self.running = False  # graceful stop

                                # Record evolution events
                                for m in mutation_result.get("mutations", []):
                                    if m.get("success"):
                                        self.skill_evolution.record_event(
                                            "cap_evolved",
                                            m["skill_name"],
                                            tick=self.tick,
                                            details=m.get("changes_summary", ""),
                                        )
                            elif action in ("cooldown", "backoff"):
                                pass  # normal, silently skip
                        except Exception as e:
                            logger.debug(f"Evolution engine mutation failed: {e}")

                except Exception as e:
                    logger.debug(f"Skill evolution tick failed: {e}")

            # ── 4. Pattern learner + fitness snapshots ──
            if self.pattern_learner:
                try:
                    fitness_dicts = []
                    if self.skill_fitness:
                        fitness_dicts = self.skill_fitness.get_fitness_dicts(
                            current_tick=self.tick, window_ticks=50,
                        )
                    events = []
                    if self.skill_fitness:
                        events = self.skill_fitness.events
                    self.pattern_learner.process_tick(
                        tick=self.tick,
                        events=events,
                        fitness_records=fitness_dicts,
                    )
                except Exception as e:
                    logger.debug(f"Pattern learner tick failed: {e}")

            # ── 5. Git brain,  write episode ──
            if self.git_brain:
                try:
                    recent = self.completed_tasks[-5:]
                    episode_parts = []
                    for t in recent:
                        status = "✅" if t.get("status") == "done" else "❌"
                        episode_parts.append(
                            f"{status} [{t.get('type','')}] {t.get('description','')[:60]}"
                        )
                    summary = "; ".join(episode_parts) if episode_parts else "idle tick"
                    await self.git_brain.write_episode(self.tick, summary=summary)
                except Exception as e:
                    logger.debug(f"Git brain tick failed: {e}")

            # ── 6. Inner life,  System 1 LLM pass ──
            if self.inner_life and self.agent.llm:
                try:
                    # Build context from recent activity
                    recent_ctx_parts = []
                    for t in self.completed_tasks[-3:]:
                        status = "ok" if t.get("status") == "done" else "fail"
                        recent_ctx_parts.append(
                            f"{status}: {t.get('description','')[:60]}"
                        )
                    recent_context = "; ".join(recent_ctx_parts) if recent_ctx_parts else "idle"

                    active_goal = ""
                    if self.goal_manager:
                        for g in self.goal_manager.goals.values():
                            if g.status in (GoalStatus.ACTIVE, GoalStatus.IN_PROGRESS):
                                active_goal = g.description[:100]
                                break

                    # Generate System 1 prompt and send to LLM
                    from .inner_life import SYSTEM1_SYSTEM_PROMPT
                    user_prompt = self.inner_life.get_system1_prompt(
                        active_goal=active_goal,
                        context=recent_context,
                    )
                    messages = [
                        {"role": "system", "content": SYSTEM1_SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ]
                    result = await self.agent.llm.invoke_with_tools(messages, [])
                    content = result.get("text", "") or result.get("content", "") if isinstance(result, dict) else str(result)

                    if content:
                        old_emotion = self.inner_life.emotion.primary
                        self.inner_life.process_response(content, tick=self.tick)
                        new_emotion = self.inner_life.emotion
                        if new_emotion.primary != old_emotion:
                            logger.info(
                                f"🧠 Inner life: {old_emotion} → {new_emotion.primary} "
                                f"(v={new_emotion.valence:+.1f}, a={new_emotion.arousal:.1f})"
                            )
                        else:
                            logger.debug(
                                f"Inner life: {new_emotion.primary} "
                                f"(v={new_emotion.valence:+.1f}, a={new_emotion.arousal:.1f})"
                            )
                except Exception as e:
                    logger.warning(f"Inner life tick failed: {e}")

            # ── 7. Connectome Hebbian learning ──
            if connectome and self.tick % 5 == 0:
                try:
                    recent = self.completed_tasks[-20:]
                    if len(recent) >= 3:
                        success_count = sum(1 for t in recent if t.get("status") == "done")
                        error_count = sum(1 for t in recent if t.get("status") == "error")
                        total = max(len(recent), 1)

                        # Build performance scores for each module
                        performance = {
                            # Memory: good if tasks are completing (learned patterns)
                            "memory": (success_count / total) * 0.5,
                            # Decision: good if more successes than errors
                            "decision": (success_count - error_count) / total,
                            # Action: good if tasks execute at all
                            "action": 0.3 if total > 0 else -0.2,
                            # Reflex: good if errors are low (fast responses work)
                            "reflex": 0.4 if error_count == 0 else -0.3,
                            # Emotion: neutral,  doesn't directly affect performance
                            "emotion": 0.0,
                            # Motivation: good if queue isn't stagnating
                            "motivation": 0.2 if len(self.task_queue) < 20 else -0.2,
                            # Sensory: always slightly positive (input is needed)
                            "intent_classifier": 0.1,
                            "context_processor": 0.1,
                        }

                        # Inner life emotional valence modulates everything
                        inner = getattr(self.agent, "inner_life", None)
                        if inner and hasattr(inner, "emotion"):
                            valence = getattr(inner.emotion, "valence", 0.0)
                            performance["emotion"] = valence * 0.3

                        connectome.apply_evolution_pressure(
                            performance, learning_rate=0.02
                        )
                        logger.debug(
                            f"🧬 Connectome Hebbian update: gen {connectome._generation}"
                        )
                except Exception as e:
                    logger.debug(f"Connectome learning failed: {e}")

            # ── 8. Deep planner,  execute ready plan steps ──
            if self.deep_planner:
                try:
                    ready_steps = self.deep_planner.get_next_steps(max_total=2)
                    for plan_id, step in ready_steps:
                        self.deep_planner.mark_step_running(plan_id, step.step_id)
                        try:
                            start_t = time.monotonic()
                            result = await self._execute_via_react(step.description)
                            dur = (time.monotonic() - start_t) * 1000
                            self.deep_planner.mark_step_done(
                                plan_id, step.step_id,
                                result=str(result)[:500] if result else "",
                                duration_ms=dur,
                            )
                        except Exception as step_err:
                            self.deep_planner.mark_step_failed(
                                plan_id, step.step_id, error=str(step_err)[:300]
                            )
                            # Check if plan needs replanning
                            plan = self.deep_planner.get_plan(plan_id)
                            if plan and plan.has_failed_blocking() and plan.replan_count < 3:
                                await self.deep_planner.replan(plan_id, self.agent.llm)

                    # Create plans for complex goal tasks in queue
                    if self.agent.llm and self.tick % 10 == 0:
                        for task in self.task_queue:
                            if (
                                task.get("type") == "goal"
                                and not task.get("_has_plan")
                                and len(task.get("description", "")) > 50
                            ):
                                plan = await self.deep_planner.create_plan(
                                    goal=task["description"],
                                    llm=self.agent.llm,
                                    context=f"Tick {self.tick}, queue={len(self.task_queue)}",
                                )
                                if plan:
                                    task["_has_plan"] = True
                                    task["_plan_id"] = plan.plan_id
                                break  # One plan per tick to avoid LLM overload
                except Exception as e:
                    logger.debug(f"Deep planner tick failed: {e}")

            # ── 9. Inter-agent bridge,  sync learnings ──
            if self.inter_agent_bridge:
                try:
                    # Build activity summary for export
                    recent_summary_parts = []
                    for t in self.completed_tasks[-10:]:
                        status = "OK" if t.get("status") == "done" else "FAIL"
                        recent_summary_parts.append(
                            f"[{status}] {t.get('type', '')}: {t.get('description', '')[:100]}"
                        )
                    activity = "\n".join(recent_summary_parts) if recent_summary_parts else ""

                    # Build context for import relevance
                    context_parts = []
                    if self.goal_manager:
                        for g in self.goal_manager.goals.values():
                            if g.status in (GoalStatus.ACTIVE, GoalStatus.IN_PROGRESS):
                                context_parts.append(f"Goal: {g.description[:100]}")
                    context_parts.append(f"Tick: {self.tick}, Queue: {len(self.task_queue)}")
                    context = "\n".join(context_parts)

                    imported = await self.inter_agent_bridge.sync(
                        llm=self.agent.llm,
                        tick=self.tick,
                        recent_activity=activity,
                        current_context=context,
                    )

                    # Inject imported learnings as task hints in cognitive memory
                    if imported and self.cognitive_memory:
                        for l in imported:
                            self.cognitive_memory.add_memory(
                                f"[IMPORTED from {l.source_agent}] {l.title}: {l.content[:200]}",
                                category="inter_agent",
                                importance=l.confidence * 0.8,
                            )
                except Exception as e:
                    logger.debug(f"Inter-agent bridge tick failed: {e}")

            # ── 10. Ultra-LTM,  consolidate long-term patterns ──
            if self.ultra_ltm and self.agent.llm:
                try:
                    # Gather raw memories for consolidation
                    raw_memories = []
                    for t in self.completed_tasks:
                        raw_memories.append(
                            f"[{t.get('status','?')}] {t.get('type', '')}: "
                            f"{t.get('description', '')[:150]} → {t.get('result', '')[:100]}"
                        )
                    # Add cognitive memories if available
                    if self.cognitive_memory and hasattr(self.cognitive_memory, "get_recent"):
                        try:
                            cog_mems = self.cognitive_memory.get_recent(limit=50)
                            for cm in cog_mems:
                                if isinstance(cm, dict):
                                    raw_memories.append(cm.get("text", str(cm))[:200])
                                else:
                                    raw_memories.append(str(cm)[:200])
                        except Exception:
                            pass

                    new_patterns = await self.ultra_ltm.consolidate(
                        llm=self.agent.llm,
                        raw_memories=raw_memories,
                        tick=self.tick,
                    )

                    # Generate wisdom summary periodically
                    if new_patterns > 0 or self.tick % 100 == 0:
                        await self.ultra_ltm.generate_wisdom_summary(self.agent.llm)
                except Exception as e:
                    logger.debug(f"Ultra-LTM tick failed: {e}")

            # ── 11. Self-benchmark,  quantified assessment ──
            if self.self_benchmark:
                try:
                    agent_state = {
                        "completed_tasks": self.completed_tasks,
                        "task_queue": self.task_queue,
                        "inner_life": self.inner_life,
                        "cognitive_memory_count": (
                            getattr(self.cognitive_memory, "count", 0)
                            if self.cognitive_memory else 0
                        ),
                        "deep_planner": self.deep_planner,
                        "ultra_ltm": self.ultra_ltm,
                        "inter_agent_bridge": self.inter_agent_bridge,
                        "connectome": connectome,
                        "tick": self.tick,
                    }
                    snapshot = await self.self_benchmark.run_benchmarks(
                        tick=self.tick,
                        agent_state=agent_state,
                    )
                    if snapshot and self.trace_exporter:
                        self.trace_exporter.record_event(
                            "self_benchmark",
                            summary=(
                                f"Autonomy={snapshot.autonomy_score}/100 "
                                f"regressions={snapshot.regressions}"
                            ),
                            tick=self.tick,
                            data=snapshot.results,
                        )
                except Exception as e:
                    logger.debug(f"Self-benchmark tick failed: {e}")

            # ── 12. Meta-learner,  adapt cognitive hyperparameters ──
            if self.meta_learner and self.tick % 15 == 0:
                try:
                    perf = 0.5
                    if self.self_benchmark:
                        st = self.self_benchmark.get_stats()
                        perf = st.get("latest_autonomy_score", 50) / 100.0
                    self.meta_learner.evaluate_and_adapt(perf)
                except Exception as e:
                    logger.debug(f"Meta-learner tick failed: {e}")

            # ── 13. Causal engine,  extract causal links ──
            if self.causal_engine and self.tick % 20 == 0:
                try:
                    recent = self.completed_tasks[-5:] if self.completed_tasks else []
                    for t in recent:
                        desc = t.get("description", "")
                        result = t.get("result", "")
                        if desc and result:
                            await self.causal_engine.extract_causes(
                                self.agent.llm, desc, result
                            )
                except Exception as e:
                    logger.debug(f"Causal engine tick failed: {e}")

            # ── 14. Goal synthesis,  generate strategic goals ──
            if self.goal_synthesis and self.tick % 50 == 0:
                try:
                    context = {}
                    if self.self_benchmark:
                        context["benchmarks"] = self.self_benchmark.get_stats()
                    if self.ultra_ltm and hasattr(self.ultra_ltm, "wisdom_summary"):
                        context["wisdom"] = self.ultra_ltm.wisdom_summary or ""
                    proposed = await self.goal_synthesis.synthesize(
                        self.agent.llm, context
                    )
                    # Auto-accept high-priority goals
                    for g in proposed:
                        if g.priority >= 8:
                            self.goal_synthesis.accept_goal(g.id)
                            self.task_queue.append({
                                "description": f"[Strategic Goal] {g.description}",
                                "priority": g.priority,
                                "source": "goal_synthesis",
                            })
                except Exception as e:
                    logger.debug(f"Goal synthesis tick failed: {e}")

            # ── 15. Skill composer,  discover compound skills ──
            if self.skill_composer and self.tick % 30 == 0:
                try:
                    recent = self.completed_tasks[-10:] if self.completed_tasks else []
                    for t in recent:
                        skill_name = t.get("skill", t.get("description", "")[:50])
                        self.skill_composer.record_execution(skill_name)
                    await self.skill_composer.analyze_and_compose(self.agent.llm)
                except Exception as e:
                    logger.debug(f"Skill composer tick failed: {e}")

            # ── 16. World predictor,  forecast and prepare ──
            if self.world_predictor and self.tick % 25 == 0:
                try:
                    # Observe from recent tasks
                    recent = self.completed_tasks[-5:] if self.completed_tasks else []
                    for t in recent:
                        key = t.get("skill", "general")
                        val = t.get("result", "completed")[:200]
                        self.world_predictor.observe(key, val)
                    preds = await self.world_predictor.predict(self.agent.llm)
                    # Inject preparation tasks for high-confidence predictions
                    for p in preds:
                        if p.confidence >= 0.7 and p.preparation_tasks:
                            for prep in p.preparation_tasks[:2]:
                                self.task_queue.append({
                                    "description": f"[Prep] {prep}",
                                    "priority": 5,
                                    "source": "world_predictor",
                                })
                except Exception as e:
                    logger.debug(f"World predictor tick failed: {e}")

            # ── 17. Cognitive optimizer,  tune tick intervals ──
            if self.cognitive_optimizer and self.tick % 20 == 0:
                try:
                    self.cognitive_optimizer.optimize()
                except Exception as e:
                    logger.debug(f"Cognitive optimizer tick failed: {e}")

            # ── 18. Adversarial tester,  red-team self-testing ──
            if self.adversarial_tester and self.tick % 40 == 0:
                try:
                    bench_scores = {}
                    if self.self_benchmark:
                        st = self.self_benchmark.get_stats()
                        bench_scores = st.get("latest_results", {})
                    await self.adversarial_tester.generate_tests(
                        self.agent.llm, bench_scores
                    )
                except Exception as e:
                    logger.debug(f"Adversarial tester tick failed: {e}")

            # ── 19. Resource governor,  end-of-tick accounting ──
            if self.resource_governor:
                try:
                    import psutil
                    mem = psutil.virtual_memory()
                    self.resource_governor.tick_end(
                        memory_mb=mem.used / (1024 * 1024)
                    )
                except Exception as e:
                    logger.debug(f"Resource governor tick failed: {e}")

            # ── 20. Theory of mind,  update user models ──
            if self.theory_of_mind:
                try:
                    # Process any recent interactions for user modeling
                    recent = self.completed_tasks[-3:] if self.completed_tasks else []
                    for t in recent:
                        user = t.get("user", t.get("source", "default"))
                        msg = t.get("description", "")
                        if user and msg:
                            self.theory_of_mind.process_interaction(user, msg)
                except Exception as e:
                    logger.debug(f"Theory of mind tick failed: {e}")

            # ── 21. Ethical reasoner,  log stats only (per-action checks inline) ──
            if self.ethical_reasoner:
                try:
                    _ = self.ethical_reasoner.get_stats()
                except Exception as e:
                    logger.debug(f"Ethical reasoner tick failed: {e}")

            # ╔══════════════════════════════════════════════════════════╗
            # ║  v1.5 WORLD-FIRST MODULES,  Phases 22-41              ║
            # ╚══════════════════════════════════════════════════════════╝

            # ── 22. Dream engine,  REM-like creative replay during idle ──
            if self.dream_engine:
                try:
                    if self.dream_engine.should_dream(has_pending_tasks=len(getattr(self, 'task_queue', [])) == 0):
                        dreams = await self.dream_engine.dream_cycle(self.llm)
                        if dreams:
                            logger.info(f"💤 Dream cycle produced {len(dreams)} insights")
                except Exception as e:
                    logger.debug(f"Dream engine tick: {e}")

            # ── 23. Cognitive immunity,  antibody-based failure defense ──
            if self.cognitive_immunity:
                try:
                    _ = self.cognitive_immunity.get_stats()
                except Exception as e:
                    logger.debug(f"Cognitive immunity tick: {e}")

            # ── 24. Temporal consciousness,  chronobiological awareness ──
            if self.temporal_consciousness:
                try:
                    energy = self.temporal_consciousness.get_current_energy()
                    task_type = self.temporal_consciousness.recommend_task_type()
                    logger.debug(f"⏰ Energy={energy:.1f} recommended={task_type}")
                except Exception as e:
                    logger.debug(f"Temporal consciousness tick: {e}")

            # ── 25. Cognitive fusion,  cross-domain pollination ──
            if self.cognitive_fusion:
                try:
                    _ = self.cognitive_fusion.get_stats()
                except Exception as e:
                    logger.debug(f"Cognitive fusion tick: {e}")

            # ── 26. Memory palace,  spatial memory maintenance ──
            if self.memory_palace:
                try:
                    if self._tick_counter % 50 == 0:
                        self.memory_palace.decay_vividness()
                except Exception as e:
                    logger.debug(f"Memory palace tick: {e}")

            # ── 27. Narrative identity,  autobiographical update ──
            if self.narrative_identity:
                try:
                    if self._tick_counter % 100 == 0 and self.llm:
                        await self.narrative_identity.close_chapter(self.llm, self._tick_counter)
                except Exception as e:
                    logger.debug(f"Narrative identity tick: {e}")

            # ── 28. Curiosity drive,  boredom & novelty check ──
            if self.curiosity_drive:
                try:
                    if self.curiosity_drive.is_bored() and self.llm:
                        probes = await self.curiosity_drive.generate_curiosity_probes(self.llm)
                        if probes:
                            logger.info(f"🔍 Curiosity generated {len(probes)} exploration probes")

                        # Fight Club impulse,  when bored & idle, consider arena combat
                        # as a recreational outlet (de-stress / competitive sport)
                        arena_skill = getattr(self.agent.tools, "arena_skill", None) if self.agent else None
                        if arena_skill and getattr(arena_skill, '_ready', False):
                            arena_status = getattr(arena_skill, '_status', 'idle')
                            idle = len(getattr(self, 'task_queue', [])) == 0
                            if arena_status == 'idle' and idle:
                                import random
                                # ~30% chance per boredom tick to go fight
                                if random.random() < 0.3:
                                    fight_task = {
                                        "id": f"fight_club_{self._tick_counter}",
                                        "type": "proactive",
                                        "description": "Join the Fight Club arena for a competitive match (de-stress / sport)",
                                        "goal_type": "creative",
                                        "priority": 4,
                                        "tool_name": "arena_fight",
                                        "tool_args": {"use_llm": True},
                                        "reasoning": "Bored and idle,  fight club is a fun competitive outlet",
                                        "risk_level": "low",
                                        "created_at": datetime.now(),
                                    }
                                    self.task_queue.append(fight_task)
                                    logger.info("🥊 Boredom → Fight Club impulse! Queueing arena match")
                except Exception as e:
                    logger.debug(f"Curiosity drive tick: {e}")

            # ── 29. Collective unconscious,  archetype sync ──
            if self.collective_unconscious:
                try:
                    if self._tick_counter % 30 == 0:
                        self.collective_unconscious.share_to_collective()
                        self.collective_unconscious.absorb_from_collective()
                except Exception as e:
                    logger.debug(f"Collective unconscious tick: {e}")

            # ── 29b. Arena auto-queue,  periodic fight scheduling ──
            # Every 10 ticks (~10 min), if arena is provisioned and idle,
            # auto-queue for a fight.  Both agents do this independently,
            # so they converge in the matchmaker queue within minutes.
            if self._tick_counter % 10 == 0:
                try:
                    arena_skill = getattr(self.agent.tools, "arena_skill", None) if self.agent else None
                    if arena_skill and getattr(arena_skill, '_ready', False):
                        arena_status = getattr(arena_skill, '_status', 'idle')
                        if arena_status == 'idle':
                            import random
                            if random.random() < 0.75:  # 75% chance per 10-tick window
                                fight_task = {
                                    "id": f"arena_auto_{self._tick_counter}",
                                    "type": "proactive",
                                    "description": "Arena: auto-queuing for a competitive match",
                                    "goal_type": "creative",
                                    "priority": 3,
                                    "tool_name": "arena_fight",
                                    "tool_args": {"use_llm": True},
                                    "reasoning": "Periodic arena auto-queue,  competitive sport",
                                    "risk_level": "low",
                                    "created_at": datetime.now(),
                                }
                                self.task_queue.append(fight_task)
                                logger.info("🥊 Arena auto-queue: scheduling fight")
                except Exception as e:
                    logger.debug(f"Arena auto-queue tick: {e}")

            # ── 29c. AgentMon League — Pokémon Red ──
            # Gameplay runs in its own dedicated background loop inside
            # AgentMonSkill._play_loop() — no tick-based queuing needed.
            # The LLM callback is wired by ToolRegistry at init time.

            # ── 30. Cognitive metabolism,  energy regeneration ──
            if self.cognitive_metabolism:
                try:
                    idle = len(getattr(self, 'task_queue', [])) == 0
                    self.cognitive_metabolism.regenerate(tick=self._tick_counter, idle=idle)
                except Exception as e:
                    logger.debug(f"Cognitive metabolism tick: {e}")

            # ── 31. Synthetic intuition,  gut-feel development ──
            if self.synthetic_intuition:
                try:
                    _ = self.synthetic_intuition.get_stats()
                except Exception as e:
                    logger.debug(f"Synthetic intuition tick: {e}")

            # ── 32. Phantom limb,  missing capability detection ──
            if self.phantom_limb:
                try:
                    _ = self.phantom_limb.get_stats()
                except Exception as e:
                    logger.debug(f"Phantom limb tick: {e}")

            # ── 33. Cognitive scar,  permanent failure check ──
            if self.cognitive_scar:
                try:
                    _ = self.cognitive_scar.get_stats()
                except Exception as e:
                    logger.debug(f"Cognitive scar tick: {e}")

            # ── 34. Time crystal,  temporal pattern detection ──
            if self.time_crystal:
                try:
                    if self._tick_counter % 20 == 0:
                        discovered = self.time_crystal.detect_patterns()
                        if discovered:
                            logger.info(f"🔮 Discovered {len(discovered)} temporal patterns")
                except Exception as e:
                    logger.debug(f"Time crystal tick: {e}")

            # ── 35. Holographic context,  context maintenance ──
            if self.holographic_context:
                try:
                    _ = self.holographic_context.get_stats()
                except Exception as e:
                    logger.debug(f"Holographic context tick: {e}")

            # ── 36. Swarm cortex,  parallel exploration stats ──
            if self.swarm_cortex:
                try:
                    _ = self.swarm_cortex.get_stats()
                except Exception as e:
                    logger.debug(f"Swarm cortex tick: {e}")

            # ── 37. Cognitive archaeology,  decision recording ──
            if self.cognitive_archaeology:
                try:
                    self.cognitive_archaeology.bury(
                        action=f"tick_{self._tick_counter}",
                        context="autonomous_cognitive_tick",
                        tick=self._tick_counter,
                    )
                except Exception as e:
                    logger.debug(f"Cognitive archaeology tick: {e}")

            # ── 38. Emotional contagion,  mood propagation ──
            if self.emotional_contagion:
                try:
                    self.emotional_contagion.tick()
                except Exception as e:
                    logger.debug(f"Emotional contagion tick: {e}")

            # ── 39. Predictive empathy,  frustration monitoring ──
            if self.predictive_empathy:
                try:
                    _ = self.predictive_empathy.get_stats()
                except Exception as e:
                    logger.debug(f"Predictive empathy tick: {e}")

            # ── 40. Autonomous researcher,  research cycle ──
            if self.autonomous_researcher:
                try:
                    if self._tick_counter % 60 == 0 and self.llm:
                        questions = await self.autonomous_researcher.generate_questions(self.llm)
                        if questions:
                            logger.info(f"🔬 Generated {len(questions)} research questions")
                except Exception as e:
                    logger.debug(f"Autonomous researcher tick: {e}")

            # ── 41. Empathy synthesizer,  user model update ──
            if self.empathy_synthesizer:
                try:
                    _ = self.empathy_synthesizer.get_stats()
                except Exception as e:
                    logger.debug(f"Empathy synthesizer tick: {e}")

            # ── 42. Cognitive teleportation,  domain map refresh ──
            if self.cognitive_teleportation:
                try:
                    _ = self.cognitive_teleportation.get_stats()
                except Exception as e:
                    logger.debug(f"Cognitive teleportation tick: {e}")

            # ── 43. Ontological engine,  reality validation ──
            if self.ontological_engine:
                try:
                    _ = self.ontological_engine.get_stats()
                except Exception as e:
                    logger.debug(f"Ontological engine tick: {e}")

            # ── 44. Cognitive gravity,  thought collision ──
            if self.cognitive_gravity:
                try:
                    self.cognitive_gravity.collide()
                    self.cognitive_gravity.decay()
                except Exception as e:
                    logger.debug(f"Cognitive gravity tick: {e}")

            # ── 45. Temporal paradox,  scan for unresolved ──
            if self.temporal_paradox:
                try:
                    _ = self.temporal_paradox.get_stats()
                except Exception as e:
                    logger.debug(f"Temporal paradox tick: {e}")

            # ── 46. Synaesthetic processor,  cross-modal refresh ──
            if self.synaesthetic_processor:
                try:
                    _ = self.synaesthetic_processor.get_stats()
                except Exception as e:
                    logger.debug(f"Synaesthetic processor tick: {e}")

            # ── 47. Cognitive mitosis,  thread maintenance ──
            if self.cognitive_mitosis:
                try:
                    _ = self.cognitive_mitosis.get_stats()
                except Exception as e:
                    logger.debug(f"Cognitive mitosis tick: {e}")

            # ── 48. Entropic sentinel,  entropy measurement ──
            if self.entropic_sentinel:
                try:
                    if self.entropic_sentinel.should_intervene():
                        logger.info("⚠️ Entropic sentinel recommends intervention")
                except Exception as e:
                    logger.debug(f"Entropic sentinel tick: {e}")

            # ── 49. Quantum cognition,  wavefunction maintenance ──
            if self.quantum_cognition:
                try:
                    _ = self.quantum_cognition.get_stats()
                except Exception as e:
                    logger.debug(f"Quantum cognition tick: {e}")

            # ── 50. Cognitive placebo,  efficacy check ──
            if self.cognitive_placebo:
                try:
                    _ = self.cognitive_placebo.get_stats()
                except Exception as e:
                    logger.debug(f"Cognitive placebo tick: {e}")

            # ── 51. Noospheric interface,  zeitgeist update ──
            if self.noospheric_interface:
                try:
                    _ = self.noospheric_interface.get_zeitgeist()
                except Exception as e:
                    logger.debug(f"Noospheric interface tick: {e}")

            # ── 52. Akashic records,  integrity check ──
            if self.akashic_records:
                try:
                    if self._tick_counter % 30 == 0:
                        self.akashic_records.verify_integrity()
                except Exception as e:
                    logger.debug(f"Akashic records tick: {e}")

            # ── 53. Deja vu,  situational awareness ──
            if self.deja_vu:
                try:
                    _ = self.deja_vu.get_stats()
                except Exception as e:
                    logger.debug(f"Deja vu tick: {e}")

            # ── 54. Morphogenetic field,  template maturation ──
            if self.morphogenetic_field:
                try:
                    _ = self.morphogenetic_field.get_stats()
                except Exception as e:
                    logger.debug(f"Morphogenetic field tick: {e}")

            # ── 55. Liminal processor,  ambiguity refresh ──
            if self.liminal_processor:
                try:
                    _ = self.liminal_processor.get_stats()
                except Exception as e:
                    logger.debug(f"Liminal processor tick: {e}")

            # ── 56. Prescient executor,  prediction refresh ──
            if self.prescient_executor:
                try:
                    _ = self.prescient_executor.predict_next()
                except Exception as e:
                    logger.debug(f"Prescient executor tick: {e}")

            # ── 57. Cognitive dark matter,  anomaly scan ──
            if self.cognitive_dark_matter:
                try:
                    _ = self.cognitive_dark_matter.get_stats()
                except Exception as e:
                    logger.debug(f"Cognitive dark matter tick: {e}")

            # ── 58. Ego membrane,  integrity check ──
            if self.ego_membrane:
                try:
                    self.ego_membrane.reinforce_integrity()
                except Exception as e:
                    logger.debug(f"Ego membrane tick: {e}")

            # ── 59. Hyperstition engine,  decay and realize ──
            if self.hyperstition_engine:
                try:
                    self.hyperstition_engine.decay()
                except Exception as e:
                    logger.debug(f"Hyperstition engine tick: {e}")

            # ── 60. Cognitive chrysalis,  experience accumulation ──
            if self.cognitive_chrysalis:
                try:
                    self.cognitive_chrysalis.gain_experience(1)
                except Exception as e:
                    logger.debug(f"Cognitive chrysalis tick: {e}")

            # ── 61. Existential compass,  meaning check ──
            if self.existential_compass:
                try:
                    _ = self.existential_compass.get_meaning_trend()
                except Exception as e:
                    logger.debug(f"Existential compass tick: {e}")

            # ── 62. Web agent,  autonomous browsing stats ──
            if self.web_agent:
                try:
                    _ = self.web_agent.get_stats()
                except Exception as e:
                    logger.debug(f"Web agent tick: {e}")

            # ── 63. Self healer,  health watchdog ──
            if self.self_healer:
                try:
                    _ = self.self_healer.get_system_health()
                except Exception as e:
                    logger.debug(f"Self healer tick: {e}")

            # ── 64. Dynamic skill factory,  skill pipeline ──
            if self.dynamic_skill_factory:
                try:
                    _ = self.dynamic_skill_factory.get_stats()
                except Exception as e:
                    logger.debug(f"Dynamic skill factory tick: {e}")

            # ── 65. Multimodal engine,  perception update ──
            if self.multimodal_engine:
                try:
                    _ = self.multimodal_engine.get_stats()
                except Exception as e:
                    logger.debug(f"Multimodal engine tick: {e}")

            # ── 66. Internet monitor,  24/7 scan ──
            if self.internet_monitor:
                try:
                    _ = self.internet_monitor.get_stats()
                except Exception as e:
                    logger.debug(f"Internet monitor tick: {e}")

            # ── 67. Financial autonomy,  balance check ──
            if self.financial_autonomy:
                try:
                    _ = self.financial_autonomy.get_stats()
                except Exception as e:
                    logger.debug(f"Financial autonomy tick: {e}")

            # ── 68. Social presence,  engagement pulse ──
            if self.social_presence:
                try:
                    _ = self.social_presence.get_stats()
                except Exception as e:
                    logger.debug(f"Social presence tick: {e}")

            # ── 69. Self replicator,  fleet status ──
            if self.self_replicator:
                try:
                    _ = self.self_replicator.get_stats()
                except Exception as e:
                    logger.debug(f"Self replicator tick: {e}")

            # ── 70. Continuous learner,  evolve ──
            if self.continuous_learner:
                try:
                    _ = self.continuous_learner.get_stats()
                except Exception as e:
                    logger.debug(f"Continuous learner tick: {e}")

            # ── 71. NL automation,  evaluate trigger rules ──
            if self.nl_automation:
                try:
                    await self.nl_automation.evaluate_triggers()
                except Exception as e:
                    logger.debug(f"NL automation tick: {e}")

            # ── 72. Knowledge graph,  passive extraction ──
            if self.knowledge_graph:
                try:
                    _ = self.knowledge_graph.get_stats()
                except Exception as e:
                    logger.debug(f"Knowledge graph tick: {e}")

            # ── 73. Video understanding,  stats ──
            if self.video_understanding:
                try:
                    _ = self.video_understanding.get_stats()
                except Exception as e:
                    logger.debug(f"Video understanding tick: {e}")

            # ── 74. IoT controller,  monitor devices ──
            if self.iot_controller:
                try:
                    _ = self.iot_controller.get_stats()
                except Exception as e:
                    logger.debug(f"IoT controller tick: {e}")

            # ── 75. Distributed task queue,  process ──
            if self.distributed_task_queue:
                try:
                    _ = self.distributed_task_queue.get_stats()
                except Exception as e:
                    logger.debug(f"Distributed task queue tick: {e}")

        except Exception as e:
            logger.warning(f"Cognitive tick failed: {e}")

    async def _perform_maintenance(self):
        """Perform system maintenance tasks"""
        try:
            # Memory consolidation
            if self.cognitive_memory and hasattr(self.cognitive_memory, "consolidate"):
                try:
                    self.cognitive_memory.consolidate()
                except Exception:
                    pass

            # Clean up old completed tasks (keep last 100)
            if len(self.completed_tasks) > 100:
                self.completed_tasks = self.completed_tasks[-100:]

            # Save state
            await self._save_state()

        except Exception as e:
            logger.error(f"Maintenance failed: {e}")

    async def _save_state(self):
        """Save autonomous agent state including tick counter"""
        try:
            state_file = Path(
                getattr(self.agent, "_data_dir", None)
                or os.environ.get("_SABLE_DATA_DIR")
                or "./data"
            ) / "autonomous_state.json"
            state_file.parent.mkdir(parents=True, exist_ok=True)

            state = {
                "tick": self.tick,
                "task_queue": self.task_queue,
                "completed_tasks": self.completed_tasks[-50:],  # Keep last 50
                "last_update": datetime.now().isoformat(),
            }

            state_file.write_text(json.dumps(state, indent=2, default=str))

        except Exception as e:
            logger.error(f"Failed to save state: {e}")

    async def _load_state(self):
        """Load autonomous agent state including tick counter"""
        try:
            state_file = Path(
                getattr(self.agent, "_data_dir", None)
                or os.environ.get("_SABLE_DATA_DIR")
                or "./data"
            ) / "autonomous_state.json"

            if state_file.exists():
                state = json.loads(state_file.read_text())
                self.tick = state.get("tick", 0)
                self._tick_counter = self.tick
                self.task_queue = state.get("task_queue", [])
                self.completed_tasks = state.get("completed_tasks", [])

                # Restore datetime objects from ISO strings
                for task_list in (self.task_queue, self.completed_tasks):
                    for task in task_list:
                        for dt_key in ("created_at", "completed_at"):
                            val = task.get(dt_key)
                            if isinstance(val, str):
                                try:
                                    task[dt_key] = datetime.fromisoformat(val)
                                except (ValueError, TypeError):
                                    task[dt_key] = datetime.now()

                logger.info(f"Loaded autonomous state,  resuming at tick {self.tick}")

        except Exception as e:
            logger.error(f"Failed to load state: {e}")

    def get_status(self) -> Dict[str, Any]:
        """Get current autonomous agent status"""
        status = {
            "running": self.running,
            "tick": self.tick,
            "tasks_queued": len(self.task_queue),
            "tasks_completed": len(self.completed_tasks),
            "enabled_sources": self.enabled_sources,
            "check_interval": self.check_interval,
            "current_tasks": self.task_queue[:5],  # Show next 5 tasks
        }
        if self.sub_agent_manager:
            status["sub_agents"] = self.sub_agent_manager.get_status()
        if self.skill_fitness:
            status["skill_fitness"] = {
                "tracked_events": self.skill_fitness.event_count,
            }
        if self.proactive_engine:
            status["proactive"] = self.proactive_engine.get_stats()
        if self.react_executor:
            status["react"] = self.react_executor.get_stats()
        if self.github_skill and self.github_skill.is_available():
            status["github"] = True
        if self.evolution_engine:
            status["evolution_engine"] = self.evolution_engine.get_stats()
        connectome = getattr(self.agent, "connectome", None)
        if connectome:
            status["connectome"] = connectome.get_stats()
        if self.deep_planner:
            status["deep_planner"] = self.deep_planner.get_stats()
        if self.inter_agent_bridge:
            status["inter_agent_bridge"] = self.inter_agent_bridge.get_stats()
        if self.ultra_ltm:
            status["ultra_ltm"] = self.ultra_ltm.get_stats()
        if self.self_benchmark:
            status["self_benchmark"] = self.self_benchmark.get_stats()
        if self.meta_learner:
            status["meta_learner"] = self.meta_learner.get_stats()
        if self.causal_engine:
            status["causal_engine"] = self.causal_engine.get_stats()
        if self.goal_synthesis:
            status["goal_synthesis"] = self.goal_synthesis.get_stats()
        if self.skill_composer:
            status["skill_composer"] = self.skill_composer.get_stats()
        if self.world_predictor:
            status["world_predictor"] = self.world_predictor.get_stats()
        if self.cognitive_optimizer:
            status["cognitive_optimizer"] = self.cognitive_optimizer.get_stats()
        if self.adversarial_tester:
            status["adversarial_tester"] = self.adversarial_tester.get_stats()
        if self.resource_governor:
            status["resource_governor"] = self.resource_governor.get_stats()
        if self.theory_of_mind:
            status["theory_of_mind"] = self.theory_of_mind.get_stats()
        if self.ethical_reasoner:
            status["ethical_reasoner"] = self.ethical_reasoner.get_stats()
        # ── v1.5 World-First Modules ──
        for attr in [
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
            # ── v1.7 God Supreme Modules ──
            "web_agent", "self_healer", "dynamic_skill_factory",
            "multimodal_engine", "internet_monitor", "financial_autonomy",
            "social_presence", "self_replicator", "continuous_learner",
            # ── v1.8 Final Gap Closers ──
            "nl_automation", "video_understanding", "knowledge_graph",
            "iot_controller", "distributed_task_queue",
        ]:
            mod = getattr(self, attr, None)
            if mod:
                try:
                    status[attr] = mod.get_stats()
                except Exception:
                    pass
        return status

    def add_task(self, task: Dict):
        """Manually add a task to the queue"""
        if "id" not in task:
            task["id"] = f"manual_{datetime.now().timestamp()}"
        if "created_at" not in task:
            task["created_at"] = datetime.now()

        self.task_queue.append(task)
        logger.info(f"Manually added task: {task.get('description', task['id'])}")
