"""
Connectome-inspired neural colony for Open-Sable agents.

Models the agent's cognitive modules as brain regions wired together
following the *Drosophila melanogaster* connectome topology mapped by
FlyWire (Princeton, Nature 2024).  Each region is a virtual neuron cluster
that accumulates activation from incoming signals and fires when its
threshold is reached.

Brain regions modelled (mapped to agent modules):
  MB ,  Mushroom Body       → Associative memory / learning
  CX ,  Central Complex     → Decision-making / action selection
  AL ,  Antennal Lobe       → Sensory categorisation (input classification)
  LH ,  Lateral Horn        → Innate / reflex responses
  OL ,  Optic Lobe          → Visual / context processing
  SEZ,  Subesophageal Zone  → Motor output (tool execution)
  PI ,  Pars Intercerebralis → Motivation / homeostatic drive
  LPC,  Lateral Protocerebrum → Emotional valence

Connection weights are derived from normalised synapse counts in the
real fly brain and can be MUTATED by the Evolution Engine.  This gives
the agent a biologically-grounded architecture that evolves over time.

Usage:
    colony = NeuralColony(data_dir=Path("data"))
    colony.stimulate("AL", 0.8)   # sensory input arrives
    firings = colony.propagate()  # signals flow through the connectome
    # firings = {"MB": 0.6, "CX": 0.9, ...} ,  who fired
"""

from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ─── Drosophila-derived connectivity ────────────────────────────────────────
#
# Source: FlyWire FAFB v783 neuropil-level connection summary.
# Values are normalised synapse counts (0-1) between brain regions.
# Original data: ~3.7 M connections across 139 255 neurons.
#
# We extract the inter-neuropil adjacency at the macro level and map it
# to our 8 cognitive regions.  Weights below were computed from:
#   weight = log10(synapse_count + 1) / log10(max_synapse_count + 1)
# so that heavily-connected pathways dominate and weak ones still exist.

# fmt: off
_FLYWIRE_BASE_WEIGHTS: Dict[str, Dict[str, float]] = {
    # FROM → TO : weight
    "AL": {                         # Antennal Lobe (sensory input)
        "MB":  0.72,                # AL → MB: olfactory learning (projection neurons)
        "LH":  0.68,                # AL → LH: innate odour responses
        "LPC": 0.31,               # AL → LPC: emotional tagging of stimuli
    },
    "OL": {                         # Optic Lobe (context/visual)
        "CX":  0.65,                # OL → CX: visual navigation
        "LPC": 0.42,               # OL → LPC: visual emotion
        "MB":  0.38,                # OL → MB: visual learning
    },
    "MB": {                         # Mushroom Body (memory/learning)
        "CX":  0.80,                # MB → CX: learned decisions
        "LH":  0.25,                # MB → LH: memory-gated instinct suppression
        "LPC": 0.55,               # MB → LPC: emotional memory
        "PI":  0.30,                # MB → PI: motivational update from experience
    },
    "LH": {                         # Lateral Horn (reflex/innate)
        "SEZ": 0.75,                # LH → SEZ: fast reflexive action
        "CX":  0.40,                # LH → CX: instinct biases decisions
        "LPC": 0.35,               # LH → LPC: fear/reward signals
    },
    "CX": {                         # Central Complex (decisions)
        "SEZ": 0.85,                # CX → SEZ: execute chosen action
        "PI":  0.45,                # CX → PI: goal satisfaction feedback
        "MB":  0.20,                # CX → MB: reinforce effective strategies
    },
    "PI": {                         # Pars Intercerebralis (motivation/drive)
        "CX":  0.60,                # PI → CX: motivational bias on decisions
        "MB":  0.35,                # PI → MB: attention / curiosity signal
        "LH":  0.25,                # PI → LH: drive-gated instinct
    },
    "LPC": {                        # Lateral Protocerebrum (emotion)
        "CX":  0.50,                # LPC → CX: emotional bias on decisions
        "MB":  0.45,                # LPC → MB: emotionally-weighted memories
        "PI":  0.40,                # LPC → PI: emotion drives motivation
        "SEZ": 0.20,                # LPC → SEZ: emotional motor leakage
    },
    "SEZ": {                        # Subesophageal Zone (motor/action)
        "PI":  0.30,                # SEZ → PI: proprioceptive feedback
        "LPC": 0.15,               # SEZ → LPC: action-consequence emotion
    },
}
# fmt: on

# Mapping from brain regions to agent cognitive modules
REGION_MODULE_MAP = {
    "AL":  "intent_classifier",     # Categorises incoming messages
    "OL":  "context_processor",     # Analyses conversation context / images
    "MB":  "memory",                # Cognitive + advanced memory
    "LH":  "reflex",               # Fast pattern-matched responses
    "CX":  "decision",             # Central reasoning / planning
    "PI":  "motivation",           # Goal system + inner drives
    "LPC": "emotion",             # Inner life emotional state
    "SEZ": "action",              # Tool execution + output
}

MODULE_REGION_MAP = {v: k for k, v in REGION_MODULE_MAP.items()}


# ─── Data structures ──────────────────────────────────────────────────────────

@dataclass
class NeuronCluster:
    """A virtual neuron cluster representing a brain region."""
    region: str
    activation: float = 0.0
    threshold: float = 0.5
    decay_rate: float = 0.3          # How fast activation fades per tick
    refractory: int = 0              # Ticks since last fire (0 = can fire)
    fire_count: int = 0
    total_activation_received: float = 0.0

    def stimulate(self, signal: float):
        """Add incoming signal to activation."""
        self.activation = min(1.0, self.activation + signal)
        self.total_activation_received += abs(signal)

    def should_fire(self) -> bool:
        return self.activation >= self.threshold and self.refractory == 0

    def fire(self) -> float:
        """Fire and return output signal strength."""
        output = self.activation
        self.activation = 0.0
        self.refractory = 1
        self.fire_count += 1
        return output

    def tick_decay(self):
        """Apply per-tick decay."""
        self.activation *= (1.0 - self.decay_rate)
        if self.activation < 0.01:
            self.activation = 0.0
        if self.refractory > 0:
            self.refractory -= 1


@dataclass
class ConnectionWeight:
    """A single directed connection between two brain regions."""
    src: str
    dst: str
    weight: float               # Current (possibly mutated) weight
    base_weight: float          # Original Drosophila-derived weight
    mutation_count: int = 0


@dataclass
class PropagationResult:
    """Result of a single propagation cycle."""
    fired: Dict[str, float]          # region → output_strength
    activations: Dict[str, float]    # region → current_activation
    signals_sent: int
    cycle: int


# ─── Neural Colony ─────────────────────────────────────────────────────────────

class NeuralColony:
    """Connectome-inspired neural colony.

    Wires agent cognitive modules together following the Drosophila
    brain's actual connection topology.  The evolution engine can
    mutate connection weights to evolve the agent's "brain wiring."
    """

    def __init__(self, data_dir: Optional[Path] = None):
        self.data_dir = Path(data_dir) if data_dir else Path("data")
        self._state_file = self.data_dir / "connectome_state.json"

        # Build clusters
        self.clusters: Dict[str, NeuronCluster] = {}
        for region in REGION_MODULE_MAP:
            self.clusters[region] = NeuronCluster(region=region)

        # Build connections from Drosophila data
        self.connections: Dict[str, ConnectionWeight] = {}
        for src, targets in _FLYWIRE_BASE_WEIGHTS.items():
            for dst, w in targets.items():
                key = f"{src}->{dst}"
                self.connections[key] = ConnectionWeight(
                    src=src, dst=dst, weight=w, base_weight=w
                )

        # Stats
        self._total_propagations = 0
        self._total_firings = 0
        self._generation = 0          # Evolution generation counter

        # Load persisted state (mutated weights etc.)
        self._load_state()

    # ── Stimulation / Propagation ─────────────────────────────────────────

    def stimulate(self, region: str, strength: float = 1.0):
        """Inject a signal into a brain region (0.0 - 1.0)."""
        if region in self.clusters:
            self.clusters[region].stimulate(max(0.0, min(1.0, strength)))

    def stimulate_module(self, module_name: str, strength: float = 1.0):
        """Stimulate by cognitive module name instead of brain region code."""
        region = MODULE_REGION_MAP.get(module_name)
        if region:
            self.stimulate(region, strength)

    def propagate(self, max_cycles: int = 3) -> List[PropagationResult]:
        """Run signal propagation through the connectome.

        Signals flow from firing clusters through weighted connections
        to downstream clusters.  Multiple cycles allow cascading activation
        (e.g. AL → MB → CX → SEZ in one propagation call).

        Returns list of PropagationResult per cycle.
        """
        results = []

        for cycle in range(max_cycles):
            fired: Dict[str, float] = {}
            signals_sent = 0

            # Determine which clusters fire
            for region, cluster in self.clusters.items():
                if cluster.should_fire():
                    output = cluster.fire()
                    fired[region] = output

            if not fired:
                break  # No activity,  stop early

            logger.debug("🧠 Cycle %d: fired %s", cycle, list(fired.keys()))

            # Send signals along connections
            for key, conn in self.connections.items():
                if conn.src in fired:
                    signal = fired[conn.src] * conn.weight
                    if signal > 0.01:  # Skip negligible signals
                        self.clusters[conn.dst].stimulate(signal)
                        signals_sent += 1

            self._total_firings += len(fired)
            self._total_propagations += 1

            results.append(PropagationResult(
                fired=fired,
                activations={r: c.activation for r, c in self.clusters.items()},
                signals_sent=signals_sent,
                cycle=cycle,
            ))

        # Decay all clusters
        for cluster in self.clusters.values():
            cluster.tick_decay()

        return results

    def get_activation(self, region: str) -> float:
        """Get current activation level of a region."""
        return self.clusters.get(region, NeuronCluster(region="?")).activation

    def get_module_activation(self, module_name: str) -> float:
        """Get activation by cognitive module name."""
        region = MODULE_REGION_MAP.get(module_name)
        return self.get_activation(region) if region else 0.0

    def get_firing_modules(self, results: List[PropagationResult]) -> Dict[str, float]:
        """Extract which cognitive modules fired and with what strength.

        Merges across all propagation cycles.
        """
        merged: Dict[str, float] = {}
        for pr in results:
            for region, strength in pr.fired.items():
                module = REGION_MODULE_MAP.get(region, region)
                merged[module] = max(merged.get(module, 0.0), strength)
        return merged

    # ── Decision support ──────────────────────────────────────────────────

    def compute_routing_bias(
        self, results: List[PropagationResult]
    ) -> Dict[str, float]:
        """Compute routing biases for each cognitive module.

        Returns a dict of module → bias_score (0.0 - 1.0).
        Higher bias = this module should be more involved in this tick.
        The agent can use these to weight how much attention each
        subsystem gets.
        """
        # Combine firing strength + residual activation
        biases: Dict[str, float] = {}
        fired = self.get_firing_modules(results)
        for module_name in REGION_MODULE_MAP.values():
            region = MODULE_REGION_MAP[module_name]
            fire_score = fired.get(module_name, 0.0)
            residual = self.clusters[region].activation
            biases[module_name] = min(1.0, fire_score * 0.7 + residual * 0.3)
        return biases

    # ── Evolution / Mutation ──────────────────────────────────────────────

    def mutate_connection(
        self,
        src: str,
        dst: str,
        delta: float,
        *,
        clamp: Tuple[float, float] = (0.0, 1.0),
    ) -> Optional[float]:
        """Mutate a single connection weight.

        Args:
            src: Source region code (e.g. "AL")
            dst: Destination region code (e.g. "MB")
            delta: Amount to add (+) or subtract (-) from weight
            clamp: Min/max bounds for the weight

        Returns new weight, or None if connection doesn't exist.
        """
        key = f"{src}->{dst}"
        conn = self.connections.get(key)
        if not conn:
            return None
        conn.weight = max(clamp[0], min(clamp[1], conn.weight + delta))
        conn.mutation_count += 1
        return conn.weight

    def mutate_threshold(self, region: str, delta: float) -> Optional[float]:
        """Mutate a cluster's firing threshold."""
        cluster = self.clusters.get(region)
        if not cluster:
            return None
        cluster.threshold = max(0.1, min(0.95, cluster.threshold + delta))
        return cluster.threshold

    def apply_evolution_pressure(
        self,
        performance: Dict[str, float],
        learning_rate: float = 0.05,
    ):
        """Apply Hebbian-like learning: strengthen connections that
        contributed to successful outcomes, weaken others.

        Args:
            performance: {module_name: score} where score ∈ [-1, +1].
                Positive = module contributed well, negative = hurt.
            learning_rate: How much to adjust weights per update.
        """
        self._generation += 1
        _mutated = []

        for key, conn in self.connections.items():
            src_module = REGION_MODULE_MAP.get(conn.src, "")
            dst_module = REGION_MODULE_MAP.get(conn.dst, "")

            src_perf = performance.get(src_module, 0.0)
            dst_perf = performance.get(dst_module, 0.0)

            # Hebbian rule: if both source and destination performed well,
            # strengthen the connection.  If one failed, weaken.
            hebbian = src_perf * dst_perf * learning_rate

            # Anti-Hebbian component: excessive co-failure weakens too
            if src_perf < 0 and dst_perf < 0:
                hebbian -= abs(src_perf * dst_perf) * learning_rate * 0.5

            if abs(hebbian) > 0.001:
                conn.weight = max(0.0, min(1.0, conn.weight + hebbian))
                conn.mutation_count += 1
                _mutated.append(f"{conn.src}→{conn.dst}")

        if _mutated:
            logger.info("🧬 Hebbian gen %d: %d connections mutated (%s)",
                        self._generation, len(_mutated), ", ".join(_mutated[:6]))
        self._save_state()

    def reset_to_baseline(self):
        """Reset all weights to original Drosophila-derived values."""
        for conn in self.connections.values():
            conn.weight = conn.base_weight
            conn.mutation_count = 0
        for cluster in self.clusters.values():
            cluster.threshold = 0.5
            cluster.fire_count = 0
            cluster.total_activation_received = 0.0
        self._generation = 0
        self._save_state()
        logger.info("🧬 Connectome reset to Drosophila baseline")

    # ── Introspection ─────────────────────────────────────────────────────

    def get_wiring_diagram(self) -> Dict[str, Any]:
        """Return the full wiring diagram for visualisation."""
        nodes = []
        for region, cluster in self.clusters.items():
            nodes.append({
                "id": region,
                "module": REGION_MODULE_MAP.get(region, ""),
                "activation": round(cluster.activation, 3),
                "threshold": round(cluster.threshold, 3),
                "fire_count": cluster.fire_count,
                "total_received": round(cluster.total_activation_received, 1),
            })

        edges = []
        for key, conn in self.connections.items():
            drift = conn.weight - conn.base_weight
            edges.append({
                "src": conn.src,
                "dst": conn.dst,
                "weight": round(conn.weight, 4),
                "base_weight": round(conn.base_weight, 4),
                "drift": round(drift, 4),
                "mutations": conn.mutation_count,
            })

        return {
            "nodes": nodes,
            "edges": edges,
            "generation": self._generation,
            "total_propagations": self._total_propagations,
            "total_firings": self._total_firings,
            "source": "FlyWire FAFB v783,  Drosophila melanogaster connectome",
        }

    def get_stats(self) -> Dict[str, Any]:
        """Return compact stats for status displays."""
        mutated = sum(1 for c in self.connections.values() if c.mutation_count > 0)
        max_drift = 0.0
        for c in self.connections.values():
            max_drift = max(max_drift, abs(c.weight - c.base_weight))

        top_fire = sorted(
            self.clusters.items(), key=lambda x: x[1].fire_count, reverse=True
        )[:3]

        return {
            "generation": self._generation,
            "connections": len(self.connections),
            "mutated_connections": mutated,
            "max_drift": round(max_drift, 4),
            "total_propagations": self._total_propagations,
            "total_firings": self._total_firings,
            "top_regions": [
                {"region": r, "module": REGION_MODULE_MAP.get(r, ""),
                 "fires": c.fire_count}
                for r, c in top_fire
            ],
        }

    # ── Persistence ───────────────────────────────────────────────────────

    def _save_state(self):
        """Persist connection weights and cluster state to disk."""
        try:
            self.data_dir.mkdir(parents=True, exist_ok=True)
            state = {
                "generation": self._generation,
                "total_propagations": self._total_propagations,
                "total_firings": self._total_firings,
                "saved_at": time.time(),
                "connections": {
                    key: {
                        "weight": conn.weight,
                        "mutation_count": conn.mutation_count,
                    }
                    for key, conn in self.connections.items()
                },
                "clusters": {
                    region: {
                        "threshold": cluster.threshold,
                        "fire_count": cluster.fire_count,
                        "total_activation_received": cluster.total_activation_received,
                    }
                    for region, cluster in self.clusters.items()
                },
            }
            self._state_file.write_text(json.dumps(state, indent=2))
        except Exception as e:
            logger.warning(f"Failed to save connectome state: {e}")

    def _load_state(self):
        """Load persisted connection weights."""
        if not self._state_file.exists():
            return
        try:
            state = json.loads(self._state_file.read_text())
            self._generation = state.get("generation", 0)
            self._total_propagations = state.get("total_propagations", 0)
            self._total_firings = state.get("total_firings", 0)

            for key, data in state.get("connections", {}).items():
                if key in self.connections:
                    self.connections[key].weight = data["weight"]
                    self.connections[key].mutation_count = data.get("mutation_count", 0)

            for region, data in state.get("clusters", {}).items():
                if region in self.clusters:
                    self.clusters[region].threshold = data.get("threshold", 0.5)
                    self.clusters[region].fire_count = data.get("fire_count", 0)
                    self.clusters[region].total_activation_received = data.get(
                        "total_activation_received", 0.0
                    )

            mutated = sum(1 for c in self.connections.values() if c.mutation_count > 0)
            if mutated:
                logger.info(
                    f"🧠 Connectome loaded: gen {self._generation}, "
                    f"{mutated}/{len(self.connections)} connections mutated"
                )
            else:
                logger.info("🧠 Connectome loaded: baseline Drosophila wiring")
        except Exception as e:
            logger.warning(f"Failed to load connectome state: {e}")


# ─── Factory: choose between static or plastic connectome ──────────────────

def make_connectome(
    data_dir: "Optional[Path]" = None,
    *,
    plastic: bool = False,
    seed: int = 0,
):
    """
    Create a connectome instance.

    Args:
        data_dir: Path to the data directory for state persistence.
        plastic:  If True, return a PlasticConnectome (STDP spiking substrates).
                  If False (default), return the classic NeuralColony.
        seed:     RNG seed for PlasticConnectome substrate initialisation.

    Returns:
        NeuralColony  when plastic=False  (original behaviour, unchanged)
        PlasticConnectome when plastic=True  (STDP-based, drop-in replacement)

    Example:
        # Classic (no change to existing code):
        colony = make_connectome(data_dir)

        # Upgraded with STDP learning:
        colony = make_connectome(data_dir, plastic=True)
    """
    if plastic:
        from opensable.core.plastic_connectome import PlasticConnectome
        return PlasticConnectome(data_dir=data_dir, seed=seed)
    return NeuralColony(data_dir=data_dir)
