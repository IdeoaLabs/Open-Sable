"""
Plastic Connectome,  STDP-based spiking upgrade for the NeuralColony.

Replaces the static activation-propagation model with per-region Izhikevich
spiking substrates.  Each of the 8 brain regions runs its own spiking network;
STDP continuously refines intra-region synaptic weights based on co-activation
patterns.  Inter-region communication still uses FlyWire-derived weights, but
now drives actual spike stimulation instead of scalar additions.

Key differences vs NeuralColony
────────────────────────────────
  NeuralColony       : fixed FlyWire weights, continuous activation (0-1)
  PlasticConnectome  : STDP within each region, weights grow from experience,
                       persist across sessions in a .npz checkpoint file

Drop-in replacement
───────────────────
PlasticConnectome exposes the exact same public API as NeuralColony so it can
be swapped in without touching any other SableCore module:

    colony = PlasticConnectome(data_dir=Path("data"))
    colony.stimulate("AL", 0.8)
    firings = colony.propagate()
    biases  = colony.compute_routing_bias(firings)

Architecture
────────────
  8 Regions × Mini-substrate (total 1 000 neurons)

  Region  Neurons  Role
  ──────  ───────  ────────────────────────────────────────
  MB       250     Associative memory / Hebbian learning
  CX       200     Decision-making / action selection
  LPC      150     Emotional valence
  PI       100     Motivational drive
  AL       100     Sensory input classification
  OL        80     Context / visual processing
  LH        60     Reflex / innate responses
  SEZ       60     Motor output / tool execution

Inter-region propagation:
  signal = firing_rate(src) × FlyWire_weight(src→dst)
  → injected as µA stimulation current into dst substrate

STDP per region:
  Potentiation (LTP): A+ = 0.005, τ+ = 20 ms
  Depression  (LTD): A- = 0.006, τ- = 25 ms
  Homeostatic target: 5 Hz mean firing rate
  Synaptic normalisation: max total excitatory input = 15 pA

Persistence:
  Weights saved as numpy .npz after every propagation cycle.
  File: <data_dir>/plastic_connectome_weights.npz
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ─── FlyWire inter-region topology (same as NeuralColony) ──────────────────

_FLYWIRE_BASE_WEIGHTS: Dict[str, Dict[str, float]] = {
    "AL": {"MB": 0.72, "LH": 0.68, "LPC": 0.31},
    "OL": {"CX": 0.65, "LPC": 0.42, "MB": 0.38},
    "MB": {"CX": 0.80, "LH": 0.25, "LPC": 0.55, "PI": 0.30},
    "LH": {"SEZ": 0.75, "CX": 0.40, "LPC": 0.35},
    "CX": {"SEZ": 0.85, "PI": 0.45, "MB": 0.20},
    "PI": {"CX": 0.60, "MB": 0.35, "LH": 0.25},
    "LPC": {"CX": 0.50, "MB": 0.45, "PI": 0.40, "SEZ": 0.20},
    "SEZ": {"PI": 0.30, "LPC": 0.15},
}

REGION_MODULE_MAP = {
    "AL": "intent_classifier",
    "OL": "context_processor",
    "MB": "memory",
    "LH": "reflex",
    "CX": "decision",
    "PI": "motivation",
    "LPC": "emotion",
    "SEZ": "action",
}
MODULE_REGION_MAP = {v: k for k, v in REGION_MODULE_MAP.items()}

# Neuron budget per region (total = 1 000)
_REGION_NEURONS: Dict[str, int] = {
    "MB": 250,
    "CX": 200,
    "LPC": 150,
    "PI": 100,
    "AL": 100,
    "OL":  80,
    "LH":  60,
    "SEZ": 60,
}

# ─── Izhikevich mini-substrate per region ──────────────────────────────────

class _RegionSubstrate:
    """
    Self-contained Izhikevich spiking network for a single brain region.

    Parameters follow the CL1_LLM_Encoder neural_substrate.py (v3) which
    was validated against real MEA recordings.  Scale is reduced to match
    the per-region neuron budget.
    """

    # STDP / homeostasis constants (identical to CL1 v3 validated settings)
    _A_PLUS      = 0.005
    _A_MINUS     = 0.006
    _TAU_PLUS    = 20.0   # ms
    _TAU_MINUS   = 25.0   # ms
    _STDP_EVERY  = 5      # apply STDP every N steps
    _TARGET_HZ   = 5.0
    _HOMEO_RATE  = 0.0001
    _HOMEO_TAU   = 1000.0
    _MAX_EXC_IN  = 15.0
    _NOISE       = 3.0    # pA tonic noise
    _STIM_GAIN   = 6.0    # µA → pA
    _DT          = 0.5    # ms timestep
    _CONN_PROB   = 0.05   # slightly denser than full-network (smaller N)
    _EXC_W_INIT  = 0.3
    _INH_W_INIT  = -0.8
    _EXC_W_MAX   = 0.8
    _INH_W_MAX   = -1.5

    # Stimulation window per propagation call
    _WINDOW_MS   = 100.0  # ms of spiking per tick (shorter = faster SableCore)

    def __init__(self, region: str, n_neurons: int, seed: int = 0):
        self.region = region
        self.N = n_neurons
        self.Ne = int(n_neurons * 0.8)   # 80% excitatory
        self.Ni = n_neurons - self.Ne
        self.rng = np.random.default_rng(seed)
        self._step_counter = 0
        self.total_spikes = 0
        self.total_steps = 0

        # ── Izhikevich parameters ──────────────────────────────────────
        re = self.rng.random(self.Ne)
        ri = self.rng.random(self.Ni)
        self.a = np.concatenate([
            0.02 * np.ones(self.Ne),
            0.02 + 0.08 * ri,
        ])
        self.b = np.concatenate([
            0.2 * np.ones(self.Ne),
            0.25 - 0.05 * ri,
        ])
        self.c = np.concatenate([
            -65 + 15 * re**2,
            -65 * np.ones(self.Ni),
        ])
        self.d = np.concatenate([
            8 - 6 * re**2,
            2 * np.ones(self.Ni),
        ])

        # ── State variables ────────────────────────────────────────────
        self.v = -65.0 * np.ones(self.N, dtype=np.float32)
        self.u = (self.b * self.v).astype(np.float32)

        # ── Synaptic weight matrix ─────────────────────────────────────
        n_conn = int(self.N * self.N * self._CONN_PROB)
        self.S = np.zeros((self.N, self.N), dtype=np.float32)
        src = self.rng.integers(0, self.N, n_conn)
        tgt = self.rng.integers(0, self.N, n_conn)
        for s, t in zip(src, tgt):
            if s != t:
                if s < self.Ne:
                    self.S[t, s] = float(self._EXC_W_INIT * self.rng.random())
                else:
                    self.S[t, s] = float(self._INH_W_INIT * self.rng.random())

        # ── STDP traces ───────────────────────────────────────────────
        self._trace_pre  = np.zeros(self.N, dtype=np.float32)
        self._trace_post = np.zeros(self.N, dtype=np.float32)

        # ── Homeostasis ───────────────────────────────────────────────
        self._rate_est   = np.zeros(self.N, dtype=np.float32)
        self._homeo_bias = np.zeros(self.N, dtype=np.float32)

        # ── Activation output (smoothed spike rate → 0-1) ────────────
        self._activation: float = 0.0
        self._spike_history: List[float] = []   # mean rate per call

    # ── Core step ─────────────────────────────────────────────────────────

    def step(self, I_ext: np.ndarray) -> np.ndarray:
        """Advance one dt timestep. Returns bool fired mask."""
        noise = self._NOISE * self.rng.standard_normal(self.N).astype(np.float32)
        I = I_ext + noise + self._homeo_bias

        fired = self.v >= 30.0

        self.v[fired] = self.c[fired]
        self.u[fired] = self.u[fired] + self.d[fired]

        if np.any(fired):
            I += self.S[:, fired].sum(axis=1)

        # Two half-steps for numerical stability
        dv = (0.04 * self.v**2 + 5 * self.v + 140 - self.u + I) * self._DT
        self.v = np.clip(self.v + dv, -100, 30).astype(np.float32)
        dv2 = (0.04 * self.v**2 + 5 * self.v + 140 - self.u + I) * self._DT
        self.v = np.clip(self.v + dv2, -100, 30).astype(np.float32)
        self.u = (self.u + self._DT * self.a * (self.b * self.v - self.u)).astype(np.float32)

        # STDP traces
        decay_pre  = float(np.exp(-self._DT / self._TAU_PLUS))
        decay_post = float(np.exp(-self._DT / self._TAU_MINUS))
        self._trace_pre  = (self._trace_pre  * decay_pre).astype(np.float32)
        self._trace_post = (self._trace_post * decay_post).astype(np.float32)
        if np.any(fired):
            self._trace_pre[fired]  += 1.0
            self._trace_post[fired] += 1.0

        self._step_counter += 1
        if np.any(fired) and self._step_counter % self._STDP_EVERY == 0:
            self._apply_stdp(fired)
            if self._step_counter % (self._STDP_EVERY * 10) == 0:
                self._apply_synaptic_norm()

        self._apply_homeostasis(fired)

        self.total_steps += 1
        self.total_spikes += int(np.sum(fired))
        return fired

    def _apply_stdp(self, fired: np.ndarray):
        fired_idx = np.where(fired)[0]
        for post in fired_idx:
            active = self._trace_pre[:self.Ne] > 0.01
            if np.any(active):
                self.S[post, :self.Ne][active] += self._A_PLUS * self._trace_pre[:self.Ne][active]
        for pre in fired_idx:
            if pre >= self.Ne:
                continue
            active = self._trace_post > 0.01
            if np.any(active):
                self.S[active, pre] -= self._A_MINUS * self._trace_post[active]
        self.S[:, :self.Ne] = np.clip(self.S[:, :self.Ne], 0, self._EXC_W_MAX)
        self.S[:, self.Ne:] = np.clip(self.S[:, self.Ne:], self._INH_W_MAX, 0)

    def _apply_synaptic_norm(self):
        exc = self.S[:, :self.Ne]
        total = exc.sum(axis=1)
        mask = total > self._MAX_EXC_IN
        if np.any(mask):
            scale = self._MAX_EXC_IN / (total[mask] + 1e-10)
            exc[mask] *= scale[:, np.newaxis]
            self.S[:, :self.Ne] = exc

    def _apply_homeostasis(self, fired: np.ndarray):
        alpha = self._DT / self._HOMEO_TAU
        instant = fired.astype(np.float32) * (1000.0 / self._DT)
        self._rate_est   = (1 - alpha) * self._rate_est + alpha * instant
        error = self._TARGET_HZ - self._rate_est
        self._homeo_bias = np.clip(
            self._homeo_bias + self._HOMEO_RATE * error, -5.0, 5.0
        ).astype(np.float32)

    # ── Public interface: stimulate + read activation ────────────────────

    def stimulate(self, strength: float) -> float:
        """
        Inject a normalised signal (0-1) as µA current and run a short
        spiking window.  Returns activation (mean spike rate, 0-1 normalised).
        """
        amp_ua = strength * 2.0   # scale: 0-1 → 0-2 µA (safe range)
        I_ext  = np.zeros(self.N, dtype=np.float32)
        I_ext  += amp_ua * self._STIM_GAIN

        n_steps = int(self._WINDOW_MS / self._DT)
        spike_count = 0
        for _ in range(n_steps):
            fired = self.step(I_ext)
            spike_count += int(np.sum(fired))

        # Mean firing rate → normalised 0-1 (saturates at ~20 Hz)
        dur_s = self._WINDOW_MS / 1000.0
        mean_hz = spike_count / (self.N * dur_s + 1e-10)
        activation = float(np.tanh(mean_hz / 10.0))   # soft-cap, not clamp

        self._activation = activation
        self._spike_history.append(mean_hz)
        if len(self._spike_history) > 100:
            self._spike_history.pop(0)

        return activation

    @property
    def activation(self) -> float:
        return self._activation

    def get_weight_stats(self) -> Dict[str, float]:
        nnz = self.S[self.S != 0]
        return {
            "mean_exc_w": float(np.mean(self.S[:, :self.Ne][self.S[:, :self.Ne] > 0])) if np.any(self.S[:, :self.Ne] > 0) else 0.0,
            "mean_inh_w": float(np.mean(self.S[:, self.Ne:][self.S[:, self.Ne:] < 0])) if np.any(self.S[:, self.Ne:] < 0) else 0.0,
            "mean_firing_hz": float(np.mean(self._spike_history)) if self._spike_history else 0.0,
            "total_spikes": self.total_spikes,
        }


# ─── PropagationResult (same shape as NeuralColony) ───────────────────────

@dataclass
class PropagationResult:
    fired: Dict[str, float]
    activations: Dict[str, float]
    signals_sent: int
    cycle: int


# ─── ConnectionWeight (same as NeuralColony for evolution compatibility) ──

@dataclass
class _Connection:
    src: str
    dst: str
    weight: float
    base_weight: float
    mutation_count: int = 0


# ─── PlasticConnectome ─────────────────────────────────────────────────────

class PlasticConnectome:
    """
    Drop-in replacement for NeuralColony with STDP-based spiking substrates.

    The 8 Drosophila-mapped brain regions each own an Izhikevich mini-network.
    STDP continuously reshapes intra-region synapses based on co-firing
    patterns, while inter-region connections use (evolvable) FlyWire weights.

    Weight matrices are persisted per session so the brain accumulates
    experience across SableCore runs.

    Public API is identical to NeuralColony:
        stimulate(), stimulate_module(), propagate(),
        get_activation(), get_module_activation(),
        get_firing_modules(), compute_routing_bias(),
        mutate_connection(), mutate_threshold(),
        apply_evolution_pressure(), reset_to_baseline(),
        get_wiring_diagram(), get_stats()
    """

    _WEIGHTS_FILE = "plastic_connectome_weights.npz"
    _STATE_FILE   = "plastic_connectome_state.json"

    def __init__(self, data_dir: Optional[Path] = None, seed: int = 0):
        self.data_dir = Path(data_dir) if data_dir else Path("data")
        self._weights_path = self.data_dir / self._WEIGHTS_FILE
        self._state_path   = self.data_dir / self._STATE_FILE

        # ── Build spiking substrate per region ────────────────────────
        self.substrates: Dict[str, _RegionSubstrate] = {}
        seed_offset = 0
        for region, n in _REGION_NEURONS.items():
            self.substrates[region] = _RegionSubstrate(
                region=region, n_neurons=n, seed=seed + seed_offset
            )
            seed_offset += 1

        # ── Inter-region connections (FlyWire topology, evolvable) ────
        self.connections: Dict[str, _Connection] = {}
        for src, targets in _FLYWIRE_BASE_WEIGHTS.items():
            for dst, w in targets.items():
                key = f"{src}->{dst}"
                self.connections[key] = _Connection(src=src, dst=dst, weight=w, base_weight=w)

        # ── Stats ─────────────────────────────────────────────────────
        self._total_propagations = 0
        self._total_firings = 0
        self._generation = 0

        # ── Thresholds (per-region firing threshold for activation → fired) ──
        self._thresholds: Dict[str, float] = {r: 0.5 for r in _REGION_NEURONS}

        # ── Load saved STDP weights if available ──────────────────────
        self._load_state()

        logger.info(
            "🧠 PlasticConnectome ready: %d regions, %d total neurons, STDP active",
            len(self.substrates),
            sum(_REGION_NEURONS.values()),
        )

    # ── Public stimulation ────────────────────────────────────────────────

    def stimulate(self, region: str, strength: float = 1.0):
        """Inject a normalised signal into a brain region (0.0–1.0)."""
        if region in self.substrates:
            self.substrates[region].stimulate(max(0.0, min(1.0, strength)))

    def stimulate_module(self, module_name: str, strength: float = 1.0):
        """Stimulate by cognitive module name instead of region code."""
        region = MODULE_REGION_MAP.get(module_name)
        if region:
            self.stimulate(region, strength)

    # ── Propagation ───────────────────────────────────────────────────────

    def propagate(self, max_cycles: int = 3) -> List[PropagationResult]:
        """
        Propagate signals through the connectome.

        Each cycle:
        1. Determine which regions "fired" (activation ≥ threshold).
        2. Send weighted signals to downstream regions via FlyWire edges.
        3. Downstream regions run their spiking simulation on the received signal.

        Returns list[PropagationResult],  one per cycle, same shape as
        NeuralColony.propagate() so all existing SableCore code still works.
        """
        results = []

        for cycle in range(max_cycles):
            fired: Dict[str, float] = {}
            signals_sent = 0

            # Determine which regions fire this cycle
            for region, sub in self.substrates.items():
                if sub.activation >= self._thresholds[region]:
                    fired[region] = sub.activation

            if not fired:
                break

            logger.debug("🧠⚡ Plastic cycle %d: fired %s", cycle, list(fired.keys()))

            # Propagate to downstream regions
            for key, conn in self.connections.items():
                if conn.src in fired:
                    signal = fired[conn.src] * conn.weight
                    if signal > 0.01:
                        # This actually runs spiking dynamics in dst region
                        self.substrates[conn.dst].stimulate(signal)
                        signals_sent += 1

            self._total_firings += len(fired)
            self._total_propagations += 1

            results.append(PropagationResult(
                fired=fired,
                activations={r: s.activation for r, s in self.substrates.items()},
                signals_sent=signals_sent,
                cycle=cycle,
            ))

        # Persist weights periodically (every 10 propagations to reduce I/O)
        if self._total_propagations % 10 == 0:
            self._save_state()

        return results

    # ── Activation queries (same API as NeuralColony) ─────────────────────

    def get_activation(self, region: str) -> float:
        sub = self.substrates.get(region)
        return sub.activation if sub else 0.0

    def get_module_activation(self, module_name: str) -> float:
        region = MODULE_REGION_MAP.get(module_name)
        return self.get_activation(region) if region else 0.0

    def get_firing_modules(self, results: List[PropagationResult]) -> Dict[str, float]:
        merged: Dict[str, float] = {}
        for pr in results:
            for region, strength in pr.fired.items():
                module = REGION_MODULE_MAP.get(region, region)
                merged[module] = max(merged.get(module, 0.0), strength)
        return merged

    def compute_routing_bias(self, results: List[PropagationResult]) -> Dict[str, float]:
        biases: Dict[str, float] = {}
        fired = self.get_firing_modules(results)
        for module_name, region in MODULE_REGION_MAP.items():
            fire_score = fired.get(module_name, 0.0)
            residual   = self.substrates[region].activation
            biases[module_name] = min(1.0, fire_score * 0.7 + residual * 0.3)
        return biases

    # ── Evolution / Mutation (same API as NeuralColony) ───────────────────

    def mutate_connection(
        self,
        src: str,
        dst: str,
        delta: float,
        *,
        clamp: Tuple[float, float] = (0.0, 1.0),
    ) -> Optional[float]:
        key = f"{src}->{dst}"
        conn = self.connections.get(key)
        if not conn:
            return None
        conn.weight = max(clamp[0], min(clamp[1], conn.weight + delta))
        conn.mutation_count += 1
        return conn.weight

    def mutate_threshold(self, region: str, delta: float) -> Optional[float]:
        if region not in self._thresholds:
            return None
        self._thresholds[region] = max(0.1, min(0.95, self._thresholds[region] + delta))
        return self._thresholds[region]

    def apply_evolution_pressure(
        self,
        performance: Dict[str, float],
        learning_rate: float = 0.05,
    ):
        """
        Hebbian update on inter-region FlyWire connections.
        Identical to NeuralColony.apply_evolution_pressure().
        """
        self._generation += 1
        _mutated = []
        for key, conn in self.connections.items():
            src_module = REGION_MODULE_MAP.get(conn.src, "")
            dst_module = REGION_MODULE_MAP.get(conn.dst, "")
            src_perf = performance.get(src_module, 0.0)
            dst_perf = performance.get(dst_module, 0.0)
            hebbian = src_perf * dst_perf * learning_rate
            if src_perf < 0 and dst_perf < 0:
                hebbian -= abs(src_perf * dst_perf) * learning_rate * 0.5
            if abs(hebbian) > 0.001:
                conn.weight = max(0.0, min(1.0, conn.weight + hebbian))
                conn.mutation_count += 1
                _mutated.append(f"{conn.src}→{conn.dst}")
        if _mutated:
            logger.info(
                "🧬 Plastic Hebbian gen %d: %d connections mutated (%s)",
                self._generation, len(_mutated), ", ".join(_mutated[:6]),
            )
        self._save_state()

    def reset_to_baseline(self):
        """Reset all inter-region weights to FlyWire values and re-init substrates."""
        for conn in self.connections.values():
            conn.weight = conn.base_weight
            conn.mutation_count = 0
        for region, sub in self.substrates.items():
            n = _REGION_NEURONS[region]
            seed = list(_REGION_NEURONS.keys()).index(region)
            new_sub = _RegionSubstrate(region=region, n_neurons=n, seed=seed)
            self.substrates[region] = new_sub
        self._thresholds = {r: 0.5 for r in _REGION_NEURONS}
        self._generation = 0
        self._save_state()
        logger.info("🧬 PlasticConnectome reset to Drosophila baseline + fresh STDP substrates")

    # ── Introspection ─────────────────────────────────────────────────────

    def get_wiring_diagram(self) -> Dict[str, Any]:
        nodes = []
        for region, sub in self.substrates.items():
            ws = sub.get_weight_stats()
            nodes.append({
                "id": region,
                "module": REGION_MODULE_MAP.get(region, ""),
                "activation": round(sub.activation, 3),
                "threshold": round(self._thresholds[region], 3),
                "neurons": _REGION_NEURONS[region],
                "mean_firing_hz": round(ws["mean_firing_hz"], 2),
                "mean_exc_weight": round(ws["mean_exc_w"], 4),
                "total_spikes": ws["total_spikes"],
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
            "total_neurons": sum(_REGION_NEURONS.values()),
            "source": "FlyWire FAFB v783 + Izhikevich STDP substrates (CL1-validated)",
        }

    def get_stats(self) -> Dict[str, Any]:
        mutated = sum(1 for c in self.connections.values() if c.mutation_count > 0)
        max_drift = max(
            abs(c.weight - c.base_weight) for c in self.connections.values()
        )
        top_regions = sorted(
            self.substrates.items(),
            key=lambda x: x[1].total_spikes,
            reverse=True,
        )[:3]
        return {
            "generation": self._generation,
            "connections": len(self.connections),
            "mutated_connections": mutated,
            "max_drift": round(max_drift, 4),
            "total_propagations": self._total_propagations,
            "total_firings": self._total_firings,
            "total_neurons": sum(_REGION_NEURONS.values()),
            "stdp_active": True,
            "top_regions": [
                {
                    "region": r,
                    "module": REGION_MODULE_MAP.get(r, ""),
                    "total_spikes": s.total_spikes,
                    "mean_hz": round(float(np.mean(s._spike_history)) if s._spike_history else 0.0, 2),
                }
                for r, s in top_regions
            ],
        }

    def get_stdp_divergence(self) -> Dict[str, Dict[str, float]]:
        """
        Return how much each region's weight matrix has diverged from init.
        Useful for tracking what the brain has 'learned' across sessions.
        """
        result = {}
        for region, sub in self.substrates.items():
            nnz = sub.S[sub.S != 0]
            result[region] = {
                "mean_weight": float(np.mean(np.abs(nnz))) if len(nnz) > 0 else 0.0,
                "std_weight":  float(np.std(np.abs(nnz)))  if len(nnz) > 0 else 0.0,
                "n_nonzero":   int(np.sum(sub.S != 0)),
                "total_spikes": sub.total_spikes,
            }
        return result

    # ── Persistence ───────────────────────────────────────────────────────

    def _save_state(self):
        """Save STDP weight matrices (numpy) + connection weights (json)."""
        try:
            self.data_dir.mkdir(parents=True, exist_ok=True)

            # Save STDP weight matrices as .npz
            arrays = {f"S_{region}": sub.S for region, sub in self.substrates.items()}
            arrays.update({f"v_{region}": sub.v for region, sub in self.substrates.items()})
            np.savez_compressed(str(self._weights_path), **arrays)

            # Save scalar state as json
            state = {
                "generation": self._generation,
                "total_propagations": self._total_propagations,
                "total_firings": self._total_firings,
                "saved_at": time.time(),
                "thresholds": self._thresholds,
                "connections": {
                    k: {"weight": c.weight, "mutation_count": c.mutation_count}
                    for k, c in self.connections.items()
                },
                "substrate_stats": {
                    r: {"total_spikes": s.total_spikes, "total_steps": s.total_steps}
                    for r, s in self.substrates.items()
                },
            }
            self._state_path.write_text(json.dumps(state, indent=2))

        except Exception as e:
            logger.warning("PlasticConnectome: failed to save state: %s", e)

    def _load_state(self):
        """Load persisted STDP weights and connection state."""
        # Load connection / scalar state
        if self._state_path.exists():
            try:
                state = json.loads(self._state_path.read_text())
                self._generation          = state.get("generation", 0)
                self._total_propagations  = state.get("total_propagations", 0)
                self._total_firings       = state.get("total_firings", 0)
                saved_thr = state.get("thresholds", {})
                for r, t in saved_thr.items():
                    if r in self._thresholds:
                        self._thresholds[r] = t
                for k, data in state.get("connections", {}).items():
                    if k in self.connections:
                        self.connections[k].weight = data["weight"]
                        self.connections[k].mutation_count = data.get("mutation_count", 0)
                for r, stats in state.get("substrate_stats", {}).items():
                    if r in self.substrates:
                        self.substrates[r].total_spikes = stats.get("total_spikes", 0)
                        self.substrates[r].total_steps  = stats.get("total_steps", 0)
                logger.info(
                    "🧠 PlasticConnectome loaded: gen=%d, propagations=%d",
                    self._generation, self._total_propagations,
                )
            except Exception as e:
                logger.warning("PlasticConnectome: failed to load state: %s", e)

        # Load STDP weight matrices
        if self._weights_path.exists():
            try:
                data = np.load(str(self._weights_path))
                for region, sub in self.substrates.items():
                    key_S = f"S_{region}"
                    key_v = f"v_{region}"
                    if key_S in data and data[key_S].shape == sub.S.shape:
                        sub.S = data[key_S].copy()
                    if key_v in data and data[key_v].shape == sub.v.shape:
                        sub.v = data[key_v].copy()
                logger.info("🧠 PlasticConnectome STDP weights restored from disk")
            except Exception as e:
                logger.warning("PlasticConnectome: failed to load STDP weights: %s", e)
