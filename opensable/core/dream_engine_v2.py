"""
Dream Engine v2 — 4-Phase Knowledge Graph Refinement

Upgrades over v1 (creative replay):
  Phase 1: Duplicate Merging     — FAISS similarity ≥0.93, merge entities
  Phase 2: Description Enrichment — LLM enriches thin entity descriptions
  Phase 3: Relationship Inference — discover missing links between co-occurring entities
  Phase 4: Confidence Decay       — age-based decay on stale relations

Plus: Rejection cache (7-day), hub diversity caps, batch rotation,
anti-contamination, Ollama busy check, dream journal logging.

Works alongside original DreamEngine (creative dreams) — this handles
structured graph maintenance.
"""
from __future__ import annotations

import json
import logging
import random
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from opensable.core.knowledge_graph_v2 import KnowledgeGraphV2

logger = logging.getLogger(__name__)


@dataclass
class DreamPhaseResult:
    phase: str
    actions_taken: int = 0
    details: List[str] = field(default_factory=list)
    duration_s: float = 0.0


@dataclass
class DreamCycleResult:
    cycle_id: int = 0
    timestamp: str = ""
    phases: List[DreamPhaseResult] = field(default_factory=list)
    total_actions: int = 0
    duration_s: float = 0.0


class DreamEngineV2:
    """
    4-phase nightly knowledge graph refinement daemon.
    Runs during idle periods to maintain and improve the knowledge graph.
    """

    MERGE_THRESHOLD = 0.93
    THIN_DESCRIPTION_CHARS = 60
    ENRICHMENT_BATCH = 10
    INFERENCE_BATCH = 15
    MAX_JOURNAL = 100
    REJECTION_CACHE_DAYS = 7

    def __init__(self, data_dir: Path, kg: "KnowledgeGraphV2"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.state_file = self.data_dir / "dream_engine_v2_state.json"
        self.kg = kg
        self._llm = None

        self.total_cycles = 0
        self.total_merges = 0
        self.total_enrichments = 0
        self.total_inferences = 0
        self.total_decays = 0
        self.journal: List[Dict[str, Any]] = []
        self._rejection_cache: Dict[str, float] = {}   # key → timestamp
        self._last_batch_offset = 0

        self._load_state()

    def set_llm(self, llm):
        self._llm = llm

    # ── Main Entry Point ──────────────────────────────────────────────

    async def run_cycle(self, llm=None) -> DreamCycleResult:
        """
        Run a full 4-phase dream cycle.
        Should be called during idle periods (e.g., no user messages for 5+ min).
        """
        if llm:
            self._llm = llm

        cycle = DreamCycleResult(
            cycle_id=self.total_cycles + 1,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
        start = time.time()

        # Check if Ollama is busy (if local)
        if await self._is_llm_busy():
            logger.info("[DreamV2] LLM busy, deferring cycle")
            return cycle

        # Clean expired rejection cache entries
        self._clean_rejection_cache()

        # Phase 1: Merge duplicates
        p1 = await self._phase_merge_duplicates()
        cycle.phases.append(p1)

        # Phase 2: Enrich thin descriptions
        p2 = await self._phase_enrich_descriptions()
        cycle.phases.append(p2)

        # Phase 3: Infer missing relationships
        p3 = await self._phase_infer_relationships()
        cycle.phases.append(p3)

        # Phase 4: Confidence decay
        p4 = self._phase_confidence_decay()
        cycle.phases.append(p4)

        cycle.total_actions = sum(p.actions_taken for p in cycle.phases)
        cycle.duration_s = time.time() - start
        self.total_cycles += 1

        # Journal
        self.journal.append(asdict(cycle))
        if len(self.journal) > self.MAX_JOURNAL:
            self.journal = self.journal[-self.MAX_JOURNAL:]

        self.kg.save()
        self._save_state()

        logger.info(f"[DreamV2] Cycle {cycle.cycle_id} complete: "
                     f"{cycle.total_actions} actions in {cycle.duration_s:.1f}s")
        return cycle

    # ── Phase 1: Duplicate Merging ────────────────────────────────────

    async def _phase_merge_duplicates(self) -> DreamPhaseResult:
        result = DreamPhaseResult(phase="merge_duplicates")
        start = time.time()

        dupes = self.kg.find_duplicates(threshold=self.MERGE_THRESHOLD)
        for keep_id, remove_id, score in dupes[:10]:
            cache_key = f"merge:{keep_id}:{remove_id}"
            if cache_key in self._rejection_cache:
                continue

            keep = self.kg.entities.get(keep_id)
            remove = self.kg.entities.get(remove_id)
            if not keep or not remove:
                continue

            self.kg.merge_entities(keep_id, remove_id)
            result.actions_taken += 1
            result.details.append(f"Merged '{remove.name}' into '{keep.name}' (sim={score:.3f})")
            self.total_merges += 1

        result.duration_s = time.time() - start
        return result

    # ── Phase 2: Description Enrichment ───────────────────────────────

    async def _phase_enrich_descriptions(self) -> DreamPhaseResult:
        result = DreamPhaseResult(phase="enrich_descriptions")
        start = time.time()

        if not self._llm:
            result.details.append("No LLM available, skipped")
            result.duration_s = time.time() - start
            return result

        # Find thin entities (short descriptions), batch rotate
        thin = [e for e in self.kg.entities.values()
                if len(e.description) < self.THIN_DESCRIPTION_CHARS]
        if not thin:
            result.duration_s = time.time() - start
            return result

        # Rotate batch offset for coverage
        batch_start = self._last_batch_offset % max(1, len(thin))
        batch = thin[batch_start:batch_start + self.ENRICHMENT_BATCH]
        self._last_batch_offset += self.ENRICHMENT_BATCH

        for ent in batch:
            cache_key = f"enrich:{ent.entity_id}"
            if cache_key in self._rejection_cache:
                continue

            # Get context from relationships
            rels = self.kg.get_entity_relations(ent.name)
            context = ", ".join(f"{r['relation']} {r['entity']}" for r in rels[:5])

            prompt = (
                f"Write a concise 1-2 sentence description of '{ent.name}' "
                f"(type: {ent.entity_type})."
            )
            if context:
                prompt += f"\nKnown relationships: {context}"
            if ent.description:
                prompt += f"\nCurrent (thin) description: {ent.description}"
            prompt += "\nRespond with ONLY the description, nothing else."

            try:
                resp = await self._llm.invoke_with_tools(
                    [{"role": "user", "content": prompt}], []
                )
                new_desc = resp.get("text", "").strip()
                if len(new_desc) >= 30 and len(new_desc) > len(ent.description):
                    ent.description = new_desc[:500]
                    ent.last_updated = datetime.now(timezone.utc).isoformat()
                    result.actions_taken += 1
                    result.details.append(f"Enriched '{ent.name}'")
                    self.total_enrichments += 1
                else:
                    self._rejection_cache[cache_key] = time.time()
            except Exception as e:
                logger.debug(f"[DreamV2] Enrichment failed for {ent.name}: {e}")
                break  # LLM might be overloaded

        result.duration_s = time.time() - start
        return result

    # ── Phase 3: Relationship Inference ───────────────────────────────

    async def _phase_infer_relationships(self) -> DreamPhaseResult:
        result = DreamPhaseResult(phase="infer_relationships")
        start = time.time()

        if not self._llm or len(self.kg.entities) < 5:
            result.duration_s = time.time() - start
            return result

        # Find entity pairs that co-occur (share relationships) but have no direct link
        candidates = self._find_inference_candidates()
        random.shuffle(candidates)

        from opensable.core.knowledge_graph_v2 import RELATION_TYPES
        relation_list = ", ".join(sorted(RELATION_TYPES.keys())[:25])

        for src_name, tgt_name in candidates[:self.INFERENCE_BATCH]:
            cache_key = f"infer:{src_name}:{tgt_name}"
            if cache_key in self._rejection_cache:
                continue

            prompt = (
                f"Do '{src_name}' and '{tgt_name}' have a direct relationship?\n"
                f"Available relation types: {relation_list}\n"
                f"If yes, respond with JSON: {{\"relation\": \"type\", \"description\": \"...\"}}\n"
                f"If no clear relationship, respond with: {{\"relation\": null}}\n"
                f"Be conservative — only infer if clearly implied."
            )

            try:
                resp = await self._llm.invoke_with_tools(
                    [{"role": "user", "content": prompt}], []
                )
                raw = resp.get("text", "")
                s = raw.find("{")
                e = raw.rfind("}") + 1
                if s >= 0 and e > s:
                    data = json.loads(raw[s:e])
                    rel_type = data.get("relation")
                    if rel_type and rel_type != "null":
                        added = self.kg.add_relationship(
                            src_name, tgt_name, rel_type,
                            description=data.get("description", ""),
                            confidence=0.6,
                            evidence=["dream_inference"],
                        )
                        if added:
                            result.actions_taken += 1
                            result.details.append(f"Inferred: {src_name} --[{rel_type}]--> {tgt_name}")
                            self.total_inferences += 1
                        else:
                            self._rejection_cache[cache_key] = time.time()
                    else:
                        self._rejection_cache[cache_key] = time.time()
            except Exception as e:
                logger.debug(f"[DreamV2] Inference failed: {e}")
                break

        result.duration_s = time.time() - start
        return result

    def _find_inference_candidates(self) -> List[tuple]:
        """Find entity pairs connected via shared neighbors but not directly linked."""
        if self.kg.graph is None:
            return []

        candidates = []
        direct = set()
        for rel in self.kg.relationships:
            direct.add((rel.source_id, rel.target_id))

        entities_list = list(self.kg.entities.values())
        if len(entities_list) > 200:
            entities_list = random.sample(entities_list, 200)

        # Find 2-hop connections without direct link
        for ent in entities_list:
            if ent.entity_id not in self.kg.graph:
                continue
            neighbors = set(self.kg.graph.neighbors(ent.entity_id))
            for n1 in neighbors:
                if n1 not in self.kg.graph:
                    continue
                for n2 in self.kg.graph.neighbors(n1):
                    if (n2 != ent.entity_id and
                            (ent.entity_id, n2) not in direct and
                            (n2, ent.entity_id) not in direct):
                        e1 = self.kg.entities.get(ent.entity_id)
                        e2 = self.kg.entities.get(n2)
                        if e1 and e2:
                            candidates.append((e1.name, e2.name))
                            if len(candidates) >= 50:
                                return candidates
        return candidates

    # ── Phase 4: Confidence Decay ─────────────────────────────────────

    def _phase_confidence_decay(self) -> DreamPhaseResult:
        result = DreamPhaseResult(phase="confidence_decay")
        start = time.time()
        decayed = self.kg.apply_confidence_decay()
        result.actions_taken = decayed
        self.total_decays += decayed
        if decayed > 0:
            result.details.append(f"Decayed confidence on {decayed} stale relations")
        result.duration_s = time.time() - start
        return result

    # ── Helpers ────────────────────────────────────────────────────────

    async def _is_llm_busy(self) -> bool:
        """Check if local Ollama is busy serving a user request."""
        try:
            import aiohttp
            async with aiohttp.ClientSession() as session:
                async with session.get("http://127.0.0.1:11434/api/ps", timeout=aiohttp.ClientTimeout(total=2)) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        models = data.get("models", [])
                        return len(models) > 0
        except Exception:
            pass
        return False

    def _clean_rejection_cache(self):
        cutoff = time.time() - (self.REJECTION_CACHE_DAYS * 86400)
        self._rejection_cache = {k: v for k, v in self._rejection_cache.items() if v > cutoff}

    def get_journal(self, last_n: int = 10) -> List[Dict]:
        return self.journal[-last_n:]

    def get_stats(self) -> Dict[str, Any]:
        return {
            "total_cycles": self.total_cycles,
            "total_merges": self.total_merges,
            "total_enrichments": self.total_enrichments,
            "total_inferences": self.total_inferences,
            "total_decays": self.total_decays,
            "rejection_cache_size": len(self._rejection_cache),
            "journal_size": len(self.journal),
        }

    # ── Persistence ───────────────────────────────────────────────────

    def _save_state(self):
        try:
            state = {
                "total_cycles": self.total_cycles,
                "total_merges": self.total_merges,
                "total_enrichments": self.total_enrichments,
                "total_inferences": self.total_inferences,
                "total_decays": self.total_decays,
                "last_batch_offset": self._last_batch_offset,
                "rejection_cache": self._rejection_cache,
                "journal": self.journal[-self.MAX_JOURNAL:],
            }
            self.state_file.write_text(json.dumps(state, indent=2, default=str), encoding="utf-8")
        except Exception as e:
            logger.error(f"[DreamV2] Save failed: {e}")

    def _load_state(self):
        if not self.state_file.exists():
            return
        try:
            state = json.loads(self.state_file.read_text(encoding="utf-8"))
            self.total_cycles = state.get("total_cycles", 0)
            self.total_merges = state.get("total_merges", 0)
            self.total_enrichments = state.get("total_enrichments", 0)
            self.total_inferences = state.get("total_inferences", 0)
            self.total_decays = state.get("total_decays", 0)
            self._last_batch_offset = state.get("last_batch_offset", 0)
            self._rejection_cache = state.get("rejection_cache", {})
            self.journal = state.get("journal", [])
        except Exception as e:
            logger.error(f"[DreamV2] Load failed: {e}")
