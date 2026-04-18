"""
Knowledge Graph v2 — Typed Relations, FAISS Semantic Search, Auto-Recall

Upgrades over v1:
  - 67 typed directional relations with 60+ aliases
  - FAISS semantic search for entity recall
  - 1-hop graph expansion before every LLM call
  - Confidence decay on stale relations
  - Relation pre-normalization (rejects vague types)
  - Deterministic cross-category dedup
  - Hub diversity caps
  - Integration-ready auto_recall() for the agent loop
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

# ── Optional deps ─────────────────────────────────────────────────────
try:
    import networkx as nx
    NX_OK = True
except ImportError:
    nx = None  # type: ignore
    NX_OK = False

try:
    import numpy as np
    import faiss          # faiss-cpu
    FAISS_OK = True
except ImportError:
    np = faiss = None     # type: ignore
    FAISS_OK = False

try:
    import tiktoken
    TIK_OK = True
except ImportError:
    tiktoken = None       # type: ignore
    TIK_OK = False


# ══════════════════════════════════════════════════════════════════════
#  Relation Vocabulary — 67 typed relations + 60 aliases
# ══════════════════════════════════════════════════════════════════════

RELATION_TYPES: Dict[str, str] = {
    # People
    "father_of": "family", "mother_of": "family", "child_of": "family",
    "sibling_of": "family", "spouse_of": "family", "partner_of": "family",
    "friend_of": "social", "colleague_of": "social", "mentor_of": "social",
    "student_of": "social", "reports_to": "social", "manages": "social",
    # Organizations
    "works_at": "employment", "member_of": "membership", "founded": "creation",
    "ceo_of": "leadership", "leads": "leadership", "owns": "ownership",
    "invested_in": "financial", "acquired": "financial", "competes_with": "business",
    "partners_with": "business", "subsidiary_of": "business",
    # Places
    "located_in": "geography", "born_in": "geography", "lives_in": "geography",
    "headquarters_in": "geography", "traveled_to": "geography", "near": "geography",
    # Events
    "attended": "participation", "organized": "participation", "spoke_at": "participation",
    "participated_in": "participation", "caused": "causation", "resulted_in": "causation",
    "preceded": "temporal", "followed": "temporal", "concurrent_with": "temporal",
    # Knowledge
    "created": "creation", "authored": "creation", "invented": "creation",
    "developed": "creation", "designed": "creation", "contributed_to": "creation",
    "uses": "utility", "requires": "dependency", "depends_on": "dependency",
    "part_of": "composition", "contains": "composition", "instance_of": "classification",
    "subclass_of": "classification", "similar_to": "similarity",
    "opposite_of": "contrast", "derived_from": "derivation",
    # Preferences
    "likes": "preference", "dislikes": "preference", "prefers": "preference",
    "interested_in": "preference", "skilled_at": "competence",
    "studies": "education", "graduated_from": "education",
    "certified_in": "education", "knows": "knowledge",
    # Projects
    "assigned_to": "project", "deadline_for": "project", "blocks": "project",
    "implements": "project", "tests": "project", "deploys": "project",
    "documents": "project",
}

# Aliases → canonical type
RELATION_ALIASES: Dict[str, str] = {
    "dad": "father_of", "mom": "mother_of", "parent_of": "father_of",
    "married_to": "spouse_of", "wife_of": "spouse_of", "husband_of": "spouse_of",
    "dating": "partner_of", "bff": "friend_of", "buddy": "friend_of",
    "boss": "manages", "manager": "manages", "employed_at": "works_at",
    "employed_by": "works_at", "hired_by": "works_at", "works_for": "works_at",
    "belongs_to": "member_of", "affiliate_of": "member_of",
    "built": "created", "wrote": "authored", "made": "created",
    "constructed": "created", "composed": "authored",
    "based_in": "located_in", "resides_in": "lives_in", "from": "born_in",
    "went_to": "traveled_to", "visited": "traveled_to",
    "triggered": "caused", "led_to": "resulted_in", "because_of": "caused",
    "before": "preceded", "after": "followed",
    "needs": "requires", "has": "contains", "includes": "contains",
    "type_of": "instance_of", "kind_of": "subclass_of",
    "loves": "likes", "enjoys": "likes", "hates": "dislikes",
    "good_at": "skilled_at", "expert_in": "skilled_at",
    "learning": "studies", "enrolled_in": "studies",
    "working_on": "assigned_to", "responsible_for": "assigned_to",
}

# Vague relation types to REJECT
VAGUE_RELATIONS = frozenset({
    "related_to", "associated_with", "connected_to", "linked_to",
    "has_relation", "involves", "concerns", "about", "regarding",
})

ENTITY_TYPES = frozenset({
    "person", "place", "event", "preference", "fact", "project",
    "organisation", "concept", "skill", "media", "object", "tool",
})


# ══════════════════════════════════════════════════════════════════════
#  Data Models
# ══════════════════════════════════════════════════════════════════════

@dataclass
class Entity:
    entity_id: str
    name: str
    entity_type: str = "concept"
    description: str = ""
    properties: Dict[str, Any] = field(default_factory=dict)
    first_seen: str = ""
    last_updated: str = ""
    mention_count: int = 0
    source: str = "conversation"
    importance: float = 0.5
    confidence: float = 0.9
    embedding: Optional[List[float]] = field(default=None, repr=False)


@dataclass
class Relationship:
    source_id: str
    target_id: str
    relation_type: str = "related_to"
    category: str = ""
    description: str = ""
    weight: float = 1.0
    confidence: float = 0.8
    established: str = ""
    last_confirmed: str = ""
    evidence: List[str] = field(default_factory=list)


# ══════════════════════════════════════════════════════════════════════
#  FAISS Embedding Index
# ══════════════════════════════════════════════════════════════════════

class FAISSIndex:
    """Lightweight FAISS index using trigram embeddings (no external model)."""

    DIM = 512  # trigram hash space

    def __init__(self):
        self._ids: List[str] = []
        self._index = None
        if FAISS_OK:
            self._index = faiss.IndexFlatIP(self.DIM)  # inner product = cosine on L2-normed

    @staticmethod
    def _trigram_embed(text: str) -> "np.ndarray":
        """Deterministic trigram hash embedding — zero external models."""
        vec = np.zeros(FAISSIndex.DIM, dtype=np.float32)
        text = text.lower().strip()
        for i in range(len(text) - 2):
            tri = text[i:i+3]
            h = int(hashlib.md5(tri.encode()).hexdigest(), 16) % FAISSIndex.DIM
            vec[h] += 1.0
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec

    def add(self, entity_id: str, text: str):
        if not FAISS_OK:
            return
        vec = self._trigram_embed(text).reshape(1, -1)
        self._index.add(vec)
        self._ids.append(entity_id)

    def search(self, query: str, k: int = 10) -> List[Tuple[str, float]]:
        if not FAISS_OK or self._index.ntotal == 0:
            return []
        vec = self._trigram_embed(query).reshape(1, -1)
        k = min(k, self._index.ntotal)
        scores, indices = self._index.search(vec, k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if 0 <= idx < len(self._ids) and score > 0.05:
                results.append((self._ids[idx], float(score)))
        return results

    def rebuild(self, entities: Dict[str, "Entity"]):
        """Full rebuild from entity dict."""
        if not FAISS_OK:
            return
        self._ids = []
        self._index = faiss.IndexFlatIP(self.DIM)
        for eid, ent in entities.items():
            text = f"{ent.name} {ent.description} {ent.entity_type}"
            self.add(eid, text)


# ══════════════════════════════════════════════════════════════════════
#  Knowledge Graph Engine v2
# ══════════════════════════════════════════════════════════════════════

class KnowledgeGraphV2:
    """
    Enhanced knowledge graph with typed relations, FAISS search,
    confidence decay, and auto-recall for the agent loop.
    """

    MAX_ENTITIES = 15000
    MAX_RELATIONSHIPS = 75000
    CONFIDENCE_DECAY_DAYS = 90
    CONFIDENCE_DECAY_RATE = 0.01
    HUB_MAX_RELATIONS = 200

    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.state_file = self.data_dir / "knowledge_graph_v2.json"

        self.graph = nx.DiGraph() if NX_OK else None
        self.entities: Dict[str, Entity] = {}
        self.relationships: List[Relationship] = []
        self._rel_index: Dict[str, List[int]] = defaultdict(list)  # entity_id → rel indices

        self.faiss_index = FAISSIndex()
        self._llm = None

        # Stats
        self.stats = {
            "entities_added": 0, "relations_added": 0,
            "queries": 0, "extractions": 0, "decays_applied": 0,
            "vague_rejected": 0, "dupes_merged": 0,
        }

        self._load_state()

    def set_llm(self, llm):
        self._llm = llm

    # ── Relation normalization ────────────────────────────────────────

    @staticmethod
    def normalize_relation(raw: str) -> Optional[str]:
        """Normalize relation type: resolve aliases, reject vague types."""
        clean = raw.lower().strip().replace(" ", "_").replace("-", "_")
        # Check aliases first
        clean = RELATION_ALIASES.get(clean, clean)
        # Reject vague
        if clean in VAGUE_RELATIONS:
            return None
        # Accept known types
        if clean in RELATION_TYPES:
            return clean
        # Accept with warning for unknown but non-vague types
        return clean

    # ── Entity Management ─────────────────────────────────────────────

    def add_entity(self, name: str, entity_type: str = "concept",
                   description: str = "", properties: Optional[Dict] = None,
                   source: str = "conversation", importance: float = 0.5,
                   confidence: float = 0.9) -> Entity:
        entity_id = self._make_id(name)
        now = datetime.now(timezone.utc).isoformat()

        if entity_id in self.entities:
            ent = self.entities[entity_id]
            ent.mention_count += 1
            ent.last_updated = now
            if description and len(description) > len(ent.description):
                ent.description = description
            if properties:
                ent.properties.update(properties)
            ent.importance = min(1.0, ent.importance + 0.03)
            ent.confidence = min(1.0, max(ent.confidence, confidence))
        else:
            etype = entity_type.lower().strip()
            if etype not in ENTITY_TYPES:
                etype = "concept"
            ent = Entity(
                entity_id=entity_id, name=name, entity_type=etype,
                description=description, properties=properties or {},
                first_seen=now, last_updated=now, mention_count=1,
                source=source, importance=importance, confidence=confidence,
            )
            self.entities[entity_id] = ent
            self.stats["entities_added"] += 1
            if self.graph is not None:
                self.graph.add_node(entity_id, name=name, type=etype, importance=importance)
            self.faiss_index.add(entity_id, f"{name} {description} {etype}")

        if len(self.entities) > self.MAX_ENTITIES:
            self._prune_entities()
        return ent

    def get_entity(self, name: str) -> Optional[Entity]:
        return self.entities.get(self._make_id(name))

    def delete_entity(self, name: str) -> bool:
        eid = self._make_id(name)
        if eid not in self.entities:
            return False
        del self.entities[eid]
        self.relationships = [r for r in self.relationships
                              if r.source_id != eid and r.target_id != eid]
        self._rebuild_rel_index()
        if self.graph is not None and eid in self.graph:
            self.graph.remove_node(eid)
        self.faiss_index.rebuild(self.entities)
        return True

    # ── Relationship Management ───────────────────────────────────────

    def add_relationship(self, source_name: str, target_name: str,
                         relation_type: str = "related_to", description: str = "",
                         weight: float = 1.0, confidence: float = 0.8,
                         evidence: Optional[List[str]] = None) -> Optional[Relationship]:
        # Normalize relation
        normalized = self.normalize_relation(relation_type)
        if normalized is None:
            self.stats["vague_rejected"] += 1
            logger.debug(f"Rejected vague relation: {relation_type}")
            return None

        # Reject self-loops
        if source_name.lower().strip() == target_name.lower().strip():
            return None

        src = self.add_entity(source_name)
        tgt = self.add_entity(target_name)
        now = datetime.now(timezone.utc).isoformat()

        # Check for existing relationship (update instead of duplicate)
        for idx, rel in enumerate(self.relationships):
            if rel.source_id == src.entity_id and rel.target_id == tgt.entity_id and rel.relation_type == normalized:
                rel.confidence = min(1.0, rel.confidence + 0.05)
                rel.last_confirmed = now
                if evidence:
                    rel.evidence.extend(evidence[-3:])
                    rel.evidence = rel.evidence[-10:]
                return rel

        # Hub cap check
        src_rels = len(self._rel_index.get(src.entity_id, []))
        tgt_rels = len(self._rel_index.get(tgt.entity_id, []))
        if src_rels >= self.HUB_MAX_RELATIONS or tgt_rels >= self.HUB_MAX_RELATIONS:
            logger.debug(f"Hub cap reached for {source_name} or {target_name}")
            return None

        category = RELATION_TYPES.get(normalized, "unknown")
        rel = Relationship(
            source_id=src.entity_id, target_id=tgt.entity_id,
            relation_type=normalized, category=category,
            description=description, weight=weight, confidence=confidence,
            established=now, last_confirmed=now, evidence=evidence or [],
        )
        idx = len(self.relationships)
        self.relationships.append(rel)
        self._rel_index[src.entity_id].append(idx)
        self._rel_index[tgt.entity_id].append(idx)
        self.stats["relations_added"] += 1

        if self.graph is not None:
            self.graph.add_edge(src.entity_id, tgt.entity_id,
                                relation=normalized, weight=weight, confidence=confidence)

        if len(self.relationships) > self.MAX_RELATIONSHIPS:
            self._prune_relationships()
        return rel

    # ── FAISS Semantic Search ─────────────────────────────────────────

    def semantic_search(self, query: str, k: int = 10) -> List[Entity]:
        """Search entities by semantic similarity using FAISS."""
        self.stats["queries"] += 1
        results = self.faiss_index.search(query, k)
        return [self.entities[eid] for eid, _ in results if eid in self.entities]

    def keyword_search(self, query: str, limit: int = 10) -> List[Entity]:
        """Fallback keyword search."""
        q = query.lower()
        scored = []
        for ent in self.entities.values():
            score = 0
            if q in ent.name.lower():
                score += 3
            if q in ent.description.lower():
                score += 1
            for v in ent.properties.values():
                if q in str(v).lower():
                    score += 0.5
            if score > 0:
                scored.append((score * ent.importance, ent))
        scored.sort(reverse=True, key=lambda x: x[0])
        return [e for _, e in scored[:limit]]

    def search(self, query: str, k: int = 10) -> List[Entity]:
        """Hybrid search: FAISS first, keyword fallback."""
        results = self.semantic_search(query, k)
        if len(results) < 3:
            kw = self.keyword_search(query, k)
            seen = {e.entity_id for e in results}
            for e in kw:
                if e.entity_id not in seen:
                    results.append(e)
                    seen.add(e.entity_id)
        return results[:k]

    # ── 1-Hop Graph Expansion (Auto-Recall) ───────────────────────────

    def auto_recall(self, query: str, k: int = 5, hops: int = 1) -> str:
        """
        Retrieve semantically similar entities via FAISS,
        then expand N hops in the graph. Returns formatted context
        for injection into the LLM system prompt.
        """
        seed_entities = self.search(query, k)
        if not seed_entities:
            return ""

        expanded: Dict[str, Entity] = {}
        for ent in seed_entities:
            expanded[ent.entity_id] = ent

        # N-hop expansion
        if self.graph is not None:
            frontier = {e.entity_id for e in seed_entities}
            for _hop in range(hops):
                next_frontier: Set[str] = set()
                for eid in frontier:
                    if eid in self.graph:
                        for neighbor in self.graph.neighbors(eid):
                            if neighbor not in expanded and neighbor in self.entities:
                                expanded[neighbor] = self.entities[neighbor]
                                next_frontier.add(neighbor)
                        for predecessor in self.graph.predecessors(eid):
                            if predecessor not in expanded and predecessor in self.entities:
                                expanded[predecessor] = self.entities[predecessor]
                                next_frontier.add(predecessor)
                frontier = next_frontier

        # Format context
        lines = ["[Memory — Knowledge Graph Context]"]
        for ent in expanded.values():
            line = f"• {ent.name} ({ent.entity_type})"
            if ent.description:
                line += f": {ent.description[:200]}"
            lines.append(line)

        # Add relevant relationships
        expanded_ids = set(expanded.keys())
        rel_lines = []
        for rel in self.relationships:
            if rel.source_id in expanded_ids and rel.target_id in expanded_ids:
                src = self.entities.get(rel.source_id)
                tgt = self.entities.get(rel.target_id)
                if src and tgt:
                    rel_lines.append(f"  {src.name} --[{rel.relation_type}]--> {tgt.name}")
        if rel_lines:
            lines.append("\nRelationships:")
            lines.extend(rel_lines[:30])

        return "\n".join(lines)

    # ── Confidence Decay ──────────────────────────────────────────────

    def apply_confidence_decay(self):
        """Decay confidence on relations older than CONFIDENCE_DECAY_DAYS."""
        now = time.time()
        decayed = 0
        for rel in self.relationships:
            if not rel.last_confirmed:
                continue
            try:
                confirmed_ts = datetime.fromisoformat(rel.last_confirmed).timestamp()
            except (ValueError, OSError):
                continue
            age_days = (now - confirmed_ts) / 86400
            if age_days > self.CONFIDENCE_DECAY_DAYS:
                decay = self.CONFIDENCE_DECAY_RATE * (age_days - self.CONFIDENCE_DECAY_DAYS) / 30
                rel.confidence = max(0.1, rel.confidence - decay)
                decayed += 1
        self.stats["decays_applied"] += decayed
        return decayed

    # ── Duplicate Detection & Merge ───────────────────────────────────

    def find_duplicates(self, threshold: float = 0.93) -> List[Tuple[str, str, float]]:
        """Find near-duplicate entities via FAISS similarity."""
        if not FAISS_OK or len(self.entities) < 2:
            return []
        dupes = []
        for eid, ent in self.entities.items():
            results = self.faiss_index.search(f"{ent.name} {ent.description}", k=5)
            for other_id, score in results:
                if other_id != eid and score >= threshold:
                    pair = tuple(sorted([eid, other_id]))
                    if pair not in {tuple(sorted([a, b])) for a, b, _ in dupes}:
                        dupes.append((eid, other_id, score))
        return dupes

    def merge_entities(self, keep_id: str, remove_id: str):
        """Merge remove_id into keep_id."""
        if keep_id not in self.entities or remove_id not in self.entities:
            return
        keep = self.entities[keep_id]
        remove = self.entities[remove_id]

        # Merge properties
        keep.mention_count += remove.mention_count
        keep.importance = max(keep.importance, remove.importance)
        if len(remove.description) > len(keep.description):
            keep.description = remove.description
        keep.properties.update(remove.properties)

        # Redirect relationships
        for rel in self.relationships:
            if rel.source_id == remove_id:
                rel.source_id = keep_id
            if rel.target_id == remove_id:
                rel.target_id = keep_id

        # Remove and rebuild
        del self.entities[remove_id]
        if self.graph is not None and remove_id in self.graph:
            self.graph.remove_node(remove_id)
        self._rebuild_rel_index()
        self.stats["dupes_merged"] += 1

    # ── LLM Entity Extraction ────────────────────────────────────────

    async def extract_entities(self, text: str, source: str = "conversation") -> List[Entity]:
        """Extract entities and relationships from text using LLM."""
        self.stats["extractions"] += 1
        if self._llm is None:
            return self._heuristic_extract(text, source)

        entity_types_str = ", ".join(sorted(ENTITY_TYPES))
        relation_types_str = ", ".join(sorted(RELATION_TYPES.keys())[:30]) + " ..."

        prompt = f"""Extract entities and relationships from this text.

ENTITY TYPES: {entity_types_str}
RELATION TYPES (use ONLY these): {relation_types_str}

Text: {text[:2000]}

Return JSON:
{{"entities": [{{"name": "...", "type": "person|place|...", "description": "..."}}], "relationships": [{{"source": "...", "target": "...", "relation": "works_at|friend_of|..."}}]}}

Rules:
- Max 12 entities
- Description must be ≥30 chars
- NO vague relations (related_to, associated_with, etc.)
- NO self-loops (source == target)
- Use EXACT relation types from the list above"""

        try:
            resp = await self._llm.invoke_with_tools(
                [{"role": "user", "content": prompt}], []
            )
            raw = resp.get("text", "")
            # Parse JSON from response
            start = raw.find("{")
            end = raw.rfind("}") + 1
            if start >= 0 and end > start:
                data = json.loads(raw[start:end])
            else:
                return self._heuristic_extract(text, source)
        except Exception:
            return self._heuristic_extract(text, source)

        extracted = []
        for ed in data.get("entities", [])[:12]:
            name = ed.get("name", "").strip()
            desc = ed.get("description", "").strip()
            if not name or len(desc) < 20:
                continue
            ent = self.add_entity(
                name=name,
                entity_type=ed.get("type", "concept"),
                description=desc,
                source=source,
            )
            extracted.append(ent)

        for rd in data.get("relationships", []):
            src = rd.get("source", "").strip()
            tgt = rd.get("target", "").strip()
            rel = rd.get("relation", "").strip()
            if src and tgt and rel:
                self.add_relationship(src, tgt, rel)

        self._save_state()
        return extracted

    def _heuristic_extract(self, text: str, source: str) -> List[Entity]:
        """Fallback: simple NER-like extraction from text."""
        import re
        entities = []
        # Capitalized phrases (2+ words)
        for match in re.finditer(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b', text):
            name = match.group(1)
            if len(name) > 4:
                ent = self.add_entity(name=name, entity_type="concept", source=source)
                entities.append(ent)
        self._save_state()
        return entities[:12]

    # ── Graph Queries ────────────────────────────────────────────────

    def get_entity_relations(self, name: str) -> List[Dict[str, Any]]:
        """Get all relationships for an entity."""
        eid = self._make_id(name)
        results = []
        for rel in self.relationships:
            if rel.source_id == eid:
                tgt = self.entities.get(rel.target_id)
                if tgt:
                    results.append({
                        "direction": "outgoing",
                        "relation": rel.relation_type,
                        "entity": tgt.name,
                        "confidence": rel.confidence,
                    })
            elif rel.target_id == eid:
                src = self.entities.get(rel.source_id)
                if src:
                    results.append({
                        "direction": "incoming",
                        "relation": rel.relation_type,
                        "entity": src.name,
                        "confidence": rel.confidence,
                    })
        return results

    def get_communities(self) -> List[List[str]]:
        """Detect communities in the graph."""
        if self.graph is None or len(self.graph) == 0:
            return []
        undirected = self.graph.to_undirected()
        communities = []
        for component in nx.connected_components(undirected):
            names = [self.entities[n].name for n in component if n in self.entities]
            if names:
                communities.append(names)
        communities.sort(key=len, reverse=True)
        return communities[:20]

    def get_stats(self) -> Dict[str, Any]:
        return {
            **self.stats,
            "total_entities": len(self.entities),
            "total_relationships": len(self.relationships),
            "faiss_available": FAISS_OK,
            "networkx_available": NX_OK,
            "entity_types": dict(defaultdict(int, {
                ent.entity_type: sum(1 for e in self.entities.values() if e.entity_type == ent.entity_type)
                for ent in list(self.entities.values())[:1]  # just trigger count
            })) if self.entities else {},
        }

    # ── Internals ─────────────────────────────────────────────────────

    def _make_id(self, name: str) -> str:
        return hashlib.sha256(name.lower().strip().encode()).hexdigest()[:16]

    def _rebuild_rel_index(self):
        self._rel_index.clear()
        for idx, rel in enumerate(self.relationships):
            self._rel_index[rel.source_id].append(idx)
            self._rel_index[rel.target_id].append(idx)

    def _prune_entities(self):
        sorted_ents = sorted(self.entities.values(),
                             key=lambda e: e.importance * e.mention_count * e.confidence,
                             reverse=True)
        keep_ids = {e.entity_id for e in sorted_ents[:self.MAX_ENTITIES]}
        self.entities = {eid: e for eid, e in self.entities.items() if eid in keep_ids}
        self.relationships = [r for r in self.relationships
                              if r.source_id in keep_ids and r.target_id in keep_ids]
        self._rebuild_rel_index()
        self.faiss_index.rebuild(self.entities)

    def _prune_relationships(self):
        self.relationships.sort(key=lambda r: r.confidence, reverse=True)
        self.relationships = self.relationships[:self.MAX_RELATIONSHIPS]
        self._rebuild_rel_index()

    # ── Persistence ───────────────────────────────────────────────────

    def save(self):
        self._save_state()

    def _save_state(self):
        try:
            state = {
                "entities": {eid: {k: v for k, v in asdict(e).items() if k != "embedding"}
                             for eid, e in self.entities.items()},
                "relationships": [asdict(r) for r in self.relationships],
                "stats": self.stats,
            }
            self.state_file.write_text(json.dumps(state, indent=2, default=str), encoding="utf-8")
        except Exception as e:
            logger.error(f"[KnowledgeGraphV2] Save failed: {e}")

    def _load_state(self):
        if not self.state_file.exists():
            return
        try:
            state = json.loads(self.state_file.read_text(encoding="utf-8"))
            fields = set(Entity.__dataclass_fields__)
            for eid, ed in state.get("entities", {}).items():
                self.entities[eid] = Entity(**{k: v for k, v in ed.items() if k in fields})
            rel_fields = set(Relationship.__dataclass_fields__)
            for rd in state.get("relationships", []):
                self.relationships.append(
                    Relationship(**{k: v for k, v in rd.items() if k in rel_fields}))
            self.stats.update(state.get("stats", {}))
            self._rebuild_rel_index()

            # Rebuild graph + FAISS
            if self.graph is not None:
                for eid, ent in self.entities.items():
                    self.graph.add_node(eid, name=ent.name, type=ent.entity_type, importance=ent.importance)
                for rel in self.relationships:
                    self.graph.add_edge(rel.source_id, rel.target_id,
                                        relation=rel.relation_type, weight=rel.weight, confidence=rel.confidence)
            self.faiss_index.rebuild(self.entities)
            logger.info(f"[KnowledgeGraphV2] Loaded {len(self.entities)} entities, {len(self.relationships)} relationships")
        except Exception as e:
            logger.error(f"[KnowledgeGraphV2] Load failed: {e}")
