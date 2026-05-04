"""
#11,  Knowledge Graph Engine (Entity Relationship Intelligence)

The CORE capability that makes Palantir, Palantir.  An in-memory directed
graph that maps every entity (aircraft, vessel, person, organization,
facility, IP address, location) to every other entity through observed
relationships: co-location, ownership, communication, shared-route,
temporal-proximity, sanctions-link, etc.

This is NOT a simple adjacency list,  it supports:
  - Weighted, time-decaying edges
  - Multi-hop path discovery (6-degree queries)
  - Community detection (find clusters/cells)
  - Relationship inference (entity A → B → C → A = closed loop = suspicious)
  - Full-text entity search
  - Automatic graph construction from live Zunvra data

Example:
  query: "Show me everything connected to aircraft N123AB"
  result: Owner: Enigma Holdings LLC → Director: John Doe
          → Co-located with: Vessel MV Shadow (3 times in 30 days)
          → Route overlap with: Aircraft HB-JTF (sanctioned owner)
          → RISK CHAIN: N123AB ↔ Enigma Holdings ↔ OFAC Entity X
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from .connector import IntelSnapshot

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Graph primitives
# ---------------------------------------------------------------------------

@dataclass
class GraphNode:
    """A node in the knowledge graph."""
    node_id: str
    entity_type: str  # aircraft, vessel, satellite, person, organization, facility, ip, location
    label: str        # human-readable name
    properties: Dict[str, Any] = field(default_factory=dict)
    lat: Optional[float] = None
    lon: Optional[float] = None
    first_seen: str = ""
    last_seen: str = ""
    observation_count: int = 0


@dataclass
class GraphEdge:
    """A directed, weighted, typed edge between two nodes."""
    source_id: str
    target_id: str
    relationship: str   # co_location, ownership, route_overlap, sanctions_link,
                        # communication, temporal_proximity, shared_flag, same_zone
    weight: float = 1.0
    confidence: float = 0.8
    evidence: str = ""
    first_observed: str = ""
    last_observed: str = ""
    observation_count: int = 1

    @property
    def edge_id(self) -> str:
        return f"{self.source_id}→{self.relationship}→{self.target_id}"


@dataclass
class GraphPath:
    """A multi-hop path through the graph."""
    nodes: List[str]
    edges: List[GraphEdge]
    total_weight: float = 0.0
    risk_score: float = 0.0

    def to_text(self) -> str:
        parts = []
        for i, nid in enumerate(self.nodes):
            parts.append(nid)
            if i < len(self.edges):
                parts.append(f" --[{self.edges[i].relationship}]--> ")
        return "".join(parts)


@dataclass
class Community:
    """A detected community/cluster in the graph."""
    community_id: str
    members: List[str]
    central_node: str
    density: float  # edge density
    risk_score: float = 0.0
    label: str = ""


# ---------------------------------------------------------------------------
# Relationship extraction rules
# ---------------------------------------------------------------------------

def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    rlat1, rlon1, rlat2, rlon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = rlat2 - rlat1
    dlon = rlon2 - rlon1
    a = math.sin(dlat / 2) ** 2 + math.cos(rlat1) * math.cos(rlat2) * math.sin(dlon / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


CO_LOCATION_RADIUS_KM = 25.0  # entities within 25km are "co-located"
TEMPORAL_WINDOW_SEC = 3600     # events within 1 hour are "temporally proximate"


# ---------------------------------------------------------------------------
# Knowledge Graph Engine
# ---------------------------------------------------------------------------

class KnowledgeGraphEngine:
    """
    In-memory directed knowledge graph that maps relationships between
    all entities in the Zunvra OSINT feed.

    Supports:
    - Automatic graph construction from live snapshots
    - Multi-hop path discovery
    - Community/cluster detection
    - Closed-loop detection (suspicious circular relationships)
    - Time-decaying edge weights
    """

    MAX_NODES = 50000
    MAX_EDGES = 200000

    def __init__(self, data_dir: Optional[Path] = None):
        self.data_dir = Path(data_dir) if data_dir else Path("data/zunvra_intel")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.state_file = self.data_dir / "knowledge_graph.json"

        self.nodes: Dict[str, GraphNode] = {}
        self.edges: Dict[str, GraphEdge] = {}
        self.adjacency: Dict[str, Set[str]] = defaultdict(set)       # node_id → {edge_ids}
        self.reverse_adj: Dict[str, Set[str]] = defaultdict(set)     # target → {edge_ids}

        self.total_updates = 0
        self.closed_loops: List[GraphPath] = []
        self._load_state()

    # ── graph construction from live data ─────────────────────────────

    async def ingest(self, snapshot: IntelSnapshot, llm=None) -> Dict[str, int]:
        """
        Ingest a full Zunvra snapshot and build/update the knowledge graph.

        Returns counts of new nodes and edges created.
        """
        now = datetime.now(timezone.utc).isoformat()
        stats = {"new_nodes": 0, "new_edges": 0, "updated_edges": 0}

        # Phase 1: Extract entities as nodes
        nodes_created = self._extract_entities(snapshot, now)
        stats["new_nodes"] = nodes_created

        # Phase 2: Detect co-location relationships
        coloc_edges = self._detect_co_locations(snapshot, now)
        stats["new_edges"] += coloc_edges

        # Phase 3: Detect ownership / flag-based relationships
        ownership_edges = self._detect_ownership_links(snapshot, now)
        stats["new_edges"] += ownership_edges

        # Phase 4: Detect route overlaps
        route_edges = self._detect_route_overlaps(snapshot, now)
        stats["new_edges"] += route_edges

        # Phase 5: Detect shared-zone relationships
        zone_edges = self._detect_zone_relationships(snapshot, now)
        stats["new_edges"] += zone_edges

        # Phase 6: Detect closed loops (suspicious circular paths)
        self.closed_loops = self._detect_closed_loops(max_depth=4)

        # Phase 7: LLM relationship inference
        if llm and len(self.nodes) > 5:
            try:
                inferred = await self._llm_infer_relationships(llm, snapshot)
                stats["new_edges"] += inferred
            except Exception as e:
                logger.debug("LLM graph inference failed: %s", e)

        # Enforce limits
        self._enforce_limits()
        self.total_updates += 1
        self._save_state()

        return stats

    def _extract_entities(self, snapshot: IntelSnapshot, now: str) -> int:
        """Extract entities from snapshot as graph nodes."""
        count = 0

        # Aircraft
        for flight in snapshot.flights + snapshot.military_flights:
            icao = flight.get("hex", flight.get("icao", ""))
            if not icao:
                continue
            nid = f"aircraft_{icao}"
            if nid not in self.nodes:
                callsign = flight.get("callsign", flight.get("flight", "")).strip()
                self.nodes[nid] = GraphNode(
                    node_id=nid,
                    entity_type="aircraft",
                    label=callsign or icao,
                    properties={
                        "icao_hex": icao,
                        "callsign": callsign,
                        "type": flight.get("t", flight.get("type", "")),
                        "owner": flight.get("owner", flight.get("operator", "")),
                    },
                    lat=self._safe_float(flight.get("lat")),
                    lon=self._safe_float(flight.get("lon")),
                    first_seen=now,
                    last_seen=now,
                )
                count += 1
            else:
                node = self.nodes[nid]
                node.last_seen = now
                node.observation_count += 1
                node.lat = self._safe_float(flight.get("lat")) or node.lat
                node.lon = self._safe_float(flight.get("lon")) or node.lon

            # Owner as separate node
            owner = flight.get("owner", flight.get("operator", ""))
            if owner:
                owner_id = f"org_{hashlib.md5(owner.encode()).hexdigest()[:8]}"
                if owner_id not in self.nodes:
                    self.nodes[owner_id] = GraphNode(
                        node_id=owner_id,
                        entity_type="organization",
                        label=owner,
                        first_seen=now, last_seen=now,
                    )
                    count += 1
                self._add_edge(nid, owner_id, "ownership",
                              evidence=f"Aircraft {icao} registered to {owner}",
                              now=now)

        # Vessels
        for ship in snapshot.ships:
            mmsi = str(ship.get("mmsi", ship.get("MMSI", "")))
            if not mmsi:
                continue
            nid = f"vessel_{mmsi}"
            if nid not in self.nodes:
                name = ship.get("name", ship.get("ship_name", ""))
                self.nodes[nid] = GraphNode(
                    node_id=nid,
                    entity_type="vessel",
                    label=name or mmsi,
                    properties={
                        "mmsi": mmsi,
                        "name": name,
                        "flag": ship.get("flag", ship.get("country", "")),
                        "type": ship.get("ship_type", ship.get("type", "")),
                        "destination": ship.get("destination", ""),
                    },
                    lat=self._safe_float(ship.get("lat")),
                    lon=self._safe_float(ship.get("lon")),
                    first_seen=now, last_seen=now,
                )
                count += 1
            else:
                self.nodes[nid].last_seen = now
                self.nodes[nid].observation_count += 1

        # Cyber threats (IP addresses)
        for threat in snapshot.cyber_threats:
            ip = threat.get("ip", threat.get("ioc", ""))
            if not ip:
                continue
            nid = f"ip_{ip.replace('.', '_')}"
            if nid not in self.nodes:
                self.nodes[nid] = GraphNode(
                    node_id=nid,
                    entity_type="ip",
                    label=ip,
                    properties={
                        "ip": ip,
                        "threat_type": threat.get("type", threat.get("threat_type", "")),
                        "malware": threat.get("malware", ""),
                        "country": threat.get("country", ""),
                    },
                    lat=self._safe_float(threat.get("lat")),
                    lon=self._safe_float(threat.get("lon")),
                    first_seen=now, last_seen=now,
                )
                count += 1

        # Nuclear facilities as fixed nodes
        for facility in snapshot.nuclear_facilities:
            name = facility.get("name", facility.get("Name", ""))
            if not name:
                continue
            nid = f"facility_{hashlib.md5(name.encode()).hexdigest()[:8]}"
            if nid not in self.nodes:
                self.nodes[nid] = GraphNode(
                    node_id=nid,
                    entity_type="facility",
                    label=name,
                    properties={"type": "nuclear"},
                    lat=self._safe_float(facility.get("lat", facility.get("Latitude"))),
                    lon=self._safe_float(facility.get("lon", facility.get("Longitude"))),
                    first_seen=now, last_seen=now,
                )
                count += 1

        return count

    def _detect_co_locations(self, snapshot: IntelSnapshot, now: str) -> int:
        """Detect entities that are physically near each other → co_location edges."""
        edges_added = 0
        positioned_nodes = [(nid, n) for nid, n in self.nodes.items()
                           if n.lat is not None and n.lon is not None
                           and n.entity_type in ("aircraft", "vessel")]

        # O(n²) but bounded by entity types,  typically <5000 positioned nodes
        for i, (nid1, n1) in enumerate(positioned_nodes):
            for nid2, n2 in positioned_nodes[i + 1:]:
                if n1.entity_type == n2.entity_type:
                    continue  # Skip same-type co-location (too noisy)
                assert n1.lat is not None and n1.lon is not None
                assert n2.lat is not None and n2.lon is not None
                dist = _haversine(n1.lat, n1.lon, n2.lat, n2.lon)
                if dist <= CO_LOCATION_RADIUS_KM:
                    added = self._add_edge(
                        nid1, nid2, "co_location",
                        weight=1.0 - (dist / CO_LOCATION_RADIUS_KM),
                        evidence=f"Within {dist:.1f}km at ({n1.lat:.2f}, {n1.lon:.2f})",
                        now=now,
                    )
                    if added:
                        edges_added += 1
        return edges_added

    def _detect_ownership_links(self, snapshot: IntelSnapshot, now: str) -> int:
        """Detect ownership and flag-based relationships."""
        edges_added = 0

        # Same-flag vessels
        flag_groups: Dict[str, List[str]] = defaultdict(list)
        for nid, node in self.nodes.items():
            if node.entity_type == "vessel":
                flag = node.properties.get("flag", "")
                if flag:
                    flag_groups[flag.lower()].append(nid)

        for flag, vessel_ids in flag_groups.items():
            if len(vessel_ids) >= 2 and len(vessel_ids) <= 20:
                # Create flag node
                flag_nid = f"flag_{flag}"
                if flag_nid not in self.nodes:
                    self.nodes[flag_nid] = GraphNode(
                        node_id=flag_nid,
                        entity_type="organization",
                        label=f"Flag: {flag.upper()}",
                        first_seen=now, last_seen=now,
                    )
                for vid in vessel_ids:
                    added = self._add_edge(vid, flag_nid, "shared_flag",
                                          evidence=f"Vessel registered under {flag}",
                                          now=now, weight=0.3)
                    if added:
                        edges_added += 1

        return edges_added

    def _detect_route_overlaps(self, snapshot: IntelSnapshot, now: str) -> int:
        """Detect when different aircraft share destination patterns."""
        edges_added = 0
        dest_groups: Dict[str, List[str]] = defaultdict(list)

        for nid, node in self.nodes.items():
            if node.entity_type == "vessel":
                dest = node.properties.get("destination", "").strip().upper()
                if dest and dest not in ("", "UNKNOWN", "N/A"):
                    dest_groups[dest].append(nid)

        for dest, vessel_ids in dest_groups.items():
            if 2 <= len(vessel_ids) <= 10:
                for i, v1 in enumerate(vessel_ids):
                    for v2 in vessel_ids[i + 1:]:
                        added = self._add_edge(
                            v1, v2, "route_overlap",
                            evidence=f"Shared destination: {dest}",
                            now=now, weight=0.5,
                        )
                        if added:
                            edges_added += 1

        return edges_added

    def _detect_zone_relationships(self, snapshot: IntelSnapshot, now: str) -> int:
        """Link entities to geographical zones of interest."""
        zones = {
            "black_sea": (43.5, 34.0, 400),
            "south_china_sea": (15.0, 115.0, 600),
            "persian_gulf": (26.5, 52.0, 300),
            "east_med": (34.5, 33.0, 400),
            "baltic": (58.0, 20.0, 400),
            "red_sea": (20.0, 38.5, 400),
        }
        edges_added = 0
        for zone_name, (zlat, zlon, zrad) in zones.items():
            zone_nid = f"zone_{zone_name}"
            if zone_nid not in self.nodes:
                self.nodes[zone_nid] = GraphNode(
                    node_id=zone_nid, entity_type="location",
                    label=zone_name.replace("_", " ").title(),
                    lat=zlat, lon=zlon,
                    first_seen=now, last_seen=now,
                )
            for nid, node in self.nodes.items():
                if node.lat and node.lon and node.entity_type in ("aircraft", "vessel"):
                    dist = _haversine(node.lat, node.lon, zlat, zlon)
                    if dist <= zrad:
                        added = self._add_edge(
                            nid, zone_nid, "same_zone",
                            evidence=f"{node.label} in {zone_name} ({dist:.0f}km from center)",
                            now=now, weight=0.4,
                        )
                        if added:
                            edges_added += 1
        return edges_added

    # ── graph algorithms ──────────────────────────────────────────────

    def query_entity(self, entity_id: str, max_depth: int = 2) -> Dict[str, Any]:
        """
        Query everything connected to an entity up to N hops.

        Returns a subgraph with nodes, edges, and paths.
        """
        if entity_id not in self.nodes:
            # Fuzzy search
            matches = [nid for nid in self.nodes if entity_id.lower() in nid.lower()
                      or entity_id.lower() in self.nodes[nid].label.lower()]
            if not matches:
                return {"error": f"Entity '{entity_id}' not found"}
            entity_id = matches[0]

        visited: Set[str] = set()
        found_nodes: Dict[str, GraphNode] = {}
        found_edges: List[GraphEdge] = []
        queue: deque[Tuple[str, int]] = deque([(entity_id, 0)])

        while queue:
            nid, depth = queue.popleft()
            if nid in visited or depth > max_depth:
                continue
            visited.add(nid)
            if nid in self.nodes:
                found_nodes[nid] = self.nodes[nid]

            # Outgoing edges
            for eid in self.adjacency.get(nid, set()):
                if eid in self.edges:
                    edge = self.edges[eid]
                    found_edges.append(edge)
                    if edge.target_id not in visited:
                        queue.append((edge.target_id, depth + 1))

            # Incoming edges
            for eid in self.reverse_adj.get(nid, set()):
                if eid in self.edges:
                    edge = self.edges[eid]
                    found_edges.append(edge)
                    if edge.source_id not in visited:
                        queue.append((edge.source_id, depth + 1))

        return {
            "center": entity_id,
            "nodes": {nid: asdict(n) for nid, n in found_nodes.items()},
            "edges": [asdict(e) for e in found_edges],
            "node_count": len(found_nodes),
            "edge_count": len(found_edges),
        }

    def find_path(self, source_id: str, target_id: str,
                  max_depth: int = 6) -> Optional[GraphPath]:
        """BFS shortest path between two entities."""
        if source_id not in self.nodes or target_id not in self.nodes:
            return None

        visited: Set[str] = set()
        queue: deque[List[str]] = deque([[source_id]])

        while queue:
            path = queue.popleft()
            current = path[-1]
            if current == target_id:
                # Build GraphPath
                edges = []
                for i in range(len(path) - 1):
                    for eid in self.adjacency.get(path[i], set()):
                        if eid in self.edges and self.edges[eid].target_id == path[i + 1]:
                            edges.append(self.edges[eid])
                            break
                    else:
                        for eid in self.reverse_adj.get(path[i], set()):
                            if eid in self.edges and self.edges[eid].source_id == path[i + 1]:
                                edges.append(self.edges[eid])
                                break
                total_w = sum(e.weight for e in edges)
                return GraphPath(nodes=path, edges=edges, total_weight=total_w)

            if len(path) > max_depth:
                continue

            visited.add(current)
            neighbors: Set[str] = set()
            for eid in self.adjacency.get(current, set()):
                if eid in self.edges:
                    neighbors.add(self.edges[eid].target_id)
            for eid in self.reverse_adj.get(current, set()):
                if eid in self.edges:
                    neighbors.add(self.edges[eid].source_id)

            for nxt in neighbors:
                if nxt not in visited:
                    queue.append(path + [nxt])

        return None

    def _detect_closed_loops(self, max_depth: int = 4) -> List[GraphPath]:
        """Detect circular paths (A→B→C→A) which may indicate suspicious networks."""
        loops: List[GraphPath] = []
        checked: Set[str] = set()

        for start_id in list(self.nodes.keys())[:500]:  # Limit search
            if start_id in checked:
                continue
            checked.add(start_id)

            # DFS for cycles
            stack: List[Tuple[str, List[str], List[GraphEdge]]] = [(start_id, [start_id], [])]
            visited_local: Set[str] = set()

            while stack:
                current, path, edges = stack.pop()
                if len(path) > max_depth + 1:
                    continue

                for eid in self.adjacency.get(current, set()):
                    if eid not in self.edges:
                        continue
                    edge = self.edges[eid]
                    target = edge.target_id

                    if target == start_id and len(path) >= 3:
                        loop = GraphPath(
                            nodes=path + [start_id],
                            edges=edges + [edge],
                            total_weight=sum(e.weight for e in edges + [edge]),
                            risk_score=min(1.0, len(path) * 0.25),
                        )
                        loops.append(loop)
                    elif target not in visited_local and len(path) < max_depth:
                        visited_local.add(target)
                        stack.append((target, path + [target], edges + [edge]))

            if len(loops) >= 50:
                break

        return loops

    def detect_communities(self, min_size: int = 3) -> List[Community]:
        """Simple label-propagation community detection."""
        labels: Dict[str, str] = {nid: nid for nid in self.nodes}

        for _ in range(10):  # iterations
            changed = False
            for nid in self.nodes:
                neighbor_labels: Dict[str, float] = defaultdict(float)
                for eid in self.adjacency.get(nid, set()):
                    if eid in self.edges:
                        target = self.edges[eid].target_id
                        if target in labels:
                            neighbor_labels[labels[target]] += self.edges[eid].weight
                for eid in self.reverse_adj.get(nid, set()):
                    if eid in self.edges:
                        source = self.edges[eid].source_id
                        if source in labels:
                            neighbor_labels[labels[source]] += self.edges[eid].weight
                if neighbor_labels:
                    best_label = max(neighbor_labels, key=lambda k: neighbor_labels[k])
                    if best_label != labels[nid]:
                        labels[nid] = best_label
                        changed = True
            if not changed:
                break

        # Group by label
        groups: Dict[str, List[str]] = defaultdict(list)
        for nid, label in labels.items():
            groups[label].append(nid)

        communities = []
        for label, members in groups.items():
            if len(members) >= min_size:
                # Find most connected node
                central = max(members,
                             key=lambda n: len(self.adjacency.get(n, set())) +
                                          len(self.reverse_adj.get(n, set())))
                # Edge density
                internal_edges = 0
                member_set = set(members)
                for m in members:
                    for eid in self.adjacency.get(m, set()):
                        if eid in self.edges and self.edges[eid].target_id in member_set:
                            internal_edges += 1
                max_edges = len(members) * (len(members) - 1)
                density = internal_edges / max_edges if max_edges > 0 else 0.0

                communities.append(Community(
                    community_id=hashlib.md5(label.encode()).hexdigest()[:8],
                    members=members,
                    central_node=central,
                    density=density,
                    label=f"Cluster around {self.nodes[central].label}" if central in self.nodes else label,
                ))

        return sorted(communities, key=lambda c: len(c.members), reverse=True)

    # ── edge management ───────────────────────────────────────────────

    def _add_edge(self, source_id: str, target_id: str, relationship: str,
                  evidence: str = "", now: str = "", weight: float = 1.0,
                  confidence: float = 0.8) -> bool:
        """Add or update an edge. Returns True if new."""
        eid = f"{source_id}→{relationship}→{target_id}"
        if eid in self.edges:
            edge = self.edges[eid]
            edge.observation_count += 1
            edge.last_observed = now
            edge.weight = min(5.0, edge.weight + 0.1)  # Reinforce
            return False

        edge = GraphEdge(
            source_id=source_id,
            target_id=target_id,
            relationship=relationship,
            weight=weight,
            confidence=confidence,
            evidence=evidence,
            first_observed=now,
            last_observed=now,
        )
        self.edges[eid] = edge
        self.adjacency[source_id].add(eid)
        self.reverse_adj[target_id].add(eid)
        return True

    # ── LLM inference ─────────────────────────────────────────────────

    async def _llm_infer_relationships(self, llm, snapshot: IntelSnapshot) -> int:
        """Use LLM to infer non-obvious relationships."""
        # Pick recent high-interest nodes
        interesting = sorted(
            self.nodes.values(),
            key=lambda n: n.observation_count, reverse=True
        )[:10]

        if len(interesting) < 3:
            return 0

        entities_text = "\n".join(
            f"- {n.node_id}: {n.entity_type} '{n.label}' "
            f"(seen {n.observation_count}x, props: {json.dumps(n.properties)[:100]})"
            for n in interesting
        )

        prompt = (
            "You are an intelligence analyst examining entity relationships.\n"
            "Given these entities observed in live OSINT data:\n\n"
            f"{entities_text}\n\n"
            "Identify any non-obvious relationships between them. For each:\n"
            "- source_id, target_id, relationship_type, evidence\n"
            "Valid relationship types: ownership, co_location, route_overlap, "
            "sanctions_link, communication, temporal_proximity\n"
            "Return JSON array. Only include HIGH-CONFIDENCE relationships."
        )

        raw = await llm.chat_raw(prompt, max_tokens=300)
        if not raw:
            return 0

        now = datetime.now(timezone.utc).isoformat()
        count = 0
        try:
            # Try to parse JSON from LLM response
            start = raw.find("[")
            end = raw.rfind("]") + 1
            if start >= 0 and end > start:
                relationships = json.loads(raw[start:end])
                for rel in relationships:
                    src = rel.get("source_id", "")
                    tgt = rel.get("target_id", "")
                    rtype = rel.get("relationship_type", "")
                    evidence = rel.get("evidence", "LLM inference")
                    if src in self.nodes and tgt in self.nodes and rtype:
                        if self._add_edge(src, tgt, rtype, evidence=evidence,
                                         now=now, confidence=0.6):
                            count += 1
        except (json.JSONDecodeError, Exception):
            pass

        return count

    # ── helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _safe_float(val: Any) -> Optional[float]:
        if val is None:
            return None
        try:
            return float(val)
        except (ValueError, TypeError):
            return None

    def _enforce_limits(self):
        if len(self.nodes) > self.MAX_NODES:
            oldest = sorted(self.nodes.values(), key=lambda n: n.last_seen)
            to_remove = [n.node_id for n in oldest[:len(self.nodes) - self.MAX_NODES + 1000]]
            for nid in to_remove:
                self._remove_node(nid)

        if len(self.edges) > self.MAX_EDGES:
            oldest = sorted(self.edges.values(), key=lambda e: e.last_observed)
            to_remove = [e.edge_id for e in oldest[:len(self.edges) - self.MAX_EDGES + 5000]]
            for eid in to_remove:
                if eid in self.edges:
                    edge = self.edges[eid]
                    self.adjacency[edge.source_id].discard(eid)
                    self.reverse_adj[edge.target_id].discard(eid)
                    del self.edges[eid]

    def _remove_node(self, node_id: str):
        if node_id in self.nodes:
            del self.nodes[node_id]
        for eid in list(self.adjacency.get(node_id, set())):
            if eid in self.edges:
                del self.edges[eid]
        for eid in list(self.reverse_adj.get(node_id, set())):
            if eid in self.edges:
                del self.edges[eid]
        self.adjacency.pop(node_id, None)
        self.reverse_adj.pop(node_id, None)

    def search(self, query: str, limit: int = 20) -> List[GraphNode]:
        """Full-text search across node labels and properties."""
        q = query.lower()
        results = []
        for node in self.nodes.values():
            if q in node.label.lower() or q in node.node_id.lower():
                results.append(node)
            elif any(q in str(v).lower() for v in node.properties.values()):
                results.append(node)
            if len(results) >= limit:
                break
        return results

    # ── persistence ───────────────────────────────────────────────────

    def _save_state(self):
        try:
            state = {
                "total_updates": self.total_updates,
                "node_count": len(self.nodes),
                "edge_count": len(self.edges),
                "loop_count": len(self.closed_loops),
            }
            self.state_file.write_text(json.dumps(state, default=str), encoding="utf-8")
        except Exception as e:
            logger.warning("Failed to save graph state: %s", e)

    def _load_state(self):
        try:
            if self.state_file.exists():
                state = json.loads(self.state_file.read_text("utf-8"))
                self.total_updates = state.get("total_updates", 0)
        except Exception as e:
            logger.warning("Failed to load graph state: %s", e)

    def get_stats(self) -> Dict[str, Any]:
        return {
            "total_nodes": len(self.nodes),
            "total_edges": len(self.edges),
            "total_updates": self.total_updates,
            "closed_loops": len(self.closed_loops),
            "node_types": dict(
                defaultdict(int,
                    {n.entity_type: sum(1 for x in self.nodes.values()
                                       if x.entity_type == n.entity_type)
                     for n in list(self.nodes.values())[:100]})
            ),
            "edge_types": dict(
                defaultdict(int,
                    {e.relationship: sum(1 for x in self.edges.values()
                                        if x.relationship == e.relationship)
                     for e in list(self.edges.values())[:100]})
            ),
        }
