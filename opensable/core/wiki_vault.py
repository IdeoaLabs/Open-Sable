"""
Wiki Vault — Obsidian-Compatible Knowledge Export

Exports the Knowledge Graph v2 into an Obsidian vault:
  - One .md file per entity with YAML frontmatter + [[wiki-links]]
  - Per-type index files (e.g., _index_person.md, _index_technology.md)
  - Master index with entity counts and statistics
  - Incremental export (only changed entities)
"""
from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from opensable.core.knowledge_graph_v2 import KnowledgeGraphV2

logger = logging.getLogger(__name__)


def _safe_filename(name: str) -> str:
    """Convert entity name to safe filename."""
    s = re.sub(r'[<>:"/\\|?*]', '_', name)
    s = re.sub(r'\s+', ' ', s).strip()
    return s[:200] if s else "unnamed"


class WikiVault:
    """
    Export KG v2 to Obsidian-compatible markdown vault.
    """

    def __init__(self, kg: "KnowledgeGraphV2", vault_dir: Path):
        self.kg = kg
        self.vault_dir = Path(vault_dir)
        self._exported_hashes: Dict[str, str] = {}
        self._state_file = self.vault_dir / ".vault_state.json"
        self._load_state()

    # ── Main Export ───────────────────────────────────────────────────

    def export(self, full: bool = False) -> Dict[str, int]:
        """
        Export knowledge graph to vault.

        Args:
            full: If True, re-export everything. If False, only changed entities.

        Returns: {"exported": N, "skipped": N, "indexes": N}
        """
        self.vault_dir.mkdir(parents=True, exist_ok=True)
        stats = {"exported": 0, "skipped": 0, "indexes": 0}

        # Export entity pages
        for eid, entity in self.kg.entities.items():
            content_hash = self._entity_hash(entity)
            if not full and self._exported_hashes.get(eid) == content_hash:
                stats["skipped"] += 1
                continue

            self._export_entity(entity)
            self._exported_hashes[eid] = content_hash
            stats["exported"] += 1

        # Build per-type indexes
        type_groups: Dict[str, list] = {}
        for entity in self.kg.entities.values():
            type_groups.setdefault(entity.entity_type, []).append(entity)

        for etype, entities in type_groups.items():
            self._write_type_index(etype, entities)
            stats["indexes"] += 1

        # Master index
        self._write_master_index(type_groups)
        stats["indexes"] += 1

        self._save_state()
        logger.info(f"[WikiVault] Export: {stats['exported']} new, "
                     f"{stats['skipped']} skipped, {stats['indexes']} indexes")
        return stats

    # ── Entity Page ───────────────────────────────────────────────────

    def _export_entity(self, entity):
        """Write one entity as an Obsidian markdown page."""
        rels = self.kg.get_entity_relations(entity.name)

        # YAML frontmatter
        fm = {
            "type": entity.entity_type,
            "aliases": entity.aliases if hasattr(entity, 'aliases') and entity.aliases else [],
            "confidence": round(getattr(entity, "confidence", 1.0), 2),
            "source": getattr(entity, "source", ""),
            "created": getattr(entity, "created_at", ""),
            "updated": getattr(entity, "last_updated", ""),
            "tags": [f"type/{entity.entity_type}"],
        }

        lines = ["---"]
        for k, v in fm.items():
            if isinstance(v, list):
                if v:
                    lines.append(f"{k}:")
                    for item in v:
                        lines.append(f"  - {item}")
                else:
                    lines.append(f"{k}: []")
            else:
                lines.append(f"{k}: {json.dumps(v) if isinstance(v, str) and ('\"' in v or ':' in v) else v}")
        lines.append("---")
        lines.append("")

        # Title
        lines.append(f"# {entity.name}")
        lines.append("")

        # Description
        if entity.description:
            lines.append(entity.description)
            lines.append("")

        # Relationships
        if rels:
            lines.append("## Relationships")
            lines.append("")

            # Group by relation type
            by_type: Dict[str, list] = {}
            for r in rels:
                by_type.setdefault(r["relation"], []).append(r)

            for rtype, items in sorted(by_type.items()):
                lines.append(f"### {rtype.replace('_', ' ').title()}")
                for item in items:
                    target_name = item.get("entity", "unknown")
                    desc = item.get("description", "")
                    link = f"[[{target_name}]]"
                    if desc:
                        lines.append(f"- {link} — {desc}")
                    else:
                        lines.append(f"- {link}")
                lines.append("")

        # Metadata footer
        lines.append("---")
        lines.append(f"*Entity ID: `{entity.entity_id}`*")
        lines.append("")

        # Write file
        fname = _safe_filename(entity.name) + ".md"
        etype_dir = self.vault_dir / _safe_filename(entity.entity_type)
        etype_dir.mkdir(parents=True, exist_ok=True)
        (etype_dir / fname).write_text("\n".join(lines), encoding="utf-8")

    # ── Index Pages ───────────────────────────────────────────────────

    def _write_type_index(self, etype: str, entities: list):
        """Write per-type index page."""
        lines = [f"# {etype.title()} Index", ""]
        lines.append(f"Total: {len(entities)}")
        lines.append("")

        # Sort alphabetically
        for ent in sorted(entities, key=lambda e: e.name.lower()):
            desc = ent.description[:80] + "..." if len(ent.description) > 80 else ent.description
            lines.append(f"- [[{ent.name}]] — {desc}" if desc else f"- [[{ent.name}]]")

        lines.append("")
        idx_file = self.vault_dir / f"_index_{_safe_filename(etype)}.md"
        idx_file.write_text("\n".join(lines), encoding="utf-8")

    def _write_master_index(self, type_groups: Dict[str, list]):
        """Write master vault index with statistics."""
        total_entities = len(self.kg.entities)
        total_rels = len(self.kg.relationships)
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

        lines = [
            "# Knowledge Vault",
            "",
            f"*Last exported: {now}*",
            "",
            f"- **Total Entities:** {total_entities}",
            f"- **Total Relationships:** {total_rels}",
            f"- **Entity Types:** {len(type_groups)}",
            "",
            "## By Type",
            "",
        ]

        for etype in sorted(type_groups.keys()):
            count = len(type_groups[etype])
            lines.append(f"- [[_index_{_safe_filename(etype)}|{etype.title()}]] ({count})")

        lines.append("")
        (self.vault_dir / "_index.md").write_text("\n".join(lines), encoding="utf-8")

    # ── Helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _entity_hash(entity) -> str:
        """Quick hash for change detection."""
        raw = f"{entity.name}|{entity.entity_type}|{entity.description}|{getattr(entity, 'last_updated', '')}"
        return str(hash(raw))

    def _save_state(self):
        try:
            self._state_file.write_text(
                json.dumps(self._exported_hashes, indent=2), encoding="utf-8"
            )
        except Exception as e:
            logger.error(f"[WikiVault] State save failed: {e}")

    def _load_state(self):
        if self._state_file.exists():
            try:
                self._exported_hashes = json.loads(
                    self._state_file.read_text(encoding="utf-8")
                )
            except Exception:
                self._exported_hashes = {}
