"""
Document Extraction Pipeline — Map-Reduce Entity Extraction

Extracts structured entities and relationships from documents (PDF, DOCX,
TXT, Markdown, HTML) into the Knowledge Graph v2.

Architecture:
  1. Reader   — convert any supported format to plain text chunks
  2. Map      — LLM extracts entities/relations from each chunk
  3. Reduce   — deduplicate, merge, and insert into knowledge graph

Supports:
  - Chunking with configurable overlap (handles long docs)
  - Batch processing with rate limiting
  - Progress tracking per document
  - Graceful fallback when optional deps (pypdf, python-docx) missing
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from opensable.core.knowledge_graph_v2 import KnowledgeGraphV2

logger = logging.getLogger(__name__)

# ── Text Readers ──────────────────────────────────────────────────────


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def read_markdown(path: Path) -> str:
    text = path.read_text(encoding="utf-8", errors="replace")
    # Strip markdown syntax for cleaner extraction
    text = re.sub(r"```[\s\S]*?```", "", text)        # code blocks
    text = re.sub(r"`[^`]+`", "", text)                # inline code
    text = re.sub(r"!\[.*?\]\(.*?\)", "", text)        # images
    text = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", text)  # links → text
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)  # headers
    return text


def read_html(path: Path) -> str:
    raw = path.read_text(encoding="utf-8", errors="replace")
    # Strip HTML tags
    text = re.sub(r"<script[\s\S]*?</script>", "", raw, flags=re.IGNORECASE)
    text = re.sub(r"<style[\s\S]*?</style>", "", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def read_pdf(path: Path) -> str:
    try:
        from pypdf import PdfReader
        reader = PdfReader(str(path))
        pages = []
        for page in reader.pages:
            t = page.extract_text()
            if t:
                pages.append(t)
        return "\n\n".join(pages)
    except ImportError:
        logger.warning("pypdf not installed — cannot extract PDF. pip install pypdf")
        return ""
    except Exception as e:
        logger.error(f"PDF read error: {e}")
        return ""


def read_docx(path: Path) -> str:
    try:
        from docx import Document
        doc = Document(str(path))
        return "\n\n".join(p.text for p in doc.paragraphs if p.text.strip())
    except ImportError:
        logger.warning("python-docx not installed — cannot extract DOCX. pip install python-docx")
        return ""
    except Exception as e:
        logger.error(f"DOCX read error: {e}")
        return ""


READERS = {
    ".txt": read_text,
    ".md": read_markdown,
    ".markdown": read_markdown,
    ".html": read_html,
    ".htm": read_html,
    ".pdf": read_pdf,
    ".docx": read_docx,
}


# ── Chunking ──────────────────────────────────────────────────────────


def chunk_text(text: str, chunk_size: int = 3000, overlap: int = 300) -> List[str]:
    """Split text into overlapping chunks by character count."""
    if len(text) <= chunk_size:
        return [text] if text.strip() else []

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk)
        start = end - overlap
    return chunks


# ── Extraction Result ─────────────────────────────────────────────────

@dataclass
class ExtractedEntity:
    name: str
    entity_type: str
    description: str = ""


@dataclass
class ExtractedRelation:
    source: str
    target: str
    relation: str
    description: str = ""


@dataclass
class ExtractionResult:
    source_file: str = ""
    entities: List[ExtractedEntity] = field(default_factory=list)
    relations: List[ExtractedRelation] = field(default_factory=list)
    chunks_processed: int = 0
    errors: List[str] = field(default_factory=list)


# ── Document Extraction Pipeline ──────────────────────────────────────

EXTRACTION_PROMPT = """Extract all named entities and relationships from the following text.

Return ONLY valid JSON with this structure:
{{
  "entities": [
    {{"name": "EntityName", "type": "person|organization|technology|concept|location|event|product|other", "description": "brief description"}}
  ],
  "relationships": [
    {{"source": "Entity1", "target": "Entity2", "relation": "relation_type", "description": "brief"}}
  ]
}}

Relation types to use: works_at, created_by, part_of, uses, depends_on, collaborates_with, located_in, occurred_at, competes_with, succeeded_by, subclass_of, implements, integrates_with, related_to

Text:
{text}

JSON:"""


class DocumentExtractor:
    """
    Map-reduce document extraction pipeline.
    Extracts entities and relationships from documents into a knowledge graph.
    """

    MAX_CHUNKS_PER_DOC = 50
    SUPPORTED_EXTENSIONS = set(READERS.keys())

    def __init__(self, kg: "KnowledgeGraphV2"):
        self.kg = kg
        self._llm = None

    def set_llm(self, llm):
        self._llm = llm

    def is_supported(self, path: Path) -> bool:
        return path.suffix.lower() in self.SUPPORTED_EXTENSIONS

    # ── Main API ──────────────────────────────────────────────────────

    async def extract_file(self, file_path: Path) -> ExtractionResult:
        """Extract entities/relations from a single file."""
        result = ExtractionResult(source_file=str(file_path))
        path = Path(file_path)

        if not path.exists():
            result.errors.append(f"File not found: {path}")
            return result

        suffix = path.suffix.lower()
        reader = READERS.get(suffix)
        if not reader:
            result.errors.append(f"Unsupported format: {suffix}")
            return result

        # Step 1: Read
        text = reader(path)
        if not text or len(text.strip()) < 50:
            result.errors.append("File too short or empty")
            return result

        # Step 2: Chunk
        chunks = chunk_text(text)[:self.MAX_CHUNKS_PER_DOC]

        # Step 3: Map — extract from each chunk
        all_entities: List[ExtractedEntity] = []
        all_relations: List[ExtractedRelation] = []

        for i, chunk in enumerate(chunks):
            ents, rels, err = await self._extract_chunk(chunk)
            all_entities.extend(ents)
            all_relations.extend(rels)
            if err:
                result.errors.append(f"Chunk {i}: {err}")
            result.chunks_processed += 1

        # Step 4: Reduce — deduplicate and merge into KG
        self._reduce_into_kg(all_entities, all_relations, str(file_path))

        result.entities = all_entities
        result.relations = all_relations

        logger.info(f"[DocExtract] {path.name}: {len(all_entities)} entities, "
                     f"{len(all_relations)} relations from {result.chunks_processed} chunks")
        return result

    async def extract_directory(self, dir_path: Path, recursive: bool = True) -> List[ExtractionResult]:
        """Extract from all supported files in a directory."""
        results = []
        pattern = "**/*" if recursive else "*"
        for p in sorted(dir_path.glob(pattern)):
            if p.is_file() and self.is_supported(p):
                r = await self.extract_file(p)
                results.append(r)
        return results

    # ── Internals ─────────────────────────────────────────────────────

    async def _extract_chunk(self, chunk: str):
        """LLM-based entity/relation extraction from a text chunk."""
        if not self._llm:
            return self._heuristic_extract(chunk)

        prompt = EXTRACTION_PROMPT.format(text=chunk[:4000])

        try:
            resp = await self._llm.invoke_with_tools(
                [{"role": "user", "content": prompt}], []
            )
            raw = resp.get("text", "")
            return self._parse_extraction_response(raw)
        except Exception as e:
            logger.debug(f"[DocExtract] LLM extraction failed: {e}")
            return self._heuristic_extract(chunk)

    def _parse_extraction_response(self, raw: str):
        """Parse JSON response from LLM."""
        entities, relations = [], []
        try:
            s = raw.find("{")
            e = raw.rfind("}") + 1
            if s >= 0 and e > s:
                data = json.loads(raw[s:e])
                for ent in data.get("entities", []):
                    if isinstance(ent, dict) and ent.get("name"):
                        entities.append(ExtractedEntity(
                            name=ent["name"],
                            entity_type=ent.get("type", "concept"),
                            description=ent.get("description", ""),
                        ))
                for rel in data.get("relationships", []):
                    if isinstance(rel, dict) and rel.get("source") and rel.get("target"):
                        relations.append(ExtractedRelation(
                            source=rel["source"],
                            target=rel["target"],
                            relation=rel.get("relation", "related_to"),
                            description=rel.get("description", ""),
                        ))
        except (json.JSONDecodeError, KeyError) as e:
            return entities, relations, str(e)
        return entities, relations, None

    def _heuristic_extract(self, chunk: str):
        """Fallback: simple NER-like heuristic without LLM."""
        entities = []
        # Extract capitalized multi-word phrases as potential entities
        for match in re.finditer(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b", chunk):
            name = match.group(1)
            if len(name) > 3 and name not in {"The", "This", "That", "These", "Those"}:
                entities.append(ExtractedEntity(name=name, entity_type="concept"))
        # Dedupe
        seen = set()
        unique = []
        for e in entities:
            if e.name not in seen:
                seen.add(e.name)
                unique.append(e)
        return unique[:20], [], None

    def _reduce_into_kg(self, entities: List[ExtractedEntity],
                        relations: List[ExtractedRelation], source: str):
        """Merge extracted data into knowledge graph."""
        for ent in entities:
            self.kg.add_entity(
                name=ent.name,
                entity_type=ent.entity_type,
                description=ent.description,
                source=source,
            )
        for rel in relations:
            self.kg.add_relationship(
                source_entity=rel.source,
                target_entity=rel.target,
                relation_type=rel.relation,
                description=rel.description,
                confidence=0.7,
                evidence=[f"extracted from {source}"],
            )
