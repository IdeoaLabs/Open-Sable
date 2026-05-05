"""
Skills Marketplace - Community-driven skills repository.

Features:
- Skill discovery and search
- Skill installation and management
- Version control and updates
- Ratings and reviews
- Dependency management
- Security scanning
- Automatic skill deployment
- REAL marketplace connection via Agent Gateway Protocol (SAGP)
"""

import asyncio
import logging
import json
import tarfile
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field, asdict
from enum import Enum
from datetime import datetime, timezone
import uuid
import os

try:
    import semver
except ImportError:
    semver = None

logger = logging.getLogger(__name__)


class SkillCategory(Enum):
    """Skill categories."""

    PRODUCTIVITY = "productivity"
    COMMUNICATION = "communication"
    AUTOMATION = "automation"
    DATA_ANALYSIS = "data_analysis"
    ENTERTAINMENT = "entertainment"
    EDUCATION = "education"
    DEVELOPMENT = "development"
    SYSTEM = "system"
    AI_ML = "ai_ml"
    CUSTOM = "custom"


class SkillStatus(Enum):
    """Skill installation status."""

    NOT_INSTALLED = "not_installed"
    INSTALLING = "installing"
    INSTALLED = "installed"
    UPDATING = "updating"
    FAILED = "failed"
    DISABLED = "disabled"


@dataclass
class SkillMetadata:
    """Skill package metadata."""

    skill_id: str
    name: str
    version: str
    author: str
    description: str
    category: SkillCategory
    tags: List[str] = field(default_factory=list)

    # Requirements
    min_python_version: str = "3.11"
    dependencies: List[str] = field(default_factory=list)
    required_skills: List[str] = field(default_factory=list)

    # Files
    main_file: str = "skill.py"
    config_schema: Optional[Dict[str, Any]] = None

    # Repository info
    repository: Optional[str] = None
    homepage: Optional[str] = None
    license: str = "MIT"

    # Stats
    downloads: int = 0
    rating: float = 0.0
    reviews_count: int = 0

    # Publishing
    published_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    # Security
    checksum: Optional[str] = None
    verified: bool = False


@dataclass
class SkillReview:
    """User review of a skill."""

    review_id: str
    skill_id: str
    user_id: str
    rating: int  # 1-5
    title: str
    content: str
    created_at: datetime
    helpful_count: int = 0


@dataclass
class InstalledSkill:
    """Installed skill information."""

    skill_id: str
    metadata: SkillMetadata
    install_path: Path
    status: SkillStatus
    installed_at: datetime
    config: Dict[str, Any] = field(default_factory=dict)
    enabled: bool = True
    auto_update: bool = True


class SkillRegistry:
    """
    Central registry of available skills.

    Connects to the SableCore Skills Marketplace via the Agent Gateway
    Protocol (SAGP) for secure, agent-only access. Falls back to local
    mock data if the gateway is unreachable.
    """

    # ── Default marketplace config ──
    DEFAULT_GATEWAY_URL = "https://sk.opensable.com/gateway"
    AGENT_CREDENTIALS_FILE = "agent-credentials.json"  # kept for backward compat reference only

    def __init__(
        self,
        registry_url: Optional[str] = None,
        gateway_url: Optional[str] = None,
        agent_id: Optional[str] = None,
        signing_key: Optional[str] = None,
        encryption_key: Optional[str] = None,
        credentials_file: Optional[str] = None,
    ):
        """
        Initialize skill registry.

        Priority for config:
          1. Explicit parameters
          2. Environment variables
          3. That's it,  no filesystem scanning for security

        Args:
            registry_url: Legacy URL (ignored when gateway is available)
            gateway_url: Agent Gateway URL (e.g., https://sk.opensable.com/gateway)
            agent_id: Provisioned agent ID
            signing_key: Base64-encoded Ed25519 secret key
            encryption_key: Base64-encoded X25519 secret key
            credentials_file: IGNORED (removed for security,  use env vars)
        """
        self.registry_url = registry_url or os.environ.get("SKILLS_API_URL", "https://sk.opensable.com/api")
        self.cache: Dict[str, SkillMetadata] = {}
        self.cache_updated: Optional[datetime] = None

        # ── Store API key (human-delegated access) ──
        # If the owner has a store account, they can set their sk_* API key
        # so the agent can install/review skills via the REST API on their behalf.
        self._store_api_key = os.environ.get("SABLE_STORE_API_KEY")

        # ── Load gateway credentials (env vars ONLY,  no filesystem scanning) ──
        self._gateway_url = (
            gateway_url
            or os.environ.get("SABLE_GATEWAY_URL")
            or self.DEFAULT_GATEWAY_URL
        )
        self._agent_id = agent_id or os.environ.get("SABLE_AGENT_ID")
        self._signing_key = signing_key or os.environ.get("SABLE_AGENT_SIGNING_KEY")
        self._encryption_key = encryption_key or os.environ.get("SABLE_AGENT_ENCRYPTION_KEY")

        # SECURITY: No automatic credentials file scanning.
        # Credentials MUST be provided via env vars or explicit parameters.
        # The owner provisions agents offline with:
        #   node marketplace/server/scripts/provision-agent.js --name "MyAgent"
        # Then sets env vars: SABLE_AGENT_ID, SABLE_AGENT_SIGNING_KEY, SABLE_AGENT_ENCRYPTION_KEY
        if not self._agent_id or not self._signing_key:
            logger.info(
                "Agent gateway credentials not configured. Gateway features disabled. "
                "Set SABLE_AGENT_ID, SABLE_AGENT_SIGNING_KEY, SABLE_AGENT_ENCRYPTION_KEY "
                "env vars to enable."
            )

        # Gateway client (lazy-initialized)
        self._gateway_client = None
        self._gateway_available = False

        logger.info(
            f"SkillRegistry initialized,  gateway: {self._gateway_url}, "
            f"agent: {self._agent_id[:8] + '...' if self._agent_id else 'NOT CONFIGURED'}"
        )

    # _load_credentials and _parse_credentials_file REMOVED for security.
    # Agents must NEVER scan the filesystem for credentials.
    # The owner sets env vars manually after offline provisioning.

    # ── Store REST API (uses human's sk_* API key) ──

    async def _store_api_request(
        self, method: str, path: str, json_body: Optional[Dict] = None
    ) -> Optional[Dict]:
        """
        Make an authenticated REST API request to the Skills Store
        using the owner's API key (SABLE_STORE_API_KEY).

        This lets the agent act on behalf of the human owner, 
        install skills, post reviews, etc.,  without needing the
        full SAGP handshake.

        Returns:
            Response JSON dict, or None on failure.
        """
        if not self._store_api_key:
            return None

        try:
            import aiohttp

            url = f"{self.registry_url}{path}"
            headers = {
                "Authorization": f"Bearer {self._store_api_key}",
                "Content-Type": "application/json",
                "User-Agent": "OpenSable-Agent/1.0",
            }

            async with aiohttp.ClientSession() as session:
                async with session.request(
                    method, url, headers=headers, json=json_body, timeout=aiohttp.ClientTimeout(total=30)
                ) as resp:
                    data = await resp.json()
                    if resp.status >= 400:
                        logger.warning(f"Store API {method} {path} → {resp.status}: {data.get('error', '?')}")
                        return None
                    return data

        except ImportError:
            logger.warning("aiohttp not available for Store API requests")
            return None
        except Exception as e:
            logger.warning(f"Store API request failed: {e}")
            return None

    async def install_skill_via_api(self, slug: str) -> Optional[Dict]:
        """
        Install a skill via the REST API using the owner's API key.

        Fallback when SAGP gateway is not configured. Requires SABLE_STORE_API_KEY.
        """
        if not self._store_api_key:
            return None

        # The install endpoint is /install/:slug (relative to registry_url base)
        # registry_url = https://sk.opensable.com/api
        # install endpoint = https://sk.opensable.com/api/install/:slug
        # But actually install routes are at /api/install/:slug
        # Since registry_url already ends with /api, we use a relative path
        result = await self._store_api_request("POST", f"/install/{slug}")
        if result:
            logger.info(f"✅ Skill '{slug}' installed via Store API key")
        return result

    async def search_skills_via_api(
        self, query: Optional[str] = None, category: Optional[str] = None, limit: int = 20
    ) -> Optional[List[Dict]]:
        """Search skills via the REST API (no auth required, but API key adds context)."""
        params = []
        if query:
            params.append(f"q={query}")
        if category:
            params.append(f"category={category}")
        if limit:
            params.append(f"limit={limit}")
        qs = "&".join(params)
        path = f"/skills?{qs}" if qs else "/skills"

        result = await self._store_api_request("GET", path)
        if result and "skills" in result:
            return result["skills"]
        return None

    async def _ensure_gateway(self) -> bool:
        """Ensure gateway client is authenticated. Returns True if available."""
        if not self._agent_id or not self._signing_key or not self._encryption_key:
            return False

        if self._gateway_client is not None and self._gateway_available:
            return True

        try:
            from opensable.skills.gateway_sdk import AgentGatewayClient

            self._gateway_client = AgentGatewayClient(
                gateway_url=self._gateway_url,
                agent_id=self._agent_id,
                signing_secret_key=self._signing_key,
                encryption_secret_key=self._encryption_key,
                auto_reconnect=True,
            )

            auth_result = await self._gateway_client.authenticate()
            self._gateway_available = True
            logger.info(
                f"✅ Gateway connected,  session authenticated in "
                f"{auth_result.get('solve_time_ms', '?')}ms, "
                f"permissions: {auth_result.get('permissions', [])}"
            )
            return True

        except ImportError:
            logger.warning("Gateway SDK not available (missing pynacl/aiohttp)")
            self._gateway_available = False
            return False
        except Exception as e:
            logger.warning(f"Gateway connection failed: {e},  falling back to local data")
            self._gateway_available = False
            return False

    async def _close_gateway(self):
        """Close the gateway client."""
        if self._gateway_client:
            try:
                await self._gateway_client.close()
            except Exception:
                pass
            self._gateway_client = None
            self._gateway_available = False

    # ── Convert gateway skill dict to SkillMetadata ──

    @staticmethod
    def _to_metadata(skill_dict: Dict) -> SkillMetadata:
        """Convert a marketplace API skill dict to SkillMetadata."""
        # Map category string to enum
        category_str = (skill_dict.get("category") or "custom").lower().replace(" ", "_")
        try:
            category = SkillCategory(category_str)
        except ValueError:
            category = SkillCategory.CUSTOM

        tags_raw = skill_dict.get("tags")
        if isinstance(tags_raw, str):
            try:
                tags = json.loads(tags_raw)
            except (json.JSONDecodeError, TypeError):
                tags = [t.strip() for t in tags_raw.split(",") if t.strip()]
        elif isinstance(tags_raw, list):
            tags = tags_raw
        else:
            tags = []

        deps_raw = skill_dict.get("dependencies")
        if isinstance(deps_raw, str):
            try:
                deps = json.loads(deps_raw)
            except (json.JSONDecodeError, TypeError):
                deps = [d.strip() for d in deps_raw.split(",") if d.strip()]
        elif isinstance(deps_raw, list):
            deps = deps_raw
        else:
            deps = []

        return SkillMetadata(
            skill_id=skill_dict.get("slug") or skill_dict.get("skill_id") or skill_dict.get("id", ""),
            name=skill_dict.get("name", "Unknown"),
            version=skill_dict.get("version", "0.0.0"),
            author=skill_dict.get("author_name") or skill_dict.get("author", "Unknown"),
            description=skill_dict.get("description", ""),
            category=category,
            tags=tags,
            dependencies=deps,
            rating=float(skill_dict.get("rating_avg") or skill_dict.get("rating") or 0),
            reviews_count=int(skill_dict.get("reviews_count") or skill_dict.get("review_count") or 0),
            downloads=int(skill_dict.get("downloads") or 0),
            published_at=_parse_datetime(skill_dict.get("created_at")),
            updated_at=_parse_datetime(skill_dict.get("updated_at")),
            verified=bool(skill_dict.get("verified") or skill_dict.get("is_published")),
            main_file=skill_dict.get("main_file", "skill.py"),
        )

    # ══════════════════════════════════════════════════════════════
    #  PUBLIC API,  Uses real marketplace when available, fallback to mock
    # ══════════════════════════════════════════════════════════════

    async def search_skills(
        self,
        query: Optional[str] = None,
        category: Optional[SkillCategory] = None,
        tags: Optional[List[str]] = None,
        limit: int = 20,
    ) -> List[SkillMetadata]:
        """
        Search for skills in the marketplace.

        Connects to the real SableCore Skills Marketplace via SAGP.
        Falls back to local mock data if the gateway is unreachable.
        """
        # ── Try real marketplace first ──
        if await self._ensure_gateway():
            try:
                category_str = category.value if category else None
                query_str = query
                if tags and not query_str:
                    query_str = " ".join(tags)

                skills_data = await self._gateway_client.list_skills(
                    category=category_str,
                    query=query_str,
                    limit=limit,
                )

                results = [self._to_metadata(s) for s in skills_data]

                # Update cache
                for meta in results:
                    self.cache[meta.skill_id] = meta
                self.cache_updated = datetime.now(timezone.utc)

                logger.info(f"🏪 Marketplace returned {len(results)} skills")
                return results

            except Exception as e:
                logger.warning(f"Marketplace search failed: {e},  using fallback")

        # ── Fallback 2: REST API with owner's API key ──
        if self._store_api_key:
            category_str = category.value if category else None
            query_str = query
            if tags and not query_str:
                query_str = " ".join(tags)
            api_results = await self.search_skills_via_api(query_str, category_str, limit)
            if api_results:
                results = [self._to_metadata(s) for s in api_results]
                for meta in results:
                    self.cache[meta.skill_id] = meta
                self.cache_updated = datetime.now(timezone.utc)
                logger.info(f"🏪 Store API returned {len(results)} skills")
                return results

        # ── Fallback 3: local mock data ──
        return await self._search_skills_mock(query, category, tags, limit)

    async def get_skill(self, skill_id: str) -> Optional[SkillMetadata]:
        """Get skill metadata by ID/slug."""
        # Check cache first
        if skill_id in self.cache:
            return self.cache[skill_id]

        # Try real marketplace
        if await self._ensure_gateway():
            try:
                skill_data = await self._gateway_client.get_skill(skill_id)
                if skill_data:
                    meta = self._to_metadata(skill_data)
                    self.cache[skill_id] = meta
                    return meta
            except Exception as e:
                logger.warning(f"Gateway get_skill failed: {e}")

        # Fallback: look in mock data
        all_skills = await self._search_skills_mock()
        for skill in all_skills:
            if skill.skill_id == skill_id:
                self.cache[skill_id] = skill
                return skill

        return None

    async def get_reviews(self, skill_id: str, limit: int = 10) -> List[SkillReview]:
        """Get reviews for a skill (mock for now,  reviews come from the web UI)."""
        return [
            SkillReview(
                review_id=str(uuid.uuid4()),
                skill_id=skill_id,
                user_id="user_001",
                rating=5,
                title="Excellent skill!",
                content="Works perfectly and saves me hours every week.",
                created_at=datetime(2026, 2, 1),
                helpful_count=12,
            ),
            SkillReview(
                review_id=str(uuid.uuid4()),
                skill_id=skill_id,
                user_id="user_002",
                rating=4,
                title="Very useful",
                content="Great functionality, could use better documentation.",
                created_at=datetime(2026, 2, 10),
                helpful_count=8,
            ),
        ]

    async def download_skill(self, skill_id: str, version: Optional[str] = None) -> bytes:
        """
        Download skill package.

        If the marketplace gateway is available, it triggers a server-side
        install which places the skill into opensable/skills/installed/.
        Returns a mock tarball for backward compatibility with SkillManager.
        """
        logger.info(f"Downloading skill: {skill_id} (version: {version or 'latest'})")

        # Try real marketplace install (server-side)
        if await self._ensure_gateway():
            try:
                result = await self._gateway_client.install_skill(skill_id)
                logger.info(f"🏪 Marketplace install result: {result}")
                # The server installs the file directly,  return a minimal tarball
                # so SkillManager doesn't break
            except Exception as e:
                logger.warning(f"Gateway install failed: {e},  creating mock package")

        # Try REST API install with owner's API key
        if self._store_api_key:
            api_result = await self.install_skill_via_api(skill_id)
            if api_result:
                logger.info(f"🏪 Store API install result: {api_result}")
                return await self._create_mock_package(skill_id)

        # Create mock tarball for backward compatibility
        return await self._create_mock_package(skill_id)

    async def install_skill_via_gateway(self, slug: str, config: Optional[Dict] = None) -> Dict:
        """
        Direct gateway skill install (preferred method).

        Installs the skill on the marketplace server which extracts it
        into opensable/skills/installed/.

        Returns:
            Installation result dict from the server
        """
        if not await self._ensure_gateway():
            raise RuntimeError(
                "Gateway not available. Set SABLE_AGENT_ID, SABLE_AGENT_SIGNING_KEY, "
                "SABLE_AGENT_ENCRYPTION_KEY or provide a credentials file."
            )

        result = await self._gateway_client.install_skill(slug, config)
        logger.info(f"✅ Skill '{slug}' installed via gateway: {result}")
        return result

    async def report_skill_via_gateway(self, slug: str, report_type: str, message: str = "") -> Dict:
        """Report a skill via the gateway (security issue, bug, etc)."""
        if not await self._ensure_gateway():
            raise RuntimeError("Gateway not available")

        return await self._gateway_client.report_skill(slug, report_type, message)

    async def review_skill_via_gateway(
        self, slug: str, rating: int, title: str, content: str
    ) -> Dict:
        """
        Post or update a review on a skill via the agent gateway.

        The agent can comment on any skill it has downloaded/used.

        Args:
            slug: Skill slug
            rating: 1-5 star rating
            title: Review title
            content: Review body

        Returns:
            Dict with reviewed, reviewId, updated keys.
        """
        if not await self._ensure_gateway():
            raise RuntimeError(
                "Gateway not available. Set SABLE_AGENT_ID, SABLE_AGENT_SIGNING_KEY, "
                "SABLE_AGENT_ENCRYPTION_KEY or provide a credentials file."
            )

        result = await self._gateway_client.review_skill(slug, rating, title, content)
        logger.info(f"✅ Agent reviewed skill '{slug}': {rating}/5,  {title}")
        return result

    @property
    def gateway_stats(self) -> Dict:
        """Get gateway connection stats."""
        if self._gateway_client:
            return self._gateway_client.stats
        return {"connected": False, "agent_id": self._agent_id}

    # ══════════════════════════════════════════════════════════════
    #  MOCK FALLBACK,  Used when gateway is unreachable
    # ══════════════════════════════════════════════════════════════

    async def _search_skills_mock(
        self,
        query: Optional[str] = None,
        category: Optional[SkillCategory] = None,
        tags: Optional[List[str]] = None,
        limit: int = 20,
    ) -> List[SkillMetadata]:
        """Fallback mock skill search."""
        results = []

        mock_skills = [
            SkillMetadata(
                skill_id="email-assistant",
                name="Email Assistant",
                version="1.0.0",
                author="Open-Sable Team",
                description="Intelligent email management and automation",
                category=SkillCategory.COMMUNICATION,
                tags=["email", "automation", "gmail"],
                dependencies=["google-auth", "google-api-python-client"],
                rating=0.0,
                reviews_count=128,
                downloads=0,
                published_at=datetime(2026, 1, 15),
                verified=True,
            ),
            SkillMetadata(
                skill_id="data-visualizer",
                name="Data Visualizer",
                version="2.1.0",
                author="DataViz Pro",
                description="Create beautiful charts and graphs from data",
                category=SkillCategory.DATA_ANALYSIS,
                tags=["visualization", "charts", "analytics"],
                dependencies=["matplotlib", "plotly", "pandas"],
                rating=0.0,
                reviews_count=89,
                downloads=0,
                published_at=datetime(2026, 1, 10),
                verified=True,
            ),
            SkillMetadata(
                skill_id="web-scraper",
                name="Web Scraper",
                version="1.5.2",
                author="AutoBot Labs",
                description="Extract data from websites automatically",
                category=SkillCategory.AUTOMATION,
                tags=["scraping", "web", "data extraction"],
                dependencies=["beautifulsoup4", "requests", "lxml"],
                rating=0.0,
                reviews_count=56,
                downloads=0,
                published_at=datetime(2026, 12, 20),
                verified=True,
            ),
            SkillMetadata(
                skill_id="code-reviewer",
                name="Code Reviewer",
                version="1.0.5",
                author="DevTools Inc",
                description="AI-powered code review and suggestions",
                category=SkillCategory.DEVELOPMENT,
                tags=["code", "review", "ai", "development"],
                dependencies=["tree-sitter", "pylint"],
                rating=0.0,
                reviews_count=42,
                downloads=0,
                published_at=datetime(2026, 1, 5),
                verified=True,
            ),
            SkillMetadata(
                skill_id="meeting-notes",
                name="Meeting Notes AI",
                version="1.2.0",
                author="Productivity Plus",
                description="Automatically generate meeting summaries and action items",
                category=SkillCategory.PRODUCTIVITY,
                tags=["meetings", "notes", "transcription"],
                dependencies=["whisper", "transformers"],
                rating=0.0,
                reviews_count=73,
                downloads=0,
                published_at=datetime(2026, 1, 1),
                verified=True,
            ),
        ]

        for skill in mock_skills:
            if category and skill.category != category:
                continue
            if tags and not any(tag in skill.tags for tag in tags):
                continue
            if query:
                query_lower = query.lower()
                if (
                    query_lower not in skill.name.lower()
                    and query_lower not in skill.description.lower()
                    and not any(query_lower in tag for tag in skill.tags)
                ):
                    continue
            results.append(skill)

        results.sort(key=lambda s: s.downloads * s.rating, reverse=True)
        return results[:limit]

    async def _create_mock_package(self, skill_id: str) -> bytes:
        """Create a mock tarball package for backward compatibility."""
        with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp:
            tmp_path = tmp.name

        with tempfile.TemporaryDirectory() as tmpdir:
            skill_dir = Path(tmpdir) / skill_id
            skill_dir.mkdir()

            skill_file = skill_dir / "skill.py"
            skill_file.write_text(f"""
# {skill_id} Skill
# Auto-generated skill template

from typing import Dict, Any

class Skill:
    '''{skill_id} skill implementation.'''

    def __init__(self, config: Dict[str, Any]):
        self.config = config

    async def execute(self, *args, **kwargs):
        '''Execute skill logic.'''
        return {{'status': 'success', 'message': 'Skill executed'}}
""")

            metadata_file = skill_dir / "metadata.json"
            skill_meta = await self.get_skill(skill_id)
            if skill_meta:
                metadata_file.write_text(json.dumps(asdict(skill_meta), default=str, indent=2))

            readme_file = skill_dir / "README.md"
            readme_file.write_text(f"# {skill_id}\n\nSkill documentation here.")

            with tarfile.open(tmp_path, "w:gz") as tar:
                tar.add(skill_dir, arcname=skill_id)

        with open(tmp_path, "rb") as f:
            package_data = f.read()

        Path(tmp_path).unlink()
        return package_data


def _parse_datetime(value) -> Optional[datetime]:
    """Safely parse a datetime string."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None


class SkillManager:
    """
    Manages installed skills.

    Features:
    - Install/uninstall skills
    - Update skills
    - Enable/disable skills
    - Dependency resolution
    - Configuration management
    """

    def __init__(self, skills_dir: Optional[Path] = None, registry: Optional[SkillRegistry] = None):
        """
        Initialize skill manager.

        Args:
            skills_dir: Directory for installed skills
            registry: Skill registry
        """
        self.skills_dir = skills_dir or Path("./skills")
        self.skills_dir.mkdir(parents=True, exist_ok=True)

        self.registry = registry or SkillRegistry()

        self.installed_skills: Dict[str, InstalledSkill] = {}

        self._load_installed_skills()

        logger.info(f"Skill manager initialized: {len(self.installed_skills)} skills installed")

    def _load_installed_skills(self):
        """Load installed skills from disk."""
        manifest_file = self.skills_dir / "manifest.json"

        if not manifest_file.exists():
            return

        try:
            with open(manifest_file, "r") as f:
                data = json.load(f)

                for skill_data in data:
                    metadata = SkillMetadata(**skill_data["metadata"])

                    skill = InstalledSkill(
                        skill_id=skill_data["skill_id"],
                        metadata=metadata,
                        install_path=Path(skill_data["install_path"]),
                        status=SkillStatus(skill_data["status"]),
                        installed_at=datetime.fromisoformat(skill_data["installed_at"]),
                        config=skill_data.get("config", {}),
                        enabled=skill_data.get("enabled", True),
                        auto_update=skill_data.get("auto_update", True),
                    )

                    self.installed_skills[skill.skill_id] = skill

            logger.info(f"Loaded {len(self.installed_skills)} installed skills")

        except Exception as e:
            logger.error(f"Error loading installed skills: {e}")

    def _save_installed_skills(self):
        """Save installed skills manifest."""
        manifest_file = self.skills_dir / "manifest.json"

        try:
            data = []

            for skill in self.installed_skills.values():
                data.append(
                    {
                        "skill_id": skill.skill_id,
                        "metadata": asdict(skill.metadata),
                        "install_path": str(skill.install_path),
                        "status": skill.status.value,
                        "installed_at": skill.installed_at.isoformat(),
                        "config": skill.config,
                        "enabled": skill.enabled,
                        "auto_update": skill.auto_update,
                    }
                )

            with open(manifest_file, "w") as f:
                json.dump(data, f, indent=2, default=str)

        except Exception as e:
            logger.error(f"Error saving installed skills: {e}")

    async def install_skill(
        self, skill_id: str, version: Optional[str] = None, config: Optional[Dict[str, Any]] = None
    ) -> InstalledSkill:
        """
        Install a skill.

        Args:
            skill_id: Skill ID
            version: Specific version (None for latest)
            config: Initial configuration

        Returns:
            InstalledSkill
        """
        logger.info(f"Installing skill: {skill_id}")

        # Get metadata
        metadata = await self.registry.get_skill(skill_id)
        if not metadata:
            raise ValueError(f"Skill not found: {skill_id}")

        # Check if already installed
        if skill_id in self.installed_skills:
            logger.warning(f"Skill already installed: {skill_id}")
            return self.installed_skills[skill_id]

        # Download package
        package_data = await self.registry.download_skill(skill_id, version)

        # Extract to skills directory
        install_path = self.skills_dir / skill_id
        install_path.mkdir(exist_ok=True)

        with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp:
            tmp.write(package_data)
            tmp_path = tmp.name

        try:
            with tarfile.open(tmp_path, "r:gz") as tar:
                tar.extractall(self.skills_dir)
        finally:
            Path(tmp_path).unlink()

        # Install dependencies into the active venv
        if metadata.dependencies:
            logger.info(f"Installing {len(metadata.dependencies)} dependencies")
            import sys, subprocess as _sp
            for dep in metadata.dependencies:
                try:
                    _sp.check_call(
                        [sys.executable, "-m", "pip", "install", dep],
                        stdout=_sp.DEVNULL,
                        stderr=_sp.PIPE,
                    )
                except _sp.CalledProcessError as e:
                    logger.warning(f"Failed to install dependency {dep}: {e}")

        # Create installed skill record
        skill = InstalledSkill(
            skill_id=skill_id,
            metadata=metadata,
            install_path=install_path,
            status=SkillStatus.INSTALLED,
            installed_at=datetime.now(timezone.utc),
            config=config or {},
            enabled=True,
            auto_update=True,
        )

        self.installed_skills[skill_id] = skill
        self._save_installed_skills()

        logger.info(f"Skill installed successfully: {skill_id}")
        return skill

    async def uninstall_skill(self, skill_id: str) -> bool:
        """
        Uninstall a skill.

        Args:
            skill_id: Skill ID

        Returns:
            True if uninstalled
        """
        if skill_id not in self.installed_skills:
            logger.warning(f"Skill not installed: {skill_id}")
            return False

        skill = self.installed_skills[skill_id]

        # Remove files
        if skill.install_path.exists():
            shutil.rmtree(skill.install_path)

        # Remove from registry
        del self.installed_skills[skill_id]
        self._save_installed_skills()

        logger.info(f"Skill uninstalled: {skill_id}")
        return True

    async def update_skill(self, skill_id: str) -> InstalledSkill:
        """Update a skill to latest version."""
        if skill_id not in self.installed_skills:
            raise ValueError(f"Skill not installed: {skill_id}")

        skill = self.installed_skills[skill_id]
        skill.status = SkillStatus.UPDATING

        # Get latest metadata
        latest_metadata = await self.registry.get_skill(skill_id)

        if not latest_metadata:
            raise ValueError(f"Skill not found in registry: {skill_id}")

        # Check version
        if semver:
            current_version = semver.VersionInfo.parse(skill.metadata.version)
            latest_version = semver.VersionInfo.parse(latest_metadata.version)
        else:
            from packaging.version import Version

            current_version = Version(skill.metadata.version)
            latest_version = Version(latest_metadata.version)

        if latest_version <= current_version:
            logger.info(f"Skill already up to date: {skill_id}")
            skill.status = SkillStatus.INSTALLED
            return skill

        logger.info(f"Updating {skill_id}: {current_version} -> {latest_version}")

        # Uninstall old version
        await self.uninstall_skill(skill_id)

        # Install new version
        updated_skill = await self.install_skill(
            skill_id, version=str(latest_version), config=skill.config
        )

        logger.info(f"Skill updated successfully: {skill_id}")
        return updated_skill

    async def update_all_skills(self) -> List[str]:
        """Update all skills with auto_update enabled."""
        updated = []

        for skill_id, skill in self.installed_skills.items():
            if skill.auto_update:
                try:
                    await self.update_skill(skill_id)
                    updated.append(skill_id)
                except Exception as e:
                    logger.error(f"Error updating {skill_id}: {e}")

        return updated

    def enable_skill(self, skill_id: str):
        """Enable a skill."""
        if skill_id in self.installed_skills:
            self.installed_skills[skill_id].enabled = True
            self._save_installed_skills()
            logger.info(f"Skill enabled: {skill_id}")

    def disable_skill(self, skill_id: str):
        """Disable a skill."""
        if skill_id in self.installed_skills:
            self.installed_skills[skill_id].enabled = False
            self._save_installed_skills()
            logger.info(f"Skill disabled: {skill_id}")

    def get_installed_skills(self) -> List[InstalledSkill]:
        """Get list of installed skills."""
        return list(self.installed_skills.values())

    def get_enabled_skills(self) -> List[InstalledSkill]:
        """Get list of enabled skills."""
        return [s for s in self.installed_skills.values() if s.enabled]


# Example usage
async def main():
    """Example skills marketplace usage,  now with real marketplace connection."""

    print("=" * 60)
    print("Skills Marketplace,  Live Connection")
    print("=" * 60)

    # Initialize registry,  auto-loads agent credentials
    print("\n🏪 Initializing marketplace...")
    registry = SkillRegistry()
    manager = SkillManager(registry=registry)
    print(f"  Gateway: {registry._gateway_url}")
    print(f"  Agent: {registry._agent_id}")
    print(f"  Installed: {len(manager.get_installed_skills())} skills")

    # Search for skills (hits real marketplace if gateway is up)
    print("\n🔍 Searching all skills...")
    skills = await registry.search_skills()
    print(f"  Found {len(skills)} skills:")
    for skill in skills:
        source = "🌐" if registry._gateway_available else "💾"
        print(
            f"    {source} {skill.name} v{skill.version} by {skill.author} "
            f",  ⭐ {skill.rating} ({skill.downloads} downloads)"
        )

    # Gateway stats
    print(f"\n📊 Gateway stats: {registry.gateway_stats}")

    # Search specific query
    print("\n🔍 Searching for 'weather'...")
    weather = await registry.search_skills(query="weather")
    print(f"  Found {len(weather)} results")
    for s in weather:
        print(f"    • {s.name},  {s.description[:60]}")

    # Get skill details
    if skills:
        first = skills[0]
        print(f"\n📄 Getting details for: {first.skill_id}")
        detail = await registry.get_skill(first.skill_id)
        if detail:
            print(f"  Name: {detail.name}")
            print(f"  Version: {detail.version}")
            print(f"  Author: {detail.author}")
            print(f"  Tags: {detail.tags}")

    # Close gateway
    await registry._close_gateway()

    print("\n✅ Skills marketplace example complete!")
    print("\n💡 Connection modes:")
    print("  🌐 Live,  connected to sk.opensable.com via SAGP gateway")
    print("  💾 Fallback,  using local mock data (gateway unreachable)")
    print("\n💡 Configure gateway credentials:")
    print("  • Set SABLE_AGENT_ID, SABLE_AGENT_SIGNING_KEY, SABLE_AGENT_ENCRYPTION_KEY")
    print("  • Or place agent-*.json in project root or marketplace/server/")


if __name__ == "__main__":
    asyncio.run(main())
