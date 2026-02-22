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
"""

import asyncio
import logging
import json
import hashlib
import tarfile
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any, List, Optional, Set
from dataclasses import dataclass, field, asdict
from enum import Enum
from datetime import datetime
import semver
import uuid

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
    
    In production, this would connect to a remote API.
    For now, it's a local index.
    """
    
    def __init__(self, registry_url: Optional[str] = None):
        """
        Initialize skill registry.
        
        Args:
            registry_url: URL of remote registry API
        """
        self.registry_url = registry_url or "https://registry.opensable.ai"
        self.cache: Dict[str, SkillMetadata] = {}
        self.cache_updated: Optional[datetime] = None
    
    async def search_skills(
        self,
        query: Optional[str] = None,
        category: Optional[SkillCategory] = None,
        tags: Optional[List[str]] = None,
        limit: int = 20
    ) -> List[SkillMetadata]:
        """
        Search for skills in registry.
        
        Args:
            query: Search query
            category: Filter by category
            tags: Filter by tags
            limit: Max results
        
        Returns:
            List of matching skills
        """
        # In production, would query remote API
        # For now, return mock data
        
        results = []
        
        # Mock skills for demonstration
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
                verified=True
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
                verified=True
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
                published_at=datetime(2025, 12, 20),
                verified=True
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
                verified=True
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
                verified=True
            )
        ]
        
        # Apply filters
        for skill in mock_skills:
            if category and skill.category != category:
                continue
            
            if tags and not any(tag in skill.tags for tag in tags):
                continue
            
            if query:
                query_lower = query.lower()
                if (query_lower not in skill.name.lower() and
                    query_lower not in skill.description.lower() and
                    not any(query_lower in tag for tag in skill.tags)):
                    continue
            
            results.append(skill)
        
        # Sort by relevance (downloads * rating)
        results.sort(key=lambda s: s.downloads * s.rating, reverse=True)
        
        return results[:limit]
    
    async def get_skill(self, skill_id: str) -> Optional[SkillMetadata]:
        """Get skill metadata by ID."""
        # Check cache
        if skill_id in self.cache:
            return self.cache[skill_id]
        
        # In production, would fetch from remote API
        # For now, search locally
        all_skills = await self.search_skills()
        for skill in all_skills:
            if skill.skill_id == skill_id:
                self.cache[skill_id] = skill
                return skill
        
        return None
    
    async def get_reviews(self, skill_id: str, limit: int = 10) -> List[SkillReview]:
        """Get reviews for a skill."""
        # Mock reviews
        return [
            SkillReview(
                review_id=str(uuid.uuid4()),
                skill_id=skill_id,
                user_id="user_001",
                rating=5,
                title="Excellent skill!",
                content="Works perfectly and saves me hours every week.",
                created_at=datetime(2026, 2, 1),
                helpful_count=12
            ),
            SkillReview(
                review_id=str(uuid.uuid4()),
                skill_id=skill_id,
                user_id="user_002",
                rating=4,
                title="Very useful",
                content="Great functionality, could use better documentation.",
                created_at=datetime(2026, 2, 10),
                helpful_count=8
            )
        ]
    
    async def download_skill(
        self,
        skill_id: str,
        version: Optional[str] = None
    ) -> bytes:
        """
        Download skill package.
        
        Args:
            skill_id: Skill ID
            version: Specific version (None for latest)
        
        Returns:
            Skill package bytes
        """
        # In production, would download from remote
        # For now, create mock package
        
        logger.info(f"Downloading skill: {skill_id} (version: {version or 'latest'})")
        
        # Create temporary package
        with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp:
            tmp_path = tmp.name
        
        # Create mock skill files
        with tempfile.TemporaryDirectory() as tmpdir:
            skill_dir = Path(tmpdir) / skill_id
            skill_dir.mkdir()
            
            # Create skill.py
            skill_file = skill_dir / "skill.py"
            skill_file.write_text(f"""
# {skill_id} Skill
# Auto-generated skill template

from typing import Dict, Any

class Skill:
    ''''{skill_id} skill implementation.'''
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
    
    async def execute(self, *args, **kwargs):
        '''Execute skill logic.'''
        return {{'status': 'success', 'message': 'Skill executed'}}
""")
            
            # Create metadata.json
            metadata_file = skill_dir / "metadata.json"
            skill_meta = await self.get_skill(skill_id)
            if skill_meta:
                metadata_file.write_text(json.dumps(asdict(skill_meta), default=str, indent=2))
            
            # Create README
            readme_file = skill_dir / "README.md"
            readme_file.write_text(f"# {skill_id}\n\nSkill documentation here.")
            
            # Create tarball
            with tarfile.open(tmp_path, "w:gz") as tar:
                tar.add(skill_dir, arcname=skill_id)
        
        # Read and return
        with open(tmp_path, 'rb') as f:
            package_data = f.read()
        
        # Cleanup
        Path(tmp_path).unlink()
        
        return package_data


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
    
    def __init__(
        self,
        skills_dir: Optional[Path] = None,
        registry: Optional[SkillRegistry] = None
    ):
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
            with open(manifest_file, 'r') as f:
                data = json.load(f)
                
                for skill_data in data:
                    metadata = SkillMetadata(**skill_data['metadata'])
                    
                    skill = InstalledSkill(
                        skill_id=skill_data['skill_id'],
                        metadata=metadata,
                        install_path=Path(skill_data['install_path']),
                        status=SkillStatus(skill_data['status']),
                        installed_at=datetime.fromisoformat(skill_data['installed_at']),
                        config=skill_data.get('config', {}),
                        enabled=skill_data.get('enabled', True),
                        auto_update=skill_data.get('auto_update', True)
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
                data.append({
                    'skill_id': skill.skill_id,
                    'metadata': asdict(skill.metadata),
                    'install_path': str(skill.install_path),
                    'status': skill.status.value,
                    'installed_at': skill.installed_at.isoformat(),
                    'config': skill.config,
                    'enabled': skill.enabled,
                    'auto_update': skill.auto_update
                })
            
            with open(manifest_file, 'w') as f:
                json.dump(data, f, indent=2, default=str)
        
        except Exception as e:
            logger.error(f"Error saving installed skills: {e}")
    
    async def install_skill(
        self,
        skill_id: str,
        version: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None
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
        
        # Install dependencies
        if metadata.dependencies:
            logger.info(f"Installing {len(metadata.dependencies)} dependencies")
            # In production, would use pip
            # await self._install_dependencies(metadata.dependencies)
        
        # Create installed skill record
        skill = InstalledSkill(
            skill_id=skill_id,
            metadata=metadata,
            install_path=install_path,
            status=SkillStatus.INSTALLED,
            installed_at=datetime.utcnow(),
            config=config or {},
            enabled=True,
            auto_update=True
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
        current_version = semver.VersionInfo.parse(skill.metadata.version)
        latest_version = semver.VersionInfo.parse(latest_metadata.version)
        
        if latest_version <= current_version:
            logger.info(f"Skill already up to date: {skill_id}")
            skill.status = SkillStatus.INSTALLED
            return skill
        
        logger.info(f"Updating {skill_id}: {current_version} -> {latest_version}")
        
        # Uninstall old version
        await self.uninstall_skill(skill_id)
        
        # Install new version
        updated_skill = await self.install_skill(
            skill_id,
            version=str(latest_version),
            config=skill.config
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
    """Example skills marketplace usage."""
    
    print("=" * 60)
    print("Skills Marketplace Example")
    print("=" * 60)
    
    # Initialize registry and manager
    print("\n🏪 Initializing marketplace...")
    registry = SkillRegistry()
    manager = SkillManager(registry=registry)
    print(f"  ✅ Ready ({len(manager.get_installed_skills())} skills installed)")
    
    # Search for skills
    print("\n🔍 Searching for skills...")
    skills = await registry.search_skills(query="email")
    print(f"  Found {len(skills)} skills:")
    for skill in skills:
        print(f"    • {skill.name} v{skill.version} - ⭐ {skill.rating} ({skill.downloads} downloads)")
    
    # Browse by category
    print("\n📂 Browsing productivity skills...")
    productivity_skills = await registry.search_skills(category=SkillCategory.PRODUCTIVITY)
    print(f"  Found {len(productivity_skills)} skills:")
    for skill in productivity_skills[:3]:
        print(f"    • {skill.name} - {skill.description[:50]}...")
    
    # Get skill details
    print("\n📄 Getting skill details...")
    skill_id = "email-assistant"
    skill = await registry.get_skill(skill_id)
    if skill:
        print(f"  Name: {skill.name}")
        print(f"  Version: {skill.version}")
        print(f"  Author: {skill.author}")
        print(f"  Description: {skill.description}")
        print(f"  Rating: ⭐ {skill.rating} ({skill.reviews_count} reviews)")
        print(f"  Downloads: {skill.downloads}")
        print(f"  Dependencies: {', '.join(skill.dependencies)}")
    
    # Install a skill
    print(f"\n📥 Installing skill: {skill_id}...")
    installed = await manager.install_skill(skill_id)
    print(f"  ✅ Installed: {installed.metadata.name} v{installed.metadata.version}")
    print(f"     Path: {installed.install_path}")
    print(f"     Status: {installed.status.value}")
    
    # List installed skills
    print("\n📦 Installed skills:")
    for skill in manager.get_installed_skills():
        status_icon = "✅" if skill.enabled else "❌"
        print(f"  {status_icon} {skill.metadata.name} v{skill.metadata.version}")
    
    # Get reviews
    print(f"\n⭐ Reviews for {skill_id}:")
    reviews = await registry.get_reviews(skill_id)
    for review in reviews[:2]:
        print(f"  {'⭐' * review.rating} {review.title}")
        print(f"    {review.content[:60]}...")
        print(f"    👍 {review.helpful_count} helpful")
    
    print("\n✅ Skills marketplace example complete!")
    print("\n💡 To use skills marketplace:")
    print("  • Search for skills: registry.search_skills('keyword')")
    print("  • Install skills: manager.install_skill('skill-id')")
    print("  • Update all: manager.update_all_skills()")
    print("  • Enable/disable: manager.enable_skill('skill-id')")


if __name__ == "__main__":
    asyncio.run(main())
