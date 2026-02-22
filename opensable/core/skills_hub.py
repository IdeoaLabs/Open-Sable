"""
Sable Skills Hub - Skills Marketplace for SableCore
Includes SkillFactory for autonomous skill creation
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict, field
from datetime import datetime
import hashlib

from opensable.core.skill_factory import SkillFactory

logger = logging.getLogger(__name__)


@dataclass
class Skill:
    """
    SableCore Skill definition.
    """

    skill_id: str
    name: str
    description: str
    category: str
    code: str
    author: str
    version: str
    downloads: int = 0
    rating: float = 0.0
    tags: List[str] = field(default_factory=list)
    created_at: str = ""
    updated_at: str = ""
    dependencies: List[str] = field(default_factory=list)

    # Community skill fields
    community_format: bool = False  # If skill is from community catalog
    trigger_words: List[str] = field(default_factory=list)
    examples: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_catalog(cls, skill_entry: Dict) -> "Skill":
        """
        Convert community catalog skill entry to SableCore Skill.

        Catalog format:
        {
            "id": "skill_name",
            "name": "Skill Name",
            "description": "...",
            "category": "...",
            "function": "code...",
            "triggers": ["keyword1", "keyword2"],
            "examples": ["example usage"]
        }
        """
        return cls(
            skill_id=skill_entry.get("id", ""),
            name=skill_entry.get("name", ""),
            description=skill_entry.get("description", ""),
            category=skill_entry.get("category", "general"),
            code=skill_entry.get("function", skill_entry.get("code", "")),
            author=skill_entry.get("author", "SableCore Community"),
            version=skill_entry.get("version", "1.0.0"),
            downloads=skill_entry.get("downloads", 0),
            rating=skill_entry.get("rating", 0.0),
            tags=skill_entry.get("tags", []),
            trigger_words=skill_entry.get("triggers", []),
            examples=skill_entry.get("examples", []),
            community_format=True,
        )


class SkillsHub:
    """
    Skills marketplace for discovering, sharing, and installing skills.

    Features:
    - Browse skills by category
    - Search skills by keyword
    - Install SableCore and community skills
    - Rate and review skills
    - Auto-updates for installed skills
    - SkillFactory for autonomous skill creation
    """

    def __init__(self, config):
        self.config = config
        self.skills_dir = Path(__file__).parent.parent.parent / "opensable" / "skills"
        self.marketplace_dir = self.skills_dir / "marketplace"
        self.installed_dir = self.skills_dir / "installed"
        self.community_dir = self.skills_dir / "community"  # Community skills
        self.cache_file = self.marketplace_dir / "cache.json"

        # Create directories
        self.marketplace_dir.mkdir(parents=True, exist_ok=True)
        self.installed_dir.mkdir(parents=True, exist_ok=True)
        self.community_dir.mkdir(parents=True, exist_ok=True)

        self.skills_catalog: Dict[str, Skill] = {}

        # Skill Factory — autonomous skill creation engine
        self.factory = SkillFactory(config)

    async def initialize(self):
        """Initialize skills hub"""
        logger.info("🛒 Initializing Skills Hub...")
        await self._load_local_catalog()
        await self._load_community_skills()  # Load community skills catalog
        await self._sync_with_remote()
        logger.info(f"✅ Skills Hub ready ({len(self.skills_catalog)} skills available)")

    async def _load_community_skills(self):
        """Load community skills catalog"""
        catalog_file = self.community_dir / "skills_catalog.json"

        if catalog_file.exists():
            try:
                with open(catalog_file, "r") as f:
                    data = json.load(f)

                    for skill_entry in data.get("skills", []):
                        # Convert catalog entry to SableCore Skill
                        skill = Skill.from_catalog(skill_entry)
                        self.skills_catalog[skill.skill_id] = skill
                        logger.info(f"📦 Loaded community skill: {skill.name}")

            except Exception as e:
                logger.error(f"Failed to load community skills: {e}")

    async def _load_local_catalog(self):
        """Load locally cached skills catalog"""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, "r") as f:
                    data = json.load(f)
                    for skill_data in data.get("skills", []):
                        skill = Skill(**skill_data)
                        self.skills_catalog[skill.skill_id] = skill
                logger.info(f"Loaded {len(self.skills_catalog)} skills from cache")
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}")

    async def _sync_with_remote(self):
        """Sync with remote skills repository (GitHub, etc)"""
        # For now, we'll populate with example skills
        # In production, this would fetch from a remote API
        await self._populate_example_skills()

    async def _populate_example_skills(self):
        """Populate with example skills"""
        example_skills = [
            {
                "skill_id": "web_scraper_pro",
                "name": "Web Scraper Pro",
                "description": "Advanced web scraping with JavaScript rendering, proxies, and anti-bot bypass",
                "category": "web",
                "code": """async def scrape_advanced(url: str, render_js: bool = True):
    import aiohttp
    from bs4 import BeautifulSoup
    
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            html = await response.text()
            soup = BeautifulSoup(html, 'html.parser')
            return soup.get_text()
""",
                "author": "sable-community",
                "version": "1.2.0",
                "downloads": 1250,
                "rating": 4.8,
                "tags": ["web", "scraping", "data"],
                "dependencies": ["aiohttp", "beautifulsoup4", "playwright"],
            },
            {
                "skill_id": "crypto_tracker",
                "name": "Crypto Price Tracker",
                "description": "Real-time cryptocurrency price tracking with alerts and portfolio management",
                "category": "finance",
                "code": """async def track_crypto(symbol: str):
    import aiohttp
    
    async with aiohttp.ClientSession() as session:
        async with session.get(f'https://api.coingecko.com/api/v3/simple/price?ids={symbol}&vs_currencies=usd') as response:
            data = await response.json()
            return data
""",
                "author": "crypto-dev",
                "version": "2.0.1",
                "downloads": 890,
                "rating": 4.5,
                "tags": ["crypto", "finance", "trading"],
                "dependencies": ["aiohttp"],
            },
            {
                "skill_id": "ai_image_gen",
                "name": "AI Image Generator",
                "description": "Generate images using Stable Diffusion API integration",
                "category": "ai",
                "code": """async def generate_image(prompt: str, style: str = 'realistic'):
    # Integration with Stable Diffusion API
    return {'url': 'https://example.com/generated.png'}
""",
                "author": "ai-artist",
                "version": "1.0.0",
                "downloads": 2100,
                "rating": 4.9,
                "tags": ["ai", "image", "generation"],
                "dependencies": ["diffusers", "torch"],
            },
            {
                "skill_id": "email_automation",
                "name": "Email Automation Suite",
                "description": "Send emails, parse inbox, auto-reply with AI-generated responses",
                "category": "productivity",
                "code": """async def send_email(to: str, subject: str, body: str):
    import smtplib
    from email.mime.text import MIMEText
    
    # Email sending logic
    pass
""",
                "author": "productivity-guru",
                "version": "1.5.2",
                "downloads": 670,
                "rating": 4.3,
                "tags": ["email", "automation", "productivity"],
                "dependencies": ["aiosmtplib"],
            },
            {
                "skill_id": "social_media_poster",
                "name": "Social Media Multi-Poster",
                "description": "Post to Twitter, Instagram, LinkedIn, and Facebook simultaneously",
                "category": "social",
                "code": """async def post_to_all(content: str, image_url: str = None):
    # Multi-platform posting
    platforms = ['twitter', 'instagram', 'linkedin']
    for platform in platforms:
        await post_to_platform(platform, content, image_url)
""",
                "author": "social-master",
                "version": "3.1.0",
                "downloads": 1450,
                "rating": 4.7,
                "tags": ["social", "marketing", "automation"],
                "dependencies": ["tweepy", "instabot", "python-linkedin"],
            },
        ]

        for skill_data in example_skills:
            skill_data["created_at"] = datetime.utcnow().isoformat()
            skill_data["updated_at"] = datetime.utcnow().isoformat()
            skill = Skill(**skill_data)
            self.skills_catalog[skill.skill_id] = skill

        # Save to cache
        await self._save_cache()

    async def _save_cache(self):
        """Save catalog to cache"""
        try:
            data = {
                "skills": [skill.to_dict() for skill in self.skills_catalog.values()],
                "updated_at": datetime.utcnow().isoformat(),
            }
            with open(self.cache_file, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save cache: {e}")

    async def browse_skills(self, category: Optional[str] = None, limit: int = 10) -> List[Skill]:
        """Browse skills, optionally filtered by category"""
        skills = list(self.skills_catalog.values())

        if category:
            skills = [s for s in skills if s.category == category]

        # Sort by downloads
        skills.sort(key=lambda s: s.downloads, reverse=True)

        return skills[:limit]

    async def search_skills(self, query: str) -> List[Skill]:
        """Search skills by keyword"""
        query = query.lower()
        results = []

        for skill in self.skills_catalog.values():
            if (
                query in skill.name.lower()
                or query in skill.description.lower()
                or any(query in tag for tag in (skill.tags or []))
            ):
                results.append(skill)

        # Sort by rating
        results.sort(key=lambda s: s.rating, reverse=True)

        return results

    async def get_skill(self, skill_id: str) -> Optional[Skill]:
        """Get skill by ID"""
        return self.skills_catalog.get(skill_id)

    async def install_skill(self, skill_id: str) -> bool:
        """Install a skill from the marketplace"""
        skill = await self.get_skill(skill_id)

        if not skill:
            logger.error(f"Skill {skill_id} not found")
            return False

        try:
            # Create skill file
            skill_file = self.installed_dir / f"{skill_id}.py"

            with open(skill_file, "w") as f:
                f.write(f'''"""
{skill.name}
{skill.description}

Author: {skill.author}
Version: {skill.version}
"""

{skill.code}
''')

            # Install dependencies
            if skill.dependencies:
                logger.info(f"Installing dependencies: {', '.join(skill.dependencies)}")
                # In production, use pip to install dependencies
                # subprocess.run(['pip', 'install'] + skill.dependencies)

            # Update download count
            skill.downloads += 1
            await self._save_cache()

            logger.info(f"✅ Installed skill: {skill.name}")
            return True

        except Exception as e:
            logger.error(f"Failed to install skill {skill_id}: {e}")
            return False

    async def publish_skill(self, skill: Skill) -> bool:
        """Publish a skill to the marketplace"""
        try:
            # Generate skill ID
            if not skill.skill_id:
                skill.skill_id = hashlib.md5((skill.name + skill.author).encode()).hexdigest()[:12]

            skill.created_at = datetime.utcnow().isoformat()
            skill.updated_at = datetime.utcnow().isoformat()

            self.skills_catalog[skill.skill_id] = skill
            await self._save_cache()

            logger.info(f"✅ Published skill: {skill.name}")
            return True

        except Exception as e:
            logger.error(f"Failed to publish skill: {e}")
            return False

    async def get_installed_skills(self) -> List[str]:
        """Get list of installed skill IDs"""
        installed = []
        for file in self.installed_dir.glob("*.py"):
            if file.stem != "__init__":
                installed.append(file.stem)
        return installed

    async def get_categories(self) -> List[str]:
        """Get all skill categories"""
        categories = set()
        for skill in self.skills_catalog.values():
            categories.add(skill.category)
        return sorted(list(categories))

    def format_skill_info(self, skill: Skill) -> str:
        """Format skill information for display"""
        info = f"""
🎯 **{skill.name}** (v{skill.version})
📝 {skill.description}

👤 Author: {skill.author}
📊 Downloads: {skill.downloads:,}
⭐ Rating: {skill.rating}/5.0
🏷️  Tags: {', '.join(skill.tags or [])}
📦 ID: {skill.skill_id}

Dependencies: {', '.join(skill.dependencies or ['None'])}
"""
        return info.strip()

    # -------------------------------------------------------------------
    # Skill Factory integration — create new skills autonomously
    # -------------------------------------------------------------------

    async def create_skill(
        self,
        name: str,
        description: str,
        category: str = "general",
        triggers: Optional[List[str]] = None,
        examples: Optional[List[str]] = None,
        custom_code: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Create a new skill using the SkillFactory engine.

        The agent can call this to autonomously build, validate, and register
        a brand-new skill from just a name and description.

        Returns:
            dict with 'success', generated code, validation results, etc.
        """
        result = await self.factory.create_skill(
            name=name,
            description=description,
            category=category,
            triggers=triggers,
            examples=examples,
            custom_code=custom_code,
        )

        # If successful, also load into the live catalog
        if result.get("success"):
            skill = Skill(
                skill_id=result["slug"],
                name=name,
                description=description,
                category=category,
                code=result["code"],
                author="SableCore SkillFactory",
                version="1.0.0",
                trigger_words=triggers or [],
                examples=examples or [],
                community_format=True,
            )
            self.skills_catalog[skill.skill_id] = skill
            logger.info(f"🏭 Factory skill '{name}' added to live catalog")

        return result
