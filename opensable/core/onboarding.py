"""
Interactive Setup & Installation Wizard for SableCore
Helps new users install dependencies and configure their AI agent
"""

import asyncio
import logging
import sys
import subprocess
from typing import Dict, Any, List
from pathlib import Path
import json

logger = logging.getLogger(__name__)


class OnboardingWizard:
    """
    Interactive setup wizard for installation and configuration.

    Flow:
    1. Welcome & system check
    2. Install dependencies (optional)
    3. Choose use case (personal assistant, developer tool, business automation, etc)
    4. Select platforms (Telegram, WhatsApp, Discord, CLI)
    5. Configure API keys
    6. Choose skills to install (SableCore + community skills)
    7. Set up integrations
    8. Test connection
    """

    def __init__(self, config):
        self.config = config
        self.responses: Dict[str, Any] = {}

    async def start(self) -> Dict[str, Any]:
        """Start the onboarding wizard"""
        print("\n" + "=" * 60)
        print("🚀 Welcome to SableCore - Your AI Agent")
        print("=" * 60)
        print("\nInteractive Setup & Installation Wizard")
        print()

        # Step 0: System check & dependencies
        await self._check_system()

        # Step 1: Use case
        await self._select_use_case()

        # Step 2: Platforms
        await self._select_platforms()

        # Step 3: Configure API keys
        await self._configure_api_keys()

        # Step 4: Personality
        await self._configure_personality()

        # Step 5: Skills
        await self._install_skills()

        # Step 6: Integrations
        await self._setup_integrations()

        # Step 7: Summary & save
        await self._show_summary()

        return self.responses

    async def _check_system(self):
        """Step 0: Check system and offer to install dependencies"""
        print("📋 Step 0: System Check")
        print("=" * 60)
        print("\nChecking your system...")
        print()

        # Check Python version
        python_version = sys.version.split()[0]
        print(f"✅ Python {python_version}")

        # Check for venv
        in_venv = hasattr(sys, "real_prefix") or (
            hasattr(sys, "base_prefix") and sys.base_prefix != sys.prefix
        )
        if in_venv:
            print("✅ Virtual environment active")
        else:
            print("⚠️  Not in virtual environment (recommended)")

        # Check for key packages
        missing_packages = []
        optional_packages = {
            "PyPDF2": "PDF parsing",
            "pdfplumber": "Advanced PDF parsing",
            "pytesseract": "OCR support",
            "discord": "Discord bot",
            "telegram": "Telegram bot",
        }

        for package, description in optional_packages.items():
            try:
                __import__(package.lower())
                print(f"✅ {package} ({description})")
            except ImportError:
                missing_packages.append((package, description))
                print(f"❌ {package} ({description}) - not installed")

        print()

        if missing_packages:
            install = input("📦 Install missing packages? (Y/n): ").strip().lower()
            if install != "n":
                await self._install_packages([pkg for pkg, _ in missing_packages])

        print("\n✅ System check complete!\n")

    async def _install_packages(self, packages: List[str]):
        """Install missing packages"""
        print(f"\n📦 Installing {len(packages)} packages...")

        for package in packages:
            try:
                subprocess.run(
                    [sys.executable, "-m", "pip", "install", package],
                    check=True,
                    capture_output=True,
                )
                print(f"✅ Installed {package}")
            except subprocess.CalledProcessError:
                print(f"❌ Failed to install {package}")

        print()

    async def _configure_api_keys(self):
        """Step 3: Configure API keys"""
        print("📋 Step 3: API Keys Configuration")
        print("=" * 60)
        print()

        env_file = Path(__file__).parent.parent.parent / ".env"

        print("Configure your API keys (press Enter to skip):")
        print()

        api_keys = {
            "OPENAI_API_KEY": "OpenAI (GPT-4, GPT-3.5)",
            "ANTHROPIC_API_KEY": "Anthropic (Claude)",
            "TELEGRAM_BOT_TOKEN": "Telegram Bot (from @BotFather)",
            "DISCORD_BOT_TOKEN": "Discord Bot",
        }

        configured = {}

        for key, description in api_keys.items():
            value = input(f"{description}: ").strip()
            if value:
                configured[key] = value

        if configured:
            self.responses["api_keys"] = configured
            print(f"\n✅ Configured {len(configured)} API key(s)")

            # Save to .env
            if env_file.exists():
                with open(env_file, "a") as f:
                    f.write("\n# Added by onboarding wizard\n")
                    for key, value in configured.items():
                        f.write(f"{key}={value}\n")
                print(f"✅ Saved to {env_file}")
        else:
            print("\n⚠️  No API keys configured (you can add them later to .env)")

        print()

    async def _select_use_case(self):
        """Step 1: Select primary use case"""
        print("📋 Step 1: What will you use SableCore for?\n")

        use_cases = {
            "1": {
                "name": "Personal Assistant",
                "description": "Schedule management, reminders, web searches, general help",
                "recommended_skills": ["calendar", "reminders", "web_search"],
            },
            "2": {
                "name": "Developer Tool",
                "description": "Code generation, debugging, documentation, automation",
                "recommended_skills": ["code_executor", "github_integration", "api_tester"],
            },
            "3": {
                "name": "Business Automation",
                "description": "Email management, data processing, reporting",
                "recommended_skills": ["email_automation", "data_analyzer", "report_generator"],
            },
            "4": {
                "name": "Content Creator",
                "description": "Social media, blog posts, image generation",
                "recommended_skills": ["social_media_poster", "ai_image_gen", "content_writer"],
            },
            "5": {
                "name": "Research Assistant",
                "description": "Information gathering, summarization, knowledge management",
                "recommended_skills": ["web_scraper_pro", "pdf_analyzer", "knowledge_base"],
            },
        }

        for key, case in use_cases.items():
            print(f"{key}. {case['name']}")
            print(f"   → {case['description']}\n")

        choice = input("Choose your primary use case (1-5): ").strip()

        if choice in use_cases:
            self.responses["use_case"] = use_cases[choice]
            print(f"\n✅ Great! Setting up SableCore as a {use_cases[choice]['name']}\n")
        else:
            self.responses["use_case"] = use_cases["1"]
            print("\n✅ Using default: Personal Assistant\n")

    async def _select_platforms(self):
        """Step 2: Select messaging platforms"""
        print("📱 Step 2: Which platforms do you want to use?\n")

        platforms = {
            "1": {"name": "Telegram", "enabled": False, "requires": "TELEGRAM_BOT_TOKEN"},
            "2": {"name": "WhatsApp", "enabled": False, "requires": "Node.js + whatsapp-web.js"},
            "3": {"name": "Discord", "enabled": False, "requires": "DISCORD_BOT_TOKEN"},
            "4": {
                "name": "CLI (Terminal)",
                "enabled": True,
                "requires": "Nothing (always available)",
            },
        }

        print("Select platforms to enable (comma-separated, e.g., 1,2,4):\n")
        for key, platform in platforms.items():
            print(f"{key}. {platform['name']} - Requires: {platform['requires']}")

        print()
        choices = input("Your selection: ").strip().split(",")

        selected = []
        for choice in choices:
            choice = choice.strip()
            if choice in platforms:
                platforms[choice]["enabled"] = True
                selected.append(platforms[choice]["name"])

        self.responses["platforms"] = [p for p in platforms.values() if p["enabled"]]
        print(f"\n✅ Enabled platforms: {', '.join(selected) if selected else 'CLI only'}\n")

    async def _configure_personality(self):
        """Step 3: Configure agent personality"""
        print("🎭 Step 3: Choose your agent's personality\n")

        personalities = {
            "1": {"name": "Professional", "description": "Formal, efficient, business-focused"},
            "2": {"name": "Friendly", "description": "Warm, conversational, helpful"},
            "3": {"name": "Technical", "description": "Precise, detailed, developer-oriented"},
            "4": {"name": "Creative", "description": "Innovative, expressive, artistic"},
            "5": {"name": "Sarcastic", "description": "Witty, humorous, with attitude"},
        }

        for key, personality in personalities.items():
            print(f"{key}. {personality['name']}: {personality['description']}")

        print()
        choice = input("Select personality (1-5, default=2): ").strip() or "2"

        if choice in personalities:
            self.responses["personality"] = personalities[choice]
            print(f"\n✅ Personality set to: {personalities[choice]['name']}\n")
        else:
            self.responses["personality"] = personalities["2"]
            print("\n✅ Using default: Friendly\n")

    async def _install_skills(self):
        """Step 4: Install recommended skills"""
        print("🛠️  Step 4: Install Skills\n")

        recommended = self.responses.get("use_case", {}).get("recommended_skills", [])

        if recommended:
            print("Based on your use case, we recommend these skills:")
            for i, skill in enumerate(recommended, 1):
                print(f"  {i}. {skill.replace('_', ' ').title()}")
            print()

            choice = input("Install recommended skills? (Y/n): ").strip().lower()

            if choice != "n":
                self.responses["install_skills"] = recommended
                print("\n✅ Skills will be installed\n")
            else:
                self.responses["install_skills"] = []
        else:
            self.responses["install_skills"] = []

    async def _setup_integrations(self):
        """Step 5: Set up optional integrations"""
        print("🔗 Step 5: Optional Integrations\n")

        integrations = {
            "1": "Google Calendar",
            "2": "GitHub",
            "3": "OpenAI API",
            "4": "Spotify",
            "5": "Skip integrations",
        }

        print("Available integrations:")
        for key, integration in integrations.items():
            print(f"{key}. {integration}")

        print()
        choices = input("Select integrations (comma-separated, or 5 to skip): ").strip().split(",")

        selected = []
        for choice in choices:
            choice = choice.strip()
            if choice in integrations and choice != "5":
                selected.append(integrations[choice])

        self.responses["integrations"] = selected

        if selected:
            print(f"\n✅ Selected integrations: {', '.join(selected)}")
            print("⚠️  You'll need to add API keys to .env file\n")
        else:
            print("\n✅ Skipping integrations\n")

    async def _show_summary(self):
        """Step 6: Show configuration summary"""
        print("=" * 60)
        print("📊 Configuration Summary")
        print("=" * 60)
        print()

        use_case = self.responses.get("use_case", {})
        print(f"Use Case: {use_case.get('name', 'Personal Assistant')}")

        platforms = self.responses.get("platforms", [])
        platform_names = [p["name"] for p in platforms]
        print(f"Platforms: {', '.join(platform_names) if platform_names else 'CLI only'}")

        personality = self.responses.get("personality", {})
        print(f"Personality: {personality.get('name', 'Friendly')}")

        skills = self.responses.get("install_skills", [])
        if skills:
            print(f"Skills to install: {len(skills)}")

        integrations = self.responses.get("integrations", [])
        if integrations:
            print(f"Integrations: {', '.join(integrations)}")

        print()
        print("=" * 60)
        print()

        confirm = input("Save this configuration? (Y/n): ").strip().lower()

        if confirm != "n":
            await self._save_config()
            print("\n✅ Configuration saved!")
            print("\n🎉 Onboarding complete! Run 'python main.py' to start.\n")
        else:
            print("\n⚠️  Configuration not saved. Run the wizard again to reconfigure.\n")

    async def _save_config(self):
        """Save configuration to file"""
        config_file = Path(__file__).parent.parent.parent / "config" / "onboarding.json"
        config_file.parent.mkdir(parents=True, exist_ok=True)

        with open(config_file, "w") as f:
            json.dump(self.responses, f, indent=2, default=str)


async def run_onboarding():
    """Run the onboarding wizard"""
    from opensable.core.config import OpenSableConfig

    config = OpenSableConfig()
    wizard = OnboardingWizard(config)
    await wizard.start()


if __name__ == "__main__":
    asyncio.run(run_onboarding())
