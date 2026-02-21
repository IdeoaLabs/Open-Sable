#!/usr/bin/env python3
"""
SableCore CLI - Command-line interface for all features
"""
import asyncio
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from opensable.core.config import OpenSableConfig
from opensable.core.skills_hub import SkillsHub
from opensable.core.onboarding import run_onboarding


async def show_skills_hub():
    """Show skills marketplace"""
    config = OpenSableConfig()
    hub = SkillsHub(config)
    await hub.initialize()
    
    while True:
        print("\n" + "="*60)
        print("🛒 Sable Skills Hub")
        print("="*60)
        print("\n1. Browse all skills")
        print("2. Search skills")
        print("3. View installed skills")
        print("4. Install a skill")
        print("5. Back to main menu")
        
        choice = input("\nSelect option: ").strip()
        
        if choice == "1":
            await browse_skills(hub)
        elif choice == "2":
            await search_skills(hub)
        elif choice == "3":
            await show_installed(hub)
        elif choice == "4":
            await install_skill(hub)
        elif choice == "5":
            break


async def browse_skills(hub):
    """Browse skills by category"""
    categories = await hub.get_categories()
    
    print("\n📂 Categories:")
    for i, cat in enumerate(categories, 1):
        print(f"{i}. {cat.title()}")
    print(f"{len(categories)+1}. All skills")
    
    choice = input("\nSelect category: ").strip()
    
    try:
        idx = int(choice) - 1
        if idx == len(categories):
            skills = await hub.browse_skills(limit=20)
        elif 0 <= idx < len(categories):
            skills = await hub.browse_skills(category=categories[idx], limit=20)
        else:
            print("❌ Invalid choice")
            return
        
        print(f"\n📦 Found {len(skills)} skills:\n")
        for skill in skills:
            print(hub.format_skill_info(skill))
            print()
            
    except ValueError:
        print("❌ Invalid input")


async def search_skills(hub):
    """Search skills"""
    query = input("\n🔍 Enter search query: ").strip()
    
    if not query:
        return
    
    skills = await hub.search_skills(query)
    
    if not skills:
        print(f"\n❌ No skills found for '{query}'")
        return
    
    print(f"\n📦 Found {len(skills)} skills:\n")
    for skill in skills:
        print(hub.format_skill_info(skill))
        print()


async def show_installed(hub):
    """Show installed skills"""
    installed = await hub.get_installed_skills()
    
    if not installed:
        print("\n📦 No skills installed yet")
        return
    
    print(f"\n📦 Installed skills ({len(installed)}):\n")
    for skill_id in installed:
        skill = await hub.get_skill(skill_id)
        if skill:
            print(f"  • {skill.name} (v{skill.version})")
    print()


async def install_skill(hub):
    """Install a skill"""
    skill_id = input("\n📥 Enter skill ID to install: ").strip()
    
    if not skill_id:
        return
    
    skill = await hub.get_skill(skill_id)
    
    if not skill:
        print(f"\n❌ Skill '{skill_id}' not found")
        return
    
    print("\nSkill details:")
    print(hub.format_skill_info(skill))
    
    confirm = input("\nInstall this skill? (y/N): ").strip().lower()
    
    if confirm == 'y':
        success = await hub.install_skill(skill_id)
        if success:
            print(f"\n✅ Successfully installed {skill.name}!")
        else:
            print(f"\n❌ Failed to install {skill.name}")


async def main_menu():
    """Main CLI menu"""
    print("\n" + "="*60)
    print("🤖 SableCore - Advanced AI Assistant Platform")
    print("="*60)
    
    print("\n📋 Main Menu:\n")
    print("1. 🎯 Run Onboarding Wizard")
    print("2. 🛒 Skills Marketplace")
    print("3. 💬 Start Chat (CLI)")
    print("4. 🌐 Start Web Dashboard")
    print("5. 📱 WhatsApp Bot Status")
    print("6. 🔧 System Configuration")
    print("7. 📊 View Statistics")
    print("8. 🚪 Exit")
    
    choice = input("\nSelect option: ").strip()
    
    if choice == "1":
        await run_onboarding()
    elif choice == "2":
        await show_skills_hub()
    elif choice == "3":
        print("\n💬 Starting CLI chat...")
        print("Use 'python cli.py chat' for full chat interface")
    elif choice == "4":
        print("\n🌐 Web dashboard URL: http://localhost:8080/dashboard_modern.html")
        print("Run: python main.py --web to start the web server")
    elif choice == "5":
        print("\n📱 WhatsApp Bot Status:")
        print("Bridge: Check if node bridge.js is running on port 3333")
        print("Run: cd whatsapp-bridge && node bridge.js")
    elif choice == "6":
        show_config()
    elif choice == "7":
        await show_stats()
    elif choice == "8":
        print("\n👋 Goodbye!")
        sys.exit(0)


def show_config():
    """Show current configuration"""
    print("\n🔧 System Configuration:\n")
    
    config_file = Path(__file__).parent / "config" / "config.yaml"
    
    if config_file.exists():
        with open(config_file, 'r') as f:
            print(f.read())
    else:
        print("No configuration file found. Run onboarding wizard to create one.")


async def show_stats():
    """Show system statistics"""
    print("\n📊 System Statistics:\n")
    
    # Check if processes are running
    import subprocess
    
    try:
        # Check WhatsApp bridge
        result = subprocess.run(['pgrep', '-f', 'bridge.js'], capture_output=True)
        whatsapp = "🟢 Running" if result.returncode == 0 else "🔴 Stopped"
    except:
        whatsapp = "❓ Unknown"
    
    print(f"WhatsApp Bridge: {whatsapp}")
    print(f"Skills Installed: {len(list((Path(__file__).parent / 'opensable' / 'skills' / 'installed').glob('*.py')))}")
    print(f"Memory Used: Check with 'python -m memory_profiler main.py'")
    print()


async def interactive_loop():
    """Interactive CLI loop"""
    while True:
        try:
            await main_menu()
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "skills":
            asyncio.run(show_skills_hub())
        elif sys.argv[1] == "onboarding":
            asyncio.run(run_onboarding())
        elif sys.argv[1] == "stats":
            asyncio.run(show_stats())
        else:
            print("Usage: sable.py [skills|onboarding|stats]")
    else:
        asyncio.run(interactive_loop())
