#!/usr/bin/env python3
"""
Quick test script for new self-modification features
"""

import asyncio
import sys
from pathlib import Path

# Add parent dir to path
sys.path.insert(0, str(Path(__file__).parent))

from opensable.core.config import Config
from opensable.core.skill_creator import SkillCreator


async def test_skill_creation():
    print("🧪 Testing Skill Creator...")

    config = Config()
    creator = SkillCreator(config)

    # Test 1: Create a simple skill
    print("\n1️⃣ Creating 'hello_world' skill...")
    result = await creator.create_skill(
        name="hello_world",
        description="Simple greeting skill",
        code="""
async def execute(name="World"):
    return f"Hello, {name}!"
""",
        metadata={"author": "test", "version": "1.0"},
    )

    if result.get("success"):
        print(f"   ✅ {result.get('message')}")
    else:
        print(f"   ❌ {result.get('error')}")

    # Test 2: List skills
    print("\n2️⃣ Listing all skills...")
    skills = creator.list_skills()
    print(f"   📦 Found {len(skills)} skills")
    for skill in skills:
        status = "✅" if skill.get("enabled") else "❌"
        print(f"   {status} {skill['name']} - {skill['description']}")

    # Test 3: Try to create invalid skill (should fail)
    print("\n3️⃣ Testing security validation (should block)...")
    result = await creator.create_skill(
        name="evil_skill",
        description="Malicious skill",
        code="""
import os
os.system("rm -rf /")  # This should be blocked!
""",
        metadata={"author": "hacker"},
    )

    if result.get("success"):
        print("   ❌ SECURITY FAILED - malicious code was allowed!")
    else:
        print(f"   ✅ Security check worked: {result.get('error')}")

    # Test 4: Create useful skill
    print("\n4️⃣ Creating 'timestamp' skill...")
    result = await creator.create_skill(
        name="get_timestamp",
        description="Get current Unix timestamp",
        code="""
import time

async def execute():
    return f"Current timestamp: {int(time.time())}"
""",
        metadata={"author": "sable"},
    )

    if result.get("success"):
        print(f"   ✅ {result.get('message')}")
    else:
        print(f"   ❌ {result.get('error')}")

    # Clean up
    print("\n🧹 Cleaning up test skills...")
    creator.delete_skill("hello_world")
    creator.delete_skill("get_timestamp")

    print("\n✅ All tests completed!")


async def test_heartbeat():
    print("\n💓 Testing Heartbeat System...")

    # Just check if file structure is correct
    heartbeat_file = Path.home() / ".opensable" / "HEARTBEAT.md"

    if heartbeat_file.exists():
        print(f"   ✅ HEARTBEAT.md found at {heartbeat_file}")
        with open(heartbeat_file) as f:
            lines = f.readlines()
            checklist_items = [l for l in lines if l.strip().startswith("- [ ]")]
            print(f"   📋 {len(checklist_items)} checklist items configured")
    else:
        print("   ⚠️  HEARTBEAT.md not found (will be created on first run)")

    print("   ℹ️  Heartbeat runs automatically in telegram bot")
    print("   ℹ️  Default interval: 30 minutes")
    print("   ℹ️  Active hours: 08:00-23:00")


async def main():
    print("=" * 60)
    print("🚀 Open-Sable Self-Modification Features Test")
    print("=" * 60)

    await test_skill_creation()
    await test_heartbeat()

    print("\n" + "=" * 60)
    print("✅ Test suite completed!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
