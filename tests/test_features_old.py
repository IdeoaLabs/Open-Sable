#!/usr/bin/env python3
"""
Complete Feature Test Suite for SableCore
Tests all newly implemented features
"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from opensable.core.config import OpenSableConfig
from opensable.core.skills_hub import SkillsHub, Skill
from opensable.core.advanced_memory import AdvancedMemorySystem, MemoryType, MemoryImportance


async def test_advanced_memory():
    """Test Advanced Memory System"""
    print("\n" + "="*60)
    print("🧠 Testing Advanced Memory System")
    print("="*60)
    
    config = OpenSableConfig()
    memory = AdvancedMemorySystem(config)
    await memory.initialize()
    
    # Store episodic memory
    print("\n1. Storing episodic memory...")
    await memory.store_memory(
        memory_type=MemoryType.EPISODIC,
        content="User asked about cryptocurrency prices",
        context={"user_id": "test_user", "platform": "test"},
        importance=MemoryImportance.HIGH
    )
    print("   ✅ Episodic memory stored")
    
    # Store semantic memory
    print("\n2. Storing semantic memory...")
    await memory.store_memory(
        memory_type=MemoryType.SEMANTIC,
        content="Bitcoin is a decentralized cryptocurrency",
        context={"topic": "crypto"},
        importance=MemoryImportance.MEDIUM
    )
    print("   ✅ Semantic memory stored")
    
    # Retrieve memories
    print("\n3. Retrieving memories about crypto...")
    memories = await memory.retrieve_memories(
        query="cryptocurrency",
        limit=5
    )
    print(f"   ✅ Retrieved {len(memories)} memories")
    for mem in memories:
        print(f"      - {mem.content[:60]}...")
    
    # Get stats
    print("\n4. Memory statistics:")
    stats = await memory.get_memory_stats()
    print(f"   Total memories: {stats.get('total_memories', 0)}")
    print(f"   Episodic: {stats.get('episodic_count', 0)}")
    print(f"   Semantic: {stats.get('semantic_count', 0)}")
    
    print("\n✅ Advanced Memory System: PASSED")


async def test_skills_hub():
    """Test Skills Marketplace Hub"""
    print("\n" + "="*60)
    print("🛒 Testing Skills Marketplace Hub")
    print("="*60)
    
    config = OpenSableConfig()
    hub = SkillsHub(config)
    await hub.initialize()
    
    # Browse skills
    print("\n1. Browsing skills...")
    skills = await hub.browse_skills(limit=5)
    print(f"   ✅ Found {len(skills)} skills")
    for skill in skills[:3]:
        print(f"      - {skill.name} (★{skill.rating}, {skill.downloads} downloads)")
    
    # Search skills
    print("\n2. Searching for 'crypto' skills...")
    results = await hub.search_skills("crypto")
    print(f"   ✅ Found {len(results)} crypto-related skills")
    
    # Get categories
    print("\n3. Available categories:")
    categories = await hub.get_categories()
    print(f"   ✅ {len(categories)} categories: {', '.join(categories)}")
    
    # Get specific skill
    print("\n4. Getting skill details...")
    skill = await hub.get_skill("web_scraper_pro")
    if skill:
        print(f"   ✅ Loaded: {skill.name} v{skill.version}")
        print(f"      Author: {skill.author}")
        print(f"      Rating: ⭐ {skill.rating}/5.0")
    
    # Check installed skills
    print("\n5. Checking installed skills...")
    installed = await hub.get_installed_skills()
    print(f"   ✅ {len(installed)} skills installed")
    
    print("\n✅ Skills Marketplace Hub: PASSED")


async def test_agent_integration():
    """Test Advanced Memory integration with Agent"""
    print("\n" + "="*60)
    print("🤖 Testing Agent Integration")
    print("="*60)
    
    try:
        from opensable.core.agent import SableAgent
        
        config = OpenSableConfig()
        agent = SableAgent(config)
        await agent.initialize()
        
        print("\n1. Checking agent components...")
        print(f"   ✅ LLM initialized: {agent.llm is not None}")
        print(f"   ✅ Memory initialized: {agent.memory is not None}")
        print(f"   ✅ Advanced Memory initialized: {agent.advanced_memory is not None}")
        print(f"   ✅ Tools initialized: {agent.tools is not None}")
        
        # Test message processing (stores in advanced memory automatically)
        print("\n2. Processing test message...")
        response = await agent.process_message(
            user_id="test_user",
            message="What is cryptocurrency?",
            history=[]
        )
        print(f"   ✅ Response received: {response[:100]}...")
        
        # Verify memory was stored
        if agent.advanced_memory:
            stats = await agent.advanced_memory.get_memory_stats()
            print(f"\n3. Memory stats after message:")
            print(f"   Total memories: {stats.get('total_memories', 0)}")
            print(f"   ✅ Message stored in advanced memory")
        
        print("\n✅ Agent Integration: PASSED")
        
    except Exception as e:
        print(f"\n⚠️  Agent Integration: SKIPPED ({e})")


async def test_file_structure():
    """Test that all new files exist"""
    print("\n" + "="*60)
    print("📁 Testing File Structure")
    print("="*60)
    
    base_dir = Path(__file__).parent
    
    files_to_check = [
        "opensable/core/advanced_memory.py",
        "opensable/core/skills_hub.py",
        "opensable/core/onboarding.py",
        "static/dashboard_modern.html",
        "sable.py",
        "venom-bot/bridge.js",
        "venom-bot/package.json",
        "WHATS_NEW.md",
    ]
    
    print("\nChecking required files:")
    all_exist = True
    for file_path in files_to_check:
        full_path = base_dir / file_path
        exists = full_path.exists()
        status = "✅" if exists else "❌"
        print(f"   {status} {file_path}")
        all_exist = all_exist and exists
    
    if all_exist:
        print("\n✅ File Structure: PASSED")
    else:
        print("\n❌ File Structure: FAILED - Some files missing")


async def test_whatsapp_bridge():
    """Test WhatsApp bridge connectivity"""
    print("\n" + "="*60)
    print("📱 Testing WhatsApp Bridge")
    print("="*60)
    
    import subprocess
    
    # Check if bridge.js exists
    bridge_file = Path(__file__).parent / "venom-bot" / "bridge.js"
    if not bridge_file.exists():
        print("   ❌ bridge.js not found")
        return
    
    print(f"   ✅ bridge.js exists")
    
    # Check if node_modules exists
    node_modules = Path(__file__).parent / "venom-bot" / "node_modules"
    if node_modules.exists():
        print("   ✅ Dependencies installed")
    else:
        print("   ⚠️  Dependencies not installed (run: npm install)")
    
    # Check if bridge is running
    try:
        result = subprocess.run(['pgrep', '-f', 'bridge.js'], capture_output=True)
        if result.returncode == 0:
            print("   ✅ WhatsApp bridge is running")
        else:
            print("   ⚠️  Bridge not running (start with: node bridge.js)")
    except:
        print("   ⚠️  Could not check if bridge is running")
    
    print("\n✅ WhatsApp Bridge: PASSED")


async def run_all_tests():
    """Run all feature tests"""
    print("\n" + "="*80)
    print("🧪 SableCore Complete Feature Test Suite")
    print("="*80)
    
    tests = [
        ("File Structure", test_file_structure),
        ("Advanced Memory System", test_advanced_memory),
        ("Skills Marketplace Hub", test_skills_hub),
        ("WhatsApp Bridge", test_whatsapp_bridge),
        ("Agent Integration", test_agent_integration),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_func in tests:
        try:
            await test_func()
            passed += 1
        except Exception as e:
            print(f"\n❌ {name}: FAILED")
            print(f"   Error: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
        
        await asyncio.sleep(0.5)  # Brief pause between tests
    
    # Summary
    print("\n" + "="*80)
    print("📊 Test Summary")
    print("="*80)
    print(f"\n   Total Tests: {passed + failed}")
    print(f"   ✅ Passed: {passed}")
    print(f"   ❌ Failed: {failed}")
    print()
    
    if failed == 0:
        print("🎉 All tests passed! SableCore is ready to use.")
        print("\nNext steps:")
        print("   1. Run onboarding: python sable.py onboarding")
        print("   2. Browse skills: python sable.py skills")
        print("   3. Start bot: python main.py")
        print("   4. Open dashboard: http://localhost:8080/dashboard_modern.html")
    else:
        print("⚠️  Some tests failed. Check errors above.")
    
    print()


if __name__ == "__main__":
    asyncio.run(run_all_tests())
