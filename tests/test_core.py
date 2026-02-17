"""
Basic tests for Open-Sable
"""
import pytest
from core.config import Open-SableConfig, load_config
from core.memory import MemoryManager


@pytest.mark.asyncio
async def test_config_loading():
    """Test configuration loading"""
    config = Open-SableConfig()
    assert config.agent_name == "Sable"
    assert config.default_model == "llama3.1:8b"


@pytest.mark.asyncio
async def test_memory_system():
    """Test memory storage and retrieval"""
    config = Open-SableConfig()
    memory = MemoryManager(config)
    await memory.initialize()
    
    # Store memory
    await memory.store("test_user", "I like pizza", {"type": "preference"})
    
    # Recall memory
    results = await memory.recall("test_user", "food preferences")
    assert len(results) > 0
    
    # Clean up
    await memory.close()


@pytest.mark.asyncio
async def test_user_preferences():
    """Test user preference management"""
    config = Open-SableConfig()
    memory = MemoryManager(config)
    await memory.initialize()
    
    # Set preference
    await memory.set_user_preference("test_user", "timezone", "UTC")
    
    # Get preference
    prefs = await memory.get_user_preferences("test_user")
    assert prefs["timezone"] == "UTC"
    
    await memory.close()


def test_personality_modes():
    """Test different personality configurations"""
    personalities = ["helpful", "professional", "sarcastic", "meme-aware"]
    
    for personality in personalities:
        config = Open-SableConfig(agent_personality=personality)
        assert config.agent_personality == personality
