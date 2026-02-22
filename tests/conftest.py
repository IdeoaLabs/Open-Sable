"""
pytest configuration for Open-Sable tests
"""

import pytest
import asyncio
import shutil


@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests"""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(autouse=True)
def temp_opensable_dir(monkeypatch, tmp_path):
    """Use temporary directory for tests"""
    # Create temp .opensable directory
    opensable_dir = tmp_path / ".opensable"
    opensable_dir.mkdir()

    # Monkeypatch home directory
    monkeypatch.setenv("HOME", str(tmp_path))

    yield opensable_dir

    # Cleanup
    if opensable_dir.exists():
        shutil.rmtree(opensable_dir)


@pytest.fixture
def mock_agent():
    """Mock agent for testing"""

    class MockAgent:
        async def send_message(self, message, **kwargs):
            """Mock send_message"""
            return {"content": f"Mock response to: {message}", "tokens": 10}

    return MockAgent()


def pytest_configure(config):
    """Configure pytest"""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "e2e: marks tests as end-to-end tests")
