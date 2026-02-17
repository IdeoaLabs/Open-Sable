"""
Tests for security and permissions
"""
import pytest
from core.security import PermissionManager, ActionType, PermissionLevel, Sandbox


def test_permission_defaults():
    """Test default permissions are set correctly"""
    from core.config import Open-SableConfig
    config = Open-SableConfig()
    pm = PermissionManager(config)
    pm.initialize()
    
    # Default should be ASK
    perms = pm.get_permissions("default")
    assert perms[ActionType.EMAIL_SEND.value] == PermissionLevel.ASK.value


@pytest.mark.asyncio
async def test_permission_setting():
    """Test setting and checking permissions"""
    from core.config import Open-SableConfig
    config = Open-SableConfig()
    pm = PermissionManager(config)
    pm.initialize()
    
    # Set permission
    pm.set_permission("test_user", ActionType.EMAIL_READ, PermissionLevel.ALWAYS_ALLOW)
    
    # Check permission
    allowed = await pm.check_permission("test_user", ActionType.EMAIL_READ)
    assert allowed is True
    
    # Deny permission
    pm.set_permission("test_user", ActionType.FILE_DELETE, PermissionLevel.DENY)
    denied = await pm.check_permission("test_user", ActionType.FILE_DELETE)
    assert denied is False


def test_path_sandboxing():
    """Test safe path validation"""
    # Safe paths
    assert Sandbox.is_safe_path("./data/test.txt", ["./data"]) is True
    assert Sandbox.is_safe_path("./logs/app.log", ["./logs"]) is True
    
    # Unsafe paths
    assert Sandbox.is_safe_path("/etc/passwd", ["./data"]) is False
    assert Sandbox.is_safe_path("../../../etc/passwd", ["./data"]) is False


def test_input_sanitization():
    """Test input sanitization"""
    dangerous = "<script>alert('xss')</script>"
    safe = Sandbox.sanitize_input(dangerous)
    assert "<script" not in safe
    
    long_text = "a" * 20000
    truncated = Sandbox.sanitize_input(long_text, max_length=10000)
    assert len(truncated) == 10000


def test_url_validation():
    """Test URL domain validation"""
    allowed_domains = ["*.google.com", "example.com"]
    
    # Allowed
    assert Sandbox.validate_url("https://mail.google.com", allowed_domains) is True
    assert Sandbox.validate_url("https://example.com", allowed_domains) is True
    
    # Not allowed
    assert Sandbox.validate_url("https://malicious.com", allowed_domains) is False


def test_audit_log():
    """Test audit logging"""
    from core.config import Open-SableConfig
    config = Open-SableConfig()
    pm = PermissionManager(config)
    pm.initialize()
    
    # Generate some audit entries
    import asyncio
    asyncio.run(pm.check_permission("user1", ActionType.EMAIL_READ))
    asyncio.run(pm.check_permission("user2", ActionType.EMAIL_SEND))
    
    # Check logs
    logs = pm.get_audit_log()
    assert len(logs) > 0
    assert logs[0]["user_id"] in ["user1", "user2"]
