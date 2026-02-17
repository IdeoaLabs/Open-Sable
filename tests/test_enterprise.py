"""
Tests for Enterprise features - Multi-tenancy, RBAC, Audit, SSO.
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock

from core.enterprise import (
    MultiTenancy, RBAC, AuditLogger, SSOProvider,
    Permission, AuditAction, Role, User, Tenant
)


class TestMultiTenancy:
    """Test multi-tenancy system"""
    
    @pytest.fixture
    def tenancy(self):
        return MultiTenancy()
    
    def test_create_tenant(self, tenancy):
        """Test tenant creation"""
        tenant = tenancy.create_tenant("Test Corp", plan="enterprise")
        
        assert tenant.name == "Test Corp"
        assert tenant.plan == "enterprise"
        assert tenant.id is not None
        assert tenant.limits is not None
    
    def test_create_user(self, tenancy):
        """Test user creation"""
        tenant = tenancy.create_tenant("Company", plan="pro")
        user = tenancy.create_user("user@company.com", tenant.id, "password123")
        
        assert user.email == "user@company.com"
        assert user.tenant_id == tenant.id
        assert user.id is not None
    
    def test_tenant_isolation(self, tenancy):
        """Test that tenants are isolated"""
        tenant1 = tenancy.create_tenant("Company A", plan="pro")
        tenant2 = tenancy.create_tenant("Company B", plan="pro")
        
        user1 = tenancy.create_user("user@a.com", tenant1.id, "pass1")
        user2 = tenancy.create_user("user@b.com", tenant2.id, "pass2")
        
        assert user1.tenant_id != user2.tenant_id
    
    def test_quota_checking(self, tenancy):
        """Test resource quota enforcement"""
        tenant = tenancy.create_tenant("Test", plan="free")
        
        # Free plan has low limits
        within_quota = tenancy.check_quota(tenant.id, "agents", 5)
        over_quota = tenancy.check_quota(tenant.id, "agents", 1000)
        
        assert within_quota is True
        assert over_quota is False
    
    def test_tenant_plans(self, tenancy):
        """Test different plan limits"""
        free = tenancy.create_tenant("Free", plan="free")
        pro = tenancy.create_tenant("Pro", plan="pro")
        enterprise = tenancy.create_tenant("Enterprise", plan="enterprise")
        
        # Enterprise should have highest limits
        assert enterprise.limits["agents"] > pro.limits["agents"]
        assert pro.limits["agents"] > free.limits["agents"]


class TestRBAC:
    """Test role-based access control"""
    
    @pytest.fixture
    def rbac(self):
        return RBAC()
    
    def test_assign_role(self, rbac):
        """Test role assignment"""
        user_id = "user_123"
        
        rbac.assign_role(user_id, "admin")
        
        roles = rbac.get_user_roles(user_id)
        assert "admin" in roles
    
    def test_check_permission(self, rbac):
        """Test permission checking"""
        user_id = "user_456"
        
        rbac.assign_role(user_id, "developer")
        
        # Developer can create agents
        assert rbac.check_permission(user_id, Permission.AGENT_CREATE)
        
        # Developer cannot delete everything
        assert not rbac.check_permission(user_id, Permission.SYSTEM_DELETE)
    
    def test_custom_role(self, rbac):
        """Test custom role creation"""
        permissions = {
            Permission.AGENT_READ,
            Permission.AGENT_EXECUTE,
            Permission.DATA_READ
        }
        
        role = rbac.create_role("analyst", permissions, "Data analyst role")
        
        assert role.name == "analyst"
        assert len(role.permissions) == 3
    
    def test_multiple_roles(self, rbac):
        """Test user with multiple roles"""
        user_id = "user_789"
        
        rbac.assign_role(user_id, "developer")
        rbac.assign_role(user_id, "operator")
        
        roles = rbac.get_user_roles(user_id)
        assert len(roles) == 2
        assert "developer" in roles
        assert "operator" in roles
    
    def test_permission_hierarchy(self, rbac):
        """Test that admin has all permissions"""
        admin_id = "admin_001"
        rbac.assign_role(admin_id, "admin")
        
        # Admin should have all permissions
        assert rbac.check_permission(admin_id, Permission.AGENT_DELETE)
        assert rbac.check_permission(admin_id, Permission.SYSTEM_DELETE)
        assert rbac.check_permission(admin_id, Permission.USER_CREATE)


class TestAuditLogger:
    """Test audit logging"""
    
    @pytest.fixture
    def audit(self):
        return AuditLogger()
    
    @pytest.mark.asyncio
    async def test_log_action(self, audit):
        """Test logging an action"""
        await audit.log(
            tenant_id="tenant_1",
            user_id="user_1",
            action=AuditAction.CREATE,
            resource_type="agent",
            resource_id="agent_123",
            details={"name": "Test Agent"}
        )
        
        assert len(audit.logs) == 1
        assert audit.logs[0].action == AuditAction.CREATE
    
    @pytest.mark.asyncio
    async def test_query_by_tenant(self, audit):
        """Test querying logs by tenant"""
        await audit.log("tenant_1", "user_1", AuditAction.CREATE, "agent", "a1")
        await audit.log("tenant_2", "user_2", AuditAction.UPDATE, "agent", "a2")
        
        tenant1_logs = await audit.query(tenant_id="tenant_1")
        
        assert len(tenant1_logs) == 1
        assert tenant1_logs[0].tenant_id == "tenant_1"
    
    @pytest.mark.asyncio
    async def test_query_by_action(self, audit):
        """Test querying logs by action type"""
        await audit.log("t1", "u1", AuditAction.CREATE, "agent", "a1")
        await audit.log("t1", "u1", AuditAction.DELETE, "agent", "a2")
        await audit.log("t1", "u1", AuditAction.CREATE, "workflow", "w1")
        
        create_logs = await audit.query(action=AuditAction.CREATE)
        
        assert len(create_logs) == 2
    
    @pytest.mark.asyncio
    async def test_query_by_time_range(self, audit):
        """Test querying logs by time range"""
        now = datetime.now()
        
        await audit.log("t1", "u1", AuditAction.LOGIN, "session", "s1")
        
        # Query last hour
        recent = await audit.query(
            start_date=now - timedelta(hours=1),
            end_date=now + timedelta(hours=1)
        )
        
        assert len(recent) >= 1


class TestSSOProvider:
    """Test Single Sign-On"""
    
    @pytest.fixture
    def sso(self):
        return SSOProvider(secret_key="test-secret-key-123")
    
    def test_create_token(self, sso):
        """Test JWT token creation"""
        token = sso.create_token(
            user_id="user_123",
            tenant_id="tenant_456",
            roles=["developer"],
            expires_in=3600
        )
        
        assert token is not None
        assert isinstance(token, str)
        assert len(token) > 0
    
    def test_verify_token(self, sso):
        """Test token verification"""
        token = sso.create_token(
            user_id="user_123",
            tenant_id="tenant_456",
            roles=["admin"]
        )
        
        payload = sso.verify_token(token)
        
        assert payload is not None
        assert payload["user_id"] == "user_123"
        assert payload["tenant_id"] == "tenant_456"
        assert "admin" in payload["roles"]
    
    def test_expired_token(self, sso):
        """Test that expired tokens are rejected"""
        # Create token that expires immediately
        token = sso.create_token(
            user_id="user_123",
            tenant_id="tenant_456",
            expires_in=-1  # Already expired
        )
        
        payload = sso.verify_token(token)
        
        # Should be None or raise exception
        assert payload is None or "error" in payload
    
    def test_session_management(self, sso):
        """Test session creation and validation"""
        session_id = sso.create_session("user_123", "tenant_456")
        
        assert session_id is not None
        assert sso.validate_session(session_id) is True
        
        # Destroy session
        sso.destroy_session(session_id)
        assert sso.validate_session(session_id) is False
    
    def test_invalid_token(self, sso):
        """Test invalid token handling"""
        payload = sso.verify_token("invalid.token.here")
        
        assert payload is None
