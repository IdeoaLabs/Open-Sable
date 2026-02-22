"""
Tests for Enterprise features - RBAC, Multi-tenancy, Audit, SSO.
"""

import pytest

from opensable.core.enterprise import (
    RBAC,
    MultiTenancy,
    AuditLogger,
    SSOProvider,
    Permission,
    AuditAction,
    Role,
    Tenant,
    User,
)


class TestPermissionEnum:
    """Test Permission enum."""

    def test_admin_permissions(self):
        assert Permission.ADMIN_ALL.value == "admin:*"
        assert Permission.ADMIN_USERS.value == "admin:users"

    def test_agent_permissions(self):
        assert Permission.AGENT_CREATE.value == "agent:create"
        assert Permission.AGENT_READ.value == "agent:read"

    def test_data_permissions(self):
        assert Permission.DATA_READ.value == "data:read"
        assert Permission.DATA_WRITE.value == "data:write"


class TestRole:
    """Test Role dataclass."""

    def test_create_role(self):
        role = Role(name="test", permissions={Permission.AGENT_READ})
        assert role.name == "test"
        assert Permission.AGENT_READ in role.permissions

    def test_has_permission(self):
        role = Role(name="r", permissions={Permission.DATA_READ, Permission.DATA_WRITE})
        assert role.has_permission(Permission.DATA_READ)
        assert not role.has_permission(Permission.ADMIN_ALL)

    def test_to_dict(self):
        role = Role(name="r", permissions={Permission.DATA_READ}, description="desc")
        d = role.to_dict()
        assert d["name"] == "r"
        assert d["description"] == "desc"


class TestMultiTenancy:
    """Test multi-tenancy management."""

    @pytest.fixture
    def mt(self, tmp_path):
        return MultiTenancy(storage_dir=str(tmp_path / "tenants"))

    def test_create_tenant(self, mt):
        tenant = mt.create_tenant("Acme Corp")
        assert isinstance(tenant, Tenant)
        assert tenant.name == "Acme Corp"
        assert tenant.plan == "free"

    def test_tenant_plans(self, mt):
        t = mt.create_tenant("Pro Co", plan="pro")
        assert t.plan == "pro"
        assert t.limits.get("agents") == 50

    def test_get_tenant(self, mt):
        t = mt.create_tenant("FindMe")
        found = mt.get_tenant(t.id)
        assert found is not None
        assert found.name == "FindMe"

    def test_get_nonexistent(self, mt):
        assert mt.get_tenant("nonexistent") is None

    def test_create_user(self, mt):
        t = mt.create_tenant("UserOrg")
        user = mt.create_user("a@b.com", t.id, "pass123")
        assert isinstance(user, User)
        assert user.email == "a@b.com"
        assert user.tenant_id == t.id

    def test_check_quota(self, mt):
        t = mt.create_tenant("Quota", plan="free")
        # free plan: agents=5
        assert mt.check_quota(t.id, "agents", 3) is True
        assert mt.check_quota(t.id, "agents", 10) is False

    def test_enterprise_unlimited(self, mt):
        t = mt.create_tenant("Big", plan="enterprise")
        assert mt.check_quota(t.id, "agents", 999999) is True


class TestRBAC:
    """Test role-based access control."""

    @pytest.fixture
    def rbac(self):
        return RBAC()

    def test_default_roles(self, rbac):
        assert "admin" in rbac.roles
        assert "developer" in rbac.roles
        assert "operator" in rbac.roles
        assert "viewer" in rbac.roles

    def test_create_role(self, rbac):
        role = rbac.create_role("custom", {Permission.DATA_READ}, "Custom role")
        assert role.name == "custom"
        assert "custom" in rbac.roles

    def test_assign_role(self, rbac):
        rbac.assign_role("user1", "viewer")
        assert "viewer" in rbac.user_roles.get("user1", [])

    def test_check_permission(self, rbac):
        rbac.assign_role("u1", "viewer")
        assert rbac.check_permission("u1", Permission.AGENT_READ) is True
        assert rbac.check_permission("u1", Permission.AGENT_DELETE) is False

    def test_admin_has_all(self, rbac):
        rbac.assign_role("admin_user", "admin")
        assert rbac.check_permission("admin_user", Permission.ADMIN_ALL) is True

    def test_multiple_roles(self, rbac):
        rbac.assign_role("u2", "viewer")
        rbac.assign_role("u2", "developer")
        assert rbac.check_permission("u2", Permission.AGENT_DELETE) is True

    def test_revoke_role(self, rbac):
        rbac.assign_role("u3", "developer")
        rbac.revoke_role("u3", "developer")
        assert rbac.check_permission("u3", Permission.AGENT_DELETE) is False

    def test_no_user_no_permission(self, rbac):
        assert rbac.check_permission("nobody", Permission.DATA_READ) is False

    def test_get_user_permissions(self, rbac):
        rbac.assign_role("u4", "viewer")
        perms = rbac.get_user_permissions("u4")
        assert Permission.AGENT_READ in perms


class TestAuditLogger:
    """Test audit logging."""

    @pytest.fixture
    def logger(self, tmp_path):
        return AuditLogger(storage_dir=str(tmp_path / "audit"))

    @pytest.mark.asyncio
    async def test_log_action(self, logger):
        await logger.log(
            tenant_id="t1",
            user_id="u1",
            action=AuditAction.CREATE,
            resource_type="agent",
            resource_id="a1",
        )
        assert len(logger.logs) == 1
        assert logger.logs[0].action == AuditAction.CREATE

    @pytest.mark.asyncio
    async def test_log_with_details(self, logger):
        await logger.log(
            tenant_id="t1",
            user_id="u1",
            action=AuditAction.UPDATE,
            resource_type="workflow",
            resource_id="w1",
            details={"field": "name"},
            ip_address="10.0.0.1",
        )
        log = logger.logs[0]
        assert log.details == {"field": "name"}
        assert log.ip_address == "10.0.0.1"

    @pytest.mark.asyncio
    async def test_query_by_tenant(self, logger):
        await logger.log("t1", "u1", AuditAction.READ, "agent", "a1")
        await logger.log("t2", "u2", AuditAction.READ, "agent", "a2")
        results = await logger.query(tenant_id="t1")
        assert len(results) == 1
        assert results[0].tenant_id == "t1"

    @pytest.mark.asyncio
    async def test_query_by_action(self, logger):
        await logger.log("t1", "u1", AuditAction.CREATE, "agent", "a1")
        await logger.log("t1", "u1", AuditAction.DELETE, "agent", "a2")
        results = await logger.query(action=AuditAction.DELETE)
        assert len(results) == 1


class TestSSOProvider:
    """Test SSO provider."""

    @pytest.fixture
    def sso(self):
        return SSOProvider(secret_key="test-secret-key-12345")

    def test_create_token(self, sso):
        token = sso.create_token(
            user_id="u1",
            tenant_id="t1",
            roles=["admin"],
        )
        assert isinstance(token, str)
        assert len(token) > 0

    def test_verify_token(self, sso):
        token = sso.create_token("u1", "t1", ["viewer"])
        payload = sso.verify_token(token)
        assert payload is not None
        assert payload["user_id"] == "u1"
        assert payload["tenant_id"] == "t1"
        assert "viewer" in payload["roles"]

    def test_invalid_token(self, sso):
        result = sso.verify_token("totally.invalid.token")
        assert result is None

    def test_create_session(self, sso):
        session_id = sso.create_session("u1", "t1")
        assert isinstance(session_id, str)
        assert len(session_id) > 0

    def test_validate_session(self, sso):
        sid = sso.create_session("u1", "t1")
        assert sso.validate_session(sid) is True

    def test_destroy_session(self, sso):
        sid = sso.create_session("u1", "t1")
        sso.destroy_session(sid)
        assert sso.validate_session(sid) is False

    def test_invalid_session(self, sso):
        assert sso.validate_session("fake-session") is False
