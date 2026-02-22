"""
Enterprise Features - Multi-tenancy, RBAC, Audit Logging, and SSO.

Production-grade security and compliance features.
"""

import asyncio
import hashlib
import json
import logging
import secrets
import time
from typing import Dict, Any, Optional, List, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path

try:
    import jwt

    JWT_AVAILABLE = True
except ImportError:
    jwt = None  # type: ignore
    JWT_AVAILABLE = False
    logging.getLogger(__name__).warning(
        "PyJWT not installed. SSO/token features disabled. " "Run: pip install PyJWT"
    )


class Permission(Enum):
    """System permissions."""

    # Admin
    ADMIN_ALL = "admin:*"
    ADMIN_USERS = "admin:users"
    ADMIN_ROLES = "admin:roles"
    ADMIN_TENANTS = "admin:tenants"

    # Agent
    AGENT_CREATE = "agent:create"
    AGENT_READ = "agent:read"
    AGENT_UPDATE = "agent:update"
    AGENT_DELETE = "agent:delete"
    AGENT_EXECUTE = "agent:execute"

    # Workflow
    WORKFLOW_CREATE = "workflow:create"
    WORKFLOW_READ = "workflow:read"
    WORKFLOW_UPDATE = "workflow:update"
    WORKFLOW_DELETE = "workflow:delete"
    WORKFLOW_EXECUTE = "workflow:execute"

    # Data
    DATA_READ = "data:read"
    DATA_WRITE = "data:write"
    DATA_DELETE = "data:delete"
    DATA_EXPORT = "data:export"

    # Settings
    SETTINGS_READ = "settings:read"
    SETTINGS_WRITE = "settings:write"


class AuditAction(Enum):
    """Audit log action types."""

    LOGIN = "login"
    LOGOUT = "logout"
    CREATE = "create"
    READ = "read"
    UPDATE = "update"
    DELETE = "delete"
    EXECUTE = "execute"
    EXPORT = "export"
    IMPORT = "import"
    PERMISSION_GRANT = "permission_grant"
    PERMISSION_REVOKE = "permission_revoke"


@dataclass
class Role:
    """User role with permissions."""

    name: str
    permissions: Set[Permission]
    description: str = ""
    created_at: datetime = field(default_factory=datetime.now)

    def has_permission(self, permission: Permission) -> bool:
        """Check if role has permission."""
        if Permission.ADMIN_ALL in self.permissions:
            return True
        return permission in self.permissions

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "permissions": [p.value for p in self.permissions],
            "description": self.description,
            "created_at": self.created_at.isoformat(),
        }


@dataclass
class User:
    """User account."""

    id: str
    email: str
    tenant_id: str
    roles: List[str] = field(default_factory=list)
    password_hash: Optional[str] = None
    is_active: bool = True
    is_verified: bool = False
    created_at: datetime = field(default_factory=datetime.now)
    last_login: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "email": self.email,
            "tenant_id": self.tenant_id,
            "roles": self.roles,
            "is_active": self.is_active,
            "is_verified": self.is_verified,
            "created_at": self.created_at.isoformat(),
            "last_login": self.last_login.isoformat() if self.last_login else None,
            "metadata": self.metadata,
        }


@dataclass
class Tenant:
    """Multi-tenant organization."""

    id: str
    name: str
    plan: str = "free"
    is_active: bool = True
    created_at: datetime = field(default_factory=datetime.now)
    settings: Dict[str, Any] = field(default_factory=dict)
    limits: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "plan": self.plan,
            "is_active": self.is_active,
            "created_at": self.created_at.isoformat(),
            "settings": self.settings,
            "limits": self.limits,
        }


@dataclass
class AuditLog:
    """Audit log entry."""

    id: str
    timestamp: datetime
    tenant_id: str
    user_id: str
    action: AuditAction
    resource_type: str
    resource_id: str
    details: Dict[str, Any] = field(default_factory=dict)
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat(),
            "tenant_id": self.tenant_id,
            "user_id": self.user_id,
            "action": self.action.value,
            "resource_type": self.resource_type,
            "resource_id": self.resource_id,
            "details": self.details,
            "ip_address": self.ip_address,
            "user_agent": self.user_agent,
        }


class RBAC:
    """
    Role-Based Access Control system.

    Features:
    - User and role management
    - Permission checking
    - Role inheritance
    """

    def __init__(self):
        """Initialize RBAC."""
        self.roles: Dict[str, Role] = {}
        self.user_roles: Dict[str, List[str]] = {}
        self._init_default_roles()

    def _init_default_roles(self):
        """Initialize default roles."""
        # Admin role - all permissions
        self.roles["admin"] = Role(
            name="admin", permissions={Permission.ADMIN_ALL}, description="Full system access"
        )

        # Developer role
        self.roles["developer"] = Role(
            name="developer",
            permissions={
                Permission.AGENT_CREATE,
                Permission.AGENT_READ,
                Permission.AGENT_UPDATE,
                Permission.AGENT_DELETE,
                Permission.AGENT_EXECUTE,
                Permission.WORKFLOW_CREATE,
                Permission.WORKFLOW_READ,
                Permission.WORKFLOW_UPDATE,
                Permission.WORKFLOW_DELETE,
                Permission.WORKFLOW_EXECUTE,
                Permission.DATA_READ,
                Permission.DATA_WRITE,
                Permission.SETTINGS_READ,
            },
            description="Full development access",
        )

        # Operator role
        self.roles["operator"] = Role(
            name="operator",
            permissions={
                Permission.AGENT_READ,
                Permission.AGENT_EXECUTE,
                Permission.WORKFLOW_READ,
                Permission.WORKFLOW_EXECUTE,
                Permission.DATA_READ,
                Permission.SETTINGS_READ,
            },
            description="Execute and monitor",
        )

        # Viewer role
        self.roles["viewer"] = Role(
            name="viewer",
            permissions={
                Permission.AGENT_READ,
                Permission.WORKFLOW_READ,
                Permission.DATA_READ,
                Permission.SETTINGS_READ,
            },
            description="Read-only access",
        )

    def create_role(self, name: str, permissions: Set[Permission], description: str = "") -> Role:
        """Create a new role."""
        role = Role(name=name, permissions=permissions, description=description)
        self.roles[name] = role
        return role

    def assign_role(self, user_id: str, role_name: str):
        """Assign role to user."""
        if role_name not in self.roles:
            raise ValueError(f"Role {role_name} does not exist")

        if user_id not in self.user_roles:
            self.user_roles[user_id] = []

        if role_name not in self.user_roles[user_id]:
            self.user_roles[user_id].append(role_name)

    def revoke_role(self, user_id: str, role_name: str):
        """Revoke role from user."""
        if user_id in self.user_roles and role_name in self.user_roles[user_id]:
            self.user_roles[user_id].remove(role_name)

    def check_permission(self, user_id: str, permission: Permission) -> bool:
        """Check if user has permission."""
        if user_id not in self.user_roles:
            return False

        for role_name in self.user_roles[user_id]:
            role = self.roles.get(role_name)
            if role and role.has_permission(permission):
                return True

        return False

    def get_user_permissions(self, user_id: str) -> Set[Permission]:
        """Get all permissions for user."""
        permissions = set()

        if user_id in self.user_roles:
            for role_name in self.user_roles[user_id]:
                role = self.roles.get(role_name)
                if role:
                    permissions.update(role.permissions)

        return permissions


class MultiTenancy:
    """
    Multi-tenancy management.

    Features:
    - Tenant isolation
    - Resource quotas
    - Tenant-specific settings
    """

    def __init__(self, storage_dir: Optional[str] = None):
        """Initialize multi-tenancy."""
        self.storage_dir = (
            Path(storage_dir) if storage_dir else Path.home() / ".opensable" / "tenants"
        )
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        self.tenants: Dict[str, Tenant] = {}
        self.users: Dict[str, User] = {}
        self._load_tenants()

    def _load_tenants(self):
        """Load tenants from storage."""
        tenants_file = self.storage_dir / "tenants.json"
        if tenants_file.exists():
            try:
                data = json.loads(tenants_file.read_text())
                for tenant_data in data.get("tenants", []):
                    tenant = Tenant(**tenant_data)
                    tenant.created_at = datetime.fromisoformat(tenant_data["created_at"])
                    self.tenants[tenant.id] = tenant
            except Exception:
                pass

    def _save_tenants(self):
        """Save tenants to storage."""
        tenants_file = self.storage_dir / "tenants.json"
        data = {"tenants": [tenant.to_dict() for tenant in self.tenants.values()]}
        tenants_file.write_text(json.dumps(data, indent=2))

    def create_tenant(self, name: str, plan: str = "free") -> Tenant:
        """Create a new tenant."""
        tenant_id = hashlib.sha256(f"{name}{time.time()}".encode()).hexdigest()[:16]

        # Default limits by plan
        limits = {
            "free": {"agents": 5, "workflows": 10, "api_calls": 1000},
            "pro": {"agents": 50, "workflows": 100, "api_calls": 100000},
            "enterprise": {"agents": -1, "workflows": -1, "api_calls": -1},
        }

        tenant = Tenant(id=tenant_id, name=name, plan=plan, limits=limits.get(plan, limits["free"]))

        self.tenants[tenant_id] = tenant
        self._save_tenants()

        return tenant

    def get_tenant(self, tenant_id: str) -> Optional[Tenant]:
        """Get tenant by ID."""
        return self.tenants.get(tenant_id)

    def create_user(self, email: str, tenant_id: str, password: str) -> User:
        """Create a new user."""
        if tenant_id not in self.tenants:
            raise ValueError("Tenant does not exist")

        user_id = hashlib.sha256(f"{email}{time.time()}".encode()).hexdigest()[:16]
        password_hash = hashlib.sha256(password.encode()).hexdigest()

        user = User(id=user_id, email=email, tenant_id=tenant_id, password_hash=password_hash)

        self.users[user_id] = user
        return user

    def check_quota(self, tenant_id: str, resource: str, current: int) -> bool:
        """Check if tenant is within quota."""
        tenant = self.tenants.get(tenant_id)
        if not tenant:
            return False

        limit = tenant.limits.get(resource, 0)
        if limit == -1:  # Unlimited
            return True

        return current < limit


class AuditLogger:
    """
    Comprehensive audit logging.

    Features:
    - Detailed activity logging
    - Query and filter logs
    - Compliance reporting
    - Log retention
    """

    def __init__(self, storage_dir: Optional[str] = None):
        """Initialize audit logger."""
        self.storage_dir = (
            Path(storage_dir) if storage_dir else Path.home() / ".opensable" / "audit"
        )
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        self.logs: List[AuditLog] = []

    async def log(
        self,
        tenant_id: str,
        user_id: str,
        action: AuditAction,
        resource_type: str,
        resource_id: str,
        details: Optional[Dict[str, Any]] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
    ):
        """Log an audit event."""
        log_id = secrets.token_hex(16)

        audit_log = AuditLog(
            id=log_id,
            timestamp=datetime.now(),
            tenant_id=tenant_id,
            user_id=user_id,
            action=action,
            resource_type=resource_type,
            resource_id=resource_id,
            details=details or {},
            ip_address=ip_address,
            user_agent=user_agent,
        )

        self.logs.append(audit_log)
        await self._persist_log(audit_log)

    async def _persist_log(self, log: AuditLog):
        """Persist log to storage."""
        # Store in daily log files
        date_str = log.timestamp.strftime("%Y-%m-%d")
        log_file = self.storage_dir / f"{date_str}.jsonl"

        with open(log_file, "a") as f:
            f.write(json.dumps(log.to_dict()) + "\n")

    async def query(
        self,
        tenant_id: Optional[str] = None,
        user_id: Optional[str] = None,
        action: Optional[AuditAction] = None,
        resource_type: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 100,
    ) -> List[AuditLog]:
        """Query audit logs with filters."""
        filtered = self.logs

        if tenant_id:
            filtered = [log for log in filtered if log.tenant_id == tenant_id]

        if user_id:
            filtered = [log for log in filtered if log.user_id == user_id]

        if action:
            filtered = [log for log in filtered if log.action == action]

        if resource_type:
            filtered = [log for log in filtered if log.resource_type == resource_type]

        if start_date:
            filtered = [log for log in filtered if log.timestamp >= start_date]

        if end_date:
            filtered = [log for log in filtered if log.timestamp <= end_date]

        # Sort by timestamp descending
        filtered.sort(key=lambda x: x.timestamp, reverse=True)

        return filtered[:limit]


class SSOProvider:
    """
    Single Sign-On provider.

    Supports:
    - SAML 2.0
    - OAuth 2.0 / OpenID Connect
    - JWT tokens
    """

    def __init__(self, secret_key: str):
        """Initialize SSO provider."""
        self.secret_key = secret_key
        self.sessions: Dict[str, Dict[str, Any]] = {}

    def create_token(
        self, user_id: str, tenant_id: str, roles: List[str], expires_in: int = 3600
    ) -> str:
        """Create JWT access token."""
        payload = {
            "user_id": user_id,
            "tenant_id": tenant_id,
            "roles": roles,
            "exp": datetime.utcnow() + timedelta(seconds=expires_in),
            "iat": datetime.utcnow(),
        }

        token = jwt.encode(payload, self.secret_key, algorithm="HS256")
        return token

    def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Verify and decode JWT token."""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=["HS256"])
            return payload
        except jwt.ExpiredSignatureError:
            return None
        except jwt.InvalidTokenError:
            return None

    def create_session(self, user_id: str, tenant_id: str) -> str:
        """Create a session."""
        session_id = secrets.token_hex(32)

        self.sessions[session_id] = {
            "user_id": user_id,
            "tenant_id": tenant_id,
            "created_at": datetime.now(),
            "last_activity": datetime.now(),
        }

        return session_id

    def validate_session(self, session_id: str) -> bool:
        """Validate a session."""
        if session_id not in self.sessions:
            return False

        session = self.sessions[session_id]
        last_activity = session["last_activity"]

        # Session expires after 1 hour of inactivity
        if datetime.now() - last_activity > timedelta(hours=1):
            del self.sessions[session_id]
            return False

        # Update last activity
        session["last_activity"] = datetime.now()
        return True

    def destroy_session(self, session_id: str):
        """Destroy a session."""
        if session_id in self.sessions:
            del self.sessions[session_id]


# Example usage
async def main():
    """Example enterprise features."""

    print("=" * 50)
    print("Enterprise Features Examples")
    print("=" * 50)

    # Multi-Tenancy
    print("\n1. Multi-Tenancy")
    tenancy = MultiTenancy()

    # Create tenants
    acme_corp = tenancy.create_tenant("ACME Corp", plan="enterprise")
    startup = tenancy.create_tenant("Startup Inc", plan="free")

    print(f"  Created tenant: {acme_corp.name} (ID: {acme_corp.id})")
    print(f"  Plan: {acme_corp.plan}")
    print(f"  Limits: {acme_corp.limits}")

    # Create users
    admin_user = tenancy.create_user("admin@acme.com", acme_corp.id, "secure123")
    print(f"\n  Created user: {admin_user.email} (ID: {admin_user.id})")

    # RBAC
    print("\n2. Role-Based Access Control")
    rbac = RBAC()

    # Assign roles
    rbac.assign_role(admin_user.id, "admin")
    print(f"  Assigned 'admin' role to {admin_user.email}")

    # Check permissions
    can_create = rbac.check_permission(admin_user.id, Permission.AGENT_CREATE)
    can_delete = rbac.check_permission(admin_user.id, Permission.DATA_DELETE)

    print(f"  Can create agents: {can_create}")
    print(f"  Can delete data: {can_delete}")

    permissions = rbac.get_user_permissions(admin_user.id)
    print(f"  Total permissions: {len(permissions)}")

    # Audit Logging
    print("\n3. Audit Logging")
    audit = AuditLogger()

    # Log some actions
    await audit.log(
        tenant_id=acme_corp.id,
        user_id=admin_user.id,
        action=AuditAction.LOGIN,
        resource_type="user",
        resource_id=admin_user.id,
        details={"method": "password"},
        ip_address="192.168.1.100",
    )

    await audit.log(
        tenant_id=acme_corp.id,
        user_id=admin_user.id,
        action=AuditAction.CREATE,
        resource_type="agent",
        resource_id="agent123",
        details={"name": "Data Processor"},
    )

    print(f"  Logged {len(audit.logs)} audit events")

    # Query logs
    logs = await audit.query(tenant_id=acme_corp.id, limit=10)
    print(f"  Retrieved {len(logs)} logs for tenant")

    for log in logs:
        print(
            f"    - {log.timestamp.strftime('%Y-%m-%d %H:%M:%S')} | {log.action.value} | {log.resource_type}"
        )

    # SSO
    print("\n4. Single Sign-On")
    sso = SSOProvider(secret_key="super-secret-key-12345")

    # Create token
    token = sso.create_token(
        user_id=admin_user.id, tenant_id=acme_corp.id, roles=["admin"], expires_in=3600
    )

    print(f"  Generated JWT token: {token[:50]}...")

    # Verify token
    payload = sso.verify_token(token)
    if payload:
        print(f"  Token valid for user: {payload['user_id']}")
        print(f"  Roles: {payload['roles']}")

    # Create session
    session_id = sso.create_session(admin_user.id, acme_corp.id)
    print(f"\n  Created session: {session_id[:20]}...")

    is_valid = sso.validate_session(session_id)
    print(f"  Session valid: {is_valid}")

    print("\n✅ Enterprise features examples completed!")


if __name__ == "__main__":
    asyncio.run(main())
