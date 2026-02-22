"""
Enterprise Features Examples - Multi-tenancy, RBAC, Audit logging, SSO.

Demonstrates tenant management, role-based access control, audit trails, and authentication.
"""

import asyncio
from core.enterprise import MultiTenancy, RBAC, AuditLogger, SSOProvider, Permission, AuditAction


async def main():
    """Run enterprise features examples."""

    print("=" * 60)
    print("Enterprise Features Examples")
    print("=" * 60)

    # Example 1: Multi-tenancy setup
    print("\n1. Multi-Tenancy Setup")
    print("-" * 40)

    tenancy = MultiTenancy()

    # Create tenants
    acme = tenancy.create_tenant("ACME Corporation", plan="enterprise")
    startup = tenancy.create_tenant("Startup Inc", plan="pro")
    free_tier = tenancy.create_tenant("Free User", plan="free")

    print(f"Created {len(tenancy.tenants)} tenants:")
    print(f"  1. {acme.name} ({acme.plan})")
    print(f"     Limits: {acme.limits}")
    print(f"  2. {startup.name} ({startup.plan})")
    print(f"     Limits: {startup.limits}")
    print(f"  3. {free_tier.name} ({free_tier.plan})")
    print(f"     Limits: {free_tier.limits}")

    # Example 2: User management
    print("\n2. User Management")
    print("-" * 40)

    # Create users
    admin = tenancy.create_user("admin@acme.com", acme.id, "secure_password_123")
    dev1 = tenancy.create_user("dev1@acme.com", acme.id, "dev_pass_456")
    dev2 = tenancy.create_user("dev2@acme.com", acme.id, "dev_pass_789")

    print(f"Created {len(tenancy.users)} users:")
    for user in [admin, dev1, dev2]:
        print(f"  - {user.email} (Tenant: {user.tenant_id[:8]}...)")

    # Example 3: RBAC - Role assignment
    print("\n3. Role-Based Access Control")
    print("-" * 40)

    rbac = RBAC()

    # Assign roles
    rbac.assign_role(admin.id, "admin")
    rbac.assign_role(dev1.id, "developer")
    rbac.assign_role(dev2.id, "operator")

    print("Assigned roles:")
    print(f"  {admin.email}: admin")
    print(f"  {dev1.email}: developer")
    print(f"  {dev2.email}: operator")

    # Example 4: Permission checking
    print("\n4. Permission Checking")
    print("-" * 40)

    permissions_to_check = [
        (admin.id, Permission.AGENT_DELETE, "admin"),
        (dev1.id, Permission.AGENT_CREATE, "developer"),
        (dev2.id, Permission.AGENT_DELETE, "operator"),
    ]

    print("Permission checks:")
    for user_id, permission, role in permissions_to_check:
        has_perm = rbac.check_permission(user_id, permission)
        status = "✅" if has_perm else "❌"
        print(f"  {status} {role} can {permission.value}: {has_perm}")

    # Example 5: Custom roles
    print("\n5. Custom Role Creation")
    print("-" * 40)

    custom_permissions = {
        Permission.AGENT_READ,
        Permission.AGENT_EXECUTE,
        Permission.DATA_READ,
        Permission.WORKFLOW_READ,
        Permission.WORKFLOW_EXECUTE,
    }

    custom_role = rbac.create_role(
        name="analyst",
        permissions=custom_permissions,
        description="Data analyst with read and execute permissions",
    )

    print(f"Created custom role: {custom_role.name}")
    print(f"Permissions: {len(custom_role.permissions)}")

    # Example 6: Audit logging
    print("\n6. Audit Logging")
    print("-" * 40)

    audit = AuditLogger()

    # Log various actions
    await audit.log(
        tenant_id=acme.id,
        user_id=admin.id,
        action=AuditAction.LOGIN,
        resource_type="session",
        resource_id="session_123",
        details={"method": "password", "ip": "192.168.1.100"},
        ip_address="192.168.1.100",
        user_agent="Mozilla/5.0...",
    )

    await audit.log(
        tenant_id=acme.id,
        user_id=dev1.id,
        action=AuditAction.CREATE,
        resource_type="agent",
        resource_id="agent_456",
        details={"name": "Data Processor", "type": "automation"},
    )

    await audit.log(
        tenant_id=acme.id,
        user_id=dev2.id,
        action=AuditAction.EXECUTE,
        resource_type="workflow",
        resource_id="workflow_789",
        details={"status": "success", "duration_ms": 1523},
    )

    await audit.log(
        tenant_id=acme.id,
        user_id=admin.id,
        action=AuditAction.PERMISSION_GRANT,
        resource_type="role",
        resource_id="analyst",
        details={"user": dev1.email, "role": "analyst"},
    )

    print(f"Logged {len(audit.logs)} audit events")

    # Example 7: Audit queries
    print("\n7. Audit Log Queries")
    print("-" * 40)

    # Query by tenant
    tenant_logs = await audit.query(tenant_id=acme.id)
    print(f"Logs for ACME Corp: {len(tenant_logs)}")

    # Query by user
    user_logs = await audit.query(user_id=dev1.id)
    print(f"Logs for dev1: {len(user_logs)}")

    # Query by action
    create_logs = await audit.query(action=AuditAction.CREATE)
    print(f"CREATE actions: {len(create_logs)}")

    # Detailed log display
    print("\nRecent audit events:")
    for log in tenant_logs[:3]:
        print(f"  [{log.timestamp.strftime('%Y-%m-%d %H:%M:%S')}]")
        print(f"  Action: {log.action.value}")
        print(f"  Resource: {log.resource_type}/{log.resource_id}")
        print(f"  User: {log.user_id[:8]}...")
        print()

    # Example 8: SSO and authentication
    print("\n8. Single Sign-On (SSO)")
    print("-" * 40)

    sso = SSOProvider(secret_key="super-secret-key-change-in-production")

    # Create JWT token
    token = sso.create_token(
        user_id=admin.id, tenant_id=acme.id, roles=["admin"], expires_in=3600  # 1 hour
    )

    print("Generated JWT token:")
    print(f"  {token[:50]}...")

    # Verify token
    payload = sso.verify_token(token)
    if payload:
        print("\nToken verified:")
        print(f"  User ID: {payload['user_id'][:8]}...")
        print(f"  Tenant ID: {payload['tenant_id'][:8]}...")
        print(f"  Roles: {payload['roles']}")
        print(f"  Expires: {payload['exp']}")

    # Example 9: Session management
    print("\n9. Session Management")
    print("-" * 40)

    # Create session
    session_id = sso.create_session(admin.id, acme.id)
    print(f"Created session: {session_id[:20]}...")

    # Validate session
    is_valid = sso.validate_session(session_id)
    print(f"Session valid: {is_valid}")

    # Destroy session
    sso.destroy_session(session_id)
    print("Session destroyed")

    is_valid = sso.validate_session(session_id)
    print(f"Session valid after destroy: {is_valid}")

    # Example 10: Resource quotas
    print("\n10. Resource Quota Management")
    print("-" * 40)

    # Check quotas
    quotas = [("agents", 3), ("workflows", 8), ("api_calls", 500)]

    print(f"Checking quotas for {acme.name} ({acme.plan}):")
    for resource, current in quotas:
        within_quota = tenancy.check_quota(acme.id, resource, current)
        status = "✅" if within_quota else "❌"
        limit = acme.limits.get(resource, 0)
        print(f"  {status} {resource}: {current}/{limit} ({within_quota})")

    # Example 11: Tenant isolation
    print("\n11. Tenant Isolation")
    print("-" * 40)

    # Create user in different tenant
    other_user = tenancy.create_user("user@startup.com", startup.id, "pass")

    print("Tenant isolation test:")
    print(f"  ACME admin tenant: {admin.tenant_id[:8]}...")
    print(f"  Startup user tenant: {other_user.tenant_id[:8]}...")
    print(f"  Isolated: {admin.tenant_id != other_user.tenant_id}")

    # Example 12: Compliance and audit reports
    print("\n12. Compliance Audit Report")
    print("-" * 40)

    from datetime import datetime, timedelta

    # Get last 24 hours of logs
    start_time = datetime.now() - timedelta(hours=24)
    recent_logs = await audit.query(tenant_id=acme.id, start_date=start_time, limit=100)

    print(f"Compliance report for {acme.name}:")
    print("  Period: Last 24 hours")
    print(f"  Total events: {len(recent_logs)}")

    # Count by action type
    action_counts = {}
    for log in recent_logs:
        action = log.action.value
        action_counts[action] = action_counts.get(action, 0) + 1

    print("  Events by type:")
    for action, count in action_counts.items():
        print(f"    - {action}: {count}")

    print("\n" + "=" * 60)
    print("✅ Enterprise features examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
