# Security Guide

## Overview

Open-Sable is designed for personal use with local-first processing. This guide helps you understand the security model and how to harden your deployment.

## Current Security Architecture

### Process-Level Sandbox

**What it does:**
- Isolates tool execution in separate processes
- Limits CPU time, memory, file descriptors
- Prevents fork bombs (RLIMIT_NPROC = 0)
- Blocks core dumps
- Basic network blocking via environment variables

**What it does NOT do:**
- Filesystem isolation (no chroot/jail)
- Network isolation (no netns/iptables)
- Full syscall filtering (no seccomp)
- Kernel-level security (no AppArmor/SELinux)

### Static Code Analysis

Blocks obvious patterns:
```python
exec()        # Code execution
eval()        # Code evaluation
os.system()   # Shell commands
subprocess.*  # Process spawning
__import__    # Dynamic imports
rm -rf        # Dangerous shell commands
```

**Limitations:** Cannot detect obfuscated code or logic-based attacks.

## Threat Model

### In Scope (Protected)

✅ **Resource exhaustion**
- CPU bombs via `RLIMIT_CPU`
- Memory bombs via `RLIMIT_AS`
- Fork bombs via `RLIMIT_NPROC`

✅ **Obvious malicious code**
- Static pattern matching
- Import validation (if using allowlist)

✅ **Process isolation**
- Separate session groups
- Independent process namespace

### Out of Scope (NOT Protected)

❌ **Filesystem attacks**
- Path traversal: `../../etc/passwd`
- Symlink attacks
- File exfiltration

❌ **Network attacks**
- SSRF via allowed libraries
- DNS exfiltration
- Reverse shells

❌ **Advanced code injection**
- Prompt injection to generate malicious tools
- Obfuscated payloads
- Time-delayed attacks

❌ **Supply chain**
- Malicious packages in marketplace
- Dependency confusion
- Typosquatting

## Production Hardening

### 1. Container Isolation (Recommended)

Run tool synthesis in Docker with strict policies:

```dockerfile
FROM python:3.11-slim

# Non-root user
RUN useradd -m -u 1000 sandbox
USER sandbox

# Read-only root
WORKDIR /workspace
VOLUME /workspace

# Minimal image
RUN pip install --no-cache-dir <minimal-deps>

ENTRYPOINT ["python3", "-I"]
```

```yaml
# docker-compose.yml
services:
  tool-runner:
    build: ./sandbox
    read_only: true
    tmpfs:
      - /tmp:size=64M,noexec
    security_opt:
      - no-new-privileges:true
      - seccomp=seccomp-profile.json
    cap_drop:
      - ALL
    network_mode: none  # No network access
    mem_limit: 256m
    cpus: 0.5
    pids_limit: 10
```

### 2. Seccomp Profile

Limit syscalls to essentials:

```json
{
  "defaultAction": "SCMP_ACT_ERRNO",
  "architectures": ["SCMP_ARCH_X86_64"],
  "syscalls": [
    {"names": ["read", "write", "close", "fstat"], "action": "SCMP_ACT_ALLOW"},
    {"names": ["brk", "mmap", "munmap"], "action": "SCMP_ACT_ALLOW"},
    {"names": ["exit", "exit_group"], "action": "SCMP_ACT_ALLOW"}
  ]
}
```

### 3. Filesystem Jail

Mount workspace as read-only with limited write:

```bash
# Podman with additional isolation
podman run \
  --rm \
  --read-only \
  --tmpfs /tmp:rw,size=64M,noexec \
  --network none \
  --pids-limit 10 \
  --memory 256m \
  --cpu-shares 512 \
  --security-opt no-new-privileges \
  --cap-drop ALL \
  -v ./workspace:/workspace:ro \
  sandbox-runner python3 tool.py
```

### 4. Network Policies

For production with network access:

```yaml
# Kubernetes NetworkPolicy
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: tool-runner-policy
spec:
  podSelector:
    matchLabels:
      app: opensable-tools
  policyTypes:
    - Egress
  egress:
    - to:
        - podSelector:
            matchLabels:
              app: opensable-core
      ports:
        - protocol: TCP
          port: 8080
    # Block all other egress
```

### 5. Monitoring & Alerting

```python
# Example: Prometheus metrics
from prometheus_client import Counter, Histogram

tool_executions = Counter('tool_executions_total', 'Total tool executions', ['status'])
tool_duration = Histogram('tool_duration_seconds', 'Tool execution duration')
tool_failures = Counter('tool_failures_total', 'Tool failures', ['error_type'])

# Alert on suspicious patterns
if failures_per_minute > 10:
    alert("Possible attack: high tool failure rate")

if cpu_time > 60 * cpu_limit:
    alert("Possible CPU bomb attempt")
```

## Marketplace Security

### Code Signing

```python
# Generate signing key (one-time)
from cryptography.hazmat.primitives.asymmetric import ed25519

private_key = ed25519.Ed25519PrivateKey.generate()
public_key = private_key.public_key()

# Sign skill package
signature = private_key.sign(package_bytes)

# Verify before installation
try:
    public_key.verify(signature, package_bytes)
except InvalidSignature:
    raise SecurityError("Invalid skill signature")
```

### Dependency Pinning

```toml
# skill.toml
[dependencies]
requests = "==2.31.0"  # Exact version, not >=
numpy = "==1.24.3"

[metadata]
min-opensable-version = "0.1.0"
max-opensable-version = "0.2.0"
```

### Review Process

1. **Automated scanning**:
   - Run static analysis on all code
   - Check for known CVEs in dependencies
   - Verify signature

2. **Manual review** (high-risk skills):
   - Code audit by maintainers
   - Sandboxed execution test
   - Community feedback period

3. **Reputation system**:
   - Developer trust score
   - Download count
   - User ratings
   - Issue reports

## Incident Response

### Detection

Monitor for:
- Unusual network activity
- High CPU/memory usage
- Frequent tool failures
- Unexpected file access
- Large data transfers

### Response Playbook

1. **Isolate**: Stop agent, disconnect network
2. **Preserve**: Backup logs and state
3. **Analyze**: Review audit logs, identify attack vector
4. **Remediate**: Remove malicious tools/skills
5. **Report**: File security issue (privately)
6. **Improve**: Update safeguards

### Reporting Security Issues

**DO NOT** open public GitHub issues for security vulnerabilities.

Email: security@opensable.dev (PGP key available)

Include:
- Attack vector description
- Steps to reproduce
- Impact assessment
- Proposed mitigation (if any)

## Security Checklist

### For Users

- [ ] Run Open-Sable behind firewall
- [ ] Use dedicated machine/VM for agent
- [ ] Enable audit logging
- [ ] Review tool synthesis output before execution
- [ ] Only install skills from trusted developers
- [ ] Keep Open-Sable and dependencies updated
- [ ] Use VPN for remote access
- [ ] Enable E2EE for multi-device sync

### For Developers

- [ ] Never execute untrusted code outside sandbox
- [ ] Validate all user inputs
- [ ] Use parameterized queries (no string concatenation)
- [ ] Log security-relevant events
- [ ] Implement rate limiting
- [ ] Add CSRF tokens for web interface
- [ ] Use secure session management
- [ ] Regular dependency audits (`pip-audit`)

### For Production Deployments

- [ ] Container isolation with seccomp
- [ ] Network policies (block egress by default)
- [ ] Read-only root filesystem
- [ ] Secret management (Vault, SOPS)
- [ ] Automated security scanning (Trivy, Snyk)
- [ ] Intrusion detection (Falco)
- [ ] Log aggregation (ELK, Loki)
- [ ] Disaster recovery plan
- [ ] Regular penetration testing

## References

- [OWASP AI Security](https://owasp.org/www-project-ai-security-and-privacy-guide/)
- [Docker Security Best Practices](https://docs.docker.com/engine/security/)
- [Kubernetes Security](https://kubernetes.io/docs/concepts/security/)
- [Seccomp Profiles](https://docs.docker.com/engine/security/seccomp/)

---

**Last Updated**: 2026-02-17
**Security Contact**: security@opensable.dev
