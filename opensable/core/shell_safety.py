"""
Shell Safety — 3-Tier Command Classification

Tier 1 (safe)    : Read-only, non-destructive — auto-execute
Tier 2 (moderate): Can modify state — require user confirmation
Tier 3 (blocked) : Dangerous / destructive — blocked outright

Supports per-task allowlists for background workflows (e.g., a build
workflow may auto-approve `npm run build` without confirmation).
"""
from __future__ import annotations

import logging
import re
import shlex
from enum import Enum, auto
from typing import Dict, FrozenSet, List, Optional, Set

logger = logging.getLogger(__name__)


class Tier(Enum):
    SAFE = auto()       # Tier 1 — auto-execute
    MODERATE = auto()   # Tier 2 — ask user
    BLOCKED = auto()    # Tier 3 — reject


# ── Command Classification Tables ─────────────────────────────────────

SAFE_COMMANDS: FrozenSet[str] = frozenset({
    # Navigation / inspection
    "ls", "dir", "pwd", "cd", "echo", "cat", "head", "tail", "less", "more",
    "wc", "file", "stat", "du", "df", "find", "locate", "which", "type",
    "whereis", "tree", "readlink", "basename", "dirname", "realpath",
    # Text processing (read-only)
    "grep", "egrep", "fgrep", "rg", "awk", "sed", "cut", "sort", "uniq",
    "tr", "diff", "comm", "paste", "column", "jq", "yq", "xmllint",
    # System info (read-only)
    "date", "cal", "uptime", "whoami", "id", "hostname", "uname",
    "lsb_release", "free", "top", "htop", "ps", "lscpu", "lsblk",
    "lsusb", "lspci", "ip", "ifconfig", "ping", "dig", "nslookup",
    "curl", "wget", "env", "printenv", "set",
    # Git (read-only)
    "git status", "git log", "git diff", "git branch", "git remote",
    "git show", "git tag", "git stash list", "git describe",
    # Python / Node (read-only)
    "python --version", "python3 --version", "node --version",
    "npm --version", "pip --version", "pip list", "pip show",
    "npm list", "npm ls",
    # Misc
    "true", "false", "test", "printf", "seq", "yes",
})

MODERATE_COMMANDS: FrozenSet[str] = frozenset({
    # File modification
    "rm", "mv", "cp", "mkdir", "rmdir", "touch", "ln", "rename",
    "chmod", "chown", "chgrp",
    # Package management
    "pip", "pip3", "npm", "npx", "yarn", "pnpm", "cargo",
    "apt", "apt-get", "brew", "dnf", "yum", "pacman", "snap",
    "gem", "composer", "go",
    # Git (write)
    "git add", "git commit", "git push", "git pull", "git fetch",
    "git merge", "git rebase", "git checkout", "git switch",
    "git reset", "git stash", "git cherry-pick", "git clone",
    # Build / run
    "make", "cmake", "gcc", "g++", "clang", "rustc", "javac",
    "python", "python3", "node", "deno", "bun",
    "docker", "docker-compose", "podman",
    # Editors (non-destructive)
    "nano", "vim", "vi", "code",
    # Network
    "ssh", "scp", "rsync", "ftp", "sftp",
    # Process control
    "kill", "killall", "pkill", "nohup", "screen", "tmux",
    # Crontab
    "crontab",
})

BLOCKED_COMMANDS: FrozenSet[str] = frozenset({
    # System destructive
    "shutdown", "reboot", "poweroff", "halt", "init",
    "systemctl poweroff", "systemctl reboot",
    # Disk destructive
    "mkfs", "fdisk", "parted", "dd", "shred", "wipefs",
    "format",  # Windows
    # Recursive force
    "rm -rf /", "rm -rf /*", "rm -rf ~",
    # Privilege escalation
    "sudo su", "su -", "passwd",
    # Network spoofing
    "iptables", "ip6tables", "nftables",
    "arp", "arping",
    # Dangerous redirects
    "> /dev/sda", "> /dev/null",  # only blocking device writes
})

# Patterns that should always block regardless of command
BLOCKED_PATTERNS: list[re.Pattern] = [
    re.compile(r"rm\s+(-[a-zA-Z]*f[a-zA-Z]*\s+)?/\s*$"),         # rm -rf /
    re.compile(r"rm\s+(-[a-zA-Z]*f[a-zA-Z]*\s+)?/\*"),            # rm -rf /*
    re.compile(r":\(\)\s*\{\s*:\|:\s*&\s*\}\s*;"),                 # fork bomb
    re.compile(r">\s*/dev/sd[a-z]"),                                # write to raw device
    re.compile(r"mkfs\.", re.IGNORECASE),                           # any mkfs variant
    re.compile(r"dd\s+.*of=/dev/sd[a-z]", re.IGNORECASE),         # dd to device
    re.compile(r"\bformat\s+[A-Z]:", re.IGNORECASE),               # Windows format
    re.compile(r"curl\s+.*\|\s*(ba)?sh", re.IGNORECASE),          # pipe curl to shell
    re.compile(r"wget\s+.*\|\s*(ba)?sh", re.IGNORECASE),          # pipe wget to shell
    re.compile(r"eval\s*\(.*base64", re.IGNORECASE),              # encoded eval
]


class ShellSafety:
    """
    Classify and gate shell commands by safety tier.

    Usage:
        safety = ShellSafety()
        tier, reason = safety.classify("ls -la")
        # tier == Tier.SAFE
    """

    def __init__(self):
        self._task_allowlists: Dict[str, Set[str]] = {}

    # ── Public API ────────────────────────────────────────────────────

    def classify(self, command: str, task_id: Optional[str] = None) -> tuple[Tier, str]:
        """
        Classify a shell command into a safety tier.

        Returns: (Tier, reason_string)
        """
        cmd = command.strip()
        if not cmd:
            return Tier.SAFE, "empty command"

        # Step 1: Check blocked patterns first (highest priority)
        for pat in BLOCKED_PATTERNS:
            if pat.search(cmd):
                return Tier.BLOCKED, f"Matches dangerous pattern: {pat.pattern}"

        # Step 2: Parse the base command
        base = self._extract_base_command(cmd)
        full_prefix = self._extract_command_prefix(cmd)

        # Step 3: Check blocked commands
        if base in BLOCKED_COMMANDS or full_prefix in BLOCKED_COMMANDS:
            return Tier.BLOCKED, f"'{base}' is a blocked command"

        # Step 4: Check task-specific allowlist
        if task_id and task_id in self._task_allowlists:
            allowed = self._task_allowlists[task_id]
            if base in allowed or full_prefix in allowed or cmd in allowed:
                return Tier.SAFE, f"Allowed by task '{task_id}' allowlist"

        # Step 5: Check safe commands
        if base in SAFE_COMMANDS or full_prefix in SAFE_COMMANDS:
            # Double-check: safe commands with pipe to shell are moderate
            if re.search(r"\|\s*(ba)?sh", cmd):
                return Tier.MODERATE, "Piped to shell interpreter"
            return Tier.SAFE, f"'{base}' is a safe read-only command"

        # Step 6: Check moderate commands
        if base in MODERATE_COMMANDS or full_prefix in MODERATE_COMMANDS:
            return Tier.MODERATE, f"'{base}' can modify system state"

        # Step 7: Unknown commands default to moderate
        return Tier.MODERATE, f"Unknown command '{base}' — requires confirmation"

    def is_allowed(self, command: str, task_id: Optional[str] = None) -> bool:
        """Quick check: is this command safe to auto-execute?"""
        tier, _ = self.classify(command, task_id)
        return tier == Tier.SAFE

    def is_blocked(self, command: str) -> bool:
        """Quick check: is this command outright blocked?"""
        tier, _ = self.classify(command)
        return tier == Tier.BLOCKED

    # ── Task Allowlists ──────────────────────────────────────────────

    def set_task_allowlist(self, task_id: str, commands: List[str]):
        """Set per-task command allowlist (e.g., build workflow)."""
        self._task_allowlists[task_id] = set(commands)

    def clear_task_allowlist(self, task_id: str):
        if task_id in self._task_allowlists:
            del self._task_allowlists[task_id]

    # ── Helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _extract_base_command(cmd: str) -> str:
        """Extract the first word/command from a shell string."""
        # Strip leading env vars, sudo, etc.
        cleaned = re.sub(r"^(\w+=\S+\s+)*", "", cmd.strip())
        cleaned = re.sub(r"^(sudo\s+)+", "", cleaned)
        try:
            parts = shlex.split(cleaned)
        except ValueError:
            parts = cleaned.split()
        return parts[0] if parts else cmd

    @staticmethod
    def _extract_command_prefix(cmd: str) -> str:
        """Extract 'command subcommand' prefix (e.g., 'git status')."""
        cleaned = re.sub(r"^(\w+=\S+\s+)*", "", cmd.strip())
        cleaned = re.sub(r"^(sudo\s+)+", "", cleaned)
        try:
            parts = shlex.split(cleaned)
        except ValueError:
            parts = cleaned.split()
        if len(parts) >= 2:
            return f"{parts[0]} {parts[1]}"
        return parts[0] if parts else cmd
