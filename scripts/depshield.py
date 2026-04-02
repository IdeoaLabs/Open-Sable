#!/usr/bin/env python3
"""
depshield — Supply-chain attack detector for any project.

Zero external dependencies. Works with:
  npm / yarn / pnpm, pip / poetry / pipenv, cargo, go, composer, ruby, gradle

Usage:
  depshield.py baseline          # Snapshot current deps → .depshield.json
  depshield.py scan              # Compare current state against baseline
  depshield.py scan --strict     # Exit code 1 on ANY new dependency
  depshield.py audit             # Full audit: new deps + risk scoring
  depshield.py lockcheck         # Enforce version pinning (no floating ranges)
  depshield.py hooks install     # Install git pre-commit hook
  depshield.py hooks ci          # Print CI config snippets (GitHub Actions, GitLab)

How it works:
  1. Parses your lockfiles to extract every dependency name + version + hash.
  2. Stores a baseline snapshot (.depshield.json) in your repo.
  3. On every scan, diffs current state against baseline.
  4. Flags new deps, removed deps, version changes, hash mismatches.
  5. Runs static risk analysis on new packages (typosquatting, suspicious patterns).

This catches supply-chain attacks like the axios/plain-crypto-js incident because
a compromised package that adds a new transitive dependency will ALWAYS show up
as a diff — no advisory database needed.

License: MIT
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import textwrap
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# ─────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────

VERSION = "1.0.0"
BASELINE_FILE = ".depshield.json"

# Popular packages per ecosystem — used for typosquatting detection
_POPULAR_NPM = {
    "express", "react", "react-dom", "lodash", "axios", "chalk", "commander",
    "webpack", "babel", "typescript", "eslint", "prettier", "vue", "angular",
    "moment", "dayjs", "uuid", "dotenv", "cors", "body-parser", "jsonwebtoken",
    "bcrypt", "mongoose", "sequelize", "socket.io", "next", "nuxt", "svelte",
    "crypto-js", "node-fetch", "got", "request", "cheerio", "puppeteer",
    "sharp", "multer", "passport", "jest", "mocha", "chai", "sinon",
    "underscore", "ramda", "rxjs", "graphql", "apollo", "prisma",
}

_POPULAR_PYPI = {
    "requests", "flask", "django", "numpy", "pandas", "scipy", "matplotlib",
    "tensorflow", "torch", "transformers", "pillow", "sqlalchemy", "celery",
    "redis", "boto3", "pytest", "black", "mypy", "pydantic", "fastapi",
    "uvicorn", "gunicorn", "cryptography", "paramiko", "fabric", "scrapy",
    "beautifulsoup4", "lxml", "httpx", "aiohttp", "click", "typer",
    "rich", "setuptools", "wheel", "pip", "poetry", "pipenv",
}

# Suspicious patterns in package source code
_SUSPICIOUS_PATTERNS = [
    # Shell execution
    (r'\bexecSync\b', "execSync (synchronous shell execution)", 8),
    (r'\bexec\s*\(', "exec() call", 5),
    (r'\bchild_process\b', "child_process module", 7),
    (r'\bspawn\s*\(', "spawn() process creation", 6),
    (r'\bsubprocess\b', "subprocess module usage", 4),
    (r'\bos\.system\b', "os.system() call", 7),
    (r'\bos\.popen\b', "os.popen() call", 7),

    # Obfuscation
    (r'\\x[0-9a-fA-F]{2}(?:\\x[0-9a-fA-F]{2}){10,}', "heavy hex-encoded strings", 9),
    (r'\\u[0-9a-fA-F]{4}(?:\\u[0-9a-fA-F]{4}){10,}', "heavy unicode-encoded strings", 9),
    (r'atob\s*\(', "atob() base64 decode", 6),
    (r'Buffer\.from\(.+,\s*[\'"]base64[\'"]\)', "Buffer.from(base64) decode", 7),
    (r'eval\s*\(', "eval() — dynamic code execution", 9),
    (r'Function\s*\(', "Function() constructor — dynamic code", 8),
    (r'fromCharCode', "String.fromCharCode — char-by-char construction", 5),

    # Filesystem access in libraries that shouldn't need it
    (r'writeFileSync|writeFile\b', "filesystem write", 4),
    (r'ProgramData', "ProgramData directory access (Windows staging)", 10),
    (r'[\'"]\/tmp\/', "/tmp directory access", 5),
    (r'\\\\AppData\\\\', "AppData directory access", 7),

    # Network exfiltration
    (r'https?://\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', "hardcoded IP address URL", 8),
    (r'\.onion\b', "Tor .onion address", 10),
    (r'pastebin\.com|hastebin|transfer\.sh', "paste/transfer service URL", 7),

    # Anti-forensics
    (r'unlinkSync|unlink\b.*\bexecSync', "file deletion after execution", 9),
    (r'renameSync.*unlinkSync|fs\.rename.*fs\.unlink', "rename + delete (artifact cleanup)", 9),

    # Crypto mining indicators
    (r'stratum\+tcp://', "mining pool connection", 10),
    (r'CoinHive|coinhive', "CoinHive miner", 10),

    # Install script hooks (npm-specific)
    (r'"preinstall"\s*:', "preinstall script defined", 6),
    (r'"postinstall"\s*:', "postinstall script defined", 5),
    (r'"install"\s*:', "install script defined", 4),
]

# Compiled for performance
_SUSPICIOUS_COMPILED = [
    (re.compile(pat, re.IGNORECASE), desc, score)
    for pat, desc, score in _SUSPICIOUS_PATTERNS
]


# ─────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────

@dataclass
class Dependency:
    name: str
    version: str
    ecosystem: str
    integrity: str = ""          # hash from lockfile (sha512, sha256, etc.)
    resolved_url: str = ""       # download URL
    has_install_script: bool = False
    direct: bool = False         # True if in the top-level manifest

    def key(self) -> str:
        return f"{self.ecosystem}:{self.name}"


@dataclass
class RiskSignal:
    signal: str
    severity: str   # low, medium, high, critical
    score: int      # 1-10
    detail: str = ""


@dataclass
class DiffResult:
    added: List[Dependency] = field(default_factory=list)
    removed: List[Dependency] = field(default_factory=list)
    changed: List[Tuple[Dependency, Dependency]] = field(default_factory=list)  # (old, new)
    hash_mismatch: List[Tuple[Dependency, Dependency]] = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────
# Lockfile parsers
# ─────────────────────────────────────────────────────────────────────

def _find_lockfiles(root: Path) -> Dict[str, Path]:
    """Discover all supported lockfiles in the project."""
    candidates = {
        "npm":      ["package-lock.json"],
        "yarn":     ["yarn.lock"],
        "pnpm":     ["pnpm-lock.yaml"],
        "pip":      ["requirements.txt", "requirements-lock.txt"],
        "pipenv":   ["Pipfile.lock"],
        "poetry":   ["poetry.lock"],
        "cargo":    ["Cargo.lock"],
        "go":       ["go.sum"],
        "composer":  ["composer.lock"],
        "ruby":     ["Gemfile.lock"],
        "gradle":   ["gradle.lockfile"],
    }
    found = {}
    for eco, names in candidates.items():
        for name in names:
            path = root / name
            if path.exists():
                found[eco] = path
                break
    return found


def _parse_npm_lock(path: Path) -> List[Dependency]:
    """Parse package-lock.json (v2/v3 format)."""
    deps = []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        _warn(f"Failed to parse {path}: {e}")
        return deps

    # Check top-level for direct deps
    manifest = path.parent / "package.json"
    direct_names: Set[str] = set()
    if manifest.exists():
        try:
            pkg = json.loads(manifest.read_text(encoding="utf-8"))
            direct_names = set(pkg.get("dependencies", {}).keys())
            direct_names |= set(pkg.get("devDependencies", {}).keys())
        except Exception:
            pass

    # v2/v3 uses "packages" key
    packages = data.get("packages", {})
    if packages:
        for pkg_path, info in packages.items():
            if not pkg_path:  # root package
                continue
            # "node_modules/axios" → "axios"
            # "node_modules/@scope/pkg" → "@scope/pkg"
            name = pkg_path.rsplit("node_modules/", 1)[-1] if "node_modules/" in pkg_path else pkg_path
            if not name:
                continue

            has_scripts = bool(info.get("hasInstallScript", False))
            deps.append(Dependency(
                name=name,
                version=info.get("version", "?"),
                ecosystem="npm",
                integrity=info.get("integrity", ""),
                resolved_url=info.get("resolved", ""),
                has_install_script=has_scripts,
                direct=name in direct_names,
            ))
        return deps

    # v1 fallback — "dependencies" key
    for name, info in data.get("dependencies", {}).items():
        deps.append(Dependency(
            name=name,
            version=info.get("version", "?"),
            ecosystem="npm",
            integrity=info.get("integrity", ""),
            resolved_url=info.get("resolved", ""),
            direct=name in direct_names,
        ))
        # Recurse into nested dependencies
        for sub_name, sub_info in info.get("dependencies", {}).items():
            deps.append(Dependency(
                name=sub_name,
                version=sub_info.get("version", "?"),
                ecosystem="npm",
                integrity=sub_info.get("integrity", ""),
                resolved_url=sub_info.get("resolved", ""),
            ))
    return deps


def _parse_yarn_lock(path: Path) -> List[Dependency]:
    """Parse yarn.lock (v1 format — regex-based)."""
    deps = []
    text = path.read_text(encoding="utf-8")
    # Pattern: "package@version": \n  version "x.y.z" \n  resolved "url" \n  integrity "sha..."
    pkg_re = re.compile(
        r'^"?(@?[^@\s]+)@[^":\n]+["\s]*:\s*\n'
        r'\s+version\s+"([^"]+)"'
        r'(?:\s+resolved\s+"([^"]*)")?'
        r'(?:\s+integrity\s+"([^"]*)")?',
        re.MULTILINE,
    )
    for m in pkg_re.finditer(text):
        deps.append(Dependency(
            name=m.group(1),
            version=m.group(2),
            ecosystem="yarn",
            resolved_url=m.group(3) or "",
            integrity=m.group(4) or "",
        ))
    return deps


def _parse_pnpm_lock(path: Path) -> List[Dependency]:
    """Parse pnpm-lock.yaml (basic — no yaml lib needed)."""
    deps = []
    text = path.read_text(encoding="utf-8")
    # Look for lines like:  /package-name@version:
    # or newer format:      package-name@version:
    pkg_re = re.compile(r'^\s+/?(@?[\w\-./]+)@(\d[^:\s]*)\s*:', re.MULTILINE)
    for m in pkg_re.finditer(text):
        name = m.group(1).lstrip("/")
        deps.append(Dependency(
            name=name,
            version=m.group(2),
            ecosystem="pnpm",
        ))
    return deps


def _parse_pip_requirements(path: Path) -> List[Dependency]:
    """Parse requirements.txt / requirements-lock.txt."""
    deps = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or line.startswith("-"):
            continue
        # Handle: package==1.2.3, package>=1.0, package~=2.0
        m = re.match(r'^([A-Za-z0-9_\-.\[\]]+)\s*([=<>~!]+)\s*([^\s;#]+)', line)
        if m:
            deps.append(Dependency(
                name=m.group(1).lower().replace("_", "-"),
                version=m.group(3),
                ecosystem="pip",
                direct=True,
            ))
        elif re.match(r'^[A-Za-z0-9_\-.]+$', line):
            # Unpinned dependency — just a name
            deps.append(Dependency(
                name=line.lower().replace("_", "-"),
                version="*",
                ecosystem="pip",
                direct=True,
            ))
    return deps


def _parse_pipenv_lock(path: Path) -> List[Dependency]:
    """Parse Pipfile.lock."""
    deps = []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return deps
    for section in ("default", "develop"):
        for name, info in data.get(section, {}).items():
            version = info.get("version", "?").lstrip("=")
            hashes = info.get("hashes", [])
            deps.append(Dependency(
                name=name.lower(),
                version=version,
                ecosystem="pipenv",
                integrity=hashes[0] if hashes else "",
            ))
    return deps


def _parse_poetry_lock(path: Path) -> List[Dependency]:
    """Parse poetry.lock (TOML-like, regex-based)."""
    deps = []
    text = path.read_text(encoding="utf-8")
    # [[package]] blocks
    blocks = re.split(r'^\[\[package\]\]\s*$', text, flags=re.MULTILINE)
    for block in blocks[1:]:  # skip header
        name_m = re.search(r'^name\s*=\s*"([^"]+)"', block, re.MULTILINE)
        ver_m = re.search(r'^version\s*=\s*"([^"]+)"', block, re.MULTILINE)
        if name_m and ver_m:
            deps.append(Dependency(
                name=name_m.group(1).lower(),
                version=ver_m.group(1),
                ecosystem="poetry",
            ))
    return deps


def _parse_cargo_lock(path: Path) -> List[Dependency]:
    """Parse Cargo.lock."""
    deps = []
    text = path.read_text(encoding="utf-8")
    blocks = re.split(r'^\[\[package\]\]\s*$', text, flags=re.MULTILINE)
    for block in blocks[1:]:
        name_m = re.search(r'^name\s*=\s*"([^"]+)"', block, re.MULTILINE)
        ver_m = re.search(r'^version\s*=\s*"([^"]+)"', block, re.MULTILINE)
        checksum_m = re.search(r'^checksum\s*=\s*"([^"]+)"', block, re.MULTILINE)
        if name_m and ver_m:
            deps.append(Dependency(
                name=name_m.group(1),
                version=ver_m.group(1),
                ecosystem="cargo",
                integrity=checksum_m.group(1) if checksum_m else "",
            ))
    return deps


def _parse_go_sum(path: Path) -> List[Dependency]:
    """Parse go.sum."""
    deps = []
    seen = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) >= 3:
            module = parts[0]
            version = parts[1].split("/")[0]
            key = f"{module}@{version}"
            if key not in seen:
                seen.add(key)
                deps.append(Dependency(
                    name=module,
                    version=version,
                    ecosystem="go",
                    integrity=parts[2] if len(parts) > 2 else "",
                ))
    return deps


def _parse_composer_lock(path: Path) -> List[Dependency]:
    """Parse composer.lock."""
    deps = []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return deps
    for section in ("packages", "packages-dev"):
        for pkg in data.get(section, []):
            deps.append(Dependency(
                name=pkg.get("name", ""),
                version=pkg.get("version", "?").lstrip("v"),
                ecosystem="composer",
                resolved_url=pkg.get("dist", {}).get("url", ""),
                integrity=pkg.get("dist", {}).get("reference", ""),
            ))
    return deps


def _parse_gemfile_lock(path: Path) -> List[Dependency]:
    """Parse Gemfile.lock."""
    deps = []
    in_specs = False
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip() == "GEM":
            in_specs = False
        if line.strip() == "specs:":
            in_specs = True
            continue
        if in_specs:
            m = re.match(r'^\s{4}(\S+)\s+\((\S+)\)', line)
            if m:
                deps.append(Dependency(
                    name=m.group(1),
                    version=m.group(2),
                    ecosystem="ruby",
                ))
    return deps


def _parse_gradle_lock(path: Path) -> List[Dependency]:
    """Parse gradle.lockfile."""
    deps = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        # format: group:artifact:version=hash
        parts = line.split("=")[0].split(":")
        if len(parts) >= 3:
            deps.append(Dependency(
                name=f"{parts[0]}:{parts[1]}",
                version=parts[2],
                ecosystem="gradle",
            ))
    return deps


# Parser dispatch
_PARSERS = {
    "npm":      _parse_npm_lock,
    "yarn":     _parse_yarn_lock,
    "pnpm":     _parse_pnpm_lock,
    "pip":      _parse_pip_requirements,
    "pipenv":   _parse_pipenv_lock,
    "poetry":   _parse_poetry_lock,
    "cargo":    _parse_cargo_lock,
    "go":       _parse_go_sum,
    "composer": _parse_composer_lock,
    "ruby":     _parse_gemfile_lock,
    "gradle":   _parse_gradle_lock,
}


# ─────────────────────────────────────────────────────────────────────
# Core engine
# ─────────────────────────────────────────────────────────────────────

def collect_deps(root: Path) -> Tuple[Dict[str, Path], List[Dependency]]:
    """Find lockfiles and parse all dependencies."""
    lockfiles = _find_lockfiles(root)
    all_deps: List[Dependency] = []
    for eco, path in lockfiles.items():
        parser = _PARSERS.get(eco)
        if parser:
            parsed = parser(path)
            all_deps.extend(parsed)
            _info(f"  {eco:10s}  {path.name:30s}  {len(parsed)} deps")
    return lockfiles, all_deps


def diff_deps(
    baseline: List[Dependency],
    current: List[Dependency],
) -> DiffResult:
    """Compare two dependency lists."""
    result = DiffResult()

    base_map = {d.key(): d for d in baseline}
    curr_map = {d.key(): d for d in current}

    base_keys = set(base_map.keys())
    curr_keys = set(curr_map.keys())

    # New deps
    for key in sorted(curr_keys - base_keys):
        result.added.append(curr_map[key])

    # Removed deps
    for key in sorted(base_keys - curr_keys):
        result.removed.append(base_map[key])

    # Changed version or hash
    for key in sorted(base_keys & curr_keys):
        old = base_map[key]
        new = curr_map[key]
        if old.version != new.version:
            result.changed.append((old, new))
        elif old.integrity and new.integrity and old.integrity != new.integrity:
            result.hash_mismatch.append((old, new))

    return result


# ─────────────────────────────────────────────────────────────────────
# Risk analysis
# ─────────────────────────────────────────────────────────────────────

def _levenshtein(a: str, b: str) -> int:
    """Levenshtein distance — detects typosquatting."""
    if len(a) < len(b):
        return _levenshtein(b, a)
    if len(b) == 0:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a):
        curr = [i + 1]
        for j, cb in enumerate(b):
            curr.append(min(
                prev[j + 1] + 1,
                curr[j] + 1,
                prev[j] + (0 if ca == cb else 1),
            ))
        prev = curr
    return prev[-1]


def _check_typosquat(name: str, ecosystem: str) -> Optional[RiskSignal]:
    """Check if a name is suspiciously close to a popular package."""
    # Check against ALL ecosystems — supply chain attacks cross boundaries
    popular = _POPULAR_NPM | _POPULAR_PYPI
    clean = name.lower().replace("_", "-").split("/")[-1]

    for pop in popular:
        if clean == pop:
            continue
        dist = _levenshtein(clean, pop)
        # Same prefix/suffix tricks: plain-crypto-js vs crypto-js
        if pop in clean and clean != pop:
            return RiskSignal(
                signal="typosquat",
                severity="high",
                score=8,
                detail=f"Name contains popular package '{pop}' — potential typosquat",
            )
        if dist <= 2 and len(clean) > 3:
            return RiskSignal(
                signal="typosquat",
                severity="high",
                score=7,
                detail=f"Name very similar to '{pop}' (edit distance {dist})",
            )
    return None


def _check_install_scripts(dep: Dependency) -> Optional[RiskSignal]:
    """Flag packages with install scripts."""
    if dep.has_install_script:
        return RiskSignal(
            signal="install_script",
            severity="medium",
            score=5,
            detail="Package defines install lifecycle scripts (preinstall/postinstall)",
        )
    return None


def _scan_package_source(root: Path, dep: Dependency) -> List[RiskSignal]:
    """Scan installed package source for suspicious patterns."""
    signals = []

    # Determine package path
    if dep.ecosystem in ("npm", "yarn", "pnpm"):
        pkg_dir = root / "node_modules" / dep.name
    elif dep.ecosystem in ("pip", "pipenv", "poetry"):
        # Try common venv locations
        for venv in (".venv", "venv", "env", ".env"):
            venv_path = root / venv
            if venv_path.exists():
                # packages could be in lib/pythonX.Y/site-packages/
                for sp in venv_path.rglob("site-packages"):
                    pkg_dir = sp / dep.name.replace("-", "_")
                    if pkg_dir.exists():
                        break
                else:
                    continue
                break
        else:
            return signals
    else:
        return signals  # Other ecosystems: skip source scan for now

    if not pkg_dir.exists():
        return signals

    # Scan files (limit to prevent huge scans)
    files_scanned = 0
    max_files = 50
    max_file_size = 512_000  # 500 KB

    exts = {".js", ".mjs", ".cjs", ".py", ".sh", ".bat", ".cmd", ".ps1"}

    for fpath in pkg_dir.rglob("*"):
        if files_scanned >= max_files:
            break
        if not fpath.is_file() or fpath.suffix not in exts:
            continue
        if fpath.stat().st_size > max_file_size:
            continue

        try:
            content = fpath.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue

        files_scanned += 1

        for pattern, desc, score in _SUSPICIOUS_COMPILED:
            if pattern.search(content):
                signals.append(RiskSignal(
                    signal="suspicious_code",
                    severity="high" if score >= 7 else "medium",
                    score=score,
                    detail=f"{desc} in {fpath.relative_to(root)}",
                ))

    return signals


def analyze_risks(dep: Dependency, root: Path) -> List[RiskSignal]:
    """Run all risk checks on a dependency."""
    signals = []

    typo = _check_typosquat(dep.name, dep.ecosystem)
    if typo:
        signals.append(typo)

    script = _check_install_scripts(dep)
    if script:
        signals.append(script)

    source_signals = _scan_package_source(root, dep)
    signals.extend(source_signals)

    return signals


# ─────────────────────────────────────────────────────────────────────
# Version pinning checker
# ─────────────────────────────────────────────────────────────────────

def check_pinning(root: Path) -> List[Dict[str, str]]:
    """Check for unpinned / floating version ranges in manifests."""
    issues = []

    # npm/yarn: package.json
    pkg_json = root / "package.json"
    if pkg_json.exists():
        try:
            data = json.loads(pkg_json.read_text(encoding="utf-8"))
            for section in ("dependencies", "devDependencies"):
                for name, ver in data.get(section, {}).items():
                    if ver.startswith("^") or ver.startswith("~") or ver in ("*", "latest"):
                        issues.append({
                            "file": "package.json",
                            "package": name,
                            "version": ver,
                            "issue": "Floating range — auto-pulls new versions",
                            "fix": f'Pin to exact: "{name}": "{ver.lstrip("^~")}"',
                        })
        except Exception:
            pass

    # pip: requirements.txt
    for req_file in root.glob("requirements*.txt"):
        try:
            for i, line in enumerate(req_file.read_text(encoding="utf-8").splitlines(), 1):
                line = line.strip()
                if not line or line.startswith("#") or line.startswith("-"):
                    continue
                if ">=" in line and "==" not in line:
                    issues.append({
                        "file": req_file.name,
                        "package": line.split(">=")[0].strip(),
                        "version": line,
                        "issue": "Minimum range (>=) — could pull compromised newer version",
                        "fix": f"Pin with ==: {line.split('>=')[0].strip()}=={line.split('>=')[1].split(',')[0].strip()}",
                    })
                elif re.match(r'^[A-Za-z0-9_\-.\[\]]+$', line):
                    issues.append({
                        "file": req_file.name,
                        "package": line,
                        "version": "(unpinned)",
                        "issue": "No version specified — pulls latest",
                        "fix": f"Pin with ==: {line}==X.Y.Z",
                    })
        except Exception:
            pass

    return issues


# ─────────────────────────────────────────────────────────────────────
# Baseline management
# ─────────────────────────────────────────────────────────────────────

def save_baseline(root: Path, deps: List[Dependency], lockfiles: Dict[str, Path]):
    """Save current dependency state as baseline."""
    base_path = root / BASELINE_FILE
    data = {
        "version": VERSION,
        "created": datetime.now(timezone.utc).isoformat(),
        "lockfiles": {k: str(v.relative_to(root)) for k, v in lockfiles.items()},
        "lockfile_hashes": {},
        "dependencies": [],
    }

    # Store lockfile content hashes for tamper detection
    for eco, path in lockfiles.items():
        try:
            content = path.read_bytes()
            data["lockfile_hashes"][eco] = hashlib.sha256(content).hexdigest()
        except OSError:
            pass

    for dep in deps:
        data["dependencies"].append(asdict(dep))

    base_path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return base_path


def load_baseline(root: Path) -> Optional[Tuple[dict, List[Dependency]]]:
    """Load baseline from disk."""
    base_path = root / BASELINE_FILE
    if not base_path.exists():
        return None

    try:
        data = json.loads(base_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        _warn(f"Failed to load baseline: {e}")
        return None

    deps = [
        Dependency(**d) for d in data.get("dependencies", [])
    ]
    return data, deps


# ─────────────────────────────────────────────────────────────────────
# Git hooks
# ─────────────────────────────────────────────────────────────────────

_PRE_COMMIT_HOOK = """\
#!/bin/sh
# depshield pre-commit hook — block commits that introduce risky dependencies
# Auto-installed by: depshield.py hooks install

DEPSHIELD="$(git rev-parse --show-toplevel)/scripts/depshield.py"
if [ ! -f "$DEPSHIELD" ]; then
    DEPSHIELD="$(which depshield.py 2>/dev/null || echo "")"
fi

if [ -z "$DEPSHIELD" ]; then
    echo "⚠️  depshield not found — skipping supply-chain check"
    exit 0
fi

# Check if any lockfile changed
LOCKFILES="package-lock.json yarn.lock pnpm-lock.yaml requirements.txt requirements-lock.txt Pipfile.lock poetry.lock Cargo.lock go.sum composer.lock Gemfile.lock gradle.lockfile"
CHANGED=0
for f in $LOCKFILES; do
    if git diff --cached --name-only | grep -q "^$f$"; then
        CHANGED=1
        break
    fi
done

if [ $CHANGED -eq 0 ]; then
    exit 0
fi

echo "🛡️  depshield: Lockfile changed — running supply-chain scan..."
python3 "$DEPSHIELD" scan --strict
EXIT=$?

if [ $EXIT -ne 0 ]; then
    echo ""
    echo "❌ depshield blocked this commit. New/changed dependencies detected."
    echo "   Review the changes above, then either:"
    echo "   1. Run: depshield.py baseline     (to accept the new deps)"
    echo "   2. Revert the lockfile changes"
    echo ""
fi

exit $EXIT
"""

_GITHUB_ACTIONS_CI = """\
# .github/workflows/depshield.yml
name: depshield
on:
  pull_request:
    paths:
      - 'package-lock.json'
      - 'yarn.lock'
      - 'pnpm-lock.yaml'
      - 'requirements*.txt'
      - 'Pipfile.lock'
      - 'poetry.lock'
      - 'Cargo.lock'
      - 'go.sum'
      - 'composer.lock'
      - 'Gemfile.lock'

jobs:
  supply-chain-scan:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.x'
      - name: Run depshield
        run: python3 scripts/depshield.py audit
"""

_GITLAB_CI = """\
# Add to .gitlab-ci.yml
depshield:
  stage: test
  image: python:3.12-slim
  rules:
    - changes:
        - package-lock.json
        - yarn.lock
        - requirements*.txt
        - Pipfile.lock
        - poetry.lock
        - Cargo.lock
        - go.sum
        - composer.lock
  script:
    - python3 scripts/depshield.py audit
"""


def install_hook(root: Path):
    """Install git pre-commit hook."""
    git_dir = root / ".git"
    if not git_dir.is_dir():
        _error("Not a git repository — run from repo root")
        return False

    hooks_dir = git_dir / "hooks"
    hooks_dir.mkdir(exist_ok=True)

    hook_path = hooks_dir / "pre-commit"

    if hook_path.exists():
        content = hook_path.read_text(encoding="utf-8")
        if "depshield" in content:
            _info("depshield hook already installed")
            return True
        # Append to existing hook
        _info("Appending depshield to existing pre-commit hook")
        with open(hook_path, "a", encoding="utf-8") as f:
            f.write("\n\n# === depshield ===\n")
            # Extract just the depshield logic (skip shebang)
            lines = _PRE_COMMIT_HOOK.splitlines()[1:]
            f.write("\n".join(lines) + "\n")
    else:
        hook_path.write_text(_PRE_COMMIT_HOOK, encoding="utf-8")

    hook_path.chmod(0o755)
    _success("Pre-commit hook installed")
    return True


# ─────────────────────────────────────────────────────────────────────
# Output helpers
# ─────────────────────────────────────────────────────────────────────

class _Colors:
    RED = "\033[91m"
    YELLOW = "\033[93m"
    GREEN = "\033[92m"
    CYAN = "\033[96m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RESET = "\033[0m"

# Disable colors if not a TTY or NO_COLOR is set
if not sys.stdout.isatty() or os.environ.get("NO_COLOR"):
    for attr in ("RED", "YELLOW", "GREEN", "CYAN", "BOLD", "DIM", "RESET"):
        setattr(_Colors, attr, "")

C = _Colors


def _info(msg: str):
    print(f"{C.DIM}  {msg}{C.RESET}")

def _success(msg: str):
    print(f"{C.GREEN}  ✅ {msg}{C.RESET}")

def _warn(msg: str):
    print(f"{C.YELLOW}  ⚠️  {msg}{C.RESET}", file=sys.stderr)

def _error(msg: str):
    print(f"{C.RED}  ❌ {msg}{C.RESET}", file=sys.stderr)

def _header(msg: str):
    print(f"\n{C.BOLD}{C.CYAN}{'─' * 60}{C.RESET}")
    print(f"{C.BOLD}{C.CYAN}  🛡️  {msg}{C.RESET}")
    print(f"{C.BOLD}{C.CYAN}{'─' * 60}{C.RESET}")

def _severity_color(severity: str) -> str:
    return {
        "critical": C.RED + C.BOLD,
        "high": C.RED,
        "medium": C.YELLOW,
        "low": C.DIM,
    }.get(severity, "")


def _print_dep(dep: Dependency, prefix: str = ""):
    ver = dep.version[:20]
    eco = dep.ecosystem
    name = dep.name
    print(f"    {prefix}{C.BOLD}{name}{C.RESET}  {C.DIM}{ver}  [{eco}]{C.RESET}")


def _print_risk(signal: RiskSignal):
    color = _severity_color(signal.severity)
    score_bar = "█" * signal.score + "░" * (10 - signal.score)
    print(f"      {color}[{signal.severity.upper():8s}] {score_bar}  {signal.detail}{C.RESET}")


# ─────────────────────────────────────────────────────────────────────
# Commands
# ─────────────────────────────────────────────────────────────────────

def cmd_baseline(root: Path):
    """Create or update baseline snapshot."""
    _header("depshield — Creating Baseline")
    print(f"  Scanning: {root}\n")

    lockfiles, deps = collect_deps(root)

    if not lockfiles:
        _error("No lockfiles found in this directory")
        return 1

    path = save_baseline(root, deps, lockfiles)
    print()
    _success(f"Baseline saved: {path.name}")
    _info(f"  {len(deps)} dependencies across {len(lockfiles)} lockfile(s)")
    _info(f"  Commit {BASELINE_FILE} to your repo so CI can use it")
    return 0


def cmd_scan(root: Path, strict: bool = False):
    """Compare current state against baseline."""
    _header("depshield — Supply Chain Scan")

    baseline = load_baseline(root)
    if baseline is None:
        _error(f"No baseline found. Run: depshield.py baseline")
        return 1

    meta, base_deps = baseline

    # Check lockfile tampering
    lockfiles = _find_lockfiles(root)
    stored_hashes = meta.get("lockfile_hashes", {})
    for eco, path in lockfiles.items():
        if eco in stored_hashes:
            current_hash = hashlib.sha256(path.read_bytes()).hexdigest()
            if current_hash != stored_hashes[eco]:
                _warn(f"Lockfile changed since baseline: {path.name}")

    print(f"  Baseline from: {meta.get('created', '?')[:19]}")
    print()

    _, curr_deps = collect_deps(root)
    print()

    diff = diff_deps(base_deps, curr_deps)

    exit_code = 0

    if diff.added:
        print(f"  {C.RED}{C.BOLD}NEW DEPENDENCIES ({len(diff.added)}):{C.RESET}")
        for dep in diff.added:
            _print_dep(dep, prefix=f"{C.RED}+ {C.RESET}")
        exit_code = 1 if strict else exit_code
        print()

    if diff.removed:
        print(f"  {C.GREEN}REMOVED DEPENDENCIES ({len(diff.removed)}):{C.RESET}")
        for dep in diff.removed:
            _print_dep(dep, prefix=f"{C.GREEN}- {C.RESET}")
        print()

    if diff.changed:
        print(f"  {C.YELLOW}VERSION CHANGES ({len(diff.changed)}):{C.RESET}")
        for old, new in diff.changed:
            print(f"    {C.BOLD}{old.name}{C.RESET}  {C.DIM}{old.version} → {C.YELLOW}{new.version}{C.RESET}")
        exit_code = 1 if strict else exit_code
        print()

    if diff.hash_mismatch:
        print(f"  {C.RED}{C.BOLD}⚠️  INTEGRITY HASH MISMATCHES ({len(diff.hash_mismatch)}):{C.RESET}")
        for old, new in diff.hash_mismatch:
            print(f"    {C.RED}{C.BOLD}{old.name}{C.RESET}  {C.DIM}v{old.version}{C.RESET}")
            print(f"      Old: {old.integrity[:40]}...")
            print(f"      New: {new.integrity[:40]}...")
        exit_code = 1  # Always fail on hash mismatch
        print()

    if not any([diff.added, diff.removed, diff.changed, diff.hash_mismatch]):
        _success("No dependency changes detected — supply chain clean")

    return exit_code


def cmd_audit(root: Path):
    """Full audit: diff + risk analysis on new/changed deps."""
    _header("depshield — Full Audit")

    baseline = load_baseline(root)
    if baseline is None:
        _warn("No baseline found — running first-time audit of all dependencies")
        lockfiles, curr_deps = collect_deps(root)
        if not lockfiles:
            _error("No lockfiles found")
            return 1
        print()
        total_risk = 0
        high_risk_count = 0

        print(f"  {C.BOLD}RISK ANALYSIS ({len(curr_deps)} packages):{C.RESET}\n")
        for dep in curr_deps:
            signals = analyze_risks(dep, root)
            if signals:
                _print_dep(dep)
                for sig in signals:
                    _print_risk(sig)
                    total_risk += sig.score
                    if sig.severity in ("high", "critical"):
                        high_risk_count += 1
                print()

        if total_risk == 0:
            _success("No risk signals detected")
        else:
            print(f"  {C.BOLD}Risk score: {total_risk}  |  High/critical signals: {high_risk_count}{C.RESET}")
            _info("Run: depshield.py baseline  to save this state as your baseline")

        return 1 if high_risk_count > 0 else 0

    meta, base_deps = baseline
    print(f"  Baseline: {meta.get('created', '?')[:19]}\n")

    _, curr_deps = collect_deps(root)
    print()

    diff = diff_deps(base_deps, curr_deps)

    exit_code = 0
    total_risk_score = 0
    high_risk = 0

    # Analyze new deps
    if diff.added:
        print(f"  {C.RED}{C.BOLD}NEW DEPENDENCIES — RISK ANALYSIS:{C.RESET}\n")
        for dep in diff.added:
            _print_dep(dep, prefix=f"{C.RED}+ {C.RESET}")
            signals = analyze_risks(dep, root)
            if signals:
                for sig in signals:
                    _print_risk(sig)
                    total_risk_score += sig.score
                    if sig.severity in ("high", "critical"):
                        high_risk += 1
            else:
                print(f"      {C.GREEN}No risk signals{C.RESET}")
            print()

    # Analyze changed deps
    if diff.changed:
        print(f"  {C.YELLOW}{C.BOLD}CHANGED DEPENDENCIES — RISK ANALYSIS:{C.RESET}\n")
        for old, new in diff.changed:
            print(f"    {C.BOLD}{old.name}{C.RESET}  {old.version} → {C.YELLOW}{new.version}{C.RESET}")
            signals = analyze_risks(new, root)
            if signals:
                for sig in signals:
                    _print_risk(sig)
                    total_risk_score += sig.score
                    if sig.severity in ("high", "critical"):
                        high_risk += 1
            print()

    # Hash mismatches are always critical
    if diff.hash_mismatch:
        print(f"  {C.RED}{C.BOLD}⚠️  INTEGRITY VIOLATIONS:{C.RESET}\n")
        for old, new in diff.hash_mismatch:
            print(f"    {C.RED}{C.BOLD}{old.name}{C.RESET}  — hash changed without version bump!")
            total_risk_score += 10
            high_risk += 1
        print()

    # Summary
    print(f"\n{'─' * 60}")
    if not any([diff.added, diff.changed, diff.hash_mismatch]):
        _success("Supply chain clean — no changes since baseline")
    else:
        changes = len(diff.added) + len(diff.changed) + len(diff.hash_mismatch)
        print(f"  {C.BOLD}Changes:          {changes}{C.RESET}")
        print(f"  {C.BOLD}Risk score:       {total_risk_score}{C.RESET}")
        if high_risk:
            print(f"  {C.RED}{C.BOLD}High/critical:    {high_risk} ← REVIEW REQUIRED{C.RESET}")
            exit_code = 1
        else:
            print(f"  {C.GREEN}High/critical:    0{C.RESET}")

    return exit_code


def cmd_lockcheck(root: Path):
    """Check for floating version ranges."""
    _header("depshield — Version Pin Check")

    issues = check_pinning(root)
    if not issues:
        _success("All dependencies properly pinned")
        return 0

    print(f"\n  {C.YELLOW}{C.BOLD}FLOATING VERSION RANGES ({len(issues)}):{C.RESET}\n")
    for issue in issues:
        print(f"    {C.BOLD}{issue['package']}{C.RESET}  {C.DIM}({issue['file']}){C.RESET}")
        print(f"      Current: {issue['version']}")
        print(f"      Issue:   {issue['issue']}")
        print(f"      Fix:     {issue['fix']}")
        print()

    _warn(f"{len(issues)} packages with floating ranges — vulnerable to supply-chain attacks")
    return 1


def cmd_hooks(root: Path, action: str):
    """Manage git hooks and CI config."""
    if action == "install":
        return 0 if install_hook(root) else 1
    elif action == "ci":
        _header("depshield — CI Configuration")
        print(f"\n  {C.BOLD}GitHub Actions:{C.RESET}\n")
        print(textwrap.indent(_GITHUB_ACTIONS_CI, "    "))
        print(f"\n  {C.BOLD}GitLab CI:{C.RESET}\n")
        print(textwrap.indent(_GITLAB_CI, "    "))
        return 0
    else:
        _error(f"Unknown hook action: {action}. Use 'install' or 'ci'")
        return 1


# ─────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        prog="depshield",
        description="🛡️  Supply-chain attack detector — zero deps, any language",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            examples:
              depshield.py baseline          # Save current dep state
              depshield.py scan              # Check for changes
              depshield.py scan --strict     # Fail on ANY new dep
              depshield.py audit             # Full risk analysis
              depshield.py lockcheck         # Find floating versions
              depshield.py hooks install     # Git pre-commit hook
              depshield.py hooks ci          # CI config snippets
        """),
    )
    parser.add_argument("--version", action="version", version=f"depshield {VERSION}")
    parser.add_argument(
        "--root", type=Path, default=Path("."),
        help="Project root directory (default: current dir)",
    )

    sub = parser.add_subparsers(dest="command", help="Command to run")

    sub.add_parser("baseline", help="Create/update dependency baseline")

    scan_p = sub.add_parser("scan", help="Compare against baseline")
    scan_p.add_argument("--strict", action="store_true", help="Fail on any new dependency")

    sub.add_parser("audit", help="Full audit with risk analysis")
    sub.add_parser("lockcheck", help="Check for floating version ranges")

    hooks_p = sub.add_parser("hooks", help="Git hooks and CI config")
    hooks_p.add_argument("action", choices=["install", "ci"], help="install hook or show CI config")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 0

    root = args.root.resolve()

    if args.command == "baseline":
        return cmd_baseline(root)
    elif args.command == "scan":
        return cmd_scan(root, strict=args.strict)
    elif args.command == "audit":
        return cmd_audit(root)
    elif args.command == "lockcheck":
        return cmd_lockcheck(root)
    elif args.command == "hooks":
        return cmd_hooks(root, args.action)

    return 0


if __name__ == "__main__":
    sys.exit(main())
