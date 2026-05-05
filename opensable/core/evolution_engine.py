"""
Evolution Engine,  Autonomous code mutation and self-restart.

Bridges the gap between evolution *analysis* (skill_evolution.py) and
actual code generation / modification.  When the evolutionary pipeline
identifies condemned, stagnant, or error-prone skills, this engine:

  1. Reads the target skill source code
  2. Builds an LLM prompt with fitness data + mutation context
  3. Asks the LLM to generate an improved version
  4. Validates the generated code (syntax, security, import test)
  5. Backs up the original → applies the evolved version
  6. Optionally hot-reloads the module or signals a full restart

Self-Restart:
  The agent can trigger its own restart when deep changes require it.
  It uses ``start.sh restart --profile <profile>`` via a detached
  subprocess that outlives the current process.

Uses:
  - skill_evolution.py  → fitness analysis, condemned lists
  - self_modify.py      → hot-reload, audit trail
  - skill_creator.py    → syntax/security validation
"""

from __future__ import annotations

import ast
import asyncio
import json
import logging
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)

# ─── Constants ─────────────────────────────────────────────────────────────────

_MAX_MUTATIONS_PER_TICK = 2          # Don't mutate too many skills at once
_MIN_TICKS_BETWEEN_MUTATIONS = 10    # Cool-down between mutation batches
_BACKUP_DIR_NAME = "evolution_backups"
_EVOLVED_LOG = "evolution_mutations.jsonl"
_RESTART_COOLDOWN_S = 300            # Min 5 min between restarts
_MAX_LLM_RETRIES = 2                # Retry code-gen if validation fails


# ─── Data ──────────────────────────────────────────────────────────────────────

@dataclass
class MutationRecord:
    """Record of a single code mutation attempt."""
    tick: int
    timestamp: float
    skill_name: str
    skill_path: str
    mutation_type: str           # "improve", "fix_errors", "evolve", "recombine"
    reason: str                  # Why this skill was selected
    original_hash: str           # SHA256 of original source
    success: bool = False
    error: Optional[str] = None
    backup_path: Optional[str] = None
    changes_summary: Optional[str] = None
    hot_reloaded: bool = False
    restart_needed: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ─── Security checks ──────────────────────────────────────────────────────────

_FORBIDDEN_PATTERNS = [
    r"os\.system\(",
    r"subprocess\..*shell\s*=\s*True",
    r"\beval\(",
    r"\bexec\(",
    r"__import__\(",
    r"open\(.*/etc/",
    r"open\(.*/root/",
    r"shutil\.rmtree\(\s*['\"]\/",
    r"rm\s+-rf\s+\/",
]


def _validate_code(source: str) -> Dict[str, Any]:
    """Validate Python code: syntax + security + sandbox load test."""
    # Syntax
    try:
        ast.parse(source)
    except SyntaxError as e:
        return {"valid": False, "error": f"SyntaxError at line {e.lineno}: {e.msg}"}

    # Security
    for pattern in _FORBIDDEN_PATTERNS:
        if re.search(pattern, source):
            return {"valid": False, "error": f"Security: forbidden pattern '{pattern}'"}

    # Sandbox load test,  run in the venv to catch missing imports
    import tempfile
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False,
    ) as tmp:
        tmp.write(source + "\nprint('__EVOLUTION_LOAD_OK__')\n")
        tmp.flush()
        try:
            result = subprocess.run(
                [sys.executable, tmp.name],
                capture_output=True, text=True, timeout=15,
            )
            if "__EVOLUTION_LOAD_OK__" not in result.stdout:
                stderr_snip = result.stderr.strip()[-300:] if result.stderr else "unknown error"
                return {"valid": False, "error": f"Load test failed: {stderr_snip}"}
        except subprocess.TimeoutExpired:
            return {"valid": False, "error": "Load test timed out (15s)"}
        except Exception as e:
            return {"valid": False, "error": f"Load test error: {e}"}
        finally:
            os.unlink(tmp.name)

    return {"valid": True}


def _hash_source(source: str) -> str:
    import hashlib
    return hashlib.sha256(source.encode()).hexdigest()[:16]


# ─── Skill Discovery ──────────────────────────────────────────────────────────

def _is_inside_project(path: Path, base_dir: Path) -> bool:
    """Verify a resolved path is inside the project directory.

    Prevents symlink escapes and path traversal.
    """
    try:
        path.resolve().relative_to(base_dir.resolve())
        return True
    except ValueError:
        return False


def _find_skill_file(skill_name: str, base_dir: Path) -> Optional[Path]:
    """Locate a skill file by name.

    Searches ONLY inside the project:
      1. opensable/skills/**/<skill_name>.py
      2. opensable/skills/**/*<skill_name>*.py   (fuzzy)
      3. opensable/core/<skill_name>.py

    All results are validated to be inside base_dir (no symlink escapes).
    """
    skills_dir = base_dir / "opensable" / "skills"
    core_dir = base_dir / "opensable" / "core"

    # Normalise name
    safe_name = re.sub(r"[^a-z0-9_]", "_", skill_name.lower())

    # Exact match in skills tree
    for candidate in skills_dir.rglob(f"{safe_name}.py"):
        if candidate.is_file() and "__pycache__" not in str(candidate):
            if _is_inside_project(candidate, base_dir):
                return candidate

    # Fuzzy match (skill_ prefix)
    for candidate in skills_dir.rglob(f"*{safe_name}*.py"):
        if candidate.is_file() and "__pycache__" not in str(candidate):
            if _is_inside_project(candidate, base_dir):
                return candidate

    # Core module
    core_file = core_dir / f"{safe_name}.py"
    if core_file.is_file() and _is_inside_project(core_file, base_dir):
        return core_file

    return None


def _skill_module_name(skill_path: Path, base_dir: Path) -> Optional[str]:
    """Convert a file path to a Python module name."""
    try:
        rel = skill_path.relative_to(base_dir)
        parts = list(rel.with_suffix("").parts)
        return ".".join(parts)
    except ValueError:
        return None


# ─── LLM Prompt Builder ───────────────────────────────────────────────────────

def _build_mutation_prompt(
    skill_name: str,
    source: str,
    mutation_type: str,
    fitness_data: Dict[str, Any],
    error_samples: List[str],
    evolution_context: str,
) -> List[Dict[str, str]]:
    """Build the LLM messages for code mutation."""

    system_msg = (
        "You are an autonomous skill evolution engine. Your job is to improve "
        "Python skill modules based on fitness data and error analysis.\n\n"
        "RULES:\n"
        "1. Return ONLY the complete improved Python source code.\n"
        "2. Do NOT include ```python or ``` markers,  raw code only.\n"
        "3. Preserve the module's public API (function/class names, signatures).\n"
        "4. Preserve all imports,  add new ones if needed.\n"
        "5. Improve error handling, performance, and reliability.\n"
        "6. Fix known bugs from the error samples provided.\n"
        "7. Add or improve docstrings.\n"
        "8. Keep all existing functionality,  do not remove features.\n"
        "9. Do NOT use os.system(), eval(), exec(), or subprocess with shell=True.\n"
        "10. The code must parse as valid Python."
    )

    fitness_summary = ""
    if fitness_data:
        fitness_summary = (
            f"\nFitness data:\n"
            f"  Score: {fitness_data.get('fitness_score', 'N/A')}\n"
            f"  Usage count: {fitness_data.get('usage_count', 0)}\n"
            f"  Error count: {fitness_data.get('error_count', 0)}\n"
            f"  Ticks alive: {fitness_data.get('ticks_alive', 0)}\n"
            f"  Times evolved: {fitness_data.get('times_evolved', 0)}\n"
            f"  Generation: {fitness_data.get('generation', 0)}\n"
        )

    errors_text = ""
    if error_samples:
        errors_text = "\nRecent errors:\n" + "\n".join(
            f"  - {e[:200]}" for e in error_samples[:5]
        )

    user_msg = (
        f"MUTATION TYPE: {mutation_type}\n"
        f"SKILL: {skill_name}\n"
        f"{fitness_summary}"
        f"{errors_text}\n"
        f"\nEVOLUTION CONTEXT:\n{evolution_context}\n"
        f"\n--- CURRENT SOURCE CODE ---\n{source}\n--- END SOURCE CODE ---\n"
        f"\nGenerate the improved version of this skill. "
        f"Return ONLY the complete Python source code."
    )

    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]


# ─── Self-Restart ──────────────────────────────────────────────────────────────

class SelfRestart:
    """Manages the agent's ability to restart itself.

    Strategy:
      1. Write a restart-request marker file
      2. Spawn a detached ``start.sh restart --profile <profile>`` process
      3. The current process exits gracefully
      4. start.sh kills the old PID and starts a fresh one
    """

    def __init__(self, base_dir: Path, profile: str = "sable"):
        self.base_dir = base_dir
        self.profile = profile
        # Per-profile markers so agents don't interfere with each other
        self._restart_marker = base_dir / "data" / f".restart_requested_{profile}"
        self._last_restart_file = base_dir / "data" / f".last_restart_{profile}"

    def can_restart(self) -> bool:
        """Check cooldown,  prevent restart storms."""
        if self._last_restart_file.exists():
            try:
                ts = float(self._last_restart_file.read_text().strip())
                if time.time() - ts < _RESTART_COOLDOWN_S:
                    remaining = _RESTART_COOLDOWN_S - (time.time() - ts)
                    logger.info(
                        f"🔄 Restart cooldown: {remaining:.0f}s remaining"
                    )
                    return False
            except (ValueError, OSError):
                pass
        return True

    def request_restart(self, reason: str = "") -> bool:
        """Request a graceful restart.

        Returns True if the restart was initiated.
        """
        if not self.can_restart():
            logger.warning("🔄 Restart blocked by cooldown")
            return False

        start_sh = self.base_dir / "start.sh"
        if not start_sh.is_file():
            logger.error("🔄 Cannot restart: start.sh not found")
            return False

        logger.warning(f"🔄 SELF-RESTART requested: {reason}")

        # Record timestamp
        self._last_restart_file.parent.mkdir(parents=True, exist_ok=True)
        self._last_restart_file.write_text(str(time.time()))

        # Write marker for post-restart inspection
        self._restart_marker.parent.mkdir(parents=True, exist_ok=True)
        self._restart_marker.write_text(json.dumps({
            "reason": reason,
            "timestamp": time.time(),
            "profile": self.profile,
            "pid": os.getpid(),
        }))

        # Spawn detached restart process
        try:
            cmd = [str(start_sh), "restart", "--profile", self.profile]
            logger.info(f"🔄 Spawning: {' '.join(cmd)}")

            # Use setsid to detach from our process group so the restart
            # script outlives us when start.sh kills our PID.
            subprocess.Popen(
                cmd,
                cwd=str(self.base_dir),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                stdin=subprocess.DEVNULL,
                start_new_session=True,
            )

            logger.warning("🔄 Restart process spawned,  shutting down gracefully")

            # Give the restart script a moment to register before we stop
            # The autonomous loop will pick up self._should_stop and exit
            return True

        except Exception as e:
            logger.error(f"🔄 Failed to spawn restart: {e}")
            self._restart_marker.unlink(missing_ok=True)
            return False

    def was_restart_requested(self) -> bool:
        """Check if there's a pending restart marker (post-restart)."""
        return self._restart_marker.is_file()

    def clear_restart_marker(self) -> Optional[Dict]:
        """Clear the restart marker and return its contents."""
        if not self._restart_marker.is_file():
            return None
        try:
            data = json.loads(self._restart_marker.read_text())
            self._restart_marker.unlink(missing_ok=True)
            return data
        except Exception:
            self._restart_marker.unlink(missing_ok=True)
            return None


# ─── Evolution Engine ──────────────────────────────────────────────────────────

class EvolutionEngine:
    """Autonomous code mutation engine.

    Integrates with:
      - SkillEvolutionManager  → fitness + condemned lists
      - SelfModificationEngine → hot-reload + audit
      - Agent LLM              → code generation

    Lifecycle:
      1. ``evaluate()`` is called from ``_cognitive_tick()`` with evolution results
      2. If condemned/stagnant/error-prone skills are found, ``mutate()`` runs
      3. LLM generates improved code
      4. Code is validated, backed up, and applied
      5. Module is hot-reloaded or restart is requested
    """

    def __init__(
        self,
        base_dir: Path,
        data_dir: Path,
        profile: str = "sable",
        *,
        max_mutations_per_tick: int = _MAX_MUTATIONS_PER_TICK,
        min_ticks_between_mutations: int = _MIN_TICKS_BETWEEN_MUTATIONS,
    ):
        self.base_dir = base_dir
        self.data_dir = data_dir
        self.profile = profile
        self.max_mutations_per_tick = max_mutations_per_tick
        self.min_ticks_between_mutations = min_ticks_between_mutations

        # Directories
        self.backup_dir = data_dir / _BACKUP_DIR_NAME
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = data_dir / _EVOLVED_LOG

        # State
        self._last_mutation_tick = 0
        self._mutation_history: List[MutationRecord] = []
        self._consecutive_failures = 0

        # Self-restart
        self.restarter = SelfRestart(base_dir, profile)

        # Check post-restart marker
        marker = self.restarter.clear_restart_marker()
        if marker:
            logger.info(
                f"🧬 Post-restart: previous restart at "
                f"{datetime.fromtimestamp(marker.get('timestamp', 0)).isoformat()} "
                f"reason={marker.get('reason', '?')}"
            )

        self._load_history()

    # ── Public API ─────────────────────────────────────────────────────────

    async def evaluate_and_mutate(
        self,
        tick: int,
        evolution_result: Dict[str, Any],
        llm,
        *,
        error_samples: Optional[Dict[str, List[str]]] = None,
    ) -> Dict[str, Any]:
        """Main entry point: evaluate evolution results and mutate if needed.

        Args:
            tick: Current tick number
            evolution_result: Output from SkillEvolutionManager.evaluate_tick()
            llm: Agent's LLM instance (must have invoke_with_tools)
            error_samples: Optional {skill_name: [error_message, ...]}

        Returns:
            Summary dict with mutations performed
        """
        error_samples = error_samples or {}

        # Cool-down check
        if tick - self._last_mutation_tick < self.min_ticks_between_mutations:
            return {"action": "cooldown", "mutations": []}

        # Backoff after consecutive failures
        if self._consecutive_failures >= 3:
            backoff_ticks = self.min_ticks_between_mutations * (2 ** min(self._consecutive_failures - 2, 4))
            if tick - self._last_mutation_tick < backoff_ticks:
                return {"action": "backoff", "mutations": [], "backoff_ticks": backoff_ticks}

        # Gather mutation targets
        targets = self._select_targets(evolution_result)
        if not targets:
            return {"action": "none", "mutations": []}

        # Perform mutations
        mutations = []
        restart_needed = False

        for target in targets[:self.max_mutations_per_tick]:
            skill_name = target["name"]
            mutation_type = target["type"]
            reason = target["reason"]

            result = await self._mutate_skill(
                tick=tick,
                skill_name=skill_name,
                mutation_type=mutation_type,
                reason=reason,
                fitness_data=target.get("fitness", {}),
                error_samples=error_samples.get(skill_name, []),
                evolution_context=self._build_context(evolution_result),
                llm=llm,
            )

            mutations.append(result)

            if result.success:
                self._consecutive_failures = 0
                if result.restart_needed:
                    restart_needed = True
            else:
                self._consecutive_failures += 1

        self._last_mutation_tick = tick

        # Trigger restart if any mutation requires it
        summary = {
            "action": "mutated",
            "mutations": [m.to_dict() for m in mutations],
            "success_count": sum(1 for m in mutations if m.success),
            "fail_count": sum(1 for m in mutations if not m.success),
            "restart_needed": restart_needed,
        }

        if restart_needed:
            reasons = [m.skill_name for m in mutations if m.restart_needed]
            self.restarter.request_restart(
                reason=f"Evolution mutated core modules: {', '.join(reasons)}"
            )
            summary["restart_initiated"] = True

        return summary

    def request_restart(self, reason: str = "Manual evolution restart") -> bool:
        """Explicitly request a restart (callable from autonomous_mode)."""
        return self.restarter.request_restart(reason)

    def get_stats(self) -> Dict[str, Any]:
        """Return engine statistics."""
        return {
            "total_mutations": len(self._mutation_history),
            "successful": sum(1 for m in self._mutation_history if m.success),
            "failed": sum(1 for m in self._mutation_history if not m.success),
            "last_mutation_tick": self._last_mutation_tick,
            "consecutive_failures": self._consecutive_failures,
            "can_restart": self.restarter.can_restart(),
        }

    # ── Target selection ───────────────────────────────────────────────────

    def _select_targets(
        self, evolution_result: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Select skills to mutate, prioritised by severity."""
        targets: List[Dict[str, Any]] = []
        fitness_lookup = {}
        for fr in evolution_result.get("fitness", []):
            fitness_lookup[fr["name"]] = fr

        # 1. Error-prone skills (highest priority)
        for name in evolution_result.get("error_driven", []):
            targets.append({
                "name": name,
                "type": "fix_errors",
                "reason": f"High error rate ({fitness_lookup.get(name, {}).get('error_count', '?')} errors)",
                "fitness": fitness_lookup.get(name, {}),
                "priority": 10,
            })

        # 2. Condemned skills
        for name in evolution_result.get("condemned", []):
            if name not in {t["name"] for t in targets}:
                targets.append({
                    "name": name,
                    "type": "evolve",
                    "reason": f"Low fitness (score={fitness_lookup.get(name, {}).get('fitness_score', '?')})",
                    "fitness": fitness_lookup.get(name, {}),
                    "priority": 8,
                })

        # 3. Stagnant skills (lower priority,  they work but haven't evolved)
        for name in evolution_result.get("stagnant", []):
            if name not in {t["name"] for t in targets}:
                targets.append({
                    "name": name,
                    "type": "improve",
                    "reason": "Stagnant,  used but never evolved",
                    "fitness": fitness_lookup.get(name, {}),
                    "priority": 5,
                })

        # Sort by priority descending
        targets.sort(key=lambda t: t["priority"], reverse=True)
        return targets

    # ── Core mutation ──────────────────────────────────────────────────────

    async def _mutate_skill(
        self,
        tick: int,
        skill_name: str,
        mutation_type: str,
        reason: str,
        fitness_data: Dict[str, Any],
        error_samples: List[str],
        evolution_context: str,
        llm,
    ) -> MutationRecord:
        """Attempt to mutate a single skill."""

        # Find the skill file
        skill_path = _find_skill_file(skill_name, self.base_dir)
        if not skill_path:
            return MutationRecord(
                tick=tick, timestamp=time.time(),
                skill_name=skill_name, skill_path="",
                mutation_type=mutation_type, reason=reason,
                original_hash="", success=False,
                error=f"Skill file not found: {skill_name}",
            )

        # Safety: verify path is inside the project (belt-and-suspenders)
        if not _is_inside_project(skill_path, self.base_dir):
            return MutationRecord(
                tick=tick, timestamp=time.time(),
                skill_name=skill_name, skill_path=str(skill_path),
                mutation_type=mutation_type, reason=reason,
                original_hash="", success=False,
                error=f"BLOCKED: path escapes project dir: {skill_path}",
            )

        # Read source
        try:
            source = skill_path.read_text(encoding="utf-8")
        except Exception as e:
            return MutationRecord(
                tick=tick, timestamp=time.time(),
                skill_name=skill_name, skill_path=str(skill_path),
                mutation_type=mutation_type, reason=reason,
                original_hash="", success=False,
                error=f"Cannot read source: {e}",
            )

        original_hash = _hash_source(source)

        # Skip very large files,  LLM context would be too long
        if len(source) > 50_000:
            return MutationRecord(
                tick=tick, timestamp=time.time(),
                skill_name=skill_name, skill_path=str(skill_path),
                mutation_type=mutation_type, reason=reason,
                original_hash=original_hash, success=False,
                error=f"Source too large ({len(source)} chars) for LLM mutation",
            )

        # Build LLM prompt
        messages = _build_mutation_prompt(
            skill_name=skill_name,
            source=source,
            mutation_type=mutation_type,
            fitness_data=fitness_data,
            error_samples=error_samples,
            evolution_context=evolution_context,
        )

        # Generate new code with retries
        new_source = None
        last_error = ""

        for attempt in range(_MAX_LLM_RETRIES + 1):
            try:
                result = await llm.invoke_with_tools(messages, [])
                content = (
                    result.get("text", "") or result.get("content", "")
                    if isinstance(result, dict)
                    else str(result)
                )

                if not content or not content.strip():
                    last_error = "LLM returned empty response"
                    continue

                # Strip markdown code fences if present
                cleaned = self._strip_code_fences(content)

                # Validate
                validation = _validate_code(cleaned)
                if not validation["valid"]:
                    last_error = validation["error"]
                    # Add error feedback for retry
                    messages.append({"role": "assistant", "content": content})
                    messages.append({
                        "role": "user",
                        "content": (
                            f"The code has an error: {validation['error']}\n"
                            "Please fix it and return the complete corrected source code."
                        ),
                    })
                    continue

                # Check it's not identical
                if _hash_source(cleaned) == original_hash:
                    last_error = "Generated code identical to original"
                    messages.append({"role": "assistant", "content": content})
                    messages.append({
                        "role": "user",
                        "content": (
                            "The code you returned is identical to the original. "
                            "Please make meaningful improvements and return the updated version."
                        ),
                    })
                    continue

                new_source = cleaned
                break

            except Exception as e:
                last_error = f"LLM error: {e}"
                logger.warning(f"🧬 Mutation LLM attempt {attempt + 1} failed: {e}")

        if new_source is None:
            record = MutationRecord(
                tick=tick, timestamp=time.time(),
                skill_name=skill_name, skill_path=str(skill_path),
                mutation_type=mutation_type, reason=reason,
                original_hash=original_hash, success=False,
                error=f"Code generation failed after {_MAX_LLM_RETRIES + 1} attempts: {last_error}",
            )
            self._append_log(record)
            return record

        # Backup original
        backup_path = self._backup_file(skill_path, tick)

        # Write new code
        try:
            skill_path.write_text(new_source, encoding="utf-8")
        except Exception as e:
            # Restore from backup
            if backup_path and backup_path.exists():
                shutil.copy2(backup_path, skill_path)
            record = MutationRecord(
                tick=tick, timestamp=time.time(),
                skill_name=skill_name, skill_path=str(skill_path),
                mutation_type=mutation_type, reason=reason,
                original_hash=original_hash, success=False,
                error=f"Write failed: {e}",
                backup_path=str(backup_path) if backup_path else None,
            )
            self._append_log(record)
            return record

        # Try hot-reload
        module_name = _skill_module_name(skill_path, self.base_dir)
        hot_reloaded = False
        restart_needed = False

        if module_name and module_name in sys.modules:
            try:
                import importlib
                importlib.reload(sys.modules[module_name])
                hot_reloaded = True
                logger.info(f"🧬 Hot-reloaded: {module_name}")
            except Exception as e:
                logger.warning(
                    f"🧬 Hot-reload failed for {module_name}: {e},  "
                    "will need restart"
                )
                restart_needed = True
        else:
            # Module not loaded (or can't determine name),  restart for safety
            if module_name:
                restart_needed = True

        # Compute changes summary
        orig_lines = len(source.splitlines())
        new_lines = len(new_source.splitlines())
        changes_summary = (
            f"Lines: {orig_lines} → {new_lines} "
            f"(Δ{new_lines - orig_lines:+d})"
        )

        record = MutationRecord(
            tick=tick, timestamp=time.time(),
            skill_name=skill_name, skill_path=str(skill_path),
            mutation_type=mutation_type, reason=reason,
            original_hash=original_hash, success=True,
            backup_path=str(backup_path) if backup_path else None,
            changes_summary=changes_summary,
            hot_reloaded=hot_reloaded,
            restart_needed=restart_needed,
        )
        self._mutation_history.append(record)
        self._append_log(record)

        logger.info(
            f"🧬 Mutation SUCCESS: {skill_name} ({mutation_type}) "
            f",  {changes_summary}"
            f"{' [hot-reloaded]' if hot_reloaded else ''}"
            f"{' [restart needed]' if restart_needed else ''}"
        )

        return record

    # ── Rollback ───────────────────────────────────────────────────────────

    def rollback_last(self, skill_name: str) -> bool:
        """Rollback the last mutation for a skill."""
        for record in reversed(self._mutation_history):
            if record.skill_name == skill_name and record.success and record.backup_path:
                backup = Path(record.backup_path)
                target = Path(record.skill_path)
                if backup.exists() and target.exists():
                    try:
                        shutil.copy2(backup, target)
                        logger.info(f"🔄 Rolled back {skill_name} to {backup.name}")

                        # Try hot-reload the rollback
                        module_name = _skill_module_name(target, self.base_dir)
                        if module_name and module_name in sys.modules:
                            import importlib
                            importlib.reload(sys.modules[module_name])

                        return True
                    except Exception as e:
                        logger.error(f"Rollback failed: {e}")
                        return False
        logger.warning(f"No rollback available for {skill_name}")
        return False

    # ── Helpers ─────────────────────────────────────────────────────────────

    def _backup_file(self, path: Path, tick: int) -> Optional[Path]:
        """Create a timestamped backup of a file."""
        try:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup = self.backup_dir / f"{path.stem}_t{tick}_{ts}{path.suffix}"
            shutil.copy2(path, backup)
            logger.debug(f"Backup: {path.name} → {backup.name}")
            return backup
        except Exception as e:
            logger.warning(f"Backup failed for {path}: {e}")
            return None

    @staticmethod
    def _strip_code_fences(text: str) -> str:
        """Remove markdown code fences from LLM output."""
        text = text.strip()
        # Remove leading ```python or ```
        if text.startswith("```"):
            first_nl = text.index("\n") if "\n" in text else len(text)
            text = text[first_nl + 1:]
        # Remove trailing ```
        if text.endswith("```"):
            text = text[:-3]
        return text.strip()

    def _build_context(self, evolution_result: Dict[str, Any]) -> str:
        """Build evolution context string from evaluation results."""
        parts = []
        niche = evolution_result.get("niche", {})
        parts.append(
            f"Active skills: {niche.get('cap_count', '?')}"
        )
        landscape = evolution_result.get("landscape", {})
        if landscape.get("ruggedness"):
            parts.append(f"Landscape ruggedness: {landscape['ruggedness']:.3f}")

        pressures = evolution_result.get("selection_pressure", [])
        if pressures:
            parts.append(f"Pressures: {' | '.join(str(p) for p in pressures)}")

        condemned = evolution_result.get("condemned", [])
        if condemned:
            parts.append(f"Condemned: {', '.join(condemned)}")

        return "\n".join(parts)

    def _append_log(self, record: MutationRecord) -> None:
        """Append a mutation record to the JSONL log."""
        try:
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(record.to_dict(), default=str) + "\n")
        except Exception as e:
            logger.warning(f"Failed to log mutation: {e}")

    def _load_history(self) -> None:
        """Load mutation history from disk."""
        if not self.log_file.exists():
            return
        try:
            records = []
            with open(self.log_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        d = json.loads(line)
                        records.append(MutationRecord(**{
                            k: v for k, v in d.items()
                            if k in MutationRecord.__dataclass_fields__
                        }))
            self._mutation_history = records
            if records:
                self._last_mutation_tick = max(r.tick for r in records)
            logger.debug(f"Loaded {len(records)} mutation records")
        except Exception as e:
            logger.warning(f"Failed to load mutation history: {e}")
