"""
SableCore Skill Factory — Autonomous Skill Creation Engine

Teaches the agent to create, validate, and publish its own skills.
Uses SKILL.md format with YAML frontmatter.

Skill structure:
  - SKILL.md with YAML frontmatter (name + description)
  - Optional scripts/, references/, assets/ directories
  - Progressive disclosure: metadata → body → bundled resources
"""

import json
import logging
import re
import ast
import subprocess
import tempfile
import os
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field, asdict
from datetime import datetime

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Skill Blueprint — describes a skill before it is generated
# ---------------------------------------------------------------------------


@dataclass
class SkillBlueprint:
    """Blueprint describing what a skill should do before code is generated."""

    name: str
    description: str
    category: str
    triggers: List[str] = field(default_factory=list)
    examples: List[str] = field(default_factory=list)
    inputs: List[Dict[str, str]] = field(default_factory=list)
    outputs: List[Dict[str, str]] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    complexity: str = "simple"  # simple | medium | complex
    async_required: bool = False
    needs_network: bool = False
    needs_filesystem: bool = False

    def to_prompt(self) -> str:
        """Convert blueprint into an LLM-ready prompt for code generation."""
        parts = [
            f"Create a Python skill called '{self.name}'.",
            f"Description: {self.description}",
            f"Category: {self.category}",
        ]
        if self.triggers:
            parts.append(f"Trigger words: {', '.join(self.triggers)}")
        if self.examples:
            parts.append(f"Usage examples: {'; '.join(self.examples)}")
        if self.inputs:
            inputs_str = ", ".join(f"{i['name']}: {i.get('type', 'str')}" for i in self.inputs)
            parts.append(f"Inputs: {inputs_str}")
        if self.outputs:
            outputs_str = ", ".join(f"{o['name']}: {o.get('type', 'Any')}" for o in self.outputs)
            parts.append(f"Expected outputs: {outputs_str}")
        if self.dependencies:
            parts.append(f"Allowed imports: {', '.join(self.dependencies)}")
        if self.async_required:
            parts.append("Use async/await pattern.")
        parts.append(
            "Return a single Python function (or set of functions) with docstrings. "
            "Include error handling. Keep it concise."
        )
        return "\n".join(parts)


# ---------------------------------------------------------------------------
# Code templates for common skill patterns
# ---------------------------------------------------------------------------

SKILL_TEMPLATES = {
    "api_fetcher": '''"""
{name} — {description}
"""
import aiohttp

async def {func_name}({params}) -> dict:
    """{description}"""
    url = "{api_url}"
    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(url, params={"{param_name}": {first_param}}) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return {{"success": True, "data": data}}
                return {{"success": False, "status": resp.status}}
        except Exception as e:
            return {{"error": str(e)}}
''',
    "data_processor": '''"""
{name} — {description}
"""
import json
from pathlib import Path

def {func_name}({params}) -> dict:
    """{description}"""
    try:
        # Process the input data
        result = {processing_logic}
        return {{"success": True, "result": result}}
    except Exception as e:
        return {{"error": str(e)}}
''',
    "file_handler": '''"""
{name} — {description}
"""
from pathlib import Path
import json

def {func_name}({params}) -> dict:
    """{description}"""
    path = Path({first_param})
    try:
        if not path.exists():
            return {{"error": f"Not found: {{{first_param}}}"}}
        content = path.read_text(errors='replace')
        # Process the file
        return {{"success": True, "path": str(path), "content": content, "size": path.stat().st_size}}
    except Exception as e:
        return {{"error": str(e)}}
''',
    "cli_wrapper": '''"""
{name} — {description}
"""
import subprocess

def {func_name}({params}) -> dict:
    """{description}"""
    try:
        result = subprocess.run(
            {command},
            capture_output=True, text=True, timeout=30
        )
        return {{
            "stdout": result.stdout.strip(),
            "stderr": result.stderr.strip(),
            "success": result.returncode == 0
        }}
    except subprocess.TimeoutExpired:
        return {{"error": "Command timed out"}}
    except Exception as e:
        return {{"error": str(e)}}
''',
    "storage_manager": '''"""
{name} — {description}
"""
import json
from pathlib import Path
from datetime import datetime

STORAGE = Path.home() / '.sablecore' / '{storage_name}.json'

def _load():
    if STORAGE.exists():
        return json.loads(STORAGE.read_text())
    return []

def _save(data):
    STORAGE.parent.mkdir(parents=True, exist_ok=True)
    STORAGE.write_text(json.dumps(data, indent=2))

def add_{item_name}({params}) -> dict:
    """{description}"""
    items = _load()
    item = {{
        'id': len(items) + 1,
        {item_fields}
        'created': datetime.now().isoformat()
    }}
    items.append(item)
    _save(items)
    return item

def list_{item_name}s() -> list:
    """List all items."""
    return _load()

def remove_{item_name}(item_id: int) -> dict:
    """Remove item by ID."""
    items = _load()
    items = [i for i in items if i['id'] != item_id]
    _save(items)
    return {{"removed": item_id}}
''',
    "generic": '''"""
{name} — {description}
"""

def {func_name}({params}) -> dict:
    """{description}"""
    try:
        # Implementation
        {implementation}
        return {{"success": True, "result": result}}
    except Exception as e:
        return {{"error": str(e)}}
''',
}


# ---------------------------------------------------------------------------
# Cross-platform templates: JavaScript and Rust
# ---------------------------------------------------------------------------

JS_TEMPLATES = {
    "api_fetcher": """// {name} — {description}

/**
 * {description}
 * @param {{{param_type}}} {first_param}
 * @returns {{Promise<Object>}}
 */
async function {func_name}({params}) {{
  try {{
    const resp = await fetch("{api_url}" + "?" + new URLSearchParams({{ {first_param} }}));
    if (!resp.ok) return {{ success: false, status: resp.status }};
    const data = await resp.json();
    return {{ success: true, data }};
  }} catch (err) {{
    return {{ error: err.message }};
  }}
}}

module.exports = {{ {func_name} }};
""",
    "data_processor": """// {name} — {description}

/**
 * {description}
 * @param {{*}} {first_param}
 * @returns {{Object}}
 */
function {func_name}({params}) {{
  try {{
    const result = {processing_logic};
    return {{ success: true, result }};
  }} catch (err) {{
    return {{ error: err.message }};
  }}
}}

module.exports = {{ {func_name} }};
""",
    "file_handler": """// {name} — {description}
const fs = require("fs");
const path = require("path");

/**
 * {description}
 * @param {{string}} filePath
 * @returns {{Object}}
 */
function {func_name}(filePath) {{
  try {{
    if (!fs.existsSync(filePath)) return {{ error: `Not found: ${{filePath}}` }};
    const content = fs.readFileSync(filePath, "utf-8");
    const stats = fs.statSync(filePath);
    return {{ success: true, path: filePath, content, size: stats.size }};
  }} catch (err) {{
    return {{ error: err.message }};
  }}
}}

module.exports = {{ {func_name} }};
""",
    "cli_wrapper": """// {name} — {description}
const {{ execSync }} = require("child_process");

/**
 * {description}
 * @param {{string}} command
 * @returns {{Object}}
 */
function {func_name}(command) {{
  try {{
    const stdout = execSync(command, {{ timeout: 30000 }}).toString().trim();
    return {{ success: true, stdout }};
  }} catch (err) {{
    return {{ success: false, stderr: err.stderr?.toString() || err.message }};
  }}
}}

module.exports = {{ {func_name} }};
""",
    "generic": """// {name} — {description}

/**
 * {description}
 * @param {{*}} {first_param}
 * @returns {{Object}}
 */
function {func_name}({params}) {{
  try {{
    {implementation}
    return {{ success: true, result }};
  }} catch (err) {{
    return {{ error: err.message }};
  }}
}}

module.exports = {{ {func_name} }};
""",
}


RUST_TEMPLATES = {
    "api_fetcher": """//! {name} — {description}

use reqwest;
use serde_json::Value;
use std::error::Error;

/// {description}
pub async fn {func_name}({params}) -> Result<Value, Box<dyn Error>> {{
    let url = format!("{api_url}?{first_param}={{}}", {first_param});
    let resp = reqwest::get(&url).await?;
    let data: Value = resp.json().await?;
    Ok(data)
}}
""",
    "data_processor": """//! {name} — {description}

use serde_json::{{json, Value}};
use std::error::Error;

/// {description}
pub fn {func_name}({params}) -> Result<Value, Box<dyn Error>> {{
    // Process the input
    let result = {processing_logic};
    Ok(json!({{ "success": true, "result": result }}))
}}
""",
    "file_handler": """//! {name} — {description}

use std::fs;
use std::path::Path;
use serde_json::{{json, Value}};
use std::error::Error;

/// {description}
pub fn {func_name}(file_path: &str) -> Result<Value, Box<dyn Error>> {{
    let path = Path::new(file_path);
    if !path.exists() {{
        return Err(format!("Not found: {{}}", file_path).into());
    }}
    let content = fs::read_to_string(path)?;
    let size = fs::metadata(path)?.len();
    Ok(json!({{
        "success": true,
        "path": file_path,
        "content": content,
        "size": size
    }}))
}}
""",
    "cli_wrapper": """//! {name} — {description}

use std::process::Command;
use serde_json::{{json, Value}};
use std::error::Error;

/// {description}
pub fn {func_name}(cmd: &str, args: &[&str]) -> Result<Value, Box<dyn Error>> {{
    let output = Command::new(cmd).args(args).output()?;
    let stdout = String::from_utf8_lossy(&output.stdout).trim().to_string();
    let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
    Ok(json!({{
        "success": output.status.success(),
        "stdout": stdout,
        "stderr": stderr
    }}))
}}
""",
    "generic": """//! {name} — {description}

use serde_json::{{json, Value}};
use std::error::Error;

/// {description}
pub fn {func_name}({params}) -> Result<Value, Box<dyn Error>> {{
    // Implementation
    {implementation}
    Ok(json!({{ "success": true, "result": result }}))
}}
""",
}


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


class SkillValidator:
    """Validate generated skill code for safety and correctness."""

    FORBIDDEN_IMPORTS = {
        "os.system",
        "subprocess.call",
        "eval",
        "exec",
        "__import__",
        "importlib",
        "ctypes",
        "pickle.loads",
    }

    DANGEROUS_PATTERNS = [
        r"os\.system\s*\(",
        r"subprocess\.call\s*\(",
        r"__import__\s*\(",
        r"eval\s*\(",
        r"exec\s*\(",
        r"shutil\.rmtree\s*\(\s*['\"/]",  # rm root/home
        r"open\s*\(['\"]\/etc",  # system files
    ]

    @classmethod
    def validate_syntax(cls, code: str) -> Dict[str, Any]:
        """Check Python syntax validity."""
        try:
            ast.parse(code)
            return {"valid": True}
        except SyntaxError as e:
            return {
                "valid": False,
                "error": str(e),
                "line": e.lineno,
                "offset": e.offset,
            }

    @classmethod
    def validate_safety(cls, code: str) -> Dict[str, Any]:
        """Check for dangerous patterns."""
        warnings = []
        for pattern in cls.DANGEROUS_PATTERNS:
            matches = re.findall(pattern, code)
            if matches:
                warnings.append(f"Dangerous pattern found: {pattern}")

        return {
            "safe": len(warnings) == 0,
            "warnings": warnings,
        }

    @classmethod
    def validate_structure(cls, code: str) -> Dict[str, Any]:
        """Check that the code defines at least one function."""
        try:
            tree = ast.parse(code)
            functions = [
                node.name
                for node in ast.walk(tree)
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            ]
            has_docstring = any(
                isinstance(node.body[0], ast.Expr)
                and isinstance(node.body[0].value, (ast.Str, ast.Constant))
                for node in ast.walk(tree)
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.body
            )
            return {
                "has_functions": len(functions) > 0,
                "functions": functions,
                "has_docstrings": has_docstring,
            }
        except Exception as e:
            return {"has_functions": False, "error": str(e)}

    @classmethod
    def run_sandbox_test(cls, code: str, timeout: int = 10) -> Dict[str, Any]:
        """Execute code in an isolated process to verify it loads cleanly."""
        test_code = code + "\n\nprint('__SKILL_LOAD_OK__')\n"
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(test_code)
            f.flush()
            try:
                result = subprocess.run(
                    ["python3", f.name],
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                )
                ok = "__SKILL_LOAD_OK__" in result.stdout
                return {
                    "loadable": ok,
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "exit_code": result.returncode,
                }
            except subprocess.TimeoutExpired:
                return {"loadable": False, "error": "Sandbox test timed out"}
            except Exception as e:
                return {"loadable": False, "error": str(e)}
            finally:
                os.unlink(f.name)

    @classmethod
    def full_validate(cls, code: str) -> Dict[str, Any]:
        """Run all validations and return a comprehensive report."""
        syntax = cls.validate_syntax(code)
        if not syntax["valid"]:
            return {
                "passed": False,
                "stage": "syntax",
                "details": syntax,
            }

        safety = cls.validate_safety(code)
        structure = cls.validate_structure(code)
        sandbox = cls.run_sandbox_test(code)

        passed = (
            syntax["valid"]
            and safety["safe"]
            and structure["has_functions"]
            and sandbox["loadable"]
        )
        return {
            "passed": passed,
            "syntax": syntax,
            "safety": safety,
            "structure": structure,
            "sandbox": sandbox,
        }


# ---------------------------------------------------------------------------
# SKILL.md Generator — creates portable skill definition files
# ---------------------------------------------------------------------------


class SkillMDGenerator:
    """Generate SKILL.md files with YAML frontmatter."""

    @staticmethod
    def generate(
        name: str,
        description: str,
        body: str,
        metadata: Optional[Dict] = None,
    ) -> str:
        """Generate a SKILL.md file with YAML frontmatter."""
        lines = ["---"]
        lines.append(f"name: {name}")
        lines.append(f"description: {description}")
        if metadata:
            lines.append(f"metadata: {json.dumps(metadata)}")
        lines.append("---")
        lines.append("")
        lines.append(body)
        return "\n".join(lines)

    @staticmethod
    def create_skill_directory(
        skill_name: str,
        base_dir: Path,
        skill_md_content: str,
        code: str,
        references: Optional[Dict[str, str]] = None,
    ) -> Path:
        """Create a full skill directory structure."""
        # Create the skill directory
        slug = skill_name.lower().replace(" ", "-").replace("_", "-")
        skill_dir = base_dir / slug
        skill_dir.mkdir(parents=True, exist_ok=True)

        # Write SKILL.md
        (skill_dir / "SKILL.md").write_text(skill_md_content)

        # Write scripts/
        scripts_dir = skill_dir / "scripts"
        scripts_dir.mkdir(exist_ok=True)
        (scripts_dir / f"{slug}.py").write_text(code)

        # Write references/ if provided
        if references:
            refs_dir = skill_dir / "references"
            refs_dir.mkdir(exist_ok=True)
            for ref_name, ref_content in references.items():
                (refs_dir / ref_name).write_text(ref_content)

        logger.info(f"📁 Created skill directory: {skill_dir}")
        return skill_dir


# ---------------------------------------------------------------------------
# Skill Factory — the main engine
# ---------------------------------------------------------------------------


class SkillFactory:
    """
    Autonomous Skill Creation Engine.

    The SkillFactory enables the SableCore agent to:
    1. Analyze a natural-language description of what a skill should do
    2. Choose the best template or generate code from scratch
    3. Validate the generated code (syntax, safety, sandbox)
    4. Create SKILL.md skill directories
    5. Register the skill in the SkillsHub marketplace
    6. Save to the installed skills directory

    Usage:
        factory = SkillFactory(config)
        result = await factory.create_skill(
            name="Weather Alerts",
            description="Monitor weather and send alerts when conditions change",
            category="utility",
            triggers=["weather alert", "storm warning"],
            examples=["Alert me if it rains in NYC"],
        )
    """

    def __init__(self, config=None):
        self.config = config
        self.base_dir = Path(__file__).parent.parent.parent
        self.skills_dir = self.base_dir / "opensable" / "skills"
        self.installed_dir = self.skills_dir / "installed"
        self.created_dir = self.skills_dir / "created"
        self.templates_dir = self.skills_dir / "templates"

        # Ensure directories exist
        self.installed_dir.mkdir(parents=True, exist_ok=True)
        self.created_dir.mkdir(parents=True, exist_ok=True)
        self.templates_dir.mkdir(parents=True, exist_ok=True)

        self.validator = SkillValidator()
        self.md_generator = SkillMDGenerator()

        # Track created skills
        self._creation_log: List[Dict] = []

    # -------------------------------------------------------------------
    # Blueprint creation — analyze what the user wants
    # -------------------------------------------------------------------

    def create_blueprint(
        self,
        name: str,
        description: str,
        category: str = "general",
        triggers: Optional[List[str]] = None,
        examples: Optional[List[str]] = None,
        inputs: Optional[List[Dict[str, str]]] = None,
        outputs: Optional[List[Dict[str, str]]] = None,
        dependencies: Optional[List[str]] = None,
    ) -> SkillBlueprint:
        """Create a skill blueprint from user description."""
        # Auto-detect properties from description
        desc_lower = description.lower()

        async_required = any(kw in desc_lower for kw in ["api", "http", "fetch", "download", "web"])
        needs_network = any(
            kw in desc_lower for kw in ["api", "web", "url", "http", "download", "fetch"]
        )
        needs_filesystem = any(
            kw in desc_lower for kw in ["file", "read", "write", "save", "directory", "path"]
        )

        # Determine complexity
        complexity = "simple"
        if any(kw in desc_lower for kw in ["multiple", "complex", "pipeline", "multi-step"]):
            complexity = "complex"
        elif any(kw in desc_lower for kw in ["process", "transform", "analyze", "parse"]):
            complexity = "medium"

        # Auto-generate triggers from name and description
        if not triggers:
            triggers = self._auto_triggers(name, description)

        # Auto-generate examples
        if not examples:
            examples = self._auto_examples(name, description)

        return SkillBlueprint(
            name=name,
            description=description,
            category=category,
            triggers=triggers,
            examples=examples,
            inputs=inputs or [],
            outputs=outputs or [],
            dependencies=dependencies or [],
            complexity=complexity,
            async_required=async_required,
            needs_network=needs_network,
            needs_filesystem=needs_filesystem,
        )

    # -------------------------------------------------------------------
    # Template selection — choose the best starting template
    # -------------------------------------------------------------------

    def select_template(self, blueprint: SkillBlueprint) -> str:
        """Select the best code template for the blueprint."""
        desc_lower = blueprint.description.lower()

        if blueprint.needs_network and "api" in desc_lower:
            return "api_fetcher"
        elif blueprint.needs_filesystem:
            return "file_handler"
        elif any(kw in desc_lower for kw in ["track", "store", "manage", "list", "add"]):
            return "storage_manager"
        elif any(kw in desc_lower for kw in ["command", "cli", "run", "execute"]):
            return "cli_wrapper"
        elif any(kw in desc_lower for kw in ["process", "transform", "parse", "convert"]):
            return "data_processor"
        else:
            return "generic"

    # -------------------------------------------------------------------
    # Code generation — generate the actual skill code
    # -------------------------------------------------------------------

    def generate_code(self, blueprint: SkillBlueprint, language: str = "python") -> str:
        """
        Generate code for the skill based on the blueprint.

        Args:
            blueprint: Skill blueprint describing what to build.
            language: Target language — "python" (default), "javascript", or "rust".

        Returns:
            Generated source code string.
        """
        if language == "javascript":
            return self._generate_js(blueprint)
        elif language == "rust":
            return self._generate_rust(blueprint)
        else:
            return self._generate_python(blueprint)

    def _generate_python(self, blueprint: SkillBlueprint) -> str:
        """Generate Python code (original path)."""
        func_name = blueprint.name.lower().replace(" ", "_").replace("-", "_")
        template_key = self.select_template(blueprint)

        # Build parameter list
        if blueprint.inputs:
            params = ", ".join(
                f"{i['name']}: {i.get('type', 'str')} = {i.get('default', 'None')}"
                for i in blueprint.inputs
            )
            first_param = blueprint.inputs[0]["name"]
        else:
            params = "input_data: str"
            first_param = "input_data"

        # Generate from template with appropriate fills
        if template_key == "api_fetcher":
            code = self._gen_api_fetcher(blueprint, func_name, params, first_param)
        elif template_key == "storage_manager":
            code = self._gen_storage_manager(blueprint, func_name, params)
        elif template_key == "file_handler":
            code = self._gen_file_handler(blueprint, func_name, params, first_param)
        elif template_key == "cli_wrapper":
            code = self._gen_cli_wrapper(blueprint, func_name, params)
        elif template_key == "data_processor":
            code = self._gen_data_processor(blueprint, func_name, params)
        else:
            code = self._gen_generic(blueprint, func_name, params)

        return code

    def _gen_api_fetcher(self, bp, func_name, params, first_param):
        return f'''"""
{bp.name} — {bp.description}
Auto-generated by SableCore SkillFactory
"""
import aiohttp
import urllib.parse

async def {func_name}({params}) -> dict:
    """{bp.description}"""
    try:
        query = urllib.parse.quote(str({first_param}))
        url = f"https://api.example.com/v1/{{query}}"
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return {{"success": True, "data": data}}
                return {{"success": False, "status": resp.status, "reason": resp.reason}}
    except aiohttp.ClientError as e:
        return {{"error": f"Network error: {{e}}"}}
    except Exception as e:
        return {{"error": str(e)}}
'''

    def _gen_storage_manager(self, bp, func_name, params):
        item_name = func_name.replace("_manager", "").replace("_tracker", "")
        return f'''"""
{bp.name} — {bp.description}
Auto-generated by SableCore SkillFactory
"""
import json
from pathlib import Path
from datetime import datetime

STORAGE_FILE = Path.home() / '.sablecore' / '{item_name}s.json'

def _load():
    if STORAGE_FILE.exists():
        return json.loads(STORAGE_FILE.read_text())
    return []

def _save(data):
    STORAGE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STORAGE_FILE.write_text(json.dumps(data, indent=2))

def add_{item_name}({params}) -> dict:
    """{bp.description}"""
    items = _load()
    item = {{
        'id': len(items) + 1,
        'data': input_data,
        'status': 'active',
        'created': datetime.now().isoformat()
    }}
    items.append(item)
    _save(items)
    return item

def list_{item_name}s() -> list:
    """List all {item_name}s."""
    return _load()

def remove_{item_name}(item_id: int) -> dict:
    """Remove a {item_name} by ID."""
    items = _load()
    original = len(items)
    items = [i for i in items if i['id'] != item_id]
    _save(items)
    return {{"removed": item_id, "success": len(items) < original}}
'''

    def _gen_file_handler(self, bp, func_name, params, first_param):
        return f'''"""
{bp.name} — {bp.description}
Auto-generated by SableCore SkillFactory
"""
from pathlib import Path

def {func_name}({params}) -> dict:
    """{bp.description}"""
    try:
        path = Path({first_param})
        if not path.exists():
            return {{"error": f"Not found: {{{first_param}}}"}}
        if path.is_dir():
            files = [{{"name": f.name, "size": f.stat().st_size}} for f in path.iterdir()]
            return {{"path": str(path), "type": "directory", "files": files, "count": len(files)}}
        content = path.read_text(errors='replace')
        return {{"path": str(path), "type": "file", "content": content, "size": path.stat().st_size}}
    except PermissionError:
        return {{"error": f"Permission denied: {{{first_param}}}"}}
    except Exception as e:
        return {{"error": str(e)}}
'''

    def _gen_cli_wrapper(self, bp, func_name, params):
        return f'''"""
{bp.name} — {bp.description}
Auto-generated by SableCore SkillFactory
"""
import subprocess

def {func_name}({params}) -> dict:
    """{bp.description}"""
    try:
        result = subprocess.run(
            str(input_data), shell=True,
            capture_output=True, text=True, timeout=30
        )
        return {{
            "stdout": result.stdout.strip(),
            "stderr": result.stderr.strip(),
            "exit_code": result.returncode,
            "success": result.returncode == 0
        }}
    except subprocess.TimeoutExpired:
        return {{"error": "Command timed out (30s limit)"}}
    except Exception as e:
        return {{"error": str(e)}}
'''

    def _gen_data_processor(self, bp, func_name, params):
        return f'''"""
{bp.name} — {bp.description}
Auto-generated by SableCore SkillFactory
"""
import json
import re

def {func_name}({params}) -> dict:
    """{bp.description}"""
    try:
        data = input_data
        # Try to parse as JSON if string
        if isinstance(data, str):
            try:
                data = json.loads(data)
            except (json.JSONDecodeError, TypeError):
                pass
        # Process the data
        if isinstance(data, dict):
            result = {{k: v for k, v in data.items()}}
        elif isinstance(data, list):
            result = {{"items": data, "count": len(data)}}
        elif isinstance(data, str):
            result = {{"text": data, "length": len(data), "words": len(data.split())}}
        else:
            result = {{"value": str(data)}}
        return {{"success": True, "result": result}}
    except Exception as e:
        return {{"error": str(e)}}
'''

    def _gen_generic(self, bp, func_name, params):
        return f'''"""
{bp.name} — {bp.description}
Auto-generated by SableCore SkillFactory
"""

def {func_name}({params}) -> dict:
    """{bp.description}"""
    try:
        result = str(input_data)
        return {{"success": True, "result": result, "skill": "{bp.name}"}}
    except Exception as e:
        return {{"error": str(e)}}
'''

    # -------------------------------------------------------------------
    # Cross-platform code generation: JavaScript
    # -------------------------------------------------------------------

    def _generate_js(self, blueprint: SkillBlueprint) -> str:
        """Generate JavaScript (Node.js) code for the skill."""
        func_name = blueprint.name.lower().replace(" ", "_").replace("-", "_")
        template_key = self.select_template(blueprint)

        if blueprint.inputs:
            params = ", ".join(i["name"] for i in blueprint.inputs)
            first_param = blueprint.inputs[0]["name"]
            param_type = blueprint.inputs[0].get("type", "string")
        else:
            params = "inputData"
            first_param = "inputData"
            param_type = "string"

        templates = JS_TEMPLATES
        tmpl = templates.get(template_key, templates["generic"])

        try:
            return tmpl.format(
                name=blueprint.name,
                description=blueprint.description,
                func_name=func_name,
                params=params,
                first_param=first_param,
                param_type=param_type,
                api_url="https://api.example.com/v1",
                processing_logic=f"JSON.parse(JSON.stringify({first_param}))",
                implementation=f"const result = {first_param};",
            )
        except KeyError:
            # Fallback generic
            return templates["generic"].format(
                name=blueprint.name,
                description=blueprint.description,
                func_name=func_name,
                params=params,
                first_param=first_param,
                implementation=f"const result = {first_param};",
            )

    # -------------------------------------------------------------------
    # Cross-platform code generation: Rust
    # -------------------------------------------------------------------

    def _generate_rust(self, blueprint: SkillBlueprint) -> str:
        """Generate Rust code for the skill."""
        func_name = blueprint.name.lower().replace(" ", "_").replace("-", "_")
        template_key = self.select_template(blueprint)

        if blueprint.inputs:
            # Rust-style parameters
            params = ", ".join(
                f"{i['name']}: &{self._rust_type(i.get('type', 'str'))}" for i in blueprint.inputs
            )
            first_param = blueprint.inputs[0]["name"]
        else:
            params = "input: &str"
            first_param = "input"

        templates = RUST_TEMPLATES
        tmpl = templates.get(template_key, templates["generic"])

        try:
            return tmpl.format(
                name=blueprint.name,
                description=blueprint.description,
                func_name=func_name,
                params=params,
                first_param=first_param,
                api_url="https://api.example.com/v1",
                processing_logic=f"serde_json::to_value({first_param})?",
                implementation=f"let result = serde_json::to_value({first_param})?;",
            )
        except KeyError:
            return templates["generic"].format(
                name=blueprint.name,
                description=blueprint.description,
                func_name=func_name,
                params=params,
                first_param=first_param,
                implementation=f"let result = serde_json::to_value({first_param})?;",
            )

    @staticmethod
    def _rust_type(python_type: str) -> str:
        """Map Python type hints to Rust types."""
        mapping = {
            "str": "str",
            "int": "i64",
            "float": "f64",
            "bool": "bool",
            "list": "Vec<serde_json::Value>",
            "dict": "serde_json::Value",
            "Any": "serde_json::Value",
        }
        return mapping.get(python_type, "str")

    # -------------------------------------------------------------------
    # Full skill creation pipeline
    # -------------------------------------------------------------------

    async def create_skill(
        self,
        name: str,
        description: str,
        category: str = "general",
        triggers: Optional[List[str]] = None,
        examples: Optional[List[str]] = None,
        inputs: Optional[List[Dict[str, str]]] = None,
        custom_code: Optional[str] = None,
        language: str = "python",
    ) -> Dict[str, Any]:
        """
        Full skill creation pipeline:
        1. Create blueprint
        2. Generate code (or use custom) in target language (python/javascript/rust)
        3. Validate (syntax, safety, sandbox)
        4. Create SKILL.md directory
        5. Register in JSON catalog
        6. Save to installed/

        Returns a detailed result dict.
        """
        logger.info(f"🏭 SkillFactory: Creating skill '{name}' [{language}]...")

        # Step 1: Blueprint
        blueprint = self.create_blueprint(
            name=name,
            description=description,
            category=category,
            triggers=triggers,
            examples=examples,
            inputs=inputs,
        )
        logger.info(
            f"  📋 Blueprint: {blueprint.complexity} complexity, template: {self.select_template(blueprint)}"
        )

        # Step 2: Generate code
        if custom_code:
            code = custom_code
            logger.info("  📝 Using custom code provided by user")
        else:
            code = self.generate_code(blueprint, language=language)
            logger.info(f"  📝 Generated {len(code)} chars of {language} code")

        # Step 3: Validate
        validation = self.validator.full_validate(code)
        if not validation["passed"]:
            logger.warning(f"  ⚠️  Validation failed: {validation}")
            return {
                "success": False,
                "stage": "validation",
                "validation": validation,
                "code": code,
                "blueprint": asdict(blueprint),
            }
        logger.info("  ✅ Validation passed (syntax, safety, sandbox)")

        # Step 4: Create SKILL.md directory
        skill_slug = name.lower().replace(" ", "-").replace("_", "-")
        skill_body = f"# {name}\n\n{description}\n\n"
        skill_body += "## Usage\n\n"
        for ex in blueprint.examples:
            skill_body += f"- {ex}\n"
        skill_body += f"\n## Implementation\n\nSee `scripts/{skill_slug}.py` for the skill code.\n"
        if blueprint.triggers:
            skill_body += "\n## Triggers\n\n"
            for t in blueprint.triggers:
                skill_body += f"- `{t}`\n"

        skill_md = self.md_generator.generate(
            name=name,
            description=description,
            body=skill_body,
            metadata={
                "skill_meta": {
                    "emoji": self._category_emoji(category),
                    "author": "SableCore SkillFactory",
                    "version": "1.0.0",
                }
            },
        )

        skill_dir = self.md_generator.create_skill_directory(
            skill_name=name,
            base_dir=self.created_dir,
            skill_md_content=skill_md,
            code=code,
        )

        # Step 5: Register in the skills catalog
        self._register_in_catalog(blueprint, code)

        # Step 6: Save to installed/
        installed_file = self.installed_dir / f"{skill_slug}.py"
        installed_file.write_text(code)
        logger.info(f"  💾 Installed to: {installed_file}")

        # Log creation
        creation_record = {
            "name": name,
            "slug": skill_slug,
            "category": category,
            "template": self.select_template(blueprint),
            "code_size": len(code),
            "created_at": datetime.now().isoformat(),
            "directory": str(skill_dir),
            "installed": str(installed_file),
            "validation": validation,
        }
        self._creation_log.append(creation_record)

        logger.info(f"  🎉 Skill '{name}' created successfully!")

        return {
            "success": True,
            "name": name,
            "slug": skill_slug,
            "directory": str(skill_dir),
            "installed_file": str(installed_file),
            "code": code,
            "blueprint": asdict(blueprint),
            "validation": validation,
        }

    def _register_in_catalog(self, blueprint: SkillBlueprint, code: str):
        """Register the new skill in the community skills catalog."""
        catalog_path = self.skills_dir / "community" / "skills_catalog.json"
        try:
            if catalog_path.exists():
                catalog = json.loads(catalog_path.read_text())
            else:
                catalog = {"registry": "SableCore", "version": "2.0.0", "skills": []}

            slug = blueprint.name.lower().replace(" ", "_").replace("-", "_")

            # Remove existing entry if any
            catalog["skills"] = [s for s in catalog["skills"] if s.get("id") != slug]

            # Add new entry
            catalog["skills"].append(
                {
                    "id": slug,
                    "name": blueprint.name,
                    "description": blueprint.description,
                    "category": blueprint.category,
                    "function": code,
                    "triggers": blueprint.triggers,
                    "examples": blueprint.examples,
                    "author": "SableCore SkillFactory",
                    "version": "1.0.0",
                    "rating": 0.0,
                    "downloads": 0,
                    "tags": [blueprint.category] + blueprint.triggers[:3],
                    "auto_generated": True,
                }
            )

            catalog["updated"] = datetime.now().isoformat()
            catalog_path.write_text(json.dumps(catalog, indent=2))
            logger.info(f"  📦 Registered in catalog: {slug}")
        except Exception as e:
            logger.error(f"  ❌ Failed to register in catalog: {e}")

    # -------------------------------------------------------------------
    # Batch creation — create multiple skills at once
    # -------------------------------------------------------------------

    async def create_skills_batch(self, skill_specs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Create multiple skills from a list of specifications."""
        results = []
        for spec in skill_specs:
            result = await self.create_skill(**spec)
            results.append(result)
        return results

    # -------------------------------------------------------------------
    # Skill improvement — iterate on existing skills
    # -------------------------------------------------------------------

    def improve_skill(self, skill_slug: str, feedback: str) -> Dict[str, Any]:
        """
        Improve an existing skill based on user feedback.
        Returns suggestions for how to improve the code.
        """
        installed_path = self.installed_dir / f"{skill_slug}.py"
        if not installed_path.exists():
            return {"error": f"Skill not found: {skill_slug}"}

        current_code = installed_path.read_text()

        # Analyze the code
        structure = self.validator.validate_structure(current_code)

        suggestions = []
        if not structure.get("has_docstrings"):
            suggestions.append("Add docstrings to all functions")
        if "error" not in current_code.lower():
            suggestions.append("Add error handling (try/except blocks)")
        if "async" in feedback.lower() and "async " not in current_code:
            suggestions.append("Convert to async for better performance")
        if "type" in feedback.lower():
            suggestions.append("Add type hints to function parameters")

        return {
            "skill": skill_slug,
            "current_functions": structure.get("functions", []),
            "feedback": feedback,
            "suggestions": suggestions,
            "code_size": len(current_code),
        }

    # -------------------------------------------------------------------
    # Introspection — what skills have been created?
    # -------------------------------------------------------------------

    def get_creation_log(self) -> List[Dict]:
        """Get the log of all skills created in this session."""
        return self._creation_log

    def get_created_skills(self) -> List[Dict[str, str]]:
        """List all skills in the created/ directory."""
        skills = []
        if self.created_dir.exists():
            for skill_dir in self.created_dir.iterdir():
                if skill_dir.is_dir() and (skill_dir / "SKILL.md").exists():
                    md = (skill_dir / "SKILL.md").read_text()
                    # Extract name from frontmatter
                    match = re.search(r"name:\s*(.+)", md)
                    name = match.group(1).strip() if match else skill_dir.name
                    match_desc = re.search(r"description:\s*(.+)", md)
                    desc = match_desc.group(1).strip() if match_desc else ""
                    skills.append(
                        {
                            "name": name,
                            "description": desc,
                            "directory": str(skill_dir),
                            "slug": skill_dir.name,
                        }
                    )
        return skills

    def get_installed_skills(self) -> List[str]:
        """List all installed skill slugs."""
        return [f.stem for f in self.installed_dir.glob("*.py") if f.stem != "__init__"]

    # -------------------------------------------------------------------
    # Helper methods
    # -------------------------------------------------------------------

    def _auto_triggers(self, name: str, description: str) -> List[str]:
        """Auto-generate trigger words from name and description."""
        triggers = []
        # Add name words
        for word in name.lower().split():
            if len(word) > 2:
                triggers.append(word)
        # Add key nouns/verbs from description
        keywords = re.findall(r"\b[a-z]{4,}\b", description.lower())
        stop = {
            "with",
            "from",
            "this",
            "that",
            "have",
            "will",
            "been",
            "were",
            "does",
            "using",
            "also",
            "very",
            "just",
            "some",
            "more",
            "than",
            "other",
            "into",
            "about",
            "between",
            "through",
            "after",
            "before",
        }
        for kw in keywords:
            if kw not in stop and kw not in triggers:
                triggers.append(kw)
                if len(triggers) >= 6:
                    break
        return triggers

    def _auto_examples(self, name: str, description: str) -> List[str]:
        """Auto-generate usage examples."""
        return [
            f"Use {name} to help me",
            f"{name}: {description.split('.')[0]}",
        ]

    def _category_emoji(self, category: str) -> str:
        """Map category to an emoji."""
        emoji_map = {
            "search": "🔍",
            "utility": "🔧",
            "system": "🖥️",
            "productivity": "📋",
            "development": "💻",
            "nlp": "📝",
            "security": "🔒",
            "data": "📊",
            "web": "🌐",
            "finance": "💰",
            "social": "👥",
            "ai": "🤖",
            "communication": "💬",
            "media": "🎬",
        }
        return emoji_map.get(category, "⚡")
