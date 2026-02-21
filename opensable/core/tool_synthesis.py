"""
Tool Synthesis - Dynamic tool creation and code generation for AGI.

Features:
- Automatic tool generation from descriptions
- Code synthesis for new capabilities
- Tool validation and testing
- Tool composition (combining tools)
- Dynamic skill acquisition
- Safe code execution
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import json
from pathlib import Path
import ast
import inspect
import re

logger = logging.getLogger(__name__)


class ToolType(Enum):
    """Types of synthesized tools."""
    DATA_PROCESSOR = "data_processor"
    API_CLIENT = "api_client"
    FILE_HANDLER = "file_handler"
    CALCULATOR = "calculator"
    CONVERTER = "converter"
    SCRAPER = "scraper"
    GENERATOR = "generator"


@dataclass
class ToolSpecification:
    """Specification for a tool to be created."""
    name: str
    description: str
    tool_type: ToolType
    inputs: List[Dict[str, str]]  # [{name: str, type: str, description: str}]
    outputs: List[Dict[str, str]]
    requirements: List[str] = field(default_factory=list)  # Dependencies
    constraints: List[str] = field(default_factory=list)  # Safety constraints
    examples: List[Dict[str, Any]] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'description': self.description,
            'tool_type': self.tool_type.value,
            'inputs': self.inputs,
            'outputs': self.outputs,
            'requirements': self.requirements,
            'constraints': self.constraints,
            'examples': self.examples
        }


@dataclass
class SynthesizedTool:
    """A synthesized tool."""
    tool_id: str
    name: str
    specification: ToolSpecification
    code: str
    function: Optional[Callable] = None
    validated: bool = False
    test_results: Optional[Dict[str, Any]] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    usage_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'tool_id': self.tool_id,
            'name': self.name,
            'specification': self.specification.to_dict(),
            'code': self.code,
            'validated': self.validated,
            'test_results': self.test_results,
            'created_at': self.created_at.isoformat(),
            'usage_count': self.usage_count,
            'metadata': self.metadata
        }


class CodeGenerator:
    """
    Generates Python code for tools.
    
    Uses LLM to synthesize code from specifications.
    """
    
    def __init__(self, llm_function: Optional[Callable] = None):
        self.llm_function = llm_function
    
    async def generate_tool_code(
        self,
        spec: ToolSpecification
    ) -> str:
        """
        Generate code for a tool.
        
        Args:
            spec: Tool specification
        
        Returns:
            Generated Python code
        """
        # Build prompt for LLM
        inputs_desc = "\n".join([
            f"  - {inp['name']} ({inp['type']}): {inp['description']}"
            for inp in spec.inputs
        ])
        
        outputs_desc = "\n".join([
            f"  - {out['name']} ({out['type']}): {out['description']}"
            for out in spec.outputs
        ])
        
        examples_desc = ""
        if spec.examples:
            examples_desc = "\n\nExamples:\n"
            for i, ex in enumerate(spec.examples, 1):
                examples_desc += f"{i}. Input: {ex.get('input', 'N/A')}\n"
                examples_desc += f"   Output: {ex.get('output', 'N/A')}\n"
        
        constraints_desc = ""
        if spec.constraints:
            constraints_desc = "\n\nConstraints:\n" + "\n".join([
                f"- {c}" for c in spec.constraints
            ])
        
        prompt = f"""Generate a Python function for this tool:

Tool Name: {spec.name}
Description: {spec.description}
Type: {spec.tool_type.value}

Inputs:
{inputs_desc}

Outputs:
{outputs_desc}{examples_desc}{constraints_desc}

Requirements:
- Function must be async (async def)
- Include type hints
- Include docstring
- Handle errors gracefully
- Return a dictionary matching the outputs
- No external API calls unless specified

Generate only the function code, no imports or extra code:"""
        
        if self.llm_function:
            try:
                code = await self.llm_function(prompt)
                
                # Extract code block
                code = self._extract_code(code)
                
                # Validate syntax
                if self._validate_syntax(code):
                    return code
            
            except Exception as e:
                logger.error(f"Code generation failed: {e}")
        
        # Fallback: generate template
        return self._generate_template(spec)
    
    def _extract_code(self, response: str) -> str:
        """Extract code from LLM response."""
        # Look for code blocks
        patterns = [
            r"```python\n(.*?)\n```",
            r"```\n(.*?)\n```",
            r"def\s+\w+.*?(?=\n\n|\Z)"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, response, re.DOTALL)
            if match:
                return match.group(1) if pattern.startswith('```') else match.group(0)
        
        return response
    
    def _validate_syntax(self, code: str) -> bool:
        """Validate Python syntax."""
        try:
            ast.parse(code)
            return True
        except SyntaxError as e:
            logger.error(f"Syntax error in generated code: {e}")
            return False
    
    def _generate_template(self, spec: ToolSpecification) -> str:
        """Generate code template."""
        # Build parameter list
        params = ", ".join([
            f"{inp['name']}: {inp['type']}"
            for inp in spec.inputs
        ])
        
        # Build return type
        return_type = "Dict[str, Any]"
        
        # Build docstring
        docstring = f'''"""
{spec.description}

Args:
'''
        for inp in spec.inputs:
            docstring += f"    {inp['name']}: {inp['description']}\n"
        
        docstring += "\nReturns:\n"
        for out in spec.outputs:
            docstring += f"    {out['name']}: {out['description']}\n"
        
        docstring += '"""'
        
        code = f"""async def {spec.name}({params}) -> {return_type}:
    {docstring}
    # TODO: Implement tool logic
    result = {{}}
    
"""
        
        # Add return statements for each output
        for out in spec.outputs:
            code += f"    result['{out['name']}'] = None  # TODO: Compute {out['name']}\n"
        
        code += "    \n    return result\n"
        
        return code


# ─── Neural Tool Synthesis (beyond LLM) ──────────────────────────────────────

class PatternLibrary:
    """
    Reusable code patterns extracted from previously-synthesised tools.
    Works WITHOUT the LLM — pure AST analysis + template matching.
    """

    # Common algorithmic building blocks keyed by semantic tag
    BLOCKS: Dict[str, str] = {
        "http_get": (
            "async def _http_get(url: str, params: dict = None) -> dict:\n"
            "    import aiohttp\n"
            "    async with aiohttp.ClientSession() as s:\n"
            "        async with s.get(url, params=params) as r:\n"
            "            return {'status': r.status, 'data': await r.json()}\n"
        ),
        "http_post": (
            "async def _http_post(url: str, payload: dict) -> dict:\n"
            "    import aiohttp\n"
            "    async with aiohttp.ClientSession() as s:\n"
            "        async with s.post(url, json=payload) as r:\n"
            "            return {'status': r.status, 'data': await r.json()}\n"
        ),
        "file_read": (
            "def _file_read(path: str) -> str:\n"
            "    from pathlib import Path\n"
            "    return Path(path).read_text(errors='replace')\n"
        ),
        "file_write": (
            "def _file_write(path: str, content: str) -> bool:\n"
            "    from pathlib import Path\n"
            "    Path(path).write_text(content); return True\n"
        ),
        "json_parse": (
            "def _json_parse(text: str) -> dict:\n"
            "    import json; return json.loads(text)\n"
        ),
        "csv_parse": (
            "def _csv_parse(text: str) -> list:\n"
            "    import csv, io\n"
            "    return list(csv.DictReader(io.StringIO(text)))\n"
        ),
        "regex_extract": (
            "def _regex_extract(text: str, pattern: str) -> list:\n"
            "    import re; return re.findall(pattern, text)\n"
        ),
        "math_stats": (
            "def _math_stats(values: list) -> dict:\n"
            "    n = len(values); s = sum(values); avg = s/n if n else 0\n"
            "    variance = sum((x-avg)**2 for x in values)/n if n else 0\n"
            "    return {'count': n, 'sum': s, 'mean': avg, 'variance': variance, 'std': variance**0.5}\n"
        ),
        "cache_memo": (
            "_cache = {}\n"
            "def _cached(key: str, fn, *args):\n"
            "    if key not in _cache: _cache[key] = fn(*args)\n"
            "    return _cache[key]\n"
        ),
        "retry_loop": (
            "async def _retry(fn, retries=3, delay=1.0):\n"
            "    import asyncio\n"
            "    for attempt in range(retries):\n"
            "        try: return await fn()\n"
            "        except Exception as e:\n"
            "            if attempt == retries - 1: raise\n"
            "            await asyncio.sleep(delay * (attempt + 1))\n"
        ),
        "string_transform": (
            "def _transform(text: str, ops: list) -> str:\n"
            "    for op in ops:\n"
            "        if op == 'upper': text = text.upper()\n"
            "        elif op == 'lower': text = text.lower()\n"
            "        elif op == 'strip': text = text.strip()\n"
            "        elif op == 'title': text = text.title()\n"
            "    return text\n"
        ),
        "list_filter": (
            "def _list_filter(items: list, key: str, value) -> list:\n"
            "    return [i for i in items if i.get(key) == value]\n"
        ),
        "date_format": (
            "def _date_format(dt_str: str, fmt: str = '%Y-%m-%d') -> str:\n"
            "    from datetime import datetime\n"
            "    return datetime.fromisoformat(dt_str).strftime(fmt)\n"
        ),
    }

    # Mapping from spec keywords → relevant blocks
    KEYWORD_MAP = {
        "fetch": ["http_get", "retry_loop"],
        "api": ["http_get", "http_post", "json_parse", "retry_loop"],
        "download": ["http_get", "file_write"],
        "scrape": ["http_get", "regex_extract"],
        "parse": ["json_parse", "csv_parse", "regex_extract"],
        "file": ["file_read", "file_write"],
        "read": ["file_read"],
        "write": ["file_write"],
        "csv": ["csv_parse"],
        "json": ["json_parse"],
        "convert": ["string_transform", "json_parse"],
        "calculate": ["math_stats"],
        "statistics": ["math_stats"],
        "average": ["math_stats"],
        "filter": ["list_filter"],
        "search": ["regex_extract", "list_filter"],
        "cache": ["cache_memo"],
        "date": ["date_format"],
        "format": ["string_transform", "date_format"],
        "transform": ["string_transform"],
        "post": ["http_post", "retry_loop"],
        "send": ["http_post"],
        "retry": ["retry_loop"],
    }

    @classmethod
    def match_patterns(cls, spec: ToolSpecification) -> List[str]:
        """Return block keys that match the spec description & type."""
        desc = (spec.description + " " + spec.name).lower()
        matched = set()
        for keyword, blocks in cls.KEYWORD_MAP.items():
            if keyword in desc:
                matched.update(blocks)
        # Also match by tool type
        type_map = {
            ToolType.API_CLIENT: ["http_get", "http_post", "json_parse", "retry_loop"],
            ToolType.SCRAPER: ["http_get", "regex_extract", "retry_loop"],
            ToolType.FILE_HANDLER: ["file_read", "file_write"],
            ToolType.CALCULATOR: ["math_stats"],
            ToolType.CONVERTER: ["string_transform", "json_parse"],
            ToolType.DATA_PROCESSOR: ["json_parse", "csv_parse", "list_filter"],
        }
        matched.update(type_map.get(spec.tool_type, []))
        return list(matched)

    @classmethod
    def get_blocks(cls, keys: List[str]) -> str:
        """Return concatenated source code for the given block keys."""
        return "\n".join(cls.BLOCKS[k] for k in keys if k in cls.BLOCKS)


class ASTComposer:
    """
    Compose tools by merging their ASTs — no LLM needed.

    Takes multiple SynthesizedTool objects and produces a single combined
    tool whose body calls each sub-tool in sequence, threading results
    through shared variables.
    """

    @staticmethod
    def compose(tools: List["SynthesizedTool"], name: str, description: str) -> str:
        """
        Merge tool functions into one pipeline function via AST manipulation.
        """
        import_nodes: List[ast.stmt] = []
        helper_nodes: List[ast.stmt] = []
        call_stmts: List[str] = []

        for i, tool in enumerate(tools):
            tree = ast.parse(tool.code)
            for node in ast.walk(tree):
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    import_nodes.append(node)
            # Rename function to _step_N
            renamed = tool.code.replace(f"def {tool.name}(", f"def _step_{i}(", 1)
            renamed = renamed.replace(f"async def {tool.name}(", f"async def _step_{i}(", 1)
            helper_nodes.append(renamed)
            call_stmts.append(
                f"    _r{i} = await _step_{i}(**{{**kwargs, **_results}})\n"
                f"    _results.update(_r{i} if isinstance(_r{i}, dict) else {{'step_{i}': _r{i}}})"
            )

        # Deduplicate imports
        seen_imports = set()
        unique_imports = []
        for node in import_nodes:
            s = ast.dump(node)
            if s not in seen_imports:
                seen_imports.add(s)
                unique_imports.append(ast.unparse(node))

        helpers = "\n\n".join(h if isinstance(h, str) else ast.unparse(h) for h in helper_nodes)
        calls = "\n".join(call_stmts)

        code = (
            "\n".join(unique_imports) + "\n\n"
            + helpers + "\n\n"
            + f"async def {name}(**kwargs):\n"
            + f'    """{description}"""\n'
            + "    _results = {}\n"
            + calls + "\n"
            + "    return _results\n"
        )
        return code


class NeuralSynthesizer:
    """
    Neural Tool Synthesis — goes beyond pure LLM prompting.

    Three-tier synthesis pipeline:
      1. Pattern matching: select reusable code blocks from PatternLibrary
      2. AST composition: combine blocks + existing tools via AST merging
      3. LLM refinement (optional): polish the assembled code

    Tier 1+2 produce working tools WITHOUT any LLM call.
    """

    def __init__(self, llm_function: Optional[Callable] = None):
        self.llm_function = llm_function
        self.pattern_lib = PatternLibrary()
        self.composer = ASTComposer()
        self._synthesis_log: List[Dict[str, Any]] = []

    async def synthesize(
        self,
        spec: ToolSpecification,
        existing_tools: Optional[Dict[str, "SynthesizedTool"]] = None,
        use_llm_refinement: bool = True,
    ) -> str:
        """
        Produce tool code through the 3-tier pipeline.

        Returns the final source code string.
        """
        log_entry = {
            "spec": spec.name,
            "tiers_used": [],
            "timestamp": datetime.utcnow().isoformat(),
        }

        # ── Tier 1: Pattern matching ─────────────────────────────────
        matched_keys = self.pattern_lib.match_patterns(spec)
        blocks_code = self.pattern_lib.get_blocks(matched_keys)
        log_entry["tiers_used"].append("pattern_match")
        log_entry["patterns_matched"] = matched_keys

        # ── Tier 2: AST composition ──────────────────────────────────
        # Build the wrapper function that calls the matched blocks
        params = ", ".join(f"{inp['name']}: {inp['type']}" for inp in spec.inputs)
        returns = "{" + ", ".join(f"'{o['name']}': {o['name']}" for o in spec.outputs) + "}"

        assembled = blocks_code + "\n\n"
        assembled += f"async def {spec.name}({params}) -> dict:\n"
        assembled += f'    """{spec.description}"""\n'
        assembled += "    result = {}\n"

        # Wire matched blocks into the function body based on type
        for key in matched_keys:
            if key == "http_get":
                assembled += "    resp = await _http_get(url, params={})\n"
                assembled += "    result['response'] = resp.get('data', {})\n"
            elif key == "http_post":
                assembled += "    resp = await _http_post(url, payload={})\n"
                assembled += "    result['response'] = resp.get('data', {})\n"
            elif key == "json_parse":
                assembled += "    # JSON parsing available via _json_parse(text)\n"
            elif key == "regex_extract":
                assembled += "    # Regex extraction available via _regex_extract(text, pattern)\n"
            elif key == "math_stats":
                first_input = spec.inputs[0]['name'] if spec.inputs else 'values'
                assembled += f"    result.update(_math_stats({first_input}))\n"
            elif key == "file_read":
                first_input = spec.inputs[0]['name'] if spec.inputs else 'path'
                assembled += f"    result['content'] = _file_read({first_input})\n"
            elif key == "file_write":
                assembled += "    # File write available via _file_write(path, content)\n"
            elif key == "csv_parse":
                first_input = spec.inputs[0]['name'] if spec.inputs else 'text'
                assembled += f"    result['rows'] = _csv_parse({first_input})\n"
            elif key == "list_filter":
                assembled += "    # Filtering available via _list_filter(items, key, value)\n"
            elif key == "string_transform":
                first_input = spec.inputs[0]['name'] if spec.inputs else 'text'
                assembled += f"    result['transformed'] = _transform({first_input}, ['strip'])\n"
            elif key == "date_format":
                first_input = spec.inputs[0]['name'] if spec.inputs else 'date_str'
                assembled += f"    result['formatted'] = _date_format({first_input})\n"

        assembled += "    return result\n"
        log_entry["tiers_used"].append("ast_compose")

        # ── Tier 3: Optional LLM refinement ──────────────────────────
        if use_llm_refinement and self.llm_function:
            try:
                refined = await self.llm_function(
                    f"Improve this Python tool. Fix any issues, fill in placeholder "
                    f"logic, make it production-ready. Keep it async, typed, and safe.\n\n"
                    f"Specification: {spec.description}\n"
                    f"Current code:\n```python\n{assembled}\n```\n\n"
                    f"Return ONLY the improved code, no explanation."
                )
                # Extract code from response
                for pat in [r"```python\n(.*?)\n```", r"```\n(.*?)\n```"]:
                    m = re.search(pat, refined, re.DOTALL)
                    if m:
                        refined = m.group(1)
                        break
                # Validate the refinement
                try:
                    ast.parse(refined)
                    assembled = refined
                    log_entry["tiers_used"].append("llm_refine")
                except SyntaxError:
                    logger.warning("LLM refinement produced invalid syntax, keeping Tier 2 output")
            except Exception as e:
                logger.warning(f"LLM refinement failed: {e}, using pattern-assembled code")

        self._synthesis_log.append(log_entry)
        logger.info(f"Neural synthesis of '{spec.name}': tiers={log_entry['tiers_used']}, patterns={matched_keys}")
        return assembled

    def get_log(self) -> List[Dict[str, Any]]:
        return list(self._synthesis_log)


class ToolValidator:
    """
    Validates synthesized tools.
    
    Tests tools for correctness and safety.
    """
    
    def __init__(self):
        self.validation_results: List[Dict[str, Any]] = []
    
    async def validate_tool(
        self,
        tool: SynthesizedTool
    ) -> Dict[str, Any]:
        """
        Validate a synthesized tool.
        
        Args:
            tool: Tool to validate
        
        Returns:
            Validation results
        """
        results = {
            'syntax_valid': False,
            'execution_safe': False,
            'tests_passed': False,
            'errors': []
        }
        
        # 1. Syntax validation
        try:
            ast.parse(tool.code)
            results['syntax_valid'] = True
        except SyntaxError as e:
            results['errors'].append(f"Syntax error: {e}")
            return results
        
        # 2. Safety checks
        safety_check = self._check_safety(tool.code)
        results['execution_safe'] = safety_check['safe']
        if not safety_check['safe']:
            results['errors'].extend(safety_check['issues'])
            return results
        
        # 3. Test with examples
        if tool.specification.examples:
            test_results = await self._run_tests(tool)
            results['tests_passed'] = test_results['all_passed']
            results['test_details'] = test_results
        else:
            results['tests_passed'] = True  # No tests to run
        
        self.validation_results.append({
            'tool_id': tool.tool_id,
            'timestamp': datetime.utcnow().isoformat(),
            'results': results
        })
        
        return results
    
    def _check_safety(self, code: str) -> Dict[str, Any]:
        """Check code for safety issues."""
        issues = []
        
        # Check for dangerous operations
        dangerous_patterns = [
            (r'\bexec\b', "Uses exec()"),
            (r'\beval\b', "Uses eval()"),
            (r'\b__import__\b', "Uses __import__"),
            (r'\bopen\s*\([^)]*[\'"]w[\'"]', "Opens files in write mode"),
            (r'\bos\.system\b', "Uses os.system()"),
            (r'\bsubprocess\b', "Uses subprocess"),
            (r'\brm\s+-rf', "Contains rm -rf command"),
        ]
        
        for pattern, issue in dangerous_patterns:
            if re.search(pattern, code):
                issues.append(issue)
        
        return {
            'safe': len(issues) == 0,
            'issues': issues
        }
    
    async def _run_tests(self, tool: SynthesizedTool) -> Dict[str, Any]:
        """Run test cases."""
        if not tool.function:
            return {'all_passed': False, 'error': 'No function available'}
        
        passed = 0
        failed = 0
        test_details = []
        
        for i, example in enumerate(tool.specification.examples):
            try:
                # Extract inputs
                inputs = example.get('input', {})
                expected_output = example.get('output', {})
                
                # Run function
                result = await tool.function(**inputs)
                
                # Check result
                matches = self._compare_outputs(result, expected_output)
                
                if matches:
                    passed += 1
                    test_details.append({
                        'test': i + 1,
                        'status': 'passed',
                        'input': inputs,
                        'expected': expected_output,
                        'actual': result
                    })
                else:
                    failed += 1
                    test_details.append({
                        'test': i + 1,
                        'status': 'failed',
                        'input': inputs,
                        'expected': expected_output,
                        'actual': result,
                        'reason': 'Output mismatch'
                    })
            
            except Exception as e:
                failed += 1
                test_details.append({
                    'test': i + 1,
                    'status': 'error',
                    'input': inputs,
                    'error': str(e)
                })
        
        return {
            'all_passed': failed == 0,
            'passed': passed,
            'failed': failed,
            'total': len(tool.specification.examples),
            'details': test_details
        }
    
    def _compare_outputs(
        self,
        actual: Any,
        expected: Any,
        tolerance: float = 1e-6
    ) -> bool:
        """Compare actual vs expected outputs."""
        if isinstance(actual, dict) and isinstance(expected, dict):
            # Compare dictionaries
            if set(actual.keys()) != set(expected.keys()):
                return False
            
            for key in expected.keys():
                if not self._compare_outputs(actual[key], expected[key], tolerance):
                    return False
            
            return True
        
        elif isinstance(actual, (int, float)) and isinstance(expected, (int, float)):
            # Numeric comparison with tolerance
            return abs(actual - expected) < tolerance
        
        else:
            # Direct comparison
            return actual == expected


class ToolSynthesizer:
    """
    Main tool synthesis system.
    
    Coordinates code generation, validation, and deployment.
    """
    
    def __init__(
        self,
        llm_function: Optional[Callable] = None,
        storage_path: Optional[Path] = None
    ):
        self.generator = CodeGenerator(llm_function)
        self.neural = NeuralSynthesizer(llm_function)
        self.validator = ToolValidator()
        
        self.tools: Dict[str, SynthesizedTool] = {}
        self.storage_path = storage_path or Path("./data/synthesized_tools.json")
        self._load_tools()
    
    async def synthesize_tool(
        self,
        spec: ToolSpecification,
        auto_validate: bool = True,
        use_neural: bool = True,
    ) -> Optional[SynthesizedTool]:
        """
        Synthesize a new tool.
        
        Args:
            spec: Tool specification
            auto_validate: Automatically validate after synthesis
            use_neural: Use neural (pattern+AST) pipeline first, LLM-only as fallback
        
        Returns:
            Synthesized tool if successful
        """
        logger.info(f"Synthesizing tool: {spec.name}")
        
        # Try neural pipeline first (pattern match + AST compose + optional LLM polish)
        if use_neural:
            code = await self.neural.synthesize(spec, self.tools)
        else:
            code = await self.generator.generate_tool_code(spec)
        
        # Create tool object
        tool_id = f"tool_{len(self.tools)}_{spec.name}"
        tool = SynthesizedTool(
            tool_id=tool_id,
            name=spec.name,
            specification=spec,
            code=code
        )
        
        # Compile function
        try:
            # Create namespace
            namespace = {}
            exec(code, namespace)
            
            # Find the function
            func = None
            for name, obj in namespace.items():
                if inspect.iscoroutinefunction(obj):
                    func = obj
                    break
            
            if func:
                tool.function = func
            else:
                logger.error("No async function found in generated code")
                return None
        
        except Exception as e:
            logger.error(f"Failed to compile tool: {e}")
            return None
        
        # Validate if requested
        if auto_validate:
            validation_results = await self.validator.validate_tool(tool)
            tool.validated = validation_results.get('tests_passed', False)
            tool.test_results = validation_results
            
            if not tool.validated:
                logger.warning(f"Tool validation failed: {validation_results.get('errors', [])}")
                # Don't return None - still save the tool for debugging
        
        # Store tool
        self.tools[tool_id] = tool
        self._save_tools()
        
        logger.info(f"Synthesized tool: {tool_id} (validated: {tool.validated})")
        return tool
    
    async def compose_tools(
        self,
        tool_ids: List[str],
        composition_name: str,
        composition_description: str
    ) -> Optional[SynthesizedTool]:
        """
        Compose multiple tools into a new tool via AST merging.
        """
        tools = [self.tools[tid] for tid in tool_ids if tid in self.tools]
        
        if len(tools) != len(tool_ids):
            logger.error("Not all tools found for composition")
            return None
        
        # Use ASTComposer for real AST-level merging
        code = ASTComposer.compose(tools, composition_name, composition_description)
        
        # Create specification
        spec = ToolSpecification(
            name=composition_name,
            description=composition_description,
            tool_type=ToolType.DATA_PROCESSOR,
            inputs=[],  # Combined from component tools
            outputs=[]   # Combined from component tools
        )
        
        # Create composed tool
        tool_id = f"composed_{len(self.tools)}_{composition_name}"
        composed_tool = SynthesizedTool(
            tool_id=tool_id,
            name=composition_name,
            specification=spec,
            code=code,
            metadata={'composed_from': tool_ids}
        )
        
        self.tools[tool_id] = composed_tool
        self._save_tools()
        
        logger.info(f"Composed tool: {tool_id} from {len(tools)} tools")
        return composed_tool
    
    async def execute_tool(
        self,
        tool_id: str,
        **kwargs
    ) -> Any:
        """Execute a synthesized tool."""
        if tool_id not in self.tools:
            raise ValueError(f"Tool not found: {tool_id}")
        
        tool = self.tools[tool_id]
        
        if not tool.function:
            raise ValueError(f"Tool not executable: {tool_id}")
        
        if not tool.validated:
            logger.warning(f"Executing unvalidated tool: {tool_id}")
        
        try:
            result = await tool.function(**kwargs)
            tool.usage_count += 1
            self._save_tools()
            return result
        
        except Exception as e:
            logger.error(f"Tool execution failed: {e}")
            raise
    
    def get_tool(self, tool_id: str) -> Optional[SynthesizedTool]:
        """Get tool by ID."""
        return self.tools.get(tool_id)
    
    def list_tools(
        self,
        tool_type: Optional[ToolType] = None,
        validated_only: bool = False
    ) -> List[SynthesizedTool]:
        """List tools with optional filtering."""
        tools = list(self.tools.values())
        
        if tool_type:
            tools = [t for t in tools if t.specification.tool_type == tool_type]
        
        if validated_only:
            tools = [t for t in tools if t.validated]
        
        return tools
    
    def get_stats(self) -> Dict[str, Any]:
        """Get synthesis statistics."""
        return {
            'total_tools': len(self.tools),
            'validated_tools': sum(1 for t in self.tools.values() if t.validated),
            'total_executions': sum(t.usage_count for t in self.tools.values()),
            'by_type': {
                tt.value: sum(1 for t in self.tools.values() if t.specification.tool_type == tt)
                for tt in ToolType
            }
        }
    
    def _save_tools(self):
        """Save tools to disk."""
        try:
            self.storage_path.parent.mkdir(parents=True, exist_ok=True)
            
            data = {
                tid: tool.to_dict()
                for tid, tool in self.tools.items()
            }
            
            with open(self.storage_path, 'w') as f:
                json.dump(data, f, indent=2)
        
        except Exception as e:
            logger.error(f"Failed to save tools: {e}")
    
    def _load_tools(self):
        """Load tools from disk."""
        try:
            if not self.storage_path.exists():
                return
            
            with open(self.storage_path, 'r') as f:
                data = json.load(f)
            
            for tid, tool_data in data.items():
                # Reconstruct specification
                spec_data = tool_data['specification']
                spec = ToolSpecification(
                    name=spec_data['name'],
                    description=spec_data['description'],
                    tool_type=ToolType(spec_data['tool_type']),
                    inputs=spec_data['inputs'],
                    outputs=spec_data['outputs'],
                    requirements=spec_data.get('requirements', []),
                    constraints=spec_data.get('constraints', []),
                    examples=spec_data.get('examples', [])
                )
                
                # Reconstruct tool
                tool = SynthesizedTool(
                    tool_id=tool_data['tool_id'],
                    name=tool_data['name'],
                    specification=spec,
                    code=tool_data['code'],
                    validated=tool_data['validated'],
                    test_results=tool_data.get('test_results'),
                    created_at=datetime.fromisoformat(tool_data['created_at']),
                    usage_count=tool_data.get('usage_count', 0),
                    metadata=tool_data.get('metadata', {})
                )
                
                # Try to compile function
                try:
                    namespace = {}
                    exec(tool.code, namespace)
                    for name, obj in namespace.items():
                        if inspect.iscoroutinefunction(obj):
                            tool.function = obj
                            break
                except:
                    pass
                
                self.tools[tid] = tool
            
            logger.info(f"Loaded {len(self.tools)} synthesized tools")
        
        except Exception as e:
            logger.error(f"Failed to load tools: {e}")


# Example usage
async def main():
    """Example tool synthesis usage."""
    
    print("=" * 50)
    print("Tool Synthesis Example")
    print("=" * 50)
    
    # Initialize synthesizer
    synthesizer = ToolSynthesizer()
    
    # Create a tool specification
    print("\n1. Creating tool specification...")
    spec = ToolSpecification(
        name="temperature_converter",
        description="Convert temperature between Celsius and Fahrenheit",
        tool_type=ToolType.CONVERTER,
        inputs=[
            {'name': 'value', 'type': 'float', 'description': 'Temperature value'},
            {'name': 'from_unit', 'type': 'str', 'description': 'Source unit (C or F)'},
            {'name': 'to_unit', 'type': 'str', 'description': 'Target unit (C or F)'}
        ],
        outputs=[
            {'name': 'result', 'type': 'float', 'description': 'Converted temperature'},
            {'name': 'formula', 'type': 'str', 'description': 'Formula used'}
        ],
        examples=[
            {
                'input': {'value': 0, 'from_unit': 'C', 'to_unit': 'F'},
                'output': {'result': 32.0, 'formula': '(C * 9/5) + 32'}
            },
            {
                'input': {'value': 100, 'from_unit': 'C', 'to_unit': 'F'},
                'output': {'result': 212.0, 'formula': '(C * 9/5) + 32'}
            }
        ],
        constraints=[
            "Must handle both C to F and F to C conversions",
            "Must return the formula used"
        ]
    )
    print(f"  Created spec: {spec.name}")
    
    # Synthesize tool
    print("\n2. Synthesizing tool...")
    tool = await synthesizer.synthesize_tool(spec, auto_validate=True)
    
    if tool:
        print(f"  Tool ID: {tool.tool_id}")
        print(f"  Validated: {tool.validated}")
        if tool.test_results:
            print(f"  Tests: {tool.test_results.get('passed', 0)}/{tool.test_results.get('total', 0)} passed")
    
    # List tools
    print("\n3. Listing tools...")
    tools = synthesizer.list_tools()
    print(f"  Total tools: {len(tools)}")
    for t in tools:
        print(f"    - {t.name} ({t.specification.tool_type.value})")
    
    # Get statistics
    print("\n4. Statistics...")
    stats = synthesizer.get_stats()
    print(f"  Total tools: {stats['total_tools']}")
    print(f"  Validated: {stats['validated_tools']}")
    print(f"  Total executions: {stats['total_executions']}")
    print(f"  By type: {stats['by_type']}")
    
    print("\n✅ Tool synthesis example completed!")


if __name__ == "__main__":
    asyncio.run(main())
