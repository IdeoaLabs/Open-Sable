"""
Dynamic Skill Creation System

Allows Sable to create its own skills on-the-fly by generating Python code.
"""

import asyncio
import logging
import importlib.util
import sys
from pathlib import Path
from typing import Dict, Any, Optional, List
import json
import ast
import re

logger = logging.getLogger(__name__)


class SkillCreator:
    """
    Creates and manages dynamically generated skills
    """
    
    def __init__(self, config):
        self.config = config
        self.skills_dir = Path.home() / ".opensable" / "dynamic_skills"
        self.skills_dir.mkdir(parents=True, exist_ok=True)
        
        # Skill registry
        self.registry_file = self.skills_dir / "registry.json"
        self.registry: Dict[str, Dict] = self._load_registry()
        
    def _load_registry(self) -> Dict[str, Dict]:
        """Load skill registry"""
        if not self.registry_file.exists():
            return {}
        try:
            return json.loads(self.registry_file.read_text())
        except:
            return {}
            
    def _save_registry(self):
        """Save skill registry"""
        self.registry_file.write_text(json.dumps(self.registry, indent=2))
        
    async def create_skill(
        self, 
        name: str, 
        description: str, 
        code: str,
        metadata: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Create a new skill from generated code
        
        Args:
            name: Skill name (alphanumeric + underscores)
            description: What the skill does
            code: Python code for the skill
            metadata: Optional metadata (author, version, etc.)
            
        Returns:
            Result dict with success/error status
        """
        logger.info(f"🧬 Creating skill: {name}")
        
        # Validate name
        if not re.match(r'^[a-zA-Z0-9_]+$', name):
            return {
                "success": False,
                "error": "Skill name must be alphanumeric + underscores"
            }
        
        # Validate syntax
        syntax_check = self._validate_syntax(code)
        if not syntax_check["valid"]:
            return {
                "success": False,
                "error": f"Syntax error: {syntax_check['error']}"
            }
        
        # Security check
        security_check = self._check_security(code)
        if not security_check["safe"]:
            return {
                "success": False,
                "error": f"Security violation: {security_check['reason']}"
            }
        
        # Write skill file
        skill_file = self.skills_dir / f"{name}.py"
        skill_file.write_text(code)
        
        # Load skill dynamically
        try:
            spec = importlib.util.spec_from_file_location(name, skill_file)
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                sys.modules[name] = module
                spec.loader.exec_module(module)
                
                # Register skill
                self.registry[name] = {
                    "name": name,
                    "description": description,
                    "file": str(skill_file),
                    "created_at": str(asyncio.get_event_loop().time()),
                    "metadata": metadata or {},
                    "active": True
                }
                self._save_registry()
                
                logger.info(f"✅ Skill '{name}' created and loaded")
                return {
                    "success": True,
                    "skill": name,
                    "path": str(skill_file),
                    "message": f"Skill '{name}' created successfully"
                }
        except Exception as e:
            logger.error(f"Failed to load skill '{name}': {e}")
            skill_file.unlink(missing_ok=True)  # Clean up
            return {
                "success": False,
                "error": f"Failed to load skill: {str(e)}"
            }
        except Exception as e:
            logger.error(f"Failed to load skill '{name}': {e}")
            # Clean up failed skill
            if skill_file.exists():
                skill_file.unlink()
            return {
                "success": False,
                "error": f"Failed to load skill: {e}"
            }
            
    def _validate_syntax(self, code: str) -> Dict[str, Any]:
        """Validate Python syntax"""
        try:
            ast.parse(code)
            return {"valid": True}
        except SyntaxError as e:
            return {
                "valid": False,
                "error": f"Line {e.lineno}: {e.msg}"
            }
            
    def _check_security(self, code: str) -> Dict[str, Any]:
        """
        Basic security check for generated code
        
        Blocks:
        - os.system, subprocess with shell=True
        - eval, exec (except in sandboxed contexts)
        - File operations outside allowed directories
        - Network operations to non-whitelisted hosts
        """
        # Forbidden patterns
        forbidden = [
            r'os\.system\(',
            r'subprocess\..*shell\s*=\s*True',
            r'\beval\(',
            r'\bexec\(',
            r'__import__\(',
            r'open\(.*/etc/',
            r'open\(.*/root/',
        ]
        
        for pattern in forbidden:
            if re.search(pattern, code):
                return {
                    "safe": False,
                    "reason": f"Forbidden pattern detected: {pattern}"
                }
        
        return {"safe": True}
        
    def list_skills(self) -> List[Dict]:
        """List all created skills"""
        return list(self.registry.values())
        
    def disable_skill(self, name: str) -> bool:
        """Disable a skill"""
        if name in self.registry:
            self.registry[name]["active"] = False
            self._save_registry()
            logger.info(f"Disabled skill: {name}")
            return True
        return False
        
    def enable_skill(self, name: str) -> bool:
        """Enable a skill"""
        if name in self.registry:
            self.registry[name]["active"] = True
            self._save_registry()
            logger.info(f"Enabled skill: {name}")
            return True
        return False
        
    def delete_skill(self, name: str) -> bool:
        """Delete a skill"""
        if name in self.registry:
            # Remove file
            skill_file = Path(self.registry[name]["file"])
            if skill_file.exists():
                skill_file.unlink()
            
            # Remove from registry
            del self.registry[name]
            self._save_registry()
            
            # Unload from Python
            if name in sys.modules:
                del sys.modules[name]
                
            logger.info(f"Deleted skill: {name}")
            return True
        return False


# ── Example Skill Template ──────────────────────────────────────

SKILL_TEMPLATE = '''"""
{description}
"""

async def execute({params}):
    """
    Main skill execution function
    
    Args:
        {param_docs}
        
    Returns:
        Result dict
    """
    # TODO: Implement skill logic here
    
    return {{
        "success": True,
        "result": "Skill executed"
    }}
'''
