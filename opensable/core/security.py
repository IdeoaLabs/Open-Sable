"""
Security and permission system for Open-Sable
"""
import logging
from typing import Dict, Any, List, Optional
from enum import Enum
from datetime import datetime
import json
from pathlib import Path

logger = logging.getLogger(__name__)


class PermissionLevel(Enum):
    """Permission levels for actions"""
    ALWAYS_ALLOW = "always_allow"
    ASK = "ask"
    DENY = "deny"


class ActionType(Enum):
    """Types of actions that require permissions"""
    EMAIL_READ = "email_read"
    EMAIL_SEND = "email_send"
    CALENDAR_READ = "calendar_read"
    CALENDAR_WRITE = "calendar_write"
    BROWSER_NAVIGATE = "browser_navigate"
    FILE_READ = "file_read"
    FILE_WRITE = "file_write"
    FILE_DELETE = "file_delete"
    SYSTEM_COMMAND = "system_command"


class PermissionManager:
    """Manages user permissions and security policies"""
    
    def __init__(self, config):
        self.config = config
        self.permissions_file = Path("./config/permissions.json")
        self.permissions: Dict[str, Dict[str, str]] = {}
        self.audit_log: List[Dict] = []
        self.audit_log_file = Path("./logs/audit.log")
    
    def initialize(self):
        """Load permissions from file"""
        if self.permissions_file.exists():
            with open(self.permissions_file, 'r') as f:
                self.permissions = json.load(f)
        else:
            # Default permissions (ask for everything)
            self.permissions = {
                "default": {
                    action.value: PermissionLevel.ASK.value
                    for action in ActionType
                }
            }
            self._save_permissions()
        
        logger.info("Permission manager initialized")
    
    async def check_permission(
        self,
        user_id: str,
        action: ActionType,
        context: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Check if action is permitted"""
        # Get user permissions (or default)
        user_perms = self.permissions.get(user_id, self.permissions.get("default", {}))
        permission_level = user_perms.get(action.value, PermissionLevel.ASK.value)
        
        # Log the attempt
        self._audit_log(user_id, action, permission_level, context)
        
        if permission_level == PermissionLevel.ALWAYS_ALLOW.value:
            return True
        elif permission_level == PermissionLevel.DENY.value:
            logger.warning(f"Action {action.value} denied for user {user_id}")
            return False
        else:  # ASK
            # In production, this would prompt the user
            # For now, we'll allow but log
            logger.info(f"Action {action.value} requires confirmation for user {user_id}")
            return True
    
    def set_permission(self, user_id: str, action: ActionType, level: PermissionLevel):
        """Set permission level for a user and action"""
        if user_id not in self.permissions:
            self.permissions[user_id] = {}
        
        self.permissions[user_id][action.value] = level.value
        self._save_permissions()
        logger.info(f"Set {action.value} = {level.value} for user {user_id}")
    
    def get_permissions(self, user_id: str) -> Dict[str, str]:
        """Get all permissions for a user"""
        return self.permissions.get(user_id, self.permissions.get("default", {}))
    
    def _save_permissions(self):
        """Save permissions to file"""
        self.permissions_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.permissions_file, 'w') as f:
            json.dump(self.permissions, f, indent=2)
    
    def _audit_log(
        self,
        user_id: str,
        action: ActionType,
        result: str,
        context: Optional[Dict] = None
    ):
        """Log action to audit trail"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "user_id": user_id,
            "action": action.value,
            "result": result,
            "context": context or {}
        }
        
        self.audit_log.append(log_entry)
        
        # Write to file
        self.audit_log_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.audit_log_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
    
    def get_audit_log(
        self,
        user_id: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict]:
        """Get audit log entries"""
        if user_id:
            return [
                entry for entry in self.audit_log
                if entry["user_id"] == user_id
            ][-limit:]
        return self.audit_log[-limit:]


class Sandbox:
    """Sandboxing utilities for safe execution"""
    
    @staticmethod
    def is_safe_path(path: str, allowed_dirs: List[str] = None) -> bool:
        """Check if file path is safe to access"""
        if allowed_dirs is None:
            allowed_dirs = ['./data', './logs']
        
        path_obj = Path(path).resolve()
        
        # Check against allowed directories
        for allowed_dir in allowed_dirs:
            allowed_path = Path(allowed_dir).resolve()
            try:
                path_obj.relative_to(allowed_path)
                return True
            except ValueError:
                continue
        
        return False
    
    @staticmethod
    def sanitize_input(text: str, max_length: int = 10000) -> str:
        """Sanitize user input"""
        # Truncate
        text = text[:max_length]
        
        # Remove potentially dangerous characters
        # This is basic - production would need more robust sanitization
        dangerous_chars = ['<script', 'javascript:', 'onerror=']
        for char in dangerous_chars:
            text = text.replace(char, '')
        
        return text
    
    @staticmethod
    def validate_url(url: str, allowed_domains: List[str]) -> bool:
        """Validate URL against allowed domains"""
        from urllib.parse import urlparse
        
        try:
            parsed = urlparse(url)
            domain = parsed.netloc
            
            for allowed in allowed_domains:
                if allowed.startswith('*.'):
                    if domain.endswith(allowed[2:]):
                        return True
                elif domain == allowed:
                    return True
            
            return False
        except Exception:
            return False
