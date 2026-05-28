"""
Custom RBAC role management service.
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from litellm.enterprise_open.config import CustomRoleConfig

logger = logging.getLogger(__name__)

DEFAULT_ROLES_PATH = os.path.expanduser("~/.litellm/custom_roles.json")

# Default permission sets for built-in roles
BUILTIN_ROLE_PERMISSIONS: Dict[str, Dict[str, List[str]]] = {
    "PROXY_ADMIN": {
        "keys": ["read", "create", "update", "delete"],
        "users": ["read", "create", "update", "delete"],
        "teams": ["read", "create", "update", "delete"],
        "models": ["read", "create", "update", "delete"],
        "organizations": ["read", "create", "update", "delete"],
        "budgets": ["read", "create", "update", "delete"],
        "configs": ["read", "create", "update", "delete"],
        "audit": ["read", "export"],
        "pricing": ["read", "create", "update", "delete"],
        "roles": ["read", "create", "update", "delete"],
    },
    "PROXY_ADMIN_VIEW_ONLY": {
        "keys": ["read"],
        "users": ["read"],
        "teams": ["read"],
        "models": ["read"],
        "organizations": ["read"],
        "budgets": ["read"],
        "configs": ["read"],
        "audit": ["read"],
        "pricing": ["read"],
        "roles": ["read"],
    },
    "ORG_ADMIN": {
        "keys": ["read", "create", "update", "delete"],
        "users": ["read", "create", "update"],
        "teams": ["read", "create", "update"],
        "models": ["read"],
        "organizations": ["read"],
        "budgets": ["read", "create", "update"],
        "configs": ["read"],
        "audit": ["read"],
        "pricing": ["read"],
        "roles": ["read"],
    },
    "INTERNAL_USER": {
        "keys": ["read", "create"],
        "users": ["read", "update"],
        "teams": ["read"],
        "models": ["read"],
        "organizations": ["read"],
        "budgets": ["read"],
        "configs": [],
        "audit": [],
        "pricing": [],
        "roles": [],
    },
    "INTERNAL_USER_VIEW_ONLY": {
        "keys": ["read"],
        "users": ["read"],
        "teams": ["read"],
        "models": ["read"],
        "organizations": ["read"],
        "budgets": [],
        "configs": [],
        "audit": [],
        "pricing": [],
        "roles": [],
    },
    "TEAM": {
        "keys": ["read", "create"],
        "users": ["read"],
        "teams": ["read", "update"],
        "models": ["read"],
        "organizations": ["read"],
        "budgets": ["read"],
        "configs": [],
        "audit": [],
        "pricing": [],
        "roles": [],
    },
    "CUSTOMER": {
        "keys": ["read"],
        "users": ["read", "update"],
        "teams": [],
        "models": ["read"],
        "organizations": [],
        "budgets": [],
        "configs": [],
        "audit": [],
        "pricing": [],
        "roles": [],
    },
}


class RoleManager:
    """Manages custom RBAC roles and permission checks."""

    def __init__(self, roles_path: str = DEFAULT_ROLES_PATH):
        self._path = Path(roles_path)
        self._custom_roles: Dict[str, CustomRoleConfig] = {}
        self._load()

    def _load(self):
        if self._path.exists():
            try:
                with open(self._path, "r") as f:
                    data = json.load(f)
                self._custom_roles = {
                    name: CustomRoleConfig(**cfg) for name, cfg in data.items()
                }
                logger.info(f"Loaded {len(self._custom_roles)} custom role(s)")
            except Exception as e:
                logger.error(f"Failed to load custom roles: {e}")
                self._custom_roles = {}

    def _save(self):
        self._path.parent.mkdir(parents=True, exist_ok=True)
        data = {name: cfg.model_dump() for name, cfg in self._custom_roles.items()}
        with open(self._path, "w") as f:
            json.dump(data, f, indent=2, default=str)

    def create_role(self, config: CustomRoleConfig) -> CustomRoleConfig:
        """Create a new custom role."""
        if config.role_name in BUILTIN_ROLE_PERMISSIONS:
            raise ValueError(f"Cannot override built-in role: {config.role_name}")
        self._custom_roles[config.role_name] = config
        self._save()
        logger.info(f"Created custom role: {config.role_name}")
        return config

    def get_role(self, role_name: str) -> Optional[CustomRoleConfig]:
        return self._custom_roles.get(role_name)

    def list_roles(self) -> List[Dict[str, Any]]:
        """List all roles (built-in + custom)."""
        roles = []
        for name, perms in BUILTIN_ROLE_PERMISSIONS.items():
            roles.append({
                "role_name": name,
                "type": "builtin",
                "permissions": perms,
            })
        for name, config in self._custom_roles.items():
            roles.append({
                "role_name": name,
                "type": "custom",
                "base_role": config.base_role,
                "permissions": config.permissions,
                "description": config.description,
                "organization_id": config.organization_id,
            })
        return roles

    def update_role(self, role_name: str, **kwargs) -> Optional[CustomRoleConfig]:
        """Update a custom role's properties."""
        if role_name in BUILTIN_ROLE_PERMISSIONS:
            raise ValueError(f"Cannot modify built-in role: {role_name}")
        config = self._custom_roles.get(role_name)
        if not config:
            return None
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
        self._save()
        return config

    def delete_role(self, role_name: str) -> bool:
        if role_name in BUILTIN_ROLE_PERMISSIONS:
            raise ValueError(f"Cannot delete built-in role: {role_name}")
        if role_name in self._custom_roles:
            del self._custom_roles[role_name]
            self._save()
            return True
        return False

    def resolve_permissions(self, role_name: str) -> Dict[str, List[str]]:
        """
        Resolve the effective permissions for a role.
        Custom roles inherit from their base role, then apply overrides.
        """
        # Check if custom role
        custom = self._custom_roles.get(role_name)
        if custom:
            # Start with base role permissions
            base_perms = BUILTIN_ROLE_PERMISSIONS.get(
                custom.base_role, {}
            )
            # Merge custom permissions (override)
            resolved = dict(base_perms)
            for resource, actions in custom.permissions.items():
                resolved[resource] = actions
            return resolved

        # Built-in role
        return BUILTIN_ROLE_PERMISSIONS.get(role_name, {})

    def check_permission(
        self, role_name: str, resource: str, action: str
    ) -> bool:
        """Check if a role has a specific permission on a resource."""
        # Admin always has access
        if role_name == "PROXY_ADMIN":
            return True

        perms = self.resolve_permissions(role_name)
        resource_perms = perms.get(resource, [])
        return action in resource_perms


# Singleton
_manager: Optional[RoleManager] = None


def get_role_manager() -> RoleManager:
    global _manager
    if _manager is None:
        _manager = RoleManager()
    return _manager
