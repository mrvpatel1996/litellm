"""
RBAC API routes — custom roles management.
"""

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, Query, Request

from litellm.enterprise_open.config import CustomRoleConfig
from litellm.enterprise_open.rbac.roles import get_role_manager

logger = logging.getLogger(__name__)
rbac_router = APIRouter()


@rbac_router.post("")
async def create_role(config: CustomRoleConfig, request: Request):
    """Create a new custom RBAC role."""
    manager = get_role_manager()
    try:
        role = manager.create_role(config)
        return {"status": "ok", "role": role.model_dump()}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@rbac_router.get("")
async def list_roles(
    request: Request,
    type: Optional[str] = Query(None, description="Filter: 'builtin' or 'custom'"),
):
    """List all roles (built-in + custom)."""
    manager = get_role_manager()
    roles = manager.list_roles()
    if type:
        roles = [r for r in roles if r["type"] == type]
    return {"roles": roles, "count": len(roles)}


@rbac_router.get("/{role_name}")
async def get_role(role_name: str, request: Request):
    """Get details and effective permissions for a role."""
    manager = get_role_manager()
    perms = manager.resolve_permissions(role_name)
    custom = manager.get_role(role_name)
    return {
        "role_name": role_name,
        "type": "custom" if custom else "builtin",
        "effective_permissions": perms,
        "config": custom.model_dump() if custom else None,
    }


@rbac_router.put("/{role_name}")
async def update_role(role_name: str, config: CustomRoleConfig, request: Request):
    """Update a custom role."""
    manager = get_role_manager()
    try:
        updated = manager.update_role(
            role_name,
            permissions=config.permissions,
            description=config.description,
            base_role=config.base_role,
        )
        if not updated:
            raise HTTPException(status_code=404, detail=f"Role '{role_name}' not found")
        return {"status": "ok", "role": updated.model_dump()}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@rbac_router.delete("/{role_name}")
async def delete_role(role_name: str, request: Request):
    """Delete a custom role."""
    manager = get_role_manager()
    try:
        deleted = manager.delete_role(role_name)
        if not deleted:
            raise HTTPException(status_code=404, detail=f"Role '{role_name}' not found")
        return {"status": "ok", "message": f"Role '{role_name}' deleted"}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@rbac_router.get("/{role_name}/check")
async def check_permission(
    role_name: str,
    resource: str = Query(..., description="Resource type"),
    action: str = Query(..., description="Action to check"),
    request: Request = None,
):
    """Check if a role has a specific permission."""
    manager = get_role_manager()
    allowed = manager.check_permission(role_name, resource, action)
    return {"role": role_name, "resource": resource, "action": action, "allowed": allowed}
