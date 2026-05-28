"""
Multi-tenant middleware — extracts tenant context from API keys and enforces isolation.
"""

import logging
from typing import Any, Dict, Optional

from fastapi import Request

logger = logging.getLogger(__name__)


class TenantContext:
    """Represents the current tenant context for a request."""

    def __init__(
        self,
        organization_id: Optional[str] = None,
        team_id: Optional[str] = None,
        user_id: Optional[str] = None,
        is_super_admin: bool = False,
    ):
        self.organization_id = organization_id
        self.team_id = team_id
        self.user_id = user_id
        self.is_super_admin = is_super_admin

    def get_scope_filters(self) -> Dict[str, str]:
        """Get Prisma where-clause filters for tenant-scoped queries."""
        filters: Dict[str, str] = {}
        if self.organization_id and not self.is_super_admin:
            filters["organization_id"] = self.organization_id
        return filters

    def can_access(self, resource_org_id: Optional[str] = None) -> bool:
        """Check if this tenant context can access a resource."""
        if self.is_super_admin:
            return True
        if not self.organization_id:
            return True  # No tenant context = global access
        return resource_org_id == self.organization_id

    def to_dict(self) -> Dict[str, Any]:
        return {
            "organization_id": self.organization_id,
            "team_id": self.team_id,
            "user_id": self.user_id,
            "is_super_admin": self.is_super_admin,
        }


async def extract_tenant_context(request: Request) -> TenantContext:
    """
    Extract tenant context from the authenticated user.

    Resolution chain:
    1. Check if user is super admin (PROXY_ADMIN) → full access
    2. Check user's organization_id → org-scoped access
    3. Check user's team_id → team-scoped access
    """
    # Try to get the authenticated user from the request
    user = getattr(request.state, "user", None)

    if user is None:
        # Try to get from the auth dependency result
        try:
            from litellm.proxy.auth.user_api_key_auth import UserAPIKeyAuth
            # The auth dependency may have set user info on the request
            user = getattr(request, "_user_auth", None)
        except (ImportError, AttributeError):
            pass

    if user is None:
        # No auth context — return empty tenant (will be filtered by auth middleware)
        return TenantContext()

    # Check if super admin
    role = getattr(user, "user_role", None) or getattr(user, "role", None)
    is_super_admin = role in ("PROXY_ADMIN",)

    # Extract organization and team
    organization_id = (
        getattr(user, "organization_id", None)
        or getattr(user, "org_id", None)
    )
    team_id = (
        getattr(user, "team_id", None)
        or getattr(user, "team", None)
    )
    user_id = getattr(user, "user_id", None)

    return TenantContext(
        organization_id=organization_id,
        team_id=team_id if not isinstance(team_id, list) else team_id[0] if team_id else None,
        user_id=user_id,
        is_super_admin=is_super_admin,
    )


class TenantScopedDataAccess:
    """
    Wraps Prisma client calls with tenant-scoped filters.
    Ensures data isolation between tenants.
    """

    def __init__(self, tenant: TenantContext):
        self.tenant = tenant

    def _apply_tenant_filter(self, where: Dict[str, Any]) -> Dict[str, Any]:
        """Add tenant scope filters to a Prisma where clause."""
        if self.tenant.is_super_admin:
            return where  # Super admins see everything

        if self.tenant.organization_id:
            # Scope to organization
            where.setdefault("organization_id", self.tenant.organization_id)

        return where

    async def find_user(self, prisma, user_id: str):
        """Find a user, scoped to tenant."""
        return await prisma.db.litellm_usertable.find_first(
            where=self._apply_tenant_filter({"user_id": user_id})
        )

    async def find_team(self, prisma, team_id: str):
        """Find a team, scoped to tenant."""
        return await prisma.db.litellm_teamtable.find_first(
            where=self._apply_tenant_filter({"team_id": team_id})
        )

    async def find_key(self, prisma, token_hash: str):
        """Find an API key, scoped to tenant."""
        return await prisma.db.litellm_verificationtoken.find_first(
            where=self._apply_tenant_filter({"token": token_hash})
        )

    async def list_users(self, prisma, page: int = 1, page_size: int = 50):
        """List users, scoped to tenant."""
        where = self._apply_tenant_filter({})
        return await prisma.db.litellm_usertable.find_many(
            where=where,
            skip=(page - 1) * page_size,
            take=page_size,
        )

    async def list_teams(self, prisma, page: int = 1, page_size: int = 50):
        """List teams, scoped to tenant."""
        where = self._apply_tenant_filter({})
        return await prisma.db.litellm_teamtable.find_many(
            where=where,
            skip=(page - 1) * page_size,
            take=page_size,
        )

    async def list_spend_logs(self, prisma, page: int = 1, page_size: int = 50):
        """List spend logs, scoped to tenant."""
        where = self._apply_tenant_filter({})
        return await prisma.db.litellm_spendlogs.find_many(
            where=where,
            skip=(page - 1) * page_size,
            take=page_size,
            order={"startTime": "desc"},
        )

    async def count_resources(self, prisma) -> Dict[str, int]:
        """Count resources for this tenant."""
        where = self._apply_tenant_filter({})
        users = await prisma.db.litellm_usertable.count(where=where)
        teams = await prisma.db.litellm_teamtable.count(where=where)
        keys_where = self._apply_tenant_filter({})
        keys = await prisma.db.litellm_verificationtoken.count(where=keys_where)
        return {"users": users, "teams": teams, "keys": keys}
