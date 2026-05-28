"""
Main Enterprise Open router — mounts all enterprise feature sub-routers.
"""

from fastapi import APIRouter, Depends, HTTPException, status

enterprise_router = APIRouter(prefix="/enterprise", tags=["Enterprise"])


def _check_enterprise_enabled():
    """Dependency to verify enterprise features are enabled."""
    from litellm.proxy.proxy_server import general_settings
    enterprise_settings = general_settings.get("enterprise_settings", {}) if general_settings else {}
    if not enterprise_settings.get("enabled", False):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Enterprise features are not enabled. Set enterprise_settings.enabled=true in config.",
        )


def mount_enterprise_routes():
    """Mount all enterprise sub-routers. Called at proxy startup."""
    from litellm.enterprise_open.saml.routes import saml_router
    from litellm.enterprise_open.rbac.routes import rbac_router
    from litellm.enterprise_open.audit.routes import audit_router
    from litellm.enterprise_open.pricing.routes import pricing_router
    from litellm.enterprise_open.multitenant.routes import tenant_router

    enterprise_router.include_router(saml_router, prefix="/saml", tags=["SAML SSO"])
    enterprise_router.include_router(rbac_router, prefix="/roles", tags=["RBAC"])
    enterprise_router.include_router(audit_router, prefix="/audit", tags=["Audit Logs"])
    enterprise_router.include_router(pricing_router, prefix="/pricing", tags=["Pricing & Billing"])
    enterprise_router.include_router(tenant_router, prefix="/tenants", tags=["Multi-Tenancy"])

    @enterprise_router.get("/health")
    async def enterprise_health():
        return {
            "status": "healthy",
            "version": "1.0.0",
            "features": {
                "saml": True,
                "rbac": True,
                "audit": True,
                "pricing": True,
                "multi_tenancy": True,
            },
        }

    @enterprise_router.get("/config")
    async def get_enterprise_config(user=Depends(_check_enterprise_enabled)):
        from litellm.proxy.proxy_server import general_settings
        enterprise_settings = general_settings.get("enterprise_settings", {}) if general_settings else {}
        return {
            "enabled": enterprise_settings.get("enabled", False),
            "saml_enabled": enterprise_settings.get("saml_enabled", False),
            "custom_pricing_enabled": enterprise_settings.get("custom_pricing_enabled", False),
            "enhanced_audit_enabled": enterprise_settings.get("enhanced_audit_enabled", False),
            "custom_roles_enabled": enterprise_settings.get("custom_roles_enabled", False),
            "multi_tenancy_enabled": enterprise_settings.get("multi_tenancy_enabled", False),
        }


mount_enterprise_routes()
