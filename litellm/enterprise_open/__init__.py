"""
LiteLLM Enterprise Open — Full enterprise features, no license gate.

Features:
- SAML 2.0 SSO
- Enhanced RBAC with custom roles
- Full Audit Log API
- Custom Pricing & Billing engine
- Multi-tenant data isolation
"""

from litellm.enterprise_open.router import enterprise_router

__all__ = ["enterprise_router"]
