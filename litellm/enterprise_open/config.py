"""
Enterprise configuration models for LiteLLM Enterprise Open.
"""

from pydantic import BaseModel, Field
from typing import Dict, List, Optional
from enum import Enum


class EnterpriseFeature(str, Enum):
    SAML = "saml"
    RBAC = "rbac"
    AUDIT = "audit"
    PRICING = "pricing"
    MULTI_TENANCY = "multi_tenancy"


class EnterpriseSettings(BaseModel):
    """Top-level enterprise settings — placed in config.yaml under `enterprise_settings`."""
    enabled: bool = False
    saml_enabled: bool = False
    custom_pricing_enabled: bool = False
    enhanced_audit_enabled: bool = False
    custom_roles_enabled: bool = False
    multi_tenancy_enabled: bool = False

    def is_feature_enabled(self, feature: EnterpriseFeature) -> bool:
        mapping = {
            EnterpriseFeature.SAML: self.saml_enabled,
            EnterpriseFeature.RBAC: self.custom_roles_enabled,
            EnterpriseFeature.AUDIT: self.enhanced_audit_enabled,
            EnterpriseFeature.PRICING: self.custom_pricing_enabled,
            EnterpriseFeature.MULTI_TENANCY: self.multi_tenancy_enabled,
        }
        return self.enabled and mapping.get(feature, False)


class SAMLProviderConfig(BaseModel):
    """Configuration for a single SAML Identity Provider."""
    provider_name: str
    entity_id: str = Field(..., description="SP Entity ID")
    sso_url: str = Field(..., description="IdP Single Sign-On URL")
    slo_url: Optional[str] = Field(None, description="IdP Single Logout URL")
    x509cert: str = Field(..., description="IdP X.509 public certificate")
    acs_url: str = Field(..., description="Our Assertion Consumer Service URL")
    sp_private_key: Optional[str] = Field(None, description="Our SP private key for signed requests")
    sp_x509cert: Optional[str] = Field(None, description="Our SP X.509 certificate")
    attribute_mapping: Dict[str, str] = Field(
        default={"email": "email", "firstName": "firstName", "lastName": "lastName"},
        description="Map SAML attribute names to LiteLLM user fields",
    )
    default_role: str = Field("INTERNAL_USER", description="Default LiteLLM role for new SAML users")
    team_mapping: Optional[Dict[str, str]] = Field(
        None, description="Map SAML group names to LiteLLM team IDs"
    )
    role_mapping: Optional[Dict[str, str]] = Field(
        None, description="Map SAML group names to LiteLLM roles"
    )
    want_assertions_signed: bool = Field(True, description="Require signed SAML assertions")
    want_responses_signed: bool = Field(False, description="Require signed SAML responses")
    allowed_clock_skew: int = Field(300, description="Allowed clock skew in seconds")


class PricingRuleConfig(BaseModel):
    """Configuration for a custom pricing rule."""
    name: str
    model_pattern: str = Field(..., description="Regex pattern to match model names")
    markup_percent: Optional[float] = Field(None, description="Percentage markup on base cost")
    flat_fee_per_request: Optional[float] = Field(None, description="Flat fee added per request")
    custom_price_input_per_1k: Optional[float] = Field(None, description="Custom price per 1K input tokens")
    custom_price_output_per_1k: Optional[float] = Field(None, description="Custom price per 1K output tokens")
    priority: int = Field(0, description="Higher priority rules take precedence")
    organization_id: Optional[str] = Field(None, description="Scope to organization")
    team_id: Optional[str] = Field(None, description="Scope to team")
    effective_from: Optional[str] = Field(None, description="ISO datetime when rule becomes active")
    effective_to: Optional[str] = Field(None, description="ISO datetime when rule expires")
    description: Optional[str] = None


class CustomRoleConfig(BaseModel):
    """Configuration for a custom RBAC role."""
    role_name: str
    description: Optional[str] = None
    base_role: str = Field("INTERNAL_USER", description="Inherit permissions from this base role")
    permissions: Dict[str, List[str]] = Field(
        default_factory=dict,
        description="Resource -> list of allowed actions, e.g. {'keys': ['read', 'create'], 'models': ['read']}",
    )
    organization_id: Optional[str] = Field(None, description="Scope to organization")
    is_default: bool = False


class TenantConfig(BaseModel):
    """Configuration for a multi-tenant organization."""
    tenant_name: str
    organization_id: str
    config_override: Optional[Dict] = Field(None, description="Tenant-specific proxy config overrides")
    max_budget: Optional[float] = Field(None, description="Maximum budget cap for tenant")
    max_users: Optional[int] = Field(None, description="Maximum number of users")
    max_teams: Optional[int] = Field(None, description="Maximum number of teams")
    allowed_models: Optional[List[str]] = Field(None, description="Models allowed for this tenant")
    custom_pricing_rules: Optional[List[PricingRuleConfig]] = None
