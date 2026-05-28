# LiteLLM Enterprise Features Implementation Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Fork LiteLLM and build a full open-source enterprise feature suite — removing the paid license gate and adding SAML 2.0, enhanced RBAC, audit log API, custom pricing engine, and proper multi-tenancy.

**Architecture:** Build on top of the existing `litellm/proxy/` FastAPI structure. Enterprise features live in a new `litellm/enterprise_open/` package (separate from the commercial `enterprise/` dir). Features are registered as FastAPI routers and loaded at proxy startup.

**Tech Stack:** Python 3.10+, FastAPI, Prisma ORM, PostgreSQL, Redis (rate limiting), python3-saml (OneLogin) for SAML 2.0

---

## Phase 1: Unlock Existing Enterprise Features (Remove License Gate)

### Task 1.1: Create enterprise_open package structure
- Create: `litellm/enterprise_open/__init__.py`
- Create: `litellm/enterprise_open/router.py` — main enterprise FastAPI router
- Create: `litellm/enterprise_open/config.py` — enterprise config models

### Task 1.2: Remove premium_user license checks
- Modify: `litellm/proxy/auth/litellm_license.py` — make `_premium_user_check()` always return True
- Modify: All files calling `premium_user` / `_premium_user_check()` — remove gates
- Search: `grep -r "premium_user" litellm/` to find all gating points

### Task 1.3: Import enterprise callbacks into enterprise_open
- Move key enterprise callbacks from `enterprise/litellm_enterprise/` into `litellm/enterprise_open/`:
  - Secret detection
  - Email notifications (SendGrid, SMTP)
  - Banned keywords filter
  - OpenAI moderation
  - LLM Guard integration

---

## Phase 2: SAML 2.0 SSO

### Task 2.1: Install SAML dependencies
- Add to pyproject.toml: `python3-saml>=1.14.0` (OneLogin SAML library)
- Add: `lxml>=4.9.0`, `xmlsec>=1.3.13`

### Task 2.2: Create SAML config model
- Create: `litellm/enterprise_open/saml/config.py`
- Pydantic model for SAML IdP settings: entity_id, sso_url, slo_url, x509_cert, acs_url

### Task 2.3: Create SAML auth handler
- Create: `litellm/enterprise_open/saml/handler.py`
- Initiate SAML AuthnRequest → redirect to IdP
- Handle ACS (Assertion Consumer Service) callback
- Parse SAML assertion → extract user attributes (email, name, groups)
- Map SAML groups to LiteLLM roles/teams

### Task 2.4: Create SAML API routes
- Create: `litellm/enterprise_open/saml/routes.py`
- `GET /enterprise/saml/login` — initiate SAML SSO
- `POST /enterprise/saml/acs` — assertion consumer endpoint
- `GET /enterprise/saml/metadata` — SP metadata XML
- `POST /enterprise/saml/logout` — SLO endpoint

### Task 2.5: Add SAML DB model to schema.prisma
- Add `LiteLLM_SAMLConfig` model with: id, provider_name, entity_id, sso_url, slo_url, x509cert, created_at, updated_at

### Task 2.6: Management API for SAML config
- `POST /enterprise/saml/config` — create/update SAML provider
- `GET /enterprise/saml/config` — list SAML providers
- `DELETE /enterprise/saml/config/{id}` — remove provider

---

## Phase 3: Enhanced RBAC with Custom Roles

### Task 3.1: Create custom roles DB model
- Add `LiteLLM_CustomRole` to schema.prisma:
  - id, role_name, permissions (JSON), is_default, created_by, organization_id
- Add `LiteLLM_RolePermission` for fine-grained permissions:
  - id, role_id, resource_type, action (CRUD), conditions (JSON)

### Task 3.2: Create role management service
- Create: `litellm/enterprise_open/rbac/roles.py`
- CRUD for custom roles
- Permission resolution: custom role → base role → deny
- Role inheritance support

### Task 3.3: Create permission checker middleware
- Create: `litellm/enterprise_open/rbac/permissions.py`
- Replace hardcoded `LitellmUserRoles` checks with permission-based system
- Backward compatible: existing roles map to default permission sets

### Task 3.4: Role management API routes
- Create: `litellm/enterprise_open/rbac/routes.py`
- `POST /enterprise/roles` — create custom role
- `GET /enterprise/roles` — list roles
- `PUT /enterprise/roles/{id}` — update role
- `DELETE /enterprise/roles/{id}` — delete role
- `POST /enterprise/roles/{id}/permissions` — set permissions
- `POST /enterprise/users/{user_id}/role` — assign role to user

---

## Phase 4: Full Audit Log API

### Task 4.1: Enhance audit log model
- Extend `LiteLLM_AuditLog` in schema.prisma:
  - Add: ip_address, user_agent, request_id, severity, resource_type, metadata (JSON)

### Task 4.2: Create audit log query service
- Create: `litellm/enterprise_open/audit/service.py`
- Query with filters: date range, user, action, resource, severity
- Aggregation: counts by action, top users, timeline
- Export to CSV/JSON

### Task 4.3: Create audit log API routes
- Create: `litellm/enterprise_open/audit/routes.py`
- `GET /enterprise/audit/logs` — query with filters (pagination, sorting)
- `GET /enterprise/audit/logs/{id}` — get single log detail
- `GET /enterprise/audit/summary` — aggregated statistics
- `GET /enterprise/audit/export` — export logs (CSV, JSON)
- `GET /enterprise/audit/stream` — SSE stream for real-time audit events

### Task 4.4: Add comprehensive audit logging hooks
- Hook into all management CRUD operations
- Log: API key creation/deletion, user changes, config changes, model updates
- Include full before/after state diff

---

## Phase 5: Custom Pricing/Billing Engine

### Task 5.1: Create pricing model
- Add to schema.prisma:
  - `LiteLLM_PricingRule`: id, name, model_pattern, markup_percent, flat_fee, custom_price_per_token, priority, organization_id, team_id, effective_from, effective_to
  - `LiteLLM_Invoice`: id, organization_id, team_id, period_start, period_end, total_cost, markup_cost, status
  - `LiteLLM_InvoiceLineItem`: id, invoice_id, model, tokens_in, tokens_out, cost, markup, custom_rate

### Task 5.2: Create pricing engine
- Create: `litellm/enterprise_open/pricing/engine.py`
- Apply pricing rules: model pattern match → markup calculation
- Priority-based rule resolution (most specific wins)
- Support: % markup, flat fee per request, custom per-token rate

### Task 5.3: Integrate pricing with spend tracking
- Hook into `proxy_track_cost_callback.py`
- Apply pricing rules during cost calculation
- Store both base cost and marked-up cost

### Task 5.4: Create billing/invoice service
- Create: `litellm/enterprise_open/pricing/billing.py`
- Generate invoices from spend data + pricing rules
- Period: daily, weekly, monthly
- PDF invoice generation (via reportlab or weasyprint)

### Task 5.5: Pricing management API routes
- Create: `litellm/enterprise_open/pricing/routes.py`
- `POST /enterprise/pricing/rules` — create pricing rule
- `GET /enterprise/pricing/rules` — list rules
- `PUT /enterprise/pricing/rules/{id}` — update rule
- `DELETE /enterprise/pricing/rules/{id}` — delete rule
- `POST /enterprise/pricing/calculate` — preview cost for a request
- `GET /enterprise/invoices` — list invoices
- `GET /enterprise/invoices/{id}` — get invoice detail
- `GET /enterprise/invoices/{id}/pdf` — download invoice PDF

---

## Phase 6: Multi-Tenant Data Isolation

### Task 6.1: Tenant context middleware
- Create: `litellm/enterprise_open/multitenant/middleware.py`
- Extract tenant from: API key → team → organization
- Set tenant context in request state
- Enforce tenant isolation in all DB queries

### Task 6.2: Tenant-scoped data access layer
- Create: `litellm/enterprise_open/multitenant/data_access.py`
- Wrapper around PrismaClient that auto-filters by tenant
- Models, keys, users, spend logs — all scoped to tenant
- Super-admin bypass for cross-tenant access

### Task 6.3: Tenant configuration isolation
- Each tenant gets: own model list, own routing config, own rate limits
- Tenant config stored in `LiteLLM_Config` with org_id/team_id scope
- Config resolution: tenant-specific → global fallback

### Task 6.4: Tenant management API
- Create: `litellm/enterprise_open/multitenant/routes.py`
- `POST /enterprise/tenants` — create tenant (org + config)
- `GET /enterprise/tenants` — list tenants
- `GET /enterprise/tenants/{id}` — tenant details + usage
- `PUT /enterprise/tenants/{id}` — update tenant config
- `GET /enterprise/tenants/{id}/usage` — usage metrics
- `GET /enterprise/tenants/{id}/limits` — rate limits & quotas

---

## Phase 7: Integration & Registration

### Task 7.1: Register enterprise router in proxy startup
- Modify: `litellm/proxy/proxy_server.py` — import and mount `enterprise_open` router
- Add config flag: `enterprise_settings.enabled: true`

### Task 7.2: Add enterprise config to _types.py
- Add `EnterpriseSettings` to ConfigYAML
- Support env vars: `ENTERPRISE_ENABLED`, `SAML_ENABLED`, `CUSTOM_PRICING_ENABLED`

### Task 7.3: Run Prisma migration
- Generate migration for new models
- Test migration on clean DB

### Task 7.4: Integration tests
- Test full SAML flow (mock IdP)
- Test RBAC with custom roles
- Test audit log query API
- Test pricing calculation accuracy
- Test multi-tenant data isolation (cross-tenant data leak test)

---

## File Structure Summary

```
litellm/enterprise_open/
├── __init__.py
├── router.py                    # Main enterprise FastAPI router
├── config.py                    # Enterprise config models
├── saml/
│   ├── __init__.py
│   ├── config.py               # SAML IdP settings
│   ├── handler.py              # SAML auth logic
│   └── routes.py               # SAML API routes
├── rbac/
│   ├── __init__.py
│   ├── roles.py                # Custom role management
│   ├── permissions.py          # Permission checker
│   └── routes.py               # Role management API
├── audit/
│   ├── __init__.py
│   ├── service.py              # Query, aggregate, export
│   └── routes.py               # Audit log API
├── pricing/
│   ├── __init__.py
│   ├── engine.py               # Pricing rule evaluation
│   ├── billing.py              # Invoice generation
│   └── routes.py               # Pricing/billing API
├── multitenant/
│   ├── __init__.py
│   ├── middleware.py            # Tenant context extraction
│   ├── data_access.py          # Tenant-scoped queries
│   └── routes.py               # Tenant management API
└── tests/
    ├── test_saml.py
    ├── test_rbac.py
    ├── test_audit.py
    ├── test_pricing.py
    └── test_multitenant.py
```
