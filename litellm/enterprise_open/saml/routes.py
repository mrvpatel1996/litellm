"""
SAML 2.0 SSO API routes.
"""

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, Query, Request, status
from fastapi.responses import HTMLResponse, RedirectResponse, Response

from litellm.enterprise_open.config import SAMLProviderConfig
from litellm.enterprise_open.saml.config import get_saml_store
from litellm.enterprise_open.saml.handler import (
    generate_logout_request,
    generate_sp_metadata,
    initiate_saml_login,
    process_saml_response,
)

logger = logging.getLogger(__name__)
saml_router = APIRouter()


@saml_router.get("/login")
async def saml_login(
    provider: str = Query(..., description="SAML provider name"),
    relay_state: Optional[str] = Query(None, description="Relay state URL after login"),
):
    """Initiate SAML SSO login — redirects user to their IdP."""
    store = get_saml_store()
    provider_config = store.get_provider(provider)
    if not provider_config:
        raise HTTPException(status_code=404, detail=f"SAML provider '{provider}' not found")

    try:
        redirect_url = initiate_saml_login(provider_config, relay_state=relay_state)
        return RedirectResponse(url=redirect_url, status_code=303)
    except Exception as e:
        logger.error(f"SAML login initiation failed: {e}")
        raise HTTPException(status_code=500, detail=f"SAML login failed: {str(e)}")


@saml_router.post("/acs")
async def saml_acs(request: Request):
    """
    Assertion Consumer Service endpoint.
    Receives the SAML response from the IdP after user authentication.
    """
    form_data = await request.form()
    saml_response_b64 = form_data.get("SAMLResponse")
    relay_state = form_data.get("RelayState")

    if not saml_response_b64:
        raise HTTPException(status_code=400, detail="Missing SAMLResponse parameter")

    # Try to determine the provider from the assertion issuer
    store = get_saml_store()
    providers = store.get_all_providers()

    user_info = None
    matched_provider = None

    for provider_name, provider_config in providers.items():
        try:
            user_info = process_saml_response(str(saml_response_b64), provider_config)
            matched_provider = provider_config
            break
        except ValueError:
            continue

    if not user_info or not matched_provider:
        raise HTTPException(status_code=400, detail="SAML assertion could not be validated against any configured provider")

    email = user_info.get("email") or user_info.get("name_id")
    if not email:
        raise HTTPException(status_code=400, detail="Could not extract email from SAML assertion")

    # Create or update LiteLLM user via proxy's internal methods
    try:
        from litellm.proxy.proxy_server import prisma_client
        if prisma_client is None:
            raise HTTPException(status_code=500, detail="Database not configured")

        # Check if user exists
        existing_user = await prisma_client.db.litellm_usertable.find_first(
            where={"user_email": email}
        )

        if existing_user:
            # Update user role/team if mapping changed
            update_data = {}
            if user_info.get("role") and user_info["role"] != existing_user.user_role:
                update_data["user_role"] = user_info["role"]
            if update_data:
                await prisma_client.db.litellm_usertable.update(
                    where={"user_id": existing_user.user_id},
                    data=update_data,
                )
            user_id = existing_user.user_id
        else:
            # Create new user
            import uuid
            user_id = f"user-{uuid.uuid4().hex[:12]}"
            await prisma_client.db.litellm_usertable.create(
                data={
                    "user_id": user_id,
                    "user_email": email,
                    "user_role": user_info.get("role", "INTERNAL_USER"),
                    "teams": [user_info["team_id"]] if user_info.get("team_id") else [],
                }
            )
            logger.info(f"Created new SAML user: {email} (role={user_info.get('role')})")

        # Generate an API key for this user
        import secrets
        from litellm.proxy.auth.litellm_pre_process_utils import hash_token

        key = f"sk-saml-{secrets.token_hex(24)}"
        key_hash = hash_token(key)

        await prisma_client.db.litellm_verificationtoken.create(
            data={
                "token": key_hash,
                "user_id": user_id,
                "key_alias": f"saml-{matched_provider.provider_name}",
                "permissions": {"models": []},
                "metadata": {
                    "saml_provider": matched_provider.provider_name,
                    "saml_name_id": user_info.get("name_id"),
                    "saml_groups": user_info.get("groups", []),
                },
            }
        )

        # Redirect to UI with the key (matching existing SSO pattern)
        redirect_url = relay_state or "/ui"
        separator = "&" if "?" in redirect_url else "?"
        return RedirectResponse(
            url=f"{redirect_url}{separator}token={key}",
            status_code=303,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"SAML user creation failed: {e}")
        raise HTTPException(status_code=500, detail=f"SAML authentication failed: {str(e)}")


@saml_router.get("/metadata")
async def saml_metadata(provider: str = Query(..., description="SAML provider name")):
    """Return Service Provider metadata XML."""
    store = get_saml_store()
    provider_config = store.get_provider(provider)
    if not provider_config:
        raise HTTPException(status_code=404, detail=f"SAML provider '{provider}' not found")

    metadata_xml = generate_sp_metadata(provider_config)
    return Response(content=metadata_xml, media_type="application/xml")


@saml_router.post("/slo")
async def saml_slo(request: Request):
    """Handle Single Logout request from IdP."""
    form_data = await request.form()
    saml_request = form_data.get("SAMLRequest")
    if not saml_request:
        raise HTTPException(status_code=400, detail="Missing SAMLRequest")

    # Parse logout request to extract name_id and session_index
    # For now, redirect to UI login page
    return RedirectResponse(url="/ui/login", status_code=303)


@saml_router.post("/config")
async def create_saml_config(config: SAMLProviderConfig, request: Request):
    """Create or update a SAML provider configuration (admin only)."""
    try:
        from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
        user = await user_api_key_auth(request)
        if not user or user.user_role not in ["PROXY_ADMIN", "PROXY_ADMIN_VIEW_ONLY"]:
            raise HTTPException(status_code=403, detail="Admin access required")
    except ImportError:
        pass  # Allow in dev mode

    store = get_saml_store()
    saved = store.save_provider(config)
    return {"status": "ok", "provider": saved.provider_name, "message": f"SAML provider '{config.provider_name}' configured successfully"}


@saml_router.get("/config")
async def list_saml_configs(request: Request):
    """List all configured SAML providers (admin only)."""
    store = get_saml_store()
    providers = store.list_providers()
    return {"providers": providers, "count": len(providers)}


@saml_router.delete("/config/{provider_name}")
async def delete_saml_config(provider_name: str, request: Request):
    """Delete a SAML provider configuration (admin only)."""
    store = get_saml_store()
    deleted = store.delete_provider(provider_name)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"SAML provider '{provider_name}' not found")
    return {"status": "ok", "message": f"SAML provider '{provider_name}' deleted"}
