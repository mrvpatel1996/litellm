"""
SAML 2.0 authentication handler.

Handles SAML AuthnRequest generation, assertion parsing, and attribute mapping.
Uses stdlib XML parsing with optional python3-saml enhancement.
"""

import base64
import hashlib
import logging
import uuid
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from urllib.parse import urlencode, quote_plus

from litellm.enterprise_open.config import SAMLProviderConfig

logger = logging.getLogger(__name__)

# SAML XML namespaces
SAML_NAMESPACES = {
    "samlp": "urn:oasis:names:tc:SAML:2.0:protocol",
    "saml": "urn:oasis:names:tc:SAML:2.0:assertion",
    "ds": "http://www.w3.org/2000/09/xmldsig#",
}

# Register namespaces to avoid ns0/ns1 prefixes
for prefix, uri in SAML_NAMESPACES.items():
    ET.register_namespace(prefix, uri)


def _generate_id() -> str:
    """Generate a random SAML ID."""
    return f"_{uuid.uuid4().hex}"


def _get_timestamp() -> str:
    """Get current UTC timestamp in ISO format."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def initiate_saml_login(provider: SAMLProviderConfig, relay_state: Optional[str] = None) -> str:
    """
    Generate a SAML AuthnRequest and return the IdP redirect URL.

    Args:
        provider: SAML provider configuration
        relay_state: Optional relay state to pass through

    Returns:
        IdP redirect URL with SAMLRequest parameter
    """
    request_id = _generate_id()
    issue_instant = _get_timestamp()

    # Build SAML AuthnRequest XML
    authn_request = f"""<samlp:AuthnRequest
        xmlns:samlp="urn:oasis:names:tc:SAML:2.0:protocol"
        xmlns:saml="urn:oasis:names:tc:SAML:2.0:assertion"
        ID="{request_id}"
        Version="2.0"
        ProtocolBinding="urn:oasis:names:tc:SAML:2.0:bindings:HTTP-POST"
        AssertionConsumerServiceURL="{provider.acs_url}"
        IssueInstant="{issue_instant}">
        <saml:Issuer>{provider.entity_id}</saml:Issuer>
        <samlp:NameIDPolicy Format="urn:oasis:names:tc:SAML:1.1:nameid-format:emailAddress" AllowCreate="true"/>
    </samlp:AuthnRequest>"""

    # Deflate + Base64 encode (SAML binding spec)
    import zlib
    compressed = zlib.compress(authn_request.encode("utf-8"))[2:-4]  # Strip zlib header/trailer
    encoded = base64.b64encode(compressed).decode("utf-8")

    # Build redirect URL
    params = {"SAMLRequest": encoded}
    if relay_state:
        params["RelayState"] = relay_state

    separator = "&" if "?" in provider.sso_url else "?"
    redirect_url = f"{provider.sso_url}{separator}{urlencode(params)}"
    logger.info(f"SAML AuthnRequest generated for provider: {provider.provider_name}")
    return redirect_url


def process_saml_response(
    saml_response_b64: str, provider: SAMLProviderConfig
) -> Dict[str, Any]:
    """
    Parse a SAML Response from the IdP and extract user attributes.

    Args:
        saml_response_b64: Base64-encoded SAML response
        provider: SAML provider configuration

    Returns:
        Dict with keys: name_id, email, attributes, session_index, conditions
    """
    try:
        # Decode
        saml_xml = base64.b64decode(saml_response_b64).decode("utf-8")
        root = ET.fromstring(saml_xml)
    except Exception as e:
        raise ValueError(f"Failed to parse SAML response: {e}")

    # Extract assertion
    assertion = root.find(".//saml:Assertion", SAML_NAMESPACES)
    if assertion is None:
        # Try without namespace
        assertion = root.find(".//{urn:oasis:names:tc:SAML:2.0:assertion}Assertion")
    if assertion is None:
        raise ValueError("No SAML assertion found in response")

    # Check conditions
    conditions = assertion.find("saml:Conditions", SAML_NAMESPACES)
    if conditions is None:
        conditions = assertion.find("{urn:oasis:names:tc:SAML:2.0:assertion}Conditions")

    not_on_or_after = None
    not_before = None
    if conditions is not None:
        not_on_or_after = conditions.get("NotOnOrAfter")
        not_before = conditions.get("NotBefore")

        now = datetime.now(timezone.utc)
        if not_on_or_after:
            expiry = datetime.fromisoformat(not_on_or_after.replace("Z", "+00:00"))
            if now >= expiry:
                raise ValueError(f"SAML assertion expired at {not_on_or_after}")
        if not_before:
            valid_from = datetime.fromisoformat(not_before.replace("Z", "+00:00"))
            if now < valid_from:
                raise ValueError(f"SAML assertion not yet valid (valid from {not_before})")

    # Extract NameID
    subject = assertion.find("saml:Subject", SAML_NAMESPACES)
    if subject is None:
        subject = assertion.find("{urn:oasis:names:tc:SAML:2.0:assertion}Subject")

    name_id = None
    if subject is not None:
        name_id_elem = subject.find("saml:NameID", SAML_NAMESPACES)
        if name_id_elem is None:
            name_id_elem = subject.find("{urn:oasis:names:tc:SAML:2.0:assertion}NameID")
        if name_id_elem is not None:
            name_id = name_id_elem.text

    # Extract session index
    session_index = None
    authn_statement = assertion.find("saml:AuthnStatement", SAML_NAMESPACES)
    if authn_statement is None:
        authn_statement = assertion.find("{urn:oasis:names:tc:SAML:2.0:assertion}AuthnStatement")
    if authn_statement is not None:
        session_index = authn_statement.get("SessionIndex")

    # Extract attributes
    attributes = {}
    attribute_statement = assertion.find("saml:AttributeStatement", SAML_NAMESPACES)
    if attribute_statement is None:
        attribute_statement = assertion.find("{urn:oasis:names:tc:SAML:2.0:assertion}AttributeStatement")

    if attribute_statement is not None:
        for attr in attribute_statement:
            # Handle both namespaced and non-namespaced
            tag = attr.tag.split("}")[-1] if "}" in attr.tag else attr.tag
            if tag == "Attribute":
                attr_name = attr.get("Name", attr.get("FriendlyName", ""))
                values = []
                for val in attr:
                    val_tag = val.tag.split("}")[-1] if "}" in val.tag else val.tag
                    if val_tag == "AttributeValue" and val.text:
                        values.append(val.text)
                if attr_name and values:
                    attributes[attr_name] = values if len(values) > 1 else values[0]

    # Map attributes to user fields
    user_info = map_attributes_to_user(attributes, provider)
    user_info["name_id"] = name_id
    user_info["session_index"] = session_index
    user_info["conditions"] = {
        "not_on_or_after": not_on_or_after,
        "not_before": not_before,
    }
    user_info["raw_attributes"] = attributes

    logger.info(f"SAML assertion parsed for user: {user_info.get('email', name_id)}")
    return user_info


def map_attributes_to_user(
    saml_attributes: Dict[str, Any], provider: SAMLProviderConfig
) -> Dict[str, Any]:
    """
    Map SAML assertion attributes to LiteLLM user fields using provider's mapping config.

    Returns:
        Dict with: email, first_name, last_name, role, team_id, groups
    """
    mapping = provider.attribute_mapping

    def _get_mapped_attr(mapping_key: str) -> Optional[str]:
        saml_attr_name = mapping.get(mapping_key)
        if not saml_attr_name:
            return None
        val = saml_attributes.get(saml_attr_name)
        if isinstance(val, list):
            return val[0] if val else None
        return val

    email = _get_mapped_attr("email") or saml_attributes.get("email")
    first_name = _get_mapped_attr("firstName") or saml_attributes.get("firstName")
    last_name = _get_mapped_attr("lastName") or saml_attributes.get("lastName")

    # Extract groups
    groups = saml_attributes.get("groups", saml_attributes.get("Group", []))
    if isinstance(groups, str):
        groups = [groups]

    # Resolve role from group mapping
    role = provider.default_role
    if provider.role_mapping and groups:
        for group in groups:
            if group in provider.role_mapping:
                role = provider.role_mapping[group]
                break

    # Resolve team from group mapping
    team_id = None
    if provider.team_mapping and groups:
        for group in groups:
            if group in provider.team_mapping:
                team_id = provider.team_mapping[group]
                break

    return {
        "email": email,
        "first_name": first_name,
        "last_name": last_name,
        "role": role,
        "team_id": team_id,
        "groups": groups,
    }


def generate_sp_metadata(provider: SAMLProviderConfig) -> str:
    """
    Generate Service Provider metadata XML for this LiteLLM instance.

    Returns:
        SP metadata XML string
    """
    entity_id = provider.entity_id
    acs_url = provider.acs_url
    sp_x509 = provider.sp_x509cert or ""

    cert_block = ""
    if sp_x509:
        cert_block = (
            '<md:KeyDescriptor use="signing">'
            '<ds:KeyInfo xmlns:ds="http://www.w3.org/2000/09/xmldsig#">'
            "<ds:X509Data><ds:X509Certificate>"
            + sp_x509
            + "</ds:X509Certificate></ds:X509Data></ds:KeyInfo></md:KeyDescriptor>"
        )

    metadata = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<md:EntityDescriptor xmlns:md="urn:oasis:names:tc:SAML:2.0:metadata"\n'
        f'                     entityID="{entity_id}">\n'
        '    <md:SPSSODescriptor protocolSupportEnumeration="urn:oasis:names:tc:SAML:2.0:protocol">\n'
        "        <md:NameIDFormat>urn:oasis:names:tc:SAML:1.1:nameid-format:emailAddress</md:NameIDFormat>\n"
        '        <md:AssertionConsumerService Binding="urn:oasis:names:tc:SAML:2.0:bindings:HTTP-POST"\n'
        f'                                     Location="{acs_url}"\n'
        '                                     index="1" isDefault="true"/>\n'
        f"        {cert_block}\n"
        "    </md:SPSSODescriptor>\n"
        "    <md:Organization>\n"
        '        <md:OrganizationName xml:lang="en">LiteLLM Enterprise</md:OrganizationName>\n'
        '        <md:OrganizationDisplayName xml:lang="en">LiteLLM Enterprise Proxy</md:OrganizationDisplayName>\n'
        "    </md:Organization>\n"
        "</md:EntityDescriptor>"
    )

    return metadata


def generate_logout_request(
    provider: SAMLProviderConfig,
    name_id: str,
    session_index: Optional[str] = None,
) -> str:
    """
    Generate a SAML LogoutRequest.

    Returns:
        Redirect URL for IdP logout
    """
    request_id = _generate_id()
    issue_instant = _get_timestamp()

    logout_request = f"""<samlp:LogoutRequest
        xmlns:samlp="urn:oasis:names:tc:SAML:2.0:protocol"
        xmlns:saml="urn:oasis:names:tc:SAML:2.0:assertion"
        ID="{request_id}"
        Version="2.0"
        IssueInstant="{issue_instant}">
        <saml:Issuer>{provider.entity_id}</saml:Issuer>
        <saml:NameID Format="urn:oasis:names:tc:SAML:1.1:nameid-format:emailAddress">{name_id}</saml:NameID>
        {"<samlp:SessionIndex>" + session_index + "</samlp:SessionIndex>" if session_index else ""}
    </samlp:LogoutRequest>"""

    import zlib
    compressed = zlib.compress(logout_request.encode("utf-8"))[2:-4]
    encoded = base64.b64encode(compressed).decode("utf-8")

    if provider.slo_url:
        separator = "&" if "?" in provider.slo_url else "?"
        return f"{provider.slo_url}{separator}SAMLRequest={quote_plus(encoded)}"
    return ""
