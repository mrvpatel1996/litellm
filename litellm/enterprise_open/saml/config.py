"""
SAML provider configuration store.

Stores SAML IdP configurations as JSON (file-based by default, swappable for DB).
"""

import json
import os
import logging
from pathlib import Path
from typing import Dict, List, Optional

from litellm.enterprise_open.config import SAMLProviderConfig

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = os.path.expanduser("~/.litellm/saml_providers.json")


class SAMLProviderStore:
    """Manages SAML IdP provider configurations."""

    def __init__(self, config_path: str = DEFAULT_CONFIG_PATH):
        self._config_path = Path(config_path)
        self._providers: Dict[str, SAMLProviderConfig] = {}
        self._load()

    def _load(self):
        """Load providers from disk."""
        if self._config_path.exists():
            try:
                with open(self._config_path, "r") as f:
                    data = json.load(f)
                self._providers = {
                    name: SAMLProviderConfig(**cfg) for name, cfg in data.items()
                }
                logger.info(f"Loaded {len(self._providers)} SAML provider(s)")
            except (json.JSONDecodeError, Exception) as e:
                logger.error(f"Failed to load SAML providers: {e}")
                self._providers = {}
        else:
            self._providers = {}

    def _save(self):
        """Persist providers to disk."""
        self._config_path.parent.mkdir(parents=True, exist_ok=True)
        data = {name: cfg.model_dump() for name, cfg in self._providers.items()}
        with open(self._config_path, "w") as f:
            json.dump(data, f, indent=2, default=str)

    def get_provider(self, name: str) -> Optional[SAMLProviderConfig]:
        """Get a SAML provider by name."""
        return self._providers.get(name)

    def list_providers(self) -> List[str]:
        """List all configured provider names."""
        return list(self._providers.keys())

    def get_all_providers(self) -> Dict[str, SAMLProviderConfig]:
        """Get all providers."""
        return dict(self._providers)

    def save_provider(self, config: SAMLProviderConfig) -> SAMLProviderConfig:
        """Create or update a SAML provider configuration."""
        self._providers[config.provider_name] = config
        self._save()
        logger.info(f"Saved SAML provider: {config.provider_name}")
        return config

    def delete_provider(self, name: str) -> bool:
        """Delete a SAML provider. Returns True if found and deleted."""
        if name in self._providers:
            del self._providers[name]
            self._save()
            logger.info(f"Deleted SAML provider: {name}")
            return True
        return False


# Singleton instance
_store: Optional[SAMLProviderStore] = None


def get_saml_store() -> SAMLProviderStore:
    """Get the global SAML provider store."""
    global _store
    if _store is None:
        _store = SAMLProviderStore()
    return _store
