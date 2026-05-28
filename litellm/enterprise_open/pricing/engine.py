"""
Custom pricing engine — applies markup rules and calculates costs.

Rules are matched by model name pattern, scoped to org/team,
and resolved by priority (higher wins).
"""

import json
import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from litellm.enterprise_open.config import PricingRuleConfig

logger = logging.getLogger(__name__)

DEFAULT_PRICING_PATH = os.path.expanduser("~/.litellm/pricing_rules.json")


class PricingEngine:
    """Evaluates pricing rules and calculates custom costs."""

    def __init__(self, rules_path: str = DEFAULT_PRICING_PATH):
        self._path = Path(rules_path)
        self._rules: List[PricingRuleConfig] = []
        self._load()

    def _load(self):
        if self._path.exists():
            try:
                with open(self._path, "r") as f:
                    data = json.load(f)
                self._rules = [PricingRuleConfig(**r) for r in data]
                logger.info(f"Loaded {len(self._rules)} pricing rule(s)")
            except Exception as e:
                logger.error(f"Failed to load pricing rules: {e}")
                self._rules = []

    def _save(self):
        self._path.parent.mkdir(parents=True, exist_ok=True)
        data = [r.model_dump() for r in self._rules]
        with open(self._path, "w") as f:
            json.dump(data, f, indent=2, default=str)

    def add_rule(self, rule: PricingRuleConfig) -> PricingRuleConfig:
        """Add a pricing rule."""
        self._rules.append(rule)
        # Sort by priority descending
        self._rules.sort(key=lambda r: r.priority, reverse=True)
        self._save()
        logger.info(f"Added pricing rule: {rule.name} (priority={rule.priority})")
        return rule

    def get_rule(self, rule_name: str) -> Optional[PricingRuleConfig]:
        for r in self._rules:
            if r.name == rule_name:
                return r
        return None

    def list_rules(self) -> List[Dict]:
        return [r.model_dump() for r in self._rules]

    def update_rule(self, rule_name: str, **kwargs) -> Optional[PricingRuleConfig]:
        for i, r in enumerate(self._rules):
            if r.name == rule_name:
                for key, value in kwargs.items():
                    if hasattr(r, key):
                        setattr(r, key, value)
                self._rules.sort(key=lambda r: r.priority, reverse=True)
                self._save()
                return r
        return None

    def delete_rule(self, rule_name: str) -> bool:
        before = len(self._rules)
        self._rules = [r for r in self._rules if r.name != rule_name]
        if len(self._rules) < before:
            self._save()
            return True
        return False

    def find_matching_rule(
        self,
        model: str,
        organization_id: Optional[str] = None,
        team_id: Optional[str] = None,
    ) -> Optional[PricingRuleConfig]:
        """
        Find the best matching pricing rule for a given request.
        Rules are already sorted by priority (desc).
        Checks: model pattern match, org/team scope, effective dates.
        """
        now = datetime.now(timezone.utc)

        for rule in self._rules:
            # Check effective dates
            if rule.effective_from:
                effective_from = datetime.fromisoformat(
                    rule.effective_from.replace("Z", "+00:00")
                )
                if now < effective_from:
                    continue
            if rule.effective_to:
                effective_to = datetime.fromisoformat(
                    rule.effective_to.replace("Z", "+00:00")
                )
                if now > effective_to:
                    continue

            # Check model pattern
            try:
                if not re.match(rule.model_pattern, model, re.IGNORECASE):
                    continue
            except re.error:
                # Invalid regex, try simple wildcard
                pattern = rule.model_pattern.replace("*", ".*")
                if not re.match(pattern, model, re.IGNORECASE):
                    continue

            # Check scope
            if rule.organization_id and rule.organization_id != organization_id:
                continue
            if rule.team_id and rule.team_id != team_id:
                continue

            return rule

        return None

    def calculate_cost(
        self,
        model: str,
        base_cost: float,
        input_tokens: int = 0,
        output_tokens: int = 0,
        organization_id: Optional[str] = None,
        team_id: Optional[str] = None,
    ) -> Dict:
        """
        Calculate the final cost by applying pricing rules.

        Returns:
            Dict with base_cost, markup, custom_cost, total, applied_rule
        """
        rule = self.find_matching_rule(
            model, organization_id=organization_id, team_id=team_id
        )

        if not rule:
            return {
                "model": model,
                "base_cost": base_cost,
                "markup": 0.0,
                "custom_cost": 0.0,
                "total": base_cost,
                "applied_rule": None,
            }

        markup = 0.0
        custom_cost = 0.0

        # Apply percentage markup
        if rule.markup_percent is not None:
            markup = base_cost * (rule.markup_percent / 100.0)

        # Apply flat fee
        flat_fee = rule.flat_fee_per_request or 0.0

        # Apply custom per-token pricing
        if rule.custom_price_input_per_1k is not None and input_tokens > 0:
            custom_cost += (input_tokens / 1000.0) * rule.custom_price_input_per_1k
        if rule.custom_price_output_per_1k is not None and output_tokens > 0:
            custom_cost += (output_tokens / 1000.0) * rule.custom_price_output_per_1k

        # If custom token pricing is set, use that instead of base cost
        if custom_cost > 0:
            total = custom_cost + markup + flat_fee
        else:
            total = base_cost + markup + flat_fee

        return {
            "model": model,
            "base_cost": base_cost,
            "markup": round(markup, 6),
            "flat_fee": round(flat_fee, 6),
            "custom_cost": round(custom_cost, 6),
            "total": round(total, 6),
            "applied_rule": rule.name,
        }


# Singleton
_engine: Optional[PricingEngine] = None


def get_pricing_engine() -> PricingEngine:
    global _engine
    if _engine is None:
        _engine = PricingEngine()
    return _engine
