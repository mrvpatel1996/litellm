"""Regression tests for the enterprise audit log service field mapping.

The LiteLLM_AuditLog Prisma model has `updated_at` (not `created_at`) and
`updated_values` (not `after_value`). Sorting/filtering on the wrong field name
caused a Prisma FieldNotFoundError -> HTTP 500 on /enterprise/audit/logs.
"""

from litellm.enterprise_open.audit.service import (
    _AUDIT_SERIALIZE_FIELDS,
    _AUDIT_TIMESTAMP_FIELD,
    _resolve_sort_field,
)


def test_timestamp_field_is_updated_at():
    assert _AUDIT_TIMESTAMP_FIELD == "updated_at"


def test_legacy_created_at_maps_to_updated_at():
    # The old default must not blow up — it maps to the real column.
    assert _resolve_sort_field("created_at") == "updated_at"


def test_empty_sort_maps_to_updated_at():
    assert _resolve_sort_field("") == "updated_at"


def test_unknown_sort_field_falls_back():
    # Prevents Prisma 500s on arbitrary/injected column names.
    assert _resolve_sort_field("'; DROP TABLE--") == "updated_at"


def test_valid_sort_field_preserved():
    assert _resolve_sort_field("action") == "action"
    assert _resolve_sort_field("table_name") == "table_name"


def test_serialize_fields_match_real_schema():
    # No phantom columns that don't exist on the model.
    assert "created_at" not in _AUDIT_SERIALIZE_FIELDS
    assert "after_value" not in _AUDIT_SERIALIZE_FIELDS
    assert "updated_at" in _AUDIT_SERIALIZE_FIELDS
    assert "updated_values" in _AUDIT_SERIALIZE_FIELDS
