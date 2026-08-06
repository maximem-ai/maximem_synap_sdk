"""Regression: the client-side CacheManager key must namespace by instance_id.

Customer- and client-scoped cache files are shared by every Instance under the
same client+customer. If the cache key omits instance_id, a second Instance
issuing the same query reads the first Instance's cached, visibility-filtered
context straight from the local cache — silently defeating server-side
cross-instance visibility. These tests pin the key to include instance_id.
"""
from __future__ import annotations

from maximem_synap.cache.manager import CacheManager, CacheScope


def _key(mgr: CacheManager) -> str:
    return mgr._build_key(
        CacheScope.CUSTOMER, "cust-1", "facts", mgr._hash_query(["same-query"])
    )


def test_distinct_instances_get_distinct_keys():
    a = CacheManager(client_id="cli_x", enabled=False, instance_id="inst_a")
    b = CacheManager(client_id="cli_x", enabled=False, instance_id="inst_b")
    # Same client + customer + type + query, different Instance -> different key.
    assert _key(a) != _key(b)
    assert "inst_a" in _key(a)
    assert "inst_b" in _key(b)


def test_same_instance_keeps_stable_key():
    # Caching still works within one Instance (identical inputs -> identical key).
    a1 = CacheManager(client_id="cli_x", enabled=False, instance_id="inst_a")
    a2 = CacheManager(client_id="cli_x", enabled=False, instance_id="inst_a")
    assert _key(a1) == _key(a2)


def test_missing_instance_id_uses_stable_placeholder():
    m = CacheManager(client_id="cli_x", enabled=False)  # no instance_id
    assert ":_:" in _key(m)  # empty instance -> "_" placeholder, still a valid key
