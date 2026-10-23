"""cache.clear() and the turn store's deletes leave no turn text on disk.

`clear_all` unlinked every *.db in the cache folder, turns.db included, while
the turn store still had it open: turns.db-wal and turns.db-shm stayed with
the text in them, and the agent wrote on into an unlinked file. A drop() or
clear() only marked rows deleted, so the text stayed in free pages too.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from maximem_synap.cache.manager import CacheManager
from maximem_synap.cache.turn_store import TurnStore
from maximem_synap.sdk import CacheInterface

SECRET = "my passport number is X1234567Q"


def _bytes_left(folder: Path) -> bytes:
    return b"".join(p.read_bytes() for p in folder.rglob("*") if p.is_file())


@pytest.fixture
def setup(tmp_path):
    manager = CacheManager(client_id="c1", instance_id="inst_1", storage_path=str(tmp_path))
    manager.base_path.mkdir(parents=True, exist_ok=True)
    store = TurnStore(path=manager.base_path / "turns.db")
    store.add("conv-1", "user", SECRET)
    assert SECRET.encode() in _bytes_left(manager.base_path), "control: the text is on disk"
    sdk = SimpleNamespace(_turn_store=store, _cache_manager=manager)
    return sdk, manager, store


def test_clear_leaves_no_turn_text_and_the_store_keeps_working(setup):
    sdk, manager, store = setup
    CacheInterface(sdk).clear()
    assert SECRET.encode() not in _bytes_left(manager.base_path)
    assert store.count("conv-1") == 0
    store.add("conv-2", "user", "still writes to a file that exists")
    assert (manager.base_path / "turns.db").exists()
    assert store.count("conv-2") == 1


def test_drop_overwrites_the_conversation(setup):
    _, manager, store = setup
    store.add("conv-keep", "user", "unrelated turn")
    store.drop("conv-1")
    assert SECRET.encode() not in _bytes_left(manager.base_path)
    assert store.count("conv-keep") == 1


def test_clear_removes_response_cache_sidecars(setup):
    sdk, manager, _ = setup
    for name in ("users.db", "users.db-wal", "users.db-shm", "stray.db-journal"):
        (manager.base_path / name).write_bytes(b"cached " + SECRET.encode())
    CacheInterface(sdk).clear()
    assert sorted(p.name for p in manager.base_path.iterdir() if "turns.db" not in p.name) == []


def test_clear_user_removes_the_file_and_its_sidecars(tmp_path):
    manager = CacheManager(client_id="c1", instance_id="inst_1", storage_path=str(tmp_path))
    users = manager.base_path / "users"
    users.mkdir(parents=True)
    for name in ("u1.db", "u1.db-wal", "u1.db-shm"):
        (users / name).write_bytes(SECRET.encode())
    manager.clear_user("u1")
    assert list(users.iterdir()) == []
