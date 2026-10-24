#!/usr/bin/env python3
"""Delete the test users a coding agent created while wiring Synap, then check.

Run it near the end of onboarding, BEFORE the setup key is signed out (the SDK
needs the runtime key from .env). It calls `sdk.data.delete(scope="user")` once
per test user (maximem-synap 0.6.9 or newer; older SDKs make it exit 2 and say
to upgrade). The server removes the user's memories, conversations, summaries,
profile and index entries inside this instance. The script waits until the
delete reaches a final status, then fetches for the user once more to confirm
nothing comes back.

THE ONE RULE: it only ever touches user ids that start with `test_user_`. Any
other id makes the whole run refuse (exit 2) before a single call is made.

Usage:
    python scripts/cleanup_test_data.py --user-id test_user_verify_a1b2c3d4 \\
        --user-id test_user_smoke_9f8e7d6c --dry-run
    python scripts/cleanup_test_data.py --user-id test_user_verify_a1b2c3d4 \\
        --user-id test_user_smoke_9f8e7d6c [--customer-id acme]
    python scripts/cleanup_test_data.py --suggest-id verify     # prints a new id

Options:
    --user-id ID         a test user id (repeatable, at least one). Must start
                         with test_user_.
    --customer-id ID     B2B only: the customer the test users were written under.
    --dry-run            ask the server for the counts it WOULD remove
                         (dry_run=True) and change nothing.
    --timeout SECONDS    how long to wait for a final status, default 120.
    --json PATH          also write the full report as JSON to PATH.
    --suggest-id PURPOSE print a new test user id and exit.

Removed on purpose (the delete API replaces list-and-delete): --memory-id,
--ingestion-id (deleting single memories is not what test cleanup needs, and a
user delete covers them), --forget-sweep (the API reaches profile text and
chunks itself), --query, --passes, --pause (no listing rounds any more). Passing
one of them exits 2 with a message. The listing is kept only for the final
check, and the poll is kept as the wait for a final status.

Exit codes: 0 clean (or dry run), 1 something remains, the status was partial,
the wait timed out, or a call failed, 2 refused or cannot start (bad id, no key,
SDK older than 0.6.9, nothing to do).

The script never prints the API key and never prints memory text, only ids,
counts and store names.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import secrets
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

TEST_PREFIX = "test_user_"
# prefix + at least one more character, from a safe alphabet.
_TEST_ID_RE = re.compile(r"^test_user_[A-Za-z0-9][A-Za-z0-9_.-]*$")

MIN_SDK = "0.6.9"
UPGRADE_MSG = (
    f"[ERROR] This maximem-synap SDK has no sdk.data (delete by scope). Upgrade it to "
    f"{MIN_SDK} or newer (pip install -U maximem-synap) and run this again. "
    "Nothing was called."
)

TERMINAL = ("complete", "partial", "failed")
POLL_INTERVAL = 3.0

CHECK_QUERIES = [
    "everything about this user",
    "facts and personal details",
    "preferences and likes",
]

# Flags of the old list-and-delete path. Refused with a clear line, not silently ignored.
REMOVED_FLAGS = {
    "memory_id": "--memory-id", "ingestion_id": "--ingestion-id",
    "forget_sweep": "--forget-sweep", "query": "--query", "passes": "--passes",
    "pause": "--pause",
}

# What a user-scope delete does NOT remove (design: delete_scope_data_design.md, sections 4, 13, 14).
# Printed on every run so the final report can copy it.
RESIDUE = [
    "LLM call logs: prompt and answer text (llm_call_bodies) stay up to 14 days, call metadata "
    "(llm_calls) up to 180 days",
    "Billing and usage counters, and the audit record of the delete (it holds no content)",
    "Data in any other instance of this client, and customer-level or client-level memories "
    "(a user delete never touches them)",
    "Ingestion still queued for the user can land after the delete: stop writing and run this again",
    "Files on this machine such as turns.db and the stream outbox, if the app uses them "
    "(delete the file with the agent stopped)",
    "Backups taken before the delete, until they expire",
]


def make_test_user_id(purpose: str) -> str:
    """`test_user_<purpose>_<8 random chars>`; the one id shape a test may use."""
    slug = re.sub(r"[^a-z0-9]+", "_", (purpose or "").lower()).strip("_")[:24] or "check"
    return f"{TEST_PREFIX}{slug}_{secrets.token_hex(4)}"


def is_test_user_id(value: Any) -> bool:
    return isinstance(value, str) and bool(_TEST_ID_RE.match(value))


def _load_dotenv(path: str = ".env") -> None:
    """Read KEY=VALUE lines from ./.env without overriding the real environment."""
    p = Path(path)
    if not p.is_file():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export "):]
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def _sleep(seconds: float):
    return asyncio.sleep(seconds)


def _err(e: BaseException) -> str:
    return f"{type(e).__name__}: {str(e)[:160]}"


def sdk_has_delete(sdk: Any) -> bool:
    data = getattr(sdk, "data", None)
    return data is not None and callable(getattr(data, "delete", None))


# ------------------------------------------------------------- final check

def _iter_items(ctx: Any):
    """Yield (id, scope, category) for every item of a fetch response."""
    ranked = list(getattr(ctx, "items", None) or [])
    if ranked:
        for it in ranked:
            md = getattr(it, "metadata", None) or {}
            yield str(it.id), md.get("scope"), getattr(it, "category", None) or "item"
        return
    for name in ("facts", "preferences", "episodes", "emotions", "temporal_events"):
        for it in getattr(ctx, name, None) or []:
            md = getattr(it, "metadata", None) or {}
            yield str(it.id), md.get("scope"), name


async def _fetch(sdk: Any, user_id: str, customer_id: Optional[str], query: str, limit: int):
    kwargs: Dict[str, Any] = {"user_id": user_id, "search_query": [query], "max_results": limit}
    if customer_id:
        kwargs["customer_id"] = customer_id
    try:
        return await sdk.user.context.fetch(response_shape="ranked", **kwargs)
    except TypeError:
        return await sdk.user.context.fetch(**kwargs)


async def still_returned(sdk, user_id, customer_id, limit) -> Dict[str, Any]:
    """Fetch for the user; return the ids of items with scope "user" that still come back.

    Customer- and client-level items in a user fetch belong to other people and
    are not counted.
    """
    ids: set = set()
    errors: List[str] = []
    for q in CHECK_QUERIES:
        try:
            ctx = await _fetch(sdk, user_id, customer_id, q, limit)
        except Exception as e:  # noqa: BLE001
            errors.append(_err(e))
            continue
        for mid, scope, category in _iter_items(ctx):
            if scope == "user" and category != "excerpts":
                ids.add(mid)
    return {"ids": sorted(ids), "errors": errors}


# ------------------------------------------------------------------ deleting

def _public(result: Any) -> Dict[str, Any]:
    """The parts of a DeleteDataResult that are safe to print: ids, counts, store names."""
    counts = {str(k): v for k, v in (getattr(result, "counts", None) or {}).items()
              if isinstance(v, (int, float))}
    stores = {str(k): str(v)[:120] for k, v in (getattr(result, "stores", None) or {}).items()}
    return {
        "deletion_id": getattr(result, "deletion_id", None),
        "status": getattr(result, "status", "unknown"),
        "found": bool(getattr(result, "found", False)),
        "counts": counts,
        "stores": stores,
        "still_present_in": [str(x) for x in (getattr(result, "still_present_in", None) or [])],
        "fully_removed": bool(getattr(result, "fully_removed", False)),
        "kept": [str(x)[:160] for x in (getattr(result, "kept", None) or [])],
        "continue": bool(getattr(result, "continue_", False)),
    }


async def delete_user(sdk, user_id: str, customer_id: Optional[str], dry_run: bool,
                      timeout: float) -> Dict[str, Any]:
    rep: Dict[str, Any] = {"result": None, "polls": 0, "timed_out": False, "errors": [],
                           "still_returned": [], "checked": False}
    kwargs: Dict[str, Any] = {"scope": "user", "user_id": user_id}
    if customer_id:
        kwargs["customer_id"] = customer_id
    if dry_run:
        kwargs["dry_run"] = True
    else:
        kwargs["confirm"] = user_id
    try:
        res = await sdk.data.delete(**kwargs)
    except Exception as e:  # noqa: BLE001
        rep["errors"].append(_err(e))
        return rep
    rep["result"] = _public(res)
    if dry_run:
        return rep

    # Wait for a final status. The SDK already repeats the call while the server
    # says "continue"; if it is still unfinished, call again (the same request is
    # safe to repeat) or read the status row.
    max_polls = max(1, int(timeout / POLL_INTERVAL))
    while rep["result"]["status"] not in TERMINAL or rep["result"]["continue"]:
        if rep["polls"] >= max_polls:
            rep["timed_out"] = True
            break
        if rep["polls"]:
            await _sleep(POLL_INTERVAL)
        rep["polls"] += 1
        try:
            if rep["result"]["continue"] or not rep["result"]["deletion_id"]:
                res = await sdk.data.delete(**kwargs)
            else:
                res = await sdk.data.status(rep["result"]["deletion_id"])
        except Exception as e:  # noqa: BLE001
            rep["errors"].append(_err(e))
            break
        pub = _public(res)
        if not pub["deletion_id"]:
            pub["deletion_id"] = rep["result"]["deletion_id"]
        rep["result"] = pub
    if rep["timed_out"] or rep["errors"]:
        return rep

    # Final check: a fresh fetch must return nothing for this user.
    try:
        sdk.cache.clear_user(user_id)
    except Exception:  # noqa: BLE001
        pass
    after = await still_returned(sdk, user_id, customer_id, 100)
    rep["checked"] = not after["errors"]
    rep["errors"].extend(after["errors"])
    rep["still_returned"] = after["ids"]
    return rep


async def run(sdk: Any, args: argparse.Namespace) -> Dict[str, Any]:
    summary: Dict[str, Any] = {"dry_run": bool(args.dry_run), "users": {}, "residue": RESIDUE}
    for uid in args.user_id:
        summary["users"][uid] = await delete_user(sdk, uid, args.customer_id, args.dry_run,
                                                  args.timeout)
    return summary


def user_ok(rep: Dict[str, Any], dry_run: bool) -> bool:
    r = rep["result"]
    if rep["errors"] or r is None:
        return False
    if dry_run:
        return True
    if rep["timed_out"] or r["status"] != "complete" or r["still_present_in"] or r["continue"]:
        return False
    if rep["still_returned"]:
        return False
    # Pessimistic: a found user must be reported fully removed by the server.
    return r["fully_removed"] or not r["found"]


def verdict(summary: Dict[str, Any]) -> str:
    ok = all(user_ok(r, summary["dry_run"]) for r in summary["users"].values())
    if summary["dry_run"]:
        return "DRY-RUN" if ok else "INCOMPLETE"
    return "CLEAN" if ok else "INCOMPLETE"


def _counts_line(counts: Dict[str, Any]) -> str:
    return ", ".join(f"{k}={v}" for k, v in sorted(counts.items())) or "no rows"


def render(summary: Dict[str, Any]) -> None:
    dry = summary["dry_run"]
    for uid, rep in summary["users"].items():
        print(f"[USER] {uid}")
        r = rep["result"]
        for e in rep["errors"]:
            print(f"       [ERROR] {e}")
        if r is None:
            continue
        if dry:
            print(f"       [WOULD DELETE] found={r['found']} {_counts_line(r['counts'])}")
            continue
        print(f"       status={r['status']} found={r['found']} fully_removed={r['fully_removed']}"
              f" deletion_id={r['deletion_id']}")
        print(f"       removed: {_counts_line(r['counts'])}")
        if rep["timed_out"]:
            print(f"       [ERROR] timed out after {rep['polls']} status checks, last status "
                  f"{r['status']}. Run this again; the delete is safe to repeat.")
        if r["status"] == "partial" or r["still_present_in"]:
            print("       [PARTIAL] the delete did not clear everything. Still present in: "
                  + (", ".join(r["still_present_in"]) or "not named by the server"))
            for store, why in r["stores"].items():
                if not why.startswith("cleared"):
                    print(f"       [PARTIAL] {store}: {why}")
        elif r["status"] == "failed":
            print("       [ERROR] the server marked the delete failed.")
        if rep["still_returned"]:
            print(f"       [ERROR] a fresh fetch still returns {len(rep['still_returned'])} "
                  f"user-scope item(s): {', '.join(rep['still_returned'][:10])}"
                  " (a server cache can hold them briefly; re-run in a minute)")
        if rep["checked"] and not rep["still_returned"]:
            print("       [OK] a fresh fetch for this user returns nothing")
        for k in r["kept"]:
            print(f"       [KEPT] {k}")
    print("[NOT REMOVED BY A USER DELETE] copy into the final report:")
    for line in RESIDUE:
        print(f"  - {line}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--user-id", action="append", default=[])
    p.add_argument("--customer-id")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--timeout", type=float, default=120.0)
    p.add_argument("--json", dest="json_path")
    p.add_argument("--suggest-id", metavar="PURPOSE")
    # Old flags: parsed only so we can refuse them with a clear line.
    for dest, flag in REMOVED_FLAGS.items():
        if dest == "forget_sweep":
            p.add_argument(flag, dest=dest, action="store_true", help=argparse.SUPPRESS)
        else:
            p.add_argument(flag, dest=dest, action="append", default=[], help=argparse.SUPPRESS)
    return p


def validate(args: argparse.Namespace) -> Optional[str]:
    """Return a refusal message, or None. Runs before any network call."""
    bad = [u for u in args.user_id if not is_test_user_id(u)]
    if bad:
        return (f"REFUSED: only ids that start with {TEST_PREFIX} may be cleaned up. "
                f"Not accepted: {', '.join(repr(b) for b in bad)}. Nothing was called.")
    used = [flag for dest, flag in REMOVED_FLAGS.items() if getattr(args, dest, None)]
    if used:
        return (f"REFUSED: {', '.join(used)} no longer exists. Cleanup now deletes whole test "
                "users with sdk.data.delete; pass --user-id test_user_... only.")
    if not args.user_id:
        return "REFUSED: pass at least one --user-id test_user_..."
    if args.timeout <= 0:
        return "REFUSED: --timeout must be positive."
    return None


async def _real_main(args: argparse.Namespace) -> int:
    _load_dotenv()
    if not os.environ.get("SYNAP_API_KEY"):
        print("[ERROR] SYNAP_API_KEY is not set (and no .env with it in this directory). "
              "Run this BEFORE signing the key out.")
        return 2
    from maximem_synap import MaximemSynapSDK  # imported late so tests need no SDK

    sdk = MaximemSynapSDK()
    if not sdk_has_delete(sdk):
        print(UPGRADE_MSG)
        print("EXIT CODE: 2")
        return 2
    try:
        await sdk.initialize()
        if not getattr(sdk, "instance_id", None):
            print("[ERROR] The API key did not resolve an instance (revoked or wrong key?).")
            return 2
        print(f"[OK] Connected to instance: {sdk.instance_id}")
        if getattr(sdk, "_user_context_isolation", None) == "strict" and not args.customer_id \
                and args.user_id:
            print("[ERROR] This is a B2B instance: pass --customer-id (the customer the test "
                  "users were written under).")
            return 2
        summary = await run(sdk, args)
    finally:
        try:
            await sdk.shutdown()
        except Exception:  # noqa: BLE001
            pass
    return finish(summary, args)


def finish(summary: Dict[str, Any], args: argparse.Namespace) -> int:
    render(summary)
    v = verdict(summary)
    if args.json_path:
        Path(args.json_path).write_text(json.dumps({**summary, "verdict": v}, indent=2), encoding="utf-8")
    print(f"RESULT: {v}")
    code = 0 if v in ("CLEAN", "DRY-RUN") else 1
    print(f"EXIT CODE: {code}")
    return code


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.suggest_id is not None:
        print(make_test_user_id(args.suggest_id))
        return 0
    problem = validate(args)
    if problem:
        print(problem)
        print("EXIT CODE: 2")
        return 2
    return asyncio.run(_real_main(args))


if __name__ == "__main__":
    sys.exit(main())
