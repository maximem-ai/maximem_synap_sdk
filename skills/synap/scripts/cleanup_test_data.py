#!/usr/bin/env python3
"""Delete the test memories a coding agent created while wiring Synap, then check.

Run it near the end of onboarding, BEFORE the setup key is signed out (the SDK
needs the runtime key from .env to delete). It uses only what the SDK offers
today: it lists a test user's memories through a context fetch, deletes each one
with `sdk.memories.delete`, and then checks that a read of each id answers
"not found" and that a fresh fetch returns nothing for that user.

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
    --memory-id UUID     also delete this memory (repeatable). Use the ids from
                         the ingestion status of YOUR OWN test writes.
    --ingestion-id ID    read this ingestion's memory ids and delete them too
                         (repeatable).
    --dry-run            list what would be deleted; delete and change nothing.
    --forget-sweep       after deleting, also run memories.forget(mode="direct")
                         for each user, to catch copies a delete by id does not
                         reach (profile text, source chunks). Best effort.
    --query TEXT         extra search text for finding memories (repeatable).
    --passes N           listing/deleting rounds, default 3. A later round finds
                         memories whose extraction finished after the first.
    --pause SECONDS      wait between rounds, default 15.
    --json PATH          also write the full report as JSON to PATH.

Exit codes: 0 clean (or dry run), 1 something was not removed or could not be
checked, 2 refused or cannot start (bad id, no key, nothing to do).

The script never prints the API key and never prints memory text, only ids.
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
_UUID_RE = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$")

DEFAULT_QUERIES = [
    "everything about this user",
    "facts and personal details",
    "preferences and likes",
    "events, plans and dates",
    "names, places and projects",
    "test verification codename",
]

# What no customer-callable call removes today. Printed on every run so the
# final report can copy it.
RESIDUE = [
    "Conversation turns and compacted summaries recorded through the live stream or "
    "record_message (no delete call exists)",
    "Customer-scope and client-scope memories (the cleanup only lists and deletes user-scope ones)",
    "Raw documents you sent (the original text kept for provenance)",
    "Usage, telemetry and request-log rows (billing and audit records)",
    "Source chunks shared with other memories (a delete keeps them while another memory rests on them)",
    "The per-user profile document (forget --forget-sweep tries; delete by id does not)",
    "Copies on this machine: run sdk.cache.clear_user(<id>) (done here); turns.db and the "
    "stream outbox need sdk.cache.clear() / deleting the file with the agent stopped",
    "Backups taken before the delete, until they expire",
    "Erasure of a whole person (preview + erase) is run by Synap staff, not callable with the runtime key",
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


def _is_not_found(exc: BaseException) -> bool:
    if any(c.__name__ == "ContextNotFoundError" for c in type(exc).__mro__):
        return True
    return getattr(exc, "status_code", None) == 404


def _is_not_ready(exc: BaseException) -> bool:
    names = {c.__name__ for c in type(exc).__mro__}
    return "InstanceNotReadyError" in names or "still being built" in str(exc)


# --------------------------------------------------------------------- listing

def _iter_items(ctx: Any):
    """Yield (id, scope, category) for every item of a fetch response.

    A ranked response lists every item once with its category, chunks apart as
    "excerpts". A by-type response keeps chunks inside `facts`, which cannot be
    told apart, so a delete of such an id may answer not-found; the final
    check by id still decides what counts as removed.
    """
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
        # An older SDK without response_shape.
        return await sdk.user.context.fetch(**kwargs)


async def list_user_memories(sdk, user_id, customer_id, queries, limit) -> Dict[str, Any]:
    """Find a test user's own memory ids. Only items whose scope is "user" count.

    A user fetch also returns customer- and client-level memories shared with
    other people; deleting those would be deleting somebody else's data, so an
    item without scope == "user" is counted and left alone.
    """
    ids: Dict[str, str] = {}
    other_scope: set = set()
    unattributed: set = set()
    chunk_ids: set = set()
    errors: List[str] = []
    for q in queries:
        try:
            ctx = await _fetch(sdk, user_id, customer_id, q, limit)
        except Exception as e:  # noqa: BLE001
            errors.append(f"{type(e).__name__}: {str(e)[:160]}")
            if _is_not_ready(e):
                break
            continue
        for mid, scope, category in _iter_items(ctx):
            if category == "excerpts":
                chunk_ids.add(mid)
            elif scope == "user":
                ids[mid] = category
            elif scope is None:
                unattributed.add(mid)
            else:
                other_scope.add(mid)
    return {"ids": ids, "skipped_other_scope": len(other_scope),
            "skipped_unattributed": len(unattributed), "chunks": len(chunk_ids), "errors": errors}


# -------------------------------------------------------------------- deleting

async def delete_ids(sdk, memory_ids, report: Dict[str, Any]) -> None:
    from uuid import UUID
    for mid in memory_ids:
        if mid in report["deleted"] or mid in report["already_gone"]:
            continue
        try:
            res = await sdk.memories.delete(UUID(mid))
        except Exception as e:  # noqa: BLE001
            if _is_not_found(e):
                report["already_gone"].append(mid)
            else:
                report["failed"][mid] = f"{type(e).__name__}: {str(e)[:160]}"
            continue
        report["failed"].pop(mid, None)
        report["deleted"].append(mid)
        # Today's SDK returns only memory_id and deleted_at. If a newer one
        # passes the server's own residue report through, keep it.
        if isinstance(res, dict):
            left = res.get("still_present_in")
            if left:
                report["still_present_in"][mid] = list(left)
            if res.get("fully_removed") is False and not left:
                report["still_present_in"][mid] = ["unspecified"]


async def check_gone(sdk, memory_ids) -> Dict[str, str]:
    """Read each deleted id back. Not-found is the proof; anything else is a finding."""
    from uuid import UUID
    bad: Dict[str, str] = {}
    for mid in memory_ids:
        try:
            await sdk.memories.get(UUID(mid))
            bad[mid] = "still readable"
        except Exception as e:  # noqa: BLE001
            if not _is_not_found(e):
                bad[mid] = f"could not check ({type(e).__name__})"
    return bad


# ------------------------------------------------------------------------- run

async def run(sdk: Any, args: argparse.Namespace) -> Dict[str, Any]:
    queries = list(DEFAULT_QUERIES) + list(args.query or [])
    summary: Dict[str, Any] = {"dry_run": bool(args.dry_run), "users": {}, "extra": None,
                               "residue": RESIDUE}

    async def one_target(label: str, user_id: Optional[str]) -> Dict[str, Any]:
        rep: Dict[str, Any] = {"found": [], "deleted": [], "already_gone": [], "failed": {},
                               "still_present_in": {}, "skipped_other_scope": 0,
                               "skipped_unattributed": 0, "chunks_seen": 0, "errors": [],
                               "forget": None, "still_returned": [], "not_verified": {}}
        if user_id is None:
            return rep
        passes = 1 if args.dry_run else max(1, args.passes)
        for n in range(passes):
            if n:
                await _sleep(args.pause)
            found = await list_user_memories(sdk, user_id, args.customer_id, queries, args.limit)
            rep["errors"].extend(found["errors"])
            rep["skipped_other_scope"] = max(rep["skipped_other_scope"], found["skipped_other_scope"])
            rep["skipped_unattributed"] = max(rep["skipped_unattributed"], found["skipped_unattributed"])
            rep["chunks_seen"] = max(rep["chunks_seen"], found["chunks"])
            fresh = [m for m in found["ids"] if m not in rep["found"]]
            rep["found"].extend(fresh)
            if args.dry_run:
                break
            todo = [m for m in found["ids"] if m not in rep["deleted"] and m not in rep["already_gone"]]
            if not todo and n > 0:
                break
            await delete_ids(sdk, todo, rep)
        if args.dry_run:
            return rep
        if args.forget_sweep:
            try:
                kw = {"description": "everything this user told me", "user_id": user_id,
                      "mode": "direct"}
                if args.customer_id:
                    kw["customer_id"] = args.customer_id
                fr = await sdk.memories.forget(**kw)
                rep["forget"] = {"status": getattr(fr, "status", None),
                                 "deleted": len(getattr(fr, "deleted_ids", []) or []),
                                 "still_present_in": list(getattr(fr, "still_present_in", []) or [])}
            except Exception as e:  # noqa: BLE001
                rep["forget"] = {"status": f"error {type(e).__name__}"}
        # A cached answer must not be mistaken for a live one.
        try:
            sdk.cache.clear_user(user_id)
        except Exception:  # noqa: BLE001
            pass
        gone = rep["deleted"] + rep["already_gone"]
        rep["not_verified"] = await check_gone(sdk, gone)
        after = await list_user_memories(sdk, user_id, args.customer_id, queries, args.limit)
        rep["errors"].extend(after["errors"])
        rep["still_returned"] = sorted(after["ids"])
        return rep

    for uid in args.user_id:
        summary["users"][uid] = await one_target(uid, uid)

    # Explicit ids and ingestions the agent recorded from its own writes.
    extra: Dict[str, Any] = {"ids": [], "deleted": [], "already_gone": [], "failed": {},
                             "still_present_in": {}, "not_verified": {}, "errors": []}
    ids = list(args.memory_id or [])
    for ing in args.ingestion_id or []:
        try:
            st = await sdk.memories.status(ing)
            ids.extend(str(m) for m in (getattr(st, "memory_ids", None) or []))
        except Exception as e:  # noqa: BLE001
            extra["errors"].append(f"ingestion {ing}: {type(e).__name__}")
    extra["ids"] = sorted(set(ids))
    if extra["ids"]:
        if not args.dry_run:
            await delete_ids(sdk, extra["ids"], extra)
            extra["not_verified"] = await check_gone(sdk, extra["deleted"] + extra["already_gone"])
        summary["extra"] = extra
    return summary


def verdict(summary: Dict[str, Any]) -> str:
    if summary["dry_run"]:
        return "DRY-RUN"
    blocks = list(summary["users"].values()) + ([summary["extra"]] if summary["extra"] else [])
    for b in blocks:
        if b["failed"] or b["not_verified"] or b["still_present_in"] or b["errors"]:
            return "INCOMPLETE"
        if b.get("still_returned"):
            return "INCOMPLETE"
        f = b.get("forget")
        if f and f.get("status") not in ("deleted", "nothing_matches"):
            return "INCOMPLETE"
    return "CLEAN"


def render(summary: Dict[str, Any]) -> None:
    dry = summary["dry_run"]
    for uid, r in summary["users"].items():
        print(f"[USER] {uid}")
        print(f"       found {len(r['found'])} user-scope memories"
              f" (left alone: {r['skipped_other_scope']} customer/client-scope,"
              f" {r['skipped_unattributed']} with no scope, {r['chunks_seen']} source excerpts)")
        if dry:
            for m in r["found"]:
                print(f"       [WOULD DELETE] {m}")
            for e in r["errors"]:
                print(f"       [WARN] listing: {e}")
            continue
        print(f"       deleted {len(r['deleted'])}, already gone {len(r['already_gone'])},"
              f" failed {len(r['failed'])}")
        for m, why in r["failed"].items():
            print(f"       [ERROR] {m}: {why}")
        for m, left in r["still_present_in"].items():
            print(f"       [WARN] {m} still present in: {', '.join(left)}")
        for m, why in r["not_verified"].items():
            print(f"       [ERROR] {m}: {why}")
        if r["forget"]:
            f = r["forget"]
            print(f"       [FORGET] status={f.get('status')} deleted={f.get('deleted', 0)}"
                  + (f" still_present_in={f['still_present_in']}" if f.get("still_present_in") else ""))
        if r["still_returned"]:
            print(f"       [ERROR] a fresh fetch still returns {len(r['still_returned'])} user-scope "
                  f"item(s): {', '.join(r['still_returned'][:10])}"
                  " (a server cache can hold them for a short time; re-run in a minute)")
        for e in r["errors"]:
            print(f"       [WARN] {e}")
    ex = summary["extra"]
    if ex:
        print(f"[EXPLICIT IDS] {len(ex['ids'])} id(s)")
        if not dry:
            print(f"       deleted {len(ex['deleted'])}, already gone {len(ex['already_gone'])},"
                  f" failed {len(ex['failed'])}")
            for m, why in {**ex["failed"], **ex["not_verified"]}.items():
                print(f"       [ERROR] {m}: {why}")
        for e in ex["errors"]:
            print(f"       [WARN] {e}")
    print("[NOT REMOVED BY ANY CUSTOMER-CALLABLE API TODAY] copy into the final report:")
    for line in RESIDUE:
        print(f"  - {line}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--user-id", action="append", default=[])
    p.add_argument("--customer-id")
    p.add_argument("--memory-id", action="append", default=[])
    p.add_argument("--ingestion-id", action="append", default=[])
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--forget-sweep", action="store_true")
    p.add_argument("--query", action="append", default=[])
    p.add_argument("--passes", type=int, default=3)
    p.add_argument("--pause", type=float, default=15.0)
    p.add_argument("--limit", type=int, default=100)
    p.add_argument("--json", dest="json_path")
    p.add_argument("--suggest-id", metavar="PURPOSE")
    return p


def validate(args: argparse.Namespace) -> Optional[str]:
    """Return a refusal message, or None. Runs before any network call."""
    bad = [u for u in args.user_id if not is_test_user_id(u)]
    if bad:
        return (f"REFUSED: only ids that start with {TEST_PREFIX} may be cleaned up. "
                f"Not accepted: {', '.join(repr(b) for b in bad)}. Nothing was called.")
    if not (args.user_id or args.memory_id or args.ingestion_id):
        return "REFUSED: pass at least one --user-id test_user_..., or --memory-id / --ingestion-id."
    badm = [m for m in args.memory_id if not _UUID_RE.match(m)]
    if badm:
        return f"REFUSED: not a memory UUID: {', '.join(badm)}"
    if args.passes < 1 or args.limit < 1:
        return "REFUSED: --passes and --limit must be at least 1."
    return None


async def _real_main(args: argparse.Namespace) -> int:
    _load_dotenv()
    if not os.environ.get("SYNAP_API_KEY"):
        print("[ERROR] SYNAP_API_KEY is not set (and no .env with it in this directory). "
              "Run this BEFORE signing the key out.")
        return 2
    from maximem_synap import MaximemSynapSDK  # imported late so tests need no SDK

    sdk = MaximemSynapSDK()
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
