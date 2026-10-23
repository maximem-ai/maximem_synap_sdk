#!/usr/bin/env python3
"""Maximem Synap verification: a real write, then a real recall.

Run this as the LAST step of any Synap integration, against the user's OWN
instance (the one their API key resolves to). It proves, in order:

  1. the SDK authenticates and resolves an instance from the key,
  2. a memory can be WRITTEN (a throwaway fact with a unique marker),
  3. the same fact can be RECALLED (polled, because extraction is eventual),
  4. the live gRPC stream opens and the server confirms a test event.

Every step prints one line starting [OK], [WARN] or [ERROR], and the recalled
text is printed verbatim. Quote this output to the user as it is. Do not
paraphrase it and do not report success the script did not print.

    python scripts/verify_synap.py

The key is read from SYNAP_API_KEY. If that is unset, a `.env` file in the
current directory is read (the file `api-keys create --write-env` writes).
The key itself is never printed.

A project with several Synap instances keeps one key per instance, written by
`api-keys create -i <id> --write-env --env-suffix NAME` as SYNAP_API_KEY_<NAME> and
SYNAP_INSTANCE_ID_<NAME>. To verify that instance, name it:

    python scripts/verify_synap.py --suffix SUPPORT_BOT
    SYNAP_VERIFY_SUFFIX=SUPPORT_BOT python scripts/verify_synap.py

NAME is letters, digits and underscore (upper-cased, as the CLI does). The script then
reads SYNAP_API_KEY_<NAME> and SYNAP_INSTANCE_ID_<NAME> (from the environment or
`.env`) and uses them in place of the plain names; it prints the variable name, never
the value. If the suffixed key is missing it stops and says which variable to add.

The output always ends with exactly two lines: `RESULT: PASS` or `RESULT: FAIL <reason>`,
then `EXIT CODE: <n>` (the code the script exits with). Paste the whole output, including
every [WARN] line, unedited. A [WARN] is reported and explained, never dropped.

Exit code 0 = write AND recall both succeeded. Non-zero = something failed;
the last [ERROR] line says what. A failed or skipped recall is never reported
as green.

Optional environment variables:

  SYNAP_VERIFY_TIMEOUT=180        seconds to keep polling for the recall
  SYNAP_VERIFY_CUSTOMER_ID=acme   B2B only. If the instance is B2B (whoami
                                  reports user_context_isolation="strict") and
                                  this is unset, a throwaway customer id
                                  "synap-verify" is used. On B2C it is ignored,
                                  because a customer_id is rejected there (HTTP 400).
  SYNAP_VERIFY_READY_TIMEOUT=300  seconds to keep retrying when Synap answers HTTP 409
                                  instance_not_ready (memory architecture still building)
  SYNAP_VERIFY_REQUIRE_STREAM=1   a stream that will not open is an [ERROR] (use this when the
                                  live stream was wired: it is the default path)
  SYNAP_VERIFY_SKIP_STREAM=1      skip the stream check (the host cannot run a stream)
  SYNAP_VERIFY_SKIP_ROUNDTRIP=1   skip write+recall. The run is then NOT a
                                  full verification and exits 0 only for
                                  steps 1 and 4. Say so if you use it.

Run it only AFTER `instances wait <id>` reports ready. While the memory architecture
is building, Synap rejects runtime writes and fetches with HTTP 409
instance_not_ready. The SDK raises InstanceNotReadyError with `retry_after_seconds`
(and `from_cooldown` when the SDK itself refused without a network call). This
script waits exactly that long (clamped to 5..300 s) and retries, bounded by
SYNAP_VERIFY_READY_TIMEOUT. Older SDKs without that class raise a plain
ConflictError; only then is the message text matched. Exit codes for this case:
2 = still building when the time ran out ([NOT READY]); 3 = the build FAILED for
good (InstanceBuildFailedError, [BUILD FAILED]): run `instances rebuild <id>`,
wait, and re-run. 1 is any other failure.

The write creates one small throwaway memory under a user id of the form
`test_user_verify_<8 random chars>`, and the stream check sends one event as
`test_user_verify_probe_<8 random chars>` (the single test-id convention of this skill,
`test_user_<purpose>_<8 random chars>`). It is harmless but real; tell the user it was
written. Both ids are printed (lines `[WRITE]`, `[PROBE]`, `[TEST IDS]`): record them and
pass them to `scripts/cleanup_test_data.py` before signing out.
"""

import argparse
import asyncio
import os
import re
import time
import uuid
from pathlib import Path

from maximem_synap import MaximemSynapSDK

_SUFFIX_RE = re.compile(r"^[A-Za-z0-9_]+$")
# Every test user id this run created, so they can be printed once and recorded.
TEST_IDS: list = []


def _test_id(purpose: str) -> str:
    """`test_user_<purpose>_<8 random chars>`, the one test-id shape of this skill."""
    uid = f"test_user_{purpose}_{uuid.uuid4().hex[:8]}"
    TEST_IDS.append(uid)
    return uid


def _apply_suffix(suffix) -> None:
    """Point the SDK at SYNAP_API_KEY_<NAME> / SYNAP_INSTANCE_ID_<NAME>.

    The SDK reads only the plain names, so the suffixed values are copied over them
    for this process. Only variable NAMES are printed.
    """
    name = (suffix or "").strip()
    if not name:
        return
    if not _SUFFIX_RE.match(name):
        print(f"[ERROR] --suffix {suffix!r} is not a valid name: use letters, digits and underscore only.")
        raise SystemExit(1)
    name = name.upper()
    key_var, id_var = f"SYNAP_API_KEY_{name}", f"SYNAP_INSTANCE_ID_{name}"
    if not os.environ.get(key_var):
        print(f"[ERROR] {key_var} is not set (and no .env with it in this directory).")
        print(f"        Run: maximem-synap api-keys create -i <instance_id> --write-env --env-suffix {name}")
        raise SystemExit(1)
    os.environ["SYNAP_API_KEY"] = os.environ[key_var]
    if os.environ.get(id_var):
        os.environ["SYNAP_INSTANCE_ID"] = os.environ[id_var]
    else:
        # Never let another instance's plain id pin the wrong instance.
        os.environ.pop("SYNAP_INSTANCE_ID", None)
    print(f"[OK] Using {key_var}" + (f" and {id_var}" if os.environ.get(id_var) else ""))


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


async def verify(suffix=None) -> None:
    _load_dotenv()
    _apply_suffix(suffix)
    if not os.environ.get("SYNAP_API_KEY"):
        print("[ERROR] SYNAP_API_KEY is not set (and no .env with it in this directory).")
        print("        Run: maximem-synap api-keys create (or: npx @maximem/synap-cli@latest api-keys create) -i <instance_id> --write-env")
        print("        (a project with several instances: add --env-suffix NAME, then run this script with --suffix NAME)")
        raise SystemExit(1)

    # No instance_id needed: it is resolved from the API key on initialize().
    sdk = MaximemSynapSDK()
    try:
        await sdk.initialize()
        print("[OK] SDK initialized")

        # initialize() does NOT raise on a bad key. It logs
        # "whoami bootstrap failed (non-fatal, will fall back to env)" and
        # returns, leaving `instance_id` empty. Without this check a junk key
        # produced green ticks and exit 0 while every request 401'd.
        if not sdk.instance_id:
            print("[ERROR] Authenticated to nothing: the API key did not resolve "
                  "an instance.")
            print("        Check SYNAP_API_KEY, and SYNAP_BASE_URL if you are "
                  "pointing at anything other than production.")
            raise SystemExit(1)
        print(f"[OK] Connected to instance: {sdk.instance_id}")

        if os.environ.get("SYNAP_VERIFY_SKIP_ROUNDTRIP"):
            print("[WARN] Write+recall skipped (SYNAP_VERIFY_SKIP_ROUNDTRIP). "
                  "This is NOT a full verification.")
        else:
            await _roundtrip(sdk)

        if not os.environ.get("SYNAP_VERIFY_SKIP_STREAM"):
            await _stream(sdk)

        if TEST_IDS:
            print("[TEST IDS] " + ", ".join(TEST_IDS)
                  + " (record them; clean up with scripts/cleanup_test_data.py before sign-out)")

        await sdk.shutdown()
        print("[OK] SDK shut down cleanly")
    except SystemExit:
        try:
            await sdk.shutdown()
        except Exception:
            pass
        raise
    except Exception as e:  # noqa: BLE001 - surface any failure to the developer
        print(f"[ERROR] {type(e).__name__}: {e}")
        # Best-effort cleanup; ignore secondary errors during teardown.
        try:
            await sdk.shutdown()
        except Exception:
            pass
        raise SystemExit(1)


_NOT_READY_TEXT = "memory architecture is still being built"
_WAIT_MIN, _WAIT_MAX, _WAIT_DEFAULT = 5.0, 300.0, 30.0


def _class_names(exc: BaseException) -> set:
    return {c.__name__ for c in type(exc).__mro__}


def _text_of(exc: BaseException) -> str:
    parts = [str(exc)]
    for attr in ("details", "body", "response_body", "response", "error_code", "code"):
        value = getattr(exc, attr, None)
        if value is not None:
            parts.append(str(value))
    return " ".join(parts)


def _build_failed(exc: BaseException):
    """Return (attempts, hint) if `exc` means the build failed for good, else None.

    Reads the typed fields of InstanceBuildFailedError. Older SDKs have no such
    class, so fall back to the code `instance_build_failed` in the error text.
    """
    if "InstanceBuildFailedError" in _class_names(exc):
        return getattr(exc, "attempts", None), getattr(exc, "hint", None)
    if "instance_build_failed" in _text_of(exc).lower():
        m = re.search(r"attempts\W+(\d+)", _text_of(exc))
        return (int(m.group(1)) if m else None), None
    return None


def _not_ready_wait(exc: BaseException):
    """Return (seconds_to_wait, attempt) if `exc` means "still building", else None.

    Typed first: `InstanceNotReadyError.retry_after_seconds` is the server's own
    estimate (or, when `from_cooldown` is true, the REMAINING wait the SDK is
    enforcing locally). We wait exactly that, clamped to 5..300 s. `attempt` is
    used only if the SDK exposes it. Older SDKs raise a plain ConflictError whose
    text says the architecture is still being built; only then is the text read.
    Any other ConflictError (for example "compaction already in progress") is a
    real conflict and is NOT treated as not-ready.
    """
    names = _class_names(exc)
    if "InstanceNotReadyError" in names:
        secs = getattr(exc, "retry_after_seconds", None)
        wait = _WAIT_DEFAULT if secs is None else min(max(float(secs), _WAIT_MIN), _WAIT_MAX)
        return wait, getattr(exc, "attempt", None)
    text = _text_of(exc)
    low = text.lower()
    if "instance_not_ready" not in low and not ("ConflictError" in names and _NOT_READY_TEXT in low):
        return None
    m = re.search(r"retry_after_seconds\W+(\d+(?:\.\d+)?)", text)
    wait = min(max(float(m.group(1)), _WAIT_MIN), _WAIT_MAX) if m else _WAIT_DEFAULT
    return wait, None


async def _when_ready(what: str, call):
    """Run `call()`, waiting the server's hint while the instance is still building.

    A build that failed for good exits 3 (waiting cannot help). Running out of
    SYNAP_VERIFY_READY_TIMEOUT exits 2.
    """
    limit = float(os.environ.get("SYNAP_VERIFY_READY_TIMEOUT", "300"))
    deadline = time.monotonic() + limit
    while True:
        try:
            return await call()
        except Exception as e:  # noqa: BLE001
            failed = _build_failed(e)
            if failed is not None:
                attempts, hint = failed
                tries = f" after {attempts} attempts" if attempts else ""
                print(f"[BUILD FAILED] {what}: the memory architecture build failed{tries}.")
                print(f"               {hint or 'Run: maximem-synap instances rebuild <instance_id>'}")
                print("               Tell the human, run the rebuild, wait for it to finish "
                      "(`instances wait <instance_id>`), then re-run this script.")
                raise SystemExit(3)
            info = _not_ready_wait(e)
            if info is None:
                raise
            wait, attempt = info
            if time.monotonic() + wait > deadline:
                print(f"[NOT READY] {what}: the instance is still building after "
                      f"{limit:.0f}s (instance_not_ready).")
                print("            Run `instances wait <instance_id>` until it reports "
                      "ready, then re-run this script.")
                raise SystemExit(2)
            of = f" (attempt {attempt} of 3)" if attempt else ""
            print(f"[..] {what}: The memory architecture is still being built: "
                  f"waiting {wait:.0f} s{of}")
            await asyncio.sleep(wait)


def _items(ctx) -> list:
    """Every returned item's text, whatever its type."""
    out = []
    for name in ("facts", "preferences", "episodes", "emotions",
                 "temporal_events", "excerpts"):
        for item in getattr(ctx, name, None) or []:
            text = getattr(item, "content", None) or getattr(item, "description", None)
            out.append(str(text) if text else item.model_dump_json())
    return out


async def _roundtrip(sdk: MaximemSynapSDK) -> None:
    """Write one throwaway fact, then poll until it is recalled (or time runs out)."""
    user_id = _test_id("verify")
    run = user_id.rsplit("_", 1)[-1]
    marker = f"Bluefin-{run}"

    # B2C sends user_id alone. B2B (user_context_isolation="strict") also needs a
    # customer_id, and a B2C instance rejects one with HTTP 400.
    scope = {"user_id": user_id}
    isolation = getattr(sdk, "_user_context_isolation", None)
    if isolation == "strict":
        scope["customer_id"] = os.environ.get("SYNAP_VERIFY_CUSTOMER_ID") or "synap-verify"
        print(f"[OK] Instance is B2B (strict); using customer_id={scope['customer_id']}")
    elif isolation == "equals_customer":
        print("[OK] Instance is B2C (equals_customer); sending user_id only")
    else:
        print("[WARN] Could not read the instance mode from whoami; sending user_id only")
        customer_id = os.environ.get("SYNAP_VERIFY_CUSTOMER_ID")
        if customer_id:
            scope["customer_id"] = customer_id

    document = (f"User: My verification project codename is {marker}.\n"
                f"Assistant: Got it, the codename is {marker}.")
    print(f"[WRITE] user_id={user_id} document={document!r}")
    await _when_ready("write", lambda: sdk.memories.create(
        document=document, document_type="ai-chat-conversation", **scope))
    print("[OK] Write accepted by Synap (extraction is asynchronous)")

    timeout = float(os.environ.get("SYNAP_VERIFY_TIMEOUT", "180"))
    deadline = time.monotonic() + timeout
    attempt, last = 0, []
    while True:
        attempt += 1
        # Extraction takes a while; in production drive retrieval from events,
        # not sleeps. Here a bounded poll is the point of the check.
        await asyncio.sleep(10 if attempt > 1 else 5)
        ctx = await _when_ready("recall", lambda: sdk.user.context.fetch(
            search_query=[f"verification project codename {marker}"], **scope))
        last = _items(ctx)
        hits = [t for t in last if marker in t]
        if hits:
            print(f"[RECALL] attempt {attempt}: {len(hits)} item(s) contain {marker}")
            for t in hits:
                print(f"[RECALLED] {t}")
            print("[OK] Write then recall succeeded on your own instance")
            return
        print(f"[..] attempt {attempt}: {len(last)} item(s) returned, "
              f"none contain {marker} yet")
        if time.monotonic() >= deadline:
            break

    print(f"[ERROR] The memory was written but NOT recalled within {timeout:.0f}s "
          f"({attempt} attempts, last fetch returned {len(last)} item(s)).")
    print("        Either extraction is still running (re-run this script in a few "
          "minutes), the instance's memory architecture is not ready "
          "(`instances wait <id>`), or the key points at a different instance.")
    raise SystemExit(1)


async def _stream(sdk: MaximemSynapSDK) -> None:
    """Open the live stream and close it. Sends nothing, stores nothing.

    Checked because the stream is the operation integrations skip, and because
    when it fails it fails quietly: fetch() still works, it is just cold every
    time, and no turn ever becomes memory on its own.
    """
    try:
        await sdk.instance.listen()
    except Exception as e:  # noqa: BLE001 - a closed network is not a code bug
        if os.environ.get("SYNAP_VERIFY_REQUIRE_STREAM"):
            # The live stream is the default path where the host can run it. When
            # it was wired, a stream that will not open is a failure, not a warning.
            print(f"[ERROR] Could not open the live stream: {type(e).__name__}: {e}")
            print("        The stream is required here (SYNAP_VERIFY_REQUIRE_STREAM=1). Check the")
            print("        dependency conflicts (for example protobuf) and gRPC egress on 443;")
            print("        see reference/streaming.md.")
            raise SystemExit(1)
        print(f"[WARN] Could not open the live stream: {type(e).__name__}: {e}")
        print("       Anticipation needs it. If this host has gRPC egress on 443,")
        print("       fix this before shipping; see reference/streaming.md.")
        return

    # ⚠ `is_listening` is true for a stream the server rejected: it reflects
    # the client's own intent, not the server's answer. Verified against a live
    # instance with an invalid key, where it read True while the stream logged
    # UNAUTHENTICATED. So prove the stream instead of trusting the flag: send
    # one real event and ask the SDK whether it got out.
    if not sdk.instance.is_listening:
        print("[ERROR] listen() returned but the stream is not connected.")
        raise SystemExit(1)

    probe_conversation = str(uuid.uuid4())
    probe_user = _test_id("verify_probe")
    print(f"[PROBE] user_id={probe_user} (one stream event)")
    await sdk.instance.send_message(
        conversation_id=probe_conversation,
        user_id=probe_user,
        event_type="user_message",
        role="user",
        content="synap verify probe",
    )
    await sdk.instance.end_session(probe_conversation)
    await sdk.instance.stop_listening()

    # `undelivered()` is the only honest signal here: every send path returns
    # None, so "it did not raise" says nothing about whether the server got it.
    transport = getattr(sdk, "_grpc_transport", None)
    left = transport.undelivered() if hasattr(transport, "undelivered") else None
    if left and (left["queued"] or left["unacknowledged"]):
        print(f"[ERROR] The stream is open but the server confirmed nothing: "
              f"{left}.")
        print("        Anticipation will receive none of your turns. Check "
              "gRPC egress on 443 and your key's instance.")
        raise SystemExit(1)
    print("[OK] Live stream opened and the server confirmed a test event")


class _Tee:
    """Copy everything printed, so the final RESULT line can quote the last problem."""

    def __init__(self, stream):
        self._stream, self.lines = stream, []

    def write(self, text):
        self.lines.extend(t for t in text.splitlines() if t.strip())
        return self._stream.write(text)

    def flush(self):
        self._stream.flush()

    def __getattr__(self, name):
        return getattr(self._stream, name)


def main() -> None:
    """Run verify() and end with exactly two lines: RESULT and EXIT CODE.

    Paste the WHOLE output to the human, unedited, warnings included.
    """
    import sys

    parser = argparse.ArgumentParser(description="Write then recall one throwaway memory on your own instance.")
    parser.add_argument("--suffix", metavar="NAME", default=os.environ.get("SYNAP_VERIFY_SUFFIX"),
                        help="verify the instance whose key is SYNAP_API_KEY_<NAME> "
                             "(env SYNAP_VERIFY_SUFFIX); written by api-keys create --env-suffix NAME")
    args = parser.parse_args()

    tee = _Tee(sys.stdout)
    sys.stdout = tee
    code = 0
    try:
        asyncio.run(verify(args.suffix))
    except SystemExit as e:
        code = e.code if isinstance(e.code, int) else (0 if e.code in (None, 0) else 1)
    except BaseException as e:  # noqa: BLE001
        print(f"[ERROR] {type(e).__name__}: {e}")
        code = 1
    sys.stdout = tee._stream
    if code == 0:
        print("RESULT: PASS")
    else:
        problems = [l for l in tee.lines if l.startswith(("[ERROR]", "[NOT READY]", "[FAILED]"))]
        reason = (problems[-1] if problems else "see the output above")
        reason = reason.split("]", 1)[-1].strip() if problems else reason
        print(f"RESULT: FAIL {reason}")
    print(f"EXIT CODE: {code}")
    raise SystemExit(code)


if __name__ == "__main__":
    main()

# Versions: run `pip show maximem-synap` for the installed version. Docs: https://docs.maximem.ai
