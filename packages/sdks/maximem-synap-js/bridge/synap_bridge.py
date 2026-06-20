#!/usr/bin/env python3
"""Synap SDK Bridge: JSON-RPC over stdin/stdout for the Node.js wrapper.

Protocol:
  stdin  -> {"id": 1, "method": "init", "params": {...}}\n
  stdout <- {"id": 1, "result": {...}, "error": null}\n
Methods:
  init, add_memory, create_memory, record_message, search_memory, get_memories,
  fetch_user_context, fetch_customer_context, fetch_client_context,
  get_context_for_prompt, delete_memory, shutdown
"""

import asyncio
import json
import logging
import sys
import time
import traceback
from typing import Dict, List, Optional
from uuid import UUID

# Keep stdout clean for protocol responses.
logging.basicConfig(
    stream=sys.stderr,
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger("synap.bridge")

try:
    from maximem_synap import MaximemSynapSDK
except Exception as import_error:  # pragma: no cover
    sys.stderr.write(
        "Failed to import maximem_synap. Run `synap-js-sdk setup` first.\n"
    )
    sys.stderr.write(f"Import error: {import_error}\n")
    raise

sdk = None
user_memory_ids: Dict[str, List[str]] = {}


def ms_since(start: float) -> int:
    return int((time.perf_counter() - start) * 1000)


def append_step(timings: List[dict], step: str, started: float) -> None:
    timings.append({"step": step, "ms": ms_since(started)})


def write_response(obj: dict) -> None:
    sys.stdout.write(json.dumps(obj, default=str) + "\n")
    sys.stdout.flush()


def tracking_key(user_id: str, customer_id: Optional[str]) -> str:
    """Scope tracked memory IDs by both customer and user."""
    return f"{customer_id or ''}::{user_id}"


def serialize_context_response(context) -> dict:
    """Serialize a Python ContextResponse for the JS bridge."""
    payload = context.model_dump(mode="json")
    payload["raw_response"] = context.raw if hasattr(context, "raw") else {}
    return payload


def serialize_context_for_prompt_response(response) -> dict:
    """Serialize a Python ContextForPromptResponse for the JS bridge."""
    return response.model_dump(mode="json")


def flatten_context_items(context) -> List[dict]:
    """Convert typed context collections into a flat memory list."""
    items: List[dict] = []
    for fact in context.facts:
        items.append({
            "id": fact.id,
            "memory": fact.content,
            "score": fact.confidence,
            "source": fact.source,
            "metadata": fact.metadata,
            "context_type": "fact",
            "event_date": str(fact.event_date) if getattr(fact, "event_date", None) else None,
            "valid_until": str(fact.valid_until) if getattr(fact, "valid_until", None) else None,
            "temporal_category": getattr(fact, "temporal_category", None),
            "temporal_confidence": getattr(fact, "temporal_confidence", 0.0),
        })
    for preference in context.preferences:
        items.append({
            "id": preference.id,
            "memory": preference.content,
            "score": preference.strength,
            "source": getattr(preference, "source", ""),
            "metadata": preference.metadata,
            "context_type": "preference",
            "event_date": str(preference.event_date) if getattr(preference, "event_date", None) else None,
            "valid_until": str(preference.valid_until) if getattr(preference, "valid_until", None) else None,
            "temporal_category": getattr(preference, "temporal_category", None),
            "temporal_confidence": getattr(preference, "temporal_confidence", 0.0),
        })
    for episode in context.episodes:
        items.append({
            "id": episode.id,
            "memory": episode.summary,
            "score": episode.significance,
            "metadata": episode.metadata,
            "context_type": "episode",
            "event_date": str(episode.event_date) if getattr(episode, "event_date", None) else None,
            "valid_until": str(episode.valid_until) if getattr(episode, "valid_until", None) else None,
            "temporal_category": getattr(episode, "temporal_category", None),
            "temporal_confidence": getattr(episode, "temporal_confidence", 0.0),
        })
    for emotion in context.emotions:
        items.append({
            "id": emotion.id,
            "memory": emotion.context,
            "score": emotion.intensity,
            "metadata": emotion.metadata,
            "context_type": "emotion",
            "event_date": str(emotion.event_date) if getattr(emotion, "event_date", None) else None,
            "valid_until": str(emotion.valid_until) if getattr(emotion, "valid_until", None) else None,
            "temporal_category": getattr(emotion, "temporal_category", None),
            "temporal_confidence": getattr(emotion, "temporal_confidence", 0.0),
        })
    for event in getattr(context, "temporal_events", []):
        items.append({
            "id": event.id,
            "memory": event.content,
            "score": event.temporal_confidence,
            "source": event.source,
            "metadata": event.metadata,
            "context_type": "temporal_event",
            "event_date": str(event.event_date) if event.event_date else None,
            "valid_until": str(event.valid_until) if event.valid_until else None,
            "temporal_category": event.temporal_category,
            "temporal_confidence": event.temporal_confidence,
        })
    return items


def messages_to_text(messages: List[dict]) -> str:
    lines: List[str] = []
    for message in messages:
        content = (message.get("content") or "").strip()
        if not content:
            continue
        role = "Assistant" if message.get("role") == "assistant" else "User"
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


def on_anticipated_context(bundle: dict) -> None:
    logger.info(
        "Anticipated context bundle received: %s",
        json.dumps(bundle, default=str)[:300],
    )


async def handle_init(params: dict) -> dict:
    global sdk

    handler_start = time.perf_counter()
    timings: List[dict] = []

    instance_id = params.get("instance_id", "")
    api_key = params.get("api_key")

    step = time.perf_counter()
    sdk = MaximemSynapSDK(
        instance_id=instance_id,
        api_key=api_key,
        _force_new=True,
    )
    append_step(timings, "construct_sdk", step)

    config_kwargs = {"log_level": "DEBUG", "cache_backend": "sqlite"}

    if params.get("base_url"):
        config_kwargs["api_base_url"] = params["base_url"]
    if params.get("grpc_host"):
        config_kwargs["grpc_host"] = params["grpc_host"]
    if params.get("grpc_port"):
        config_kwargs["grpc_port"] = int(params["grpc_port"])
    if "grpc_use_tls" in params:
        config_kwargs["grpc_use_tls"] = bool(params["grpc_use_tls"])

    step = time.perf_counter()
    sdk.configure(**config_kwargs)
    append_step(timings, "configure_sdk", step)

    step = time.perf_counter()
    await sdk.initialize()
    append_step(timings, "initialize_sdk", step)

    grpc_listening = False

    step = time.perf_counter()
    try:
        await sdk.instance.listen(
            on_context=on_anticipated_context,
            on_reconnect=lambda attempt: logger.info(
                "gRPC reconnect attempt %d", attempt
            ),
            on_disconnect=lambda reason: logger.warning(
                "gRPC disconnected: %s", reason
            ),
        )
        grpc_listening = sdk.instance.is_listening
    except Exception as exc:  # non-fatal
        logger.warning("gRPC listen failed (non-fatal): %s", exc)
    append_step(timings, "start_grpc_listener", step)

    return {
        "success": True,
        "instance_id": instance_id,
        "grpc_listening": grpc_listening,
        "bridgeTiming": {
            "python_total_ms": ms_since(handler_start),
            "steps": timings,
        },
    }


async def handle_add_memory(params: dict) -> dict:
    handler_start = time.perf_counter()
    timings: List[dict] = []

    user_id = params["user_id"]
    customer_id = params.get("customer_id")
    messages = params["messages"]
    conversation_id = params.get("conversation_id")
    session_id = params.get("session_id")

    if not customer_id:
        raise ValueError("customer_id is required")

    step = time.perf_counter()
    transcript = messages_to_text(messages)
    append_step(timings, "build_transcript", step)

    if not transcript:
        return {
            "success": True,
            "latencyMs": 0,
            "rawResponse": {"note": "Empty input; skipped"},
            "note": "No text content to ingest",
            "bridgeTiming": {
                "python_total_ms": ms_since(handler_start),
                "steps": timings,
            },
        }

    start = time.perf_counter()

    step = time.perf_counter()
    mode = params.get("mode", "long-range")
    document_type = params.get("document_type", "ai-chat-conversation")
    document_id = params.get("document_id")
    document_created_at = params.get("document_created_at")
    metadata = params.get("metadata")

    create_kwargs: dict = {
        "document": transcript,
        "document_type": document_type,
        "user_id": user_id,
        "customer_id": customer_id,
        "mode": mode,
    }
    if document_id is not None:
        create_kwargs["document_id"] = document_id
    if document_created_at is not None:
        create_kwargs["document_created_at"] = document_created_at
    if metadata is not None:
        create_kwargs["metadata"] = metadata

    create_result = await sdk.memories.create(**create_kwargs)
    append_step(timings, "memories_create", step)

    ingestion_id = create_result.ingestion_id

    step = time.perf_counter()
    if sdk.instance.is_listening:
        for message in messages:
            content = (message.get("content") or "").strip()
            if not content:
                continue
            try:
                role = message.get("role", "user")
                await sdk.instance.send_message(
                    content=content,
                    role=role,
                    conversation_id=conversation_id,
                    user_id=user_id,
                    customer_id=customer_id,
                    session_id=session_id,
                    event_type="assistant_message" if role == "assistant" else "user_message",
                    metadata=message.get("metadata"),
                )
            except Exception as exc:
                logger.debug("gRPC send_message failed (non-fatal): %s", exc)
    append_step(timings, "grpc_send_messages", step)

    step = time.perf_counter()
    try:
        final_status = await sdk.memories.wait_for_completion(
            ingestion_id,
            timeout_seconds=60,
            poll_interval_seconds=2,
        )
    except TimeoutError:
        append_step(timings, "wait_for_completion_timeout", step)
        return {
            "success": True,
            "latencyMs": ms_since(start),
            "rawResponse": {"ingestion_id": str(ingestion_id), "status": "timeout"},
            "note": "Ingestion timed out after 60s and may still complete later",
            "bridgeTiming": {
                "python_total_ms": ms_since(handler_start),
                "steps": timings,
            },
        }
    append_step(timings, "wait_for_completion", step)

    memory_ids = [str(memory_id) for memory_id in (final_status.memory_ids or [])]
    if memory_ids:
        user_memory_ids.setdefault(tracking_key(user_id, customer_id), []).extend(memory_ids)

    return {
        "success": final_status.status.value != "failed",
        "latencyMs": ms_since(start),
        "rawResponse": {
            "ingestion_id": str(ingestion_id),
            "status": final_status.status.value,
            "memories_created": final_status.memories_created,
            "memory_ids": memory_ids,
        },
        "note": f"Ingestion {final_status.status.value}; created {final_status.memories_created}",
        "bridgeTiming": {
            "python_total_ms": ms_since(handler_start),
            "steps": timings,
        },
    }


async def handle_search_memory(params: dict) -> dict:
    handler_start = time.perf_counter()
    timings: List[dict] = []

    user_id = params["user_id"]
    customer_id = params.get("customer_id")
    query = params["query"]
    max_results = params.get("max_results", 10)
    mode = params.get("mode", "fast")
    conversation_id = params.get("conversation_id")
    types = params.get("types", ["all"])

    start = time.perf_counter()

    step = time.perf_counter()
    context = await sdk.user.context.fetch(
        user_id=user_id,
        customer_id=customer_id,
        conversation_id=conversation_id,
        search_query=[query],
        max_results=max_results,
        types=types,
        mode=mode,
    )
    append_step(timings, "context_fetch", step)

    step = time.perf_counter()
    results = flatten_context_items(context)
    append_step(timings, "map_context_results", step)

    return {
        "success": True,
        "latencyMs": ms_since(start),
        "results": results,
        "resultsCount": len(results),
        "rawResponse": context.raw if hasattr(context, "raw") else {},
        "source": context.metadata.source if context.metadata else "unknown",
        "bridgeTiming": {
            "python_total_ms": ms_since(handler_start),
            "steps": timings,
        },
    }


async def handle_get_memories(params: dict) -> dict:
    handler_start = time.perf_counter()
    timings: List[dict] = []

    user_id = params["user_id"]
    customer_id = params.get("customer_id")
    mode = params.get("mode", "fast")
    conversation_id = params.get("conversation_id")
    max_results = params.get("max_results", 100)
    types = params.get("types", ["all"])

    start = time.perf_counter()

    step = time.perf_counter()
    context = await sdk.user.context.fetch(
        user_id=user_id,
        customer_id=customer_id,
        conversation_id=conversation_id,
        search_query=[],
        max_results=max_results,
        types=types,
        mode=mode,
    )
    append_step(timings, "context_fetch_all", step)

    step = time.perf_counter()
    memories = flatten_context_items(context)
    append_step(timings, "map_memories", step)

    return {
        "success": True,
        "latencyMs": ms_since(start),
        "memories": memories,
        "memoriesCount": len(memories),
        "totalCount": len(memories),
        "rawResponse": context.raw if hasattr(context, "raw") else {},
        "source": context.metadata.source if context.metadata else "unknown",
        "bridgeTiming": {
            "python_total_ms": ms_since(handler_start),
            "steps": timings,
        },
    }


async def handle_fetch_user_context(params: dict) -> dict:
    handler_start = time.perf_counter()
    timings: List[dict] = []
    start = time.perf_counter()

    step = time.perf_counter()
    context = await sdk.user.context.fetch(
        user_id=params["user_id"],
        customer_id=params.get("customer_id"),
        conversation_id=params.get("conversation_id"),
        search_query=params.get("search_query"),
        max_results=params.get("max_results", 10),
        types=params.get("types"),
        mode=params.get("mode", "fast"),
    )
    append_step(timings, "context_fetch", step)

    return {
        "success": True,
        "latencyMs": ms_since(start),
        "context": serialize_context_response(context),
        "bridgeTiming": {
            "python_total_ms": ms_since(handler_start),
            "steps": timings,
        },
    }


async def handle_fetch_customer_context(params: dict) -> dict:
    handler_start = time.perf_counter()
    timings: List[dict] = []
    start = time.perf_counter()

    step = time.perf_counter()
    context = await sdk.customer.context.fetch(
        customer_id=params["customer_id"],
        conversation_id=params.get("conversation_id"),
        search_query=params.get("search_query"),
        max_results=params.get("max_results", 10),
        types=params.get("types"),
        mode=params.get("mode", "fast"),
    )
    append_step(timings, "context_fetch", step)

    return {
        "success": True,
        "latencyMs": ms_since(start),
        "context": serialize_context_response(context),
        "bridgeTiming": {
            "python_total_ms": ms_since(handler_start),
            "steps": timings,
        },
    }


async def handle_fetch_client_context(params: dict) -> dict:
    handler_start = time.perf_counter()
    timings: List[dict] = []
    start = time.perf_counter()

    step = time.perf_counter()
    context = await sdk.client.context.fetch(
        conversation_id=params.get("conversation_id"),
        search_query=params.get("search_query"),
        max_results=params.get("max_results", 10),
        types=params.get("types"),
        mode=params.get("mode", "fast"),
    )
    append_step(timings, "context_fetch", step)

    return {
        "success": True,
        "latencyMs": ms_since(start),
        "context": serialize_context_response(context),
        "bridgeTiming": {
            "python_total_ms": ms_since(handler_start),
            "steps": timings,
        },
    }


async def handle_get_context_for_prompt(params: dict) -> dict:
    handler_start = time.perf_counter()
    timings: List[dict] = []
    start = time.perf_counter()

    step = time.perf_counter()
    response = await sdk.conversation.get_context_for_prompt(
        conversation_id=params["conversation_id"],
        style=params.get("style", "structured"),
    )
    append_step(timings, "get_context_for_prompt", step)

    return {
        "success": True,
        "latencyMs": ms_since(start),
        "context_for_prompt": serialize_context_for_prompt_response(response),
        "bridgeTiming": {
            "python_total_ms": ms_since(handler_start),
            "steps": timings,
        },
    }


async def handle_delete_memory(params: dict) -> dict:
    handler_start = time.perf_counter()
    timings: List[dict] = []

    user_id = params["user_id"]
    customer_id = params.get("customer_id")
    memory_id = params.get("memory_id")

    start = time.perf_counter()

    step = time.perf_counter()
    if memory_id:
        await sdk.memories.delete(UUID(memory_id))
        append_step(timings, "delete_single_memory", step)
        return {
            "success": True,
            "latencyMs": ms_since(start),
            "deletedCount": 1,
            "rawResponse": {"deleted": 1},
            "bridgeTiming": {
                "python_total_ms": ms_since(handler_start),
                "steps": timings,
            },
        }

    if not customer_id:
        raise ValueError("customer_id is required when memory_id is not provided")

    tracked_ids = user_memory_ids.get(tracking_key(user_id, customer_id), [])
    if not tracked_ids:
        return {
            "success": True,
            "latencyMs": 0,
            "deletedCount": 0,
            "rawResponse": None,
            "note": "No tracked memory IDs for this user",
            "bridgeTiming": {
                "python_total_ms": ms_since(handler_start),
                "steps": timings,
            },
        }

    last_error: Optional[str] = None

    step = time.perf_counter()
    for tracked_id in tracked_ids:
        try:
            await sdk.memories.delete(UUID(tracked_id))
        except Exception as exc:
            last_error = str(exc)
    append_step(timings, "delete_tracked_memories", step)

    user_memory_ids.pop(tracking_key(user_id, customer_id), None)

    if last_error:
        return {
            "success": False,
            "latencyMs": ms_since(start),
            "error": last_error,
            "bridgeTiming": {
                "python_total_ms": ms_since(handler_start),
                "steps": timings,
            },
        }

    return {
        "success": True,
        "latencyMs": ms_since(start),
        "deletedCount": len(tracked_ids),
        "rawResponse": {"deleted": len(tracked_ids)},
        "note": f"Deleted {len(tracked_ids)} memories",
        "bridgeTiming": {
            "python_total_ms": ms_since(handler_start),
            "steps": timings,
        },
    }


async def handle_shutdown(_params: dict) -> dict:
    handler_start = time.perf_counter()
    timings: List[dict] = []

    if sdk:
        step = time.perf_counter()
        try:
            await sdk.instance.stop_listening()
        except Exception:
            pass
        append_step(timings, "stop_listener", step)

        step = time.perf_counter()
        await sdk.shutdown()
        append_step(timings, "sdk_shutdown", step)
    return {
        "success": True,
        "bridgeTiming": {
            "python_total_ms": ms_since(handler_start),
            "steps": timings,
        },
    }


async def handle_record_message(params: dict) -> dict:
    """Record a single conversation turn (registers the conversation_id).

    Mirrors sdk.conversation.record_message(...) so the JS client can expose
    the same namespaced surface the integration packages duck-type against.
    """
    handler_start = time.perf_counter()
    timings: List[dict] = []
    start = time.perf_counter()
    step = time.perf_counter()
    result = await sdk.conversation.record_message(
        conversation_id=params["conversation_id"],
        role=params["role"],
        content=params["content"],
        user_id=params["user_id"],
        customer_id=params["customer_id"],
        session_id=params.get("session_id"),
        metadata=params.get("metadata"),
    )
    append_step(timings, "record_message", step)
    return {
        "success": True,
        "latencyMs": ms_since(start),
        "result": result,
        "bridgeTiming": {"python_total_ms": ms_since(handler_start), "steps": timings},
    }


async def handle_create_memory(params: dict) -> dict:
    """Create a durable memory from a single document string.

    Mirrors sdk.memories.create(document=...). Distinct from add_memory,
    which builds a transcript from a messages array.
    """
    handler_start = time.perf_counter()
    timings: List[dict] = []
    start = time.perf_counter()
    create_kwargs: dict = {"document": params["document"]}
    for key in ("user_id", "customer_id", "document_type", "document_id",
                "document_created_at", "mode", "metadata"):
        if params.get(key) is not None:
            create_kwargs[key] = params[key]
    step = time.perf_counter()
    result = await sdk.memories.create(**create_kwargs)
    append_step(timings, "memories_create", step)
    status = result.status.value if hasattr(result.status, "value") else result.status
    return {
        "success": True,
        "latencyMs": ms_since(start),
        "result": {
            "ingestion_id": str(result.ingestion_id),
            "document_id": result.document_id,
            "status": status,
        },
        "bridgeTiming": {"python_total_ms": ms_since(handler_start), "steps": timings},
    }


HANDLERS = {
    "init": handle_init,
    "add_memory": handle_add_memory,
    "create_memory": handle_create_memory,
    "record_message": handle_record_message,
    "search_memory": handle_search_memory,
    "get_memories": handle_get_memories,
    "fetch_user_context": handle_fetch_user_context,
    "fetch_customer_context": handle_fetch_customer_context,
    "fetch_client_context": handle_fetch_client_context,
    "get_context_for_prompt": handle_get_context_for_prompt,
    "delete_memory": handle_delete_memory,
    "shutdown": handle_shutdown,
}


async def main() -> None:
    logger.info("Synap bridge starting")

    reader = asyncio.StreamReader()
    protocol = asyncio.StreamReaderProtocol(reader)
    await asyncio.get_event_loop().connect_read_pipe(lambda: protocol, sys.stdin)

    while True:
        line = await reader.readline()
        if not line:
            logger.info("stdin closed; shutting down bridge")
            if sdk and getattr(sdk, "_initialized", False):
                await handle_shutdown({})
            break

        payload = line.decode().strip()
        if not payload:
            continue

        try:
            request = json.loads(payload)
        except json.JSONDecodeError as exc:
            write_response({"id": None, "result": None, "error": f"Invalid JSON: {exc}"})
            continue

        req_id = request.get("id")
        method = request.get("method")
        params = request.get("params", {})

        handler = HANDLERS.get(method)
        if not handler:
            write_response({"id": req_id, "result": None, "error": f"Unknown method: {method}"})
            continue

        try:
            handler_started = time.perf_counter()
            result = await handler(params)
            if isinstance(result, dict):
                bridge_timing = result.get("bridgeTiming")
                if isinstance(bridge_timing, dict):
                    bridge_timing.setdefault("python_total_ms", ms_since(handler_started))
                    bridge_timing.setdefault("steps", [])
                else:
                    result["bridgeTiming"] = {
                        "python_total_ms": ms_since(handler_started),
                        "steps": [],
                    }
            write_response({"id": req_id, "result": result, "error": None})
        except Exception as exc:
            logger.error("Handler error for %s: %s", method, traceback.format_exc())
            write_response({"id": req_id, "result": None, "error": str(exc), "error_type": type(exc).__name__})


if __name__ == "__main__":
    asyncio.run(main())
