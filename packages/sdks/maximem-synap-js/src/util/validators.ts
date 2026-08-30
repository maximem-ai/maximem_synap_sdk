/**
 * Client-side identifier validation, mirroring Python's
 * ``maximem_synap/utils/validators.py``.
 *
 * These raise the *specific* ``InvalidInputError`` subtypes so callers can
 * catch them directly (``catch (e) { if (e instanceof InvalidConversationIdError) }``)
 * instead of the generic parent. Validation happens before any network call,
 * so a malformed id fails locally rather than becoming a billed round trip.
 *
 * Both errors were exported from day one and thrown at zero sites until this
 * module existed, which meant a typo'd conversation id reached the server and
 * came back as a generic 4xx.
 */

import { InvalidConversationIdError, InvalidInstanceIdError } from '../errors.js';

// Instance ids are issued as `inst_` followed by 16 hex characters.
const INSTANCE_ID_RE = /^inst_[0-9a-fA-F]{16}$/;

// Accepts any RFC 4122 shape. Deliberately laxer than a version-specific
// check: Python defers to `uuid.UUID(...)`, which likewise accepts any
// variant, and rejecting a valid-but-unexpected version here would diverge.
const UUID_RE = /^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$/;

/**
 * Throw `InvalidConversationIdError` if a non-empty `conversationId` is not a
 * UUID string.
 *
 * Empty / null / undefined values pass through untouched: callers that require
 * a conversation id surface that separately, so this only rejects a clearly
 * malformed value such as `"conv_123"`.
 */
export function validateConversationId(conversationId: unknown): void {
  if (conversationId === undefined || conversationId === null || conversationId === '') return;
  if (typeof conversationId !== 'string' || !UUID_RE.test(conversationId)) {
    // Message text matches Python's `f"Invalid conversation ID: {id}"` byte for byte.
    throw new InvalidConversationIdError(`Invalid conversation ID: ${String(conversationId)}`);
  }
}

/**
 * Throw `InvalidInstanceIdError` if a non-empty `instanceId` is not in the
 * `inst_<hex16>` format.
 *
 * Empty values pass through: the instance id is normally resolved from the API
 * key during `initialize()`, so only an explicitly provided malformed id is
 * rejected.
 */
export function validateInstanceId(instanceId: unknown): void {
  if (instanceId === undefined || instanceId === null || instanceId === '') return;
  if (typeof instanceId !== 'string' || !INSTANCE_ID_RE.test(instanceId)) {
    throw new InvalidInstanceIdError(`Invalid instance ID: ${String(instanceId)}`);
  }
}
