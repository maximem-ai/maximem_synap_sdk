/**
 * `client.as_tool(...)`: an LLM-ready tool definition for fetching context.
 *
 * Ports Python's `MaximemSynapSDK.as_tool` plus its three module helpers.
 *
 * The point of the helper is that the scope identifiers are CLOSED OVER rather
 * than exposed in the JSON schema. The LLM chooses the query and the limits; it
 * cannot choose whose memory to read. Handing the model a `user_id` parameter
 * is how the per-user privacy filter gets dropped by accident, so the schema
 * deliberately has no such field.
 */

import { InvalidInputError } from '../errors.js';
import type { Json } from '../context/types.js';

export type ToolScope = 'conversation' | 'user' | 'customer' | 'client' | 'unified';
export type ToolStyle = 'openai' | 'anthropic';

const VALID_SCOPES: readonly ToolScope[] = ['conversation', 'user', 'customer', 'client', 'unified'];

export interface AsToolOptions {
  scope?: string;
  user_id?: string;
  userId?: string;
  customer_id?: string;
  customerId?: string;
  conversation_id?: string;
  conversationId?: string;
  name?: string;
  description?: string;
  style?: string;
}

export type ToolHandler = (callArgs?: Json) => Promise<Json>;

export interface OpenAiToolDefinition {
  type: 'function';
  function: { name: string; description: string; parameters: Json };
  handler: ToolHandler;
}

export interface AnthropicToolDefinition {
  name: string;
  description: string;
  input_schema: Json;
  handler: ToolHandler;
}

export type ToolDefinition = OpenAiToolDefinition | AnthropicToolDefinition;

const BASE_DESCRIPTION =
  'Retrieve stored context (facts, preferences, recent episodes, ' +
  'emotions, temporal events) from Synap memory.';

/** Python's `_default_tool_description`. Wording is load-bearing: it primes the model to call the tool. */
export function defaultToolDescription(scope: ToolScope): string {
  switch (scope) {
    case 'conversation':
      return (
        BASE_DESCRIPTION +
        ' Scoped to a specific conversation. Call this when you ' +
        'need facts already established in the current conversation, ' +
        'or to reload context after a topic shift.'
      );
    case 'user':
      return (
        BASE_DESCRIPTION +
        " Scoped to the user's long-term memory. Call this when " +
        'you need to recall who the user is, their preferences, or ' +
        'their history across conversations.'
      );
    case 'customer':
      return (
        BASE_DESCRIPTION +
        ' Scoped to the customer/organization. Call this when ' +
        'you need facts about the customer org rather than an individual.'
      );
    case 'client':
      return (
        BASE_DESCRIPTION +
        ' Scoped to the integrating client/product. Call this for ' +
        'product-level knowledge (e.g. policies, documentation).'
      );
    default:
      return (
        BASE_DESCRIPTION +
        ' Cross-scope: merges conversation, user, customer, and ' +
        'client memory in one call. Call this at the start of a turn when ' +
        'you need general grounding before responding.'
      );
  }
}

/**
 * Python's `_tool_input_schema`.
 *
 * `conversation_id` appears ONLY for `scope="conversation"` when it was not
 * closed over, in which case it is also the one required field.
 */
export function toolInputSchema(scope: ToolScope, options: { hasConversationId: boolean }): Json {
  const props: Json = {
    search_query: {
      type: 'array',
      items: { type: 'string' },
      description:
        'Optional list of search queries describing what context ' +
        'you need. Free-form natural language is fine. Omit to ' +
        'retrieve the most relevant recent context.',
    },
    max_results: {
      type: 'integer',
      minimum: 1,
      maximum: 50,
      description: 'Maximum items to return (default 10).',
    },
    types: {
      type: 'array',
      items: {
        type: 'string',
        enum: ['facts', 'preferences', 'episodes', 'emotions', 'temporal_events'],
      },
      description: 'Filter to specific memory categories. Omit to retrieve all.',
    },
    mode: {
      type: 'string',
      enum: ['fast', 'accurate'],
      description:
        "Retrieval mode. 'fast' = low-latency (~50ms); 'accurate' = " +
        "LLM-decomposed multi-query (~200-500ms). Default 'fast'.",
    },
    precision_level: {
      type: 'string',
      enum: ['high', 'medium'],
      description:
        "Result filtering precision. 'high' (default) applies an extra " +
        "relevance-refinement pass; 'medium' skips it for faster, less " +
        'precisely filtered results (recall unaffected).',
    },
  };

  let required: string[] = [];
  if (scope === 'conversation' && !options.hasConversationId) {
    props['conversation_id'] = {
      type: 'string',
      description: 'The conversation id to fetch context for.',
    };
    required = ['conversation_id'];
  }

  return { type: 'object', properties: props, required, additionalProperties: false };
}

/** What `as_tool` needs from the client, so this module stays testable in isolation. */
export interface ToolDispatchTarget {
  fetch(options: Json): Promise<{
    formatted_context: string;
    scopes_queried: string[];
    total_items: number;
  }>;
  conversation: { context: { fetch(options: Json): Promise<Json> } };
  user: { context: { fetch(options: Json): Promise<Json> } };
  customer: { context: { fetch(options: Json): Promise<Json> } };
  client: { context: { fetch(options: Json): Promise<Json> } };
}

const RESULT_COLLECTIONS = ['facts', 'preferences', 'episodes', 'emotions', 'temporal_events'] as const;

/**
 * Python's `_invoke_scope_fetch`.
 *
 * Returns a plain object so the host runtime can JSON-serialise it straight
 * into the LLM's tool-result message.
 */
export async function invokeScopeFetch(params: {
  target: ToolDispatchTarget;
  scope: ToolScope;
  userId: string | undefined;
  customerId: string | undefined;
  conversationId: string | undefined;
  callArgs: Json;
}): Promise<Json> {
  const { target, scope, userId, customerId, conversationId, callArgs } = params;

  const searchQuery = callArgs['search_query'];
  // Python: `int(call_args.get("max_results") or 10)` -- so 0 falls back to 10.
  const rawMax = callArgs['max_results'];
  const maxResults = typeof rawMax === 'number' && rawMax !== 0 ? Math.trunc(rawMax) : 10;
  const types = callArgs['types'];
  const mode = String(callArgs['mode'] ?? '') || 'fast';
  const precisionLevel = String(callArgs['precision_level'] ?? '') || 'high';
  const callConvId =
    (typeof callArgs['conversation_id'] === 'string' && callArgs['conversation_id'] !== ''
      ? (callArgs['conversation_id'] as string)
      : undefined) ?? conversationId;

  const shared: Json = {
    search_query: searchQuery,
    max_results: maxResults,
    types,
    mode,
    precision_level: precisionLevel,
  };

  if (scope === 'unified') {
    const response = await target.fetch({
      conversation_id: callConvId,
      user_id: userId,
      customer_id: customerId,
      ...shared,
    });
    return {
      formatted_context: response.formatted_context,
      scopes_queried: response.scopes_queried,
      total_items: response.total_items,
    };
  }

  let response: Json;
  if (scope === 'conversation') {
    if (callConvId === undefined || callConvId === '') {
      // Python returns this as DATA, not an exception: the LLM sees the error
      // string in its tool result and can retry with an id.
      return { error: 'conversation_id is required', items: [] };
    }
    response = await target.conversation.context.fetch({
      conversation_id: callConvId,
      ...shared,
      user_id: userId,
    });
  } else if (scope === 'user') {
    response = await target.user.context.fetch({
      user_id: userId,
      conversation_id: callConvId,
      ...shared,
      customer_id: customerId,
    });
  } else if (scope === 'customer') {
    response = await target.customer.context.fetch({
      customer_id: customerId,
      conversation_id: callConvId,
      ...shared,
    });
  } else {
    response = await target.client.context.fetch({
      conversation_id: callConvId,
      ...shared,
    });
  }

  const out: Json = {};
  for (const collection of RESULT_COLLECTIONS) {
    out[collection] = Array.isArray(response[collection]) ? response[collection] : [];
  }
  return out;
}

/**
 * Build the tool definition.
 *
 * `onWarning` receives Python's `logger.warning` for the
 * `scope="conversation"` without `user_id` case, which is a soft warning
 * rather than an error.
 */
export function buildTool(
  target: ToolDispatchTarget,
  options: AsToolOptions = {},
  onWarning: (message: string) => void = () => {},
): ToolDefinition {
  const scope = String(options.scope ?? 'user').toLowerCase() as ToolScope;
  if (!VALID_SCOPES.includes(scope)) {
    // Python formats the valid set with sorted() and repr() on the input.
    const sorted = [...VALID_SCOPES].sort();
    throw new InvalidInputError(
      `scope must be one of ['${sorted.join("', '")}'], got '${String(options.scope)}'`,
    );
  }

  const userId = options.user_id ?? options.userId;
  const customerId = options.customer_id ?? options.customerId;
  const conversationId = options.conversation_id ?? options.conversationId;

  if (scope === 'user' && (userId === undefined || userId === '')) {
    throw new InvalidInputError("scope='user' requires user_id");
  }
  if (scope === 'customer' && (customerId === undefined || customerId === '')) {
    throw new InvalidInputError("scope='customer' requires customer_id");
  }
  if (scope === 'conversation' && (userId === undefined || userId === '')) {
    // Not a hard error: the anticipation cache refuses cross-user matches
    // anyway. But the caller is leaving privacy on the floor, so say so.
    onWarning(
      "as_tool(scope='conversation') without user_id: the SDK " +
        'anticipation cache cannot apply per-user filtering. Pass ' +
        'user_id to enable the Section 15 privacy guarantee.',
    );
  }

  const style = String(options.style ?? 'openai');
  if (style !== 'openai' && style !== 'anthropic') {
    throw new InvalidInputError(`style must be 'openai' or 'anthropic', got '${style}'`);
  }

  const toolName = options.name ?? `synap_fetch_${scope}_context`;
  const toolDesc = options.description ?? defaultToolDescription(scope);
  const schema = toolInputSchema(scope, { hasConversationId: conversationId !== undefined });

  const handler: ToolHandler = (callArgs: Json = {}) =>
    invokeScopeFetch({ target, scope, userId, customerId, conversationId, callArgs });

  if (style === 'openai') {
    return {
      type: 'function',
      function: { name: toolName, description: toolDesc, parameters: schema },
      handler,
    };
  }
  return { name: toolName, description: toolDesc, input_schema: schema, handler };
}
