/**
 * Which model the agent runs on, sent with each conversation turn.
 *
 * The server sizes compaction to the agent's context window: it compacts at
 * about 11.7% of the window (15K tokens on a 128K model) and caps the
 * compacted record at half that. It reads the window from the latest turn's
 * metadata, under `agent_model` and `agent_context_window`. A model name is
 * enough: the server has a model -> window table. Give `context_window` only
 * for a model the table would not know, or to override it.
 *
 * Precedence, highest first: the value passed on the call, the client option
 * (`agent_model` / `agent_context_window`), a value already in the caller's
 * own metadata. With none of them the server assumes 128K. Mirrors Python's
 * `agent_model.py`.
 */
export const AGENT_MODEL_KEY = 'agent_model';
export const AGENT_CONTEXT_WINDOW_KEY = 'agent_context_window';

export interface AgentModel {
  model?: string | undefined;
  contextWindow?: number | undefined;
}

/** The per-call `model` / `context_window` (either spelling) off an options object. */
export function agentModelOf(options: unknown): AgentModel {
  const o = (options ?? {}) as Record<string, unknown>;
  const window = o['context_window'] ?? o['contextWindow'];
  return {
    model: typeof o['model'] === 'string' ? (o['model'] as string) : undefined,
    contextWindow: typeof window === 'number' ? window : undefined,
  };
}

/**
 * `metadata` with the agent's model folded in. `asStrings` for the gRPC
 * stream, whose metadata is a string map. Returns the input unchanged when
 * there is nothing to add.
 */
export function withAgentModel<T extends Record<string, unknown>>(
  metadata: T | undefined,
  perCall: AgentModel,
  defaults: AgentModel,
  asStrings = false,
): T | undefined {
  const model = perCall.model || defaults.model;
  const window = perCall.contextWindow || defaults.contextWindow;
  if (!model && !window) return metadata;
  const out: Record<string, unknown> = { ...(metadata ?? {}) };
  if (model) out[AGENT_MODEL_KEY] = String(model);
  if (window) out[AGENT_CONTEXT_WINDOW_KEY] = asStrings ? String(Math.trunc(window)) : Math.trunc(window);
  return out as T;
}
