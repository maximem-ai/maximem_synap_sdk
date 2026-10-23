import { wrapLanguageModel, type LanguageModelV1 } from 'ai';
import { CredentialManager } from './auth/credential-manager.js';
import { AnticipationCache } from './context/anticipation-cache.js';
import { createSynapMiddleware } from './middleware.js';
import { StreamManager } from './stream.js';
import type { SynapProviderOptions, SynapModelOptions, Credentials } from './types.js';

let warnedSpec = false;

export class SynapProvider {
  private readonly credManager: CredentialManager;
  private readonly cache: AnticipationCache;
  private readonly opts: SynapProviderOptions;
  private readonly streams: StreamManager;

  private credentials: Credentials | null = null;

  constructor(opts: SynapProviderOptions) {
    this.opts = opts;
    this.credManager = new CredentialManager(undefined, opts.baseUrl);
    this.cache = new AnticipationCache();
    this.streams = new StreamManager(() => this.credentials, this.cache, opts);
  }

  /**
   * Wrap any Vercel AI SDK LanguageModelV1 with Synap context injection.
   *
   * Usage:
   *   const model = provider.wrap(anthropic('claude-sonnet-4-6'), { userId: 'u_123' });
   *   const { text } = await generateText({ model, messages });
   *
   * With a conversationId, once Synap has compacted that conversation the
   * model sees Synap's compaction in place of the earlier turns in `messages`.
   */
  wrap(model: LanguageModelV1, modelOptions: SynapModelOptions = {}): LanguageModelV1 {
    const creds = this.credentials;

    if (!creds) {
      throw new Error(
        'SynapProvider not initialized. Call await provider.init() before wrapping a model.'
      );
    }

    if (!modelOptions.userId && !modelOptions.customerId && !modelOptions.conversationId) {
      throw new Error(
        'wrap() requires at least one of: userId, customerId, conversationId. ' +
        'Without an identity field, context fetch and memory writes are no-ops.'
      );
    }

    const spec = (model as { specificationVersion?: string }).specificationVersion;
    if (spec && spec !== 'v1' && !warnedSpec) {
      warnedSpec = true;
      console.warn(
        `[synap] this model implements specification ${spec}; the middleware is written for `
        + 'AI SDK 3/4 (LanguageModelV1). Context is injected, but replies, tool calls and '
        + 'reasoning are not recorded.',
      );
    }

    return wrapLanguageModel({
      model,
      middleware: createSynapMiddleware({
        // Synap sizes compaction to the agent model's context window
        // (COMPACTION_V3_PLAN.md §4.3); an explicit option wins.
        agentModel: model.modelId,
        ...modelOptions,
        credentials: creds,
        anticipationCache: this.cache,
        // Read at call time, not now: a model wrapped before listen(), or across
        // stopListening() + listen(), still reaches the live stream.
        streams: this.streams,
        baseUrl: this.opts.baseUrl,
      }),
    });
  }

  /**
   * Initialize credentials from the explicit API key or SYNAP_API_KEY.
   * Must be called once before wrap() or listen().
   */
  async init(): Promise<this> {
    this.credentials = await this.credManager.load(
      this.opts.apiKey,
    );
    return this;
  }

  /**
   * Open the gRPC bidirectional stream now.
   *
   * Optional: the first model call opens it on its own (set
   * `SYNAP_AUTO_LISTEN=false` or `autoListen: false` to stop that). Call this
   * at app startup to have it ready before the first request, e.g. in Next.js
   * instrumentation.ts.
   *
   * Node.js only. Safe in Edge environments: it silently no-ops and everything
   * goes over REST.
   */
  async listen(): Promise<void> {
    if (!this.credentials) {
      throw new Error('Call await provider.init() before provider.listen()');
    }

    // Guard: don't attempt gRPC in Edge / browser where dynamic import will fail
    if (typeof process === 'undefined' || process.versions?.node == null) {
      return;
    }

    await this.streams.listen();
  }

  /**
   * Gracefully close the gRPC stream. It stays closed until listen() is
   * called again; the first model call does not reopen it.
   */
  async stopListening(): Promise<void> {
    await this.streams.stop();
  }

  /** Whether the gRPC anticipation stream is currently connected. */
  get isListening(): boolean {
    return this.streams.current !== null;
  }

  /** Current anticipation cache size (diagnostic). */
  get cacheSize(): number {
    return this.cache.size();
  }
}

// ─── Factory function — primary public API ────────────────────────────────────

/**
 * Create a SynapProvider instance.
 *
 * @example
 * // app/lib/synap.ts
 * import { createSynap } from '@maximem/synap-vercel-adk';
 *
 * export const synap = await createSynap({
 *   apiKey: process.env.SYNAP_API_KEY,
 * });
 *
 * // The gRPC stream opens on the first model call. To have it ready earlier
 * // (Node.js only, optional):
 * await synap.listen();
 */
export async function createSynap(opts: SynapProviderOptions): Promise<SynapProvider> {
  const provider = new SynapProvider(opts);
  await provider.init();
  return provider;
}
