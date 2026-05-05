import { wrapLanguageModel, type LanguageModelV1 } from 'ai';
import { CredentialManager } from './auth/credential-manager.js';
import { AnticipationCache } from './context/anticipation-cache.js';
import { createSynapMiddleware } from './middleware.js';
import type { SynapProviderOptions, SynapModelOptions, Credentials } from './types.js';

// Lazy import — only loaded in Node.js environments, never in Edge/browser
type GrpcStreamClientModule = typeof import('./grpc/stream-client.js');

export class SynapProvider {
  private readonly credManager: CredentialManager;
  private readonly cache: AnticipationCache;
  private readonly opts: SynapProviderOptions;

  private credentials: Credentials | null = null;
  private grpcClient: import('./grpc/stream-client.js').GrpcStreamClient | null = null;
  private grpcMod: GrpcStreamClientModule | null = null;

  constructor(opts: SynapProviderOptions) {
    this.opts = opts;
    this.credManager = new CredentialManager(undefined, opts.baseUrl);
    this.cache = new AnticipationCache();
  }

  /**
   * Wrap any Vercel AI SDK LanguageModelV1 with Synap context injection.
   *
   * Usage:
   *   const model = provider.wrap(anthropic('claude-sonnet-4-6'), { userId: 'u_123' });
   *   const { text } = await generateText({ model, messages });
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

    return wrapLanguageModel({
      model,
      middleware: createSynapMiddleware({
        ...modelOptions,
        credentials: creds,
        anticipationCache: this.cache,
        grpcClient: this.grpcClient,
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
   * Open the gRPC bidirectional stream for anticipation cache updates.
   *
   * Node.js only — call this once at app startup (e.g. in Next.js instrumentation.ts).
   * Safe to call in Edge environments — it will silently no-op.
   */
  async listen(): Promise<void> {
    if (!this.credentials) {
      throw new Error('Call await provider.init() before provider.listen()');
    }

    // Guard: don't attempt gRPC in Edge / browser where dynamic import will fail
    if (typeof process === 'undefined' || process.versions?.node == null) {
      return;
    }

    try {
      this.grpcMod ??= await import('./grpc/stream-client.js');
      this.grpcClient = new this.grpcMod.GrpcStreamClient(
        this.credentials,
        this.cache,
        (state) => {
          if (state === 'disconnected') {
            // Non-fatal — HTTP context fetch fallback remains active
          }
        },
        {
          host: this.opts.grpcHost,
          port: this.opts.grpcPort,
          useTls: this.opts.grpcUseTls,
        },
      );
      await this.grpcClient.connect();
    } catch (err) {
      console.warn('[synap] gRPC listen() failed — falling back to HTTP context fetch:', err);
      this.grpcClient = null;
    }
  }

  /** Gracefully close the gRPC stream. */
  async stopListening(): Promise<void> {
    await this.grpcClient?.disconnect();
    this.grpcClient = null;
  }

  /** Whether the gRPC anticipation stream is currently connected. */
  get isListening(): boolean {
    return this.grpcClient?.isConnected ?? false;
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
 * // Start gRPC listener for anticipation cache (Node.js only, optional)
 * await synap.listen();
 */
export async function createSynap(opts: SynapProviderOptions): Promise<SynapProvider> {
  const provider = new SynapProvider(opts);
  await provider.init();
  return provider;
}
