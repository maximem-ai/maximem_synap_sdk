/**
 * gRPC anticipation stream: availability and loading.
 *
 * This is a SUBPATH export (`@maximem/synap-js-sdk/grpc`) and is never
 * imported by the main entry point. Three reasons, all of which have bitten
 * someone already:
 *
 *  1. `@grpc/grpc-js` plus its protos is roughly 5MB. HTTP-only and Edge users
 *     should not pay for it (gotcha G-Q).
 *  2. grpc-js is built on `node:http2`, `node:net` and `node:tls`. Merely
 *     importing it on Edge or Workers fails the *build*, not the call. Keeping
 *     it behind a subpath plus a lazy `import()` means importing the SDK on
 *     Edge stays safe.
 *  3. It is an `optionalDependency`, so it may legitimately be absent.
 *
 * ## Proto loading
 *
 * The stream is live. The proto ships as an INLINED JSON descriptor
 * (./descriptor.ts) loaded through `proto-loader.fromJSON`, not through
 * `loadSync(path)`. proto-loader's file-based path reads the `.proto` off disk
 * at runtime, which breaks under webpack/turbopack bundling and Vercel's `nft`
 * file tracing: at deploy time rather than at test time (gotcha G-P). Inlining
 * gets the same result as ts-proto codegen without a codegen toolchain.
 */

import { ListeningNotActiveError } from '../errors.js';
export { GrpcStreamClient } from './stream-client.js';
export type {
  StreamState, StreamCredentials, StreamClientOptions, ConversationEventInput,
} from './stream-client.js';
export { PROTO_SHA } from './descriptor.js';

export type GrpcAvailability =
  | { available: true }
  | { available: false; reason: 'not-node' | 'module-missing'; detail: string };

/**
 * Whether a gRPC stream could run here.
 *
 * Deliberately does not throw: callers use this to decide whether to offer the
 * stream at all, and a hard failure at that point is unhelpful.
 */
export async function checkGrpcAvailability(): Promise<GrpcAvailability> {
  const isNode =
    typeof process !== 'undefined' &&
    process.versions?.node !== undefined &&
    typeof (globalThis as { EdgeRuntime?: unknown }).EdgeRuntime === 'undefined';

  if (!isNode) {
    return {
      available: false,
      reason: 'not-node',
      detail:
        'gRPC needs raw TCP and node:http2, which Edge runtimes and Workers do ' +
        'not provide. HTTP context fetching works normally here; only the ' +
        'anticipation stream is unavailable. Reaching Edge would require a ' +
        'gRPC-Web or Connect endpoint on the server, which does not exist yet.',
    };
  }

  try {
    await import(/* @vite-ignore */ '@grpc/grpc-js');
    return { available: true };
  } catch (error) {
    return {
      available: false,
      reason: 'module-missing',
      detail:
        '@grpc/grpc-js is an optional dependency and is not installed. Run ' +
        '`npm install @grpc/grpc-js` to enable the anticipation stream. ' +
        `(${(error as Error)?.message ?? String(error)})`,
    };
  }
}

export interface ListenOptions {
  conversationId: string;
  userId?: string;
  customerId?: string;
  onBundle?: (bundle: unknown) => void;
}

export interface ListeningSession {
  stop(): Promise<void>;
  readonly active: boolean;
}

/**
 * Start the anticipation stream directly, without a SynapClient.
 *
 * Most callers want `client.instance.listen()` instead, which shares the
 * client's credentials and anticipation cache. This exists for the case where
 * only the stream is wanted.
 *
 * **Opt-in, and off by default.** The removed wrapper called `listen()`
 * unconditionally during init, so every user had a stream whether they asked
 * for one or not. Making it opt-in is the right default, but it is not a
 * neutral change: fewer streams means fewer anticipation hits and therefore
 * MORE billed cloud fetches for existing users. That has to be announced
 * rather than discovered (gotcha G-Y).
 */
export async function listen(
  options: ListenOptions & {
    apiKey?: string;
    clientId?: string;
    instanceId?: string;
    host?: string;
    port?: number;
    useTls?: boolean;
  },
): Promise<ListeningSession> {
  if (!options.conversationId) {
    // conversation_id is not a scope tier, but the stream genuinely requires
    // one: it is what the server keys the anticipation session on.
    throw new ListeningNotActiveError('conversation_id is required to start a listening stream');
  }

  const availability = await checkGrpcAvailability();
  if (!availability.available) {
    throw new ListeningNotActiveError(availability.detail);
  }

  const [{ GrpcStreamClient: StreamClient }, { AnticipationCache }, { getEnv }] = await Promise.all([
    import('./stream-client.js'),
    import('../context/anticipation-cache.js'),
    import('../util/env.js'),
  ]);

  const apiKey = options.apiKey ?? getEnv('SYNAP_API_KEY') ?? '';
  if (apiKey === '') {
    throw new ListeningNotActiveError(
      'No Synap API key found. Set SYNAP_API_KEY or pass apiKey.',
    );
  }

  const cache = new AnticipationCache();
  const client = new StreamClient(
    {
      apiKey,
      clientId: options.clientId ?? getEnv('SYNAP_CLIENT_ID') ?? '',
      instanceId: options.instanceId ?? getEnv('SYNAP_INSTANCE_ID') ?? '',
    },
    cache,
    {
      ...(options.host !== undefined ? { host: options.host } : {}),
      ...(options.port !== undefined ? { port: options.port } : {}),
      ...(options.useTls !== undefined ? { useTls: options.useTls } : {}),
      ...(options.onBundle !== undefined ? { onContext: options.onBundle } : {}),
    },
  );
  await client.connect();
  client.sendSessionControl({
    action: 'start',
    conversation_id: options.conversationId,
    ...(options.userId !== undefined ? { user_id: options.userId } : {}),
    ...(options.customerId !== undefined ? { customer_id: options.customerId } : {}),
  });

  return {
    stop: () => client.disconnect(),
    get active() { return client.isConnected; },
  };
}
