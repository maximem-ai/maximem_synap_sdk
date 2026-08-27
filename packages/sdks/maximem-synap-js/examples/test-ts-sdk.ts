/**
 * Per-method exerciser for the native Synap SDK in TypeScript.
 *
 *   npx tsc --noEmit          # compile-time checks: the types themselves
 *   npx tsx test-ts-sdk.ts    # runtime checks
 *
 * This is NOT a retyped copy of test-js-sdk.mjs. It tests the thing the JS
 * script structurally cannot: the shipped `.d.ts`. Roughly half of what
 * follows is assertions that never execute, because their whole job is to fail
 * `tsc`. A published SDK can pass every runtime test and still be unusable in
 * a strict TypeScript project.
 *
 * tsconfig.json here is deliberately strict (`exactOptionalPropertyTypes`,
 * `noUncheckedIndexedAccess`, `skipLibCheck: false`). If the SDK's types only
 * hold up under loose settings, that should fail here rather than in a
 * customer's repo.
 */
import { randomUUID } from 'node:crypto';

import {
  SynapClient,
  SynapError,
  SynapTransientError,
  SynapPermanentError,
  InvalidInputError,
  InvalidConversationIdError,
  InvalidInstanceIdError,
  ContextNotFoundError,
  ConflictError,
  TranscriptConflictError,
  RateLimitError,
  ListeningAlreadyActiveError,
  isSynapError,
  AnticipationCache,
  BM25,
  tokenize,
  stem,
  isRecallQuery,
  flattenContextItems,
  DEFAULT_DOCUMENT_TYPE,
  DEFAULT_INGEST_MODE,
  DOCUMENT_TYPES,
  INGEST_MODES,
  SDK_VERSION,
  // Types. These are the real deliverable of this file.
  type SynapClientOptions,
  type ConfigureOptions,
  type UnifiedFetchOptions,
  type CreateMemoryOptions,
  type CreateMemoryResult,
  type UpdateMemoryOptions,
  type CreateFromFileOptions,
  type RecordMessageOptions,
  type CompactOptions,
  type FetchOptions,
  type RawContext,
  type RawContextItem,
  type NormalisedContext,
  type Json,
  type DocumentType,
  type IngestMode,
  type FlatMemory,
  type UnifiedContext,
  type AnticipationCacheSnapshot,
  type LookupResult,
  type ToolDefinition,
  type AsToolOptions,
  type ListenOptions,
  type SendMessageOptions,
} from '@maximem/synap-js-sdk';

// The gRPC stream lives behind a subpath so HTTP-only and Edge builds never
// pay for grpc-js. Importing types from it must work under every
// moduleResolution mode, which is what broke once already (TS2307 under
// "node", fixed with typesVersions).
import {
  checkGrpcAvailability,
  GrpcStreamClient,
  PROTO_SHA,
  type GrpcAvailability,
  type StreamState,
  type StreamCredentials,
} from '@maximem/synap-js-sdk/grpc';

// ═══════════════════════════════════════════════════════════════════════════
// PART 1 — COMPILE-TIME ASSERTIONS
//
// Nothing below runs. Every line exists to fail `tsc` if a type regresses.
// ═══════════════════════════════════════════════════════════════════════════

/** Fails to compile unless T and U are mutually assignable. */
type Exact<T, U> = [T] extends [U] ? ([U] extends [T] ? true : never) : never;
function staticAssert<_T extends true>(): void {}

// ─── Return types are what the docs claim ───────────────────────────────────

declare const c: SynapClient;

staticAssert<Exact<ReturnType<typeof c.initialize>, Promise<void>>>();
staticAssert<Exact<ReturnType<typeof c.shutdown>, Promise<void>>>();
staticAssert<Exact<ReturnType<typeof c.configure>, void>>();
staticAssert<Exact<ReturnType<typeof c.fetch>, Promise<UnifiedContext>>>();
staticAssert<Exact<ReturnType<typeof c.as_tool>, ToolDefinition>>();
staticAssert<Exact<ReturnType<typeof c.anticipation_cache_snapshot>, AnticipationCacheSnapshot>>();
staticAssert<Exact<typeof c.instance_id, string>>();
staticAssert<Exact<typeof c.client_id, string>>();

// `is_listening` must be a boolean PROPERTY, not a method. It was a frozen
// literal at one point, which typechecked identically and was always false.
staticAssert<Exact<typeof c.instance.is_listening, boolean>>();

// get_compacted returns null rather than throwing when nothing is compacted,
// matching Python. A non-nullable return here would be a lie.
staticAssert<Exact<ReturnType<typeof c.conversation.context.get_compacted>, Promise<Json | null>>>();

// The deprecated top-level surface returns the NORMALISED camelCase shape,
// while the namespaced surface returns RAW snake_case. They differ on purpose.
staticAssert<Exact<ReturnType<typeof c.fetchUserContext>, Promise<NormalisedContext>>>();
staticAssert<Exact<ReturnType<typeof c.user.context.fetch>, Promise<RawContext>>>();

// An unsubscribe thunk, not void.
staticAssert<Exact<
  ReturnType<typeof c.conversation.context.subscribe_to_compaction_updates>,
  () => void
>>();
staticAssert<Exact<
  ReturnType<typeof c.conversation.context.unsubscribe_all_compaction_updates>,
  number
>>();

// ─── Every namespace method returns a Promise ───────────────────────────────
// A synchronous throw from a Promise-returning method bypasses `.catch()`, so
// "is it async" is a correctness property, not a style one.

type IsPromise<T> = T extends Promise<unknown> ? true : false;
staticAssert<IsPromise<ReturnType<typeof c.memories.create>>>();
staticAssert<IsPromise<ReturnType<typeof c.memories.batch_create>>>();
staticAssert<IsPromise<ReturnType<typeof c.memories.create_from_file>>>();
staticAssert<IsPromise<ReturnType<typeof c.memories.get>>>();
staticAssert<IsPromise<ReturnType<typeof c.memories.update>>>();
staticAssert<IsPromise<ReturnType<typeof c.memories.delete>>>();
staticAssert<IsPromise<ReturnType<typeof c.memories.status>>>();
staticAssert<IsPromise<ReturnType<typeof c.memories.wait_for_completion>>>();
staticAssert<IsPromise<ReturnType<typeof c.customer.context.fetch>>>();
staticAssert<IsPromise<ReturnType<typeof c.client.context.fetch>>>();
staticAssert<IsPromise<ReturnType<typeof c.conversation.record_message>>>();
staticAssert<IsPromise<ReturnType<typeof c.conversation.record_messages_batch>>>();
staticAssert<IsPromise<ReturnType<typeof c.conversation.ingest_transcript>>>();
staticAssert<IsPromise<ReturnType<typeof c.conversation.context.compact>>>();
staticAssert<IsPromise<ReturnType<typeof c.credits.get_balance>>>();
staticAssert<IsPromise<ReturnType<typeof c.credits.get_ledger>>>();
staticAssert<IsPromise<ReturnType<typeof c.credits.estimate>>>();
staticAssert<IsPromise<ReturnType<typeof c.credits.redeem>>>();
staticAssert<IsPromise<ReturnType<typeof c.user.get_profile>>>();
staticAssert<IsPromise<ReturnType<typeof c.instance.listen>>>();
staticAssert<IsPromise<ReturnType<typeof c.instance.send_message>>>();
staticAssert<IsPromise<ReturnType<typeof c.instance.record_thinking>>>();
staticAssert<IsPromise<ReturnType<typeof c.instance.stop_listening>>>();

// The cache facade is deliberately SYNCHRONOUS, mirroring Python.
staticAssert<Exact<ReturnType<typeof c.cache.clear>, void>>();
staticAssert<Exact<ReturnType<typeof c.cache.stats>, Json>>();

// ─── Literal unions are real unions, not `string` ───────────────────────────

staticAssert<Exact<DocumentType, (typeof DOCUMENT_TYPES)[number]>>();
staticAssert<Exact<IngestMode, 'fast' | 'long-range'>>();

// `document_type` accepts the union AND a bare string, because the server adds
// types faster than the SDK ships. Both must compile.
const typedDoc: CreateMemoryOptions = { document: 'x', document_type: 'meeting-transcript' };
const looseDoc: CreateMemoryOptions = { document: 'x', document_type: 'some-future-type' };
void typedDoc; void looseDoc;

// ─── snake_case and camelCase are both accepted ─────────────────────────────
// Python users write snake_case; JS users write camelCase. Rejecting either
// would make the SDK feel foreign to half its audience.

const snakeStyle: CreateMemoryOptions = {
  document: 'x', user_id: 'u', customer_id: 'c',
  document_type: 'document', document_id: 'd', document_created_at: '2026-01-01T00:00:00Z',
  mode: 'fast', metadata: { k: 'v' },
};
const camelStyle: CreateMemoryOptions = {
  document: 'x', userId: 'u', customerId: 'c',
  documentType: 'document', documentId: 'd', documentCreatedAt: new Date(),
  mode: 'fast', metadata: { k: 'v' },
};
void snakeStyle; void camelStyle;

const snakeUpdate: UpdateMemoryOptions = { memory_id: 'm', document: 'x', merge_strategy: 'replace' };
const camelUpdate: UpdateMemoryOptions = { memoryId: 'm', document: 'x', mergeStrategy: 'replace' };
void snakeUpdate; void camelUpdate;

const snakeMsg: RecordMessageOptions = { conversation_id: 'c', user_id: 'u', role: 'user', content: 'x' };
const camelMsg: RecordMessageOptions = { conversationId: 'c', userId: 'u', role: 'user', content: 'x' };
void snakeMsg; void camelMsg;

const snakeCompact: CompactOptions = { conversation_id: 'c', target_tokens: 100 };
const camelCompact: CompactOptions = { conversationId: 'c', targetTokens: 100 };
void snakeCompact; void camelCompact;

// ─── Required fields stay required ──────────────────────────────────────────

// @ts-expect-error `document` is required on memories.create
const missingDocument: CreateMemoryOptions = { user_id: 'u' };
void missingDocument;

// @ts-expect-error `document` is required on memories.update
const missingUpdateDocument: UpdateMemoryOptions = { memory_id: 'm' };
void missingUpdateDocument;

// @ts-expect-error `content` is required on a recorded message
const missingContent: RecordMessageOptions = { conversation_id: 'c', role: 'user' };
void missingContent;

// @ts-expect-error `text` is not a valid option name here; it is `content`
const wrongFieldName: RecordMessageOptions = { conversation_id: 'c', role: 'user', text: 'x' };
void wrongFieldName;

// ─── exactOptionalPropertyTypes ─────────────────────────────────────────────
// Under this flag, `{ x: undefined }` is NOT assignable to `{ x?: string }`.
// Optional fields must therefore be omitted, not set to undefined. Getting
// this wrong is the single most common reason an SDK fails to compile in a
// strict project.

const omitted: SynapClientOptions = { apiKey: 'k' };
void omitted;

// @ts-expect-error under exactOptionalPropertyTypes, an explicit undefined is not allowed
const explicitUndefined: SynapClientOptions = { apiKey: 'k', baseUrl: undefined };
void explicitUndefined;

// ─── Configure accepts the Python-only keys ─────────────────────────────────
// Accepted and ignored, so one config object stays portable across both SDKs.

const portableConfig: ConfigureOptions = {
  timeouts: { read: 10 },
  retryPolicy: { maxAttempts: 3 },
  storage_path: '/tmp/x',
  cache_backend: 'sqlite',
  session_timeout_minutes: 30,
  log_level: 'DEBUG',
};
void portableConfig;

// ─── Error narrowing ────────────────────────────────────────────────────────

function narrowError(error: unknown): string {
  // The `.code` discriminant works across a dual ESM/CJS install, where two
  // copies of the module mean two class identities and `instanceof` alone can
  // be false for a genuinely correct error.
  if (!isSynapError(error)) return 'not a synap error';
  const code: string = error.code;
  const transient: boolean = error.transient;

  if (error instanceof RateLimitError) {
    const retryAfter: number | null = error.retryAfterSeconds;
    return `rate limited, retry after ${retryAfter ?? 'unknown'}`;
  }
  if (error instanceof TranscriptConflictError) return 'transcript conflict';
  if (error instanceof ConflictError) return 'some other conflict';
  if (error instanceof InvalidConversationIdError) return 'bad conversation id';
  if (error instanceof InvalidInputError) return 'bad input';
  if (error instanceof ContextNotFoundError) return 'not found';
  if (error instanceof SynapTransientError) return `transient (${code})`;
  if (error instanceof SynapPermanentError) return `permanent (${code})`;
  return `unknown synap error, transient=${transient}`;
}

// Subclass relationships must hold at the TYPE level too, not just at runtime.
staticAssert<TranscriptConflictError extends ConflictError ? true : never>();
staticAssert<InvalidConversationIdError extends InvalidInputError ? true : never>();
staticAssert<InvalidInstanceIdError extends InvalidInputError ? true : never>();
staticAssert<RateLimitError extends SynapTransientError ? true : never>();
staticAssert<ConflictError extends SynapPermanentError ? true : never>();
staticAssert<SynapTransientError extends SynapError ? true : never>();

// ─── noUncheckedIndexedAccess ───────────────────────────────────────────────
// Array access yields `T | undefined`, so the SDK's own collections must be
// usable without a non-null assertion at every call site.

function readFirstFact(context: RawContext): string {
  const first: RawContextItem | undefined = context.facts?.[0];
  return first?.content ?? '(none)';
}

function readUnified(unified: UnifiedContext): string {
  const scope: string | undefined = unified.scopes_queried[0];
  const fact: RawContextItem | undefined = unified.facts[0];
  const mapped: string | undefined = unified.scope_map[fact?.id ?? ''];
  return `${scope ?? '-'}/${mapped ?? '-'}: ${fact?.content ?? '-'}`;
}

// ─── The tool definition is a discriminated union ───────────────────────────

function readTool(tool: ToolDefinition): string {
  // Narrowing on `type` must work: the OpenAI shape nests under `function`
  // while the Anthropic shape is flat with `input_schema`.
  if ('type' in tool) return `openai:${tool.function.name}`;
  return `anthropic:${tool.name}`;
}

const toolOptions: AsToolOptions = { scope: 'unified', user_id: 'u', style: 'anthropic' };
void toolOptions;

// ─── gRPC subpath types ─────────────────────────────────────────────────────

function readAvailability(a: GrpcAvailability): string {
  // A discriminated union on `available`, so `reason` is only reachable when
  // it exists.
  if (a.available) return 'available';
  const reason: 'not-node' | 'module-missing' = a.reason;
  return `${reason}: ${a.detail}`;
}

const states: StreamState[] = ['disconnected', 'connecting', 'connected', 'reconnecting', 'closed'];
const streamCreds: StreamCredentials = { apiKey: 'k', clientId: 'c', instanceId: 'i' };
void states; void streamCreds;

// @ts-expect-error 'reconnected' is not a StreamState
const badState: StreamState = 'reconnected';
void badState;

// ─── Options accepted by the live methods ───────────────────────────────────

const unifiedOptions: UnifiedFetchOptions = {
  conversation_id: randomUUID(), user_id: 'u', customer_id: 'c',
  search_query: ['q'], max_results: 20, types: ['facts'],
  mode: 'fast', precision_level: 'high',
  include_conversation_context: true, scopes: ['user', 'customer'],
  include_scope_labels: true, context_mode: 'in-conversation',
  include_profile: true, last_n_conversations: 1,
};
const fileOptions: CreateFromFileOptions = {
  user_id: 'u', customer_id: 'c', text: 'x', relationship_type: 'b2c', mode: 'fast',
};
const fetchOptions: FetchOptions = { user_id: 'u', search_query: ['q'], max_results: 5 };
const listenOptions: ListenOptions = { host: 'localhost', port: 50051, use_tls: false };
const sendOptions: SendMessageOptions = { content: 'x', role: 'user', conversation_id: 'c' };
void unifiedOptions; void fileOptions; void fetchOptions; void listenOptions; void sendOptions;

// ═══════════════════════════════════════════════════════════════════════════
// PART 2 — RUNTIME CHECKS
// ═══════════════════════════════════════════════════════════════════════════

const API_KEY: string = process.env['SYNAP_API_KEY'] ?? '';
const BASE_URL: string | undefined = process.env['SYNAP_BASE_URL'];
const OFFLINE: boolean = API_KEY === '' || process.argv.includes('--offline');
/** Skip anything that stores or bills. See the JS script for the rationale. */
const READ_ONLY: boolean = process.argv.includes('--read-only');

const RUN = randomUUID().slice(0, 8);
const USER_ID = `ts-test-user-${RUN}`;
const CUSTOMER_ID = `ts-test-cust-${RUN}`;
const CONVERSATION_ID = randomUUID();

interface Result { name: string; status: 'pass' | 'fail' | 'skip'; detail?: string }
const results: Result[] = [];

function section(title: string): void {
  console.log();
  console.log(`\x1b[1m${title}\x1b[0m`);
  console.log('─'.repeat(74));
}

async function check(
  name: string,
  fn: () => Promise<string | undefined>,
  options: { skip?: boolean; skipReason?: string } = {},
): Promise<void> {
  if (options.skip === true) {
    console.log(`  \x1b[90m skip \x1b[0m ${name.padEnd(48)} ${options.skipReason ?? ''}`);
    results.push({ name, status: 'skip' });
    return;
  }
  try {
    const detail = await fn();
    console.log(`  \x1b[32m pass \x1b[0m ${name.padEnd(48)} ${detail ?? ''}`);
    results.push({ name, status: 'pass', ...(detail !== undefined ? { detail } : {}) });
  } catch (error) {
    console.log(`  \x1b[31m FAIL \x1b[0m ${name}`);
    console.log(`         ${narrowError(error)}`);
    console.log(`         ${(error as Error)?.message}`);
    results.push({ name, status: 'fail' });
  }
}

/**
 * Declared `asserts condition`, not `: void`. A plain boolean check narrows
 * nothing, so every use of a nullable below it stays nullable and the file
 * fails to compile. This is the idiomatic fix and worth showing, because it is
 * the first thing a strict-mode consumer hits.
 */
function assert(condition: unknown, message: string): asserts condition {
  if (!condition) throw new Error(message);
}

const stub = (async () =>
  new Response('{}', { status: 200, headers: { 'content-type': 'application/json' } })
) as unknown as typeof fetch;

console.log(`\x1b[1mSynap SDK — TypeScript test\x1b[0m`);
console.log(`  sdk version   ${SDK_VERSION}`);
console.log(`  proto sha     ${PROTO_SHA}`);
console.log(`  mode          ${OFFLINE ? 'LOCAL ONLY (no key given)' : (READ_ONLY ? 'LOCAL + LIVE (read-only)' : 'LOCAL + LIVE')}`);
console.log();
console.log(`  \x1b[90mThe compile-time half of this file already passed by running at all.\x1b[0m`);
console.log(`  \x1b[90mRun \`npx tsc --noEmit\` to check it properly.\x1b[0m`);

section('1. Types are usable at runtime');

await check('typed client construction', async () => {
  const options: SynapClientOptions = { apiKey: 'k', fetchImpl: stub };
  const client = new SynapClient(options);
  await client.shutdown();
  return 'SynapClientOptions accepted';
});

await check('typed error narrowing', async () => {
  const cases: Array<[unknown, string]> = [
    [new RateLimitError('x', { retryAfterSeconds: 30 }), 'rate limited'],
    [new TranscriptConflictError('x'), 'transcript conflict'],
    [new ConflictError('x'), 'some other conflict'],
    [new InvalidConversationIdError('x'), 'bad conversation id'],
    [new ContextNotFoundError('x'), 'not found'],
    [new Error('plain'), 'not a synap error'],
  ];
  for (const [error, expected] of cases) {
    const got = narrowError(error);
    assert(got.startsWith(expected), `expected "${expected}", got "${got}"`);
  }
  return `${cases.length} branches narrow correctly`;
});

await check('noUncheckedIndexedAccess helpers', async () => {
  const empty: RawContext = {};
  assert(readFirstFact(empty) === '(none)', 'empty context should be safe to read');
  const populated: RawContext = { facts: [{ id: 'f1', content: 'a fact' }] };
  assert(readFirstFact(populated) === 'a fact', 'should read the first fact');
  return 'optional indexing is safe without assertions';
});

await check('tool definition narrowing', async () => {
  const client = new SynapClient({ apiKey: 'k', fetchImpl: stub });
  const openai = readTool(client.as_tool({ scope: 'user', user_id: USER_ID }));
  const anthropic = readTool(
    client.as_tool({ scope: 'user', user_id: USER_ID, style: 'anthropic' }),
  );
  assert(openai.startsWith('openai:'), `got ${openai}`);
  assert(anthropic.startsWith('anthropic:'), `got ${anthropic}`);
  await client.shutdown();
  return `${openai} / ${anthropic}`;
});

await check('typed snapshot', async () => {
  const client = new SynapClient({ apiKey: 'k', fetchImpl: stub });
  const snapshot: AnticipationCacheSnapshot = client.anticipation_cache_snapshot();
  const entries: number = snapshot.total_entries;
  const vocab: string[] = snapshot.corpus_vocab_sample;
  await client.shutdown();
  return `total_entries=${entries} vocab=${vocab.length}`;
});

await check('typed cache and BM25', async () => {
  const cache = new AnticipationCache({ ttlSeconds: 600, maxBundles: 10 });
  cache.store({
    bundleId: 'b1', entityId: USER_ID, conversationId: CONVERSATION_ID,
    bundleType: 'anticipation', searchQueries: ['refund'],
    itemsByType: { facts: [{ item_id: 'i1', content: 'Refunds take five days', scope: 'user' }] },
  });
  const hit: LookupResult | null = cache.lookup({
    searchQuery: ['how long do refunds take'], entityId: USER_ID, conversationId: CONVERSATION_ID,
  });
  // `assert` narrows because it is declared `asserts condition`, so `hit` is
  // LookupResult from here on with no non-null assertion needed.
  assert(hit !== null, 'expected a cache hit');
  const bm25 = new BM25([tokenize('refund policy'), tokenize('seat preference')]);
  const score: number = bm25.score(tokenize('refund'), 0);
  const stemmed: string = stem('running');
  const recall: boolean = isRecallQuery(['what did I say earlier']);
  assert(recall, 'should be recognised as a recall query');
  cache.clear();
  return `score=${score.toFixed(3)} stem=${stemmed} items=${hit.bundleIds.length}`;
});

await check('typed flatten', async () => {
  const flat: FlatMemory[] = flattenContextItems({
    facts: [{ id: 'f1', content: 'a fact', confidence: 0.9 }],
    emotions: [{ id: 'm1', emotion_type: 'joy', context: 'good news', intensity: 0.8 }],
  });
  assert(flat.length === 2, `expected 2, got ${flat.length}`);
  const first: FlatMemory | undefined = flat[0];
  return `${flat.length} items, first=${first?.memory ?? '-'}`;
});

await check('typed constants', async () => {
  const docType: DocumentType = DEFAULT_DOCUMENT_TYPE;
  const mode: IngestMode = DEFAULT_INGEST_MODE;
  assert(DOCUMENT_TYPES.includes(docType), 'default not in DOCUMENT_TYPES');
  assert(INGEST_MODES.includes(mode), 'default not in INGEST_MODES');
  return `${docType} / ${mode}`;
});

section('2. gRPC subpath');

await check('checkGrpcAvailability', async () => {
  const availability: GrpcAvailability = await checkGrpcAvailability();
  const summary = readAvailability(availability);
  // Not a failure when unavailable: grpc-js is an optionalDependency and Edge
  // runtimes cannot host it at all.
  return summary;
});

await check('GrpcStreamClient is constructible', async () => {
  const client = new GrpcStreamClient(
    { apiKey: 'k', clientId: 'c', instanceId: 'inst_0123456789abcdef' },
    new AnticipationCache(),
    { host: '127.0.0.1', port: 50051, useTls: false },
  );
  const state: StreamState = client.currentState;
  assert(state === 'disconnected', `expected disconnected, got ${state}`);
  assert(client.isConnected === false, 'should not be connected');
  return `state=${state} proto=${PROTO_SHA}`;
});

section(`3. Live methods${OFFLINE ? ' (skipped: no SYNAP_API_KEY)' : ''}`);

const live: SynapClient | null = OFFLINE
  ? null
  : new SynapClient({ apiKey: API_KEY, ...(BASE_URL !== undefined ? { baseUrl: BASE_URL } : {}) });
const skip = { skip: OFFLINE, skipReason: 'set SYNAP_API_KEY to run' };

await check('initialize', async () => {
  assert(live !== null, 'no client');
  await live.initialize();
  return `client_id=${live.client_id} instance_id=${live.instance_id || '(none)'}`;
}, skip);

const skipWrite = {
  skip: OFFLINE || READ_ONLY,
  skipReason: READ_ONLY ? 'skipped: --read-only' : 'set SYNAP_API_KEY to run',
};

await check('memories.create (typed result)', async () => {
  assert(live !== null, 'no client');
  const options: CreateMemoryOptions = {
    document: 'The customer is a TypeScript user who prefers strict mode.',
    user_id: USER_ID,
    customer_id: CUSTOMER_ID,
  };
  const result: CreateMemoryResult = await live.memories.create(options);
  const ingestionId: string | undefined = result.ingestion_id;
  assert(ingestionId !== undefined, 'no ingestion_id on the typed result');
  return `ingestion_id=${ingestionId}`;
}, skipWrite);

await check('user.context.fetch (raw shape)', async () => {
  assert(live !== null, 'no client');
  const context: RawContext = await live.user.context.fetch({
    user_id: USER_ID, customer_id: CUSTOMER_ID, max_results: 5,
  });
  return `facts=${(context.facts ?? []).length}`;
}, skip);

await check('fetchUserContext (normalised shape)', async () => {
  assert(live !== null, 'no client');
  const normalised: NormalisedContext = await live.fetchUserContext({
    user_id: USER_ID, customer_id: CUSTOMER_ID,
  });
  // camelCase here, snake_case above. Deliberately different.
  return `facts=${(normalised.facts ?? []).length} (camelCase)`;
}, skip);

await check('fetch (typed UnifiedContext)', async () => {
  assert(live !== null, 'no client');
  const unified: UnifiedContext = await live.fetch({
    user_id: USER_ID, customer_id: CUSTOMER_ID, search_query: ['typescript'],
  });
  return readUnified(unified);
}, skip);

await check('credits.get_balance', async () => {
  assert(live !== null, 'no client');
  const balance: Json = await live.credits.get_balance();
  return `balance=${String(balance['balance_credits'] ?? 'n/a')}`;
}, skip);

await check('validation rejects with a typed error', async () => {
  assert(live !== null, 'no client');
  let caught: unknown = null;
  await live.conversation.context.fetch({ conversation_id: 'not-a-uuid' })
    .catch((e: unknown) => { caught = e; });
  assert(caught instanceof InvalidConversationIdError, 'expected InvalidConversationIdError');
  return narrowError(caught);
}, skip);

await check('a second listen() is refused', async () => {
  assert(live !== null, 'no client');
  // Without SYNAP_GRPC_HOST the first listen() fails to connect, which leaves
  // the client un-listening on purpose so a retry is possible.
  let caught: unknown = null;
  await live.instance.listen({ host: '127.0.0.1', port: 1, use_tls: false })
    .catch((e: unknown) => { caught = e; });
  assert(caught !== null, 'connecting to a dead port should fail');
  assert(isSynapError(caught), 'error should be in the Synap taxonomy');
  assert(live.instance.is_listening === false, 'a failed listen must not latch');
  return `failed cleanly as ${(caught as SynapError).name}, retry still possible`;
}, skip);

await check('shutdown', async () => {
  assert(live !== null, 'no client');
  await live.shutdown();
  await live.shutdown();
  return 'idempotent';
}, skip);

// ─── Summary ────────────────────────────────────────────────────────────────

const pass = results.filter((r) => r.status === 'pass').length;
const fail = results.filter((r) => r.status === 'fail').length;
const skipped = results.filter((r) => r.status === 'skip').length;

console.log();
console.log('═'.repeat(74));
console.log(`\x1b[1mRESULT\x1b[0m  ${pass} passed, ${fail} failed, ${skipped} skipped`);
console.log(`        plus every compile-time assertion, checked by \`npx tsc --noEmit\``);
console.log('═'.repeat(74));

if (OFFLINE) {
  console.log();
  console.log('Only the local group ran. For the live methods:');
  console.log('  export SYNAP_API_KEY=...');
  console.log('  export SYNAP_BASE_URL=https://your-synap-host');
  console.log('  npx tsx test-ts-sdk.ts');
}

process.exitCode = fail > 0 ? 1 : 0;
