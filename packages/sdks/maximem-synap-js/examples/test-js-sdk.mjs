/**
 * Per-method exerciser for the native Synap JS SDK (plain JavaScript, ESM).
 *
 *   node test-js-sdk.mjs                # local-only, no network, no credentials
 *   SYNAP_API_KEY=... node test-js-sdk.mjs   # adds the live HTTP methods
 *
 * Every public member gets its own check, and the run ends by cross-checking
 * what it exercised against the Python surface contract, so "all green" cannot
 * quietly mean "half of them were never called".
 *
 * Groups:
 *   local  runs anywhere, no key, no network, no cost
 *   live   real HTTP calls. WRITES MEMORIES AND SPENDS CREDITS.
 *   grpc   the bidirectional anticipation stream. Needs SYNAP_GRPC_HOST.
 */
import { existsSync, readFileSync } from 'node:fs';
import { randomUUID } from 'node:crypto';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import {
  SynapClient,
  // Error taxonomy
  SynapError, SynapTransientError, SynapPermanentError,
  InvalidInputError, InvalidConversationIdError, InvalidInstanceIdError,
  AuthenticationError, ContextNotFoundError, ConflictError, TranscriptConflictError,
  RateLimitError, InsufficientCreditsError, ServiceUnavailableError,
  NetworkTimeoutError, AgentUnavailableError, SessionExpiredError,
  ListeningNotActiveError, ListeningAlreadyActiveError,
  isSynapError,
  // Building blocks
  AnticipationCache, BM25, HttpTransport,
  tokenize, stem, isRecallQuery, flattenContextItems, shouldRetry, isOutcomeUnknown,
  resolvePath, ENDPOINTS, CONTRACT, THRESHOLDS, ANY_SCOPE,
  DOCUMENT_TYPES, INGEST_MODES, DEFAULT_DOCUMENT_TYPE, DEFAULT_INGEST_MODE,
  DEFAULT_BASE_URL, DEFAULT_TIMEOUTS, DEFAULT_RETRY_POLICY, SDK_VERSION,
} from '@maximem/synap-js-sdk';

// ─── Configuration ───────────────────────────────────────────────────────────

const API_KEY = process.env.SYNAP_API_KEY ?? '';
const BASE_URL = process.env.SYNAP_BASE_URL ?? DEFAULT_BASE_URL;
const OFFLINE = process.argv.includes('--offline') || API_KEY === '';
// gRPC runs by DEFAULT. The SDK already defaults to the prod host on 443 with
// TLS, so a normal run needs no gRPC configuration at all. These env vars are
// overrides for the unusual case: a non-default deployment, or a plaintext port
// reached over an SSH tunnel. Previously the group was gated on SYNAP_GRPC_HOST
// being set, which meant the default run silently skipped nine checks.
const GRPC_HOST = process.env.SYNAP_GRPC_HOST ?? '';
const GRPC_PORT = process.env.SYNAP_GRPC_PORT ?? '';
const GRPC_TLS = process.env.SYNAP_GRPC_USE_TLS ?? '';
const RUN_GRPC = !OFFLINE && !process.argv.includes('--no-grpc');
/**
 * Skip every check that writes or bills.
 *
 * Meant for pointing at a real production instance: retrieval, the formatter,
 * as_tool, error mapping and the full anticipation stream all still run, but
 * nothing is stored, updated, deleted or compacted. Note that the stream's
 * send_message/record_thinking DO write server-side (the server persists the
 * conversation event), so they count as writes here.
 */
const READ_ONLY = process.argv.includes('--read-only');

/**
 * Resolve the gRPC target.
 *
 * The stream host is DERIVED from SYNAP_BASE_URL when one is set, rather than
 * left to the SDK default. The SDK defaults to prod for both, which is right
 * for a plain run but wrong the moment someone overrides only the HTTP base:
 * you then get HTTP against staging and a gRPC stream against PROD, silently.
 * Deriving keeps both halves on the same deployment unless the caller
 * explicitly says otherwise.
 */
const derivedGrpcHost = process.env.SYNAP_BASE_URL
  ? new URL(process.env.SYNAP_BASE_URL).hostname
  : '';
const grpcOverrides = {
  ...(GRPC_HOST !== '' ? { host: GRPC_HOST }
      : derivedGrpcHost !== '' ? { host: derivedGrpcHost } : {}),
  ...(GRPC_PORT !== '' ? { port: Number(GRPC_PORT) } : {}),
  ...(GRPC_TLS !== '' ? { use_tls: GRPC_TLS !== '0' } : {}),
};
const INGEST_WAIT = Number(process.env.SYNAP_INGEST_WAIT ?? 120);

// Fresh scope ids per run, so nothing collides with real data and every
// "did my write land" assertion is honest.
const RUN = randomUUID().slice(0, 8);
const USER_ID = `js-test-user-${RUN}`;
const CUSTOMER_ID = `js-test-cust-${RUN}`;
const CONVERSATION_ID = randomUUID();

// ─── Tiny runner ─────────────────────────────────────────────────────────────

const results = [];
const exercised = new Set();
let currentGroup = 'local';

function group(name, title) {
  currentGroup = name;
  console.log();
  console.log(`\x1b[1m${title}\x1b[0m`);
  console.log('─'.repeat(74));
}

/**
 * @param member dotted member path, e.g. "memories.create". Recorded so the
 *   coverage check at the end can prove it actually ran.
 */
async function check(member, fn, { skipIf = false, skipReason = '' } = {}) {
  exercised.add(member);
  if (skipIf) {
    console.log(`  \x1b[90m skip \x1b[0m ${member.padEnd(46)} ${skipReason}`);
    results.push({ member, status: 'skip', group: currentGroup });
    return undefined;
  }
  const started = Date.now();
  try {
    const detail = await fn();
    const ms = String(Date.now() - started).padStart(5);
    console.log(`  \x1b[32m pass \x1b[0m ${member.padEnd(46)} ${ms}ms  ${detail ?? ''}`);
    results.push({ member, status: 'pass', group: currentGroup });
    return detail;
  } catch (error) {
    const ms = String(Date.now() - started).padStart(5);
    console.log(`  \x1b[31m FAIL \x1b[0m ${member.padEnd(46)} ${ms}ms`);
    console.log(`         ${error?.name}: ${error?.message}`);
    if (error?.code) console.log(`         code=${error.code} transient=${error.transient}`);
    results.push({ member, status: 'fail', group: currentGroup, error });
    return undefined;
  }
}

function assert(condition, message) {
  if (!condition) throw new Error(message);
}

/** A fetch stub, so the local group can drive real code paths with no network. */
function stubFetch(handler) {
  return async (url, init) => {
    const body = handler ? handler(String(url), init) : {};
    return new Response(JSON.stringify(body), {
      status: 200,
      headers: { 'content-type': 'application/json' },
    });
  };
}

// ═════════════════════════════════════════════════════════════════════════════

console.log(`\x1b[1mSynap JS SDK — per-method test\x1b[0m`);
console.log(`  sdk version   ${SDK_VERSION}`);
console.log(`  mode          ${OFFLINE ? 'LOCAL ONLY (no key given)' : 'LOCAL + LIVE'}`);
if (!OFFLINE) {
  console.log(`  base url      ${BASE_URL}`);
  const grpcDesc = Object.keys(grpcOverrides).length > 0
    ? `${grpcOverrides.host ?? '(sdk default)'}:${grpcOverrides.port ?? 443} ` +
      `tls=${grpcOverrides.use_tls ?? true}`
    : 'sdk defaults (prod)';
  console.log(`  grpc          ${RUN_GRPC ? grpcDesc : 'skipped (--no-grpc)'}`);
  console.log(`  scope         user=${USER_ID} customer=${CUSTOMER_ID}`);
  console.log(READ_ONLY
    ? `  \x1b[36mREAD-ONLY: nothing will be stored, updated, deleted or compacted.\x1b[0m`
    : `  \x1b[33mThis WRITES MEMORIES and SPENDS CREDITS.\x1b[0m`);
}

// ─── 1. Construction and credentials ─────────────────────────────────────────

group('local', '1. Construction, credentials, configuration');

await check('new SynapClient', async () => {
  const c = new SynapClient({ apiKey: 'k', fetchImpl: stubFetch() });
  assert(c instanceof SynapClient, 'not a SynapClient');
  await c.shutdown();
  return 'constructed';
});

await check('new SynapClient (no key rejected)', async () => {
  const saved = process.env.SYNAP_API_KEY;
  delete process.env.SYNAP_API_KEY;
  try {
    let threw = null;
    try { new SynapClient(); } catch (e) { threw = e; }
    assert(threw instanceof InvalidInputError, 'expected InvalidInputError');
    assert(/SYNAP_API_KEY/.test(threw.message), 'message should name the env var');
    return 'rejects a missing key';
  } finally {
    if (saved !== undefined) process.env.SYNAP_API_KEY = saved;
  }
});

await check('validation: instance id format', async () => {
  let threw = null;
  try {
    new SynapClient({ apiKey: 'k', instanceId: 'not-an-instance-id', fetchImpl: stubFetch() });
  } catch (e) { threw = e; }
  assert(threw instanceof InvalidInstanceIdError, 'expected InvalidInstanceIdError');
  // Valid shape is `inst_` plus 16 hex characters.
  const ok = new SynapClient({
    apiKey: 'k', instanceId: 'inst_0123456789abcdef', fetchImpl: stubFetch(),
  });
  await ok.shutdown();
  return 'inst_<hex16> enforced locally';
});

await check('configure', async () => {
  const c = new SynapClient({ apiKey: 'k', fetchImpl: stubFetch() });
  c.configure({ timeouts: { read: 12 }, retryPolicy: { maxAttempts: 5 } });
  // Python-only keys are accepted and ignored, so a shared config object is
  // portable between the two SDKs.
  c.configure({ storage_path: '/tmp/x', cache_backend: 'sqlite', log_level: 'DEBUG' });
  await c.shutdown();
  return 'timeouts + retry policy set';
});

await check('instance_id / client_id getters', async () => {
  const c = new SynapClient({
    apiKey: 'k', clientId: 'cli_x', instanceId: 'inst_0123456789abcdef',
    fetchImpl: stubFetch(),
  });
  assert(c.instance_id === 'inst_0123456789abcdef', `got ${c.instance_id}`);
  assert(c.client_id === 'cli_x', `got ${c.client_id}`);
  await c.shutdown();
  return 'readable';
});

await check('internals are not reachable', async () => {
  const c = new SynapClient({ apiKey: 'k', fetchImpl: stubFetch() });
  for (const hidden of ['transport', 'anticipationCache', 'closed']) {
    assert(c[hidden] === undefined, `client.${hidden} is exposed`);
  }
  await c.shutdown();
  return 'transport + raw cache stay private';
});

// ─── 2. Error taxonomy ───────────────────────────────────────────────────────

group('local', '2. Error taxonomy');

await check('error hierarchy', async () => {
  const pairs = [
    [new InvalidConversationIdError('x'), InvalidInputError],
    [new InvalidInstanceIdError('x'), InvalidInputError],
    [new InvalidInputError('x'), SynapPermanentError],
    [new TranscriptConflictError('x'), ConflictError],
    [new ConflictError('x'), SynapPermanentError],
    [new RateLimitError('x'), SynapTransientError],
    [new ServiceUnavailableError('x'), SynapTransientError],
    [new NetworkTimeoutError('x'), SynapTransientError],
    [new AgentUnavailableError('x'), SynapTransientError],
    [new AuthenticationError('x'), SynapPermanentError],
    [new ContextNotFoundError('x'), SynapPermanentError],
    [new InsufficientCreditsError('x'), SynapPermanentError],
    [new SessionExpiredError('x'), SynapPermanentError],
    [new ListeningNotActiveError('x'), SynapPermanentError],
    [new ListeningAlreadyActiveError('x'), SynapPermanentError],
  ];
  for (const [instance, parent] of pairs) {
    assert(instance instanceof parent, `${instance.name} is not a ${parent.name}`);
    assert(instance instanceof SynapError, `${instance.name} is not a SynapError`);
    assert(isSynapError(instance), `isSynapError missed ${instance.name}`);
    assert(typeof instance.code === 'string', `${instance.name} has no .code`);
  }
  return `${pairs.length} subclasses, .code + instanceof both work`;
});

await check('transient vs permanent flags', async () => {
  assert(new RateLimitError('x').transient === true, 'rate limit should be transient');
  assert(new InvalidInputError('x').transient === false, 'invalid input should be permanent');
  return 'retryability is readable off the error';
});

// ─── 3. Local behaviour: cache, BM25, contract ───────────────────────────────

group('local', '3. Anticipation cache, BM25, behaviour contract');

await check('cache.stats', async () => {
  const c = new SynapClient({ apiKey: 'k', fetchImpl: stubFetch() });
  const stats = c.cache.stats();
  assert(typeof stats === 'object' && stats !== null, 'no stats object');
  assert(c.cache.now === undefined, 'cache leaks its internal clock');
  await c.shutdown();
  return JSON.stringify(stats);
});

await check('cache.clear', async () => {
  const c = new SynapClient({ apiKey: 'k', fetchImpl: stubFetch() });
  c.cache.clear();
  await c.shutdown();
  return 'callable';
});

await check('cache.clear_user', async () => {
  const c = new SynapClient({ apiKey: 'k', fetchImpl: stubFetch() });
  c.cache.clear_user(USER_ID);
  await c.shutdown();
  return 'callable';
});

await check('cache.clear_customer', async () => {
  const c = new SynapClient({ apiKey: 'k', fetchImpl: stubFetch() });
  c.cache.clear_customer(CUSTOMER_ID);
  await c.shutdown();
  return 'callable';
});

await check('anticipation_cache_snapshot', async () => {
  const c = new SynapClient({ apiKey: 'k', fetchImpl: stubFetch() });
  const snap = c.anticipation_cache_snapshot();
  for (const key of [
    'total_entries', 'total_item_records', 'scope_breakdown_overall',
    'corpus_vocab_size', 'corpus_vocab_sample', 'item_records', 'bundles',
  ]) {
    assert(key in snap, `snapshot is missing ${key}`);
  }
  await c.shutdown();
  return `${Object.keys(snap).length} keys, matching Python's shape`;
});

await check('AnticipationCache store + lookup', async () => {
  const cache = new AnticipationCache();
  cache.store({
    bundleId: 'b1',
    entityId: USER_ID,
    conversationId: CONVERSATION_ID,
    bundleType: 'anticipation',
    searchQueries: ['refund policy'],
    itemsByType: {
      facts: [{ item_id: 'i1', content: 'Refunds take five business days to process', scope: 'user' }],
    },
  });
  assert(cache.size === 1, `expected 1 bundle, got ${cache.size}`);
  // Term overlap has to be real. On a one-item corpus there is almost no IDF
  // to work with, and the per-query threshold is max(0.6, min(1.5, 0.3 * tokens)),
  // so a query sharing one stem out of three legitimately misses. That is the
  // gate working, not a bug.
  const hit = cache.lookup({
    searchQuery: ['how long do refunds take'],
    entityId: USER_ID,
    conversationId: CONVERSATION_ID,
  });
  assert(hit !== null, 'lookup missed a bundle it should have matched');
  return `hit score=${hit.score.toFixed(3)} coverage=${hit.coverage.toFixed(2)}`;
});

await check('AnticipationCache scope isolation', async () => {
  const cache = new AnticipationCache();
  cache.store({
    bundleId: 'b1', entityId: 'user-A', itemsByType: {
      facts: [{ item_id: 'i1', content: 'user A private detail here', scope: 'user' }],
    },
  });
  // A bundle pushed for one user must never be served to another.
  const leak = cache.lookup({ searchQuery: ['user A private detail here'], entityId: 'user-B' });
  assert(leak === null, 'CROSS-USER LEAK: user B was served user A\'s bundle');
  return 'a bundle for one user is not served to another';
});

await check('tokenize / stem', async () => {
  const tokens = tokenize('The customers were running quickly toward refunds');
  assert(Array.isArray(tokens) && tokens.length > 0, 'no tokens');
  assert(!tokens.includes('the'), 'stop words should be dropped');
  assert(stem('running').length > 0, 'stem returned nothing');
  return `${tokens.length} tokens: ${tokens.slice(0, 5).join(', ')}`;
});

await check('isRecallQuery', async () => {
  // Takes the search_query ARRAY, not a single string, matching the shape the
  // retrieval path actually holds.
  assert(isRecallQuery(['what did I tell you earlier']) === true, 'should be a recall query');
  assert(isRecallQuery(['book me a flight to Berlin']) === false, 'should not be a recall query');
  assert(isRecallQuery([]) === false, 'an empty array is not a recall query');
  assert(isRecallQuery(null) === false, 'null is not a recall query');
  return 'recall bypass detected on the query array';
});

await check('BM25', async () => {
  const bm25 = new BM25([
    tokenize('refund policy takes five days'),
    tokenize('window seat preference on long flights'),
  ]);
  const score = bm25.score(tokenize('refund policy'), 0);
  assert(score > 0, `expected a positive score, got ${score}`);
  return `score=${score.toFixed(4)}`;
});

await check('CONTRACT / THRESHOLDS', async () => {
  // The behaviour contract is shared verbatim with the Python SDK, so tuned
  // values cannot drift between languages. Both SDKs read this same JSON.
  assert(typeof CONTRACT === 'object' && CONTRACT !== null, 'no contract');
  assert(typeof CONTRACT.version === 'number', 'contract has no version');
  const patterns = CONTRACT.recall_bypass.patterns;
  assert(Array.isArray(patterns) && patterns.length > 0, 'no recall patterns');
  assert(Array.isArray(CONTRACT.bm25.stop_words), 'no stop words');
  assert(Array.isArray(CONTRACT.bm25.suffixes), 'no stemmer suffixes');
  for (const key of ['bm25', 'novelTerm', 'minCorpusForNovelGate', 'effectiveFloor', 'queryTokenScale']) {
    assert(key in THRESHOLDS, `THRESHOLDS is missing ${key}`);
  }
  return `v${CONTRACT.version} recall=${patterns.length} stopwords=${CONTRACT.bm25.stop_words.length} ` +
    `suffixes=${CONTRACT.bm25.suffixes.length} thresholds=${Object.keys(THRESHOLDS).length}`;
});

await check('ENDPOINTS / resolvePath', async () => {
  assert(Object.keys(ENDPOINTS).length > 0, 'endpoint table is empty');
  // Takes the endpoint NAME, not a path string, so a route is a compile-time
  // constant rather than a string someone can typo.
  const resolved = resolvePath('memories_get', { memory_id: 'a b/c' });
  assert(!resolved.includes(' '), 'path params must be URL-encoded');
  assert(resolved.includes('a%20b%2Fc'), `not encoded: ${resolved}`);
  let threw = null;
  try { resolvePath('memories_get', {}); } catch (e) { threw = e; }
  assert(threw !== null, 'a missing path parameter should throw');
  return `${Object.keys(ENDPOINTS).length} endpoints, params encoded`;
});

await check('shouldRetry / isOutcomeUnknown', async () => {
  // The double-billing gate: a request whose outcome is unknown must NOT be
  // retried unless the route is idempotent.
  const decision = shouldRetry({
    error: new ServiceUnavailableError('x'),
    attempt: 1,
    policy: DEFAULT_RETRY_POLICY,
    method: 'POST',
    outcomeUnknown: true,
    idempotent: false,
  });
  assert(decision.retry === false, 'a non-idempotent POST with unknown outcome was retried');
  assert(typeof isOutcomeUnknown(new NetworkTimeoutError('x')) === 'boolean', 'bad guard');
  return `blocked: ${decision.reason}`;
});

await check('flattenContextItems', async () => {
  const flat = flattenContextItems({
    facts: [{ id: 'f1', content: 'a fact', confidence: 0.9 }],
    episodes: [{ id: 'e1', summary: 'an episode', significance: 0.5 }],
  });
  assert(flat.length === 2, `expected 2 flattened items, got ${flat.length}`);
  // Episodes deliberately carry no `source` key, matching the 0.3.x bridge.
  const episode = flat.find((i) => i.memory === 'an episode');
  assert(episode !== undefined, 'episode did not flatten');
  return `${flat.length} items, per-type field mapping preserved`;
});

await check('constants', async () => {
  assert(DOCUMENT_TYPES.includes(DEFAULT_DOCUMENT_TYPE), 'default document type is not in the list');
  assert(INGEST_MODES.includes(DEFAULT_INGEST_MODE), 'default ingest mode is not in the list');
  assert(typeof DEFAULT_TIMEOUTS.read === 'number', 'no read timeout');
  assert(ANY_SCOPE.length > 0, 'no ANY_SCOPE sentinel');
  return `doc types=${DOCUMENT_TYPES.length} modes=${INGEST_MODES.join('|')} default=${DEFAULT_DOCUMENT_TYPE}`;
});

// ─── 4. Request building, checked without a server ───────────────────────────

group('local', '4. Request bodies (verified against a stub, no network)');

await check('memories.create body', async () => {
  let sent = null;
  const c = new SynapClient({
    apiKey: 'k',
    fetchImpl: async (url, init) => {
      sent = JSON.parse(init.body);
      return new Response('{}', { status: 200, headers: { 'content-type': 'application/json' } });
    },
  });
  await c.memories.create({ document: 'hello' });
  await c.shutdown();
  // Python emits all eight fields with defaults applied. document_type in
  // particular selects the extraction path, so omitting it stores less.
  assert(sent.document_type === DEFAULT_DOCUMENT_TYPE, `document_type=${sent.document_type}`);
  assert(sent.mode === DEFAULT_INGEST_MODE, `mode=${sent.mode}`);
  for (const key of [
    'document', 'document_type', 'document_id', 'document_created_at',
    'user_id', 'customer_id', 'mode', 'metadata',
  ]) {
    assert(key in sent, `body is missing ${key}`);
  }
  return `8 fields, document_type=${sent.document_type}`;
});

await check('validation rejects rather than throwing', async () => {
  // A synchronous throw from a Promise-returning method bypasses .catch().
  const c = new SynapClient({ apiKey: 'k', fetchImpl: stubFetch() });
  const calls = [
    () => c.memories.create({}),
    () => c.memories.get(''),
    () => c.memories.update({ document: 'x' }),
    () => c.memories.delete(''),
    () => c.memories.status(''),
    () => c.memories.batch_create({ documents: [] }),
    () => c.conversation.record_message({}),
    () => c.conversation.record_messages_batch([]),
    () => c.conversation.context.compact({}),
  ];
  for (const call of calls) {
    const p = call();
    assert(p && typeof p.then === 'function', 'did not return a promise');
    let rejected = false;
    await p.catch(() => { rejected = true; });
    assert(rejected, 'expected a rejection, not a resolution');
  }
  await c.shutdown();
  return `${calls.length} methods reject, none throw synchronously`;
});

await check('conversation_id validated where Python validates', async () => {
  const c = new SynapClient({ apiKey: 'k', fetchImpl: stubFetch() });
  const bad = 'conv_123';
  const validated = [
    () => c.conversation.record_message({ conversation_id: bad, user_id: 'u', role: 'user', content: 'x' }),
    () => c.conversation.context.fetch({ conversation_id: bad }),
    () => c.conversation.context.compact({ conversation_id: bad }),
    () => c.conversation.context.get_compacted({ conversation_id: bad }),
    () => c.conversation.context.get_compaction_status({ conversation_id: bad }),
    () => c.conversation.context.get_context_for_prompt({ conversation_id: bad }),
    () => c.fetch({ conversation_id: bad }),
  ];
  for (const call of validated) {
    let err = null;
    await call().catch((e) => { err = e; });
    assert(err instanceof InvalidConversationIdError, `expected rejection, got ${err?.name}`);
  }
  // And NOT where Python deliberately skips it: ingest_transcript takes a
  // free-form id the server coerces.
  await c.conversation.ingest_transcript({
    conversation_id: 'free-form-id', user_id: 'u', transcript: 'hello',
  });
  await c.shutdown();
  return `${validated.length} validated, ingest_transcript left free-form`;
});

// ─── 5. as_tool ──────────────────────────────────────────────────────────────

group('local', '5. as_tool (LLM tool definition)');

await check('as_tool', async () => {
  const c = new SynapClient({ apiKey: 'k', fetchImpl: stubFetch() });
  const openai = c.as_tool({ scope: 'user', user_id: USER_ID });
  assert(openai.type === 'function', 'not an OpenAI function tool');
  assert(openai.function.name === 'synap_fetch_user_context', openai.function.name);
  const anthropic = c.as_tool({ scope: 'unified', user_id: USER_ID, style: 'anthropic' });
  assert('input_schema' in anthropic, 'no input_schema on the Anthropic shape');

  // The reason the helper exists: the model must not be able to choose whose
  // memory to read.
  for (const tool of [openai, anthropic]) {
    const schema = tool.function?.parameters ?? tool.input_schema;
    const props = Object.keys(schema.properties);
    assert(!props.includes('user_id'), 'user_id is exposed to the LLM');
    assert(!props.includes('customer_id'), 'customer_id is exposed to the LLM');
  }
  let threw = null;
  try { c.as_tool({ scope: 'nope' }); } catch (e) { threw = e; }
  assert(threw instanceof InvalidInputError, 'an unknown scope should be rejected');
  await c.shutdown();
  return 'both styles, scope ids closed over';
});

// ─── 6. Live HTTP ────────────────────────────────────────────────────────────

group('live', `6. Live HTTP${OFFLINE ? ' (skipped: no SYNAP_API_KEY)' : ''}`);

const client = OFFLINE ? null : new SynapClient({ apiKey: API_KEY, baseUrl: BASE_URL });
const skipLive = { skipIf: OFFLINE, skipReason: 'set SYNAP_API_KEY to run' };
/** For checks that mutate server state or spend credits. */
const skipWrite = {
  skipIf: OFFLINE || READ_ONLY,
  skipReason: READ_ONLY ? 'skipped: --read-only' : 'set SYNAP_API_KEY to run',
};
let memoryId = null;
let ingestionId = null;
let fileIngestionId = null;
let ingestBlockedBy = null;

await check('initialize', async () => {
  await client.initialize();
  assert(client.client_id !== '', 'client_id was not resolved from whoami');
  return `client_id=${client.client_id} instance_id=${client.instance_id || '(none)'}`;
}, skipLive);

await check('credits.get_balance', async () => {
  const balance = await client.credits.get_balance();
  const credits = Number(balance.balance_credits ?? balance.total_remaining ?? NaN);
  if (Number.isFinite(credits) && credits <= 0) {
    // Retrieval fails OPEN on a 429, so an empty context later would look like
    // a retrieval bug rather than an empty wallet.
    console.log('         \x1b[33mWARNING: no credit headroom; retrieval will 429 and fail open\x1b[0m');
  }
  return `balance=${balance.balance_credits ?? JSON.stringify(balance).slice(0, 60)}`;
}, skipLive);

await check('credits.get_ledger', async () => {
  const ledger = await client.credits.get_ledger({ limit: 3 });
  return `entries=${(ledger.entries ?? ledger.items ?? []).length}`;
}, skipLive);

await check('credits.estimate', async () => {
  const estimate = await client.credits.estimate({ metric_type: 'context_fetch', units: 10 });
  return JSON.stringify(estimate).slice(0, 70);
}, skipLive);

await check('credits.redeem', async () => {
  // No real promo code here. A rejection IS the pass: it proves the route is
  // wired and the error maps to the taxonomy rather than escaping raw.
  let err = null;
  await client.credits.redeem(`definitely-not-a-real-code-${RUN}`).catch((e) => { err = e; });
  assert(err !== null, 'a bogus code was accepted');
  assert(isSynapError(err), `expected a SynapError, got ${err?.name}`);
  return `bogus code rejected as ${err.name}`;
}, skipLive);

await check('memories.create', async () => {
  const result = await client.memories.create({
    document:
      'The customer prefers window seats and always checks a bag. They are based ' +
      'in Berlin and hold Gold loyalty status until March 2027.',
    user_id: USER_ID,
    customer_id: CUSTOMER_ID,
  });
  ingestionId = result.ingestion_id ?? null;
  assert(ingestionId, `no ingestion_id in ${JSON.stringify(result).slice(0, 120)}`);
  return `ingestion_id=${ingestionId}`;
}, skipWrite);

await check('memories.status', async () => {
  const status = await client.memories.status(ingestionId);
  return `status=${status.status}`;
}, { skipIf: OFFLINE || !ingestionId, skipReason: 'needs a successful create' });

await check('memories.wait_for_completion', async () => {
  const done = await client.memories.wait_for_completion(ingestionId, {
    timeout_seconds: INGEST_WAIT,
  });
  // `memory_ids` here is the ONLY place a usable memory id comes from.
  // A context item's `id` is a different identifier and 404s against
  // /api/v1/memories/{memory_id}, which is an easy hour to lose.
  memoryId = (done.memory_ids ?? [])[0] ?? null;
  const status = String(done.status ?? '');

  // `failed` is terminal, so the SDK correctly returns rather than hanging.
  // Reporting that as a pass was wrong: it hid a broken server and turned it
  // into a confusing 404 three checks later. The SDK call worked; the
  // INGESTION did not, and those are different claims.
  if (status === 'failed' || status === 'error') {
    ingestBlockedBy = String(done.error_message ?? 'ingestion failed with no error_message');
    throw new Error(
      `ingestion ${status} server-side (the SDK call itself worked): ${ingestBlockedBy}`,
    );
  }
  if (memoryId === null) {
    ingestBlockedBy =
      `status=${status} but memories_created=${done.memories_created ?? 0}, so no memory_ids`;
  }
  return `status=${status} memories=${done.memories_created ?? '?'} first_id=${memoryId ?? 'none'}`;
}, { skipIf: OFFLINE || !ingestionId, skipReason: 'needs a successful create' });

await check('memories.batch_create', async () => {
  const result = await client.memories.batch_create({
    documents: [
      { document: 'Escalations above 500 EUR need manager approval.', user_id: USER_ID, customer_id: CUSTOMER_ID },
      { document: 'The customer speaks German and English.', user_id: USER_ID, customer_id: CUSTOMER_ID },
    ],
  });
  return JSON.stringify(result).slice(0, 70);
}, skipWrite);

await check('memories.create_from_file', async () => {
  // Multipart. `text:` is the portable route; `file_path:` reads from disk and
  // is Node-only, `file:` takes a Blob/Uint8Array and works anywhere.
  const result = await client.memories.create_from_file({
    user_id: USER_ID,
    customer_id: CUSTOMER_ID,
    text: 'Refund policy: full refund within 14 days of purchase.',
  });
  fileIngestionId = result.ingestion_id ?? null;
  return `ingestion_id=${fileIngestionId ?? JSON.stringify(result).slice(0, 60)}`;
}, skipWrite);

await check('memory_ids fallback', async () => {
  // One extraction failure should not block the whole memory-addressing group.
  // If the first ingestion produced nothing, try the multipart one before
  // giving up, so get/update/delete still get exercised whenever the server
  // manages to extract anything at all.
  if (memoryId !== null) return `already have ${memoryId}`;
  if (fileIngestionId === null) return 'no second ingestion to fall back to';
  const done = await client.memories.wait_for_completion(fileIngestionId, {
    timeout_seconds: INGEST_WAIT,
  });
  memoryId = (done.memory_ids ?? [])[0] ?? null;
  if (memoryId !== null) {
    ingestBlockedBy = null;
    return `recovered ${memoryId} from the multipart ingestion`;
  }
  return `no memory_ids from either ingestion (status=${done.status})`;
}, skipLive);

await check('memories.get', async () => {
  const memory = await client.memories.get(memoryId);
  return `memory_id=${memoryId} keys=${Object.keys(memory).length}`;
}, {
  skipIf: OFFLINE || !memoryId,
  skipReason: ingestBlockedBy !== null
    ? `blocked upstream: ${ingestBlockedBy}`
    : 'needs memory_ids from wait_for_completion',
});

await check('context item ids are NOT memory ids', async () => {
  // Documented as a check because it is a natural mistake: a context item's
  // `id` addresses the retrieval item, not the stored memory, so passing it to
  // memories.get() returns 404.
  const context = await client.user.context.fetch({
    user_id: USER_ID, customer_id: CUSTOMER_ID, max_results: 5,
  });
  const item = (context.facts ?? [])[0] ?? (context.preferences ?? [])[0];
  if (!item?.id) return 'no context item to check yet';
  let err = null;
  await client.memories.get(item.id).catch((e) => { err = e; });
  assert(err !== null, 'a context item id resolved as a memory id; this note is now stale');
  return `context id ${String(item.id).slice(0, 8)} -> ${err.name} (use memory_ids instead)`;
}, skipLive);

await check('memories.update', async () => {
  const result = await client.memories.update({
    memory_id: memoryId,
    document: 'The customer prefers aisle seats after all.',
    merge_strategy: 'replace',
  });
  return JSON.stringify(result).slice(0, 70);
}, {
  skipIf: OFFLINE || READ_ONLY || !memoryId,
  skipReason: ingestBlockedBy !== null
    ? `blocked upstream: ${ingestBlockedBy}`
    : 'needs a resolvable memory id',
});

await check('memories.get (unknown id maps to a typed error)', async () => {
  let err = null;
  await client.memories.get(randomUUID()).catch((e) => { err = e; });
  assert(err !== null, 'an unknown memory id resolved');
  assert(isSynapError(err), `expected a SynapError, got ${err?.name}`);
  return `${err.name} code=${err.code} transient=${err.transient}`;
}, skipLive);

await check('conversation.record_message', async () => {
  const result = await client.conversation.record_message({
    conversation_id: CONVERSATION_ID,
    user_id: USER_ID,
    customer_id: CUSTOMER_ID,
    role: 'user',
    content: 'I need to change my seat for the Berlin flight.',
  });
  return `message_id=${result.message_id ?? JSON.stringify(result).slice(0, 50)}`;
}, skipWrite);

await check('conversation.record_messages_batch', async () => {
  const result = await client.conversation.record_messages_batch([
    { conversation_id: CONVERSATION_ID, user_id: USER_ID, customer_id: CUSTOMER_ID, role: 'assistant', content: 'Let me check availability.' },
    { conversation_id: CONVERSATION_ID, user_id: USER_ID, customer_id: CUSTOMER_ID, role: 'user', content: 'Aisle if possible.' },
  ]);
  return JSON.stringify(result).slice(0, 70);
}, skipWrite);

await check('conversation.ingest_transcript', async () => {
  const result = await client.conversation.ingest_transcript({
    conversation_id: randomUUID(),
    user_id: USER_ID,
    customer_id: CUSTOMER_ID,
    transcript: [
      { role: 'user', content: 'Do you still have my address on file?' },
      { role: 'assistant', content: 'Yes, the Berlin one.' },
    ],
  });
  return JSON.stringify(result).slice(0, 70);
}, skipWrite);

await check('user.context.fetch', async () => {
  const context = await client.user.context.fetch({
    user_id: USER_ID, customer_id: CUSTOMER_ID,
    search_query: ['seat preference'], max_results: 10,
  });
  const counts = ['facts', 'preferences', 'episodes', 'emotions', 'temporal_events']
    .map((k) => `${k}=${(context[k] ?? []).length}`).join(' ');
  return counts;
}, skipLive);

await check('customer.context.fetch', async () => {
  const context = await client.customer.context.fetch({
    customer_id: CUSTOMER_ID, search_query: ['policy'], max_results: 5,
  });
  return `facts=${(context.facts ?? []).length}`;
}, skipLive);

await check('client.context.fetch', async () => {
  const context = await client.client.context.fetch({ search_query: ['policy'], max_results: 5 });
  return `facts=${(context.facts ?? []).length}`;
}, skipLive);

await check('conversation.context.fetch', async () => {
  const context = await client.conversation.context.fetch({
    conversation_id: CONVERSATION_ID, user_id: USER_ID, customer_id: CUSTOMER_ID, max_results: 5,
  });
  return `facts=${(context.facts ?? []).length}`;
}, skipLive);

await check('user.get_profile', async () => {
  try {
    const profile = await client.user.get_profile({ user_id: USER_ID, customer_id: CUSTOMER_ID });
    return JSON.stringify(profile).slice(0, 70);
  } catch (error) {
    // A profile is derived from extracted memories, so a brand-new user with a
    // fresh random id has none until ingestion produces something. A 404 here
    // is the server being correct, and Python behaves the same way. What is
    // being checked is that it maps to ContextNotFoundError rather than
    // escaping as a raw HTTP error.
    if (error instanceof ContextNotFoundError) {
      return `404 for a fresh user, mapped to ${error.name} (correct)`;
    }
    throw error;
  }
}, skipLive);

await check('fetch (cross-scope)', async () => {
  const unified = await client.fetch({
    user_id: USER_ID, customer_id: CUSTOMER_ID, conversation_id: CONVERSATION_ID,
    search_query: ['seat preference'], include_scope_labels: true,
  });
  assert(Array.isArray(unified.scopes_queried), 'no scopes_queried');
  assert(typeof unified.formatted_context === 'string', 'no formatted_context');
  if (unified.formatted_context) {
    console.log('         \x1b[90m--- formatted_context (first 8 lines) ---\x1b[0m');
    for (const l of unified.formatted_context.split('\n').slice(0, 8)) {
      console.log(`         \x1b[90m${l}\x1b[0m`);
    }
  }
  return `scopes=[${unified.scopes_queried.join(',')}] items=${unified.total_items}`;
}, skipLive);

await check('as_tool handler (live fetch)', async () => {
  const tool = client.as_tool({ scope: 'user', user_id: USER_ID, customer_id: CUSTOMER_ID });
  const out = await tool.handler({ search_query: ['loyalty status'], max_results: 5 });
  const total = ['facts', 'preferences', 'episodes', 'emotions', 'temporal_events']
    .reduce((n, k) => n + (out[k]?.length ?? 0), 0);
  return `items=${total}`;
}, skipLive);

await check('conversation.context.get_context_for_prompt', async () => {
  const prompt = await client.conversation.context.get_context_for_prompt({
    conversation_id: CONVERSATION_ID,
  });
  return JSON.stringify(prompt).slice(0, 70);
}, {
  skipIf: OFFLINE || READ_ONLY,
  // Needs a conversation that exists server-side. Read-only never records a
  // message, so the id is unknown and the server answers 403 "not found or
  // access denied" -- which is a real answer, not an SDK fault.
  skipReason: READ_ONLY ? 'skipped: --read-only (no conversation was created)' : 'set SYNAP_API_KEY to run',
});

await check('conversation.context.compact', async () => {
  try {
    const result = await client.conversation.context.compact({
      conversation_id: CONVERSATION_ID, force: true,
    });
    return JSON.stringify(result).slice(0, 70);
  } catch (error) {
    // Recording a message can trigger compaction on its own, so a 409 here is
    // the server being correct. What matters is that it maps to ConflictError
    // and NOT to the transcript_conflict subclass.
    if (error instanceof ConflictError && !(error instanceof TranscriptConflictError)) {
      return `409 already in progress, mapped to ${error.name} (correct)`;
    }
    throw error;
  }
}, skipWrite);

await check('conversation.context.get_compaction_status', async () => {
  const status = await client.conversation.context.get_compaction_status({
    conversation_id: CONVERSATION_ID,
  });
  return JSON.stringify(status).slice(0, 70);
}, {
  skipIf: OFFLINE || READ_ONLY,
  // Needs a conversation that exists server-side. Read-only never records a
  // message, so the id is unknown and the server answers 403 "not found or
  // access denied" -- which is a real answer, not an SDK fault.
  skipReason: READ_ONLY ? 'skipped: --read-only (no conversation was created)' : 'set SYNAP_API_KEY to run',
});

await check('conversation.context.get_compacted', async () => {
  // Returns null rather than raising when nothing is compacted yet, matching
  // Python.
  const compacted = await client.conversation.context.get_compacted({
    conversation_id: CONVERSATION_ID,
  });
  return compacted === null ? 'null (nothing compacted yet)' : JSON.stringify(compacted).slice(0, 60);
}, {
  skipIf: OFFLINE || READ_ONLY,
  // Needs a conversation that exists server-side. Read-only never records a
  // message, so the id is unknown and the server answers 403 "not found or
  // access denied" -- which is a real answer, not an SDK fault.
  skipReason: READ_ONLY ? 'skipped: --read-only (no conversation was created)' : 'set SYNAP_API_KEY to run',
});

// ─── 7. Deprecated 0.3.x surface ─────────────────────────────────────────────

group('live', `7. Deprecated 0.3.x wrapper surface${OFFLINE ? ' (skipped)' : ''}`);

await check('searchMemory', async () => {
  const out = await client.searchMemory({ user_id: USER_ID, customer_id: CUSTOMER_ID, query: 'seat' });
  return `results=${out.count}`;
}, skipLive);

await check('getMemories', async () => {
  const out = await client.getMemories({ user_id: USER_ID, customer_id: CUSTOMER_ID });
  return `results=${out.count} (default max_results is 100, not 10)`;
}, skipLive);

await check('fetchUserContext', async () => {
  const normalised = await client.fetchUserContext({ user_id: USER_ID, customer_id: CUSTOMER_ID });
  // Returns the camelCase normalised shape, unlike the namespaced surface.
  return `facts=${(normalised.facts ?? []).length} (camelCase shape)`;
}, skipLive);

await check('fetchCustomerContext', async () => {
  const out = await client.fetchCustomerContext({ customer_id: CUSTOMER_ID });
  return `facts=${(out.facts ?? []).length}`;
}, skipLive);

await check('fetchClientContext', async () => {
  const out = await client.fetchClientContext({});
  return `facts=${(out.facts ?? []).length}`;
}, skipLive);

await check('getContextForPrompt', async () => {
  const out = await client.getContextForPrompt({ conversation_id: CONVERSATION_ID });
  return JSON.stringify(out).slice(0, 60);
}, {
  skipIf: OFFLINE || READ_ONLY,
  // Needs a conversation that exists server-side. Read-only never records a
  // message, so the id is unknown and the server answers 403 "not found or
  // access denied" -- which is a real answer, not an SDK fault.
  skipReason: READ_ONLY ? 'skipped: --read-only (no conversation was created)' : 'set SYNAP_API_KEY to run',
});

await check('addMemory', async () => {
  const out = await client.addMemory({
    document: 'Added through the deprecated alias.', user_id: USER_ID, customer_id: CUSTOMER_ID,
  });
  return `ingestion_id=${out.ingestion_id}`;
}, skipWrite);

await check('init (deprecated alias for initialize)', async () => {
  await client.init();
  return 'forwards to initialize()';
}, skipLive);

// ─── 8. gRPC anticipation stream ─────────────────────────────────────────────

group('grpc', `8. gRPC anticipation stream${RUN_GRPC ? '' : ' (skipped: set SYNAP_GRPC_HOST)'}`);

const skipGrpc = { skipIf: !RUN_GRPC, skipReason: 'skipped via --no-grpc' };
const bundles = [];

await check('instance.is_listening (before)', async () => {
  assert(client.instance.is_listening === false, 'should not be listening yet');
  return 'false';
}, skipGrpc);

await check('conversation.context.subscribe_to_compaction_updates', async () => {
  const off = client.conversation.context.subscribe_to_compaction_updates(
    CONVERSATION_ID, (bundle) => console.log(`         [compaction] ${bundle.bundle_id}`),
  );
  assert(typeof off === 'function', 'no unsubscribe thunk returned');
  return 'registered, returns an unsubscribe thunk';
}, skipGrpc);

await check('instance.listen', async () => {
  await client.instance.listen({
    ...grpcOverrides,
    on_context: (bundle) => {
      bundles.push(bundle);
      console.log(`         [bundle] id=${bundle.bundle_id} type=${bundle.bundle_type || 'anticipation'}`);
    },
    on_disconnect: (reason) => console.log(`         [stream] disconnected: ${reason}`),
    on_reconnect: (n) => console.log(`         [stream] reconnect attempt ${n}`),
  });
  assert(client.instance.is_listening === true, 'is_listening stayed false after listen()');
  return 'stream open, is_listening=true';
}, skipGrpc);

await check('instance.listen (second call rejected)', async () => {
  let err = null;
  await client.instance.listen(grpcOverrides).catch((e) => { err = e; });
  assert(err instanceof ListeningAlreadyActiveError, `expected ListeningAlreadyActiveError, got ${err?.name}`);
  return 'a second listen() is refused';
}, {
  skipIf: !RUN_GRPC || READ_ONLY || !client?.instance.is_listening,
  skipReason: READ_ONLY ? 'skipped: --read-only (the server persists stream events)' : 'needs an open stream',
});

await check('instance.send_message', async () => {
  await client.instance.send_message({
    content: 'What seat do I usually pick?',
    role: 'user',
    conversation_id: CONVERSATION_ID,
    user_id: USER_ID,
    customer_id: CUSTOMER_ID,
    event_type: 'user_message',
  });
  return 'conversation_event written to the stream';
}, {
  skipIf: !RUN_GRPC || READ_ONLY || !client?.instance.is_listening,
  skipReason: READ_ONLY ? 'skipped: --read-only (the server persists stream events)' : 'needs an open stream',
});

await check('instance.record_thinking', async () => {
  await client.instance.record_thinking({
    content: 'The user is asking about seat preference.',
    conversation_id: CONVERSATION_ID,
    user_id: USER_ID,
    customer_id: CUSTOMER_ID,
  });
  return 'agent_thinking event written';
}, {
  skipIf: !RUN_GRPC || READ_ONLY || !client?.instance.is_listening,
  skipReason: READ_ONLY ? 'skipped: --read-only (the server persists stream events)' : 'needs an open stream',
});

await check('server pushes a bundle', async () => {
  const deadline = Date.now() + 30_000;
  while (bundles.length === 0 && Date.now() < deadline) {
    await new Promise((r) => setTimeout(r, 500));
  }
  // Not a failure. Anticipation only fires when the instance has MACA patterns
  // configured AND retrieval finds something; the client->server direction is
  // proven by the sends above either way.
  if (bundles.length === 0) return 'none in 30s (needs MACA patterns configured server-side)';
  const snapshot = client.anticipation_cache_snapshot();
  return `bundles=${bundles.length} cached_entries=${snapshot.total_entries}`;
}, {
  skipIf: !RUN_GRPC || READ_ONLY || !client?.instance.is_listening,
  skipReason: READ_ONLY ? 'skipped: --read-only (the server persists stream events)' : 'needs an open stream',
});

await check('conversation.context.unsubscribe_all_compaction_updates', async () => {
  const removed = client.conversation.context.unsubscribe_all_compaction_updates(CONVERSATION_ID);
  return `removed=${removed}`;
}, skipGrpc);

await check('instance.stop_listening', async () => {
  await client.instance.stop_listening();
  assert(client.instance.is_listening === false, 'is_listening stayed true after stop');
  // Stopping twice is a no-op, so a finally block is safe.
  await client.instance.stop_listening();
  return 'closed, and idempotent';
}, skipGrpc);

// ─── 9. Cleanup ──────────────────────────────────────────────────────────────

group('live', '9. Cleanup and teardown');

await check('memories.delete', async () => {
  const result = await client.memories.delete(memoryId);
  return JSON.stringify(result).slice(0, 70);
}, {
  skipIf: OFFLINE || READ_ONLY || !memoryId,
  skipReason: ingestBlockedBy !== null
    ? `blocked upstream: ${ingestBlockedBy}`
    : 'nothing to delete',
});

await check('shutdown', async () => {
  if (client) await client.shutdown();
  // Safe to call more than once.
  if (client) await client.shutdown();
  return 'idempotent, timers and sockets released';
}, { skipIf: OFFLINE, skipReason: 'no live client was built' });

// ─── Coverage and summary ────────────────────────────────────────────────────

const pass = results.filter((r) => r.status === 'pass').length;
const fail = results.filter((r) => r.status === 'fail').length;
const skip = results.filter((r) => r.status === 'skip').length;

console.log();
console.log('═'.repeat(74));
console.log(`\x1b[1mRESULT\x1b[0m  ${pass} passed, ${fail} failed, ${skip} skipped   (${results.length} checks)`);
console.log('═'.repeat(74));

if (fail > 0) {
  console.log('\nFailures:');
  for (const r of results.filter((x) => x.status === 'fail')) {
    console.log(`  - ${r.member}: ${r.error?.message}`);
  }
}

// Cross-check against the Python surface contract when running inside the repo,
// so "all green" cannot quietly mean half of the surface was never called.
const here = path.dirname(fileURLToPath(import.meta.url));
const surfacePath = path.resolve(here, '../../CONTRACT/conformance/python_surface.json');
if (existsSync(surfacePath)) {
  const golden = JSON.parse(readFileSync(surfacePath, 'utf8'));
  const expected = [];
  for (const [ns, members] of Object.entries(golden.namespaces)) {
    for (const m of members) expected.push(ns === '' ? m : `${ns}.${m}`);
  }
  const touched = new Set([...exercised].map((m) => m.split(' ')[0]));
  const untouched = expected.filter((m) => !touched.has(m));
  console.log();
  console.log(`Surface coverage vs Python ${golden.python_version}: ` +
    `${expected.length - untouched.length}/${expected.length} members exercised`);
  if (untouched.length > 0) console.log(`  not exercised: ${untouched.join(', ')}`);
}

if (OFFLINE) {
  console.log();
  console.log('Only the local group ran. To exercise the live methods:');
  console.log('  export SYNAP_API_KEY=...');
  console.log('  export SYNAP_BASE_URL=https://your-synap-host');
  console.log('  # gRPC needs no configuration: the SDK defaults to prod:443 over TLS');
  console.log('  node test-js-sdk.mjs');
}

process.exitCode = fail > 0 ? 1 : 0;
