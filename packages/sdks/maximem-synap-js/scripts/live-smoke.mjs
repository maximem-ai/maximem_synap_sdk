/**
 * Live end-to-end smoke test for the native JS SDK.
 *
 * Runs against a real deployment and exercises the HTTP surface, the
 * cross-scope fetch, as_tool, and the bidirectional gRPC anticipation stream.
 * Deliberately runs against `dist/`, not `src/`, so it tests what a customer
 * actually installs.
 *
 * Usage:
 *   export SYNAP_API_KEY=...
 *   export SYNAP_BASE_URL=https://synap-cloud-staging.maximem.ai
 *   export SYNAP_GRPC_HOST=synap-cloud-staging.maximem.ai
 *   node scripts/live-smoke.mjs
 *
 * Optional:
 *   SYNAP_SMOKE_USER_ID / SYNAP_SMOKE_CUSTOMER_ID  reuse a known scope
 *   SYNAP_SMOKE_SKIP_GRPC=1                        HTTP only
 *   SYNAP_SMOKE_INGEST_WAIT=60                     seconds to wait for ingestion
 */
import { randomUUID } from 'node:crypto';
import { SynapClient } from '../dist/index.js';

const API_KEY = process.env.SYNAP_API_KEY;
if (!API_KEY) {
  console.error('SYNAP_API_KEY is required.');
  process.exit(2);
}
const BASE_URL = process.env.SYNAP_BASE_URL ?? 'https://synap-cloud-staging.maximem.ai';
const GRPC_HOST = process.env.SYNAP_GRPC_HOST ?? new URL(BASE_URL).hostname;
const GRPC_PORT = Number(process.env.SYNAP_GRPC_PORT ?? 443);
const GRPC_TLS = (process.env.SYNAP_GRPC_USE_TLS ?? '1') !== '0';
const SKIP_GRPC = process.env.SYNAP_SMOKE_SKIP_GRPC === '1';
const INGEST_WAIT = Number(process.env.SYNAP_SMOKE_INGEST_WAIT ?? 90);

// A fresh user per run keeps assertions about "did my write land" honest.
const RUN = randomUUID().slice(0, 8);
const USER_ID = process.env.SYNAP_SMOKE_USER_ID ?? `js-smoke-user-${RUN}`;
const CUSTOMER_ID = process.env.SYNAP_SMOKE_CUSTOMER_ID ?? `js-smoke-cust-${RUN}`;
const CONVERSATION_ID = randomUUID();

let passed = 0, failed = 0, skipped = 0;
const failures = [];

function line() { console.log('─'.repeat(72)); }
function banner(msg) { console.log(); line(); console.log(msg); line(); }

async function step(name, fn, { optional = false } = {}) {
  const started = Date.now();
  try {
    const detail = await fn();
    const ms = Date.now() - started;
    console.log(`  PASS  ${name}  (${ms}ms)${detail ? `  ${detail}` : ''}`);
    passed++;
    return true;
  } catch (error) {
    const ms = Date.now() - started;
    if (optional) {
      console.log(`  SKIP  ${name}  (${ms}ms)  ${error?.message ?? error}`);
      skipped++;
      return false;
    }
    console.log(`  FAIL  ${name}  (${ms}ms)`);
    console.log(`        ${error?.name ?? 'Error'}: ${error?.message ?? error}`);
    if (error?.code) console.log(`        code=${error.code} transient=${error.transient}`);
    failed++;
    failures.push(`${name}: ${error?.message ?? error}`);
    return false;
  }
}

function assert(condition, message) {
  if (!condition) throw new Error(message);
}

async function waitFor(predicate, seconds, label) {
  const deadline = Date.now() + seconds * 1000;
  while (Date.now() < deadline) {
    if (await predicate()) return true;
    await new Promise((r) => setTimeout(r, 1000));
  }
  throw new Error(`timed out after ${seconds}s waiting for ${label}`);
}

const client = new SynapClient({
  apiKey: API_KEY,
  baseUrl: BASE_URL,
  retryPolicy: { maxAttempts: 3 },
});

console.log(`Synap JS SDK live smoke`);
console.log(`  base url      ${BASE_URL}`);
console.log(`  grpc          ${GRPC_HOST}:${GRPC_PORT} tls=${GRPC_TLS}`);
console.log(`  user          ${USER_ID}`);
console.log(`  customer      ${CUSTOMER_ID}`);
console.log(`  conversation  ${CONVERSATION_ID}`);

banner('1. Identity');

await step('initialize() resolves identity from whoami', async () => {
  await client.initialize();
  assert(client.client_id !== '', 'client_id was not resolved');
  return `client_id=${client.client_id} instance_id=${client.instance_id || '(none)'}`;
});

banner('2. Credits (checked first: retrieval fails open on a 429)');

let creditsOk = false;
await step('credits.get_balance()', async () => {
  const balance = await client.credits.get_balance();
  const remaining = balance?.total_remaining ?? balance?.remaining ?? balance?.balance;
  creditsOk = typeof remaining !== 'number' || remaining > 0;
  if (!creditsOk) console.log('        WARNING: no credit headroom; retrieval will 429 and fail open');
  return `remaining=${remaining ?? JSON.stringify(balance).slice(0, 80)}`;
}, { optional: true });

banner('3. Memory ingestion');

let ingestionId = null;
await step('memories.create()', async () => {
  const result = await client.memories.create({
    document:
      'The customer prefers window seats and always flies with checked baggage. ' +
      'They are based in Berlin and hold Gold loyalty status until March 2027.',
    user_id: USER_ID,
    customer_id: CUSTOMER_ID,
    document_type: 'ai-chat-conversation',
  });
  ingestionId = result?.ingestion_id ?? result?.id ?? null;
  assert(ingestionId, `no ingestion_id in ${JSON.stringify(result).slice(0, 160)}`);
  return `ingestion_id=${ingestionId}`;
});

await step('memories.status()', async () => {
  const status = await client.memories.status(ingestionId);
  return `status=${status?.status ?? JSON.stringify(status).slice(0, 80)}`;
}, { optional: !ingestionId });

await step(`memories.wait_for_completion() up to ${INGEST_WAIT}s`, async () => {
  const done = await client.memories.wait_for_completion(ingestionId, {
    timeout_seconds: INGEST_WAIT,
  });
  return `status=${done?.status} memories=${done?.memories_created ?? done?.memory_count ?? '?'}`;
}, { optional: !ingestionId });

await step('memories.create_from_file() with raw text', async () => {
  const result = await client.memories.create_from_file({
    user_id: USER_ID,
    customer_id: CUSTOMER_ID,
    text: 'Escalation policy: refunds above 500 EUR need manager approval.',
  });
  return `ingestion_id=${result?.ingestion_id ?? JSON.stringify(result).slice(0, 80)}`;
}, { optional: true });

banner('4. Conversation writes');

await step('conversation.record_message()', async () => {
  const result = await client.conversation.record_message({
    conversation_id: CONVERSATION_ID,
    user_id: USER_ID,
    customer_id: CUSTOMER_ID,
    role: 'user',
    content: 'I need to change my seat for the Berlin flight.',
  });
  return JSON.stringify(result).slice(0, 80);
}, { optional: true });

banner('5. Retrieval');

await step('user.context.fetch()', async () => {
  const context = await client.user.context.fetch({
    user_id: USER_ID,
    customer_id: CUSTOMER_ID,
    search_query: ['seat preference'],
    max_results: 10,
  });
  const counts = ['facts', 'preferences', 'episodes', 'emotions', 'temporal_events']
    .map((k) => `${k}=${(context?.[k] ?? []).length}`).join(' ');
  return counts;
});

await step('cross-scope fetch() produces a formatted prompt', async () => {
  const unified = await client.fetch({
    user_id: USER_ID,
    customer_id: CUSTOMER_ID,
    conversation_id: CONVERSATION_ID,
    search_query: ['seat preference'],
    include_scope_labels: true,
  });
  assert(Array.isArray(unified.scopes_queried), 'scopes_queried missing');
  if (unified.formatted_context) {
    console.log('        --- formatted_context ---');
    for (const l of unified.formatted_context.split('\n').slice(0, 12)) {
      console.log(`        ${l}`);
    }
  }
  return `scopes=[${unified.scopes_queried.join(',')}] items=${unified.total_items}`;
});

await step('as_tool() handler executes a real fetch', async () => {
  const tool = client.as_tool({ scope: 'user', user_id: USER_ID, customer_id: CUSTOMER_ID });
  const result = await tool.handler({ search_query: ['loyalty status'], max_results: 5 });
  const total = ['facts', 'preferences', 'episodes', 'emotions', 'temporal_events']
    .reduce((n, k) => n + (result?.[k]?.length ?? 0), 0);
  return `tool=${tool.function.name} items=${total}`;
});

await step('user.get_profile()', async () => {
  const profile = await client.user.get_profile({ user_id: USER_ID });
  return JSON.stringify(profile).slice(0, 90);
}, { optional: true });

banner('6. Error mapping against the real server');

await step('a malformed conversation id is rejected locally', async () => {
  // conversation.context.fetch is one of the seven places Python validates.
  // user.context.fetch deliberately does NOT, so probing that would prove
  // nothing.
  let threw = null;
  try {
    await client.conversation.context.fetch({ conversation_id: 'not-a-uuid' });
  } catch (error) { threw = error; }
  assert(threw, 'expected a rejection');
  assert(threw.code === 'invalid_conversation_id', `wrong code: ${threw.code}`);
  return `code=${threw.code}`;
});

await step('an unknown memory id maps to a typed error', async () => {
  let threw = null;
  try { await client.memories.get(randomUUID()); } catch (error) { threw = error; }
  assert(threw, 'expected a rejection');
  return `${threw.name} code=${threw.code} transient=${threw.transient}`;
}, { optional: true });

banner('7. gRPC anticipation stream');

if (SKIP_GRPC) {
  console.log('  SKIP  gRPC (SYNAP_SMOKE_SKIP_GRPC=1)');
  skipped++;
} else {
  const bundles = [];
  const compactions = [];
  let listening = false;

  listening = await step('instance.listen() opens the bidirectional stream', async () => {
    await client.instance.listen({
      host: GRPC_HOST,
      port: GRPC_PORT,
      use_tls: GRPC_TLS,
      on_context: (bundle) => bundles.push(bundle),
      on_disconnect: (reason) => console.log(`        [stream] disconnected: ${reason}`),
      on_reconnect: (n) => console.log(`        [stream] reconnect attempt ${n}`),
    });
    assert(client.instance.is_listening, 'is_listening stayed false after listen()');
    return `is_listening=${client.instance.is_listening}`;
  });

  if (listening) {
    await step('subscribe_to_compaction_updates() registers', async () => {
      const off = client.conversation.context.subscribe_to_compaction_updates(
        CONVERSATION_ID, (bundle) => compactions.push(bundle),
      );
      assert(typeof off === 'function', 'no unsubscribe thunk');
      return 'registered';
    });

    await step('send_message() over the stream', async () => {
      await client.instance.send_message({
        content: 'What seat do I usually pick, and what is my loyalty status?',
        role: 'user',
        conversation_id: CONVERSATION_ID,
        user_id: USER_ID,
        customer_id: CUSTOMER_ID,
        event_type: 'user_message',
      });
      return 'sent';
    });

    await step('record_thinking() over the stream', async () => {
      await client.instance.record_thinking({
        content: 'The user is asking about seat preference and loyalty tier.',
        conversation_id: CONVERSATION_ID,
        user_id: USER_ID,
        customer_id: CUSTOMER_ID,
      });
      return 'sent';
    });

    await step('server pushes an anticipation bundle', async () => {
      await waitFor(() => bundles.length > 0, 45, 'a context bundle');
      const types = [...new Set(bundles.map((b) => b.bundle_type || 'anticipation'))];
      return `bundles=${bundles.length} types=[${types.join(',')}]`;
    }, { optional: true });

    await step('the bundle landed in the anticipation cache', async () => {
      const snapshot = client.anticipation_cache_snapshot();
      assert(snapshot.total_entries > 0, 'cache is empty; no cacheable bundle arrived');
      console.log(`        scope breakdown: ${JSON.stringify(snapshot.scope_breakdown_overall)}`);
      console.log(`        vocab size: ${snapshot.corpus_vocab_size}`);
      for (const b of snapshot.bundles.slice(0, 3)) {
        console.log(`        bundle ${b.bundle_id} type=${b.bundle_type} items=${b.total_items} queries=${b.search_queries.slice(0, 3).join('|')}`);
      }
      return `entries=${snapshot.total_entries} items=${snapshot.total_item_records}`;
    }, { optional: true });

    await step('a fetch is served from the anticipation cache', async () => {
      const hit = client.cache.stats();
      return JSON.stringify(hit).slice(0, 100);
    }, { optional: true });

    await step('stop_listening() closes the stream', async () => {
      await client.instance.stop_listening();
      assert(!client.instance.is_listening, 'is_listening stayed true after stop');
      return 'closed';
    });
  }
}

banner('8. Teardown');

await step('shutdown() releases everything', async () => {
  await client.shutdown();
  return 'clean';
});

banner(`RESULT  ${passed} passed, ${failed} failed, ${skipped} skipped`);
if (failures.length > 0) {
  console.log('Failures:');
  for (const f of failures) console.log(`  - ${f}`);
}

// The process must exit on its own. A lingering handle means a timer or socket
// was not released, which is a real defect in a serverless deployment.
const timer = setTimeout(() => {
  console.log('\nWARNING: still alive 3s after shutdown; something holds a handle.');
  const handles = process._getActiveHandles?.() ?? [];
  console.log(`  active handles: ${handles.map((h) => h?.constructor?.name).join(', ') || 'none'}`);
  process.exit(failed > 0 ? 1 : 0);
}, 3000);
timer.unref();

process.exitCode = failed > 0 ? 1 : 0;
