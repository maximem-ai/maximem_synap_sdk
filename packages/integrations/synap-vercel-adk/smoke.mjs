// End-to-end smoke test for @maximem/synap-vercel-adk.
// Runs against the built dist/. No network required.

import {
  AnticipationCache,
  buildContextSystemBlock,
  injectContextIntoPrompt,
  extractSearchQuery,
  promptToTranscript,
  createSynapMiddleware,
} from './dist/index.js';

let passed = 0, failed = 0;
const results = [];

function eq(a, b, msg) {
  const ok = JSON.stringify(a) === JSON.stringify(b);
  if (!ok) throw new Error(`${msg}: expected ${JSON.stringify(b)}, got ${JSON.stringify(a)}`);
}
function truthy(v, msg) { if (!v) throw new Error(msg); }
function contains(s, sub, msg) { if (!String(s).includes(sub)) throw new Error(`${msg}: ${JSON.stringify(s)} does not contain ${JSON.stringify(sub)}`); }

async function run(name, fn) {
  try { await fn(); passed++; results.push(`  PASS  ${name}`); }
  catch (e) { failed++; results.push(`  FAIL  ${name}\n        ${e.message}`); }
}

// ── Mock data ──────────────────────────────────────────────────────────────
const RICH_CTX = {
  facts: [{ id: 'f1', content: 'Senior engineer at Acme', contextType: 'fact', confidence: 0.9, source: 'profile' }],
  preferences: [{ id: 'p1', content: 'Prefers concise answers', contextType: 'preference', confidence: 0.8, source: 'history' }],
  episodes: [{ id: 'e1', content: 'Worked on distributed systems', contextType: 'episode', confidence: 0.7, source: 'chat' }],
  emotions: [],
  temporalEvents: [],
  conversationContext: {
    summary: 'Discussing architecture choices',
    currentState: {},
    keyExtractions: {},
    recentTurns: [
      { role: 'user', content: 'What about sharding?', timestamp: '2026-04-21T10:00:00Z' },
      { role: 'assistant', content: 'Sharding works if...', timestamp: '2026-04-21T10:00:01Z' },
    ],
    compactionId: null,
    conversationId: 'conv-1',
  },
  source: 'cache',
  correlationId: 'corr-1',
};
const EMPTY_CTX = { facts: [], preferences: [], episodes: [], emotions: [], temporalEvents: [], conversationContext: null, source: 'cache', correlationId: '' };
const UNICODE_CTX = {
  ...EMPTY_CTX,
  facts: [{ id: 'f', content: '🚀 ships to 東京 in "Q2"', contextType: 'fact', confidence: 1, source: 'x' }],
};

const FAKE_CREDS = { api_key: 'sk-test', client_id: 'cli_x', instance_id: 'inst_x' };

// ── transforms ─────────────────────────────────────────────────────────────

await run('buildContextSystemBlock: RICH → structured sections', () => {
  const s = buildContextSystemBlock(RICH_CTX);
  contains(s, '<synap_context>', 'open tag');
  contains(s, '## Known Facts', 'facts heading');
  contains(s, 'Senior engineer at Acme', 'fact body');
  contains(s, '## User Preferences', 'prefs heading');
  contains(s, '## Recent Conversation', 'recent heading');
  contains(s, 'What about sharding?', 'recent turn');
});

await run('buildContextSystemBlock: EMPTY → empty string', () => {
  eq(buildContextSystemBlock(EMPTY_CTX), '', 'empty');
});

await run('buildContextSystemBlock: UNICODE preserved', () => {
  const s = buildContextSystemBlock(UNICODE_CTX);
  contains(s, '🚀', 'rocket');
  contains(s, '東京', 'cjk');
});

await run('injectContextIntoPrompt: no existing system → prepends', () => {
  const prompt = [{ role: 'user', content: [{ type: 'text', text: 'hi' }] }];
  const out = injectContextIntoPrompt(prompt, RICH_CTX);
  eq(out[0].role, 'system', 'first is system');
  contains(out[0].content, '<synap_context>', 'context injected');
  eq(out[1], prompt[0], 'user preserved');
});

await run('injectContextIntoPrompt: existing system → prepends block to content', () => {
  const prompt = [
    { role: 'system', content: 'You are helpful.' },
    { role: 'user', content: [{ type: 'text', text: 'hi' }] },
  ];
  const out = injectContextIntoPrompt(prompt, RICH_CTX);
  eq(out.length, 2, 'length preserved');
  eq(out[0].role, 'system', 'still system');
  contains(out[0].content, '<synap_context>', 'block present');
  contains(out[0].content, 'You are helpful.', 'original preserved');
});

await run('injectContextIntoPrompt: EMPTY ctx → prompt unchanged', () => {
  const prompt = [{ role: 'user', content: [{ type: 'text', text: 'hi' }] }];
  const out = injectContextIntoPrompt(prompt, EMPTY_CTX);
  eq(out, prompt, 'unchanged');
});

await run('extractSearchQuery: takes last user message text', () => {
  const prompt = [
    { role: 'user', content: [{ type: 'text', text: 'first' }] },
    { role: 'assistant', content: [{ type: 'text', text: 'reply' }] },
    { role: 'user', content: [{ type: 'text', text: 'second Q' }, { type: 'text', text: '?' }] },
  ];
  eq(extractSearchQuery(prompt), ['second Q ?'], 'concatenated');
});

await run('extractSearchQuery: no user → []', () => {
  eq(extractSearchQuery([{ role: 'system', content: 'x' }]), [], 'empty');
});

await run('extractSearchQuery: truncated at 512 chars', () => {
  const long = 'x'.repeat(1000);
  const prompt = [{ role: 'user', content: [{ type: 'text', text: long }] }];
  const [q] = extractSearchQuery(prompt);
  eq(q.length, 512, '512 cap');
});

await run('promptToTranscript: filters non-text parts + system role', () => {
  const prompt = [
    { role: 'system', content: 'sys' },
    { role: 'user', content: [{ type: 'text', text: 'hello' }, { type: 'image', image: 'x' }] },
    { role: 'assistant', content: [{ type: 'text', text: 'hi' }] },
  ];
  eq(promptToTranscript(prompt), [
    { role: 'user', content: 'hello' },
    { role: 'assistant', content: 'hi' },
  ], 'transcript');
});

await run('promptToTranscript: drops empty-content messages', () => {
  const prompt = [{ role: 'user', content: [{ type: 'image', image: 'x' }] }];
  eq(promptToTranscript(prompt), [], 'dropped');
});

// ── AnticipationCache ──────────────────────────────────────────────────────

function makeBundle(overrides = {}) {
  return {
    itemsByType: {
      facts: [{
        item_id: 'f1', content: 'User loves sharding and distributed systems architecture',
        context_type: 'fact', source: 's', confidence: 0.9, similarity_score: 0.9,
        relevance_score: 0.9, scope: 'user', entity_id: '', event_date: '', valid_until: '',
        temporal_category: '', temporal_confidence: 0,
      }],
    },
    conversationContext: null,
    bundleType: 'anticipation',
    userId: 'u1', customerId: 'c1', conversationId: 'conv-1',
    searchKeywords: ['sharding', 'distributed', 'systems'],
    searchQueries: ['sharding architecture'],
    storedAt: Date.now(), ttl: 300_000,
    ...overrides,
  };
}

await run('AnticipationCache: store + lookup scope hit', () => {
  const c = new AnticipationCache();
  c.store(makeBundle());
  const hit = c.lookup({ userId: 'u1', customerId: 'c1', conversationId: 'conv-1', searchQuery: ['sharding'] });
  truthy(hit, 'got hit');
  eq(hit.facts.length, 1, 'one fact');
  eq(hit.facts[0].content.includes('sharding'), true, 'content preserved');
});

await run('AnticipationCache: scope mismatch → null', () => {
  const c = new AnticipationCache();
  c.store(makeBundle());
  eq(c.lookup({ userId: 'other', searchQuery: ['sharding'] }), null, 'miss');
});

await run('AnticipationCache: novel-term gate rejects unknown query', () => {
  const c = new AnticipationCache();
  c.store(makeBundle());
  eq(c.lookup({ userId: 'u1', searchQuery: ['completely foreign unrelated tokens xyzzy'] }), null, 'gate');
});

await run('AnticipationCache: invalidateConversation removes entry', () => {
  const c = new AnticipationCache();
  c.store(makeBundle());
  c.invalidateConversation('conv-1');
  eq(c.size(), 0, 'cleared');
});

await run('AnticipationCache: lookupUserSummary filters by bundleType', () => {
  const c = new AnticipationCache();
  c.store(makeBundle({ bundleType: 'user_summary' }));
  truthy(c.lookupUserSummary('u1'), 'found');
  eq(c.lookupUserSummary('nobody'), null, 'miss');
});

// ── middleware (via AnticipationCache hit — no HTTP) ───────────────────────

function makeMw({ cache, injectContext = true, ids = { userId: 'u1', customerId: 'c1', conversationId: 'conv-1' } } = {}) {
  return createSynapMiddleware({
    credentials: FAKE_CREDS,
    anticipationCache: cache ?? new AnticipationCache(),
    grpcClient: null,
    baseUrl: 'http://127.0.0.1:1',  // unreachable; fire-and-forget writes may fail silently
    injectContext,
    writeMemory: false,
    ...ids,
  });
}

await run('middleware.transformParams: cache hit → injects context', async () => {
  const cache = new AnticipationCache();
  cache.store(makeBundle());
  const mw = makeMw({ cache });
  const prompt = [{ role: 'user', content: [{ type: 'text', text: 'sharding distributed systems' }] }];
  const out = await mw.transformParams({ type: 'generate', params: { prompt } });
  eq(out.prompt[0].role, 'system', 'system injected');
  contains(out.prompt[0].content, '<synap_context>', 'context block');
  eq(out.prompt[1], prompt[0], 'user preserved');
});

await run('middleware.transformParams: no ids → pass-through', async () => {
  const mw = createSynapMiddleware({
    credentials: FAKE_CREDS, anticipationCache: new AnticipationCache(),
    grpcClient: null, baseUrl: 'http://127.0.0.1:1',
  });
  const prompt = [{ role: 'user', content: [{ type: 'text', text: 'hi' }] }];
  const out = await mw.transformParams({ type: 'generate', params: { prompt } });
  eq(out.prompt, prompt, 'unchanged');
});

await run('middleware.transformParams: injectContext=false → pass-through even with ids/cache', async () => {
  const cache = new AnticipationCache(); cache.store(makeBundle());
  const mw = makeMw({ cache, injectContext: false });
  const prompt = [{ role: 'user', content: [{ type: 'text', text: 'sharding' }] }];
  const out = await mw.transformParams({ type: 'generate', params: { prompt } });
  eq(out.prompt, prompt, 'unchanged');
});

await run('middleware.wrapGenerate: returns underlying result, pops prompt stack', async () => {
  const mw = makeMw();
  const prompt = [{ role: 'user', content: [{ type: 'text', text: 'hi' }] }];
  await mw.transformParams({ type: 'generate', params: { prompt } });
  const result = await mw.wrapGenerate({ doGenerate: async () => ({ text: 'ok response', finishReason: 'stop', usage: {} }) });
  eq(result.text, 'ok response', 'result passthrough');
});

await run('middleware.wrapStream: accumulates text-delta, closes stream', async () => {
  const mw = makeMw();
  const prompt = [{ role: 'user', content: [{ type: 'text', text: 'hi' }] }];
  await mw.transformParams({ type: 'generate', params: { prompt } });

  const deltas = ['Hello', ', ', 'world'];
  const src = new ReadableStream({
    start(controller) {
      for (const d of deltas) controller.enqueue({ type: 'text-delta', textDelta: d });
      controller.enqueue({ type: 'finish', finishReason: 'stop', usage: {} });
      controller.close();
    },
  });
  const { stream } = await mw.wrapStream({ doStream: async () => ({ stream: src }) });
  const reader = stream.getReader();
  const out = [];
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    out.push(value);
  }
  eq(out.length, 4, 'all parts forwarded');
  eq(out[0].textDelta, 'Hello', 'first delta');
  eq(out[3].type, 'finish', 'finish forwarded');
});

// ── Report ─────────────────────────────────────────────────────────────────
console.log('\n=== synap-vercel-adk smoke test ===');
for (const r of results) console.log(r);
console.log(`\n${passed} passed, ${failed} failed`);
process.exit(failed === 0 ? 0 : 1);
