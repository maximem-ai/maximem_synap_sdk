/**
 * Local turn store and history search. Python twin:
 * synap/tests/sdk/test_history_search.py — the same cases, the same answers.
 */
import { describe, expect, it } from 'vitest';
import { SynapClient } from '../client.js';
import { renderExcerpts, searchTurns, splitPassages, toolResultText, type Encoder, type Excerpt } from '../cache/history-search.js';
import { HISTORY_SEARCH } from '../behavior/contract.js';
import { TurnStore, type StoreLimits } from '../cache/turn-store.js';

const TURNS: [string, string][] = [
  ['user', 'We are moving to Lisbon in February. The cat Biscuit flies in cabin.'],
  ['assistant', 'Noted. The CFIA endorsement must be done within ten days of the flight.'],
  ['user', 'The moving quote from Harbourline came in at CAD 11,200 for a shared container.'],
  ['assistant', 'That is CAD 3,400 cheaper than the other quote. Do you want to book the video survey?'],
  ['user', 'Yes. Also my sister Corinne said something awful about mom again.'],
  ['user', 'Update: Harbourline moved the pickup to 19 January, reference HL-22918.'],
];

const LIMITS: StoreLimits = {
  max_conversations: 200,
  max_turns_per_conversation: 5000,
  max_bytes_per_conversation: 8 << 20,
  idle_ttl_seconds: 7 * 86400,
  max_file_bytes: 200 << 20,
};

function storeWith(turns = TURNS, limits: Partial<StoreLimits> = {}): TurnStore {
  const store = new TurnStore({ limits: { ...LIMITS, ...limits } });
  turns.forEach(([role, content], i) => store.add('c1', role, content, '', { messageId: `id${i + 1}`, position: i + 1 }));
  return store;
}

describe('turn store', () => {
  it('a turn is held once and learns its position from the server', () => {
    const store = new TurnStore();
    store.add('c1', 'user', 'streamed turn');
    store.add('c1', 'user', 'streamed turn');
    store.addMany('c1', [{ role: 'user', content: 'streamed turn', message_id: 'm-9', position: 9 }]);
    const turns = store.turns('c1');
    expect(turns.map((t) => [t.message_id, t.position])).toEqual([['m-9', 9]]);
  });

  it('backfilled turns take their place in order', () => {
    const store = new TurnStore();
    store.add('c1', 'user', 'third', '', { position: 3 });
    store.addMany('c1', [
      { role: 'user', content: 'first', position: 1 },
      { role: 'assistant', content: 'second', position: 2 },
    ]);
    store.add('c1', 'assistant', 'fourth, no position yet');
    expect(store.turns('c1').map((t) => t.content)).toEqual(['first', 'second', 'third', 'fourth, no position yet']);
  });

  it('limits drop oldest turns then least recent conversations', () => {
    let clock = 1000;
    const store = new TurnStore({
      limits: { ...LIMITS, max_conversations: 2, max_turns_per_conversation: 3, idle_ttl_seconds: 100 },
      now: () => clock,
    });
    for (let i = 0; i < 5; i++) store.add('a', 'user', `turn ${i}`, '', { position: i + 1 });
    expect(store.turns('a').map((t) => t.content)).toEqual(['turn 2', 'turn 3', 'turn 4']);
    clock += 10;
    store.add('b', 'user', 'b1');
    clock += 10;
    store.add('c', 'user', 'c1');
    store.sweep();
    expect(store.conversations()).toEqual(['b', 'c']);
    clock += 1000;
    store.sweep();
    expect(store.conversations()).toEqual([]);
  });
});

const found = (xs: Excerpt[]) => xs.filter((e) => e.via !== 'previous');

describe('history search', () => {
  it('a literal value finds its turn', async () => {
    const hits = found(await searchTurns(storeWith().turns('c1'), 'what was reference HL-22918 about?'));
    expect(hits[0]?.position).toBe(6);
    expect(hits[0]?.via).toBe('literal');
  });

  it('keywords find the turns and later ones read last', async () => {
    const positions = found(await searchTurns(storeWith().turns('c1'), 'When is the Harbourline pickup?')).map((h) => h.position);
    expect(positions).toContain(6);
    expect(positions).toEqual([...positions].sort((a, b) => (a ?? 0) - (b ?? 0)));
  });

  it('turns already in the prompt are not repeated', async () => {
    const hits = found(await searchTurns(storeWith().turns('c1'), 'Harbourline pickup reference', { exclude: ['id6'] }));
    expect(hits.map((h) => h.position)).not.toContain(6);
  });

  it('an embedding model fills in when keywords come up short', async () => {
    const fake: Encoder = {
      model: 'fake-1',
      encode: (texts) => texts.map((t) => (t.includes('sister') || t.includes('family') ? [1, 0] : [0, 1])),
    };
    const store = storeWith();
    const hits = found(await searchTurns(store.turns('c1'), 'how is the family situation', { encoder: fake, store }));
    expect(hits.map((h) => h.position)).toEqual([5]);
    expect(hits[0]?.via).toBe('embedding');
    expect(store.vectors(store.turns('c1').map((t) => t.seq), 'fake-1').size).toBe(TURNS.length);
  });

  it('excerpts are cut to the matching sentences', async () => {
    const long = Array.from({ length: 40 }, (_, i) => `Filler sentence number ${i} about nothing.`).join(' ');
    const store = storeWith([['user', `${long} The deposit is EUR 2,280 due by 28 October. ${long}`]]);
    const [hit] = found(await searchTurns(store.turns('c1'), 'how much is the deposit and when is it due?'));
    expect(hit!.text).toContain('EUR 2,280');
    expect(hit!.text.split(/\s+/).length).toBeLessThanOrEqual(70);
    expect(hit!.text.startsWith('...')).toBe(true);
  });

  it('hindi turns are searchable by keywords', async () => {
    const store = storeWith([
      ['user', 'हम फरवरी में लिस्बन जा रहे हैं।'],
      ['user', 'मूवर्स ने पिकअप 19 जनवरी कर दिया है।'],
    ]);
    const hits = found(await searchTurns(store.turns('c1'), 'पिकअप कब है?'));
    expect(hits.map((h) => h.position)).toEqual([2]);
  });

  it('the turn before a hit comes along, cut to its end', async () => {
    const xs = await searchTurns(storeWith().turns('c1'), 'what was reference HL-22918 about?');
    expect(xs.map((e) => [e.position, e.via])).toEqual([[5, 'previous'], [6, 'literal']]);
    const long = storeWith([
      ['assistant', Array.from({ length: 100 }, (_, i) => `w${i}`).join(' ') + ' Has it arrived?'],
      ['user', 'Yes, the housing letter arrived this morning.'],
    ]);
    const [before, hit] = await searchTurns(long.turns('c1'), 'has the housing letter arrived?');
    expect(hit!.position).toBe(2);
    expect(before!.text.startsWith('... ')).toBe(true);
    expect(before!.text.endsWith('Has it arrived?')).toBe(true);
    expect(before!.text.split(' ').length).toBe(41);
  });

  it('irregular past forms match their base', async () => {
    const store = storeWith([['user', 'We will sell the car to Marguerite.'], ['user', 'Lunch was good.']]);
    expect(found(await searchTurns(store.turns('c1'), 'has the car been sold?')).map((h) => h.position)).toEqual([1]);
  });

  it('one common word is not enough', async () => {
    const turns: [string, string][] = Array.from({ length: 60 }, (_, i) => ['user', `The car needs tyres, note ${i}.`]);
    turns.push(['user', 'Lunch was good.']);
    expect(found(await searchTurns(storeWith(turns).turns('c1'), 'where is the car parked?'))).toEqual([]);
  });

  it('render', () => {
    const text = renderExcerpts([{ position: 6, role: 'user', text: 'moved the pickup', score: 1, timestamp: '', via: '' }]);
    expect(text.split('\n')[0]!.startsWith('### From the earlier turns')).toBe(true);
    expect(text.split('\n')[1]).toBe('- turn 6, user: "moved the pickup"');
    expect(renderExcerpts([])).toBe('');
  });

  it('render with timestamps', () => {
    const line = (timestamp: string, latest: string) =>
      renderExcerpts([{ position: 6, role: 'user', text: 'moved', score: 1, timestamp, via: '' }], latest).split('\n')[1];
    const latest = '2026-09-23T10:00:00Z';
    expect(line('2026-09-20T09:12:00+00:00', latest)).toBe(
      '- turn 6, user, 2026-09-20T09:12:00+00:00 (3 days before the latest message): "moved"');
    expect(line('2026-09-23T08:59:00Z', latest)).toBe(
      '- turn 6, user, 2026-09-23T08:59:00Z (1 hour before the latest message): "moved"');
    expect(line('2026-09-23T09:58:30Z', latest)).toBe(
      '- turn 6, user, 2026-09-23T09:58:30Z (1 minute before the latest message): "moved"');
    expect(line('2026-09-23T15:29:40+05:30', latest)).toBe(
      '- turn 6, user, 2026-09-23T15:29:40+05:30 (less than a minute before the latest message): "moved"');
    expect(line('2026-09-20 09:12', latest)).toBe('- turn 6, user, 2026-09-20 09:12: "moved"');
  });
});

describe('get_context_for_prompt with history search', () => {
  const CONV = '0b0f6a52-6f0e-4c58-9f0b-2f6f4d1f3a11';
  const BLOCK = '## Conversation so far (compacted: turns 1-6)\n[CONTEXT COMPACTION, REFERENCE ONLY]\n\n### Where things stand\n- Moving to Lisbon';
  const json = (b: unknown) => new Response(JSON.stringify(b), { status: 200, headers: { 'content-type': 'application/json' } });

  function cloud(o: { compacted?: boolean; serverTurns?: boolean } = {}) {
    const compacted = o.compacted ?? true;
    const state = { paths: [] as string[], recorded: 0 };
    const recent = [
      { role: 'assistant', content: 'Anything else on the move?', timestamp: '2026-10-19T10:00:00+00:00', message_id: 'id7' },
      { role: 'user', content: 'When is the Harbourline pickup now?', timestamp: '2026-10-19T10:01:00+00:00', message_id: 'id8' },
    ];
    const fetchImpl = (async (url: string, init?: { body?: string }) => {
      const path = new URL(String(url)).pathname;
      state.paths.push(path);
      if (path.endsWith('/messages') && init?.body === undefined) {
        const turns = o.serverTurns
          ? TURNS.map(([role, content], i) => ({ position: i + 1, role, content, message_id: `id${i + 1}`, timestamp: '' }))
          : [];
        return json({ turns, total_message_count: 8 });
      }
      if (path.endsWith('/context-for-prompt')) {
        return json({
          formatted_context: (compacted ? `${BLOCK}\n\n` : '') + 'recent...',
          available: true,
          recent_messages: recent,
          recent_message_count: 2,
          compacted_message_count: compacted ? 6 : 0,
          total_message_count: compacted ? 8 : 2,
          compaction_id: compacted ? 'k1' : null,
          context_block: compacted ? BLOCK : '',
        });
      }
      state.recorded += 1;
      return json({ message_id: `id${state.recorded}`, conversation_id: CONV, session_id: 's1', recorded_at: '2026-10-19T09:00:00+00:00' });
    }) as unknown as typeof fetch;
    return { state, fetchImpl };
  }

  const record = async (c: SynapClient) => {
    for (const [role, content] of TURNS) {
      await c.conversation.record_message({ conversation_id: CONV, user_id: 'u1', customer_id: 'c1', role, content });
    }
  };

  it('after a compaction the earlier turns are searched locally', async () => {
    const { state, fetchImpl } = cloud();
    const c = new SynapClient({ apiKey: 'k', _force_new: true, clientId: 'cli_test', instanceId: 'inst_0123456789abcdef', fetchImpl });
    await record(c);
    const response = await c.conversation.context.get_context_for_prompt({ conversation_id: CONV });
    const excerpts = response['history_excerpts'] as Array<{ text: string }>;
    expect(excerpts.some((e) => e.text.includes('HL-22918'))).toBe(true);
    const text = String(response['formatted_context']);
    expect(text.indexOf(BLOCK)).toBeLessThan(text.indexOf('### From the earlier turns'));
    expect(text).toContain(String(response['history_block']));
    expect(state.paths.some((p) => p.endsWith('/messages') && p.includes(CONV))).toBe(false);
  });

  it('a caller-supplied encoder is used', async () => {
    const { fetchImpl } = cloud();
    const calls: string[][] = [];
    const encoder = {
      model: 'fake-2',
      encode: (texts: string[]) => {
        calls.push(texts);
        return texts.map(() => [1, 0]);
      },
    };
    const c = new SynapClient({ apiKey: 'k', _force_new: true, fetchImpl, history_search_encoder: encoder });
    await record(c);
    await c.conversation.context.get_context_for_prompt({ conversation_id: CONV });
    expect(calls.length).toBeGreaterThan(0);
  });

  it('it can be turned off', async () => {
    const { fetchImpl } = cloud();
    const c = new SynapClient({ apiKey: 'k', _force_new: true, fetchImpl });
    await record(c);
    const response = await c.conversation.context.get_context_for_prompt({ conversation_id: CONV, search_history: false });
    expect(response['history_excerpts']).toBeUndefined();
    expect(String(response['formatted_context'])).not.toContain('From the earlier turns');
  });

  it('without a compaction nothing is added', async () => {
    const { fetchImpl } = cloud({ compacted: false });
    const c = new SynapClient({ apiKey: 'k', _force_new: true, fetchImpl });
    const response = await c.conversation.context.get_context_for_prompt({ conversation_id: CONV });
    expect(response['history_excerpts']).toBeUndefined();
  });

  it('turns written elsewhere are fetched in the background', async () => {
    const { state, fetchImpl } = cloud({ serverTurns: true });
    const c = new SynapClient({ apiKey: 'k', _force_new: true, fetchImpl });
    const first = await c.conversation.context.get_context_for_prompt({ conversation_id: CONV });
    expect(first['history_excerpts']).toBeUndefined(); // nothing local yet
    await new Promise((r) => setTimeout(r, 10));
    expect(state.paths.some((p) => p.endsWith(`/${CONV}/messages`))).toBe(true);
    const second = await c.conversation.context.get_context_for_prompt({ conversation_id: CONV });
    const excerpts = second['history_excerpts'] as Array<{ position: number }>;
    expect(excerpts.map((e) => e.position)).toContain(6);
  });
});

describe('tool result archive (decision B, 20 Oct): searchable, never server turns', () => {
  it('an archived tool result is searchable and not a server turn', async () => {
    const store = storeWith();
    const before = store.count('c1');
    store.add('c1', 'tool', toolResultText({ order: 'ORD-55120', status: 'refunded' }, 'get_order'));
    expect(store.count('c1')).toBe(before);
    const hits = found(await searchTurns(store.turns('c1'), 'what happened to order ORD-55120?'));
    expect(hits[0]!.role).toBe('tool');
    expect(hits[0]!.text).toBe('[tool result: get_order] {"order":"ORD-55120","status":"refunded"}');
  });

  it('a long result is cut', () => {
    const text = toolResultText('x'.repeat(20_000), 'read_file');
    expect(text.startsWith('[tool result: read_file] xxx')).toBe(true);
    expect(text.endsWith(' ...[cut]')).toBe(true);
    expect(text.length).toBeLessThan(16_100);
  });
});

// The same text and passages as the Python test: the two splitters must agree.
const LONG_TURN = "Here is budget v3 from Keiko, with the storage line she asked about.\n\n| Vendor | Line | Qty | Rate | Amount |\n|---|---|---|---|---|\n| Corvane | Storage | 2,100 pallets | $18.50 | $388,500 |\n| Corvane | Returns | 9,000 returns | $2.75 | $24,750 |\n| TOTAL FY27 | | | | $2,181,418 |\n\nThe storage row uses the proposal rate. The agreed rate since Renata's first reply is $17.00 per pallet, so this overstates storage. The contingency on top of it is also a little high. Everything else in the model matches the signed schedule, including receiving, kitting and the peak surcharge for November and December. I would fix the storage row before Martin sees it, and then re-run the headroom against the $2.25M line.\n\n```\n2026-10-19 07:02:41 ERROR pod ledgerhook-api port 9187 refused\n2026-10-19 07:03:02 INFO retry 2 of 6\n```\n";
const LONG_TURN_PASSAGES = [
  "Here is budget v3 from Keiko, with the storage line she asked about.",
  "| Vendor | Line | Qty | Rate | Amount | | Corvane | Storage | 2,100 pallets | $18.50 | $388,500 |",
  "| Vendor | Line | Qty | Rate | Amount | | Corvane | Returns | 9,000 returns | $2.75 | $24,750 |",
  "| Vendor | Line | Qty | Rate | Amount | | TOTAL FY27 | | | | $2,181,418 |",
  "The storage row uses the proposal rate. The agreed rate since Renata's first reply is $17.00 per pallet, so this overstates storage. The contingency on top of it is also a little high. Everything else in the model matches the signed schedule, including receiving, kitting and the peak surcharge for November and December.",
  "I would fix the storage row before Martin sees it, and then re-run the headroom against the $2.25M line.",
  "2026-10-19 07:02:41 ERROR pod ledgerhook-api port 9187 refused 2026-10-19 07:03:02 INFO retry 2 of 6"
];

describe('passages (20 Oct, behind the contract switch)', () => {
  it('a long turn splits into rows with their header, sentences and log lines', () => {
    expect(splitPassages(LONG_TURN, 60)).toEqual(LONG_TURN_PASSAGES);
    expect(splitPassages('A short turn.', 60)).toEqual(['A short turn.']);
  });

  it('the matching row is what is shown', async () => {
    const params = HISTORY_SEARCH as unknown as Record<string, unknown>;
    const saved = { passages: params['passages'], passages_per_turn: params['passages_per_turn'] };
    try {
      const store = storeWith([['user', LONG_TURN], ['assistant', 'Noted.']]);
      Object.assign(params, { passages: true, passages_per_turn: 1 });
      let [hit] = found(await searchTurns(store.turns('c1'), 'how much do Corvane returns cost each?'));
      expect(hit!.text).toBe('... ' + LONG_TURN_PASSAGES[2] + ' ...');
      [hit] = found(await searchTurns(store.turns('c1'), 'port 9187 refused?'));
      expect(hit!.text).toBe('... ' + LONG_TURN_PASSAGES[LONG_TURN_PASSAGES.length - 1]);
      Object.assign(params, { passages_per_turn: 2 });
      [hit] = found(await searchTurns(store.turns('c1'), 'why is the storage row too high?'));
      expect(hit!.text).toBe('... ' + LONG_TURN_PASSAGES[4] + ' ... ' + LONG_TURN_PASSAGES[5] + ' ...');
    } finally {
      Object.assign(params, saved);
    }
  });
});

// The same turns and answers as the Python test (20 Oct, documents-case fixes).
const KIND_TURNS: [string, string][] = [
  ['user', 'The first quote from Harbourline was CAD 11,200 for a shared container.'],
  ['assistant', 'Noted the Harbourline quote.'],
  ['user', 'Harbourline updated their quote to CAD 12,050 after the survey.'],
  ['user', "Hi Nadia, thanks for the call today. Renata here again with the updated numbers you asked for. We can move the pick fee to $3.55 if the minimum stays where it is, and we can look at the storage rate again next week once our pricing team has reviewed the volumes you sent. I'll send the revised schedule by Friday.\n\nBest,\nRenata\n\nRenata Okafor\nSenior Account Executive, West Region\nDirect: (775) 555-0148 | Mobile: (775) 555-0193"],
];

describe('question shape (20 Oct)', () => {
  const order = async (q: string) =>
    found(await searchTurns(storeWith(KIND_TURNS).turns('c1'), q)).sort((a, b) => b.score - a.score).map((h) => h.position);

  it('a question about how it started gets no recency prior', async () => {
    expect((await order('what was the first quote from Harbourline?'))[0]).toBe(1);
    expect((await order('what is the Harbourline quote now?'))[0]).toBe(3);
  });

  it('a question for a number shows the line with the number', async () => {
    const hits = found(await searchTurns(storeWith(KIND_TURNS).turns('c1'), "what's the best number to reach Renata?"));
    expect(hits.map((h) => h.text)).toEqual(["... Best, Renata ... Renata Okafor Senior Account Executive, West Region Direct: (775) 555-0148 | Mobile: (775) 555-0193"]);
  });
});
