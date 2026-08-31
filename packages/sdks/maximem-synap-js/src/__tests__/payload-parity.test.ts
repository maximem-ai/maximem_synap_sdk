import { describe, it, expect } from 'vitest';
import { readFileSync, existsSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { buildFetchRequest } from '../context/fetch.js';
import { buildCreateBody, DOCUMENT_TYPES, INGEST_MODES } from '../memories/interface.js';
import type { FetchOptions } from '../context/types.js';
import type { CreateMemoryOptions } from '../memories/interface.js';

const here = path.dirname(fileURLToPath(import.meta.url));
const corpusPath = path.resolve(here, '../../../CONTRACT/conformance/request_payloads.json');
const memoryCorpusPath = path.resolve(here, '../../../CONTRACT/conformance/memory_payloads.json');

/**
 * The wire contract, checked against payloads captured from the Python SDK.
 *
 * A unit test that only checks this SDK against itself cannot catch a
 * divergence from Python, and a divergence from Python is what actually breaks:
 * the server is tuned against Python's payload.
 */
describe('request payload parity with the Python SDK', () => {
  if (!existsSync(corpusPath)) return;
  const golden = JSON.parse(readFileSync(corpusPath, 'utf8')) as {
    scope: string;
    cases: { args: Record<string, unknown>; payload: Record<string, unknown> }[];
  };

  it.each(golden.cases.map((c, i) => ({ i, label: JSON.stringify(c.args) })))(
    'case[$i] $label',
    ({ i }) => {
      const c = golden.cases[i]!;
      const built = buildFetchRequest('user', c.args as FetchOptions).body;
      // Compare key sets too: an extra key is as much a divergence as a wrong
      // value, and toEqual alone would not flag a missing one on either side.
      expect(Object.keys(built).sort()).toEqual(Object.keys(c.payload).sort());
      expect(built).toEqual(c.payload);
    },
  );
});

/**
 * Memory ingestion bodies, generated from Python's own pydantic models.
 *
 * This is the highest-stakes payload in the SDK: `document_type` selects the
 * extraction path, so a body that merely looks tidier can silently store far
 * less from identical input.
 */
describe('memory payload parity with the Python SDK', () => {
  if (!existsSync(memoryCorpusPath)) return;
  const golden = JSON.parse(readFileSync(memoryCorpusPath, 'utf8')) as {
    create: { args: Record<string, unknown>; body: Record<string, unknown> }[];
    update: { args: Record<string, unknown>; body: Record<string, unknown> }[];
    document_types: string[];
    ingest_modes: string[];
    merge_strategies: string[];
  };

  it.each(golden.create.map((c, i) => ({ i, label: JSON.stringify(c.args) })))(
    'create[$i] $label',
    ({ i }) => {
      const c = golden.create[i]!;
      const built = buildCreateBody(c.args as unknown as CreateMemoryOptions);
      expect(Object.keys(built).sort()).toEqual(Object.keys(c.body).sort());
      expect(built).toEqual(c.body);
    },
  );

  it('mirrors the Python enums exactly', () => {
    // A value Python accepts and JS rejects (or vice versa) is a divergence
    // that only shows up at a customer's runtime.
    expect([...DOCUMENT_TYPES]).toEqual(golden.document_types);
    expect([...INGEST_MODES]).toEqual(golden.ingest_modes);
  });

  it('defaults document_type and mode the way Python does', () => {
    const b = buildCreateBody({ document: 'x' });
    expect(b['document_type']).toBe('ai-chat-conversation');
    expect(b['mode']).toBe('long-range');
  });
});

describe('conversation-summary mode (user scope)', () => {
  /**
   * These three parameters existed in Python and had no JS equivalent at all:
   * neither `FetchOptions` nor the request builder knew them, so the unified
   * fetch passed them in and they were silently dropped. Summary mode was
   * unreachable from this SDK.
   */
  it('sends nothing extra in the default mode', () => {
    const { body } = buildFetchRequest('user', { user_id: 'u1' });
    for (const key of ['context_mode', 'include_profile', 'last_n_conversations', 'scope']) {
      expect(body, `${key} must not appear on an ordinary fetch`).not.toHaveProperty(key);
    }
  });

  it('sends all three only in summary mode', () => {
    const { body } = buildFetchRequest('user', {
      user_id: 'u1', context_mode: 'conversation-summary',
    });
    expect(body['context_mode']).toBe('conversation-summary');
    // Python's defaults, applied here rather than left to the server.
    expect(body['include_profile']).toBe(true);
    expect(body['last_n_conversations']).toBe(1);
  });

  it('carries explicit summary options through', () => {
    const { body } = buildFetchRequest('user', {
      user_id: 'u1', context_mode: 'conversation-summary',
      include_profile: false, last_n_conversations: 7,
    });
    expect(body['include_profile']).toBe(false);
    expect(body['last_n_conversations']).toBe(7);
  });

  it('accepts the camelCase spellings too', () => {
    const { body } = buildFetchRequest('user', {
      user_id: 'u1', contextMode: 'conversation-summary', lastNConversations: 3,
    });
    expect(body['context_mode']).toBe('conversation-summary');
    expect(body['last_n_conversations']).toBe(3);
  });

  it('rejects an unknown context_mode locally', () => {
    expect(() => buildFetchRequest('user', { user_id: 'u1', context_mode: 'nope' }))
      .toThrow(/Invalid context_mode/);
  });

  it('enforces Python\'s 0 to 20 bound, and integers only', () => {
    for (const bad of [-1, 21, 1.5]) {
      expect(() => buildFetchRequest('user', {
        user_id: 'u1', context_mode: 'conversation-summary', last_n_conversations: bad,
      }), String(bad)).toThrow(/last_n_conversations/);
    }
    expect(() => buildFetchRequest('user', {
      user_id: 'u1', context_mode: 'conversation-summary', last_n_conversations: 0,
    })).not.toThrow();
  });

  it('sends scope_path as `scope`, only when supplied', () => {
    const { body } = buildFetchRequest('user', {
      user_id: 'u1', scope_path: { tenant: 't1' },
    });
    expect(body['scope']).toEqual({ tenant: 't1' });
  });
});
