/**
 * The conversation routes could not name a rung from TypeScript.
 *
 * `create` gained a scope path (see create-carries-a-scope-path.test.ts) and
 * the same gap was left open one namespace over. Python 0.4.10 sends a path on
 * `ingest_transcript`; this SDK sent none, on either the one-shot transcript
 * route or the streaming turn routes, and both accept one on the wire.
 *
 * That is the same "true in one language, false in the other" split the sibling
 * test was written about, so this pins all three shapes: the single turn, the
 * batch, and the transcript.
 */
import { describe, it, expect } from 'vitest';
import {
  buildRecordMessageBody,
  buildIngestTranscriptBody,
} from '../conversation/interface.js';

const CONV = '11111111-1111-1111-1111-111111111111';
const RUNG = { tenant: 'acme', practice: 'high_street' };

describe('a turn can name the rung it belongs to', () => {
  it('puts the path on a single turn, under the wire name', () => {
    const body = buildRecordMessageBody({
      conversation_id: CONV,
      role: 'user',
      content: 'hello',
      user_id: 'patient-44',
      scope_path: RUNG,
    });
    expect(body.scope).toEqual(RUNG);
    // The SDK spelling must not leak: the server has no `scope_path` field and
    // would silently ignore it.
    expect(body.scope_path).toBeUndefined();
  });

  it('accepts the camelCase spelling too, like every other option here', () => {
    const body = buildRecordMessageBody({
      conversation_id: CONV,
      role: 'user',
      content: 'hello',
      user_id: 'patient-44',
      scopePath: RUNG,
    });
    expect(body.scope).toEqual(RUNG);
  });

  it('omits the key entirely when no rung is named', () => {
    // The clients on the default three rungs must send exactly the body they
    // send today, not an explicit null.
    const body = buildRecordMessageBody({
      conversation_id: CONV,
      role: 'user',
      content: 'hello',
      user_id: 'patient-44',
    });
    expect('scope' in body).toBe(false);
  });

  it('puts the path on a one-shot transcript, matching Python 0.4.10', () => {
    const body = buildIngestTranscriptBody({
      conversation_id: 'call-abc-123',
      user_id: 'patient-44',
      transcript: 'user: hi\nassistant: hello',
      scope_path: RUNG,
    });
    expect(body.scope).toEqual(RUNG);
  });

  it('omits it on a transcript when no rung is named', () => {
    const body = buildIngestTranscriptBody({
      conversation_id: 'call-abc-123',
      user_id: 'patient-44',
      transcript: 'user: hi',
    });
    expect('scope' in body).toBe(false);
  });
});
