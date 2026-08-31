/**
 * The JS SDK could not name the level a write belongs to.
 *
 * The Python SDK gained a `scope` path on reads in 0.4.6 and on writes in
 * 0.4.8. This one had neither, so a JS client on an account whose ladder has a
 * level between customer and user could define that level, see it, accept it,
 * and never file a record on it. The write was refused by the server with a
 * message asking for a path this SDK had no way to send.
 *
 * Worth stating because it nearly shipped that way: the Python SDK was updated
 * and this one was not, which would have made "custom depth works" true in one
 * language and false in the other, with nothing to say which.
 */
import { describe, it, expect } from 'vitest';
import { buildCreateBody } from '../memories/interface.js';

describe('a write can name the level it belongs to', () => {
  it('puts the path in the body', () => {
    const body = buildCreateBody({
      document: 'patient notes',
      user_id: 'patient-44',
      customer_id: 'bright-smile',
      scope: { customer: 'bright-smile', practice: 'high-street', user: 'patient-44' },
    });
    expect(body.scope).toEqual({
      customer: 'bright-smile',
      practice: 'high-street',
      user: 'patient-44',
    });
  });

  it('sends null when omitted, never an empty object', () => {
    // An empty path is not absence. The server treats one as an error, because
    // a caller who asks to be narrowed and is ignored gets widened instead.
    const body = buildCreateBody({
      document: 'hello',
      user_id: 'dana',
      customer_id: 'acme',
    });
    expect(body.scope).toBeNull();
    expect(body.scope).not.toEqual({});
  });

  it('leaves every other field exactly as it was', () => {
    // The body is key-ordered to match the Python model so a captured-body diff
    // stays readable. Adding a field must not disturb the rest.
    const body = buildCreateBody({
      document: 'hello',
      user_id: 'dana',
      customer_id: 'acme',
      scope: { customer: 'acme', team: 'payments' },
    });
    expect(body.document).toBe('hello');
    expect(body.user_id).toBe('dana');
    expect(body.customer_id).toBe('acme');
    expect(body.metadata).toEqual({});
  });
});
