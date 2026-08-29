/**
 * The SDK refuses a customer_id on a B2C instance, at the call site.
 *
 * The server rejects independently. This layer exists because the old failure
 * was silent: sending a customer_id on a B2C instance produced no exception
 * anywhere, the write was filed under the customer, the read asked for the
 * user, and both returned success. One client ran 4,634 consecutive empty
 * fetches over seven days with nothing to look at.
 */
import { describe, expect, it } from 'vitest';
import { B2C_ISOLATION, checkCustomerId } from '../scoping.js';
import { InvalidInputError } from '../errors.js';

describe('B2C single-identifier contract', () => {
  it('rejects a customer_id on a B2C instance', () => {
    expect(() => checkCustomerId(B2C_ISOLATION, 'acme', 'probe', 'inst_1'))
      .toThrow(InvalidInputError);
  });

  it('says what to send instead, and echoes the rejected value', () => {
    let msg = '';
    try {
      checkCustomerId(B2C_ISOLATION, 'acme', 'probe', 'inst_1');
    } catch (e) {
      msg = (e as Error).message;
    }
    expect(msg).toContain('Send user_id only');
    // Without the offending value in the message the caller cannot trace which
    // call site sent it.
    expect(msg).toContain('acme');
    expect(msg).toContain('inst_1');
  });

  it('accepts the correct B2C shape', () => {
    expect(() => checkCustomerId(B2C_ISOLATION, undefined, 'probe')).not.toThrow();
    expect(() => checkCustomerId(B2C_ISOLATION, null, 'probe')).not.toThrow();
    expect(() => checkCustomerId(B2C_ISOLATION, '', 'probe')).not.toThrow();
  });

  it('never refuses on B2B, where customer_id is REQUIRED', () => {
    expect(() => checkCustomerId('strict', 'acme', 'probe')).not.toThrow();
    expect(() => checkCustomerId('maca_suggests', 'acme', 'probe')).not.toThrow();
  });

  it('never refuses when the server did not report a mode', () => {
    // undefined is what every server older than the whoami field returns. A new
    // SDK against an old server must refuse nothing, or it breaks B2B callers
    // everywhere it cannot confirm the mode.
    expect(() => checkCustomerId(undefined, 'acme', 'probe')).not.toThrow();
  });
});
