import { describe, it, expect } from 'vitest';
import {
  SynapError, SynapTransientError, SynapPermanentError,
  RateLimitError, NetworkTimeoutError, InsufficientCreditsError,
  InvalidInputError, InvalidInstanceIdError, ConflictError,
  TranscriptConflictError, AuthenticationError, isSynapError,
  SDKError, TransientError, PermanentError, ConnectionError,
} from '../errors.js';

describe('error taxonomy', () => {
  it('mirrors the Python hierarchy', () => {
    expect(new RateLimitError('x')).toBeInstanceOf(SynapTransientError);
    expect(new RateLimitError('x')).toBeInstanceOf(SynapError);
    expect(new InvalidInstanceIdError('x')).toBeInstanceOf(InvalidInputError);
    expect(new TranscriptConflictError('x')).toBeInstanceOf(ConflictError);
    expect(new AuthenticationError('x')).toBeInstanceOf(SynapPermanentError);
  });

  it('does not treat a permanent error as transient', () => {
    expect(new AuthenticationError('x')).not.toBeInstanceOf(SynapTransientError);
    expect(new RateLimitError('x')).not.toBeInstanceOf(SynapPermanentError);
    // Credits are permanent on purpose: retrying without a top-up never works.
    expect(new InsufficientCreditsError('x')).not.toBeInstanceOf(SynapTransientError);
    expect(new InsufficientCreditsError('x').transient).toBe(false);
  });

  it('keeps name, message, stack and cause intact', () => {
    const cause = new Error('root');
    const e = new NetworkTimeoutError('timed out', { correlationId: 'abc', cause });
    expect(e.name).toBe('NetworkTimeoutError');
    expect(e.message).toBe('timed out');
    expect(e.correlationId).toBe('abc');
    expect(e.cause).toBe(cause);
    expect(e.stack).toContain('NetworkTimeoutError');
  });

  it('carries retry metadata', () => {
    expect(new RateLimitError('x', { retryAfterSeconds: 30 }).retryAfterSeconds).toBe(30);
    expect(new RateLimitError('x').retryAfterSeconds).toBeNull();
    const c = new InsufficientCreditsError('x', { balanceCredits: 4, requiredCredits: 10 });
    expect(c.balanceCredits).toBe(4);
    expect(c.requiredCredits).toBe(10);
  });

  it('exposes the Python back-compat aliases', () => {
    expect(SDKError).toBe(SynapError);
    expect(TransientError).toBe(SynapTransientError);
    expect(PermanentError).toBe(SynapPermanentError);
    expect(ConnectionError).toBe(NetworkTimeoutError);
  });

  // ── The dual-package hazard (gotcha G-G) ────────────────────────────────────
  // A consumer can load both the ESM and CJS builds, getting two independent
  // copies of every class. Simulate that with a structurally identical error
  // built from a *different* class object.
  describe('survives the dual ESM/CJS package hazard', () => {
    class OtherCopyRateLimitError extends Error {
      code = 'rate_limit';
      constructor() {
        super('from the other copy');
        Object.defineProperty(this, Symbol.for('@maximem/synap-js-sdk.SynapError'), {
          value: true, enumerable: false,
        });
      }
    }

    it('recognises an error from the other copy', () => {
      const foreign = new OtherCopyRateLimitError();
      expect(foreign).toBeInstanceOf(SynapError);
      expect(foreign).toBeInstanceOf(SynapTransientError);
      expect(foreign).toBeInstanceOf(RateLimitError);
      expect(isSynapError(foreign)).toBe(true);
    });

    it('still narrows correctly across copies', () => {
      const foreign = new OtherCopyRateLimitError();
      expect(foreign).not.toBeInstanceOf(AuthenticationError);
      expect(foreign).not.toBeInstanceOf(SynapPermanentError);
    });

    it('rejects unrelated values', () => {
      expect(new Error('plain')).not.toBeInstanceOf(SynapError);
      expect(isSynapError(new Error('plain'))).toBe(false);
      expect(isSynapError(null)).toBe(false);
      expect(isSynapError('rate_limit')).toBe(false);
      expect(isSynapError({ code: 'rate_limit' })).toBe(false);
    });
  });

  it('gives every error a stable string code', () => {
    const codes = [
      new SynapError('x'), new SynapTransientError('x'), new SynapPermanentError('x'),
      new NetworkTimeoutError('x'), new RateLimitError('x'), new InsufficientCreditsError('x'),
      new InvalidInputError('x'), new InvalidInstanceIdError('x'), new ConflictError('x'),
      new TranscriptConflictError('x'), new AuthenticationError('x'),
    ].map((e) => e.code);
    expect(new Set(codes).size).toBe(codes.length);
    expect(codes.every((c) => typeof c === 'string' && c.length > 0)).toBe(true);
  });
});
