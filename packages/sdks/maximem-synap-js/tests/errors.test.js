import { describe, it, expect } from 'vitest';

import {
  SynapError,
  SynapTransientError,
  SynapPermanentError,
  NetworkTimeoutError,
  RateLimitError,
  ServiceUnavailableError,
  AgentUnavailableError,
  InvalidInputError,
  InvalidInstanceIdError,
  InvalidConversationIdError,
  AuthenticationError,
  InsufficientCreditsError,
  ContextNotFoundError,
  SessionExpiredError,
  ListeningAlreadyActiveError,
  ListeningNotActiveError,
  createSynapError,
} from '../src/errors.js';

describe('error hierarchy', () => {
  it('SynapError extends Error', () => {
    const err = new SynapError('boom');
    expect(err).toBeInstanceOf(Error);
    expect(err).toBeInstanceOf(SynapError);
    expect(err.message).toBe('boom');
    expect(err.name).toBe('SynapError');
    expect(err.correlationId).toBeNull();
  });

  it('SynapError stores correlationId', () => {
    const err = new SynapError('msg', 'corr-123');
    expect(err.correlationId).toBe('corr-123');
  });

  it('SynapTransientError extends SynapError', () => {
    const err = new SynapTransientError('transient');
    expect(err).toBeInstanceOf(SynapError);
    expect(err).toBeInstanceOf(SynapTransientError);
    expect(err.name).toBe('SynapTransientError');
  });

  it('SynapPermanentError extends SynapError', () => {
    const err = new SynapPermanentError('permanent');
    expect(err).toBeInstanceOf(SynapError);
    expect(err).toBeInstanceOf(SynapPermanentError);
    expect(err.name).toBe('SynapPermanentError');
  });

  it('NetworkTimeoutError extends SynapTransientError', () => {
    const err = new NetworkTimeoutError('timeout');
    expect(err).toBeInstanceOf(SynapTransientError);
    expect(err.name).toBe('NetworkTimeoutError');
  });

  it('RateLimitError stores retryAfterSeconds', () => {
    const err = new RateLimitError('rate limited', 30, 'corr-456');
    expect(err).toBeInstanceOf(SynapTransientError);
    expect(err.name).toBe('RateLimitError');
    expect(err.retryAfterSeconds).toBe(30);
    expect(err.correlationId).toBe('corr-456');
  });

  it('RateLimitError retryAfterSeconds defaults to null', () => {
    const err = new RateLimitError('rate limited');
    expect(err.retryAfterSeconds).toBeNull();
  });

  it('ServiceUnavailableError extends SynapTransientError', () => {
    expect(new ServiceUnavailableError('down')).toBeInstanceOf(SynapTransientError);
  });

  it('AgentUnavailableError extends SynapTransientError', () => {
    expect(new AgentUnavailableError('down')).toBeInstanceOf(SynapTransientError);
  });

  it('InvalidInputError extends SynapPermanentError', () => {
    expect(new InvalidInputError('bad')).toBeInstanceOf(SynapPermanentError);
  });

  it('InvalidInstanceIdError extends InvalidInputError', () => {
    const err = new InvalidInstanceIdError('bad id');
    expect(err).toBeInstanceOf(InvalidInputError);
    expect(err.name).toBe('InvalidInstanceIdError');
  });

  it('InvalidConversationIdError extends InvalidInputError', () => {
    const err = new InvalidConversationIdError('bad conv');
    expect(err).toBeInstanceOf(InvalidInputError);
    expect(err.name).toBe('InvalidConversationIdError');
  });

  it('AuthenticationError extends SynapPermanentError', () => {
    expect(new AuthenticationError('unauth')).toBeInstanceOf(SynapPermanentError);
  });

  it('InsufficientCreditsError parses structured details', () => {
    const err = new InsufficientCreditsError('no credits', {
      balance_credits: 5,
      minimum_required_credits: 10,
      recovery_url: 'https://example.com/recover',
      redeem_url: 'https://example.com/redeem',
    });
    expect(err).toBeInstanceOf(SynapPermanentError);
    expect(err.name).toBe('InsufficientCreditsError');
    expect(err.balanceCredits).toBe(5);
    expect(err.minimumRequiredCredits).toBe(10);
    expect(err.recoveryUrl).toBe('https://example.com/recover');
    expect(err.redeemUrl).toBe('https://example.com/redeem');
  });

  it('InsufficientCreditsError handles missing details gracefully', () => {
    const err = new InsufficientCreditsError('no credits');
    expect(err.balanceCredits).toBeNull();
    expect(err.minimumRequiredCredits).toBeNull();
    expect(err.recoveryUrl).toBeNull();
    expect(err.redeemUrl).toBeNull();
  });

  it('ContextNotFoundError extends SynapPermanentError', () => {
    expect(new ContextNotFoundError('not found')).toBeInstanceOf(SynapPermanentError);
  });

  it('SessionExpiredError extends SynapPermanentError', () => {
    expect(new SessionExpiredError('expired')).toBeInstanceOf(SynapPermanentError);
  });

  it('ListeningAlreadyActiveError extends SynapPermanentError', () => {
    expect(new ListeningAlreadyActiveError('active')).toBeInstanceOf(SynapPermanentError);
  });

  it('ListeningNotActiveError extends SynapPermanentError', () => {
    expect(new ListeningNotActiveError('not active')).toBeInstanceOf(SynapPermanentError);
  });
});

describe('createSynapError', () => {
  it('returns the correct typed class for known error types', () => {
    const cases = [
      ['NetworkTimeoutError', NetworkTimeoutError],
      ['RateLimitError', RateLimitError],
      ['ServiceUnavailableError', ServiceUnavailableError],
      ['AgentUnavailableError', AgentUnavailableError],
      ['InvalidInputError', InvalidInputError],
      ['InvalidInstanceIdError', InvalidInstanceIdError],
      ['InvalidConversationIdError', InvalidConversationIdError],
      ['AuthenticationError', AuthenticationError],
      ['InsufficientCreditsError', InsufficientCreditsError],
      ['ContextNotFoundError', ContextNotFoundError],
      ['SessionExpiredError', SessionExpiredError],
      ['ListeningAlreadyActiveError', ListeningAlreadyActiveError],
      ['ListeningNotActiveError', ListeningNotActiveError],
      ['SynapError', SynapError],
      ['SynapTransientError', SynapTransientError],
      ['SynapPermanentError', SynapPermanentError],
    ];

    for (const [errorType, ExpectedClass] of cases) {
      const err = createSynapError('test message', errorType);
      expect(err).toBeInstanceOf(ExpectedClass);
      expect(err.message).toBe('test message');
    }
  });

  it('falls back to SynapError for unknown error types', () => {
    const err = createSynapError('unknown', 'SomeUnknownError');
    expect(err).toBeInstanceOf(SynapError);
    expect(err.name).toBe('SynapError');
  });

  it('falls back to SynapError when no errorType provided', () => {
    const err = createSynapError('oops');
    expect(err).toBeInstanceOf(SynapError);
  });
});
