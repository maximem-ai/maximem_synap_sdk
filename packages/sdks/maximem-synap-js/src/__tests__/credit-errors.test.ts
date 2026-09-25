import { describe, it, expect } from 'vitest';
import { HttpTransport } from '../transport/http.js';
import {
  GrpcStreamClient,
  creditErrorFromRpc,
  isCreditStop,
} from '../grpc/stream-client.js';
import { AnticipationCache } from '../context/anticipation-cache.js';
import { SYNAP_PROTO_DESCRIPTOR } from '../grpc/descriptor.js';
import { InsufficientCreditsError, RateLimitError, SynapPermanentError } from '../errors.js';

/**
 * The SDK says why a request was refused for credits.
 *
 * Mirrors `synap/tests/sdk/test_credit_errors.py` case for case. Three live
 * refusals, one fix each: `overages_disabled` (402, allow overages or add
 * credits), `trial_limit_reached` (429, upgrade or wait for the cycle) and
 * `subscription_inactive` (429, Plan & billing). On gRPC all three arrive as
 * the same RESOURCE_EXHAUSTED abort with the reason in trailing metadata.
 */

const CREDS = { apiKey: 'k', clientId: 'c', instanceId: 'inst_0123456789abcdef' };

function transport(fetchImpl: typeof fetch) {
  return new HttpTransport({
    credentials: CREDS,
    fetchImpl,
    retryPolicy: { maxAttempts: 1, backoffBase: 0, backoffMax: 0, backoffJitter: false },
  });
}

const respond = (status: number, body: unknown, headers: Record<string, string> = {}) =>
  new Response(JSON.stringify(body), {
    status,
    headers: { 'content-type': 'application/json', ...headers },
  });

/** The gate's 402 body, verbatim from `_insufficient_body()`. */
const OVERAGES_DISABLED_BODY = {
  error: 'insufficient_credits',
  reason: 'overages_disabled',
  message:
    'Your credit balance is spent and this account does not allow usage past zero. ' +
    'Allow overages or add credits to continue.',
  balance_credits: 0,
  minimum_required_credits: 2,
  recovery_url: '/v1/credits/balance',
  redeem_url: '/credits/redeem',
  manage_url: '/dashboard/billing',
  support_contact: 'support@maximem.ai',
  request_id: 'req-402',
};

const TRIAL_BODY = {
  error: 'trial_limit_reached',
  message: 'Your trial credits are spent.',
  balance_credits: 0,
  minimum_required_credits: 2,
  upgrade_url: '/dashboard/billing',
  request_id: 'req-429',
};

const LAPSED_PLAN_BODY = {
  error: 'trial_limit_reached',
  reason: 'subscription_inactive',
  message: 'Your subscription is not active.',
  balance_credits: 0,
  minimum_required_credits: 2,
  upgrade_url: '/dashboard/billing',
  manage_url: '/dashboard/billing',
  request_id: 'req-429b',
};

async function raised(status: number, body: unknown, headers: Record<string, string> = {}) {
  const t = transport((async () => respond(status, body, headers)) as unknown as typeof fetch);
  return await t.request('whoami').then(
    () => { throw new Error('expected the request to be refused'); },
    (e: unknown) => e,
  );
}

describe('HTTP credit refusals', () => {
  it('reads reason and manage_url from a 402', async () => {
    const error = await raised(402, { detail: OVERAGES_DISABLED_BODY });

    expect(error).toBeInstanceOf(InsufficientCreditsError);
    const credit = error as InsufficientCreditsError;
    expect(credit.reason).toBe('overages_disabled');
    expect(credit.manageUrl).toBe('/dashboard/billing');
    // The gate raises an HTTPException, so its body arrives nested in `detail`.
    // Reading only the top level left every one of these null in production.
    expect(credit.balanceCredits).toBe(0);
    expect(credit.requiredCredits).toBe(2);
    expect(credit.recoveryUrl).toBe('/v1/credits/balance');
    expect(credit.redeemUrl).toBe('/credits/redeem');
    expect(credit.message).toContain('Allow overages or add credits');
  });

  it('reads the same body at the top level', async () => {
    // What the burn service returns, without the `detail` wrapper.
    const error = (await raised(402, OVERAGES_DISABLED_BODY)) as InsufficientCreditsError;
    expect(error.reason).toBe('overages_disabled');
    expect(error.manageUrl).toBe('/dashboard/billing');
    expect(error.balanceCredits).toBe(0);
  });

  it('still parses a 402 from a server too old to send the new fields', async () => {
    const error = (await raised(402, {
      detail: {
        balance_credits: 1,
        minimum_required_credits: 2,
        recovery_url: '/v1/credits/balance',
        redeem_url: '/credits/redeem',
      },
    })) as InsufficientCreditsError;

    expect(error.reason).toBeNull();
    expect(error.manageUrl).toBeNull();
    expect(error.balanceCredits).toBe(1);
    expect(error.requiredCredits).toBe(2);
    expect(error.recoveryUrl).toBe('/v1/credits/balance');
  });

  it('names the Trial cap on a 429', async () => {
    const error = (await raised(429, { detail: TRIAL_BODY }, { 'retry-after': '60' })) as RateLimitError;

    expect(error).toBeInstanceOf(RateLimitError);
    expect(error.reason).toBe('trial_limit_reached');
    expect(error.upgradeUrl).toBe('/dashboard/billing');
    expect(error.manageUrl).toBeNull();
    expect(error.retryAfterSeconds).toBe(60);
  });

  it('names a lapsed subscription on a 429', async () => {
    const error = (await raised(429, { detail: LAPSED_PLAN_BODY })) as RateLimitError;

    expect(error.reason).toBe('subscription_inactive');
    expect(error.manageUrl).toBe('/dashboard/billing');
    expect(error.upgradeUrl).toBe('/dashboard/billing');
  });

  it('leaves reason null on an ordinary rate limit', async () => {
    const error = (await raised(429, { detail: 'Too many requests' }, { 'retry-after': '5' })) as RateLimitError;

    expect(error.reason).toBeNull();
    expect(error.upgradeUrl).toBeNull();
    expect(error.manageUrl).toBeNull();
    expect(error.retryAfterSeconds).toBe(5);
  });

  it('keeps the existing constructor shape', () => {
    const credit = new InsufficientCreditsError('spent', { balanceCredits: 1, requiredCredits: 2 });
    expect(credit.balanceCredits).toBe(1);
    expect(credit.reason).toBeNull();
    expect(credit.manageUrl).toBeNull();

    const rate = new RateLimitError('slow down', { retryAfterSeconds: 30 });
    expect(rate.retryAfterSeconds).toBe(30);
    expect(rate.reason).toBeNull();
  });
});

// ─── gRPC ─────────────────────────────────────────────────────────────────────

const RESOURCE_EXHAUSTED = 8;

function abort(reason: string | null, extra: Record<string, string> = {}) {
  const metadata: Record<string, string> = {
    'credit-balance': '0.0',
    'credit-minimum-required': '2.0',
    'credit-recovery-url': '/v1/credits/balance',
    'credit-redeem-url': '/credits/redeem',
    'request-id': 'req-grpc',
    ...extra,
  };
  if (reason !== null) metadata['credit-reason'] = reason;
  return {
    code: RESOURCE_EXHAUSTED,
    details: 'insufficient_credits',
    metadata: { get: (k: string) => (metadata[k] === undefined ? [] : [metadata[k]]) },
  };
}

describe('gRPC credit refusals', () => {
  it('maps overages_disabled to a permanent InsufficientCreditsError', () => {
    const error = creditErrorFromRpc(abort('overages_disabled', { 'credit-manage-url': '/dashboard/billing' }));

    expect(error).toBeInstanceOf(InsufficientCreditsError);
    expect(error).toBeInstanceOf(SynapPermanentError);
    const credit = error as InsufficientCreditsError;
    expect(credit.transient).toBe(false);
    expect(credit.reason).toBe('overages_disabled');
    expect(credit.manageUrl).toBe('/dashboard/billing');
    expect(credit.balanceCredits).toBe(0);
    expect(credit.requiredCredits).toBe(2);
    expect(credit.correlationId).toBe('req-grpc');
    expect(isCreditStop(credit)).toBe(true);
  });

  it('maps trial_limit_reached to a RateLimitError', () => {
    const error = creditErrorFromRpc(
      abort('trial_limit_reached', { 'credit-upgrade-url': '/dashboard/billing' }),
    ) as RateLimitError;

    expect(error).toBeInstanceOf(RateLimitError);
    expect(error.reason).toBe('trial_limit_reached');
    expect(error.upgradeUrl).toBe('/dashboard/billing');
    expect(isCreditStop(error)).toBe(true);
  });

  it('maps subscription_inactive to a RateLimitError', () => {
    const error = creditErrorFromRpc(
      abort('subscription_inactive', { 'credit-manage-url': '/dashboard/billing' }),
    ) as RateLimitError;

    expect(error.reason).toBe('subscription_inactive');
    expect(error.manageUrl).toBe('/dashboard/billing');
  });

  it('leaves an unrelated RESOURCE_EXHAUSTED alone', () => {
    // Server overload, a flow-control limit, an unknown future reason: none of
    // them are a credit refusal, so the stream must go on reconnecting.
    expect(creditErrorFromRpc({ code: RESOURCE_EXHAUSTED, details: 'bandwidth exhausted' })).toBeNull();
    expect(creditErrorFromRpc(abort(null))).toBeNull();
    expect(creditErrorFromRpc(abort('some_future_reason'))).toBeNull();
    expect(creditErrorFromRpc({ code: 14, details: 'insufficient_credits' })).toBeNull();
    expect(creditErrorFromRpc(null)).toBeNull();
  });

  it('does not call an ordinary rate limit a credit stop', () => {
    expect(isCreditStop(new RateLimitError('slow down'))).toBe(false);
    expect(isCreditStop(new InsufficientCreditsError('spent'))).toBe(false);
    expect(isCreditStop(new Error('nope'))).toBe(false);
  });

  it('ends the stream instead of reconnecting, against a real server', async () => {
    // The reconnect loop used to retry a refusal with backoff and then report
    // nothing but "disconnected".
    const grpc = await import('@grpc/grpc-js');
    const protoLoader = await import('@grpc/proto-loader');
    const pkgDef = protoLoader.fromJSON(
      SYNAP_PROTO_DESCRIPTOR as unknown as Parameters<typeof protoLoader.fromJSON>[0],
      { keepCase: true, longs: Number, enums: String, defaults: true, oneofs: true },
    );
    const pkg = grpc.loadPackageDefinition(pkgDef) as unknown as {
      synap: { v1: { SynapService: { service: never } } };
    };

    let calls = 0;
    const server = new grpc.Server();
    server.addService(pkg.synap.v1.SynapService.service, {
      Listen: (call: { emit: (e: string, a: unknown) => void }) => {
        calls += 1;
        const metadata = new grpc.Metadata();
        metadata.add('credit-balance', '0');
        metadata.add('credit-minimum-required', '2');
        metadata.add('credit-recovery-url', '/v1/credits/balance');
        metadata.add('credit-redeem-url', '/credits/redeem');
        metadata.add('credit-reason', 'overages_disabled');
        metadata.add('credit-manage-url', '/dashboard/billing');
        metadata.add('request-id', 'req-grpc');
        call.emit('error', {
          code: grpc.status.RESOURCE_EXHAUSTED,
          details: 'insufficient_credits',
          metadata,
        });
      },
      IngestTelemetry: () => { /* unused here */ },
    });

    const port = await new Promise<number>((resolve, reject) => {
      server.bindAsync('127.0.0.1:0', grpc.ServerCredentials.createInsecure(), (err, p) => {
        if (err) reject(err); else resolve(p);
      });
    });

    const states: string[] = [];
    const client = new GrpcStreamClient(CREDS, new AnticipationCache(), {
      host: '127.0.0.1',
      port,
      useTls: false,
      onStateChange: (s) => states.push(s),
    });

    try {
      await client.connect();
      const deadline = Date.now() + 5_000;
      while (client.currentState !== 'disconnected' && Date.now() < deadline) {
        await new Promise((r) => setTimeout(r, 25));
      }

      expect(client.currentState).toBe('disconnected');
      expect(client.lastError).toBeInstanceOf(InsufficientCreditsError);
      const credit = client.lastError as InsufficientCreditsError;
      expect(credit.reason).toBe('overages_disabled');
      expect(credit.manageUrl).toBe('/dashboard/billing');
      expect(credit.balanceCredits).toBe(0);
      expect(states).not.toContain('reconnecting');
      expect(calls).toBe(1);
    } finally {
      await client.disconnect();
      await new Promise<void>((resolve) => { server.tryShutdown(() => resolve()); });
    }
  }, 20_000);
});
