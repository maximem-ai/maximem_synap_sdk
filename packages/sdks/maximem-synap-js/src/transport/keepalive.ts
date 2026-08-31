/**
 * Connection reuse for the HTTP transport.
 *
 * ## Why this file exists at all
 *
 * Node's global `fetch` keeps an idle socket for roughly 4 seconds. Our real
 * call pattern is one fetch per conversation turn, often minutes apart, so
 * essentially every call would pay a full connection setup. Through the CDN to
 * the origin that was measured at 600-950ms.
 *
 * The Python SDK already solved this: `keepalive_expiry=300s` plus a 240s
 * `/health` heartbeat that keeps the socket warm for as long as the client
 * lives. Porting the SDK without porting this would ship a "native and faster"
 * release that is measurably *slower* than the Python-subprocess wrapper it
 * replaces, on every call, from day one. That is gotcha G-A.
 *
 * ## Why it is optional and lazily loaded
 *
 * `undici` is Node-only. Edge runtimes and Workers have their own connection
 * pooling and no `dispatcher` option, so there the correct behavior is to use
 * global `fetch` unchanged. Importing undici eagerly would break those builds
 * at bundle time, so it is behind a lazy `import()` and every failure path
 * degrades to plain `fetch` rather than throwing.
 */

/** Mirrors HTTPTransport.KEEPALIVE_EXPIRY_SECONDS in the Python SDK. */
export const KEEPALIVE_EXPIRY_MS = 300_000;
/** Mirrors HTTPTransport.HEARTBEAT_INTERVAL_SECONDS. */
export const HEARTBEAT_INTERVAL_MS = 240_000;
export const MAX_CONNECTIONS = 20;

export interface KeepAliveOptions {
  keepAliveExpiryMs?: number;
  maxConnections?: number;
}

/** An undici Dispatcher, typed loosely so undici is not a hard type dependency. */
export type Dispatcher = object;

let cached: Dispatcher | null | undefined;

/** True when we are on a runtime that can host an undici dispatcher. */
export function supportsDispatcher(): boolean {
  return (
    typeof process !== 'undefined' &&
    process.versions?.node !== undefined &&
    // Edge builds of Next expose a partial `process` shim, but no real Node.
    typeof (globalThis as { EdgeRuntime?: unknown }).EdgeRuntime === 'undefined'
  );
}

/**
 * Build (once) a dispatcher with Python-equivalent keep-alive.
 *
 * Returns null on any runtime where that is not possible, in which case the
 * caller should simply omit the dispatcher.
 */
export async function getKeepAliveDispatcher(
  options: KeepAliveOptions = {},
): Promise<Dispatcher | null> {
  if (cached !== undefined) return cached;
  if (!supportsDispatcher()) {
    cached = null;
    return cached;
  }
  try {
    // Bundlers must not try to follow this statically on Edge builds.
    const undici = (await import(/* @vite-ignore */ 'undici')) as {
      Agent: new (opts: Record<string, unknown>) => Dispatcher;
    };
    cached = new undici.Agent({
      keepAliveTimeout: options.keepAliveExpiryMs ?? KEEPALIVE_EXPIRY_MS,
      // undici will not honour a keepAliveTimeout above this ceiling.
      keepAliveMaxTimeout: options.keepAliveExpiryMs ?? KEEPALIVE_EXPIRY_MS,
      connections: options.maxConnections ?? MAX_CONNECTIONS,
    });
  } catch {
    // undici is an optionalDependency. Missing or unloadable is not fatal:
    // global fetch still works, just with a short-lived pool.
    cached = null;
  }
  return cached;
}

/** Test seam. */
export function _resetDispatcherCache(): void {
  cached = undefined;
}

/**
 * Keep an idle connection warm.
 *
 * The timer is `unref`'d. Without that the Node event loop stays alive for as
 * long as the interval exists and any CLI built on this SDK hangs forever
 * instead of exiting. Python has no equivalent symptom, so no parity test would
 * ever catch it: that is gotcha G-J.
 */
export class Heartbeat {
  private timer: ReturnType<typeof setInterval> | null = null;

  constructor(
    private readonly ping: () => void | Promise<void>,
    private readonly intervalMs: number = HEARTBEAT_INTERVAL_MS,
  ) {}

  start(): void {
    if (this.timer !== null) return;
    // 0 disables, matching the Python transport's convention.
    if (this.intervalMs <= 0) return;
    if (typeof setInterval !== 'function') return;

    this.timer = setInterval(() => {
      // A failed heartbeat is not an error the caller should ever see. The
      // next real request will surface a genuine connectivity problem.
      void Promise.resolve(this.ping()).catch(() => {});
    }, this.intervalMs);

    (this.timer as { unref?: () => void }).unref?.();
  }

  stop(): void {
    if (this.timer !== null) {
      clearInterval(this.timer);
      this.timer = null;
    }
  }

  get running(): boolean {
    return this.timer !== null;
  }
}
