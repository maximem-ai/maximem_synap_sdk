/**
 * Environment access.
 *
 * Every read happens at CALL time, never at module load. Python does
 * `os.environ.get(...)` inside the function bodies, and tests plus runtime
 * toggles depend on that: a value captured at import time makes a later
 * `process.env` mutation look like the flag simply does not work. That is
 * gotcha G-S.
 *
 * `process` is absent on Workers and in the browser, so every accessor
 * tolerates it being missing.
 */

const TRUTHY = new Set(['true', '1', 'yes']);

function readEnv(name: string): string | undefined {
  if (typeof process === 'undefined' || process.env === undefined) return undefined;
  return process.env[name];
}

/** Boolean flag. Defaults to false, matching the Python SDK's flags. */
export function getEnvFlag(name: string, fallback = false): boolean {
  const raw = readEnv(name);
  if (raw === undefined || raw.trim() === '') return fallback;
  return TRUTHY.has(raw.trim().toLowerCase());
}

/** Numeric setting. Returns null when unset OR malformed: a bad value must not fail closed. */
export function getEnvFloat(name: string): number | null {
  const raw = readEnv(name)?.trim();
  if (!raw) return null;
  const n = Number(raw);
  return Number.isFinite(n) ? n : null;
}

export function getEnv(name: string): string | undefined {
  const v = readEnv(name);
  return v === undefined || v === '' ? undefined : v;
}
