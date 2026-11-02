/**
 * Local mode: talking to the open-source Synap server, which has no auth.
 *
 * It is on when no API key is given (option or SYNAP_API_KEY) and a base URL
 * is configured. A key always wins, so Cloud callers never see a difference.
 */
import { getEnv } from './util/env.js';

export function isLocalMode(apiKey: string | undefined, baseUrl: string | undefined): boolean {
  if (apiKey || getEnv('SYNAP_API_KEY')) return false;
  return baseUrl !== undefined && baseUrl.trim() !== '';
}

/**
 * The optional shared token the open-source server may require. Read from the
 * environment only. Empty means send no Authorization header.
 */
export function localToken(): string {
  return getEnv('SYNAP_API_TOKEN')?.trim() ?? '';
}
