import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import { CredentialManager } from '../auth/credential-manager.js';

describe('CredentialManager', () => {
  const originalEnv = process.env;

  beforeEach(() => {
    process.env = { ...originalEnv };
    delete process.env['SYNAP_API_KEY'];
    delete process.env['SYNAP_CLIENT_ID'];
  });

  afterEach(() => {
    process.env = originalEnv;
  });

  it('returns credentials when explicit apiKey is provided', async () => {
    const manager = new CredentialManager();
    const creds = await manager.load('sk-test-key');
    expect(creds.api_key).toBe('sk-test-key');
    expect(creds.instance_id).toBe('');
  });

  it('returns credentials from SYNAP_API_KEY env var', async () => {
    process.env['SYNAP_API_KEY'] = 'sk-env-key';
    process.env['SYNAP_CLIENT_ID'] = 'client-123';
    const manager = new CredentialManager();
    const creds = await manager.load();
    expect(creds.api_key).toBe('sk-env-key');
    expect(creds.client_id).toBe('client-123');
  });

  it('prefers explicit apiKey over env var', async () => {
    process.env['SYNAP_API_KEY'] = 'sk-env-key';
    const manager = new CredentialManager();
    const creds = await manager.load('sk-explicit-key');
    expect(creds.api_key).toBe('sk-explicit-key');
  });

  it('caches credentials on second load()', async () => {
    const manager = new CredentialManager();
    const first = await manager.load('sk-key');
    const second = await manager.load('sk-other-key');
    expect(second).toBe(first);
  });

  it('throws when no API key is available', async () => {
    const manager = new CredentialManager();
    await expect(manager.load()).rejects.toThrow('No Synap API key found');
  });
});
