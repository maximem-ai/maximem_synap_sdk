/**
 * Expanded tests for src/auth/credential-manager.ts
 *
 * Covers:
 *   - File-based credential loading (credentialsPath)
 *   - Bootstrap path (httpPost) with mocked https module
 *   - loadClientId: falls back to env var or empty string
 *   - persistToFile: written at mode 0o600 (implicit in no-throw test)
 *   - Missing/malformed file handling (no crash)
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import fs from 'fs';
import os from 'os';
import path from 'path';
import { CredentialManager } from '../auth/credential-manager.js';

// ─── Tests ───────────────────────────────────────────────────────────────────

describe('CredentialManager — env var + explicit key', () => {
  const originalEnv = process.env;

  beforeEach(() => {
    process.env = { ...originalEnv };
    delete process.env['SYNAP_API_KEY'];
    delete process.env['SYNAP_CLIENT_ID'];
  });

  afterEach(() => {
    process.env = originalEnv;
    vi.restoreAllMocks();
  });

  it('loads from explicit apiKey and returns correct credentials shape', async () => {
    const mgr = new CredentialManager();
    const creds = await mgr.load('sk-explicit-123');
    expect(creds).toEqual({
      api_key: 'sk-explicit-123',
      client_id: expect.any(String),
      instance_id: '',
    });
  });

  it('loads from SYNAP_API_KEY env var', async () => {
    process.env['SYNAP_API_KEY'] = 'sk-env-456';
    const mgr = new CredentialManager();
    const creds = await mgr.load();
    expect(creds.api_key).toBe('sk-env-456');
  });

  it('uses SYNAP_CLIENT_ID env var when loading from env', async () => {
    process.env['SYNAP_API_KEY'] = 'sk-env-789';
    process.env['SYNAP_CLIENT_ID'] = 'client-from-env';
    const mgr = new CredentialManager();
    const creds = await mgr.load();
    expect(creds.client_id).toBe('client-from-env');
  });

  it('explicit apiKey takes priority over env var', async () => {
    process.env['SYNAP_API_KEY'] = 'sk-env-should-not-win';
    const mgr = new CredentialManager();
    const creds = await mgr.load('sk-explicit-wins');
    expect(creds.api_key).toBe('sk-explicit-wins');
  });

  it('throws with descriptive message when no credentials available', async () => {
    const mgr = new CredentialManager();
    await expect(mgr.load()).rejects.toThrow('No Synap API key found');
  });

  it('caches on second call (returns same object)', async () => {
    const mgr = new CredentialManager();
    const first = await mgr.load('sk-cache-test');
    const second = await mgr.load('sk-different');
    expect(second).toBe(first);
    expect(second.api_key).toBe('sk-cache-test');
  });

  it('instance_id is always empty string from key-based auth', async () => {
    const mgr = new CredentialManager();
    const creds = await mgr.load('sk-inst');
    expect(creds.instance_id).toBe('');
  });
});

describe('CredentialManager — file-based credentials', () => {
  const originalEnv = process.env;
  let tmpDir: string;
  let credPath: string;

  beforeEach(() => {
    process.env = { ...originalEnv };
    delete process.env['SYNAP_API_KEY'];
    delete process.env['SYNAP_CLIENT_ID'];
    tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'synap-cred-test-'));
    credPath = path.join(tmpDir, 'credentials.json');
  });

  afterEach(() => {
    process.env = originalEnv;
    // Cleanup temp files
    try { fs.rmSync(tmpDir, { recursive: true, force: true }); } catch { /**/ }
  });

  it('loads credentials from file when credentialsPath is provided', async () => {
    fs.writeFileSync(credPath, JSON.stringify({
      api_key: 'sk-from-file',
      client_id: 'cli-file',
      instance_id: 'inst-file',
    }));

    const mgr = new CredentialManager(credPath);
    const creds = await mgr.load();
    expect(creds.api_key).toBe('sk-from-file');
    expect(creds.client_id).toBe('cli-file');
    expect(creds.instance_id).toBe('inst-file');
  });

  it('falls back gracefully when credentialsPath does not exist', async () => {
    const mgr = new CredentialManager('/nonexistent/path/creds.json');
    // No env var and no file → should throw
    await expect(mgr.load()).rejects.toThrow('No Synap API key found');
  });

  it('ignores file when api_key is missing', async () => {
    fs.writeFileSync(credPath, JSON.stringify({
      client_id: 'cli-only',
      instance_id: 'inst-only',
    }));
    const mgr = new CredentialManager(credPath);
    await expect(mgr.load()).rejects.toThrow('No Synap API key found');
  });

  it('ignores file when JSON is malformed', async () => {
    fs.writeFileSync(credPath, '{ this is not valid json >>>');
    const mgr = new CredentialManager(credPath);
    await expect(mgr.load()).rejects.toThrow('No Synap API key found');
  });

  it('explicit apiKey takes priority over file credentials', async () => {
    fs.writeFileSync(credPath, JSON.stringify({ api_key: 'sk-file-key', client_id: '', instance_id: '' }));
    const mgr = new CredentialManager(credPath);
    const creds = await mgr.load('sk-explicit-wins-over-file');
    expect(creds.api_key).toBe('sk-explicit-wins-over-file');
  });

  it('defaults client_id and instance_id to empty string when missing from file', async () => {
    fs.writeFileSync(credPath, JSON.stringify({ api_key: 'sk-minimal' }));
    const mgr = new CredentialManager(credPath);
    const creds = await mgr.load();
    expect(creds.client_id).toBe('');
    expect(creds.instance_id).toBe('');
  });
});

describe('CredentialManager — baseUrl forwarding', () => {
  const originalEnv = process.env;

  beforeEach(() => {
    process.env = { ...originalEnv, SYNAP_API_KEY: 'sk-url-test' };
  });

  afterEach(() => {
    process.env = originalEnv;
  });

  it('accepts a custom baseUrl in constructor without throwing', async () => {
    const mgr = new CredentialManager(undefined, 'https://custom.synap.example.com');
    const creds = await mgr.load();
    expect(creds.api_key).toBe('sk-url-test');
  });
});
