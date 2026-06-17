/**
 * runtime.js unit tests.
 *
 * Covers: getSdkHome, getVenvPythonPath, resolveBridgeScriptPath,
 * resolvePythonBin, resolveInstanceId, runCommand.
 *
 * NOTE ON MOCKING STRATEGY:
 * runtime.js is a CommonJS module that does:
 *   const { spawn } = require('child_process')   ← destructured at load time
 *   const fs = require('fs')                      ← reference captured at load time
 *
 * `vi.mock('child_process')` only patches the ESM namespace — the CJS
 * destructured `spawn` local is already set and cannot be re-patched.
 *
 * `vi.spyOn(fs, 'existsSync')` DOES work because `fs` is an object reference
 * shared between the test and the CJS module.  The spy mutates the same
 * property on the same object both modules hold.
 *
 * For `runCommand` (which calls `spawn` directly), we fall back to running
 * real, safe, non-Python commands (`node --version`, `node -e "..."`) instead
 * of mocking spawn.  These are always available in a Node.js test environment,
 * are non-destructive, and take < 500ms each.
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import os from 'os';
import path from 'path';
import fs from 'fs';

import {
  getSdkHome,
  getVenvPythonPath,
  resolveBridgeScriptPath,
  resolvePythonBin,
  resolveInstanceId,
  runCommand,
} from '../src/runtime.js';

// ---------------------------------------------------------------------------
// Shared setup
// ---------------------------------------------------------------------------

const homeDir = os.homedir();

beforeEach(() => {
  // Clean env vars that affect runtime behaviour
  delete process.env.SYNAP_JS_SDK_HOME;
  delete process.env.SYNAP_PYTHON_BIN;
  delete process.env.SYNAP_INSTANCE_ID;
});

afterEach(() => {
  vi.restoreAllMocks();
});

// ---------------------------------------------------------------------------
// getSdkHome
// ---------------------------------------------------------------------------

describe('getSdkHome', () => {
  it('returns customHome when provided', () => {
    expect(getSdkHome('/custom/home')).toBe('/custom/home');
  });

  it('returns SYNAP_JS_SDK_HOME env var when set', () => {
    process.env.SYNAP_JS_SDK_HOME = '/env/sdk/home';
    expect(getSdkHome()).toBe('/env/sdk/home');
  });

  it('returns ~/.synap-js-sdk when no override provided', () => {
    expect(getSdkHome()).toBe(path.join(homeDir, '.synap-js-sdk'));
  });

  it('customHome takes priority over env var', () => {
    process.env.SYNAP_JS_SDK_HOME = '/env/home';
    expect(getSdkHome('/explicit')).toBe('/explicit');
  });
});

// ---------------------------------------------------------------------------
// getVenvPythonPath
// ---------------------------------------------------------------------------

describe('getVenvPythonPath', () => {
  it('returns Scripts/python.exe on Windows', () => {
    const origPlatform = process.platform;
    Object.defineProperty(process, 'platform', { value: 'win32', configurable: true });

    expect(getVenvPythonPath('/my/venv')).toBe(path.join('/my/venv', 'Scripts', 'python.exe'));

    Object.defineProperty(process, 'platform', { value: origPlatform, configurable: true });
  });

  it('returns bin/python on non-Windows', () => {
    if (process.platform !== 'win32') {
      expect(getVenvPythonPath('/my/venv')).toBe(path.join('/my/venv', 'bin', 'python'));
    }
  });
});

// ---------------------------------------------------------------------------
// resolveBridgeScriptPath
// ---------------------------------------------------------------------------

describe('resolveBridgeScriptPath', () => {
  it('returns custom path when provided', () => {
    expect(resolveBridgeScriptPath('/custom/bridge.py')).toBe('/custom/bridge.py');
  });

  it('resolves to bridge/synap_bridge.py relative to src/', () => {
    const result = resolveBridgeScriptPath();
    expect(result).toContain('synap_bridge.py');
    expect(result).toContain('bridge');
  });
});

// ---------------------------------------------------------------------------
// resolvePythonBin
// ---------------------------------------------------------------------------

describe('resolvePythonBin', () => {
  it('returns candidate list starting with explicit pythonBin when given', () => {
    const candidates = resolvePythonBin({ pythonBin: '/usr/local/bin/python3' });
    expect(candidates[0]).toBe('/usr/local/bin/python3');
  });

  it('includes SYNAP_PYTHON_BIN env var in candidates', () => {
    process.env.SYNAP_PYTHON_BIN = '/env/python3';
    const candidates = resolvePythonBin({});
    expect(candidates).toContain('/env/python3');
  });

  it('includes venv python path in candidates', () => {
    const candidates = resolvePythonBin({ venvPath: '/my/venv' });
    const hasVenv = candidates.some((c) => c.includes('/my/venv'));
    expect(hasVenv).toBe(true);
  });

  it('falls back to python3 or python on non-Windows', () => {
    if (process.platform !== 'win32') {
      const candidates = resolvePythonBin({});
      expect(candidates).toContain('python3');
      expect(candidates).toContain('python');
    }
  });

  it('returns candidates as a non-empty array', () => {
    const candidates = resolvePythonBin({});
    expect(Array.isArray(candidates)).toBe(true);
    expect(candidates.length).toBeGreaterThan(0);
  });
});

// ---------------------------------------------------------------------------
// resolveInstanceId
// Uses vi.spyOn on the real fs module — works because CJS runtime.js holds a
// reference to the same object we spy on.
// ---------------------------------------------------------------------------

describe('resolveInstanceId', () => {
  let existsSpy;
  let readdirSpy;
  let readFileSpy;

  beforeEach(() => {
    existsSpy = vi.spyOn(fs, 'existsSync');
    readdirSpy = vi.spyOn(fs, 'readdirSync');
    readFileSpy = vi.spyOn(fs, 'readFileSync');
  });

  it('returns explicit id when provided', () => {
    // Explicit id bypasses all FS calls
    expect(resolveInstanceId('inst_abc123')).toBe('inst_abc123');
  });

  it('returns SYNAP_INSTANCE_ID env var when set', () => {
    process.env.SYNAP_INSTANCE_ID = 'inst_from_env';
    expect(resolveInstanceId()).toBe('inst_from_env');
  });

  it('returns empty string when instances dir does not exist', () => {
    existsSpy.mockReturnValue(false);
    expect(resolveInstanceId()).toBe('');
  });

  it('picks the most recently issued non-expired instance', () => {
    const now = new Date();
    const future = new Date(now.getTime() + 86400 * 1000).toISOString();
    const earlier = new Date(now.getTime() - 3600 * 1000).toISOString();
    const later = new Date(now.getTime() - 1800 * 1000).toISOString();

    existsSpy.mockReturnValue(true);
    readdirSpy.mockReturnValue(['inst-a', 'inst-b']);
    readFileSpy.mockImplementation((p) => {
      if (String(p).includes('inst-a')) {
        return JSON.stringify({ instance_id: 'inst_A', issued_at: earlier, expires_at: future });
      }
      return JSON.stringify({ instance_id: 'inst_B', issued_at: later, expires_at: future });
    });

    expect(resolveInstanceId()).toBe('inst_B');
  });

  it('skips expired instances', () => {
    const now = new Date();
    const past = new Date(now.getTime() - 86400 * 1000).toISOString();
    const issuedAgo = new Date(now.getTime() - 2 * 86400 * 1000).toISOString();

    existsSpy.mockReturnValue(true);
    readdirSpy.mockReturnValue(['inst-expired']);
    readFileSpy.mockReturnValue(
      JSON.stringify({ instance_id: 'inst_EXPIRED', issued_at: issuedAgo, expires_at: past })
    );

    expect(resolveInstanceId()).toBe('');
  });

  it('returns empty string when metadata is malformed JSON', () => {
    existsSpy.mockReturnValue(true);
    readdirSpy.mockReturnValue(['bad-dir']);
    readFileSpy.mockReturnValue('{ not valid json ');

    expect(resolveInstanceId()).toBe('');
  });

  it('skips entries with NaN dates', () => {
    existsSpy.mockReturnValue(true);
    readdirSpy.mockReturnValue(['broken-dates']);
    readFileSpy.mockReturnValue(
      JSON.stringify({ instance_id: 'inst_X', issued_at: 'not-a-date', expires_at: 'not-a-date' })
    );

    expect(resolveInstanceId()).toBe('');
  });
});

// ---------------------------------------------------------------------------
// runCommand
//
// `spawn` is destructured from require('child_process') at CJS module-load
// time; vi.spyOn / vi.mock cannot intercept it after that point.
//
// We test runCommand's real behaviour using safe, always-available commands:
//   `node --version`  →  exit 0, stdout contains 'v'
//   `node -e "process.exit(2)"` → exit 2, rejects
//   `node -e "process.stdout.write('out'); process.stderr.write('err'); process.exit(3)"`
//      → exit 3, err.stdout / err.stderr populated
//
// No Python, no cloud, no network.
// ---------------------------------------------------------------------------

describe('runCommand (integration via node CLI)', () => {
  it('resolves with { code, stdout, stderr } on exit 0', async () => {
    const result = await runCommand('node', ['--version']);
    expect(result.code).toBe(0);
    expect(result.stdout).toMatch(/^v\d+/);
    expect(result.stderr).toBe('');
  });

  it('rejects on non-zero exit code', async () => {
    const err = await runCommand('node', ['-e', 'process.exit(1)']).catch((e) => e);
    expect(err).toBeInstanceOf(Error);
    expect(err.message).toContain('exit 1');
  });

  it('attaches numeric exit code, stdout, and stderr to rejected error', async () => {
    const script = `process.stdout.write('out-msg\\n'); process.stderr.write('err-msg\\n'); process.exit(2)`;
    const err = await runCommand('node', ['-e', script]).catch((e) => e);
    expect(err.code).toBe(2);
    expect(err.stdout).toContain('out-msg');
    expect(err.stderr).toContain('err-msg');
  });

  it('rejects with ENOENT when command does not exist', async () => {
    const err = await runCommand('totally-nonexistent-binary-synap-xyz', []).catch((e) => e);
    expect(err).toBeInstanceOf(Error);
    expect(err.code).toBe('ENOENT');
  });
});
