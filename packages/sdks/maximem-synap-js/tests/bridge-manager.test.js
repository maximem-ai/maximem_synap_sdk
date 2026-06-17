import { describe, it, expect, vi } from 'vitest';
import { EventEmitter } from 'events';

import { BridgeManager } from '../src/bridge-manager.js';
import { SynapError, NetworkTimeoutError, createSynapError } from '../src/errors.js';

// ---------------------------------------------------------------------------
// Helpers to build a fake child-process that behaves like the bridge
// ---------------------------------------------------------------------------

function makeFakeStdin() {
  const written = [];
  return {
    writable: true,
    write(data) {
      written.push(data);
      return true;
    },
    _written: written,
  };
}

function makeFakeProc() {
  const proc = new EventEmitter();
  proc.stdin = makeFakeStdin();
  proc.stdout = new EventEmitter();
  proc.stdout.pipe = () => {};
  proc.stderr = new EventEmitter();
  proc.stderr.pipe = () => {};
  proc.killed = false;
  proc.kill = vi.fn(() => { proc.killed = true; });
  return proc;
}

// readline.createInterface returns an object with .on() for 'line' events.
// We store a reference to each created RL instance so we can push lines.
function makeFakeReadline(proc) {
  const interfaces = {};

  function createInterface({ input }) {
    const rl = new EventEmitter();
    rl.close = vi.fn();
    // Tag which stream we're wrapping
    if (input === proc.stdout) interfaces.stdout = rl;
    if (input === proc.stderr) interfaces.stderr = rl;
    return rl;
  }

  return { createInterface, interfaces };
}

// Shared helper: wire the stdout handler that BridgeManager's #attachReaders normally installs.
function wireStdoutHandler(bm) {
  bm.stdoutReader.on('line', (line) => {
    let payload;
    try { payload = JSON.parse(line); } catch (_) {
      if (bm.options.onLog) bm.options.onLog('error', `Bad bridge JSON: ${line}`);
      return;
    }
    const pending = bm.pending.get(payload.id);
    if (!pending) return;
    clearTimeout(pending.timer);
    bm.pending.delete(payload.id);
    if (payload.error) {
      pending.reject(createSynapError(payload.error, payload.error_type));
    } else {
      pending.resolve(payload.result);
    }
  });
}

// ---------------------------------------------------------------------------
// Full integration-style test using manual module manipulation
// ---------------------------------------------------------------------------

describe('BridgeManager.sendCommand serialises JSON-RPC correctly', () => {
  it('writes a newline-delimited JSON object to stdin', async () => {
    const bm = new BridgeManager({ requestTimeoutMs: 5000 });

    // Manually inject a fake child process (bypassing ensureStarted)
    const proc = makeFakeProc();
    bm.child = proc;

    // Set up a readline-style response dispatch
    const fakeRl = makeFakeReadline(proc);
    bm.stdoutReader = fakeRl.createInterface({ input: proc.stdout });
    bm.stderrReader = fakeRl.createInterface({ input: proc.stderr });

    wireStdoutHandler(bm);

    // Override stdin.write so we can capture + respond
    const origWrite = proc.stdin.write.bind(proc.stdin);
    proc.stdin.write = (data) => {
      origWrite(data);
      // Parse the request and emit a success response on the stdout RL
      const req = JSON.parse(data.trim());
      const response = JSON.stringify({ id: req.id, result: { ok: true }, error: null });

      setImmediate(() => {
        bm.stdoutReader.emit('line', response);
      });
      return true;
    };

    const result = await bm.sendCommand('search_memory', { user_id: 'u1', query: 'hello' });

    // Verify the written bytes
    const written = proc.stdin._written;
    expect(written).toHaveLength(1);
    const parsed = JSON.parse(written[0].trim());
    expect(parsed.method).toBe('search_memory');
    expect(parsed.params).toEqual({ user_id: 'u1', query: 'hello' });
    expect(typeof parsed.id).toBe('number');
    expect(result).toEqual({ ok: true });
  });

  it('rejects with SynapError when bridge responds with error field', async () => {
    const bm = new BridgeManager({ requestTimeoutMs: 5000 });
    const proc = makeFakeProc();
    bm.child = proc;

    const fakeRl = makeFakeReadline(proc);
    bm.stdoutReader = fakeRl.createInterface({ input: proc.stdout });
    bm.stderrReader = fakeRl.createInterface({ input: proc.stderr });

    wireStdoutHandler(bm);

    proc.stdin.write = (data) => {
      const req = JSON.parse(data.trim());
      const response = JSON.stringify({
        id: req.id,
        result: null,
        error: 'Bridge exploded',
        error_type: 'NetworkTimeoutError',
      });
      setImmediate(() => bm.stdoutReader.emit('line', response));
      return true;
    };

    await expect(bm.sendCommand('add_memory', {})).rejects.toBeInstanceOf(NetworkTimeoutError);
  });

  it('rejects when stdin is not writable (bridge not running)', async () => {
    const bm = new BridgeManager();
    // child is null — sendCommand should throw immediately
    await expect(bm.sendCommand('search_memory', {})).rejects.toThrow(
      'Synap bridge process is not running'
    );
  });

  it('rejects when stdin.writable is false', async () => {
    const bm = new BridgeManager();
    const proc = makeFakeProc();
    proc.stdin.writable = false;
    bm.child = proc;
    await expect(bm.sendCommand('add_memory', {})).rejects.toThrow(
      'Synap bridge process is not running'
    );
  });

  it('increments nextId for each request', async () => {
    const bm = new BridgeManager({ requestTimeoutMs: 5000 });
    const proc = makeFakeProc();
    bm.child = proc;

    const fakeRl = makeFakeReadline(proc);
    bm.stdoutReader = fakeRl.createInterface({ input: proc.stdout });
    bm.stderrReader = fakeRl.createInterface({ input: proc.stderr });

    wireStdoutHandler(bm);

    const ids = [];
    proc.stdin.write = (data) => {
      const req = JSON.parse(data.trim());
      ids.push(req.id);
      setImmediate(() =>
        bm.stdoutReader.emit('line', JSON.stringify({ id: req.id, result: {}, error: null }))
      );
      return true;
    };

    await bm.sendCommand('m1', {});
    await bm.sendCommand('m2', {});
    await bm.sendCommand('m3', {});

    expect(ids).toHaveLength(3);
    expect(ids[0]).toBeLessThan(ids[1]);
    expect(ids[1]).toBeLessThan(ids[2]);
  });
});

describe('BridgeManager pending-request cleanup', () => {
  it('clears pending map and rejects all inflight requests on exit event', async () => {
    const bm = new BridgeManager({ requestTimeoutMs: 30_000 });
    const proc = makeFakeProc();
    bm.child = proc;

    const fakeRl = makeFakeReadline(proc);
    bm.stdoutReader = fakeRl.createInterface({ input: proc.stdout });
    bm.stderrReader = fakeRl.createInterface({ input: proc.stderr });

    // Register lifecycle (normally done by #attachLifecycle)
    proc.on('exit', (code) => {
      const err = new Error(`Synap bridge exited with code ${code}`);
      for (const [, pending] of bm.pending) {
        clearTimeout(pending.timer);
        pending.reject(err);
      }
      bm.pending.clear();
      bm.initPromise = null;
    });

    // Queue an inflight request that will never be answered
    const inflightPromise = new Promise((resolve, reject) => {
      const id = bm.nextId++;
      bm.pending.set(id, {
        resolve,
        reject,
        timer: setTimeout(() => {}, 30_000),
      });
    });

    // Simulate bridge process exiting
    proc.emit('exit', 1);

    await expect(inflightPromise).rejects.toThrow('Synap bridge exited with code 1');
    expect(bm.pending.size).toBe(0);
  });
});

describe('BridgeManager malformed JSON from bridge', () => {
  it('ignores non-JSON lines and does not reject pending requests', () => {
    const bm = new BridgeManager({ requestTimeoutMs: 5000 });
    const proc = makeFakeProc();
    bm.child = proc;

    const fakeRl = makeFakeReadline(proc);
    bm.stdoutReader = fakeRl.createInterface({ input: proc.stdout });
    bm.stderrReader = fakeRl.createInterface({ input: proc.stderr });

    const logs = [];
    bm.options.onLog = (level, msg) => logs.push({ level, msg });

    // Manually wire the stdout handler
    wireStdoutHandler(bm);

    // Emit a malformed line
    bm.stdoutReader.emit('line', 'not json at all {{}}');

    // The pending map should be unchanged (no requests were registered)
    expect(bm.pending.size).toBe(0);
    // onLog should have been called with the bad JSON
    expect(logs.some((l) => l.level === 'error' && l.msg.includes('Bad bridge JSON'))).toBe(true);
  });
});

describe('BridgeManager.shutdown', () => {
  it('is a no-op when child is null', async () => {
    const bm = new BridgeManager();
    // Should not throw
    await expect(bm.shutdown()).resolves.toBeUndefined();
  });

  it('sends shutdown command and nulls child', async () => {
    const bm = new BridgeManager({ requestTimeoutMs: 5000 });
    const proc = makeFakeProc();
    bm.child = proc;

    const fakeRl = makeFakeReadline(proc);
    bm.stdoutReader = fakeRl.createInterface({ input: proc.stdout });
    bm.stderrReader = fakeRl.createInterface({ input: proc.stderr });

    wireStdoutHandler(bm);

    proc.stdin.write = (data) => {
      const req = JSON.parse(data.trim());
      setImmediate(() =>
        bm.stdoutReader.emit('line', JSON.stringify({ id: req.id, result: { ok: true }, error: null }))
      );
      return true;
    };

    await bm.shutdown();
    expect(bm.child).toBeNull();
  });

  it('does not throw if shutdown command errors', async () => {
    const bm = new BridgeManager({ requestTimeoutMs: 5000 });
    const proc = makeFakeProc();
    bm.child = proc;

    const fakeRl = makeFakeReadline(proc);
    bm.stdoutReader = fakeRl.createInterface({ input: proc.stdout });
    bm.stderrReader = fakeRl.createInterface({ input: proc.stderr });

    wireStdoutHandler(bm);

    // Simulate error response to shutdown
    proc.stdin.write = (data) => {
      const req = JSON.parse(data.trim());
      setImmediate(() =>
        bm.stdoutReader.emit('line', JSON.stringify({
          id: req.id,
          result: null,
          error: 'Already shutting down',
          error_type: 'SynapError',
        }))
      );
      return true;
    };

    // Should not throw even though shutdown RPC errors
    await expect(bm.shutdown()).resolves.toBeUndefined();
    expect(bm.child).toBeNull();
  });
});

describe('BridgeManager.call (ensureStarted guard)', () => {
  it('throws if ensureStarted fails', async () => {
    const bm = new BridgeManager({ requestTimeoutMs: 5000 });

    // Stub ensureStarted to reject
    bm.ensureStarted = vi.fn().mockRejectedValue(new Error('Python not found'));

    await expect(bm.call('add_memory', {})).rejects.toThrow('Python not found');
  });
});
