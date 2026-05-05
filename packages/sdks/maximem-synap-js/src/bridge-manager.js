const { spawn, spawnSync } = require('child_process');
const readline = require('readline');
const fs = require('fs');
const {
  resolveBridgeScriptPath,
  resolvePythonBin,
  resolveInstanceId,
  setupPythonRuntime,
} = require('./runtime');
const { createSynapError } = require('./errors');

class BridgeManager {
  constructor(options = {}) {
    this.options = {
      requestTimeoutMs: 30_000,
      initTimeoutMs: 45_000,
      ingestTimeoutMs: 120_000,
      autoSetup: false,
      ...options,
    };

    this.child = null;
    this.stdoutReader = null;
    this.stderrReader = null;
    this.pending = new Map();
    this.nextId = 1;
    this.initPromise = null;
  }

  async ensureStarted() {
    if (this.initPromise) return this.initPromise;

    this.initPromise = (async () => {
      const bridgeScript = resolveBridgeScriptPath(this.options.bridgeScriptPath);
      const pythonBin = await this.#resolvePythonBinary();
      this.#preflightPythonImport(pythonBin);

      if (!fs.existsSync(bridgeScript)) {
        throw new Error(`Bridge script not found at ${bridgeScript}`);
      }

      const bridgeEnv = {
        ...process.env,
        PYTHONUNBUFFERED: '1',
        ...(this.options.pythonEnv || {}),
      };

      this.child = spawn(pythonBin, [bridgeScript], {
        stdio: ['pipe', 'pipe', 'pipe'],
        env: bridgeEnv,
      });

      this.#attachReaders();
      this.#attachLifecycle();

      const instanceId = resolveInstanceId(this.options.instanceId);

      await this.sendCommand(
        'init',
        {
          instance_id: instanceId,
          api_key: this.options.apiKey
            || process.env.SYNAP_API_KEY,
          base_url: this.options.baseUrl || process.env.SYNAP_BASE_URL,
          grpc_host: this.options.grpcHost || process.env.SYNAP_GRPC_HOST,
          grpc_port: Number(this.options.grpcPort || process.env.SYNAP_GRPC_PORT || 50051),
          grpc_use_tls: this.options.grpcUseTls ?? (process.env.SYNAP_GRPC_TLS === 'true'),
        },
        this.options.initTimeoutMs
      );
    })()
      .catch((error) => {
        this.initPromise = null;
        throw error;
      });

    return this.initPromise;
  }

  async sendCommand(method, params = {}, timeoutMs = this.options.requestTimeoutMs) {
    if (!this.child || !this.child.stdin || !this.child.stdin.writable) {
      throw new Error('Synap bridge process is not running');
    }

    const id = this.nextId++;

    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => {
        this.pending.delete(id);
        reject(new Error(`Bridge command '${method}' timed out after ${timeoutMs}ms`));
      }, timeoutMs);

      this.pending.set(id, { resolve, reject, timer });

      try {
        this.child.stdin.write(`${JSON.stringify({ id, method, params })}\n`);
      } catch (error) {
        clearTimeout(timer);
        this.pending.delete(id);
        reject(error);
      }
    });
  }

  async call(method, params = {}, timeoutMs) {
    await this.ensureStarted();
    return this.sendCommand(method, params, timeoutMs);
  }

  async shutdown() {
    if (!this.child) return;

    try {
      await this.sendCommand('shutdown', {}, 10_000);
    } catch (_) {
      // Ignore shutdown RPC errors and continue with forced process stop.
    }

    const proc = this.child;
    this.child = null;

    setTimeout(() => {
      if (proc && !proc.killed) proc.kill();
    }, 1_500);

    this.#resetState();
  }

  #attachReaders() {
    this.stdoutReader = readline.createInterface({ input: this.child.stdout });
    this.stderrReader = readline.createInterface({ input: this.child.stderr });

    this.stdoutReader.on('line', (line) => {
      let payload;

      try {
        payload = JSON.parse(line);
      } catch (_) {
        if (this.options.onLog) this.options.onLog('error', `Bad bridge JSON: ${line}`);
        return;
      }

      const pending = this.pending.get(payload.id);
      if (!pending) return;

      clearTimeout(pending.timer);
      this.pending.delete(payload.id);

      if (payload.error) {
        pending.reject(createSynapError(payload.error, payload.error_type));
      } else {
        pending.resolve(payload.result);
      }
    });

    this.stderrReader.on('line', (line) => {
      if (this.options.onLog) this.options.onLog('debug', line);
    });
  }

  #attachLifecycle() {
    this.child.on('exit', (code) => {
      const err = new Error(`Synap bridge exited with code ${code}`);
      for (const [, pending] of this.pending) {
        clearTimeout(pending.timer);
        pending.reject(err);
      }
      this.#resetState();
      if (this.options.onLog) this.options.onLog('error', err.message);
    });

    this.child.on('error', (error) => {
      for (const [, pending] of this.pending) {
        clearTimeout(pending.timer);
        pending.reject(error);
      }
      this.#resetState();
      if (this.options.onLog) this.options.onLog('error', error.message);
    });
  }

  #resetState() {
    this.pending.clear();
    this.initPromise = null;

    if (this.stdoutReader) {
      this.stdoutReader.close();
      this.stdoutReader = null;
    }
    if (this.stderrReader) {
      this.stderrReader.close();
      this.stderrReader = null;
    }
  }

  async #resolvePythonBinary() {
    const candidates = resolvePythonBin(this.options);

    for (const candidate of candidates) {
      if (!candidate) continue;
      if (candidate.includes('/') || candidate.includes('\\')) {
        if (fs.existsSync(candidate)) return candidate;
        continue;
      }

      // Command-style candidate (python3/python).
      return candidate;
    }

    if (this.options.autoSetup) {
      const setupResult = await setupPythonRuntime({
        sdkHome: this.options.sdkHome,
        venvPath: this.options.venvPath,
        pythonBootstrap: this.options.pythonBootstrap,
        pythonPackage: this.options.pythonPackage,
        pythonSdkVersion: this.options.pythonSdkVersion,
        noDeps: this.options.noDeps,
        noBuildIsolation: this.options.noBuildIsolation,
        upgrade: this.options.upgrade,
        forceRecreateVenv: this.options.forceRecreateVenv,
      });
      return setupResult.pythonBin;
    }

    throw new Error(
      'No usable Python runtime found. Run `npx synap-js-sdk setup` or pass `pythonBin` in SynapClient options.'
    );
  }

  #preflightPythonImport(pythonBin) {
    const envBase = {
      ...process.env,
      ...(this.options.pythonEnv || {}),
    };

    const checkImport = (envExtra = {}) => {
      const env = { ...envBase, ...envExtra };
      return spawnSync(
        pythonBin,
        ['-c', 'import maximem_synap,sys; print(maximem_synap.__file__)'],
        { env, encoding: 'utf8' }
      );
    };

    const noPath = checkImport();
    if (noPath.status !== 0) {
      const details = [
        `Python SDK import failed.`,
        `pythonBin=${pythonBin}`,
        `error=${(noPath.stderr || noPath.stdout || '').trim()}`,
        `Hint: run 'synap-js-sdk setup --sdk-version <version>' to install maximem-synap from PyPI.`,
      ].join('\n');
      throw new Error(details);
    }

    if (this.options.onLog) {
      this.options.onLog('debug', 'Python SDK import OK from installed package');
    }
  }
}

module.exports = { BridgeManager };
