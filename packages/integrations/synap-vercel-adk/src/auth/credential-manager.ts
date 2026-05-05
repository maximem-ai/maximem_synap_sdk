import fs from 'fs';
import path from 'path';
import https from 'https';
import type { Credentials } from '../types.js';

const DEFAULT_BASE_URL = 'https://synap-cloud-prod.maximem.ai';

export class CredentialManager {
  private cached: Credentials | null = null;

  constructor(
    private readonly credentialsPath?: string,
    private readonly baseUrl: string = DEFAULT_BASE_URL,
  ) {}

  async load(explicitApiKey?: string, bootstrapToken?: string): Promise<Credentials> {
    if (this.cached) return this.cached;

    // 1. Direct API key passed in constructor options
    if (explicitApiKey) {
      this.cached = { api_key: explicitApiKey, client_id: this.loadClientId(), instance_id: '' };
      return this.cached;
    }

    // 2. SYNAP_API_KEY env var
    const envKey = process.env['SYNAP_API_KEY'];
    if (envKey) {
      const clientId = process.env['SYNAP_CLIENT_ID'] ?? this.loadClientId();
      this.cached = { api_key: envKey, client_id: clientId, instance_id: '' };
      return this.cached;
    }

    // 3. Stored credentials file (explicit credentialsPath only)
    const fileCreds = this.loadFromFile();
    if (fileCreds) {
      this.cached = fileCreds;
      return this.cached;
    }

    // 4. Bootstrap with one-time token → exchange for API key
    if (bootstrapToken) {
      const creds = await this.bootstrap(bootstrapToken);
      this.cached = creds;
      return this.cached;
    }

    throw new Error(
      'No Synap API key found. Set SYNAP_API_KEY in your environment, ' +
      'or pass apiKey in provider options.'
    );
  }

  private loadFromFile(): Credentials | null {
    if (!this.credentialsPath) return null;

    const credPath = this.credentialsPath;

    if (!fs.existsSync(credPath)) return null;

    try {
      const raw = JSON.parse(fs.readFileSync(credPath, 'utf-8')) as Record<string, string>;
      if (!raw['api_key']) return null;
      return {
        api_key: raw['api_key'],
        client_id: raw['client_id'] ?? '',
        instance_id: raw['instance_id'] ?? '',
      };
    } catch {
      return null;
    }
  }

  private loadClientId(): string {
    const fileCreds = this.loadFromFile();
    return fileCreds?.client_id ?? process.env['SYNAP_CLIENT_ID'] ?? '';
  }

  private async bootstrap(bootstrapToken: string): Promise<Credentials> {
    const url = new URL('/api/v1/keys/bootstrap', this.baseUrl);
    const body = JSON.stringify({ bootstrap_token: bootstrapToken, instance_id: '' });

    const data = await this.httpPost(url.toString(), body);
    const creds: Credentials = {
      api_key: data['api_key'] as string,
      client_id: data['client_id'] as string,
      instance_id: '',
    };

    // Persist to disk for future startups
    this.persistToFile(creds);
    return creds;
  }

  private persistToFile(creds: Credentials): void {
    if (!this.credentialsPath) return;
    try {
      fs.mkdirSync(path.dirname(this.credentialsPath), { recursive: true });
      fs.writeFileSync(this.credentialsPath, JSON.stringify(creds, null, 2), { mode: 0o600 });
    } catch {
      // Non-fatal — credentials are still usable in memory
    }
  }

  private httpPost(url: string, body: string): Promise<Record<string, unknown>> {
    return new Promise((resolve, reject) => {
      const req = https.request(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'Content-Length': Buffer.byteLength(body) },
      }, (res) => {
        let raw = '';
        res.on('data', (chunk: Buffer) => { raw += chunk.toString(); });
        res.on('end', () => {
          if ((res.statusCode ?? 0) >= 400) {
            reject(new Error(`Bootstrap failed (HTTP ${res.statusCode}): ${raw}`));
            return;
          }
          try { resolve(JSON.parse(raw) as Record<string, unknown>); }
          catch { reject(new Error(`Bootstrap response parse error: ${raw}`)); }
        });
      });
      req.on('error', reject);
      req.write(body);
      req.end();
    });
  }
}
