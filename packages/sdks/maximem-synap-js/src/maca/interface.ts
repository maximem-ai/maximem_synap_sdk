/**
 * MACA plugin namespace (Synap Core). Mirrors `MacaInterface` in the Python SDK.
 *
 * A Synap Core server can have Synap Cloud write its memory config from a short
 * description of your use case. The server holds the Synap Cloud key; this
 * namespace only starts the job and reads it back. Synap Cloud has no such
 * route and answers 404.
 */

import { InvalidInputError } from '../errors.js';
import type { HttpTransport } from '../transport/http.js';

type BlobSource = NonNullable<ConstructorParameters<typeof Blob>[0]>[number];

/** What a job can end as. Anything else is still running. */
export const MACA_FINAL_STATUSES = ['succeeded', 'invalid', 'failed'] as const;

export type MacaRelationship = 'b2c' | 'b2b';

export interface MacaJob {
  job_id: string;
  status: string;
  message: string;
  created_at: string | null;
  updated_at: string | null;
  removed_fields: string[];
  problems: string[];
  config_file: string | null;
  /** The generated config, once the job has ended with `succeeded` or `invalid`. */
  yaml: string | null;
}

export interface MacaGenerateOptions {
  /** The use-case file (UTF-8 text, at most 256 KB), uploaded as is. */
  file_path?: string;
  filePath?: string;
  /** `b2c`: each user is their own customer. `b2b`: users sit under a customer. */
  user_relationship?: MacaRelationship;
  userRelationship?: MacaRelationship;
}

/** The server refuses a larger use-case file. */
export const MACA_MAX_FILE_BYTES = 256 * 1024;

export interface MacaWaitOptions {
  timeout_seconds?: number;
  timeoutSeconds?: number;
  poll_interval_seconds?: number;
  pollIntervalSeconds?: number;
}

export interface MacaNamespace {
  generate(options: MacaGenerateOptions): Promise<MacaJob>;
  get_job(jobId: string): Promise<MacaJob>;
  wait_for_completion(jobId: string, options?: MacaWaitOptions): Promise<MacaJob>;
}

function toJob(raw: Record<string, unknown>): MacaJob {
  return {
    job_id: String(raw['job_id']),
    status: String(raw['status']),
    message: typeof raw['message'] === 'string' ? raw['message'] : '',
    created_at: typeof raw['created_at'] === 'string' ? raw['created_at'] : null,
    updated_at: typeof raw['updated_at'] === 'string' ? raw['updated_at'] : null,
    removed_fields: Array.isArray(raw['removed_fields']) ? (raw['removed_fields'] as string[]) : [],
    problems: Array.isArray(raw['problems']) ? (raw['problems'] as string[]) : [],
    config_file: typeof raw['config_file'] === 'string' ? raw['config_file'] : null,
    yaml: typeof raw['yaml'] === 'string' ? raw['yaml'] : null,
  };
}

export function createMacaNamespace(transport: HttpTransport): MacaNamespace {
  const namespace: MacaNamespace = {
    async generate(options) {
      const filePath = options.file_path ?? options.filePath;
      const relationship = options.user_relationship ?? options.userRelationship;
      if (relationship !== 'b2c' && relationship !== 'b2b') {
        throw new InvalidInputError("user_relationship must be 'b2c' or 'b2b'");
      }
      if (typeof filePath !== 'string' || !filePath) throw new InvalidInputError('file_path is required');

      // Lazy, so importing the SDK on Edge or in a Worker stays safe: a top-level `node:fs` import fails the BUILD there.
      const { readFile, stat } = await import('node:fs/promises');
      const { basename } = await import('node:path');
      let size: number;
      try {
        size = (await stat(filePath)).size;
      } catch {
        throw new InvalidInputError(`file_path does not point to a readable file: ${filePath}`);
      }
      if (size === 0) throw new InvalidInputError('The use-case file is empty');
      if (size > MACA_MAX_FILE_BYTES) {
        throw new InvalidInputError(`The use-case file is larger than ${MACA_MAX_FILE_BYTES / 1024} KB`);
      }
      const bytes = await readFile(filePath);

      const form = new FormData();
      form.set('user_relationship', relationship);
      form.set('file', new Blob([bytes as unknown as BlobSource]), basename(filePath));
      // Non-idempotent by contract: a second call starts a second job (the server refuses it with 409).
      return toJob(await transport.request<Record<string, unknown>>('maca_generate', { body: form, rawBody: true }));
    },

    async get_job(jobId) {
      if (!jobId) throw new InvalidInputError('job_id is required');
      return toJob(await transport.request<Record<string, unknown>>('maca_job', { pathParams: { job_id: jobId } }));
    },

    async wait_for_completion(jobId, options = {}) {
      const timeout = (options.timeout_seconds ?? options.timeoutSeconds ?? 3000) * 1000;
      const interval = (options.poll_interval_seconds ?? options.pollIntervalSeconds ?? 15) * 1000;
      const start = Date.now();
      while (Date.now() - start < timeout) {
        const job = await namespace.get_job(jobId);
        if ((MACA_FINAL_STATUSES as readonly string[]).includes(job.status)) return job;
        await new Promise((resolve) => setTimeout(resolve, interval));
      }
      throw new Error(`MACA job ${jobId} did not finish within ${timeout / 1000}s`);
    },
  };
  return namespace;
}
