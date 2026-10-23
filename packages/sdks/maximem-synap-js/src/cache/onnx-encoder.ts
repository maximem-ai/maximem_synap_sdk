/**
 * The local embedding model, Node only: the same int8 ONNX file the Python SDK
 * runs (python/maximem_synap/cache/encoders.py, OnnxEncoder), run the same way:
 * the tokenizer, the model, mean pooling over the attention mask, L2 norm. Both
 * SDKs therefore produce the same vectors, and the thresholds tuned on them in
 * CONTRACT/behavior/anticipation.json (`semantic_match`) hold for both.
 *
 * Used by history search after a compaction and by the anticipation cache to
 * match fetch queries to prefetched items by meaning.
 *
 * `onnxruntime-node` and `@huggingface/tokenizers` are optionalDependencies,
 * reached only through the lazy `import()`s below and only from Node: importing
 * this SDK on Edge or in a browser never loads them (bundle-hygiene.test.ts).
 * When either is missing, `loadOnnxEncoder` returns null and the SDK matches by
 * words, as before.
 *
 * The files are downloaded once, pinned to a revision, into
 * `~/.synap/models/onnx/<repo>/<revision>/` (SYNAP_MODELS_DIR overrides the
 * root; HF_ENDPOINT points at a mirror). For an air-gapped machine, copy the
 * three files there beforehand.
 */
import type { Encoder } from './history-search.js';

/** The model's max_seq_length, as in Python. */
const MAX_TOKENS = 128;
const FILES = (onnxFile: string) => ['tokenizer.json', 'tokenizer_config.json', onnxFile];

export interface ModelRef {
  repo: string;
  revision: string;
  file: string;
}

/** `onnx:<repo>@<revision>/<file>`, or null for any other kind. */
export function parseModelId(model: string): ModelRef | null {
  const m = /^onnx:([^@]+)@([0-9a-f]+)\/(.+)$/.exec(model);
  return m ? { repo: m[1]!, revision: m[2]!, file: m[3]! } : null;
}

/** True when this process can run the model at all (Node, not Edge or a browser). */
export function canRunLocalModel(): boolean {
  const p = (globalThis as { process?: { versions?: { node?: string } } }).process;
  return typeof p?.versions?.node === 'string';
}

async function ensureFiles(ref: ModelRef, modelsDir: string | undefined): Promise<string> {
  const fs = await import('node:fs/promises');
  const path = await import('node:path');
  const os = await import('node:os');
  const env = (globalThis as { process?: { env?: Record<string, string | undefined> } }).process?.env ?? {};
  const root = modelsDir ?? env['SYNAP_MODELS_DIR'] ?? path.join(os.homedir(), '.synap', 'models');
  const dir = path.join(root, 'onnx', ...ref.repo.split('/'), ref.revision);
  const endpoint = (env['HF_ENDPOINT'] ?? 'https://huggingface.co').replace(/\/+$/, '');
  for (const file of FILES(ref.file)) {
    const target = path.join(dir, ...file.split('/'));
    try {
      await fs.access(target);
      continue;
    } catch {
      /* not downloaded yet */
    }
    await fs.mkdir(path.dirname(target), { recursive: true });
    const res = await fetch(`${endpoint}/${ref.repo}/resolve/${ref.revision}/${file}`);
    if (!res.ok) throw new Error(`model download failed: ${file}: HTTP ${res.status}`);
    // Written aside and renamed, so an interrupted download is never mistaken
    // for a complete file on the next start.
    const partial = `${target}.partial-${Date.now()}`;
    await fs.writeFile(partial, new Uint8Array(await res.arrayBuffer()));
    await fs.rename(partial, target);
  }
  return dir;
}

/**
 * Load the model, downloading it on first use. Returns null when the optional
 * packages are not installed, this is not Node, or anything fails: the caller
 * then matches by words, as it did without a model.
 */
export async function loadOnnxEncoder(
  model: string,
  options: { modelsDir?: string } = {},
): Promise<Encoder | null> {
  const ref = parseModelId(model);
  if (!ref || !canRunLocalModel()) return null;
  try {
    const [ort, tok] = await Promise.all([
      import(/* @vite-ignore */ 'onnxruntime-node'),
      import(/* @vite-ignore */ '@huggingface/tokenizers'),
    ]);
    const dir = await ensureFiles(ref, options.modelsDir);
    const fs = await import('node:fs/promises');
    const path = await import('node:path');
    const os = await import('node:os');
    const read = async (f: string) => JSON.parse(await fs.readFile(path.join(dir, ...f.split('/')), 'utf8'));
    const tokenizer = new tok.Tokenizer(await read('tokenizer.json'), await read('tokenizer_config.json'));
    const threads = Math.max(1, Math.min(4, Math.floor((os.cpus()?.length ?? 2) / 2)));
    const session = await ort.InferenceSession.create(path.join(dir, ...ref.file.split('/')), {
      intraOpNumThreads: threads,
      executionProviders: ['cpu'],
    });
    const needsTypeIds = session.inputNames.includes('token_type_ids');

    const embedOne = async (text: string): Promise<number[]> => {
      let { ids } = tokenizer.encode(text);
      // <s> A </s>, A cut so the whole fits: what Python's truncation does.
      if (ids.length > MAX_TOKENS) ids = [...ids.slice(0, MAX_TOKENS - 1), ids[ids.length - 1]!];
      const n = ids.length;
      const shape = [1, n];
      const feeds: Record<string, unknown> = {
        input_ids: new ort.Tensor('int64', BigInt64Array.from(ids, (x) => BigInt(x)), shape),
        attention_mask: new ort.Tensor('int64', new BigInt64Array(n).fill(1n), shape),
      };
      if (needsTypeIds) feeds['token_type_ids'] = new ort.Tensor('int64', new BigInt64Array(n), shape);
      const out = await session.run(feeds as never);
      const hidden = out[session.outputNames[0]!]!;
      const data = hidden.data as Float32Array;
      const dim = hidden.dims[2]!;
      const pooled = new Array<number>(dim).fill(0);
      for (let t = 0; t < n; t++) for (let d = 0; d < dim; d++) pooled[d]! += data[t * dim + d]!;
      let norm = 0;
      for (let d = 0; d < dim; d++) {
        pooled[d] = pooled[d]! / n;
        norm += pooled[d]! * pooled[d]!;
      }
      norm = Math.sqrt(norm) || 1;
      return pooled.map((x) => x / norm);
    };

    return {
      model,
      async encode(texts: string[]): Promise<number[][]> {
        const out: number[][] = [];
        for (const t of texts) out.push(await embedOne(t));
        return out;
      },
    };
  } catch {
    return null;
  }
}
