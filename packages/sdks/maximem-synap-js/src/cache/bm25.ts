/**
 * BM25 scorer for the anticipation cache.
 *
 * A faithful port of `maximem_synap/cache/bm25.py`. Every tuned value comes
 * from the shared behavior contract, so the two implementations cannot drift
 * on parameters. What they *can* still drift on is arithmetic, which is what
 * the shared conformance corpus is for.
 */

import { BM25_PARAMS, STOP_WORDS, SUFFIXES, ALIAS_TOKEN_SOURCE } from '../behavior/contract.js';

const MIN_STEM_LEN = BM25_PARAMS.min_stem_len;
const MIN_TOKEN_LEN = BM25_PARAMS.min_token_len;

export function stem(word: string): string {
  // Order is load-bearing: longest suffix first, return on first match.
  for (const suffix of SUFFIXES) {
    if (word.endsWith(suffix) && word.length - suffix.length >= MIN_STEM_LEN) {
      return word.slice(0, word.length - suffix.length);
    }
  }
  return word;
}

/**
 * An alias label, as it appears in text: `[[PERSON_PHONE_h2n7v5cx8m0d]]`.
 *
 * Kept whole rather than split into `person`, `phone` and the suffix, because
 * the first two are boilerplate shared by every label of that field type and
 * would make two different people's memories score alike.
 *
 * Mirrors `synap/cloud/shared/tokenizer.py`. Keep the two in step.
 */
function aliasTokenRe(): RegExp {
  // Constructed per call because `g` regexes carry `lastIndex`, and this one
  // is used with replace(). A module-level shared instance would produce
  // different results on alternating calls.
  return new RegExp(ALIAS_TOKEN_SOURCE, 'g');
}

const WORD_RE = /[a-z0-9]+/g;

/** Tokenize, lowercase, remove stop words, and stem. Alias labels survive whole. */
export function tokenize(text: string | null | undefined): string[] {
  if (!text) return [];

  const labels: string[] = [];
  const remainder = text.replace(aliasTokenRe(), (_match, captured: string) => {
    labels.push(captured.toLowerCase());
    return ' ';
  });

  const words = remainder.toLowerCase().match(WORD_RE) ?? [];
  const out = labels.slice();
  for (const w of words) {
    if (w.length >= MIN_TOKEN_LEN && !STOP_WORDS.has(w)) out.push(stem(w));
  }
  return out;
}

/** Okapi BM25 scorer over a small in-memory corpus. */
export class BM25 {
  static readonly IDF_FLOOR = BM25_PARAMS.idf_floor;

  readonly k1: number;
  readonly b: number;
  readonly corpus: readonly (readonly string[])[];
  readonly docCount: number;
  readonly avgdl: number;
  readonly docLens: readonly number[];
  readonly idf: ReadonlyMap<string, number>;

  constructor(
    corpus: readonly (readonly string[])[],
    k1: number = BM25_PARAMS.k1,
    b: number = BM25_PARAMS.b,
  ) {
    this.k1 = k1;
    this.b = b;
    this.corpus = corpus;
    this.docCount = corpus.length;
    this.docLens = corpus.map((doc) => doc.length);
    // Python falls back to 1.0 rather than dividing by zero on an empty corpus.
    this.avgdl = this.docCount
      ? this.docLens.reduce((a, n) => a + n, 0) / this.docCount
      : 1.0;

    const idf = new Map<string, number>();
    const allTerms = new Set<string>();
    for (const doc of corpus) for (const t of doc) allTerms.add(t);
    for (const term of allTerms) {
      let df = 0;
      for (const doc of corpus) if (doc.includes(term)) df++;
      const raw = Math.log((this.docCount - df + 0.5) / (df + 0.5) + 1.0);
      idf.set(term, Math.max(raw, BM25.IDF_FLOOR));
    }
    this.idf = idf;
  }

  score(queryTokens: readonly string[], docIdx: number): number {
    const doc = this.corpus[docIdx];
    if (doc === undefined) return 0;
    const dl = this.docLens[docIdx] ?? 0;

    const tfMap = new Map<string, number>();
    for (const t of doc) tfMap.set(t, (tfMap.get(t) ?? 0) + 1);

    let s = 0;
    for (const term of queryTokens) {
      const tf = tfMap.get(term);
      if (tf === undefined) continue;
      const idf = this.idf.get(term) ?? BM25.IDF_FLOOR;
      const numerator = tf * (this.k1 + 1);
      const denominator = tf + this.k1 * (1 - this.b + (this.b * dl) / this.avgdl);
      s += (idf * numerator) / denominator;
    }
    return s;
  }

  scores(queryTokens: readonly string[]): number[] {
    return this.corpus.map((_, i) => this.score(queryTokens, i));
  }
}
