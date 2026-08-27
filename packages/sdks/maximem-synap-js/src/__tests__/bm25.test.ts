import { describe, it, expect } from 'vitest';
import { readFileSync, existsSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { BM25, tokenize, stem } from '../cache/bm25.js';

const here = path.dirname(fileURLToPath(import.meta.url));
const corpusPath = path.resolve(here, '../../../CONTRACT/conformance/bm25.json');

describe('BM25 + tokenizer', () => {
  it('stems only when the remainder is long enough', () => {
    expect(stem('running')).toBe('runn');
    expect(stem('ties')).toBe('ties');      // remainder < min_stem_len
    expect(stem('address')).toBe('addres');
    expect(stem('xy')).toBe('xy');
  });

  it('keeps alias labels whole', () => {
    expect(tokenize('call [[PERSON_PHONE_h2n7v5cx8m0d]] now'))
      .toContain('person_phone_h2n7v5cx8m0d');
  });

  it('handles an empty corpus without dividing by zero', () => {
    const bm = new BM25([]);
    expect(bm.avgdl).toBe(1.0);
    expect(bm.scores(['anything'])).toEqual([]);
    expect(Number.isFinite(bm.avgdl)).toBe(true);
  });

  it('scores an out-of-range index as 0 rather than throwing', () => {
    expect(new BM25([['a']]).score(['a'], 99)).toBe(0);
  });

  // ── Cross-language parity ───────────────────────────────────────────────────
  // This is the check that matters. Both SDKs implement BM25 independently, so
  // the shared contract only guarantees identical *parameters*. Identical
  // *arithmetic* is guaranteed here or not at all.
  describe('matches the Python implementation exactly', () => {
    if (!existsSync(corpusPath)) return;
    const golden = JSON.parse(readFileSync(corpusPath, 'utf8')) as {
      texts: string[];
      tokens: string[][];
      queries: { query: string; tokens: string[]; scores: number[] }[];
      idf_sample: Record<string, number>;
      avgdl: number;
      doc_count: number;
    };

    const corpusTokens = golden.texts.map((t) => tokenize(t));
    const bm = new BM25(corpusTokens);

    it.each(golden.texts.map((t, i) => ({ i, text: t })))(
      'tokenizes text[$i] identically',
      ({ i, text }) => {
        expect(tokenize(text)).toEqual(golden.tokens[i]);
      },
    );

    it('derives the same corpus statistics', () => {
      expect(bm.docCount).toBe(golden.doc_count);
      expect(round10(bm.avgdl)).toBe(golden.avgdl);
    });

    it('computes the same IDF values', () => {
      for (const [term, expected] of Object.entries(golden.idf_sample)) {
        expect(round10(bm.idf.get(term)!)).toBe(expected);
      }
    });

    it.each(golden.queries.map((q, i) => ({ i, q: q.query })))(
      'scores query[$i] "$q" identically',
      ({ i }) => {
        const g = golden.queries[i]!;
        expect(tokenize(g.query)).toEqual(g.tokens);
        expect(bm.scores(g.tokens).map(round10)).toEqual(g.scores);
      },
    );
  });
});

function round10(n: number): number {
  return Number(n.toFixed(10));
}
