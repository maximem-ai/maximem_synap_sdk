import { describe, it, expect } from 'vitest';
import { readFileSync, existsSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { CONTRACT, RECALL_PATTERNS, THRESHOLDS, isRecallQuery, STOP_WORDS, SUFFIXES } from '../behavior/contract.js';

const here = path.dirname(fileURLToPath(import.meta.url));
const vendored = path.resolve(here, '../behavior/anticipation.json');
const canonical = path.resolve(here, '../../../CONTRACT/behavior/anticipation.json');
const corpusPath = path.resolve(here, '../../../CONTRACT/conformance/recall_queries.json');

describe('behavior contract', () => {
  // Mirrors the established proto-parity pattern: the vendored copy must match
  // the canonical one, and the check no-ops outside the monorepo (a published
  // tarball has no canonical file to compare against).
  it('vendored copy is byte-identical to the canonical contract', () => {
    if (!existsSync(canonical)) return;
    expect(readFileSync(vendored, 'utf8')).toBe(readFileSync(canonical, 'utf8'));
  });

  it('loads the tuned values', () => {
    expect(THRESHOLDS.bm25).toBe(1.5);
    expect(THRESHOLDS.novelTerm).toBe(0.45);
    expect(THRESHOLDS.minCorpusForNovelGate).toBe(200);
    expect(RECALL_PATTERNS.length).toBe(CONTRACT.recall_bypass.patterns.length);
    expect(RECALL_PATTERNS.length).toBeGreaterThan(0);
    expect(STOP_WORDS.size).toBeGreaterThan(50);
    expect(SUFFIXES.length).toBeGreaterThan(0);
  });

  it('never compiles a global regex', () => {
    // A `g` regex keeps lastIndex between calls, so a shared instance matches
    // on one query then misses on the next. Intermittent and traffic-dependent.
    for (const p of RECALL_PATTERNS) expect(p.global).toBe(false);
  });

  it('keeps suffixes ordered longest-first', () => {
    // stem() returns on first match, so a short suffix ahead of a long one
    // silently changes every stem and therefore every BM25 score.
    for (let i = 1; i < SUFFIXES.length; i++) {
      expect(SUFFIXES[i]!.length).toBeLessThanOrEqual(SUFFIXES[i - 1]!.length);
    }
  });

  // The golden corpus is the actual cross-language guarantee: Python runs the
  // identical file and must produce the identical verdicts.
  describe('recall-bypass conformance corpus', () => {
    if (!existsSync(corpusPath)) return;
    const cases = JSON.parse(readFileSync(corpusPath, 'utf8')).cases as {
      query: string; expected: boolean; why: string;
    }[];

    it.each(cases)('$why -- "$query"', ({ query, expected }) => {
      expect(isRecallQuery([query])).toBe(expected);
    });

    it('handles null, empty and mixed arrays like Python', () => {
      expect(isRecallQuery(null)).toBe(false);
      expect(isRecallQuery(undefined)).toBe(false);
      expect(isRecallQuery([])).toBe(false);
      expect(isRecallQuery([null, undefined, ''])).toBe(false);
      expect(isRecallQuery(['book a ride', 'remind me of my address'])).toBe(true);
    });

    it('is stable across repeated calls with the same instance', () => {
      for (let i = 0; i < 3; i++) expect(isRecallQuery(['what is my email'])).toBe(true);
    });
  });
});
