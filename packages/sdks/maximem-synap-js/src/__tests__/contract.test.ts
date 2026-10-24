import { describe, it, expect } from 'vitest';
import { readFileSync, existsSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { CONTRACT, THRESHOLDS, STOP_WORDS, SUFFIXES } from '../behavior/contract.js';

const here = path.dirname(fileURLToPath(import.meta.url));
const vendored = path.resolve(here, '../behavior/anticipation.json');
const canonical = path.resolve(here, '../../../CONTRACT/behavior/anticipation.json');

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
    // The recall-phrase bypass is gone (23 Oct): no list of phrases decides
    // whether the cache may answer.
    expect('recall_bypass' in CONTRACT).toBe(false);
    expect(STOP_WORDS.size).toBeGreaterThan(50);
    expect(SUFFIXES.length).toBeGreaterThan(0);
  });

  it('keeps suffixes ordered longest-first', () => {
    // stem() returns on first match, so a short suffix ahead of a long one
    // silently changes every stem and therefore every BM25 score.
    for (let i = 1; i < SUFFIXES.length; i++) {
      expect(SUFFIXES[i]!.length).toBeLessThanOrEqual(SUFFIXES[i - 1]!.length);
    }
  });
});
