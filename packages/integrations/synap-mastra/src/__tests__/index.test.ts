// Tests for the public surface of synap-mastra (index.ts re-exports).
// Asserts that every documented export is present and of the expected kind.
// No SDK or network calls — purely structural.

import { describe, it, expect } from 'vitest';
import * as pkg from '../index.js';

describe('@maximem/synap-mastra public surface', () => {
  // ── Classes ──────────────────────────────────────────────────────────────

  it('exports SynapMemory as a constructor', () => {
    expect(typeof pkg.SynapMemory).toBe('function');
    // Mastra base class extends MastraMemory; ensure prototype chain is intact
    expect(pkg.SynapMemory.prototype).toBeDefined();
  });

  // ── Tool factories ────────────────────────────────────────────────────────

  it('exports synapSearchTool as a function', () => {
    expect(typeof pkg.synapSearchTool).toBe('function');
  });

  it('exports synapStoreTool as a function', () => {
    expect(typeof pkg.synapStoreTool).toBe('function');
  });

  // ── Short-term functions ──────────────────────────────────────────────────

  it('exports fetchSynapShortTerm as a function', () => {
    expect(typeof pkg.fetchSynapShortTerm).toBe('function');
  });

  it('exports buildSynapShortTermSystem as a function', () => {
    expect(typeof pkg.buildSynapShortTermSystem).toBe('function');
  });

  it('exports synapShortTermInstructions as a function', () => {
    expect(typeof pkg.synapShortTermInstructions).toBe('function');
  });

  // ── No unexpected extras (surface stability) ──────────────────────────────

  it('does not export unexpected runtime values', () => {
    const EXPECTED_RUNTIME_EXPORTS = new Set([
      'SynapMemory',
      'synapSearchTool',
      'synapStoreTool',
      'fetchSynapShortTerm',
      'buildSynapShortTermSystem',
      'synapShortTermInstructions',
    ]);
    const actual = Object.keys(pkg).filter(
      (k) => typeof (pkg as Record<string, unknown>)[k] !== 'undefined',
    );
    for (const name of actual) {
      expect(EXPECTED_RUNTIME_EXPORTS.has(name), `Unexpected export: ${name}`).toBe(true);
    }
  });
});
