import type { CachedBundle, FetchedContext, RawContextItem } from '../types.js';
import { rawItemToContextItem } from '../transform/messages.js';

const DEFAULT_TTL_MS = 1_800_000; // 30 minutes — matches Python SDK (ttl_seconds=1800)
const MAX_ENTRIES = 100;
const BM25_THRESHOLD = 1.5;
const NOVEL_TERM_GATE = 0.45;     // cache miss if >45% query terms are unknown

interface CacheEntry {
  bundle: CachedBundle;
  touchedAt: number;
}

export class AnticipationCache {
  private entries = new Map<string, CacheEntry>();
  private vocabulary = new Set<string>();

  store(bundle: CachedBundle): void {
    const key = this.bundleKey(bundle);

    // Evict oldest if at cap
    if (this.entries.size >= MAX_ENTRIES && !this.entries.has(key)) {
      const oldest = [...this.entries.entries()]
        .sort(([, a], [, b]) => a.touchedAt - b.touchedAt)[0];
      if (oldest) this.entries.delete(oldest[0]);
    }

    this.entries.set(key, { bundle, touchedAt: Date.now() });

    // Index vocabulary from all item contents
    for (const list of Object.values(bundle.itemsByType)) {
      for (const item of list) {
        for (const token of this.tokenize(item.content)) {
          this.vocabulary.add(token);
        }
      }
    }
    for (const kw of bundle.searchKeywords) {
      this.vocabulary.add(kw.toLowerCase());
    }
  }

  lookup(opts: {
    userId?: string;
    customerId?: string;
    conversationId?: string;
    searchQuery?: string[];
  }): FetchedContext | null {
    const now = Date.now();
    this.evictExpired(now);

    const candidates = [...this.entries.values()].filter(({ bundle }) =>
      this.scopeMatches(bundle, opts)
    );

    if (candidates.length === 0) return null;

    // Score each candidate against the query
    const query = (opts.searchQuery ?? []).join(' ');
    const queryTerms = this.tokenize(query);

    if (queryTerms.length > 0) {
      const novelRatio = queryTerms.filter(t => !this.vocabulary.has(t)).length / queryTerms.length;
      if (novelRatio > NOVEL_TERM_GATE) return null;
    }

    let best: CacheEntry | null = null;
    let bestScore = 0;

    for (const entry of candidates) {
      const score = queryTerms.length > 0
        ? this.bm25Score(queryTerms, entry.bundle)
        : 1.0; // no query → take most recent

      if (score > bestScore || (score === bestScore && entry.touchedAt > (best?.touchedAt ?? 0))) {
        bestScore = score;
        best = entry;
      }
    }

    if (!best || (queryTerms.length > 0 && bestScore < BM25_THRESHOLD)) return null;

    // Touch LRU
    best.touchedAt = now;

    return this.bundleToContext(best.bundle);
  }

  lookupUserSummary(userId: string): CachedBundle | null {
    const now = Date.now();
    this.evictExpired(now);

    for (const { bundle } of this.entries.values()) {
      if (bundle.bundleType === 'user_summary' && bundle.userId === userId) {
        return bundle;
      }
    }
    return null;
  }

  invalidateConversation(conversationId: string): void {
    for (const [key, { bundle }] of this.entries) {
      if (bundle.conversationId === conversationId) {
        this.entries.delete(key);
      }
    }
  }

  size(): number { return this.entries.size; }

  private bundleKey(bundle: CachedBundle): string {
    return `${bundle.bundleType}:${bundle.userId}:${bundle.customerId}:${bundle.conversationId}`;
  }

  private scopeMatches(bundle: CachedBundle, opts: {
    userId?: string;
    customerId?: string;
    conversationId?: string;
  }): boolean {
    if (opts.userId && bundle.userId && bundle.userId !== opts.userId) return false;
    if (opts.customerId && bundle.customerId && bundle.customerId !== opts.customerId) return false;
    if (opts.conversationId && bundle.conversationId && bundle.conversationId !== opts.conversationId) return false;
    return true;
  }

  private evictExpired(now: number): void {
    for (const [key, { bundle, touchedAt }] of this.entries) {
      if (now - touchedAt > (bundle.ttl || DEFAULT_TTL_MS)) {
        this.entries.delete(key);
      }
    }
  }

  private bm25Score(queryTerms: string[], bundle: CachedBundle): number {
    const k1 = 1.5;
    const b = 0.75;

    // Collect all text from bundle items
    const docs: string[] = [];
    for (const list of Object.values(bundle.itemsByType)) {
      for (const item of list) docs.push(item.content);
    }
    docs.push(...bundle.searchKeywords);
    docs.push(...bundle.searchQueries);

    const corpus = docs.join(' ');
    const docTerms = this.tokenize(corpus);
    const docLength = docTerms.length;
    const avgDocLength = 50; // approximate

    let score = 0;
    for (const term of queryTerms) {
      const tf = docTerms.filter(t => t === term).length;
      if (tf === 0) continue;
      const idf = Math.log((MAX_ENTRIES - 1 + 0.5) / (1 + 0.5) + 1);
      const tfNorm = (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * (docLength / avgDocLength)));
      score += idf * tfNorm;
    }
    return score;
  }

  private tokenize(text: string): string[] {
    return text.toLowerCase().split(/\W+/).filter(t => t.length > 2);
  }

  private bundleToContext(bundle: CachedBundle): FetchedContext {
    const servedItemIds: string[] = [];
    for (const list of Object.values(bundle.itemsByType)) {
      for (const item of list) {
        if (item.item_id) servedItemIds.push(item.item_id);
      }
    }

    return {
      facts: (bundle.itemsByType['facts'] ?? []).map(rawItemToContextItem),
      preferences: (bundle.itemsByType['preferences'] ?? []).map(rawItemToContextItem),
      episodes: (bundle.itemsByType['episodes'] ?? []).map(rawItemToContextItem),
      emotions: (bundle.itemsByType['emotions'] ?? []).map(rawItemToContextItem),
      temporalEvents: (bundle.itemsByType['temporal_events'] ?? []).map(rawItemToContextItem),
      conversationContext: bundle.conversationContext ? {
        summary: bundle.conversationContext.summary ?? null,
        currentState: this.parseJson(bundle.conversationContext.current_state_json),
        keyExtractions: this.parseJson(bundle.conversationContext.key_extractions_json),
        recentTurns: bundle.conversationContext.recent_turns ?? [],
        compactionId: bundle.conversationContext.compaction_id ?? null,
        conversationId: bundle.conversationContext.conversation_id ?? null,
      } : null,
      source: 'anticipation',
      correlationId: '',
      bundleId: bundle.bundleId,
      servedItemIds,
      sourceBundleIds: bundle.sourceBundleIds.length ? bundle.sourceBundleIds : (bundle.bundleId ? [bundle.bundleId] : []),
      totalTokens: bundle.totalTokens,
      assemblySource: 'anticipation_cache',
      cacheHit: true,
    };
  }

  private parseJson(s: string): Record<string, unknown> {
    try { return JSON.parse(s) as Record<string, unknown>; }
    catch { return {}; }
  }
}

// Re-export for convenience
export type { CachedBundle };
