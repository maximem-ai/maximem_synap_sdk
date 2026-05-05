import type { LanguageModelV1Prompt, LanguageModelV1Message } from '@ai-sdk/provider';
import type { ContextItem, FetchedContext, RawContextItem } from '../types.js';

// ─── Raw proto item → typed ContextItem ──────────────────────────────────────

export function rawItemToContextItem(raw: RawContextItem): ContextItem {
  return {
    id: raw.item_id ?? '',
    content: raw.content ?? '',
    contextType: raw.context_type ?? '',
    confidence: raw.confidence ?? 0,
    source: raw.source ?? '',
    eventDate: raw.event_date || undefined,
    validUntil: raw.valid_until || undefined,
    temporalCategory: raw.temporal_category || undefined,
  };
}

// ─── Build system prompt block from fetched context ───────────────────────────

export function buildContextSystemBlock(ctx: FetchedContext): string {
  const sections: string[] = [];

  if (ctx.conversationContext?.summary) {
    sections.push(`## Conversation Summary\n${ctx.conversationContext.summary}`);
  }

  if (ctx.conversationContext?.recentTurns.length) {
    const turns = ctx.conversationContext.recentTurns
      .slice(-6)
      .map(t => `${t.role}: ${t.content}`)
      .join('\n');
    sections.push(`## Recent Conversation\n${turns}`);
  }

  if (ctx.facts.length) {
    sections.push(`## Known Facts\n${ctx.facts.map(f => `- ${f.content}`).join('\n')}`);
  }

  if (ctx.preferences.length) {
    sections.push(`## User Preferences\n${ctx.preferences.map(p => `- ${p.content}`).join('\n')}`);
  }

  if (ctx.episodes.length) {
    sections.push(`## Past Episodes\n${ctx.episodes.map(e => `- ${e.content}`).join('\n')}`);
  }

  if (ctx.emotions.length) {
    sections.push(`## Emotional Context\n${ctx.emotions.map(e => `- ${e.content}`).join('\n')}`);
  }

  if (ctx.temporalEvents.length) {
    sections.push(`## Upcoming / Temporal\n${ctx.temporalEvents.map(t => `- ${t.content}`).join('\n')}`);
  }

  if (sections.length === 0) return '';
  return `<synap_context>\n${sections.join('\n\n')}\n</synap_context>`;
}

// ─── Inject context block into the prompt ─────────────────────────────────────

export function injectContextIntoPrompt(
  prompt: LanguageModelV1Prompt,
  ctx: FetchedContext,
): LanguageModelV1Prompt {
  const block = buildContextSystemBlock(ctx);
  if (!block) return prompt;

  const hasSystem = prompt.some(m => m.role === 'system');

  if (hasSystem) {
    return prompt.map((m): LanguageModelV1Message => {
      if (m.role !== 'system') return m;
      // system content is a plain string in LanguageModelV1Message
      return { role: 'system', content: `${block}\n\n${m.content}` };
    });
  }

  const systemMsg: LanguageModelV1Message = { role: 'system', content: block };
  return [systemMsg, ...prompt];
}

// ─── Extract a search query from the latest user message ─────────────────────

export function extractSearchQuery(prompt: LanguageModelV1Prompt): string[] {
  const userMessages = prompt.filter(m => m.role === 'user');
  const last = userMessages[userMessages.length - 1];
  if (!last || last.role !== 'user') return [];

  const text = last.content
    .filter((p): p is { type: 'text'; text: string } => p.type === 'text')
    .map(p => p.text)
    .join(' ')
    .trim();

  return text ? [text.slice(0, 512)] : [];
}

// ─── Convert prompt messages to plain text for memory writing ─────────────────

export function promptToTranscript(prompt: LanguageModelV1Prompt): Array<{ role: string; content: string }> {
  return prompt
    .filter((m): m is Extract<LanguageModelV1Message, { role: 'user' | 'assistant' }> =>
      m.role === 'user' || m.role === 'assistant'
    )
    .map(m => ({
      role: m.role,
      content: m.content
        .filter((p): p is { type: 'text'; text: string } => p.type === 'text')
        .map(p => p.text)
        .join(''),
    }))
    .filter(m => m.content.length > 0);
}
