# Docs map (full)

## Docs map (the docs are the source of truth; `.md` after any URL gives clean markdown)

Prefix every path with `https://docs.maximem.ai`. "Re-visit" says when to open the page again, not just first time.

| Topic | Pages | Re-visit when |
|---|---|---|
| Orientation | /getting-started/quickstart, /getting-started/your-first-memory, /concepts/memory-model-cheat-sheet, /resources/faq, /resources/glossary | Step 0; any term you are unsure of |
| CLI and onboarding | /cli/overview, /cli/commands, /integrations/ai-coding-agents | Before each CLI step |
| SDK install, auth, init | /setup/installation, /setup/authentication, /setup/agent-integration, /sdk/initialization, /sdk/configuration, /sdk-reference/lifecycle/initialize, /sdk-reference/lifecycle/configure, /sdk-reference/lifecycle/shutdown | Before writing init and shutdown; after, compare |
| Scoping, B2C vs B2B | /concepts/memory-scopes, /concepts/scope-ladder, /concepts/agent-topologies, /guides/multi-user-scoping | Before every call that takes `user_id` or `customer_id` |
| Writes | /sdk/ingestion, /concepts/how-ingestion-works, /sdk-reference/memories/create, /sdk-reference/memories/batch-create, /sdk-reference/memories/create-from-file, /sdk-reference/memories/status, /sdk-reference/memories/wait-for-completion, /sdk-reference/conversation/record-message, /sdk-reference/conversation/record-messages-batch, /sdk-reference/conversation/ingest-transcript | Before and after writing each write call |
| Reads | /sdk/context-fetch, /concepts/retrieval-modes, /sdk/response-shapes, /concepts/context-end-to-end, /sdk-reference/context/fetch, /sdk-reference/context/user-fetch, /sdk-reference/context/customer-fetch, /sdk-reference/context/client-fetch, /sdk-reference/conversation-context/fetch, /sdk-reference/conversation-context/get-context-for-prompt | Before and after writing each fetch call |
| Live stream | /concepts/real-time-anticipation, /concepts/event-delivery, /patterns/real-time-anticipation-server, /sdk-reference/instance/listen, /sdk-reference/instance/send-message, /sdk-reference/instance/record-thinking, /sdk-reference/instance/record-tool-call, /sdk-reference/instance/record-tool-result, /sdk-reference/instance/end-session, /sdk-reference/instance/stop-listening | Before wiring the stream; after, re-read and compare |
| MACA and the use-case file | /concepts/memory-architecture, /guides/configuring-memory | Before drafting the use-case |
| Frameworks | /integrations/overview, then /integrations/<name> (names in `reference/frameworks/_index.md`; Vercel is `vercel-ai-sdk`) | Before and after wiring the framework |
| Patterns by agent type | /patterns/multi-tenant-saas, /patterns/rag-user-history, /patterns/replay-history, /patterns/slack-bot, /patterns/voice-agent-livekit, /patterns/graceful-degradation, /cookbook/overview (then the cookbook page closest to the agent: b2b-sdr, b2b-salesforce, consumer-uber, consumer-amazon, personal-ai-coach, personal-ai-companion, support-tier-escalation, voice-concierge, whatsapp-single-campaign) | When the repo matches the pattern; before step 9 for graceful-degradation |
| Privacy and data control | /guides/pii-protection, /guides/erasure, /guides/instance-visibility, /dashboard/pii-and-data-controls | When the human names compliance or sensitive data |
| Production and errors | /guides/production-checklist, /sdk/error-handling, /sdk-reference/errors, /resources/performance-limits | Before step 9, and on any error |
| Credits and instances | /sdk-reference/credits/get-balance, /sdk-reference/credits/estimate | On a credits error |
| Several instances, sharing or isolating memory | /guides/instance-visibility (read before the visibility question), /concepts/agent-topologies | Before asking whether agents see each other's memories |
| Instances (what they are, their states) | /concepts/memory-scopes ("What is an Instance?"), /dashboard/overview ("Managing instances"), /concepts/agent-topologies, /setup/authentication ("Multiple keys per instance"); plain-words glossary: `instances-explained.md` | Before explaining any status word; when the repo has several agents |
| Dashboard (optional view) | /dashboard/overview | Only if the human asks |

