# Use-Case Markdown — the instance's brief

When you create an Instance (Dashboard step 3, see `reference/dashboard-setup.md`) you
upload a short Markdown file describing what the agent does. Synap reads it to generate the
instance's **Memory Architecture (MACA)** — what kinds of facts, preferences, episodes, and
entities it should extract and how to weight them at retrieval.

A good brief makes extraction sharper. A vague one makes memory noisy. It's worth 5 minutes.

- Accepted formats: `.md`, `.markdown`, `.txt`, ≤512 KB.
- The Dashboard form has a **"Download Template"** button — this file is the same idea, so
  you can fill it in here and the developer uploads it.
- You can refine the MACA later in the Dashboard; this file just seeds it.

## Template

```markdown
# Agent Objective
<One or two sentences: what is this agent for, and what does "remembering well" mean for it?
 e.g. "A B2B support copilot for our analytics product. It should remember each user's
 environment, past tickets, and stated preferences so it never re-asks setup questions.">

# Target Users
<Who talks to it, and how are they organized? Name the tenancy explicitly.
 e.g. "Engineers at customer companies. Multi-tenant (B2B): each user belongs to a
 customer org; memory must never leak across orgs.">

# Task Examples
<3–6 concrete things a user says or asks, spanning the memory you care about. Cover
 facts, preferences, and recurring episodes.>
- "We run Postgres 15 on AWS, not the managed cloud version."   (durable fact about env)
- "Always give me Terraform, never the console steps."          (preference)
- "Same timeout error as last week's incident."                 (episode / continuity)
- "Add my teammate Dana to this workspace."                     (entity / relationship)

# Out of Scope (optional)
<Anything the agent should NOT try to remember — e.g. transient values, secrets,
 PII you don't want persisted.>
```

## Tips

- Be specific about **tenancy** (B2C vs B2B) — it must match the instance's User
  Relationship setting and the IDs you pass at call time.
- Prefer real example utterances over abstract categories; the extractor calibrates to them.
- If the agent is multi-domain, list the domains; the MACA can weight them differently.

---
> **Accurate as of** `maximem-synap` 0.2.6 (Python) · `@maximem/synap-js-sdk` 0.3.0 (JS) — verified 2026-06-20.
> MACA / memory-architecture detail: https://docs.maximem.ai/concepts/customized-memory-architectures
