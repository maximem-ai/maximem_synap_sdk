# Use-Case Markdown — the instance's brief

> **Read this next:** [Memory Architecture (MACA) and the use-case file](https://docs.maximem.ai/concepts/memory-architecture.md); [Shaping your MACA](https://docs.maximem.ai/guides/configuring-memory.md). The live docs win over this file.

The use-case file is a short Markdown brief describing what the agent does. Synap reads it to generate the
instance's **Memory Architecture (MACA)**: what kinds of facts, preferences, episodes and entities it extracts
and how it weights them at retrieval. A good brief makes extraction sharper; a vague or invented one makes
memory noisy.

**It cannot be edited after the instance is created.** So the human approves the draft first, and you never
fill a gap with a guess. Write the literal text `not stated` instead.

- Accepted formats: `.md`, `.markdown`, `.txt`, up to 512 KB.
- Get the current template with `instances template --output-file usecase.md`. It is the source of truth for
  section headings; if it differs from the example below, follow the CLI's version.
- Pass it with `instances create ... --from-file usecase.md`.

## The sections (per the docs: three required, five optional)

| Section | Required | May the agent infer it from code? | Rule |
|---|---|---|---|
| Agent Objective | Yes | Yes: README, prompts, entry points | Quote what you found. Ask if the repo does not say what "remembering well" means. |
| Target Users | Yes | **No. Always ask.** | Includes B2C vs B2B. Show the evidence (user model, tenant or org tables, auth) and still ask. Immutable after create; must match `--relationship`. |
| Task Examples | Yes | Partly: real utterances from tests, fixtures, prompts | Use real examples found in the repo, marked as such, and ask the human for the ones that matter. Never invent utterances. |
| Behavioral Guidelines | Optional | Only if the code states them | Otherwise `not stated`. |
| Role Descriptions | Optional | Only if the code states who the client, customer and user are | Otherwise `not stated`; ask if tenancy is unclear. |
| Compliance & Data Sensitivity | Optional | **No. Always ask.** | Do not assume "no PII". Unanswered means `not stated`, and ask again before approval. |
| Memory Priorities | Optional | **No. Always ask.** | The human's words: what to prioritize, deprioritize, or never remember. |
| Additional Context | Optional | Yes, for deployment and integrations visible in code | Otherwise `not stated`. |

A draft with `not stated` in it is fine to show. A draft with an invented answer is not. Show the whole file to
the human and wait for an explicit yes before `instances create`. Section names above come from
`https://docs.maximem.ai/concepts/memory-architecture.md`; `instances template` is the source of truth if they differ.

## Template (shape only; the CLI's template wins)

```markdown
## Agent Objective
<what the agent does and the problem it solves, or `not stated`>

## Target Users
<who talks to it, how they are organized, B2C or B2B, or `not stated`>

## Task Examples
- **User**: "<a real user message>"
  **Agent**: <what the agent should do and remember>

## Behavioral Guidelines
## Role Descriptions
## Compliance & Data Sensitivity
## Memory Priorities
## Additional Context
```

## Tips

- Be specific about **tenancy** (B2C vs B2B) — it must match the instance's User
  Relationship setting and the IDs you pass at call time.
- Prefer real example utterances over abstract categories; the extractor calibrates to them. Real means from the repo or from the human, not made up.
- If the agent is multi-domain, list the domains; the MACA can weight them differently.

---
> **Accurate as of** `maximem-synap` 0.5.1 (Python) · `@maximem/synap-js-sdk` 0.5.1 (JS) — verified 2026-09-25.
> MACA / memory-architecture detail: https://docs.maximem.ai/concepts/memory-architecture
