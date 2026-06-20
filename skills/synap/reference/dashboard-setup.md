# Dashboard setup — manual provisioning (no CLI)

Synap has **no CLI**. Instances and API keys are created by hand in the Dashboard
at **https://synap.maximem.ai**. The SDK cannot provision them — it only *uses* a key
that already exists. So before any code runs, walk the developer through these steps,
then **PAUSE and wait for them to paste their `synap_...` key.**

## Steps (developer does these in the browser)

| # | Action in the Dashboard | Result |
|---|---|---|
| 1 | Sign up or accept an invite (Auth0 — email/password or SSO) | An account |
| 2 | **Create a Client** (enter your org name) | `cli_<hex16>` — org-level account that holds instances |
| 3 | **Create an Instance** → set **Name**, **Agent Type**, and **User Relationship (B2C / B2B)**; upload a **Use-Case Markdown** file (Agent Objective, Target Users, Task Examples — "Download Template" in the form; `.md/.markdown/.txt`, ≤512 KB) | `inst_<hex16>` + an auto-generated memory architecture (MACA) |
| 4 | Open the instance → **API Keys** → **Generate API Key** → label it → **copy it now** (shown once) | A `synap_...` key |

A starter Use-Case Markdown file is in `reference/use-case-markdown.md` — offer to fill it
in for the developer based on what their agent does, so they can upload it in step 3.

## Things to surface while they provision

- **B2C vs B2B is chosen once, here, by "User Relationship"** — and it changes how you
  call the SDK later:
  - **B2C** — memory is per end-user. You only reason about `user_id`. (Note: the
    turn-by-turn `record_message` call still requires a `customer_id` argument — on B2C,
    pass the same value as `user_id`. See `reference/core-concepts.md`.)
  - **B2B** — memory is per tenant. Each `user_id` lives under a `customer_id`.
- **Roles:** Owner/Admin can create instances and keys; Member is read-only.
- **The instance is resolved from the API key** — you do *not* pass an instance ID on each
  call. (You may optionally set `SYNAP_INSTANCE_ID` to pin one, but it's not required.)
- **One key = one instance.** Use separate instances (and keys) for staging vs production.

## The PAUSE (do not skip)

After step 4, stop and ask the developer:

> Paste your `synap_...` API key here (or set it yourself and tell me when it's ready).

Then set it in their environment — **never commit it**:

```bash
export SYNAP_API_KEY=synap_...
```

For apps, put it in `.env` (git-ignored) and load it (e.g. `python-dotenv`, or your
framework's env loader). Resolution order and auth errors: `reference/sdk-setup.md`.

Only after the key is set do you install the SDK and write integration code.

---
> **Accurate as of** `maximem-synap` 0.2.6 (Python) · `@maximem/synap-js-sdk` 0.3.0 (JS) — verified 2026-06-20.
> Live, changing detail: https://docs.maximem.ai (Mintlify serves a clean `.md` for any page).
