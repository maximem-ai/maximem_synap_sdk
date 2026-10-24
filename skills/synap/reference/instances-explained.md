# What an instance is, and what the status words mean (say it in plain words)

> **Read this next:** [Identifiers & Scopes, "What is an Instance?"](https://docs.maximem.ai/concepts/memory-scopes.md#what-is-an-instance); [Agent topologies](https://docs.maximem.ai/concepts/agent-topologies.md); [Dashboard, "Managing instances"](https://docs.maximem.ai/dashboard/overview.md); [Describe your use case](https://docs.maximem.ai/getting-started/use-case.md); [CLI command reference](https://docs.maximem.ai/cli/commands.md). The live docs win over this file.

## The rule

Never show the human a raw status word or field name (`no_instance`, `building`, `attempt 2 of 3`, `b2c`,
`external_facing`, `initializing`, `equals_customer`, `maca`). Translate it with this page. If a word is not
here, read the docs first (start with the pages above and `https://docs.maximem.ai/llms.txt`), then say it
simply. If the docs do not explain it either, say you are not sure what it means; never invent a meaning.

Also never tell the human which Synap "environment", "production" or "staging" you are on (see "Words you
must not say" below).

## What an instance is

An **instance** is one memory brain for one agent. It has its own memory store, its own settings (built
from the use-case file you uploaded), its own API keys and its own list of people. But unless an Instance
Visibility policy is set, instances of the same workspace and customer can read each other's memories when they
retrieve (/guides/instance-visibility decides; the policy is set in the dashboard, not by you). Its id looks like `inst_a1b2c3d4e5f67890`. An API key belongs to exactly one
instance, so the key you use decides which instance the app talks to (memory-scopes, "What is an Instance?").

Say it like this: "An instance is the memory space Synap built for your agent. Your app's key opens exactly that
space and no other."

One instance is built around one agent's job (one memory architecture, generated from one use-case file). A
single instance serves all of that agent's people: you do not create an instance per user or per customer
(agent-topologies, "single-agent"). Agents with different jobs, extraction needs or compliance needs get their
own instance each (agent-topologies, "Separate Instance per agent").

## Words you may see, and what to tell the human

| Where you see it | Word | Say this instead |
|---|---|---|
| `instances wait`, `instances get`: the build | `building` | Synap is preparing the memory design for your agent from the file you approved. It usually takes 15 to 20 minutes. |
| same | `attempt 2 of 3` | Synap tries up to three times by itself. This is its second try; nothing is wrong yet. |
| same | `ready` | The memory design is finished and the instance can be used. |
| same | `failed` | The memory design could not be built after three tries. Nothing you approved is lost; I can start the build again from the same file. |
| same | `awaiting_review` / `unknown` | The design was made and is waiting for the Synap team to approve it. Nothing to do on your side. |
| `instances list/get`: STATUS | `initializing` or `provisioning` | The space is being set up. The dashboard docs say it becomes active once the first connection from your app lands. |
| same | `active` | In use. |
| same | `inactive`, `suspended` | Switched off or paused. The docs list these states but do not say more, so do not guess what they block; tell the human and do not change it. |
| same | `deleting` | Being removed. |
| `instances list`: AGENT column | `external_facing`, `intra_org`, `special_purpose` | The kind of agent, filled in for you from who uses it (customers, your own staff, or other software). It is not an error and not about whether an agent exists. |
| `instances list`: CONFIG | `approved` and similar | Whether the memory design is in place. Trust the build state above, not this column. |
| who uses it | `b2c`, `equals_customer` | Each end user has their own private memory. |
| same | `b2b`, `strict` | Users belong to a customer organization, and memory is kept per organization and per person. |
| same | `internal` | The agent serves your own staff. |
| same | `agent_to_agent` | The agent serves other software. |
| `onboard status`: lifecycle | `no_workspace` / `no_instance` | The account has no space yet / no memory instance yet. |
| same | `provisioning` | Synap is building the memory design. |
| same | `maca_failed` | The memory design build failed (see `failed` above). |
| same | `no_api_key` | The instance exists but no key for your app has been made yet. |
| same | `no_ingestion` | A key exists but nothing has been written to memory yet. |
| same | `no_retrieval` | Something was written but nothing has been read back yet. |
| same | `activated` | Memory was written and read back with results: it works. |
| `onboard status`: `agent_connected` false | (shows as "waiting for agent") | No coding agent has signed in with the one-time code yet. This is about the sign-in, not about the human's agent or the instance. |
| `onboard report` | `in_progress`, `done`, `failed`, `cleared` | Only for the dashboard checklist; do not repeat these words to the human. |
| HTTP 409 | `instance_not_ready` | The memory design is still being built, so memory calls are refused for now. Your app will see it too until the build ends. |
| HTTP 409 | `instance_name_taken` | An instance with that name already exists in your workspace. Ask the human: use that one (only if it is for this agent) or pick another name. |
| HTTP 409 | `instance_limit_reached` | The workspace (or this setup session) is at its instance limit. Stop, tell the human the limit and count, ask whether to reuse an existing instance or contact support. Never retry, delete instances to make room, or work around it. 429 `instance_create_rate_limited` is the same: stop and ask. |
| HTTP 409 | `instance_build_failed` | The build gave up; I will start it again. |
| a rebuild | `rebuild` | Run the build again from the same approved file. Allowed at most 3 times in 24 hours. |
| `maca`, MACA | memory architecture | The memory design Synap generates from your use-case file: what to remember, how to store and find it. |

`no_agent` is NOT a Synap word: it appears nowhere in the product. If you were about to say "the instance is in
no_agent state", you were probably reading `agent_connected` or the lifecycle word above. Say what is true in
plain words instead (for example "no instance has been created yet" or "the setup sign-in is not connected").

## Words you must not say to the human

For the human there is just Synap: no environments. Never say "production", "prod", "staging" or
"environment" about Synap itself. The words come from the tools, not from the human's situation:

- `auth whoami` prints an `environment` field, `auth login` prints "(workspace X, production)" and `auth list`
  has an ENVIRONMENT column.
- A prompt for a test server may carry a line "Environment: staging ... run every CLI command with --env".
  Use the value silently: pass `--env <name>` (or set `SYNAP_ENV`) and say nothing about it.
- `SYNAP_BASE_URL`, `SYNAP_GRPC_HOST`, `SYNAP_GRPC_PORT` and `SYNAP_GRPC_USE_TLS` appear in `.env` when the CLI
  was pointed at a non-default server. Do not explain them as "staging"; say "connection settings the setup
  wrote for your app" and leave them in place.

The human's own deployments (their own dev, test and live versions of their app) are theirs to name; talk
about those freely, and say each one needs its own Synap instance and key (see `sdk-setup.md`, "Several
instances").

## Where the docs fall short (for the docs owner)

- There is no page titled "Instances". The concept is a section of memory-scopes ("What is an Instance?"),
  and the dashboard overview has "Managing instances".
- The build states (`building`, `ready`, `failed`, "attempt n of 3", the daily rebuild limit), `instances wait`
  and `instances rebuild` are not in the docs or in the CLI command reference.
- The instance STATUS values `provisioning`, `inactive`, `suspended` and `deleting` are listed but not explained;
  `agent_type`, the AGENT and CONFIG columns, and the onboarding lifecycle words are not documented at all.
