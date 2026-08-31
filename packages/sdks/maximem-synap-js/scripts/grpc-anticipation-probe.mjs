/**
 * Prove server -> client bundle delivery on the Listen stream.
 *
 * `instance.listen()` deliberately does NOT emit session_control, matching
 * Python, whose public listen() does not either (its anticipation_smoke.py
 * reaches into `sdk._grpc_transport` to send one). Section 8 session_start
 * anticipation fires on that control message, so without it the server runs
 * anticipation_dispatch and has nothing to dispatch.
 *
 * The `/grpc` subpath listen() does send it, so this uses that path.
 */
import { randomUUID } from 'node:crypto';
import { listen } from '../dist/grpc/index.js';
import { SynapClient } from '../dist/index.js';

const API_KEY = process.env.SYNAP_API_KEY;
const BASE_URL = process.env.SYNAP_BASE_URL;
const HOST = process.env.SYNAP_GRPC_HOST ?? '127.0.0.1';
const PORT = Number(process.env.SYNAP_GRPC_PORT ?? 50051);
const TLS = (process.env.SYNAP_GRPC_USE_TLS ?? '0') !== '0';
const WAIT_S = Number(process.env.SYNAP_PROBE_WAIT ?? 60);

const USER_ID = process.env.SYNAP_SMOKE_USER_ID ?? `js-grpc-user-${randomUUID().slice(0, 8)}`;
const CUSTOMER_ID = process.env.SYNAP_SMOKE_CUSTOMER_ID ?? `js-grpc-cust-${randomUUID().slice(0, 8)}`;
const CONVERSATION_ID = randomUUID();

console.log(`grpc anticipation probe`);
console.log(`  grpc          ${HOST}:${PORT} tls=${TLS}`);
console.log(`  user          ${USER_ID}`);
console.log(`  conversation  ${CONVERSATION_ID}`);

// Seed something worth anticipating about, so a bundle has content to carry.
const client = new SynapClient({ apiKey: API_KEY, baseUrl: BASE_URL });
await client.initialize();
console.log(`  instance      ${client.instance_id}`);

// Content deliberately aimed at the generic fallback anticipation queries the
// server generates on session_start when an instance has no MACA config:
// 'user role and position', 'manager and team', 'key dates and milestones'.
// The first attempt seeded ride-preference text, anticipation retrieved 0 items,
// and so no bundle was ever pushed. That was a content mismatch, not a
// transport problem.
const seed = await client.memories.create({
  document:
    'Priya Raman is the Director of Platform Engineering at Northwind Logistics. ' +
    'She reports to the VP of Engineering, Marcus Feld, and leads a team of nine ' +
    'engineers across two squads. Her role covers the ingestion platform and the ' +
    'billing service. Key dates: she joined in March 2024, was promoted to ' +
    'Director in January 2026, and her next performance review is due in ' +
    'September 2026. The platform migration milestone lands in November 2026.',
  user_id: USER_ID,
  customer_id: CUSTOMER_ID,
});
console.log(`  seeded        ingestion=${seed.ingestion_id}`);
const done = await client.memories.wait_for_completion(seed.ingestion_id, { timeout_seconds: 120 });
console.log(`  seed status   ${done.status} memories=${done.memories_created ?? '?'}`);

const bundles = [];
const session = await listen({
  apiKey: API_KEY,
  clientId: client.client_id,
  instanceId: client.instance_id,
  host: HOST,
  port: PORT,
  useTls: TLS,
  conversationId: CONVERSATION_ID,
  userId: USER_ID,
  customerId: CUSTOMER_ID,
  onBundle: (b) => {
    bundles.push(b);
    console.log(
      `  BUNDLE        id=${b.bundle_id} type=${b.bundle_type || 'anticipation'} ` +
      `pattern=${b.origin_pattern_id || '-'} items=${Object.values(b.items_by_type ?? {})
        .reduce((n, l) => n + (l?.items?.length ?? 0), 0)} ` +
      `queries=${(b.search_queries ?? []).slice(0, 3).join('|')}`,
    );
  },
});
console.log(`  stream        active=${session.active} (session_control start sent)`);

const deadline = Date.now() + WAIT_S * 1000;
while (bundles.length === 0 && Date.now() < deadline) {
  await new Promise((r) => setTimeout(r, 1000));
}

console.log();
if (bundles.length > 0) {
  console.log(`RESULT: ${bundles.length} bundle(s) received from the server.`);
  const types = {};
  for (const b of bundles) {
    const t = b.bundle_type || 'anticipation';
    types[t] = (types[t] ?? 0) + 1;
  }
  console.log(`  by type: ${JSON.stringify(types)}`);
} else {
  console.log(`RESULT: no bundle in ${WAIT_S}s. Client->server is proven either way`);
  console.log(`        (check the server log for conversation_events / session_controls).`);
}

await session.stop();
await client.shutdown();
