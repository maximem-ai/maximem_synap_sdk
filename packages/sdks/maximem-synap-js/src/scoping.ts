/**
 * The B2C identifier contract, client side.
 *
 *   B2C   send user_id, and nothing else. customer_id is not accepted.
 *   B2B   customer_id is required.
 *
 * The server is authoritative and rejects independently. This exists so the
 * failure happens at the call site, with a message that says what to do,
 * instead of as a round trip that returns an empty result.
 *
 * That last part is the whole reason. Sending a customer_id on a B2C instance
 * used to produce no exception anywhere: the write was filed under the
 * customer, the read asked for the user, and both returned success. One client
 * ran 4,634 consecutive empty fetches across seven days without a single error
 * to look at. An SDK that stays quiet about a misuse it can see is not being
 * permissive, it is hiding the bug.
 */
import { InvalidInputError } from './errors.js';

export const B2C_ISOLATION = 'equals_customer';

/**
 * Throw if this call breaks the contract.
 *
 * `isolation` is what `GET /api/v1/auth/whoami` reported. **undefined means the
 * server did not tell us**, which is true of any deployment older than that
 * field, and it must mean "do nothing". Guessing B2C there would make this SDK
 * refuse a B2B client's mandatory field against every server not yet upgraded,
 * a far worse failure than the one it prevents.
 */
export function checkCustomerId(
  isolation: string | undefined,
  customerId: unknown,
  where: string,
  instanceId?: string,
): void {
  if (customerId === undefined || customerId === null || customerId === '') return;
  if (isolation !== B2C_ISOLATION) return;
  throw new InvalidInputError(
    `${where}: customer_id is not accepted on this instance` +
      `${instanceId ? ` (${instanceId})` : ''}, which is B2C ` +
      `(user_context_isolation='${B2C_ISOLATION}'). Send user_id only; it is the ` +
      `whole identity, and the server files and reads your data under it. ` +
      `Received customer_id=${JSON.stringify(customerId)}.`,
  );
}
