#!/usr/bin/env node
/**
 * Compatibility shim.
 *
 * `npx synap-js-sdk setup` used to build a Python virtualenv and install the
 * Python SDK into it. Since 0.4 the SDK is native TypeScript, so there is
 * nothing to set up.
 *
 * The command is kept, and kept exiting 0, because it appears in published
 * documentation and in customers' postinstall and CI scripts. Removing the
 * binary would turn a no-longer-needed step into a hard failure on upgrade.
 *
 * ESM, not CommonJS: package.json declares "type": "module", so a `require`
 * here would throw ERR_REQUIRE_ESM on every invocation.
 */

const [, , command] = process.argv;

const NOTE = `
@maximem/synap-js-sdk 0.4+ is a native TypeScript SDK. It does not use Python,
so there is no setup step and this command does nothing.

If you are upgrading from 0.3.x:
  - Remove any "npx synap-js-sdk setup" step from your install or CI scripts.
  - ~/.synap-js-sdk/.venv is now orphaned and safe to delete.
  - Node 20 or newer is required.
  - listen() is now opt-in rather than started automatically.

See MIGRATING.md in the package for the full list.
`.trim();

switch (command) {
  case 'setup':
  case 'setup-ts':
  case undefined:
    console.log(NOTE);
    break;
  case '--version':
  case '-v':
    console.log(process.env['npm_package_version'] ?? '0.4+');
    break;
  default:
    console.log(`Unknown command: ${command}\n\n${NOTE}`);
    // Still exit 0: an unknown subcommand in a legacy script should not break
    // a customer's build on upgrade.
    break;
}
