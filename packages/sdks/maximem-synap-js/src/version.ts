/**
 * Kept in sync with package.json by the release workflow, in the
 * "Sync SDK_VERSION to the resolved version" step of publish-npm-js-sdk.yml.
 *
 * Until that step existed nothing kept it in sync, so this constant sat at
 * 1.0.0-beta.0 while the package shipped 0.4.x, and every request's User-Agent
 * and every telemetry event reported a version that was never released.
 *
 * ⚠ The workflow only covers a workflow release. 0.4.8 was published by hand
 * and shipped reporting 0.4.7, so every request from it carries a User-Agent
 * for a version it is not. `version-matches-package.test.ts` now fails the
 * build when the two disagree, which is the only thing that makes a manual
 * publish safe.
 */
export const SDK_VERSION = '0.4.9';
