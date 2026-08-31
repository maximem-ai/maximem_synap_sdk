/**
 * Kept in sync with package.json by the release workflow, in the
 * "Sync SDK_VERSION to the resolved version" step of publish-npm-js-sdk.yml.
 *
 * Until that step existed nothing kept it in sync, so this constant sat at
 * 1.0.0-beta.0 while the package shipped 0.4.x, and every request's User-Agent
 * and every telemetry event reported a version that was never released.
 */
export const SDK_VERSION = '0.4.7';
