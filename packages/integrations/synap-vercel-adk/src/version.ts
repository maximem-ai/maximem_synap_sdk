// Single source of truth for the SDK version string.
// Keep in sync with package.json "version" — surfaced in the User-Agent header
// and in ContextAssembledEvent.sdk_version for server-side audit correlation.
export const SDK_VERSION = '0.3.0';
