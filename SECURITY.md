# Security policy

## Reporting a vulnerability
Please do not open a public issue for security problems.

Report privately through GitHub: go to the **Security** tab of this repository and choose **Report a vulnerability**. You can also email [security@maximem.ai](mailto:security@maximem.ai). We acknowledge reports within 72 hours and keep you updated until the issue is fixed.

## Scope
- The Python and JavaScript SDKs in this repository
- The framework integration packages under `packages/integrations/`
- The MCP server adapter under `packages/mcps/`

Issues in the hosted Synap service can be reported the same way.

## Handling API keys
The SDKs send your Synap API key only to the Synap API endpoint you configure. Never commit keys to source control. If a key is exposed, revoke it from the dashboard at https://synap.maximem.ai and create a new one.
