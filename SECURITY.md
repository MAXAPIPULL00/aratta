# Security Policy

## Supported Versions

| Version | Supported |
|---------|-----------|
| 0.1.x   | Yes       |

## Reporting a Vulnerability

If you discover a security vulnerability in Aratta, please report it responsibly.

**Do not open a public GitHub issue for security vulnerabilities.**

Instead, use [GitHub Security Advisories](https://github.com/scri-labs/aratta/security/advisories/new) to report privately. Include:

- A description of the vulnerability.
- Steps to reproduce.
- Potential impact.
- Suggested fix (if any).

We will acknowledge receipt within 48 hours and aim to provide a fix or mitigation plan within 7 days for critical issues.

## Scope

Security issues we care about include:

- Sandbox escapes in the agent execution environment.
- Unauthorized access to API keys or credentials.
- Server-side request forgery (SSRF) through provider adapters.
- Injection attacks through tool definitions or chat messages.
- Denial of service through circuit breaker or health check manipulation.

## Disclosure

We follow coordinated disclosure. Once a fix is released, we will credit reporters (unless they prefer anonymity) in the changelog.
