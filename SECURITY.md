# Security

Report vulnerabilities to **security@lica.world** (not the public issue tracker). We'll respond within 48 hours.

GDB is a benchmark evaluation framework — no auth, no payments, no user data. The relevant surface area is API key handling (env vars / credential files), dataset path resolution, and subprocess execution via `--custom-entry`.
