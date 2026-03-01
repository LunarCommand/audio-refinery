# Security Policy

## Reporting a Vulnerability

If you discover a security vulnerability, please **do not open a public GitHub issue**.

Instead, email [chris@lunarcommand.xyz](mailto:chris@lunarcommand.xyz) with:

- A description of the vulnerability
- Steps to reproduce
- Potential impact
- Any suggested mitigations

You can expect an acknowledgement within 48 hours and a resolution or status update within 7 days.

## Scope

Security issues relevant to this project include:

- Unsafe handling of user-supplied file paths
- Credential or token exposure
- Arbitrary code execution via malicious audio files or pipeline configs
- Dependency vulnerabilities with direct impact on users

## Out of Scope

- Issues in third-party models (Demucs, Pyannote, WhisperX, HuggingFace) — report those upstream
- Issues requiring physical access to the machine running the pipeline
