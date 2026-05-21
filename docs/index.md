# Documentation

Audio Refinery is a GPU-accelerated audio processing pipeline — vocal separation
→ speaker diarization → transcription → sentiment — that runs in two modes
sharing one core: a [command-line tool](cli.md) for workstation use and an
[HTTP service](service.md) for deployment at scale.

Start at the [README](../README.md) for installation and the "choose your path"
overview.

## Guides

| Document | What it covers |
|----------|----------------|
| [CLI Reference](cli.md) | Every command (`separate`, `diarize`, `transcribe`, `sentiment`, `pipeline`, `pipeline-parallel`, `serve`), flags, examples, GPU targeting, and troubleshooting. |
| [Service Guide](service.md) | HTTP API endpoints, container deployment, output schemas, the full environment-variable reference, readiness probes, scaling, and the `file://` dev loop. |
| [Architecture](architecture.md) | Ghost Track pipeline design, the three-layer split (core / CLI / service), model selection rationale, and the data model. |
| [Use Cases](use-cases.md) | Who uses Audio Refinery and for what — a decision guide for CLI vs service vs neither. |
| [Performance](performance.md) | VRAM budgets, throughput benchmarks, and optimization knobs. |
| [Deployment](deployment.md) | Production deployment patterns for both CLI (workstation install) and service (Docker, orchestrators). |
| [Development](development.md) | Dev environment setup, running tests, running the service locally, and the release process. |

## By task

- **I want to transcribe a few files right now** → [CLI Reference](cli.md)
- **I want to deploy a transcription API** → [Service Guide](service.md)
- **I'm deciding whether this tool fits my use case** → [Use Cases](use-cases.md)
- **I need to know how fast it is / how much VRAM it needs** → [Performance](performance.md)
- **I want to contribute or run it from source** → [Development](development.md)
- **I want to understand how it works internally** → [Architecture](architecture.md)
