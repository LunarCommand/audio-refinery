# Audio Refinery

GPU-accelerated audio processing pipeline — vocal separation (Demucs), speaker
diarization (Pyannote), transcription (WhisperX), and sentiment analysis —
packaged as a long-lived HTTP service. Point it at audio by URI and get back
speaker-attributed transcripts with word-level timestamps as JSON. Models load
once at container startup and stay resident across jobs.

## Quick start

```bash
docker run --gpus all -p 8000:8000 \
  -e REFINERY_API_KEYS=your-secret-key \
  -e HF_TOKEN=hf_your_token \
  lunarcommand/audio-refinery:latest
```

Wait for `GET /health` to return `200` (model warmup, ~10–30 s), then
`POST /transcribe` a batch of jobs. Full API and environment reference:
[Service Guide](https://github.com/LunarCommand/audio-refinery/blob/main/docs/service.md).

## Requirements

- NVIDIA GPU — **24 GB VRAM recommended** (holds all models resident)
- NVIDIA driver ≥ 525.85.12 and the NVIDIA Container Toolkit on the host
- A HuggingFace token for the gated Pyannote models (`HF_TOKEN`)

## Tags

- `latest` — most recent stable release
- `X.Y.Z` — pinned release (e.g. `0.2.0`)

## Key environment variables

| Variable | Purpose |
|----------|---------|
| `REFINERY_API_KEYS` | **required** — bearer-token allowlist |
| `HF_TOKEN` | **required** — HuggingFace token for diarization |
| `REFINERY_WHISPER_MODEL` | Whisper variant (default `large-v3`) |
| `REFINERY_SCRATCH_DIR` | per-job scratch; mount a tmpfs for throughput |

Full environment reference, CLI usage, and architecture:
**https://github.com/LunarCommand/audio-refinery**

## License

MIT. Pyannote model weights are gated under separate HuggingFace terms — verify
your account's accepted terms cover your use case.
