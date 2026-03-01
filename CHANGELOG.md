# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-02-27

### Added

- Initial release
- `separate` command: GPU-accelerated vocal separation via Demucs `htdemucs`
- `diarize` command: Speaker diarization via Pyannote `speaker-diarization-3.1`
- `transcribe` command: Transcription with word-level alignment via WhisperX `large-v3`
- `sentiment` command: Per-segment sentiment analysis via `cardiffnlp/twitter-roberta-base-sentiment-latest`
- `pipeline` command: Single-GPU batch processing through all four stages
- `pipeline-parallel` command: Multi-GPU batch processing with worker distribution
- GPU pre-flight checks via `nvidia-smi` with active process detection
- Thermal monitoring with configurable shutdown threshold (default 80°C)
- Slack webhook notifications for pipeline completion and thermal events
- Resume behavior for batch pipelines (skips already-completed files)
- Pydantic data models for all pipeline outputs with full provenance tracking
- Rich terminal output with progress spinners and result tables
- VRAM usage tracking and peak memory reporting per stage
- Scratch space management with automatic cleanup of intermediate files
- GPU performance-based ordering via `gpu_tflops.toml` lookup table

[Unreleased]: https://github.com/LunarCommand/audio-refinery/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/LunarCommand/audio-refinery/releases/tag/v0.1.0
