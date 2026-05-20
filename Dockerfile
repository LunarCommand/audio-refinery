# syntax=docker/dockerfile:1.7
# CUDA runtime variant (not devel) — saves ~3GB by dropping the full CUDA SDK
# that's only needed at compile time. Matches the torch==2.1.2+cu121 wheel
# pinned in pyproject.toml.
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# System dependencies
#   python3.11{,-dev,-venv}: pinned interpreter (pyannote + WhisperX need 3.11)
#   ffmpeg                 : Demucs audio I/O
#   curl                   : container HEALTHCHECK + occasional debug
#   git                    : whisperx is installed from a pinned git ref
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-dev python3-pip python3.11-venv \
    ffmpeg git curl \
    && rm -rf /var/lib/apt/lists/*

# Non-root runtime user
RUN useradd -m -u 1000 refinery
WORKDIR /app
USER refinery

# Install uv into the refinery user's PATH
RUN pip install --user uv
ENV PATH="/home/refinery/.local/bin:${PATH}"

# Copy and install the package (resolves main deps; may pull CPU-only torch)
COPY --chown=refinery:refinery . .
RUN uv pip install --system -e .

# Install WhisperX at the pinned commit — no-deps to avoid overwriting torch.
# v3.1.1 tag has the old API without device_index; use the correct commit.
RUN uv pip install --system --no-deps \
    "whisperx @ git+https://github.com/m-bain/whisperX.git@741ab9a2a8a1076c171e785363b23c55a91ceff1"

# Install pinned WhisperX runtime deps.
# transformers stays <4.40.0 — 4.40+ uses torch.utils._pytree.register_pytree_node
# added in PyTorch 2.2, which breaks with the pinned 2.1.2.
RUN uv pip install --system \
    "av==16.1.0" "ctranslate2==4.7.1" "faster-whisper==1.2.1" \
    "flatbuffers==25.12.19" "nltk==3.9.2" "onnxruntime==1.24.1" \
    "transformers>=4.30.0,<4.40.0"

# Reinstall PyTorch with CUDA 12.1 wheels last — `uv pip install -e .` above
# may have pulled CPU-only builds; this guarantees the CUDA wheel is what's
# actually used at runtime.
RUN uv pip install --system torch==2.1.2+cu121 torchaudio==2.1.2+cu121 \
    --extra-index-url https://download.pytorch.org/whl/cu121

# Per-job Demucs scratch lives here. The directory is declared as a VOLUME
# so operators can bind a tmpfs mount (recommended on RAM-rich hosts for the
# RAM-disk benefit Demucs throughput likes) or a fast disk on RAM-tight VMs.
# The env var is honored by `tempfile.TemporaryDirectory(dir=...)` in the
# worker. Unset REFINERY_SCRATCH_DIR to fall back to the system /tmp.
USER root
RUN mkdir -p /scratch && chown refinery:refinery /scratch
USER refinery
ENV REFINERY_SCRATCH_DIR=/scratch
VOLUME ["/scratch"]

# Service mode binds REFINERY_PORT (default 8000) on all interfaces.
EXPOSE 8000

# Orchestrator-friendly health probe. /health returns 503 with status="loading"
# during the ~10s model warmup, then flips to 200 status="ok". start-period
# of 60s gives warmup ample headroom without flapping the container.
HEALTHCHECK --interval=10s --timeout=5s --start-period=60s --retries=3 \
    CMD curl -fsS "http://localhost:${REFINERY_PORT:-8000}/health" || exit 1

# Service mode is the default. CLI is still invocable as an override:
#   docker run --gpus all <image> audio-refinery pipeline --help
CMD ["audio-refinery-service"]
