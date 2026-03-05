FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

# System dependencies
RUN apt-get update && apt-get install -y \
    python3.11 python3.11-dev python3-pip python3.11-venv \
    ffmpeg git curl \
    && rm -rf /var/lib/apt/lists/*

# Non-root user
RUN useradd -m -u 1000 refinery
WORKDIR /app
USER refinery

# Install uv
RUN pip install --user uv

# Copy and install the package (resolves main deps; may pull CPU-only torch)
COPY --chown=refinery:refinery . .
RUN uv pip install -e .

# Install WhisperX at the pinned commit — no-deps to avoid overwriting torch
# v3.1.1 tag has the old API without device_index; use the correct commit instead
RUN uv pip install --no-deps \
    "whisperx @ git+https://github.com/m-bain/whisperX.git@741ab9a2a8a1076c171e785363b23c55a91ceff1"

# Install pinned WhisperX runtime deps
# transformers must stay <4.40.0 — 4.40+ uses torch.utils._pytree.register_pytree_node
# which was added in PyTorch 2.2 and breaks with the pinned 2.1.2
RUN uv pip install \
    "av==16.1.0" "ctranslate2==4.7.1" "faster-whisper==1.2.1" \
    "flatbuffers==25.12.19" "nltk==3.9.2" "onnxruntime==1.24.1" \
    "transformers>=4.30.0,<4.40.0"

# Reinstall PyTorch with CUDA 12.1 wheels last — uv pip install -e . above may have
# pulled CPU-only builds; this guarantees the CUDA wheel is what's actually used
RUN uv pip install torch==2.1.2+cu121 torchaudio==2.1.2+cu121 \
    --extra-index-url https://download.pytorch.org/whl/cu121

CMD ["audio-refinery", "--help"]
