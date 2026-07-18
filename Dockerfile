# syntax=docker/dockerfile:1
#
# GPU image for aibackends, built for CUDA clouds such as RunPod and Modal.
#
# Build:
#   docker build -t aibackends:cuda .
#
# Run (needs the NVIDIA container toolkit / a GPU cloud):
#   docker run --rm --gpus all aibackends:cuda \
#     aibackends task summarize --input notes.txt --runtime llamacpp --model gemma4-e2b
#
# See docs/docker.md for RunPod and Modal usage.

ARG CUDA_VERSION=12.8.1
ARG UBUNTU_VERSION=24.04

# ---------------------------------------------------------------------------
# Builder: compiles llama-cpp-python with CUDA and installs everything into
# a virtualenv that is copied into the slimmer runtime image below.
# ---------------------------------------------------------------------------
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION} AS builder

# CUDA compute capabilities baked into the llama.cpp kernels:
#   75 = T4            80 = A100          86 = A10/A40/RTX 30xx
#   89 = L4/L40S/RTX 40xx                 90 = H100/H200
#   100 = B200         120 = RTX 5090/PRO 6000
# Trim this list (e.g. --build-arg CUDA_ARCHITECTURES="89") for faster builds
# when you know which GPU you will run on.
ARG CUDA_ARCHITECTURES="75;80;86;89;90;100;120"

# Extras baked into the image. Trim for a smaller image, e.g.
# --build-arg AIBACKENDS_EXTRAS="llamacpp,transformers"
ARG AIBACKENDS_EXTRAS="llamacpp,transformers,pdf,audio,video,pii"

# PyTorch wheel index; cu128 wheels include Blackwell (sm_120) support.
ARG TORCH_INDEX_URL="https://download.pytorch.org/whl/cu128"

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        cmake \
        git \
        ninja-build \
        python3.12 \
        python3.12-dev \
        python3.12-venv \
    && rm -rf /var/lib/apt/lists/*

RUN python3.12 -m venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"
RUN pip install --upgrade pip setuptools wheel

# Compile llama-cpp-python against CUDA. This is the slow step; it is kept
# in its own layer so image rebuilds after source changes stay fast.
RUN CMAKE_ARGS="-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCHITECTURES}" \
    FORCE_CMAKE=1 \
    pip install llama-cpp-python

# Install torch from the CUDA wheel index before the project so the
# transformers extra does not pull a different build from PyPI.
RUN pip install torch --index-url "${TORCH_INDEX_URL}"

WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install ".[${AIBACKENDS_EXTRAS}]"

# ---------------------------------------------------------------------------
# Runtime: CUDA runtime + cuDNN (needed by faster-whisper/ctranslate2 on GPU),
# ffmpeg for the audio/video extras, and the prebuilt virtualenv.
# ---------------------------------------------------------------------------
FROM nvidia/cuda:${CUDA_VERSION}-cudnn-runtime-ubuntu${UBUNTU_VERSION}

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:${PATH}" \
    HF_HOME=/models

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        ffmpeg \
        libgomp1 \
        python3.12 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv

WORKDIR /app
COPY examples ./examples

# Downloaded models land here; mount a persistent volume to keep them
# across container restarts (RunPod network volume, Modal volume, ...).
RUN mkdir -p /models
VOLUME ["/models"]

CMD ["aibackends", "--help"]
