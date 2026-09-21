# syntax=docker/dockerfile:1.7
#
# ComfyRanch worker image
# ======================
# The image pulled by a rented Vast.ai GPU. Built and pushed to GHCR by
# .github/workflows/build-worker-image.yml — never built on the operator machine.
#
# Base image
# ----------
# A CUDA runtime base (Ubuntu 24.04 / Python 3.12) plus a pinned PyTorch (cu128)
# and a pinned ComfyUI checkout. Model weights are NOT baked in: the bootstrap
# fetches or mounts them at boot, so the image stays reproducible and small.
# Override the base with --build-arg COMFYUI_BASE_IMAGE=...
#
# Alternatives audited before choosing this (see workflow_provisioning.md §8):
#   comfyui/base:latest                    CUDA-only; no Python/ComfyUI, ~13.5 GB, stale (2024-03)
#   runpod/worker-comfyui                  serverless worker wrapper; reported fragile when layered
#   ghcr.io/saladtechnologies/comfyui-api  bundles an API server we do not need
#
# Build:
#   docker buildx build worker -f worker/Dockerfile \
#     -t ghcr.io/hype-mode/comfyranch-comfyui:latest --push

ARG COMFYUI_BASE_IMAGE=nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04
FROM ${COMFYUI_BASE_IMAGE}

ARG DEBIAN_FRONTEND=noninteractive
ARG COMFYUI_REPO=https://github.com/Comfy-Org/ComfyUI.git
ARG COMFYUI_REF=v0.37.0
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128
ARG TORCH_VERSION=2.9.1
ARG TORCHVISION_VERSION=0.24.1
ARG TORCHAUDIO_VERSION=2.9.1
ARG COMFYUI_HOME=/opt/ComfyUI
ARG COMFRANCH_HOME=/opt/comfyranch

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_PREFER_BINARY=1 \
    PIP_NO_CACHE_DIR=1 \
    COMFYUI_HOME=${COMFYUI_HOME} \
    COMFRANCH_HOME=${COMFRANCH_HOME}

# System packages: python runtime, build bits for wheels without prebuilts, and
# the CLI tools the bootstrap relies on (git for custom nodes, aria2c for fast
# parallel model downloads, ffmpeg for video/audio workflows).
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-venv python3-dev python3-pip \
        build-essential pkg-config \
        git curl wget ca-certificates aria2 jq \
        libgl1 libglib2.0-0 libsm6 libxext6 \
        ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Isolated Python environment shared by ComfyUI and its custom nodes.
RUN python3 -m venv ${COMFRANCH_HOME}/venv
ENV VIRTUAL_ENV=${COMFRANCH_HOME}/venv
ENV PATH=${COMFRANCH_HOME}/venv/bin:$PATH

RUN pip install --upgrade pip wheel setuptools

# Pinned ComfyUI checkout (>= v0.3.75 is required for Z-Image support).
RUN git clone --depth 1 --branch ${COMFYUI_REF} ${COMFYUI_REPO} ${COMFYUI_HOME}

# Pinned PyTorch first (explicit CUDA wheel index), then ComfyUI's requirements.
RUN pip install \
        "torch==${TORCH_VERSION}" \
        "torchvision==${TORCHVISION_VERSION}" \
        "torchaudio==${TORCHAUDIO_VERSION}" \
        --index-url ${TORCH_INDEX_URL} \
    && pip install -r ${COMFYUI_HOME}/requirements.txt

# Bootstrap + version stamp.
COPY bootstrap/ ${COMFRANCH_HOME}/bootstrap/
COPY VERSION ${COMFRANCH_HOME}/VERSION
COPY entrypoint.sh ${COMFRANCH_HOME}/entrypoint.sh
RUN chmod +x ${COMFRANCH_HOME}/entrypoint.sh ${COMFRANCH_HOME}/bootstrap/comfyranch_bootstrap.py

ENV COMFY_DIR=${COMFYUI_HOME} \
    COMFRANCH_PORT=8188

EXPOSE 8188
WORKDIR ${COMFYUI_HOME}
ENTRYPOINT ["/opt/comfyranch/entrypoint.sh"]
