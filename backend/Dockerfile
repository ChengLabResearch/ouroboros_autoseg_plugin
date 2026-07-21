# ── Stage 1: build the Rust backend for CPU runtimes ─────────────────────────
# Rust 1.88+ is required by the locked image dependency; it also supports edition 2024 manifests.
FROM rust:1.88-slim-bookworm AS builder

RUN apt-get update && apt-get install -y \
    git \
    pkg-config \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Layout inside the build container mirrors the local tava_hut structure so
# the Cargo.toml path deps (../../candle_sam3_main/…) resolve correctly:
#
#   /build/
#     candle_sam3_main/    ← cloned from GitHub
#     plugin/
#       backend/           ← copied from build context (this directory)
#         Cargo.toml       ← path "../../candle_sam3_main/…" resolves to /build/candle_sam3_main/

WORKDIR /build

# Clone candle_sam3 at the pinned commit used during development.
ARG CANDLE_SAM3_COMMIT=ee8555977d885ddeefd4e6e445c5ce5b978221a2
ARG PLUGIN_GIT_SHA=unknown
ENV CANDLE_SAM3_GIT_SHA=${CANDLE_SAM3_COMMIT}
ENV PLUGIN_GIT_SHA=${PLUGIN_GIT_SHA}
RUN git clone https://github.com/den-sq/candle_sam3.git candle_sam3_main \
    && git -C candle_sam3_main checkout "${CANDLE_SAM3_COMMIT}"

WORKDIR /build/plugin
COPY . backend/

WORKDIR /build/plugin/backend
RUN cargo build --release --bin ouroboros-autoseg-plugin-backend

# ── Stage 2: build the Rust backend for CUDA runtimes ────────────────────────
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04 AS cuda-builder

ENV PATH="/root/.cargo/bin:${PATH}"
ARG RUST_VERSION=1.88.0

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    git \
    pkg-config \
    libssl-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
    | sh -s -- -y --profile minimal --default-toolchain "${RUST_VERSION}"

WORKDIR /build

ARG CANDLE_SAM3_COMMIT=ee8555977d885ddeefd4e6e445c5ce5b978221a2
ARG PLUGIN_GIT_SHA=unknown
ENV CANDLE_SAM3_GIT_SHA=${CANDLE_SAM3_COMMIT}
ENV PLUGIN_GIT_SHA=${PLUGIN_GIT_SHA}
RUN git clone https://github.com/den-sq/candle_sam3.git candle_sam3_main \
    && git -C candle_sam3_main checkout "${CANDLE_SAM3_COMMIT}"

WORKDIR /build/plugin
COPY . backend/

WORKDIR /build/plugin/backend
ARG CANDLE_FEATURES=cuda
ARG CUDA_COMPUTE_CAP=75
ENV CUDA_COMPUTE_CAP=${CUDA_COMPUTE_CAP}
RUN cargo build --release --features "${CANDLE_FEATURES}" --bin ouroboros-autoseg-plugin-backend

# ── Stage 3: minimal CPU runtime image ───────────────────────────────────────
FROM debian:bookworm-slim AS runtime

RUN apt-get update && apt-get install -y \
    ca-certificates \
    libssl3 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder \
    /build/plugin/backend/target/release/ouroboros-autoseg-plugin-backend \
    /usr/local/bin/ouroboros-autoseg-plugin-backend

EXPOSE 8686

CMD ["/usr/local/bin/ouroboros-autoseg-plugin-backend"]

# ── Stage 4: CUDA runtime image ──────────────────────────────────────────────
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04 AS cuda-runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    libssl3 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=cuda-builder \
    /build/plugin/backend/target/release/ouroboros-autoseg-plugin-backend \
    /usr/local/bin/ouroboros-autoseg-plugin-backend

EXPOSE 8686

CMD ["/usr/local/bin/ouroboros-autoseg-plugin-backend"]
