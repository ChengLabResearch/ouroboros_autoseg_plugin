# ── Stage 1: build the Rust backend ──────────────────────────────────────────
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
ARG CANDLE_SAM3_COMMIT=770d20ca8db4f834ba4c89c845bca196fbfc97ea
RUN git clone https://github.com/den-sq/candle_sam3.git candle_sam3_main \
    && git -C candle_sam3_main checkout "${CANDLE_SAM3_COMMIT}"

WORKDIR /build/plugin
COPY . backend/

WORKDIR /build/plugin/backend
RUN cargo build --release --bin ouroboros-autoseg-plugin-backend

# ── Stage 2: minimal runtime image ───────────────────────────────────────────
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
