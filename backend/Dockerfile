ARG CPU_BUILDER_IMAGE=rust:1.88-slim-bookworm@sha256:38bc5a86d998772d4aec2348656ed21438d20fcdce2795b56ca434cf21430d89
ARG CPU_RUNTIME_IMAGE=debian:bookworm-slim@sha256:734a72b3d6142a71993b4bc580ca01d8ca687f77a2c422d96a961d3376b179bc
ARG CUDA_BUILDER_IMAGE=nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04@sha256:622e78a1d02c0f90ed900e3985d6c975d8e2dc9ee5e61643aed587dcf9129f42
ARG CUDA_RUNTIME_IMAGE=nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04@sha256:2fcc4280646484290cc50dce5e65f388dd04352b07cbe89a635703bd1f9aedb6

# ── Stage 1: build the Rust backend for CPU runtimes ─────────────────────────
# Rust 1.88+ is required by the locked image dependency; it also supports edition 2024 manifests.
FROM ${CPU_BUILDER_IMAGE} AS builder

RUN apt-get update && apt-get install -y \
    git \
    pkg-config \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Candle and candle-sam3 are git dependencies pinned in Cargo.toml and Cargo.lock;
# Cargo fetches them during the dependency build below.
# build.rs derives the logged engine revision from the resolved Cargo.lock source.

# Compile the stable dependency graph before copying plugin source. The dummy
# package is replaced below, while its target artifacts remain reusable when
# ordinary plugin source changes.
WORKDIR /build/plugin/backend
COPY Cargo.toml Cargo.lock ./
RUN mkdir -p src \
    && printf 'fn main() {}\n' > src/main.rs \
    && printf '' > src/lib.rs \
    && cargo build --release --locked --bin ouroboros-autoseg-plugin-backend \
    && rm -rf src

# cargo-about generates the third-party licence notices shipped in the runtime
# image. It is installed after the dependency build so that layer's cache is
# unaffected, and before the plugin source so source edits do not reinstall it.
ARG CARGO_ABOUT_VERSION=0.9.3
RUN cargo install cargo-about --version "${CARGO_ABOUT_VERSION}" --locked --features cli

COPY build.rs ./
COPY src/ src/
RUN touch src/lib.rs src/main.rs \
    && cargo build --release --locked --bin ouroboros-autoseg-plugin-backend

COPY about.toml about.hbs ./
RUN cargo about generate --locked --fail about.hbs -o THIRD_PARTY_LICENSES.txt

# ── CPU tests gate its candidate; main/release promotion requires both jobs ──
FROM builder AS test

COPY tests/ tests/
RUN cargo test --release --locked

# ── Stage 2: build the Rust backend for CUDA runtimes ────────────────────────
FROM ${CUDA_BUILDER_IMAGE} AS cuda-builder

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

WORKDIR /build/plugin/backend
ARG CANDLE_FEATURES=cuda,cudnn
ARG CUDA_COMPUTE_CAP=75
ENV CUDA_COMPUTE_CAP=${CUDA_COMPUTE_CAP}
COPY Cargo.toml Cargo.lock ./
RUN test "${CANDLE_FEATURES}" = "cuda,cudnn" \
    || { echo "CANDLE_FEATURES must equal the canonical cuda,cudnn feature set" >&2; exit 1; } \
    && mkdir -p src \
    && printf 'fn main() {}\n' > src/main.rs \
    && printf '' > src/lib.rs \
    && cargo build --release --locked --features "${CANDLE_FEATURES}" \
        --bin ouroboros-autoseg-plugin-backend \
    && rm -rf src

ARG CARGO_ABOUT_VERSION=0.9.3
RUN cargo install cargo-about --version "${CARGO_ABOUT_VERSION}" --locked --features cli

COPY build.rs ./
COPY src/ src/
RUN touch src/lib.rs src/main.rs \
    && cargo build --release --locked --features "${CANDLE_FEATURES}" \
    --bin ouroboros-autoseg-plugin-backend

COPY about.toml about.hbs ./
RUN cargo about generate --locked --fail --features "${CANDLE_FEATURES}" \
    about.hbs -o THIRD_PARTY_LICENSES.txt

# ── Stage 3: minimal CPU runtime image ───────────────────────────────────────
FROM ${CPU_RUNTIME_IMAGE} AS runtime

RUN apt-get update && apt-get install -y \
    ca-certificates \
    libssl3 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder \
    /build/plugin/backend/target/release/ouroboros-autoseg-plugin-backend \
    /usr/local/bin/ouroboros-autoseg-plugin-backend
COPY licenses/LICENSE licenses/LICENSE-SAM /usr/share/licenses/ouroboros-autoseg-plugin-backend/
COPY --from=builder \
    /build/plugin/backend/THIRD_PARTY_LICENSES.txt \
    /usr/share/licenses/ouroboros-autoseg-plugin-backend/

EXPOSE 8686

CMD ["/usr/local/bin/ouroboros-autoseg-plugin-backend"]

# ── Stage 4: CUDA runtime image ──────────────────────────────────────────────
FROM ${CUDA_RUNTIME_IMAGE} AS cuda-runtime

ARG CANDLE_FEATURES=cuda,cudnn
ENV OUROBOROS_COMPILED_FEATURES=${CANDLE_FEATURES} \
    OUROBOROS_COMPILED_CUDA=true \
    OUROBOROS_COMPILED_CUDNN=true
LABEL io.ouroboros.backend.compiled.features=${CANDLE_FEATURES} \
      io.ouroboros.backend.compiled.cuda="true" \
      io.ouroboros.backend.compiled.cudnn="true"

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    libssl3 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=cuda-builder \
    /build/plugin/backend/target/release/ouroboros-autoseg-plugin-backend \
    /usr/local/bin/ouroboros-autoseg-plugin-backend
COPY licenses/ /usr/share/licenses/ouroboros-autoseg-plugin-backend/
COPY --from=cuda-builder \
    /build/plugin/backend/THIRD_PARTY_LICENSES.txt \
    /usr/share/licenses/ouroboros-autoseg-plugin-backend/

# Verify capabilities and cuDNN linkage before a reusable candidate can publish.
# This does not initialize a GPU and runs on the hosted CPU-only builders.
RUN set -eu; \
    test "${OUROBOROS_COMPILED_FEATURES}" = "cuda,cudnn"; \
    test "${OUROBOROS_COMPILED_CUDA}" = "true"; \
    test "${OUROBOROS_COMPILED_CUDNN}" = "true"; \
    linkage="$(ldd /usr/local/bin/ouroboros-autoseg-plugin-backend)"; \
    printf '%s\n' "${linkage}"; \
    printf '%s\n' "${linkage}" | grep -Eq 'libcudnn[.]so([.][0-9]+)* => /'; \
    ! printf '%s\n' "${linkage}" | grep -Eq 'libcudnn[.]so([.][0-9]+)* => not found'

EXPOSE 8686

CMD ["/bin/sh", "-c", "printf '%s\\n' 'backend compiled capabilities: cuda=true cudnn=true features=cuda,cudnn' >&2; exec /usr/local/bin/ouroboros-autoseg-plugin-backend"]
