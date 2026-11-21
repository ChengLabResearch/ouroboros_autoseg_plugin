# syntax=docker/dockerfile:1

ARG CPU_BUILDER_IMAGE=rust:1.88-slim-bookworm@sha256:38bc5a86d998772d4aec2348656ed21438d20fcdce2795b56ca434cf21430d89
ARG CPU_RUNTIME_IMAGE=debian:bookworm-slim@sha256:734a72b3d6142a71993b4bc580ca01d8ca687f77a2c422d96a961d3376b179bc
ARG CUDA_BUILDER_IMAGE=nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04@sha256:622e78a1d02c0f90ed900e3985d6c975d8e2dc9ee5e61643aed587dcf9129f42
ARG CUDA_RUNTIME_IMAGE=nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04@sha256:2fcc4280646484290cc50dce5e65f388dd04352b07cbe89a635703bd1f9aedb6

# ── Stage 1: build the Rust backend for CPU runtimes ─────────────────────────
# Rust 1.88+ is required by the locked image dependency; it also supports edition 2024 manifests.
# The optional probe secret stops an uncached dependency ancestor before any
# installation or compilation. Normal builds omit it.
FROM ${CPU_BUILDER_IMAGE} AS builder-base

RUN --mount=type=secret,id=backend-dependency-probe \
    if [ -f /run/secrets/backend-dependency-probe ]; then \
        printf 'OUROBOROS_DEPENDENCY_CACHE_MISS\n' >&2; \
        exit 83; \
    fi; \
    apt-get update && apt-get install -y \
    git \
    pkg-config \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build/plugin/backend

# Tool installation is independent of the plugin dependency graph. Both build
# branches inherit identical dummy targets, including the build dependencies.
FROM builder-base AS builder-tools
ARG CARGO_ABOUT_VERSION=0.9.3
RUN --mount=type=secret,id=backend-dependency-probe \
    if [ -f /run/secrets/backend-dependency-probe ]; then \
        printf 'OUROBOROS_DEPENDENCY_CACHE_MISS\n' >&2; \
        exit 83; \
    fi; \
    cargo install cargo-about --version "${CARGO_ABOUT_VERSION}" --locked --features cli \
    && mkdir -p src \
    && printf 'fn main() {}\n' > src/main.rs \
    && printf '' > src/lib.rs \
    && printf 'fn main() {}\n' > build.rs

# Mutable Cargo caches are used only for dependency changes. The snapshot
# preserves their canonical paths and mtimes for the ordinary builder layers.
FROM builder-tools AS builder-dependencies
COPY Cargo.toml Cargo.lock ./
ARG BACKEND_CACHE_NAMESPACE=cpu-default
RUN --mount=type=cache,id=${BACKEND_CACHE_NAMESPACE}-target,target=/build/plugin/backend/target,sharing=locked \
    --mount=type=cache,id=${BACKEND_CACHE_NAMESPACE}-registry,target=/usr/local/cargo/registry,sharing=locked \
    --mount=type=cache,id=${BACKEND_CACHE_NAMESPACE}-git,target=/usr/local/cargo/git,sharing=locked \
    --mount=type=secret,id=backend-dependency-probe \
    set -eu; \
    if [ -f /run/secrets/backend-dependency-probe ]; then \
        printf 'OUROBOROS_DEPENDENCY_CACHE_MISS\n' >&2; \
        exit 83; \
    fi; \
    cargo build --release --locked --bin ouroboros-autoseg-plugin-backend; \
    cargo test --release --locked --no-run; \
    mkdir -p /cache-snapshot; \
    cd /; \
    cp -a --parents build/plugin/backend/target usr/local/cargo/registry usr/local/cargo/git /cache-snapshot/

FROM builder-tools AS builder-dependency-layer
COPY Cargo.toml Cargo.lock ./
COPY --from=builder-dependencies /cache-snapshot/ /

FROM builder-dependency-layer AS builder

# Notices depend on the manifest, lockfile and licensing configuration, so
# source-only changes can reuse them. Dummy targets keep Cargo metadata valid.
COPY about.toml about.hbs ./
RUN cargo about generate --locked --fail about.hbs -o THIRD_PARTY_LICENSES.txt

# The real build script derives engine provenance from Cargo.lock. Touch all
# replaced targets so restored Cargo artifacts cannot hide source changes.
COPY build.rs ./
COPY src/ src/
RUN touch build.rs src/lib.rs src/main.rs \
    && cargo build --release --locked --bin ouroboros-autoseg-plugin-backend

# ── CPU tests gate its candidate; main/release promotion requires both jobs ──
FROM builder AS test

COPY tests/ tests/
RUN cargo test --release --locked

# Export the same dependency snapshot used by the builder for archive saves.
FROM scratch AS cpu-cache-export
COPY --from=builder-dependencies /cache-snapshot/ /

# ── Stage 2: build the Rust backend for CUDA runtimes ────────────────────────
FROM ${CUDA_BUILDER_IMAGE} AS cuda-builder-base

ENV PATH="/root/.cargo/bin:${PATH}"
ARG RUST_VERSION=1.88.0

RUN --mount=type=secret,id=backend-dependency-probe \
    if [ -f /run/secrets/backend-dependency-probe ]; then \
        printf 'OUROBOROS_DEPENDENCY_CACHE_MISS\n' >&2; \
        exit 83; \
    fi; \
    apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    git \
    pkg-config \
    libssl-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN --mount=type=secret,id=backend-dependency-probe \
    if [ -f /run/secrets/backend-dependency-probe ]; then \
        printf 'OUROBOROS_DEPENDENCY_CACHE_MISS\n' >&2; \
        exit 83; \
    fi; \
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
    | sh -s -- -y --profile minimal --default-toolchain "${RUST_VERSION}"

WORKDIR /build/plugin/backend
ARG CANDLE_FEATURES=cuda,cudnn
ARG CUDA_COMPUTE_CAP=75
ENV CUDA_COMPUTE_CAP=${CUDA_COMPUTE_CAP}

FROM cuda-builder-base AS cuda-builder-tools
ARG CARGO_ABOUT_VERSION=0.9.3
RUN --mount=type=secret,id=backend-dependency-probe \
    if [ -f /run/secrets/backend-dependency-probe ]; then \
        printf 'OUROBOROS_DEPENDENCY_CACHE_MISS\n' >&2; \
        exit 83; \
    fi; \
    cargo install cargo-about --version "${CARGO_ABOUT_VERSION}" --locked --features cli \
    && mkdir -p src \
    && printf 'fn main() {}\n' > src/main.rs \
    && printf '' > src/lib.rs \
    && printf 'fn main() {}\n' > build.rs

FROM cuda-builder-tools AS cuda-builder-dependencies
COPY Cargo.toml Cargo.lock ./
ARG BACKEND_CACHE_NAMESPACE=cuda-default
RUN --mount=type=cache,id=${BACKEND_CACHE_NAMESPACE}-target,target=/build/plugin/backend/target,sharing=locked \
    --mount=type=cache,id=${BACKEND_CACHE_NAMESPACE}-registry,target=/root/.cargo/registry,sharing=locked \
    --mount=type=cache,id=${BACKEND_CACHE_NAMESPACE}-git,target=/root/.cargo/git,sharing=locked \
    --mount=type=secret,id=backend-dependency-probe \
    set -eu; \
    if [ -f /run/secrets/backend-dependency-probe ]; then \
        printf 'OUROBOROS_DEPENDENCY_CACHE_MISS\n' >&2; \
        exit 83; \
    fi; \
    test "${CANDLE_FEATURES}" = "cuda,cudnn" \
        || { echo "CANDLE_FEATURES must equal the canonical cuda,cudnn feature set" >&2; exit 1; }; \
    cargo build --release --locked --features "${CANDLE_FEATURES}" \
        --bin ouroboros-autoseg-plugin-backend; \
    mkdir -p /cache-snapshot; \
    cd /; \
    cp -a --parents build/plugin/backend/target root/.cargo/registry root/.cargo/git /cache-snapshot/

FROM cuda-builder-tools AS cuda-builder-dependency-layer
COPY Cargo.toml Cargo.lock ./
COPY --from=cuda-builder-dependencies /cache-snapshot/ /

FROM cuda-builder-dependency-layer AS cuda-builder

COPY about.toml about.hbs ./
RUN cargo about generate --locked --fail --features "${CANDLE_FEATURES}" \
    about.hbs -o THIRD_PARTY_LICENSES.txt

COPY build.rs ./
COPY src/ src/
RUN touch build.rs src/lib.rs src/main.rs \
    && cargo build --release --locked --features "${CANDLE_FEATURES}" \
    --bin ouroboros-autoseg-plugin-backend

FROM scratch AS cuda-cache-export
COPY --from=cuda-builder-dependencies /cache-snapshot/ /

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
COPY --from=test \
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
