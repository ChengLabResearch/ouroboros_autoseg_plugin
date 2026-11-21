ARG CPU_BUILDER_IMAGE=rust:1.88-slim-bookworm
ARG CPU_RUNTIME_IMAGE=debian:bookworm-slim
ARG CUDA_BUILDER_IMAGE=nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04
ARG CUDA_RUNTIME_IMAGE=nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

# ── Stage 1: build the Rust backend for CPU runtimes ─────────────────────────
# Rust 1.88+ is required by the locked image dependency; it also supports edition 2024 manifests.
FROM ${CPU_BUILDER_IMAGE} AS builder

RUN apt-get update && apt-get install -y \
    git \
    pkg-config \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# cargo-about generates the third-party licence notices shipped in the runtime
# image. Installing it before the lockfile is copied keeps it cached across
# dependency changes.
ARG CARGO_ABOUT_VERSION=0.9.3
RUN cargo install cargo-about --version "${CARGO_ABOUT_VERSION}" --locked --features cli

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

# Licence notices depend only on the lockfile and the cargo-about config, so
# they are generated before the plugin source is copied.
COPY about.toml about.hbs ./
RUN mkdir -p src \
    && printf 'fn main() {}\n' > src/main.rs \
    && printf '' > src/lib.rs \
    && cargo about generate --locked --fail about.hbs -o THIRD_PARTY_LICENSES.txt \
    && rm -rf src

COPY build.rs ./
COPY src/ src/
RUN touch src/lib.rs src/main.rs \
    && cargo build --release --locked --bin ouroboros-autoseg-plugin-backend

# ── Test target used by CI before either runtime image can publish ───────────
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

ARG CARGO_ABOUT_VERSION=0.9.3
RUN cargo install cargo-about --version "${CARGO_ABOUT_VERSION}" --locked --features cli

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

COPY about.toml about.hbs ./
RUN mkdir -p src \
    && printf 'fn main() {}\n' > src/main.rs \
    && printf '' > src/lib.rs \
    && cargo about generate --locked --fail --features "${CANDLE_FEATURES}" \
        about.hbs -o THIRD_PARTY_LICENSES.txt \
    && rm -rf src

COPY build.rs ./
COPY src/ src/
RUN touch src/lib.rs src/main.rs \
    && cargo build --release --locked --features "${CANDLE_FEATURES}" \
    --bin ouroboros-autoseg-plugin-backend

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

EXPOSE 8686

CMD ["/bin/sh", "-c", "printf '%s\\n' 'backend compiled capabilities: cuda=true cudnn=true features=cuda,cudnn' >&2; exec /usr/local/bin/ouroboros-autoseg-plugin-backend"]
