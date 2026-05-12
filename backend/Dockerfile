# ── Stage 1: build candle-sam3 binary ─────────────────────────────────────────
FROM rust:1-slim-bookworm AS candle-builder

# Repo and commit to build from (override at build-time via --build-arg)
ARG CANDLE_SAM3_REPO=https://github.com/den-sq/candle_sam3.git
ARG CANDLE_SAM3_REV=770d20ca

RUN apt-get update && apt-get install -y \
    pkg-config \
    libssl-dev \
    cmake \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Shallow clone (CANDLE_SAM3_REV is currently HEAD of main; depth=1 suffices).
# To pin to an older commit, switch to a full clone: git clone ... && git checkout.
RUN git clone --depth=1 "${CANDLE_SAM3_REPO}" . && \
    git checkout "${CANDLE_SAM3_REV}"

# Build only the sam3 example binary (CPU-only; no CUDA features required here).
# The Python backend forwards --cpu when torch.cuda.is_available() is False.
RUN cargo build --release -p candle-examples --example sam3

# ── Stage 2: Python runtime ────────────────────────────────────────────────────
FROM python:3.10.19-slim-bookworm

WORKDIR /app

RUN apt-get update && apt-get install -y \
    git \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install the candle-sam3 binary built in stage 1.
COPY --from=candle-builder /build/target/release/examples/sam3 /opt/candle-sam3/sam3
RUN chmod +x /opt/candle-sam3/sam3

# Tell the Python adapter where to find the binary.
ENV SAM3_BINARY_PATH=/opt/candle-sam3/sam3

# Mitigate pip CVEs (including CVE-2023-5752 and CVE-2025-8869).
RUN python -m pip install --no-cache-dir --upgrade "pip>=25.3"

COPY . .
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu121
RUN pip install --no-cache-dir "wheel>=0.46.2"
RUN pip install --no-cache-dir .

EXPOSE 8686

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8686"]
