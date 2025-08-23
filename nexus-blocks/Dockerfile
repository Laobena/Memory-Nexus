# Multi-stage Docker build for Memory Nexus Blocks
# Optimized for <5MB final image size with maximum performance

# ===== Stage 1: Builder =====
FROM rust:1.75-slim AS builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    pkg-config \
    libssl-dev \
    cmake \
    gcc \
    g++ \
    clang \
    lld \
    && rm -rf /var/lib/apt/lists/*

# Set up build environment
ENV RUSTFLAGS="-C target-cpu=native -C link-arg=-fuse-ld=lld -C opt-level=3"
ENV CARGO_BUILD_JOBS=8
ENV CARGO_REGISTRIES_CRATES_IO_PROTOCOL=sparse

# Create app directory
WORKDIR /build

# Copy workspace files
COPY Cargo.toml /build/
COPY ../Cargo.toml /build/parent/
COPY ../src /build/parent/src

# Create dummy source files for dependency caching
RUN mkdir -p src && \
    echo "pub fn lib() {}" > src/lib.rs

# Build dependencies only (cached layer)
WORKDIR /build
RUN cargo build --release --features full && \
    rm -rf src

# Copy actual source code
COPY src ./src
COPY benches ./benches

# Touch source to ensure rebuild
RUN touch src/lib.rs

# Build the final binary with all optimizations
RUN cargo build --release --features full

# Strip symbols for smaller binary (if they exist)
RUN if [ -f /build/target/release/libnexus_blocks.so ]; then strip /build/target/release/libnexus_blocks.so; fi && \
    if [ -f /build/target/release/libnexus_blocks.a ]; then strip /build/target/release/libnexus_blocks.a; fi

# ===== Stage 2: Runtime =====
FROM gcr.io/distroless/cc-debian12:nonroot AS runtime

# Copy the libraries (if they exist)
COPY --from=builder /build/target/release/libnexus_blocks* /lib/

# Set library path
ENV LD_LIBRARY_PATH=/lib

# Health check (if running as service)
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD ["/bin/sh", "-c", "test -f /lib/libnexus_blocks.so"]

# ===== Stage 3: Scratch (Ultra-minimal) =====
FROM scratch AS minimal

# Copy only the shared library
COPY --from=builder /build/target/release/libnexus_blocks* /

# This image is <5MB and contains only the library

# ===== Stage 4: Development =====
FROM rust:1.75 AS development

# Install development tools
RUN apt-get update && apt-get install -y \
    pkg-config \
    libssl-dev \
    cmake \
    gcc \
    g++ \
    clang \
    lld \
    lldb \
    valgrind \
    perf-tools-unstable \
    linux-tools-generic \
    heaptrack \
    && rm -rf /var/lib/apt/lists/*

# Install Rust tools
RUN rustup component add \
    rustfmt \
    clippy \
    rust-src \
    rust-analyzer

# Install cargo tools
RUN cargo install \
    cargo-watch \
    cargo-edit \
    cargo-expand \
    cargo-flamegraph \
    cargo-criterion \
    cargo-audit \
    cargo-outdated

WORKDIR /workspace

# Copy source
COPY . .

# Set development environment
ENV RUST_BACKTRACE=1
ENV RUST_LOG=debug
ENV CARGO_WATCH_CLEAR=true

# Development command
CMD ["cargo", "watch", "-x", "test", "-x", "clippy"]

# ===== Stage 5: Benchmark =====
FROM rust:1.75-slim AS benchmark

# Install minimal dependencies
RUN apt-get update && apt-get install -y \
    pkg-config \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /bench

# Copy built artifacts and benchmarks
COPY --from=builder /build/target/release/deps /bench/target/release/deps
COPY benches /bench/benches
COPY Cargo.toml /bench/

# Run benchmarks
CMD ["cargo", "bench", "--features", "full"]

# ===== Stage 6: PGO Builder =====
FROM builder AS pgo-builder

# First pass: Build with PGO instrumentation
ENV RUSTFLAGS="-C target-cpu=native -C profile-generate=/tmp/pgo-data"
RUN cargo build --release --features full

# Run representative workload to generate profile data
RUN cargo test --release --features full || true
RUN cargo bench --features full -- --profile-time 10 || true

# Second pass: Build with PGO optimization
ENV RUSTFLAGS="-C target-cpu=native -C profile-use=/tmp/pgo-data"
RUN cargo clean && \
    cargo build --release --features full

# Strip for production (if exists)
RUN if [ -f /build/target/release/libnexus_blocks.so ]; then strip /build/target/release/libnexus_blocks.so; fi

# ===== Stage 7: WASM Target =====
FROM rust:1.75 AS wasm-builder

# Install wasm-pack
RUN curl https://rustwasm.github.io/wasm-pack/installer/init.sh -sSf | sh

WORKDIR /build

# Copy source
COPY . .

# Build for WASM
WORKDIR /build
RUN wasm-pack build --target web --release

# ===== Default: Multi-target =====
FROM runtime AS default

# Labels
LABEL org.opencontainers.image.title="Memory Nexus Blocks"
LABEL org.opencontainers.image.description="High-performance pipeline blocks for AI memory systems"
LABEL org.opencontainers.image.version="0.1.0"
LABEL org.opencontainers.image.authors="Memory Nexus Team"
LABEL org.opencontainers.image.source="https://github.com/memory-nexus/nexus-blocks"

# The default image provides the runtime libraries