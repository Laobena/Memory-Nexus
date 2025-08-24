# Multi-stage build for optimized image
FROM rust:1.81 as builder

# Install dependencies
RUN apt-get update && apt-get install -y \
    pkg-config \
    libssl-dev \
    cmake \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /usr/src/app

# Copy manifests
COPY Cargo.toml Cargo.lock ./

# Build dependencies (cached layer)
RUN mkdir src && \
    echo "fn main() {}" > src/main.rs && \
    cargo build --release && \
    rm -rf src

# Copy source code
COPY src ./src
COPY build.rs ./

# Build for release with all optimizations
ENV RUSTFLAGS="-C target-cpu=native -C opt-level=3 -C lto=fat -C codegen-units=1 -C link-arg=-s"
RUN cargo build --release --features full

# Runtime stage
FROM debian:bookworm-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    ca-certificates \
    libssl3 \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -u 1000 nexus

# Copy binary from builder
COPY --from=builder /usr/src/app/target/release/memory-nexus-bare /usr/local/bin/memory-nexus

# Create data directory
RUN mkdir -p /data && chown nexus:nexus /data

# Switch to non-root user
USER nexus

# Set environment variables
ENV RUST_LOG=info,memory_nexus_pipeline=debug
ENV RUST_BACKTRACE=1

# Expose ports
EXPOSE 8086 9090

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD ["/usr/local/bin/memory-nexus", "health"] || exit 1

# Run the binary
ENTRYPOINT ["/usr/local/bin/memory-nexus"]
CMD ["--config", "/data/config.toml"]