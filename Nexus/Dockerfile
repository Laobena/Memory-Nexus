# Memory Nexus Bare - Minimal Docker Image

FROM rust:1.82-slim AS builder

# Install system dependencies
RUN apt-get update && apt-get install -y \
    pkg-config \
    libssl-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy Cargo files first for better caching
COPY Cargo.toml Cargo.lock ./
COPY crates/ ./crates/

# Copy source code
COPY src/ ./src/

# Build the application
RUN cargo build --release

# Runtime stage
FROM debian:bookworm-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    ca-certificates \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create app user
RUN useradd -m -u 1000 app

# Create app directory
WORKDIR /app

# Copy the binary
COPY --from=builder /app/target/release/memory-nexus-bare /usr/local/bin/memory-nexus-bare

# Copy configuration
COPY .env.example .env

# Change ownership
RUN chown -R app:app /app

# Switch to app user
USER app

# Expose ports
EXPOSE 8080 8082

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8082/health || exit 1

# Run the application
CMD ["memory-nexus-bare"]