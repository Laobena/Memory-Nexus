# Memory Nexus Build & Utility Scripts

This directory contains build and utility scripts for the Memory Nexus pipeline.

## Build Scripts

### 1. **build_optimized.sh**
Production build script with full optimizations:
- Architecture detection (x86_64, ARM64)
- CPU feature detection (AVX2, AVX-512)
- Optimization flags (LTO, native CPU, opt-level 3)
- Binary stripping for size
- PGO/BOLT suggestions

**Usage:**
```bash
./scripts/build_optimized.sh
```

### 2. **quick_test.sh**
Fast development iteration helper:
- Uses `dev-fast` profile for quick builds
- Runs core tests only
- Optional Docker services with `--with-services`

**Usage:**
```bash
./scripts/quick_test.sh
./scripts/quick_test.sh --with-services
```

### 3. **performance_monitor.sh**
Real-time performance monitoring (if exists)

## Build Optimization Scripts

Additional optimization scripts in `/build_optimization/scripts/`:

### 1. **build_dev_fast.sh**
- Fastest compilation for development
- opt-level=0, 256 codegen-units
- No debug symbols

### 2. **build_ci.sh**
- Optimized for CI/CD pipelines
- Balanced speed and optimization

### 3. **00-interactive-profile-selector.sh**
- Interactive build profile selection
- Helps choose the right profile

## Build Profiles

| Profile | Compilation Speed | Runtime Speed | Use Case |
|---------|------------------|---------------|----------|
| dev-fast | ⚡⚡⚡ | 🐢 | Development |
| release | 🐢 | ⚡⚡⚡ | Production |
| bench | 🐢 | ⚡⚡⚡ | Benchmarking |

## Environment Variables

```bash
# Disable sccache if having issues
export RUSTC_WRAPPER=""

# Set optimization flags
export RUSTFLAGS="-C target-cpu=native -C opt-level=3"
```

## Common Commands

```bash
# Quick development build
./scripts/quick_test.sh

# Production build
./scripts/build_optimized.sh

# Run with specific profile
cargo build --profile=dev-fast
cargo build --release
```