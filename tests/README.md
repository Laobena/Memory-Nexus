# Memory Nexus Test Suite

This directory contains all testing scripts for the Memory Nexus pipeline.

## Test Scripts

### 1. **test_integration.sh**
Complete 7-phase integration test suite covering:
- Build and compilation
- Unit tests
- Benchmarks
- Docker deployment
- API testing
- Load testing (100 concurrent requests)
- Performance validation

**Usage:**
```bash
./tests/test_integration.sh
```

### 2. **test_skeleton.sh**
Validates all 13 phases of the skeleton implementation:
- Core infrastructure (Phases 1-3)
- Database layer (Phase 4)
- Router (Phase 5)
- Build system (Phase 6)
- Pipeline components (Phases 7-10)
- Main application (Phase 11)
- Unified pipeline (Phase 12)
- Test scripts (Phase 13)

**Usage:**
```bash
./tests/test_skeleton.sh
```

### 3. **test_pipeline.sh**
Original pipeline test from Phase 6 with:
- CPU feature detection
- SIMD capability validation
- Memory optimization tests
- Performance benchmarks

**Usage:**
```bash
./tests/test_pipeline.sh
```

### 4. **validate_performance.sh**
Performance validation script that checks:
- CPU features (AVX2, SSE, etc.)
- SIMD operations (4-7x speedup)
- Lock-free cache performance
- Binary embeddings compression
- Memory pool efficiency
- Pipeline path performance

**Usage:**
```bash
./tests/validate_performance.sh
```

## Running All Tests

To run the complete test suite:

```bash
# Quick validation
./tests/validate_performance.sh

# Skeleton structure test
./tests/test_skeleton.sh

# Full integration test (includes Docker)
./tests/test_integration.sh
```

## Test Results

Expected performance targets:
- **Average Latency**: 6.5ms
- **Cache Hit Rate**: >70%
- **P99 Latency**: <20ms
- **Accuracy**: 94.8-98.4%
- **Concurrent Users**: 10,000+

## Test Profiles

The tests use different Cargo profiles:
- `dev-fast`: Quick compilation for development
- `release`: Full optimizations
- `bench`: Optimized for benchmarking

## Notes

- All tests disable `sccache` by setting `RUSTC_WRAPPER=""`
- Tests include timeout protection
- Color-coded output for easy reading
- Automatic cleanup on exit