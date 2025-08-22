# Memory Nexus Project Structure

## 📁 Organized Directory Layout

```
nexus/
├── src/                        # Source code (50+ Rust modules)
│   ├── core/                   # Core types and optimizations
│   │   ├── types.rs           # High-performance types
│   │   ├── simd_ops.rs        # SIMD operations
│   │   ├── binary_embeddings.rs
│   │   ├── lock_free_cache.rs
│   │   └── aligned_alloc.rs
│   ├── pipeline/               # Pipeline components
│   │   ├── unified_pipeline.rs # Main pipeline orchestrator
│   │   ├── intelligent_router.rs
│   │   ├── fusion.rs
│   │   ├── preprocessor_enhanced.rs
│   │   └── search_orchestrator.rs
│   ├── engines/                # 4 processing engines
│   │   ├── accuracy.rs
│   │   ├── intelligence.rs
│   │   ├── learning.rs
│   │   └── mining.rs
│   ├── database/               # Database layer
│   │   ├── connection_pool.rs
│   │   ├── enhanced_pool.rs
│   │   └── database_connections.rs
│   ├── optimizations/          # Performance optimizations
│   │   └── memory_pool.rs
│   ├── api/                    # REST API
│   │   ├── routes.rs
│   │   └── middleware/
│   ├── monitoring/             # Metrics and tracing
│   └── main.rs                 # Application entry point
│
├── tests/                      # All test scripts
│   ├── README.md              # Test documentation
│   ├── test_integration.sh    # 7-phase integration test
│   ├── test_skeleton.sh       # Skeleton validation
│   ├── test_pipeline.sh       # Pipeline test
│   ├── validate_performance.sh # Performance validation
│   └── integration_test.rs    # Rust integration tests
│
├── scripts/                    # Build and utility scripts
│   ├── README.md              # Scripts documentation
│   ├── build_optimized.sh     # Production build
│   ├── quick_test.sh          # Dev iteration helper
│   └── performance_monitor.sh # Performance monitoring
│
├── docs/                       # Documentation
│   ├── README.md              # Documentation index
│   ├── goal.md                # Architecture goals
│   ├── ROADMAP.md             # Development roadmap
│   ├── dependencies.md        # External dependencies
│   ├── SKELETON_TEST_RESULTS.md
│   ├── SKELETON_ADVANTAGES.md
│   └── RUST_OPTIMIZATIONS_USED.md
│
├── benches/                    # Benchmarks
│   ├── core_types.rs
│   ├── intelligent_router.rs
│   └── memory_pools.rs
│
├── build_optimization/         # Additional build scripts
│   └── scripts/
│       ├── build_dev_fast.sh
│       ├── build_ci.sh
│       └── 00-interactive-profile-selector.sh
│
├── monitoring/                 # Monitoring configurations
│
├── k8s/                       # Kubernetes configs (if needed)
│
├── build.rs                   # Cargo build script (CPU detection)
├── Cargo.toml                 # Project dependencies
├── Cargo.lock                 # Locked dependencies
├── Dockerfile                 # Production Docker image
├── docker-compose.yml         # Complete stack deployment
├── k8s-deployment.yaml        # Kubernetes deployment
├── run_tests.sh              # Master test runner
├── README.md                 # Project overview
├── CLAUDE.md                 # AI assistant context
└── LICENSE                   # License file
```

## 🎯 Key Directories

### `/src` - Source Code
- **50+ Rust modules** implementing the complete pipeline
- Organized by functionality
- Clean separation of concerns

### `/tests` - Test Suite
- Shell scripts for different test scenarios
- Rust integration tests
- Performance validation tools

### `/scripts` - Build & Utilities
- Build optimization scripts
- Development helpers
- Performance monitoring

### `/docs` - Documentation
- Architecture documentation
- Performance analysis
- Implementation status

### `/benches` - Benchmarks
- Criterion benchmarks
- Performance tests
- Optimization validation

## 🚀 Quick Commands

### Running Tests
```bash
# Interactive test menu
./run_tests.sh

# Specific tests
./tests/validate_performance.sh
./tests/test_skeleton.sh
./tests/test_integration.sh
```

### Building
```bash
# Development
cargo build --profile=dev-fast

# Production
./scripts/build_optimized.sh

# Docker
docker-compose up -d
```

### Development Workflow
```bash
# Quick iteration
./scripts/quick_test.sh

# Full validation
./run_tests.sh  # Choose option 5 for all tests
```

## 📊 File Count Summary

- **Source files**: ~50 Rust modules
- **Test scripts**: 5 shell scripts + Rust tests
- **Build scripts**: 6 scripts
- **Documentation**: 7 markdown files
- **Configuration**: Docker, Cargo, K8s files

## 🧹 Clean Organization Benefits

1. **Clear separation** - Tests, scripts, docs in dedicated directories
2. **Easy navigation** - Logical structure matches functionality
3. **Maintainable** - Each directory has its own README
4. **Scalable** - Easy to add new components
5. **Professional** - Production-ready structure

## 📝 Notes

- `build.rs` remains in root (required by Cargo)
- `Cargo.toml` and `Cargo.lock` in root (required)
- Docker files in root for easy deployment
- `run_tests.sh` in root for quick access
- All other files organized by type