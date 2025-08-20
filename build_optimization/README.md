# 🛠️ Essential Build Scripts for Memory Nexus Bare

## 🚀 **Quick Start**

### **Interactive Profile Selector** (Recommended)
```bash
cd scripts/
./00-interactive-profile-selector.sh
```

### **Direct Build Commands**
```bash
# Fast development iteration (5m 37s)
./scripts/build_dev_fast.sh

# CI/CD pipeline build (8m 47s)
./scripts/build_ci.sh
```

## 📂 **Essential Scripts**

| Script | Purpose | Build Time |
|--------|---------|------------|
| `00-interactive-profile-selector.sh` | Interactive menu to choose build profile | - |
| `build_dev_fast.sh` | Fastest compilation for rapid iteration | ~5m 30s |
| `build_ci.sh` | Optimized for CI/CD pipelines | ~8m 45s |

## ⚡ **Performance Benefits**

- **SIMD Optimizations**: All scripts use .cargo/config.toml settings (AVX2+FMA)
- **Parallel Compilation**: Maximized job parallelization
- **Profile Optimization**: Tailored for different use cases
- **Memory Efficiency**: Optimized memory usage during builds

## 🎯 **Integration with 27ms Pipeline**

These build scripts are optimized for:
- **Fast iteration**: dev-fast profile for rapid pipeline development
- **CI/CD deployment**: ci profile for production builds
- **SIMD performance**: Full AVX2+FMA optimization for vector operations
- **Memory efficiency**: Optimized for large codebases with vector processing

## 📋 **Usage Notes**

1. **Development**: Use `build_dev_fast.sh` for rapid iteration
2. **Testing**: CI profile provides good balance of speed and optimization
3. **Production**: Full release builds should use standard `cargo build --release`
4. **SIMD**: All profiles respect .cargo/config.toml SIMD settings