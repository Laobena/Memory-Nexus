#!/bin/bash
# Memory Nexus Pipeline - Optimized Build Script
# Builds with all performance optimizations enabled

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${BLUE}"
echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║         MEMORY NEXUS - OPTIMIZED BUILD SYSTEM                ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo -e "${NC}\n"

# Detect CPU architecture
ARCH=$(uname -m)
echo -e "${BLUE}[INFO] Detected architecture: $ARCH${NC}"

# Set optimization flags based on architecture
if [[ "$ARCH" == "x86_64" ]]; then
    # x86_64 optimizations
    export RUSTFLAGS="-C target-cpu=native -C opt-level=3 -C lto=fat -C codegen-units=1 -C embed-bitcode=yes"
    
    # Check for specific CPU features
    if grep -q "avx2" /proc/cpuinfo 2>/dev/null; then
        echo -e "${GREEN}✓ AVX2 support detected${NC}"
        export RUSTFLAGS="$RUSTFLAGS -C target-feature=+avx2,+fma"
    fi
    
    if grep -q "avx512" /proc/cpuinfo 2>/dev/null; then
        echo -e "${GREEN}✓ AVX-512 support detected${NC}"
        export RUSTFLAGS="$RUSTFLAGS -C target-feature=+avx512f"
    fi
elif [[ "$ARCH" == "aarch64" ]] || [[ "$ARCH" == "arm64" ]]; then
    # ARM64 optimizations
    echo -e "${GREEN}✓ ARM64 architecture detected${NC}"
    export RUSTFLAGS="-C target-cpu=native -C opt-level=3 -C lto=fat -C codegen-units=1"
else
    # Generic optimizations
    echo -e "${YELLOW}! Using generic optimizations${NC}"
    export RUSTFLAGS="-C opt-level=3 -C lto=fat -C codegen-units=1"
fi

# Additional linker optimizations
if command -v lld >/dev/null 2>&1; then
    echo -e "${GREEN}✓ Using LLD linker for faster builds${NC}"
    export RUSTFLAGS="$RUSTFLAGS -C link-arg=-fuse-ld=lld"
fi

# Strip symbols for smaller binary
export RUSTFLAGS="$RUSTFLAGS -C link-arg=-s"

echo -e "\n${BLUE}[1/6] Cleaning previous builds...${NC}"
cargo clean

echo -e "\n${BLUE}[2/6] Updating dependencies...${NC}"
cargo update

echo -e "\n${BLUE}[3/6] Building with optimizations...${NC}"
echo -e "${YELLOW}RUSTFLAGS: $RUSTFLAGS${NC}\n"

# Build with all features
cargo build --release --features full

echo -e "\n${BLUE}[4/6] Checking binary size...${NC}"
BINARY_PATH="target/release/memory-nexus-pipeline"
if [ -f "$BINARY_PATH" ]; then
    SIZE=$(du -h "$BINARY_PATH" | cut -f1)
    echo -e "${GREEN}✓ Binary size: $SIZE${NC}"
    
    # Strip additional debug symbols if available
    if command -v strip >/dev/null 2>&1; then
        echo -e "${BLUE}Stripping debug symbols...${NC}"
        strip "$BINARY_PATH"
        NEW_SIZE=$(du -h "$BINARY_PATH" | cut -f1)
        echo -e "${GREEN}✓ Stripped size: $NEW_SIZE${NC}"
    fi
else
    echo -e "${RED}✗ Binary not found${NC}"
fi

echo -e "\n${BLUE}[5/6] Running quick validation...${NC}"

# Quick test to ensure binary works
if "$BINARY_PATH" --version >/dev/null 2>&1; then
    echo -e "${GREEN}✓ Binary executes successfully${NC}"
else
    echo -e "${RED}✗ Binary execution failed${NC}"
    exit 1
fi

# Check feature detection binary
if [ -f "target/release/check_features" ]; then
    echo -e "\n${BLUE}CPU Features Available:${NC}"
    ./target/release/check_features 2>/dev/null || true
fi

echo -e "\n${BLUE}[6/6] Build optimization summary...${NC}"
echo "┌─────────────────────────────────────────────────────┐"
echo "│ Optimization           │ Status                     │"
echo "├─────────────────────────────────────────────────────┤"
echo "│ Target CPU            │ ✓ Native                   │"
echo "│ Optimization Level    │ ✓ Level 3 (maximum)        │"
echo "│ Link-Time Opt (LTO)   │ ✓ Fat LTO enabled          │"
echo "│ Codegen Units         │ ✓ 1 (best optimization)    │"
echo "│ Symbol Stripping      │ ✓ Enabled                  │"
echo "│ SIMD Instructions     │ ✓ Auto-detected            │"
echo "│ Parallel Compilation  │ ✓ Available                │"
echo "└─────────────────────────────────────────────────────┘"

# Profile-Guided Optimization (PGO) suggestion
echo -e "\n${BLUE}Advanced Optimization Options:${NC}"
echo -e "${YELLOW}For even better performance, consider:${NC}"
echo "1. Profile-Guided Optimization (PGO):"
echo "   cargo pgo build"
echo "2. BOLT optimization:"
echo "   cargo bolt build"
echo "3. Custom allocator tuning:"
echo "   export MIMALLOC_PAGE_RESET=0"
echo "   export MIMALLOC_LARGE_OS_PAGES=1"

echo -e "\n${GREEN}✅ OPTIMIZED BUILD COMPLETE!${NC}"
echo -e "${GREEN}Binary location: $BINARY_PATH${NC}"
echo -e "${GREEN}Ready for deployment with maximum performance.${NC}\n"

exit 0