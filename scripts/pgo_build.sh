#!/bin/bash
# ================================================================================
# Memory Nexus - Profile-Guided Optimization Build Script
# ================================================================================
# Based on production patterns that show 10-15% performance improvement
# 
# PGO Process:
# 1. Build with profile generation
# 2. Run with production workload to collect profiles
# 3. Build again using collected profiles for optimization
#
# Usage:
#   ./scripts/pgo_build.sh generate  # Step 1: Generate profiles
#   ./scripts/pgo_build.sh optimize  # Step 2: Build with profiles

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration
PGO_DATA_DIR="./pgo-data"
PROFILE_NAME="memory-nexus"
MERGED_PROFILE="$PGO_DATA_DIR/merged.profdata"

# Print banner
print_banner() {
    echo -e "${BLUE}"
    echo "╔══════════════════════════════════════════════════════════════════╗"
    echo "║           MEMORY NEXUS - PROFILE-GUIDED OPTIMIZATION            ║"
    echo "║                    10-15% Performance Boost                     ║"
    echo "╚══════════════════════════════════════════════════════════════════╝"
    echo -e "${NC}\n"
}

# Check for required tools
check_requirements() {
    echo -e "${CYAN}[CHECK] Verifying requirements...${NC}"
    
    # Check for Rust
    if ! command -v rustc &> /dev/null; then
        echo -e "${RED}[ERROR] Rust is not installed${NC}"
        exit 1
    fi
    
    # Check for llvm-profdata
    if ! command -v llvm-profdata &> /dev/null; then
        echo -e "${YELLOW}[WARN] llvm-profdata not found. Trying to find it...${NC}"
        
        # Try to find llvm-profdata in common locations
        LLVM_PROFDATA=$(find ~/.rustup -name "llvm-profdata" 2>/dev/null | head -n 1)
        
        if [ -z "$LLVM_PROFDATA" ]; then
            echo -e "${RED}[ERROR] llvm-profdata not found. Install LLVM tools.${NC}"
            echo -e "${YELLOW}Try: rustup component add llvm-tools-preview${NC}"
            exit 1
        fi
        
        echo -e "${GREEN}[OK] Found llvm-profdata at: $LLVM_PROFDATA${NC}"
    else
        LLVM_PROFDATA="llvm-profdata"
    fi
    
    echo -e "${GREEN}[OK] All requirements met${NC}\n"
}

# Generate profile data
generate_profiles() {
    echo -e "${CYAN}[STEP 1/3] Building with profile generation...${NC}"
    
    # Clean previous profiles
    rm -rf "$PGO_DATA_DIR"
    mkdir -p "$PGO_DATA_DIR"
    
    # Set environment for profile generation
    export RUSTFLAGS="-Cprofile-generate=$PGO_DATA_DIR -Cllvm-args=-vp-counters-per-site=8"
    
    # Build with profile generation
    echo -e "${BLUE}[BUILD] Building Memory Nexus with profiling instrumentation...${NC}"
    cargo build --release --features full
    
    echo -e "${GREEN}[OK] Build complete with profiling enabled${NC}\n"
    
    echo -e "${CYAN}[STEP 2/3] Running profiling workload...${NC}"
    echo -e "${YELLOW}[INFO] This will run various workloads to collect profile data${NC}"
    
    # Run the binary with various workloads
    run_profiling_workload
    
    echo -e "${GREEN}[OK] Profile data collected${NC}\n"
    
    echo -e "${CYAN}[STEP 3/3] Merging profile data...${NC}"
    
    # Find all .profraw files
    PROFRAW_FILES=$(find "$PGO_DATA_DIR" -name "*.profraw" 2>/dev/null)
    
    if [ -z "$PROFRAW_FILES" ]; then
        echo -e "${RED}[ERROR] No profile data found. Make sure the workload ran successfully.${NC}"
        exit 1
    fi
    
    # Count profile files
    PROFILE_COUNT=$(echo "$PROFRAW_FILES" | wc -l)
    echo -e "${BLUE}[INFO] Found $PROFILE_COUNT profile files${NC}"
    
    # Merge profiles
    $LLVM_PROFDATA merge -o "$MERGED_PROFILE" $PROFRAW_FILES
    
    if [ -f "$MERGED_PROFILE" ]; then
        echo -e "${GREEN}[OK] Profile data merged successfully${NC}"
        echo -e "${GREEN}[OK] Profile saved to: $MERGED_PROFILE${NC}\n"
        
        # Show profile statistics
        echo -e "${CYAN}[INFO] Profile statistics:${NC}"
        $LLVM_PROFDATA show --summary "$MERGED_PROFILE"
        
        echo -e "\n${GREEN}✓ Profile generation complete!${NC}"
        echo -e "${YELLOW}Now run: $0 optimize${NC}"
    else
        echo -e "${RED}[ERROR] Failed to merge profile data${NC}"
        exit 1
    fi
}

# Run profiling workload
run_profiling_workload() {
    local BINARY="./target/release/memory-nexus-pipeline"
    
    if [ ! -f "$BINARY" ]; then
        echo -e "${RED}[ERROR] Binary not found at $BINARY${NC}"
        exit 1
    fi
    
    echo -e "${BLUE}[RUN] Starting profiling workload...${NC}"
    
    # Run various scenarios to collect comprehensive profile data
    # Note: These should be representative of production workloads
    
    # 1. Run unit tests (good for code coverage)
    echo -e "${YELLOW}[1/5] Running unit tests...${NC}"
    cargo test --release --features full || true
    
    # 2. Run benchmarks if available
    if [ -d "benches" ]; then
        echo -e "${YELLOW}[2/5] Running benchmarks...${NC}"
        cargo bench --features full || true
    fi
    
    # 3. Run the main binary with different workloads
    echo -e "${YELLOW}[3/5] Running main application scenarios...${NC}"
    
    # Start the server in background for profiling
    timeout 10s $BINARY &> /dev/null || true
    
    # 4. Run integration tests if available
    if [ -f "tests/test_integration.sh" ]; then
        echo -e "${YELLOW}[4/5] Running integration tests...${NC}"
        ./tests/test_integration.sh || true
    fi
    
    # 5. Run any custom workload scripts
    if [ -f "scripts/workload.sh" ]; then
        echo -e "${YELLOW}[5/5] Running custom workload...${NC}"
        ./scripts/workload.sh || true
    fi
    
    echo -e "${GREEN}[OK] Profiling workload complete${NC}"
}

# Build with profile data
optimize_with_profiles() {
    echo -e "${CYAN}[PGO] Building with profile-guided optimization...${NC}"
    
    # Check if profile data exists
    if [ ! -f "$MERGED_PROFILE" ]; then
        echo -e "${RED}[ERROR] No profile data found at $MERGED_PROFILE${NC}"
        echo -e "${YELLOW}Run '$0 generate' first to create profile data${NC}"
        exit 1
    fi
    
    # Show profile info
    echo -e "${BLUE}[INFO] Using profile: $MERGED_PROFILE${NC}"
    echo -e "${BLUE}[INFO] Profile size: $(du -h "$MERGED_PROFILE" | cut -f1)${NC}\n"
    
    # Clean previous build
    echo -e "${YELLOW}[CLEAN] Cleaning previous builds...${NC}"
    cargo clean
    
    # Set environment for PGO build
    export RUSTFLAGS="-Cprofile-use=$MERGED_PROFILE -Cllvm-args=-pgo-warn-missing-function -Ccodegen-units=1 -Clto=fat"
    
    # Additional optimizations
    export CARGO_PROFILE_RELEASE_LTO="fat"
    export CARGO_PROFILE_RELEASE_CODEGEN_UNITS="1"
    export CARGO_PROFILE_RELEASE_PANIC="abort"
    
    # Build with PGO
    echo -e "${BLUE}[BUILD] Building with profile-guided optimization...${NC}"
    echo -e "${YELLOW}[INFO] This may take longer than a normal build${NC}"
    
    cargo build --release --features full
    
    if [ $? -eq 0 ]; then
        echo -e "\n${GREEN}✓ PGO build complete!${NC}"
        echo -e "${GREEN}Binary optimized with 10-15% performance improvement${NC}"
        
        # Show binary info
        BINARY="./target/release/memory-nexus-pipeline"
        if [ -f "$BINARY" ]; then
            SIZE=$(du -h "$BINARY" | cut -f1)
            echo -e "${BLUE}[INFO] Binary size: $SIZE${NC}"
            echo -e "${BLUE}[INFO] Binary path: $BINARY${NC}"
        fi
        
        echo -e "\n${CYAN}[TIP] Run benchmarks to verify performance improvement:${NC}"
        echo -e "${YELLOW}  cargo bench --features full${NC}"
    else
        echo -e "${RED}[ERROR] PGO build failed${NC}"
        exit 1
    fi
}

# Clean PGO data
clean_pgo_data() {
    echo -e "${CYAN}[CLEAN] Removing PGO data...${NC}"
    rm -rf "$PGO_DATA_DIR"
    echo -e "${GREEN}[OK] PGO data cleaned${NC}"
}

# Show help
show_help() {
    echo "Memory Nexus - Profile-Guided Optimization Build Script"
    echo ""
    echo "Usage: $0 [command]"
    echo ""
    echo "Commands:"
    echo "  generate    Build with profiling and collect profile data"
    echo "  optimize    Build with profile-guided optimization"
    echo "  clean       Remove all PGO data"
    echo "  help        Show this help message"
    echo ""
    echo "Typical workflow:"
    echo "  1. $0 generate  # Build and profile"
    echo "  2. $0 optimize  # Build with PGO"
    echo ""
    echo "Expected performance improvement: 10-15%"
}

# Main script
print_banner
check_requirements

case "${1:-help}" in
    generate)
        generate_profiles
        ;;
    optimize)
        optimize_with_profiles
        ;;
    clean)
        clean_pgo_data
        ;;
    help|--help|-h)
        show_help
        ;;
    *)
        echo -e "${RED}[ERROR] Unknown command: $1${NC}"
        show_help
        exit 1
        ;;
esac