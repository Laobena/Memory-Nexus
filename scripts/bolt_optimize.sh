#!/bin/bash
# ================================================================================
# Memory Nexus - BOLT Binary Optimization Script
# ================================================================================
# BOLT (Binary Optimization and Layout Tool) provides post-link optimization
# Can provide 5-20% additional performance on top of PGO
#
# Requirements:
#   - LLVM BOLT (usually comes with LLVM 14+)
#   - A PGO-optimized binary
#   - perf tool for profiling
#
# Usage:
#   ./scripts/bolt_optimize.sh profile  # Collect BOLT profile
#   ./scripts/bolt_optimize.sh optimize # Apply BOLT optimization

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

# Configuration
BINARY_PATH="./target/release/memory-nexus-pipeline"
BOLT_PROFILE="bolt-profile.fdata"
BOLT_OUTPUT="${BINARY_PATH}.bolt"
PERF_DATA="perf.data"

# Print banner
print_banner() {
    echo -e "${BLUE}"
    echo "╔══════════════════════════════════════════════════════════════════╗"
    echo "║              MEMORY NEXUS - BOLT OPTIMIZATION                   ║"
    echo "║                5-20% Additional Performance                     ║"
    echo "╚══════════════════════════════════════════════════════════════════╝"
    echo -e "${NC}\n"
}

# Check requirements
check_requirements() {
    echo -e "${CYAN}[CHECK] Verifying BOLT requirements...${NC}"
    
    # Check for BOLT
    if ! command -v llvm-bolt &> /dev/null; then
        echo -e "${YELLOW}[WARN] llvm-bolt not found in PATH${NC}"
        
        # Try common locations
        BOLT_PATHS=(
            "/usr/bin/llvm-bolt"
            "/usr/local/bin/llvm-bolt"
            "/opt/llvm/bin/llvm-bolt"
            "$HOME/.local/bin/llvm-bolt"
        )
        
        for path in "${BOLT_PATHS[@]}"; do
            if [ -f "$path" ]; then
                LLVM_BOLT="$path"
                echo -e "${GREEN}[OK] Found BOLT at: $LLVM_BOLT${NC}"
                break
            fi
        done
        
        if [ -z "$LLVM_BOLT" ]; then
            echo -e "${RED}[ERROR] BOLT not found. Install LLVM 14+ with BOLT support${NC}"
            echo -e "${YELLOW}Ubuntu/Debian: sudo apt install llvm-bolt${NC}"
            echo -e "${YELLOW}Arch: sudo pacman -S llvm${NC}"
            echo -e "${YELLOW}macOS: brew install llvm${NC}"
            exit 1
        fi
    else
        LLVM_BOLT="llvm-bolt"
    fi
    
    # Check for perf (Linux only)
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        if ! command -v perf &> /dev/null; then
            echo -e "${YELLOW}[WARN] perf not found. Install linux-tools-common${NC}"
            echo -e "${YELLOW}Ubuntu/Debian: sudo apt install linux-tools-common linux-tools-generic${NC}"
            echo -e "${YELLOW}Will use alternative profiling method${NC}"
            USE_PERF=false
        else
            USE_PERF=true
        fi
    else
        echo -e "${YELLOW}[INFO] Non-Linux system detected. Using sampling profiler${NC}"
        USE_PERF=false
    fi
    
    # Check if binary exists
    if [ ! -f "$BINARY_PATH" ]; then
        echo -e "${RED}[ERROR] Binary not found at: $BINARY_PATH${NC}"
        echo -e "${YELLOW}Build the project first with: cargo build --release${NC}"
        exit 1
    fi
    
    echo -e "${GREEN}[OK] Requirements verified${NC}\n"
}

# Collect BOLT profile
collect_bolt_profile() {
    echo -e "${CYAN}[BOLT] Collecting execution profile...${NC}"
    
    if [ "$USE_PERF" = true ]; then
        collect_perf_profile
    else
        collect_sampling_profile
    fi
}

# Collect profile using perf (Linux)
collect_perf_profile() {
    echo -e "${BLUE}[PERF] Using perf for profiling...${NC}"
    
    # Clean old profile
    rm -f "$PERF_DATA"
    
    echo -e "${YELLOW}[INFO] Starting profiling session...${NC}"
    echo -e "${YELLOW}[INFO] Run your workload now. Press Ctrl+C when done.${NC}"
    
    # Start perf recording
    sudo perf record -e cycles:u -j any,u -o "$PERF_DATA" -- "$BINARY_PATH" &
    PERF_PID=$!
    
    # Wait for user to stop
    echo -e "${CYAN}[WAIT] Collecting profile data...${NC}"
    sleep 10  # Run for 10 seconds or until interrupted
    
    # Stop perf
    sudo kill -INT $PERF_PID 2>/dev/null || true
    wait $PERF_PID 2>/dev/null || true
    
    if [ -f "$PERF_DATA" ]; then
        echo -e "${GREEN}[OK] Perf data collected${NC}"
        
        # Convert perf data to BOLT format
        echo -e "${BLUE}[CONVERT] Converting perf data to BOLT format...${NC}"
        sudo perf2bolt "$BINARY_PATH" -p "$PERF_DATA" -o "$BOLT_PROFILE"
        
        if [ -f "$BOLT_PROFILE" ]; then
            echo -e "${GREEN}[OK] BOLT profile created: $BOLT_PROFILE${NC}"
        else
            echo -e "${RED}[ERROR] Failed to create BOLT profile${NC}"
            exit 1
        fi
    else
        echo -e "${RED}[ERROR] No perf data collected${NC}"
        exit 1
    fi
}

# Collect profile using BOLT's built-in sampler
collect_sampling_profile() {
    echo -e "${BLUE}[SAMPLE] Using BOLT sampling profiler...${NC}"
    
    # Create instrumented binary
    echo -e "${YELLOW}[INFO] Creating instrumented binary...${NC}"
    $LLVM_BOLT "$BINARY_PATH" \
        -instrument \
        -instrumentation-file="$BOLT_PROFILE" \
        -o "${BINARY_PATH}.instrumented"
    
    if [ ! -f "${BINARY_PATH}.instrumented" ]; then
        echo -e "${RED}[ERROR] Failed to create instrumented binary${NC}"
        exit 1
    fi
    
    echo -e "${GREEN}[OK] Instrumented binary created${NC}"
    
    # Run instrumented binary
    echo -e "${CYAN}[RUN] Running instrumented binary to collect profile...${NC}"
    echo -e "${YELLOW}[INFO] This will run various workloads${NC}"
    
    # Make it executable
    chmod +x "${BINARY_PATH}.instrumented"
    
    # Run workload
    run_bolt_workload "${BINARY_PATH}.instrumented"
    
    if [ -f "$BOLT_PROFILE" ]; then
        echo -e "${GREEN}[OK] BOLT profile collected: $BOLT_PROFILE${NC}"
        
        # Clean up instrumented binary
        rm -f "${BINARY_PATH}.instrumented"
    else
        echo -e "${RED}[ERROR] No profile data generated${NC}"
        exit 1
    fi
}

# Run workload for profiling
run_bolt_workload() {
    local BINARY="$1"
    
    echo -e "${BLUE}[WORKLOAD] Running profiling workload...${NC}"
    
    # Run the binary with typical workloads
    # This should represent real production usage
    
    # 1. Start server and send requests
    echo -e "${YELLOW}[1/3] Running server workload...${NC}"
    timeout 10s "$BINARY" &> /dev/null || true
    
    # 2. If there are benchmarks, run them
    if [ -f "target/release/deps/benchmark" ]; then
        echo -e "${YELLOW}[2/3] Running benchmarks...${NC}"
        ./target/release/deps/benchmark || true
    fi
    
    # 3. Run any test scenarios
    echo -e "${YELLOW}[3/3] Running test scenarios...${NC}"
    if [ -f "scripts/workload.sh" ]; then
        ./scripts/workload.sh || true
    fi
    
    echo -e "${GREEN}[OK] Workload complete${NC}"
}

# Apply BOLT optimization
apply_bolt_optimization() {
    echo -e "${CYAN}[BOLT] Applying binary optimization...${NC}"
    
    # Check if profile exists
    if [ ! -f "$BOLT_PROFILE" ]; then
        echo -e "${RED}[ERROR] No BOLT profile found at: $BOLT_PROFILE${NC}"
        echo -e "${YELLOW}Run '$0 profile' first to collect profile data${NC}"
        exit 1
    fi
    
    # Show profile info
    PROFILE_SIZE=$(du -h "$BOLT_PROFILE" | cut -f1)
    echo -e "${BLUE}[INFO] Using profile: $BOLT_PROFILE (${PROFILE_SIZE})${NC}"
    
    # Apply BOLT optimization
    echo -e "${BLUE}[OPTIMIZE] Applying BOLT optimization...${NC}"
    
    $LLVM_BOLT "$BINARY_PATH" \
        -o "$BOLT_OUTPUT" \
        -data="$BOLT_PROFILE" \
        -reorder-blocks=ext-tsp \
        -reorder-functions=hfsort+ \
        -split-functions \
        -split-all-cold \
        -split-eh \
        -dyno-stats \
        -icf=1 \
        -use-gnu-stack
    
    if [ -f "$BOLT_OUTPUT" ]; then
        echo -e "${GREEN}[OK] BOLT optimization complete!${NC}"
        
        # Compare sizes
        ORIG_SIZE=$(du -h "$BINARY_PATH" | cut -f1)
        BOLT_SIZE=$(du -h "$BOLT_OUTPUT" | cut -f1)
        
        echo -e "\n${CYAN}[RESULTS] Optimization Results:${NC}"
        echo -e "Original binary: $ORIG_SIZE"
        echo -e "Optimized binary: $BOLT_SIZE"
        echo -e "Output: $BOLT_OUTPUT"
        
        # Make executable
        chmod +x "$BOLT_OUTPUT"
        
        echo -e "\n${GREEN}✓ BOLT optimization successful!${NC}"
        echo -e "${YELLOW}Expected performance improvement: 5-20%${NC}"
        echo -e "${CYAN}Run benchmarks to verify:${NC}"
        echo -e "${YELLOW}  $BOLT_OUTPUT${NC}"
    else
        echo -e "${RED}[ERROR] BOLT optimization failed${NC}"
        exit 1
    fi
}

# Clean BOLT data
clean_bolt_data() {
    echo -e "${CYAN}[CLEAN] Removing BOLT data...${NC}"
    rm -f "$BOLT_PROFILE" "$PERF_DATA" "${BINARY_PATH}.instrumented" "$BOLT_OUTPUT"
    echo -e "${GREEN}[OK] BOLT data cleaned${NC}"
}

# Show statistics
show_stats() {
    if [ ! -f "$BOLT_OUTPUT" ]; then
        echo -e "${RED}[ERROR] No BOLT-optimized binary found${NC}"
        exit 1
    fi
    
    echo -e "${CYAN}[STATS] BOLT Optimization Statistics:${NC}"
    
    # Run BOLT with -dyno-stats to show optimization stats
    $LLVM_BOLT "$BOLT_OUTPUT" -dyno-stats -o /dev/null 2>&1 | grep -E "BOLT-INFO|Functions|Blocks|Instructions"
}

# Show help
show_help() {
    echo "Memory Nexus - BOLT Binary Optimization Script"
    echo ""
    echo "Usage: $0 [command]"
    echo ""
    echo "Commands:"
    echo "  profile     Collect execution profile for BOLT"
    echo "  optimize    Apply BOLT optimization using profile"
    echo "  stats       Show optimization statistics"
    echo "  clean       Remove all BOLT data"
    echo "  help        Show this help message"
    echo ""
    echo "Typical workflow:"
    echo "  1. Build with PGO first (recommended)"
    echo "  2. $0 profile   # Collect BOLT profile"
    echo "  3. $0 optimize  # Apply BOLT optimization"
    echo ""
    echo "Expected improvement: 5-20% on top of PGO"
}

# Main
print_banner
check_requirements

case "${1:-help}" in
    profile)
        collect_bolt_profile
        ;;
    optimize)
        apply_bolt_optimization
        ;;
    stats)
        show_stats
        ;;
    clean)
        clean_bolt_data
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