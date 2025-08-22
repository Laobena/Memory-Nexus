#!/bin/bash

# Memory Nexus Performance Monitoring Script
# Comprehensive performance analysis and monitoring

set -e

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration
RESULTS_DIR="performance_results"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
REPORT_FILE="$RESULTS_DIR/report_$TIMESTAMP.md"
METRICS_FILE="$RESULTS_DIR/metrics_$TIMESTAMP.json"

# Ensure results directory exists
mkdir -p $RESULTS_DIR

# Print header
print_header() {
    echo -e "${CYAN}═══════════════════════════════════════════════════════════════${NC}"
    echo -e "${CYAN}       Memory Nexus Performance Monitor v2.0${NC}"
    echo -e "${CYAN}═══════════════════════════════════════════════════════════════${NC}"
    echo
}

# Check prerequisites
check_prerequisites() {
    echo -e "${YELLOW}▶ Checking Prerequisites${NC}"
    
    # Check for required tools
    TOOLS=("cargo" "rustc" "perf" "hyperfine" "flamegraph")
    for tool in "${TOOLS[@]}"; do
        if command -v $tool &> /dev/null; then
            echo -e "  ${GREEN}✓${NC} $tool found"
        else
            echo -e "  ${RED}✗${NC} $tool not found"
            case $tool in
                "perf")
                    echo "    Install: sudo apt-get install linux-tools-common"
                    ;;
                "hyperfine")
                    echo "    Install: cargo install hyperfine"
                    ;;
                "flamegraph")
                    echo "    Install: cargo install flamegraph"
                    ;;
            esac
        fi
    done
    
    echo
}

# Build with different optimization levels
benchmark_builds() {
    echo -e "${YELLOW}▶ Benchmarking Build Times${NC}"
    
    # Clean build
    cargo clean
    
    # Dev build
    echo -n "  Dev build: "
    DEV_TIME=$( { time cargo build --profile dev-fast 2>&1 1>/dev/null; } 2>&1 | grep real | awk '{print $2}' )
    echo -e "${GREEN}$DEV_TIME${NC}"
    
    # Release build
    cargo clean
    echo -n "  Release build: "
    RELEASE_TIME=$( { time cargo build --release 2>&1 1>/dev/null; } 2>&1 | grep real | awk '{print $2}' )
    echo -e "${GREEN}$RELEASE_TIME${NC}"
    
    # Native CPU build
    cargo clean
    echo -n "  Native CPU build: "
    NATIVE_TIME=$( { time RUSTFLAGS="-C target-cpu=native" cargo build --release 2>&1 1>/dev/null; } 2>&1 | grep real | awk '{print $2}' )
    echo -e "${GREEN}$NATIVE_TIME${NC}"
    
    echo
}

# Run cargo benchmarks
run_benchmarks() {
    echo -e "${YELLOW}▶ Running Benchmarks${NC}"
    
    # Run all benchmarks and save results
    cargo bench --all-features 2>&1 | tee "$RESULTS_DIR/bench_$TIMESTAMP.txt"
    
    # Extract key metrics
    echo -e "\n${BLUE}Key Metrics:${NC}"
    grep -E "time:|ns/iter|throughput" "$RESULTS_DIR/bench_$TIMESTAMP.txt" | head -20
    
    echo
}

# CPU profiling with perf
cpu_profiling() {
    echo -e "${YELLOW}▶ CPU Profiling${NC}"
    
    if command -v perf &> /dev/null; then
        # Build with debug symbols
        cargo build --release --features full
        
        # Run perf record
        echo "  Recording CPU profile..."
        timeout 10s perf record -F 99 -g ./target/release/memory-nexus-pipeline 2>/dev/null || true
        
        # Generate report
        perf report --stdio > "$RESULTS_DIR/perf_report_$TIMESTAMP.txt" 2>/dev/null
        
        # Show hot functions
        echo -e "\n${BLUE}Top 10 Hot Functions:${NC}"
        perf report --stdio 2>/dev/null | grep -v '^#' | head -10
        
        # Clean up
        rm -f perf.data perf.data.old
    else
        echo -e "  ${YELLOW}Skipping: perf not available${NC}"
    fi
    
    echo
}

# Memory profiling
memory_profiling() {
    echo -e "${YELLOW}▶ Memory Profiling${NC}"
    
    # Get binary size
    if [ -f "target/release/memory-nexus-pipeline" ]; then
        SIZE=$(du -h target/release/memory-nexus-pipeline | cut -f1)
        echo "  Binary size: $SIZE"
        
        # Check memory allocations with valgrind if available
        if command -v valgrind &> /dev/null; then
            echo "  Running valgrind memory check..."
            timeout 5s valgrind --tool=massif --massif-out-file="$RESULTS_DIR/massif_$TIMESTAMP.out" \
                ./target/release/memory-nexus-pipeline 2>/dev/null || true
            
            if [ -f "$RESULTS_DIR/massif_$TIMESTAMP.out" ]; then
                ms_print "$RESULTS_DIR/massif_$TIMESTAMP.out" > "$RESULTS_DIR/massif_$TIMESTAMP.txt" 2>/dev/null || true
                echo "  Memory profile saved to massif_$TIMESTAMP.txt"
            fi
        else
            echo -e "  ${YELLOW}Skipping valgrind: not available${NC}"
        fi
    fi
    
    echo
}

# Latency testing with hyperfine
latency_testing() {
    echo -e "${YELLOW}▶ Latency Testing${NC}"
    
    if command -v hyperfine &> /dev/null; then
        # Test startup time
        echo "  Testing startup latency..."
        hyperfine --warmup 3 --min-runs 10 \
            --export-json "$RESULTS_DIR/startup_$TIMESTAMP.json" \
            './target/release/memory-nexus-pipeline --version' 2>/dev/null || true
        
        # Show results
        if [ -f "$RESULTS_DIR/startup_$TIMESTAMP.json" ]; then
            MEAN=$(jq '.results[0].mean' "$RESULTS_DIR/startup_$TIMESTAMP.json")
            STDDEV=$(jq '.results[0].stddev' "$RESULTS_DIR/startup_$TIMESTAMP.json")
            echo -e "  Startup time: ${GREEN}${MEAN}s ± ${STDDEV}s${NC}"
        fi
    else
        echo -e "  ${YELLOW}Skipping hyperfine: not available${NC}"
    fi
    
    echo
}

# Generate flamegraph
generate_flamegraph() {
    echo -e "${YELLOW}▶ Generating Flamegraph${NC}"
    
    if command -v flamegraph &> /dev/null; then
        # Build with debug symbols
        cargo build --release --features full
        
        # Generate flamegraph
        echo "  Generating flamegraph..."
        timeout 10s cargo flamegraph --release --features full \
            -o "$RESULTS_DIR/flamegraph_$TIMESTAMP.svg" 2>/dev/null || true
        
        if [ -f "$RESULTS_DIR/flamegraph_$TIMESTAMP.svg" ]; then
            echo -e "  ${GREEN}✓${NC} Flamegraph saved to flamegraph_$TIMESTAMP.svg"
        fi
    else
        echo -e "  ${YELLOW}Skipping flamegraph: not available${NC}"
    fi
    
    echo
}

# SIMD optimization analysis
simd_analysis() {
    echo -e "${YELLOW}▶ SIMD Optimization Analysis${NC}"
    
    # Check CPU features
    ./target/release/check_features 2>/dev/null || cargo run --release --bin check_features
    
    # Check if SIMD is being used in assembly
    echo -e "\n${BLUE}SIMD Instructions in Binary:${NC}"
    if command -v objdump &> /dev/null; then
        echo -n "  AVX2 instructions: "
        objdump -d target/release/memory-nexus-pipeline 2>/dev/null | grep -c "vp" || echo "0"
        
        echo -n "  SSE instructions: "
        objdump -d target/release/memory-nexus-pipeline 2>/dev/null | grep -c "xmm" || echo "0"
        
        echo -n "  AVX512 instructions: "
        objdump -d target/release/memory-nexus-pipeline 2>/dev/null | grep -c "zmm" || echo "0"
    fi
    
    echo
}

# Generate performance report
generate_report() {
    echo -e "${YELLOW}▶ Generating Performance Report${NC}"
    
    cat > "$REPORT_FILE" << EOF
# Memory Nexus Performance Report
**Date:** $(date)
**Timestamp:** $TIMESTAMP

## System Information
- **CPU:** $(lscpu | grep "Model name" | cut -d: -f2 | xargs)
- **Cores:** $(nproc)
- **Memory:** $(free -h | grep Mem | awk '{print $2}')
- **Rust Version:** $(rustc --version)
- **Target:** $(rustc --print target-list | grep native || echo "generic")

## Build Performance
- Dev build time: $DEV_TIME
- Release build time: $RELEASE_TIME
- Native CPU build time: $NATIVE_TIME

## Optimization Status
\`\`\`
$(./target/release/check_features 2>/dev/null | tail -5 || echo "Feature detection not available")
\`\`\`

## Benchmark Results
Key performance metrics from cargo bench:
\`\`\`
$(grep -E "time:|ns/iter|throughput" "$RESULTS_DIR/bench_$TIMESTAMP.txt" 2>/dev/null | head -10 || echo "No benchmark results")
\`\`\`

## Recommendations
1. **CPU Optimizations:** Enable native CPU targeting with RUSTFLAGS="-C target-cpu=native"
2. **Profile-Guided Optimization:** Consider using PGO for 10-20% additional performance
3. **BOLT Optimization:** Post-link optimization can provide 5-15% improvement
4. **Memory Allocator:** Test with different allocators (mimalloc vs jemalloc)

## Files Generated
- Benchmark results: bench_$TIMESTAMP.txt
- Performance metrics: metrics_$TIMESTAMP.json
- Flamegraph: flamegraph_$TIMESTAMP.svg
- CPU profile: perf_report_$TIMESTAMP.txt
EOF
    
    echo -e "  ${GREEN}✓${NC} Report saved to $REPORT_FILE"
    echo
}

# Collect metrics in JSON format
collect_metrics() {
    echo -e "${YELLOW}▶ Collecting Metrics${NC}"
    
    # Get binary size
    BINARY_SIZE=$(stat -c%s target/release/memory-nexus-pipeline 2>/dev/null || echo 0)
    
    # Count SIMD instructions
    AVX2_COUNT=$(objdump -d target/release/memory-nexus-pipeline 2>/dev/null | grep -c "vp" || echo 0)
    SSE_COUNT=$(objdump -d target/release/memory-nexus-pipeline 2>/dev/null | grep -c "xmm" || echo 0)
    
    # Create JSON metrics
    cat > "$METRICS_FILE" << EOF
{
  "timestamp": "$TIMESTAMP",
  "system": {
    "cpu_cores": $(nproc),
    "rust_version": "$(rustc --version | cut -d' ' -f2)",
    "target": "$(rustc -vV | grep host | cut -d' ' -f2)"
  },
  "build": {
    "dev_time": "$DEV_TIME",
    "release_time": "$RELEASE_TIME",
    "native_time": "$NATIVE_TIME",
    "binary_size": $BINARY_SIZE
  },
  "optimizations": {
    "avx2_instructions": $AVX2_COUNT,
    "sse_instructions": $SSE_COUNT,
    "lto_enabled": true,
    "codegen_units": 1
  }
}
EOF
    
    echo -e "  ${GREEN}✓${NC} Metrics saved to $METRICS_FILE"
    echo
}

# Main execution
main() {
    print_header
    check_prerequisites
    
    # Build the project first
    echo -e "${YELLOW}▶ Building Project${NC}"
    cargo build --release --features full
    echo
    
    # Run all analyses
    benchmark_builds
    run_benchmarks
    cpu_profiling
    memory_profiling
    latency_testing
    generate_flamegraph
    simd_analysis
    
    # Generate outputs
    collect_metrics
    generate_report
    
    # Summary
    echo -e "${CYAN}═══════════════════════════════════════════════════════════════${NC}"
    echo -e "${GREEN}✓ Performance monitoring complete!${NC}"
    echo -e "Results saved in: ${BLUE}$RESULTS_DIR/${NC}"
    echo -e "Report: ${BLUE}$REPORT_FILE${NC}"
    echo -e "Metrics: ${BLUE}$METRICS_FILE${NC}"
    echo -e "${CYAN}═══════════════════════════════════════════════════════════════${NC}"
}

# Run main function
main "$@"