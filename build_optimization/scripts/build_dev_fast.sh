#!/bin/bash
# Memory Nexus dev-fast Profile Build Script
# Optimized for rapid iteration and development speed

set -e

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"

echo -e "${BLUE}🚀 Memory Nexus Bare dev-fast Build${NC}"
echo -e "${BLUE}Profile: Minimal optimization for rapid iteration${NC}"
echo -e "${YELLOW}→ opt-level=0, no debug symbols, max parallelization${NC}"
echo

cd "$PROJECT_DIR"

# Record start time
start_time=$(date +%s)

# Build with dev-fast profile
echo -e "${YELLOW}Building with dev-fast profile...${NC}"
cargo build --profile=dev-fast

end_time=$(date +%s)
build_time=$((end_time - start_time))

echo
echo -e "${GREEN}✅ dev-fast build completed in ${build_time}s${NC}"

# Show binary info if it exists
binary_path="target/dev-fast/memory-nexus-bare"
if [[ -f "$binary_path" ]]; then
    binary_size=$(ls -lh "$binary_path" | awk '{print $5}')
    echo -e "${YELLOW}Binary size: ${binary_size}${NC}"
fi

echo -e "${BLUE}💡 Use this profile for:${NC}"
echo -e "  • Quick code changes and testing"
echo -e "  • Rapid prototyping"
echo -e "  • Maximum compilation speed"