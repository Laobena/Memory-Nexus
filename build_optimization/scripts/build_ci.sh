#!/bin/bash
# Memory Nexus CI Profile Build Script
# Optimized for CI/CD pipelines with balanced performance and build time

set -e

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"

echo -e "${BLUE}🚀 Memory Nexus Bare CI Build${NC}"
echo -e "${BLUE}Profile: Balanced for CI/CD pipelines${NC}"
echo -e "${YELLOW}→ opt-level=2, thin LTO, some parallelization${NC}"
echo

cd "$PROJECT_DIR"

# Record start time
start_time=$(date +%s)

# Build with ci profile
echo -e "${YELLOW}Building with ci profile...${NC}"
cargo build --profile=ci --workspace

end_time=$(date +%s)
build_time=$((end_time - start_time))

echo
echo -e "${GREEN}✅ CI build completed in ${build_time}s${NC}"

# Show binary info if it exists
binary_path="target/ci/memory-nexus-bare"
if [[ -f "$binary_path" ]]; then
    binary_size=$(ls -lh "$binary_path" | awk '{print $5}')
    echo -e "${YELLOW}Binary size: ${binary_size}${NC}"
fi

echo -e "${BLUE}💡 Use this profile for:${NC}"
echo -e "  • Continuous integration builds"
echo -e "  • Balanced performance and compile time"
echo -e "  • CI/CD pipeline optimization"

# Optionally run tests if --test flag is provided
if [[ "$1" == "--test" ]]; then
    echo
    echo -e "${BLUE}🧪 Running tests with ci profile...${NC}"
    cargo test --profile=ci --workspace
fi