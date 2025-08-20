#!/bin/bash
# Memory Nexus Build Profile Selector
# Easy selection and execution of optimized build profiles

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"

print_header() {
    echo -e "${BLUE}🚀 Memory Nexus Bare Build Profile Selector${NC}"
    echo -e "${BLUE}======================================${NC}"
    echo
}

print_profiles() {
    echo -e "${CYAN}Available Build Profiles:${NC}"
    echo
    echo -e "${GREEN}1. dev-fast${NC}      - Fastest compilation for rapid iteration"
    echo -e "   ${YELLOW}→${NC} opt-level=0, no debug symbols, max parallelization"
    echo -e "   ${YELLOW}→${NC} Best for: Quick code changes and testing"
    echo
    echo -e "${GREEN}2. ci${NC}            - Optimized for CI/CD pipelines"
    echo -e "   ${YELLOW}→${NC} opt-level=2, thin LTO, some parallelization"
    echo -e "   ${YELLOW}→${NC} Best for: Continuous integration builds"
    echo
    echo -e "${GREEN}3. release${NC}       - Production builds with maximum performance"
    echo -e "   ${YELLOW}→${NC} opt-level=3, full LTO, single codegen unit"
    echo -e "   ${YELLOW}→${NC} Best for: Production deployment"
    echo
}

build_with_profile() {
    local profile="$1"
    local target="${2:-}"
    local additional_args="${3:-}"
    
    echo -e "${BLUE}🔨 Building Memory Nexus with profile: ${GREEN}$profile${NC}"
    
    cd "$PROJECT_DIR"
    
    # Construct cargo command
    local cargo_cmd="cargo build --profile=$profile"
    
    if [[ -n "$target" ]]; then
        cargo_cmd="$cargo_cmd $target"
    fi
    
    if [[ -n "$additional_args" ]]; then
        cargo_cmd="$cargo_cmd $additional_args"
    fi
    
    echo -e "${YELLOW}Command:${NC} $cargo_cmd"
    echo
    
    # Record start time
    local start_time=$(date +%s)
    
    # Execute build
    if eval $cargo_cmd; then
        local end_time=$(date +%s)
        local build_time=$((end_time - start_time))
        
        echo
        echo -e "${GREEN}✅ Build completed successfully!${NC}"
        echo -e "${CYAN}Build time: ${build_time}s${NC}"
        
        # Show binary info if it exists
        local binary_path="target/$profile/memory-nexus"
        if [[ -f "$binary_path" ]]; then
            local binary_size=$(ls -lh "$binary_path" | awk '{print $5}')
            echo -e "${CYAN}Binary size: ${binary_size}${NC}"
        fi
    else
        echo -e "${RED}❌ Build failed!${NC}"
        return 1
    fi
}

show_usage() {
    echo -e "${PURPLE}Usage:${NC}"
    echo -e "  $0 [profile] [target] [options]"
    echo
    echo -e "${PURPLE}Examples:${NC}"
    echo -e "  $0                           # Interactive mode"
    echo -e "  $0 dev-fast                  # Quick development build"
    echo -e "  $0 dev-test --lib            # Library build for testing"
    echo -e "  $0 release --release         # Production build"
    echo -e "  $0 release-small --release   # Size-optimized production build"
    echo -e "  $0 ci --workspace            # CI build for all workspace members"
    echo
}

# Main execution
main() {
    print_header
    
    # If profile provided as argument
    if [[ $# -gt 0 ]]; then
        local profile="$1"
        local target="${2:-}"
        local additional_args="${@:3}"
        
        # Validate profile
        case "$profile" in
            dev-fast|dev-test|ci|release|release-small)
                build_with_profile "$profile" "$target" "$additional_args"
                ;;
            -h|--help|help)
                print_profiles
                echo
                show_usage
                ;;
            *)
                echo -e "${RED}❌ Invalid profile: $profile${NC}"
                echo
                print_profiles
                exit 1
                ;;
        esac
        return
    fi
    
    # Interactive mode
    print_profiles
    
    while true; do
        echo -e "${PURPLE}Select build profile (1-5):${NC} "
        read -r choice
        
        case $choice in
            1)
                build_with_profile "dev-fast"
                break
                ;;
            2)
                build_with_profile "dev-test"
                break
                ;;
            3)
                build_with_profile "ci"
                break
                ;;
            4)
                build_with_profile "release"
                break
                ;;
            5)
                build_with_profile "release-small"
                break
                ;;
            q|quit|exit)
                echo -e "${YELLOW}Exiting...${NC}"
                exit 0
                ;;
            *)
                echo -e "${RED}Invalid choice. Please select 1-5 or 'q' to quit.${NC}"
                ;;
        esac
    done
}

# Run main function
main "$@"