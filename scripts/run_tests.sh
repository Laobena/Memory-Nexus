#!/bin/bash
# Master test runner - runs tests from organized directories

set -e

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${BLUE}╔════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║            MEMORY NEXUS - MASTER TEST RUNNER          ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════╝${NC}"
echo

# Function to run a test
run_test() {
    local test_name=$1
    local test_path=$2
    
    echo -e "\n${BLUE}Running: $test_name${NC}"
    echo -e "${YELLOW}────────────────────────────────────────${NC}"
    
    if [ -f "$test_path" ]; then
        if [ -x "$test_path" ]; then
            "$test_path"
            echo -e "${GREEN}✓ $test_name completed${NC}"
        else
            echo -e "${YELLOW}! Making $test_name executable${NC}"
            chmod +x "$test_path"
            "$test_path"
            echo -e "${GREEN}✓ $test_name completed${NC}"
        fi
    else
        echo -e "${YELLOW}⚠ $test_name not found at $test_path${NC}"
    fi
}

# Menu
echo "Select test to run:"
echo "1) Quick Performance Validation"
echo "2) Skeleton Structure Test"
echo "3) Full Integration Test"
echo "4) Pipeline Test"
echo "5) All Tests (Sequential)"
echo "6) Quick Development Test"
echo "q) Quit"
echo
read -p "Choice: " choice

case $choice in
    1)
        run_test "Performance Validation" "./tests/validate_performance.sh"
        ;;
    2)
        run_test "Skeleton Test" "./tests/test_skeleton.sh"
        ;;
    3)
        run_test "Integration Test" "./tests/test_integration.sh"
        ;;
    4)
        run_test "Pipeline Test" "./tests/test_pipeline.sh"
        ;;
    5)
        echo -e "${BLUE}Running all tests sequentially...${NC}"
        run_test "Performance Validation" "./tests/validate_performance.sh"
        run_test "Skeleton Test" "./tests/test_skeleton.sh"
        run_test "Pipeline Test" "./tests/test_pipeline.sh"
        echo -e "\n${YELLOW}Note: Skipping full integration test (requires Docker)${NC}"
        echo -e "${YELLOW}Run separately with: ./tests/test_integration.sh${NC}"
        ;;
    6)
        run_test "Quick Test" "./scripts/quick_test.sh"
        ;;
    q|Q)
        echo "Exiting..."
        exit 0
        ;;
    *)
        echo -e "${YELLOW}Invalid choice${NC}"
        exit 1
        ;;
esac

echo -e "\n${GREEN}Test execution complete!${NC}"