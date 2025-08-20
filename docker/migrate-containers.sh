#!/bin/bash
# Memory Nexus Container Migration Script
# Cleans up scattered containers and networks, prepares for unified architecture

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}Memory Nexus Container Migration${NC}"
echo -e "${BLUE}=================================${NC}"
echo ""

# Stop and remove old scattered containers
echo -e "${YELLOW}Step 1: Stopping and removing old scattered containers...${NC}"

# List of old container patterns to clean up
OLD_CONTAINERS=(
    "practical_taussig"
    "memory-nexus-qdrant-test"
    "memory-nexus-ollama-test"  
    "memory-nexus-surrealdb-test"
    "memory-nexus-dev-fast"
    "memory-nexus-qdrant"
    "memory-nexus-ollama"
    "mn-qdrant-lightning"
    "mn-surrealdb-lightning"
)

for container in "${OLD_CONTAINERS[@]}"; do
    if docker ps -a --format "{{.Names}}" | grep -q "^${container}$"; then
        echo -e "  ${RED}Removing container: $container${NC}"
        docker stop "$container" 2>/dev/null || true
        docker rm "$container" 2>/dev/null || true
    else
        echo -e "  ${GREEN}Container $container already removed${NC}"
    fi
done

# Remove old networks
echo -e "${YELLOW}Step 2: Removing old scattered networks...${NC}"

OLD_NETWORKS=(
    "compose_memory_nexus_dev_network"
    "memory-nexus-full-build-network"
    "memory-nexus-test-network"
)

for network in "${OLD_NETWORKS[@]}"; do
    if docker network ls --format "{{.Name}}" | grep -q "^${network}$"; then
        echo -e "  ${RED}Removing network: $network${NC}"
        docker network rm "$network" 2>/dev/null || true
    else
        echo -e "  ${GREEN}Network $network already removed${NC}"
    fi
done

# Clean up old volumes with inconsistent naming
echo -e "${YELLOW}Step 3: Listing old volumes for review...${NC}"
echo -e "${BLUE}Old volumes found (review before manual cleanup):${NC}"

docker volume ls --format "{{.Name}}" | grep -E "(compose_|memory-nexus-|mn-)" | while read volume; do
    echo -e "  ${YELLOW}$volume${NC}"
done

echo ""
echo -e "${GREEN}Migration completed!${NC}"
echo ""
echo -e "${YELLOW}What was cleaned up:${NC}"
echo -e "  ✅ Removed scattered containers with inconsistent naming"
echo -e "  ✅ Removed fragmented Docker networks"
echo -e "  📋 Listed old volumes for manual review"
echo ""
echo -e "${GREEN}What's ready:${NC}"
echo -e "  🚀 Unified docker-compose.yml with consistent naming"
echo -e "  🔧 Management script for easy operations"
echo -e "  📦 All containers now follow 'memory-nexus-[service]-[env]' pattern"
echo ""
echo -e "${BLUE}Next steps:${NC}"
echo -e "  1. Review old volumes above and remove manually if not needed"
echo -e "  2. Use: ${GREEN}./manage.sh dev${NC} to start development environment"
echo -e "  3. Use: ${GREEN}./manage.sh status${NC} to see the new organized containers"
echo ""

# Show the new architecture
echo -e "${YELLOW}New Unified Architecture:${NC}"
echo -e "${GREEN}Production:${NC} memory-nexus-[service]-prod"
echo -e "${GREEN}Development:${NC} memory-nexus-[service]-dev" 
echo -e "${GREEN}Testing:${NC} memory-nexus-[service]-test"
echo -e "${GREEN}Network:${NC} memory-nexus-network (single unified network)"
echo -e "${GREEN}Volumes:${NC} memory-nexus-[service]-[env]-data"
echo ""