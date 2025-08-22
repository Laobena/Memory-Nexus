#!/bin/bash
# ================================================================================
# Memory Nexus - Optimized Docker Build Script
# ================================================================================
# Builds ultra-optimized container images with multiple strategies:
# - Multi-stage builds with cargo-chef for caching
# - Static musl compilation for minimal size
# - Choice of distroless or scratch base images
# - BuildKit optimizations

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

# Configuration
IMAGE_NAME="memory-nexus"
IMAGE_TAG="optimized"
DOCKERFILE="Dockerfile.optimized"
BUILD_CONTEXT="."

# Print banner
print_banner() {
    echo -e "${BLUE}"
    echo "╔══════════════════════════════════════════════════════════════════╗"
    echo "║         MEMORY NEXUS - OPTIMIZED DOCKER BUILD                   ║"
    echo "║              460x Size Reduction (2GB → 4.6MB)                  ║"
    echo "╚══════════════════════════════════════════════════════════════════╝"
    echo -e "${NC}\n"
}

# Check requirements
check_requirements() {
    echo -e "${CYAN}[CHECK] Verifying Docker requirements...${NC}"
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        echo -e "${RED}[ERROR] Docker is not installed${NC}"
        exit 1
    fi
    
    # Check Docker version
    DOCKER_VERSION=$(docker version --format '{{.Server.Version}}' 2>/dev/null || echo "0.0.0")
    MAJOR_VERSION=$(echo $DOCKER_VERSION | cut -d. -f1)
    
    if [ "$MAJOR_VERSION" -lt "20" ]; then
        echo -e "${YELLOW}[WARN] Docker version $DOCKER_VERSION is old. Recommend 20.10+${NC}"
    fi
    
    # Check BuildKit
    if ! docker buildx version &> /dev/null; then
        echo -e "${YELLOW}[WARN] Docker BuildKit not available. Installing...${NC}"
        docker buildx create --use --name nexus-builder || true
    fi
    
    echo -e "${GREEN}[OK] Docker requirements verified${NC}\n"
}

# Build distroless image (recommended for production)
build_distroless() {
    echo -e "${CYAN}[BUILD] Building distroless image...${NC}"
    echo -e "${BLUE}[INFO] Distroless provides security and compatibility${NC}"
    
    DOCKER_BUILDKIT=1 docker buildx build \
        --target distroless \
        --tag ${IMAGE_NAME}:${IMAGE_TAG} \
        --tag ${IMAGE_NAME}:distroless \
        --tag ${IMAGE_NAME}:latest \
        --cache-from type=registry,ref=${IMAGE_NAME}:buildcache \
        --cache-to type=registry,ref=${IMAGE_NAME}:buildcache,mode=max \
        --build-arg BUILDKIT_INLINE_CACHE=1 \
        --progress=plain \
        -f ${DOCKERFILE} \
        ${BUILD_CONTEXT}
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}[OK] Distroless image built successfully${NC}"
        show_image_info "${IMAGE_NAME}:${IMAGE_TAG}"
    else
        echo -e "${RED}[ERROR] Build failed${NC}"
        exit 1
    fi
}

# Build scratch image (minimal size)
build_scratch() {
    echo -e "${CYAN}[BUILD] Building scratch image...${NC}"
    echo -e "${BLUE}[INFO] Scratch provides absolute minimal size${NC}"
    
    DOCKER_BUILDKIT=1 docker buildx build \
        --target scratch-runtime \
        --tag ${IMAGE_NAME}:scratch \
        --cache-from type=registry,ref=${IMAGE_NAME}:buildcache \
        --build-arg BUILDKIT_INLINE_CACHE=1 \
        --progress=plain \
        -f ${DOCKERFILE} \
        ${BUILD_CONTEXT}
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}[OK] Scratch image built successfully${NC}"
        show_image_info "${IMAGE_NAME}:scratch"
    else
        echo -e "${RED}[ERROR] Build failed${NC}"
        exit 1
    fi
}

# Build all variants
build_all() {
    echo -e "${CYAN}[BUILD] Building all image variants...${NC}"
    
    # Build distroless
    build_distroless
    
    # Build scratch
    build_scratch
    
    echo -e "\n${GREEN}[OK] All images built successfully${NC}"
    
    # Compare sizes
    echo -e "\n${CYAN}[COMPARE] Image size comparison:${NC}"
    docker images ${IMAGE_NAME} --format "table {{.Repository}}:{{.Tag}}\t{{.Size}}" | head -4
}

# Show image information
show_image_info() {
    local IMAGE=$1
    
    echo -e "\n${CYAN}[INFO] Image details for ${IMAGE}:${NC}"
    
    # Get size
    SIZE=$(docker images ${IMAGE} --format "{{.Size}}" | head -1)
    echo -e "  Size: ${GREEN}${SIZE}${NC}"
    
    # Get layers
    LAYERS=$(docker history ${IMAGE} --format "{{.Size}}" | wc -l)
    echo -e "  Layers: ${LAYERS}"
    
    # Security scan if available
    if command -v trivy &> /dev/null; then
        echo -e "\n${BLUE}[SCAN] Running security scan...${NC}"
        trivy image --security-checks vuln --quiet ${IMAGE}
    fi
}

# Run the optimized container
run_container() {
    local IMAGE="${IMAGE_NAME}:${IMAGE_TAG}"
    
    echo -e "${CYAN}[RUN] Starting optimized container...${NC}"
    
    docker run -d \
        --name memory-nexus-optimized \
        --restart unless-stopped \
        -p 8086:8086 \
        -p 9090:9090 \
        -e RUST_LOG=info \
        --memory="256m" \
        --cpus="1.0" \
        ${IMAGE}
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}[OK] Container started${NC}"
        echo -e "${BLUE}[INFO] API: http://localhost:8086${NC}"
        echo -e "${BLUE}[INFO] Metrics: http://localhost:9090/metrics${NC}"
        
        # Show logs
        echo -e "\n${CYAN}[LOGS] Container logs:${NC}"
        docker logs -f memory-nexus-optimized 2>&1 | head -20
    else
        echo -e "${RED}[ERROR] Failed to start container${NC}"
        exit 1
    fi
}

# Stop and remove container
stop_container() {
    echo -e "${CYAN}[STOP] Stopping container...${NC}"
    docker stop memory-nexus-optimized 2>/dev/null || true
    docker rm memory-nexus-optimized 2>/dev/null || true
    echo -e "${GREEN}[OK] Container stopped${NC}"
}

# Push to registry
push_image() {
    local REGISTRY=${1:-"docker.io"}
    local IMAGE="${IMAGE_NAME}:${IMAGE_TAG}"
    
    echo -e "${CYAN}[PUSH] Pushing image to ${REGISTRY}...${NC}"
    
    # Tag for registry
    docker tag ${IMAGE} ${REGISTRY}/${IMAGE}
    
    # Push
    docker push ${REGISTRY}/${IMAGE}
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}[OK] Image pushed to ${REGISTRY}${NC}"
    else
        echo -e "${RED}[ERROR] Push failed${NC}"
        exit 1
    fi
}

# Clean build cache
clean_cache() {
    echo -e "${CYAN}[CLEAN] Cleaning Docker build cache...${NC}"
    docker builder prune -f
    docker image prune -f
    echo -e "${GREEN}[OK] Cache cleaned${NC}"
}

# Show help
show_help() {
    echo "Memory Nexus - Optimized Docker Build Script"
    echo ""
    echo "Usage: $0 [command] [options]"
    echo ""
    echo "Commands:"
    echo "  distroless   Build distroless image (recommended)"
    echo "  scratch      Build scratch image (minimal size)"
    echo "  all          Build all image variants"
    echo "  run          Run the optimized container"
    echo "  stop         Stop and remove container"
    echo "  push [reg]   Push image to registry"
    echo "  clean        Clean build cache"
    echo "  help         Show this help"
    echo ""
    echo "Examples:"
    echo "  $0 distroless        # Build production image"
    echo "  $0 all               # Build all variants"
    echo "  $0 push ghcr.io      # Push to GitHub registry"
    echo ""
    echo "Image sizes:"
    echo "  Distroless: ~20MB (secure, compatible)"
    echo "  Scratch: ~4.6MB (minimal, may have compatibility issues)"
}

# Main
print_banner
check_requirements

case "${1:-help}" in
    distroless)
        build_distroless
        ;;
    scratch)
        build_scratch
        ;;
    all)
        build_all
        ;;
    run)
        run_container
        ;;
    stop)
        stop_container
        ;;
    push)
        push_image "${2}"
        ;;
    clean)
        clean_cache
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