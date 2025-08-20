#!/bin/bash
# Memory Nexus Docker Management Script
# Unified container management with consistent naming and organization

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
COMPOSE_FILE="$SCRIPT_DIR/docker-compose.yml"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Logo and branding
print_logo() {
    echo -e "${PURPLE}"
    cat << "EOF"
    ███╗   ███╗███████╗███╗   ███╗ ██████╗ ██████╗ ██╗   ██╗    ███╗   ██╗███████╗██╗  ██╗██╗   ██╗███╗   ███╗
    ████╗ ████║██╔════╝████╗ ████║██╔═══██╗██╔══██╗╚██╗ ██╔╝    ████╗  ██║██╔════╝╚██╗██╔╝██║   ██║████╗ ████║
    ██╔████╔██║█████╗  ██╔████╔██║██║   ██║██████╔╝ ╚████╔╝     ██╔██╗ ██║█████╗   ╚███╔╝ ██║   ██║██╔████╔██║
    ██║╚██╔╝██║██╔══╝  ██║╚██╔╝██║██║   ██║██╔══██╗  ╚██╔╝      ██║╚██╗██║██╔══╝   ██╔██╗ ██║   ██║██║╚██╔╝██║
    ██║ ╚═╝ ██║███████╗██║ ╚═╝ ██║╚██████╔╝██║  ██║   ██║       ██║ ╚████║███████╗██╔╝ ██╗╚██████╔╝██║ ╚═╝ ██║
    ╚═╝     ╚═╝╚══════╝╚═╝     ╚═╝ ╚═════╝ ╚═╝  ╚═╝   ╚═╝       ╚═╝  ╚═══╝╚══════╝╚═╝  ╚═╝ ╚═════╝ ╚═╝     ╚═╝
EOF
    echo -e "${NC}"
    echo -e "${CYAN}Universal Intelligence Amplification System${NC}"
    echo -e "${CYAN}Enterprise Docker Management (2025)${NC}"
    echo ""
}

# Help function
show_help() {
    print_logo
    echo -e "${YELLOW}Usage: $0 [COMMAND] [OPTIONS]${NC}"
    echo ""
    echo -e "${GREEN}ENVIRONMENTS:${NC}"
    echo -e "  ${BLUE}dev${NC}        Start development environment (ports 8081, 9001)"
    echo -e "  ${BLUE}prod${NC}       Start production environment (ports 8080, 9000)"
    echo -e "  ${BLUE}test${NC}       Start testing environment (ports 8082, 9002)"
    echo -e "  ${BLUE}monitor${NC}    Start monitoring stack (Prometheus, Grafana)"
    echo -e "  ${BLUE}all${NC}        Start everything (use with caution)"
    echo ""
    echo -e "${GREEN}MANAGEMENT:${NC}"
    echo -e "  ${BLUE}stop [env]${NC}    Stop environment (or all if no env specified)"
    echo -e "  ${BLUE}restart [env]${NC} Restart environment"
    echo -e "  ${BLUE}logs [env]${NC}    Show logs for environment"
    echo -e "  ${BLUE}status${NC}       Show status of all containers"
    echo -e "  ${BLUE}clean${NC}        Clean up stopped containers and unused volumes"
    echo -e "  ${BLUE}reset${NC}        Reset everything (DANGER: deletes all data)"
    echo ""
    echo -e "${GREEN}UTILITIES:${NC}"
    echo -e "  ${BLUE}build${NC}        Build all Docker images"
    echo -e "  ${BLUE}pull${NC}         Pull latest base images"
    echo -e "  ${BLUE}shell [env]${NC}  Open shell in Memory Nexus container"
    echo ""
    echo -e "${GREEN}EXAMPLES:${NC}"
    echo -e "  ${YELLOW}$0 dev${NC}                 # Start development environment"
    echo -e "  ${YELLOW}$0 prod${NC}                # Start production environment"
    echo -e "  ${YELLOW}$0 stop dev${NC}            # Stop development environment"
    echo -e "  ${YELLOW}$0 logs prod${NC}           # Show production logs"
    echo -e "  ${YELLOW}$0 shell dev${NC}           # Open shell in dev container"
    echo ""
}

# Check if docker and docker compose are available
check_requirements() {
    if ! command -v docker &> /dev/null; then
        echo -e "${RED}ERROR: Docker is not installed or not in PATH${NC}"
        exit 1
    fi
    
    if ! docker compose version &> /dev/null; then
        echo -e "${RED}ERROR: Docker Compose is not available${NC}"
        exit 1
    fi
}

# Start environment function
start_env() {
    local env=$1
    local valid_envs="development production testing monitoring all"
    
    if [[ ! " $valid_envs " =~ " $env " ]]; then
        echo -e "${RED}ERROR: Invalid environment '$env'${NC}"
        echo -e "${YELLOW}Valid environments: $valid_envs${NC}"
        exit 1
    fi
    
    print_logo
    echo -e "${GREEN}Starting Memory Nexus $env environment...${NC}"
    echo ""
    
    case $env in
        "development"|"dev")
            echo -e "${CYAN}Development Environment:${NC}"
            echo -e "  • Memory Nexus API: http://localhost:8081"
            echo -e "  • MCP Server: http://localhost:9001"
            echo -e "  • SurrealDB: http://localhost:8001"
            echo -e "  • Qdrant: http://localhost:6335"
            echo -e "  • Ollama: http://localhost:11435"
            echo ""
            docker compose -f "$COMPOSE_FILE" --profile development up -d
            ;;
        "production"|"prod")
            echo -e "${CYAN}Production Environment:${NC}"
            echo -e "  • Memory Nexus API: http://localhost:8080"
            echo -e "  • MCP Server: http://localhost:9000"
            echo -e "  • SurrealDB: http://localhost:8000"
            echo -e "  • Qdrant: http://localhost:6333"
            echo -e "  • Ollama: http://localhost:11434"
            echo ""
            docker compose -f "$COMPOSE_FILE" --profile production up -d
            ;;
        "testing"|"test")
            echo -e "${CYAN}Testing Environment:${NC}"
            echo -e "  • Test Orchestrator: memory-nexus-test-orchestrator"
            echo -e "  • SurrealDB: http://localhost:8002"
            echo -e "  • Qdrant: http://localhost:6337"
            echo -e "  • Ollama: http://localhost:11436"
            echo ""
            docker compose -f "$COMPOSE_FILE" --profile testing up -d
            ;;
        "monitoring"|"monitor")
            echo -e "${CYAN}Monitoring Stack:${NC}"
            echo -e "  • Prometheus: http://localhost:9090"
            echo -e "  • Grafana: http://localhost:3000 (admin/memory_nexus_admin_2025)"
            echo ""
            docker compose -f "$COMPOSE_FILE" --profile monitoring up -d
            ;;
        "all")
            echo -e "${YELLOW}WARNING: Starting ALL services (this will use significant resources)${NC}"
            echo ""
            docker compose -f "$COMPOSE_FILE" --profile all up -d
            ;;
    esac
    
    echo ""
    echo -e "${GREEN}Environment started successfully!${NC}"
    echo -e "${YELLOW}Use '$0 status' to check container health${NC}"
}

# Stop environment function
stop_env() {
    local env=$1
    
    if [[ -z "$env" ]]; then
        echo -e "${YELLOW}Stopping all Memory Nexus containers...${NC}"
        docker compose -f "$COMPOSE_FILE" down
    else
        case $env in
            "development"|"dev")
                docker compose -f "$COMPOSE_FILE" --profile development down
                ;;
            "production"|"prod")
                docker compose -f "$COMPOSE_FILE" --profile production down
                ;;
            "testing"|"test")
                docker compose -f "$COMPOSE_FILE" --profile testing down
                ;;
            "monitoring"|"monitor")
                docker compose -f "$COMPOSE_FILE" --profile monitoring down
                ;;
            *)
                echo -e "${RED}ERROR: Invalid environment '$env'${NC}"
                exit 1
                ;;
        esac
    fi
    
    echo -e "${GREEN}Environment stopped successfully!${NC}"
}

# Show logs function
show_logs() {
    local env=$1
    
    if [[ -z "$env" ]]; then
        docker compose -f "$COMPOSE_FILE" logs -f
    else
        case $env in
            "development"|"dev")
                docker compose -f "$COMPOSE_FILE" --profile development logs -f
                ;;
            "production"|"prod")
                docker compose -f "$COMPOSE_FILE" --profile production logs -f
                ;;
            "testing"|"test")
                docker compose -f "$COMPOSE_FILE" --profile testing logs -f
                ;;
            "monitoring"|"monitor")
                docker compose -f "$COMPOSE_FILE" --profile monitoring logs -f
                ;;
            *)
                echo -e "${RED}ERROR: Invalid environment '$env'${NC}"
                exit 1
                ;;
        esac
    fi
}

# Show status function
show_status() {
    print_logo
    echo -e "${GREEN}Memory Nexus Container Status:${NC}"
    echo ""
    
    # Show all Memory Nexus containers
    docker ps -a --filter "name=memory-nexus" --format "table {{.Names}}\t{{.Image}}\t{{.Status}}\t{{.Ports}}"
    
    echo ""
    echo -e "${GREEN}Memory Nexus Networks:${NC}"
    docker network ls --filter "name=memory-nexus"
    
    echo ""
    echo -e "${GREEN}Memory Nexus Volumes:${NC}"
    docker volume ls --filter "name=memory-nexus" --format "table {{.Name}}\t{{.Driver}}\t{{.Scope}}"
}

# Open shell function
open_shell() {
    local env=$1
    local container_name
    
    case $env in
        "development"|"dev")
            container_name="memory-nexus-dev"
            ;;
        "production"|"prod")
            container_name="memory-nexus-prod"
            ;;
        *)
            echo -e "${RED}ERROR: Invalid environment '$env' for shell access${NC}"
            echo -e "${YELLOW}Valid environments: dev, prod${NC}"
            exit 1
            ;;
    esac
    
    if ! docker ps --format "{{.Names}}" | grep -q "^${container_name}$"; then
        echo -e "${RED}ERROR: Container $container_name is not running${NC}"
        echo -e "${YELLOW}Start the environment first: $0 $env${NC}"
        exit 1
    fi
    
    echo -e "${GREEN}Opening shell in $container_name...${NC}"
    docker exec -it "$container_name" /bin/bash
}

# Main script logic
main() {
    check_requirements
    
    case "${1:-}" in
        "help"|"-h"|"--help")
            show_help
            ;;
        "dev"|"development")
            start_env "development"
            ;;
        "prod"|"production")
            start_env "production"
            ;;
        "test"|"testing")
            start_env "testing"
            ;;
        "monitor"|"monitoring")
            start_env "monitoring"
            ;;
        "all")
            start_env "all"
            ;;
        "stop")
            stop_env "${2:-}"
            ;;
        "restart")
            stop_env "${2:-}"
            sleep 2
            start_env "${2:-development}"
            ;;
        "logs")
            show_logs "${2:-}"
            ;;
        "status")
            show_status
            ;;
        "shell")
            open_shell "${2:-dev}"
            ;;
        "build")
            echo -e "${GREEN}Building Memory Nexus Docker images...${NC}"
            docker compose -f "$COMPOSE_FILE" build
            ;;
        "pull")
            echo -e "${GREEN}Pulling latest base images...${NC}"
            docker compose -f "$COMPOSE_FILE" pull
            ;;
        "clean")
            echo -e "${YELLOW}Cleaning up stopped containers and unused volumes...${NC}"
            docker system prune -f
            docker volume prune -f
            ;;
        "reset")
            echo -e "${RED}WARNING: This will delete ALL Memory Nexus data!${NC}"
            read -p "Are you sure? (type 'yes' to confirm): " confirm
            if [[ $confirm == "yes" ]]; then
                docker compose -f "$COMPOSE_FILE" down -v
                docker system prune -a -f --volumes
                echo -e "${GREEN}Reset complete!${NC}"
            else
                echo -e "${YELLOW}Reset cancelled.${NC}"
            fi
            ;;
        "")
            show_help
            ;;
        *)
            echo -e "${RED}ERROR: Unknown command '$1'${NC}"
            echo -e "${YELLOW}Use '$0 help' to see available commands${NC}"
            exit 1
            ;;
    esac
}

# Run main function with all arguments
main "$@"