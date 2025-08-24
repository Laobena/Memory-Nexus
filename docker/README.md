# Docker Configuration

This folder contains all Docker-related files for Memory Nexus.

## Files

### Dockerfiles
- **Dockerfile.dev** - Development build with debugging tools and hot reload
- **Dockerfile.prod** - Production build with optimizations (LTO, native CPU, minimal size)
- **Dockerfile.simple** - Minimal build for quick testing

### Docker Compose Files
- **docker-compose.dev.yml** - Development environment with all services
- **docker-compose.prod.yml** - Production environment with optimizations
- **docker-compose.simple.yml** - Minimal setup for testing (if needed)

## Usage

### Development (Default)
```bash
# From project root, just run:
docker-compose up -d

# This automatically uses docker/docker-compose.dev.yml
```

### Production
```bash
# From project root:
docker-compose -f docker/docker-compose.prod.yml up -d
```

### Build Only
```bash
# Development build
docker build -f docker/Dockerfile.dev -t memory-nexus:dev .

# Production build
docker build -f docker/Dockerfile.prod -t memory-nexus:prod .
```

## Services Included

All compose files include:
- **memory-nexus** - Main application
- **surrealdb** - Graph database for UUID tracking
- **qdrant** - Vector database for embeddings
- **redis** - Cache (optional)
- **ollama** - LLM service for embeddings

## Environment Variables

Create a `.env` file in the project root:
```env
SURREALDB_URL=ws://surrealdb:8000
SURREALDB_USER=root
SURREALDB_PASS=root
QDRANT_URL=http://qdrant:6334
REDIS_URL=redis://redis:6379
```

## Volumes

- `memory-nexus-data` - Application data
- `surrealdb-data` - SurrealDB persistence
- `qdrant-data` - Qdrant vector storage
- `redis-data` - Redis cache persistence

## Networks

All services use `memory-nexus-network` for internal communication.