# Memory Nexus Bare - Essential Infrastructure

**Clean foundation ready for new pipeline implementation**

This is a stripped-down version of Memory Nexus containing only the essential infrastructure components. All search logic, Context Master, and pipeline implementations have been removed, providing a clean slate for implementing new processing architectures.

## 🏗️ What's Included

### ✅ Essential Infrastructure
- **Complete Database Layer**: SurrealDB + Qdrant adapters with connection pooling
- **Sync Engine**: Resilient dual-database coordination
- **HTTP Server**: Axum-based server with health endpoints
- **Docker Environment**: Complete containerization setup
- **Configuration Management**: Environment-based configuration
- **All Dependencies**: Complete Cargo.toml with all required crates

### ❌ What's Removed
- IntegratedSearchPipeline
- All Context Master modules (7 stages)
- Search scoring algorithms
- Pipeline processing logic
- Cache implementations
- AI processing components

## 🚀 Quick Start

### Option 1: Local Development
```bash
# Copy environment file
cp .env.example .env

# Run with cargo
cargo run

# Or with development profile
cargo run --profile dev-fast
```

### Option 2: Docker Development
```bash
# Start development environment
docker compose --profile development -f docker/docker-compose.yml up -d

# Check status
docker ps --filter "name=memory-nexus"
```

## 📡 API Endpoints

### Main API (Port 8080)
- `GET /` - Service information
- `POST /process` - Pipeline processing (placeholder)
- `GET /status` - System status

### Health API (Port 8082)  
- `GET /health` - Health check
- `GET /health/ready` - Readiness check
- `GET /health/live` - Liveness check

## 🏢 Architecture

```
memory-nexus-bare/
├── Cargo.toml              # Complete dependencies
├── .env.example            # Configuration template
├── docker/                 # Complete Docker environment
│   ├── docker-compose.yml  # Multi-profile setup
│   └── configs/            # Database configurations
├── src/
│   ├── main.rs            # Minimal HTTP server
│   ├── config.rs          # Configuration management
│   ├── db.rs              # Database coordination
│   ├── health.rs          # Health endpoints
│   ├── pipeline.rs        # Empty pipeline (ready for implementation)
│   └── types.rs           # Common types
└── crates/
    ├── database-adapters/  # SurrealDB + Qdrant connections
    └── sync-engine/       # Dual-database sync strategy
```

## 🔗 Database Connections

### SurrealDB (Source of Truth)
- **URL**: `ws://localhost:8000/rpc`
- **Purpose**: Graph relationships, persistent storage
- **Features**: RocksDB backend, graph traversal

### Qdrant (Vector Operations)
- **URL**: `http://localhost:6333`
- **Purpose**: Vector similarity search
- **Features**: HNSW indexing, 1024D embeddings ready

### Sync Strategy
- **Pattern**: Write-through with automatic conflict resolution
- **Resilience**: Automatic failover and retry logic
- **Health**: Continuous monitoring and recovery

## 🛠️ Ready for Implementation

The `src/pipeline.rs` module is empty and ready for your new pipeline implementation:

```rust
// This is where your new 27ms pipeline goes
pub async fn process(&self, request: PipelineRequest) -> Result<PipelineResponse, PipelineError> {
    // Implement your ultra-simplified architecture here
    // Target: 27ms processing time
    // Infrastructure: Already connected (SurrealDB + Qdrant + Sync)
}
```

## 📊 Performance Targets

- **Processing Time**: 27ms target (vs 80ms current)
- **Architecture**: `Intelligent Search → Response Formatter (2ms) → Answer`
- **Database Response**: <20ms (already achieved: 3.14ms Qdrant + 8-12ms SurrealDB)
- **Infrastructure Overhead**: <5ms

## 🐳 Docker Profiles

```bash
# Development (ports: 8081, 8001, 6335, 11435)
docker compose --profile development up -d

# Production (ports: 8080, 8000, 6333, 11434)
docker compose --profile production up -d

# Testing (ports: 8082, 8002, 6337, 11436)
docker compose --profile testing up -d
```

## 🧪 Testing Infrastructure

```bash
# Test database connections
cargo test

# Test with development profile
cargo test --profile dev-test

# Health check
curl http://localhost:8082/health
```

## 🔧 Environment Variables

```bash
# Application
APP_PORT=8080
HEALTH_PORT=8082

# Databases
SURREALDB_URL=ws://localhost:8000/rpc
SURREALDB_USER=root
SURREALDB_PASS=memory_nexus_2025
QDRANT_URL=http://localhost:6333

# AI
OLLAMA_URL=http://localhost:11434
OLLAMA_MODEL=mxbai-embed-large
```

## 🎯 Implementation Strategy

1. **Start with `src/pipeline.rs`** - Implement your processing logic
2. **Use `DatabaseManager`** - Access SurrealDB + Qdrant through sync strategy
3. **Target 27ms** - All infrastructure overhead is <5ms
4. **Leverage existing types** - `MemoryEntry`, `PipelineRequest`, `PipelineResponse`
5. **Test with Docker** - Complete environment ready

## 📝 Next Steps

1. Implement your pipeline logic in `src/pipeline.rs`
2. Use the existing database infrastructure via `DatabaseManager`
3. Test with the included Docker environment
4. Deploy using the production Docker profile

**The infrastructure is ready. Now build your 27ms pipeline! 🚀**