//! Database management module

// Export the modern connection pool implementation
pub mod enhanced_pool;
pub mod database_connections;
pub mod qdrant_setup;

// Export enhanced pool as the standard connection pool
pub use enhanced_pool::{
    EnhancedConnectionPool as ConnectionPool,
    PoolConfig,
    PoolStats,
    CircuitBreaker,
    HealthMonitor,
    HealthStatus,
    RetryPolicy,
    PooledConnection,
    PoolStatsSnapshot,
    PoolError,
    // Add aliases for compatibility
    PooledConnection as PoolableConnection,
};

// Export unified database pool for the pipeline
pub use database_connections::{
    UnifiedDatabasePool, DatabaseConfig, SurrealDBConfig, QdrantConfig, RedisConfig,
    SurrealDBConnection, QdrantConnection, RedisPooledConnection,
    OverallHealth, DatabaseHealth, UnifiedPoolStats, DatabasePoolError
};

// Export Qdrant setup utilities
pub use qdrant_setup::{
    setup_qdrant_collections, create_temporal_point, temporal_vector_search,
    MEMORY_VECTORS_COLLECTION, TRUTH_VECTORS_COLLECTION, EVOLUTION_VECTORS_COLLECTION,
};