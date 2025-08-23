//! Storage and Persistence Blocks (Wrapped)
//! 
//! High-performance storage systems:
//! - Cache: 3-tier lock-free caching (<1ms L1 hits) - WRAPPED
//! - Vector: Vector store placeholder 
//! - WAL: Write-ahead logging placeholder
//! - Transaction: Transaction manager placeholder

pub mod cache_block;
pub mod vector_block;
pub mod wal_block;
pub mod transaction_block;

// Re-export wrapped storage blocks
pub use cache_block::{
    TieredCache,
    TieredCacheConfig,
};

pub use vector_block::{
    VectorStore,
};

pub use wal_block::{
    WriteAheadLog,
};

pub use transaction_block::{
    TransactionManager,
};

// Re-export types from the wrapped implementation
pub use memory_nexus::core::lock_free_cache::{
    CacheStats, CacheEntry,
};