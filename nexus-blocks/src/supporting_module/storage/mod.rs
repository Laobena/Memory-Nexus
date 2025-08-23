//! Resilient storage block with tiered caching and write-ahead logging
//!
//! Provides adaptive storage with crash recovery and transaction support.

mod tiered_cache;
mod write_coalescer;
mod wal;
mod transaction;
mod recovery;
mod validation;

#[cfg(feature = "storage")]
mod vector_store;

#[cfg(feature = "storage")]
mod compression;

#[cfg(test)]
mod tests;

pub use tiered_cache::{TieredCache, CacheConfig, CacheTier};
pub use write_coalescer::{WriteCoalescer, WriteStrategy, BatchConfig};
pub use wal::{WriteAheadLog, WalEntry, WalConfig};
pub use transaction::{TransactionManager, Transaction, TransactionId, IsolationLevel};
pub use recovery::{RecoveryManager, RecoveryState, RecoveryStats};
pub use validation::{DataValidator, ChecksumType, ValidationResult};

#[cfg(feature = "storage")]
pub use vector_store::{VectorStore, QuantizationConfig};

#[cfg(feature = "storage")]
pub use compression::{CompressionStrategy, Compressor};

use crate::core::{BlockError, BlockResult};
use std::sync::Arc;
use uuid::Uuid;

/// Storage metrics
#[derive(Debug, Clone, Default)]
pub struct StorageMetrics {
    pub total_writes: u64,
    pub total_reads: u64,
    pub cache_hits: u64,
    pub cache_misses: u64,
    pub wal_entries: u64,
    pub transactions_committed: u64,
    pub transactions_rolled_back: u64,
    pub compression_ratio: f32,
    pub bytes_saved: u64,
}

/// Storage operation types
#[derive(Debug, Clone)]
pub enum Operation {
    Write(Uuid, Vec<u8>),
    Delete(Uuid),
    Update(Uuid, Vec<u8>),
    Batch(Vec<Operation>),
}

/// Storage configuration
#[derive(Debug, Clone)]
pub struct StorageConfig {
    /// Enable write-ahead logging
    pub enable_wal: bool,
    /// Enable transactions
    pub enable_transactions: bool,
    /// Cache configuration
    pub cache_config: CacheConfig,
    /// Write strategy
    pub write_strategy: WriteStrategy,
    /// Compression strategy
    pub compression: CompressionStrategy,
    /// Vector quantization
    pub enable_quantization: bool,
    /// Recovery on startup
    pub auto_recovery: bool,
}