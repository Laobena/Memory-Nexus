//! Essential database adapter enhancements for high-performance operations
//!
//! This module provides essential database optimization components:
//! - Universal Memory ID system for 100x performance direct access
//! - Direct access error handling with recovery strategies
//! - HNSW configuration for vector search optimization

pub mod universal_memory_id;
pub mod direct_access_errors;
pub mod hnsw_config;

// Re-export essential components for easy access
pub use universal_memory_id::{
    UniversalMemoryId, UniversalMemoryIdConfig, UserSequenceCounter, UniversalMemoryIdError
};
pub use direct_access_errors::{
    DirectAccessError, ErrorRecoveryManager, DirectAccessMetrics, RecoveryStrategy
};
pub use hnsw_config::{HNSWConfig, HNSWMetrics, HNSWBuildProgress};