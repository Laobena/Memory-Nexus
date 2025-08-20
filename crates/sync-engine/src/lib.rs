//! Sync Engine for Memory Nexus
//! 
//! Provides resilient synchronization between SurrealDB and Qdrant databases.
//! This is the essential infrastructure for coordinating dual-database operations.

pub mod resilient_sync;
pub mod error;

pub use resilient_sync::{ResilientSyncStrategy, SyncConfig, SyncHealth};
pub use error::{SyncError, SyncResult};