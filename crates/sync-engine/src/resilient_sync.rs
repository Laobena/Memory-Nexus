//! Resilient synchronization strategy for dual-database architecture

use crate::error::{SyncError, SyncResult};
use database_adapters::{MemoryEntry, QdrantAdapter, SurrealDBAdapter, HealthCheck};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::Duration;
use tracing::{debug, error, info, warn};

/// Configuration for sync strategy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SyncConfig {
    pub retry_attempts: usize,
    pub retry_delay: Duration,
    pub sync_timeout: Duration,
    pub health_check_interval: Duration,
    pub conflict_resolution: ConflictResolution,
}

impl Default for SyncConfig {
    fn default() -> Self {
        Self {
            retry_attempts: 3,
            retry_delay: Duration::from_millis(100),
            sync_timeout: Duration::from_secs(30),
            health_check_interval: Duration::from_secs(60),
            conflict_resolution: ConflictResolution::LastWriteWins,
        }
    }
}

/// Conflict resolution strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConflictResolution {
    LastWriteWins,
    FirstWriteWins,
    Manual,
}

/// Health status of sync engine
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SyncHealth {
    pub overall_health: String,
    pub surrealdb_status: String,
    pub qdrant_status: String,
    pub last_sync: Option<chrono::DateTime<chrono::Utc>>,
    pub sync_operations_count: u64,
    pub failed_operations_count: u64,
}

/// Resilient synchronization strategy
pub struct ResilientSyncStrategy {
    surrealdb: Arc<SurrealDBAdapter>,
    qdrant: Arc<QdrantAdapter>,
    config: SyncConfig,
    sync_stats: Arc<tokio::sync::RwLock<SyncStats>>,
}

#[derive(Debug, Default)]
struct SyncStats {
    operations_count: u64,
    failed_operations: u64,
    last_sync: Option<chrono::DateTime<chrono::Utc>>,
}

impl ResilientSyncStrategy {
    /// Create new resilient sync strategy
    pub async fn new(
        surrealdb: Arc<SurrealDBAdapter>,
        qdrant: Arc<QdrantAdapter>,
        config: Option<SyncConfig>,
    ) -> SyncResult<Self> {
        let config = config.unwrap_or_default();

        let strategy = Self {
            surrealdb,
            qdrant,
            config,
            sync_stats: Arc::new(tokio::sync::RwLock::new(SyncStats::default())),
        };

        info!("✅ Resilient sync strategy initialized");
        Ok(strategy)
    }

    /// Store a memory entry to both databases
    pub async fn store_memory(&self, content: String, user_id: String) -> SyncResult<String> {
        let memory = MemoryEntry::new(content, user_id);
        let memory_id = memory.id.to_string();

        debug!("Storing memory: {}", memory_id);

        // Store in SurrealDB first (source of truth)
        self.surrealdb
            .store_memory(&memory)
            .await
            .map_err(|e| SyncError::database(format!("SurrealDB store failed: {}", e)))?;

        // Store in Qdrant for vector operations
        self.qdrant
            .store_memory(&memory)
            .await
            .map_err(|e| SyncError::database(format!("Qdrant store failed: {}", e)))?;

        // Update stats
        let mut stats = self.sync_stats.write().await;
        stats.operations_count += 1;
        stats.last_sync = Some(chrono::Utc::now());

        info!("✅ Memory stored successfully: {}", memory_id);
        Ok(memory_id)
    }

    /// Search memories across both databases
    pub async fn search_memories(&self, query: &str, limit: usize, user_id: &str) -> SyncResult<Vec<MemoryEntry>> {
        debug!("Searching memories: query='{}', user='{}', limit={}", query, user_id, limit);

        // For now, just use SurrealDB as primary search
        let results = self.surrealdb
            .search_memories(query, user_id, limit)
            .await
            .map_err(|e| SyncError::database(format!("Search failed: {}", e)))?;

        debug!("Found {} memories", results.len());
        Ok(results)
    }

    /// Get memory by ID from either database
    pub async fn get_memory(&self, memory_id: &str) -> SyncResult<Option<MemoryEntry>> {
        debug!("Getting memory: {}", memory_id);

        // Try SurrealDB first
        match self.surrealdb.get_memory(memory_id).await {
            Ok(Some(memory)) => {
                debug!("Found memory in SurrealDB");
                return Ok(Some(memory));
            }
            Ok(None) => {
                debug!("Memory not found in SurrealDB, trying Qdrant");
            }
            Err(e) => {
                warn!("SurrealDB error, trying Qdrant fallback: {}", e);
            }
        }

        // Fallback to Qdrant
        match self.qdrant.get_memory(memory_id).await {
            Ok(memory) => {
                debug!("Found memory in Qdrant");
                Ok(memory)
            }
            Err(e) => {
                error!("Memory not found in either database: {}", e);
                Ok(None)
            }
        }
    }

    /// List memories for a user
    pub async fn list_memories(&self, user_id: &str, limit: usize) -> SyncResult<Vec<MemoryEntry>> {
        debug!("Listing memories for user: {}, limit: {}", user_id, limit);

        // Use SurrealDB as primary for listing
        let memories = self.surrealdb
            .list_memories(user_id, limit)
            .await
            .map_err(|e| SyncError::database(format!("List memories failed: {}", e)))?;

        debug!("Found {} memories for user", memories.len());
        Ok(memories)
    }

    /// Get sync health status
    pub async fn get_sync_health(&self) -> SyncResult<SyncHealth> {
        let stats = self.sync_stats.read().await;

        // Check database health
        let surrealdb_healthy = self.surrealdb.liveness_check().await.unwrap_or(false);
        let qdrant_healthy = self.qdrant.liveness_check().await.unwrap_or(false);

        let overall_health = if surrealdb_healthy && qdrant_healthy {
            "healthy".to_string()
        } else if surrealdb_healthy || qdrant_healthy {
            "degraded".to_string()
        } else {
            "unhealthy".to_string()
        };

        Ok(SyncHealth {
            overall_health,
            surrealdb_status: if surrealdb_healthy { "healthy" } else { "unhealthy" }.to_string(),
            qdrant_status: if qdrant_healthy { "healthy" } else { "unhealthy" }.to_string(),
            last_sync: stats.last_sync,
            sync_operations_count: stats.operations_count,
            failed_operations_count: stats.failed_operations,
        })
    }

    /// Get database adapters
    pub fn surrealdb(&self) -> &SurrealDBAdapter {
        &self.surrealdb
    }

    pub fn qdrant(&self) -> &QdrantAdapter {
        &self.qdrant
    }
}