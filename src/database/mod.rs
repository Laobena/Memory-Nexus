//! Database management module

use crate::config::Config;
use database_adapters::{QdrantAdapter, SurrealDBAdapter, HealthCheck};
use std::sync::Arc;
use sync_engine::{ResilientSyncStrategy, SyncConfig, SyncHealth};
use tracing::{error, info};

/// Database manager coordinating all database connections
pub struct DatabaseManager {
    pub surrealdb: Arc<SurrealDBAdapter>,
    pub qdrant: Arc<QdrantAdapter>,
    pub sync_strategy: Arc<ResilientSyncStrategy>,
}

/// Combined health status of all databases
#[derive(Debug, Clone)]
pub struct DatabaseHealthStatus {
    pub surrealdb_healthy: bool,
    pub qdrant_healthy: bool,
    pub sync_healthy: bool,
    pub overall_healthy: bool,
}

impl DatabaseManager {
    /// Initialize all database connections
    pub async fn new(config: &Config) -> Result<Self, DatabaseError> {
        info!("🔗 Initializing database connections...");

        // Initialize SurrealDB
        let surrealdb = Arc::new(
            SurrealDBAdapter::new(&config.surrealdb_url, &config.surrealdb_user, &config.surrealdb_pass)
                .await
                .map_err(|e| DatabaseError::Connection(format!("SurrealDB init failed: {}", e)))?,
        );
        info!("✅ SurrealDB connected: {}", config.surrealdb_url);

        // Initialize Qdrant
        let qdrant = Arc::new(
            QdrantAdapter::new(&config.qdrant_url)
                .await
                .map_err(|e| DatabaseError::Connection(format!("Qdrant init failed: {}", e)))?,
        );
        info!("✅ Qdrant connected: {}", config.qdrant_url);

        // Initialize sync strategy
        let sync_config = SyncConfig::default();
        let sync_strategy = Arc::new(
            ResilientSyncStrategy::new(surrealdb.clone(), qdrant.clone(), Some(sync_config))
                .await
                .map_err(|e| DatabaseError::Sync(format!("Sync strategy init failed: {}", e)))?,
        );
        info!("✅ Resilient sync strategy initialized");

        Ok(Self {
            surrealdb,
            qdrant,
            sync_strategy,
        })
    }

    /// Get combined health status of all databases
    pub async fn get_health_status(&self) -> DatabaseHealthStatus {
        let surrealdb_healthy = self
            .surrealdb
            .liveness_check()
            .await
            .unwrap_or(false);

        let qdrant_healthy = self
            .qdrant
            .liveness_check()
            .await
            .unwrap_or(false);

        let sync_health = self
            .sync_strategy
            .get_sync_health()
            .await
            .unwrap_or_else(|_| SyncHealth {
                overall_health: "unhealthy".to_string(),
                surrealdb_status: "unknown".to_string(),
                qdrant_status: "unknown".to_string(),
                last_sync: None,
                sync_operations_count: 0,
                failed_operations_count: 0,
            });

        let sync_healthy = sync_health.overall_health == "healthy";
        let overall_healthy = surrealdb_healthy && qdrant_healthy && sync_healthy;

        DatabaseHealthStatus {
            surrealdb_healthy,
            qdrant_healthy,
            sync_healthy,
            overall_healthy,
        }
    }

    /// Test all database connections
    pub async fn test_connections(&self) -> Result<(), DatabaseError> {
        // Test SurrealDB
        self.surrealdb
            .liveness_check()
            .await
            .map_err(|e| DatabaseError::Connection(format!("SurrealDB connection test failed: {}", e)))?;

        // Test Qdrant
        self.qdrant
            .liveness_check()
            .await
            .map_err(|e| DatabaseError::Connection(format!("Qdrant connection test failed: {}", e)))?;

        info!("✅ All database connections tested successfully");
        Ok(())
    }

    /// Get sync strategy reference
    pub fn sync_strategy(&self) -> &ResilientSyncStrategy {
        &self.sync_strategy
    }

    /// Get SurrealDB adapter reference
    pub fn surrealdb(&self) -> &SurrealDBAdapter {
        &self.surrealdb
    }

    /// Get Qdrant adapter reference  
    pub fn qdrant(&self) -> &QdrantAdapter {
        &self.qdrant
    }
}

/// Database management errors
#[derive(Debug, thiserror::Error)]
pub enum DatabaseError {
    #[error("Database connection error: {0}")]
    Connection(String),

    #[error("Sync engine error: {0}")]
    Sync(String),

    #[error("Health check error: {0}")]
    HealthCheck(String),
}