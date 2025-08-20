//! SurrealDB database adapter - Connection and basic operations only

use crate::error::{DatabaseError, DatabaseResult};
use crate::health::{HealthCheck, HealthReport};
use crate::MemoryEntry;
use async_trait::async_trait;
use surrealdb::{engine::remote::ws::{Ws, Wss}, opt::auth::Root, Surreal};
use std::sync::Arc;
use std::time::Instant;
use tracing::{debug, error, info};

/// SurrealDB adapter for graph operations
pub struct SurrealDBAdapter {
    client: Arc<Surreal<surrealdb::engine::remote::ws::Client>>,
    url: String,
    database: String,
    namespace: String,
}

impl SurrealDBAdapter {
    /// Create new SurrealDB adapter
    pub async fn new(url: &str, username: &str, password: &str) -> DatabaseResult<Self> {
        // Determine if this is a secure WebSocket connection
        let client = if url.starts_with("wss://") {
            Surreal::new::<Wss>(url)
        } else {
            Surreal::new::<Ws>(url)
        }
        .await
        .map_err(|e| DatabaseError::connection(format!("Failed to connect to SurrealDB: {}", e)))?;

        // Sign in
        client
            .signin(Root {
                username,
                password,
            })
            .await
            .map_err(|e| DatabaseError::connection(format!("SurrealDB authentication failed: {}", e)))?;

        // Use namespace and database
        let namespace = "memory_nexus";
        let database = "main";
        
        client
            .use_ns(namespace)
            .use_db(database)
            .await
            .map_err(|e| DatabaseError::connection(format!("Failed to use SurrealDB namespace/database: {}", e)))?;

        let adapter = Self {
            client: Arc::new(client),
            url: url.to_string(),
            database: database.to_string(),
            namespace: namespace.to_string(),
        };

        info!("✅ SurrealDB adapter initialized: {}", url);
        Ok(adapter)
    }

    /// Get SurrealDB client reference
    pub fn client(&self) -> &Surreal<surrealdb::engine::remote::ws::Client> {
        &self.client
    }

    /// Get database name
    pub fn database(&self) -> &str {
        &self.database
    }

    /// Get namespace
    pub fn namespace(&self) -> &str {
        &self.namespace
    }

    /// Basic connection test
    pub async fn test_connection(&self) -> DatabaseResult<()> {
        // Simple query to test connection
        let _result: Vec<surrealdb::sql::Value> = self.client
            .query("SELECT 1 as test")
            .await
            .map_err(|e| DatabaseError::connection(format!("SurrealDB connection test failed: {}", e)))?
            .take(0)
            .map_err(|e| DatabaseError::connection(format!("SurrealDB query parsing failed: {}", e)))?;
        
        debug!("SurrealDB connection test successful");
        Ok(())
    }

    /// Store a memory entry (basic implementation)
    pub async fn store_memory(&self, memory: &MemoryEntry) -> DatabaseResult<()> {
        // This is a placeholder - actual storage would be implemented here
        debug!("Store memory called for ID: {}", memory.id);
        Ok(())
    }

    /// Retrieve memory by ID (basic implementation)
    pub async fn get_memory(&self, memory_id: &str) -> DatabaseResult<Option<MemoryEntry>> {
        // This is a placeholder - actual retrieval would be implemented here  
        debug!("Get memory called for ID: {}", memory_id);
        Ok(None)
    }

    /// List all memories for a user (basic implementation)
    pub async fn list_memories(&self, user_id: &str, limit: usize) -> DatabaseResult<Vec<MemoryEntry>> {
        // This is a placeholder - actual listing would be implemented here
        debug!("List memories called for user: {}, limit: {}", user_id, limit);
        Ok(Vec::new())
    }

    /// Search memories by content (basic implementation)
    pub async fn search_memories(&self, query: &str, user_id: &str, limit: usize) -> DatabaseResult<Vec<MemoryEntry>> {
        // This is a placeholder - actual search would be implemented here
        debug!("Search memories called for query: '{}', user: {}, limit: {}", query, user_id, limit);
        Ok(Vec::new())
    }
}

#[async_trait]
impl HealthCheck for SurrealDBAdapter {
    async fn health_check(&self) -> DatabaseResult<HealthReport> {
        let start = Instant::now();
        
        match self.test_connection().await {
            Ok(_) => {
                let response_time = start.elapsed().as_millis() as u64;
                Ok(HealthReport::healthy(response_time))
            }
            Err(e) => {
                error!("SurrealDB health check failed: {}", e);
                Ok(HealthReport::unhealthy(format!("SurrealDB connection failed: {}", e)))
            }
        }
    }

    async fn readiness_check(&self) -> DatabaseResult<bool> {
        self.test_connection().await.map(|_| true).or(Ok(false))
    }

    async fn liveness_check(&self) -> DatabaseResult<bool> {
        self.test_connection().await.map(|_| true).or(Ok(false))
    }
}