//! Memory Nexus Database Adapters
//! 
//! Essential database connection infrastructure for SurrealDB and Qdrant.
//! This module provides only connection management and basic operations.
//! All search logic and complex queries are removed.

pub mod error;
pub mod health;
pub mod qdrant_adapter;
pub mod surreal_adapter;

// Re-export essential types
pub use error::{DatabaseError, DatabaseResult};
pub use health::{HealthCheck, HealthStatus};
pub use qdrant_adapter::QdrantAdapter;
pub use surreal_adapter::SurrealDBAdapter;

// Essential data types
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

/// Basic memory entry structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryEntry {
    pub id: Uuid,
    pub content: String,
    pub user_id: String,
    pub created_at: u64,
    pub updated_at: u64,
    pub embedding: Option<Vec<f32>>,
    pub metadata: HashMap<String, String>,
}

impl MemoryEntry {
    pub fn new(content: String, user_id: String) -> Self {
        let now = chrono::Utc::now().timestamp() as u64;
        Self {
            id: Uuid::new_v4(),
            content,
            user_id,
            created_at: now,
            updated_at: now,
            embedding: None,
            metadata: HashMap::new(),
        }
    }
}