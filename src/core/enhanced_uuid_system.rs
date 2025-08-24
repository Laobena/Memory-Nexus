//! Enhanced UUID System - Database-Powered Reference System
//! Leverages SurrealDB and Qdrant capabilities for intelligent memory management

use anyhow::{Context, Result, bail};
use chrono::{DateTime, Utc};
use qdrant_client::{
    client::QdrantClient,
    qdrant::{
        PointStruct, SearchPoints, Filter, Condition, Range,
        with_payload_selector::SelectorOptions, WithPayloadSelector,
    },
};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::collections::HashMap;
use std::sync::Arc;
use surrealdb::{engine::any::Any, Surreal};
use tracing::{debug, info, warn, error, instrument};
use uuid::Uuid;

use crate::core::uuid_types::{
    MemoryType, EvolutionType, RelationshipType, OriginalTruth, Memory,
    MemoryRelationship, MemoryEvolution, ProcessingLog, TemporalSearchResult,
    UserPatterns, MemoryChain, MemoryStats, UUIDConfig, UUIDError,
    calculate_content_hash, calculate_time_weight, get_time_context,
};
use crate::database::qdrant_setup::{
    create_temporal_point, temporal_vector_search,
    MEMORY_VECTORS_COLLECTION, TRUTH_VECTORS_COLLECTION,
};

/// Record type for SurrealDB queries
type Record = surrealdb::sql::Thing;

/// Enhanced UUID System - Database coordinator
pub struct EnhancedUUIDSystem {
    /// SurrealDB connection for graph and relational data
    surrealdb: Arc<Surreal<Any>>,
    
    /// Qdrant client for vector operations
    qdrant: Arc<QdrantClient>,
    
    /// System configuration
    config: UUIDConfig,
}

impl EnhancedUUIDSystem {
    /// Create a new Enhanced UUID System
    pub async fn new(
        surrealdb: Arc<Surreal<Any>>,
        qdrant: Arc<QdrantClient>,
    ) -> Result<Self> {
        info!("🚀 Initializing Enhanced UUID System");
        
        Ok(Self {
            surrealdb,
            qdrant,
            config: UUIDConfig::default(),
        })
    }
    
    /// Create with custom configuration
    pub async fn with_config(
        surrealdb: Arc<Surreal<Any>>,
        qdrant: Arc<QdrantClient>,
        config: UUIDConfig,
    ) -> Result<Self> {
        info!("🚀 Initializing Enhanced UUID System with custom config");
        
        Ok(Self {
            surrealdb,
            qdrant,
            config,
        })
    }
    
    /// Create from database pool (for pipeline integration)
    pub async fn with_database_pool(
        db_pool: Arc<crate::database::UnifiedDatabasePool>,
    ) -> Result<Self> {
        info!("🚀 Initializing Enhanced UUID System from database pool");
        
        // For now, we'll create new connections as we need the Arc<Surreal<Any>> type
        // In a production setup, we'd refactor the pool to share these connections
        
        // Get database config from environment
        let surreal_url = std::env::var("SURREALDB_URL")
            .unwrap_or_else(|_| "ws://localhost:8000".to_string());
        let surreal_ns = std::env::var("SURREALDB_NS")
            .unwrap_or_else(|_| "nexus".to_string());
        let surreal_db = std::env::var("SURREALDB_DB")
            .unwrap_or_else(|_| "memory".to_string());
        let surreal_user = std::env::var("SURREALDB_USER")
            .unwrap_or_else(|_| "root".to_string());
        let surreal_pass = std::env::var("SURREALDB_PASS")
            .unwrap_or_else(|_| "root".to_string());
        
        // Create SurrealDB connection
        let surrealdb = surrealdb::Surreal::new::<surrealdb::engine::any::Any>(surreal_url.as_str())
            .await
            .context("Failed to connect to SurrealDB")?;
        
        surrealdb.signin(surrealdb::opt::auth::Root {
            username: &surreal_user,
            password: &surreal_pass,
        }).await.context("Failed to authenticate with SurrealDB")?;
        
        surrealdb.use_ns(&surreal_ns).use_db(&surreal_db).await
            .context("Failed to select namespace and database")?;
        
        // Get Qdrant URL from environment
        let qdrant_url = std::env::var("QDRANT_URL")
            .unwrap_or_else(|_| "http://localhost:6334".to_string());
        
        // Create Qdrant client
        let qdrant = QdrantClient::from_url(&qdrant_url)
            .build()
            .context("Failed to create Qdrant client")?;
        
        Ok(Self {
            surrealdb: Arc::new(surrealdb),
            qdrant: Arc::new(qdrant),
            config: UUIDConfig::default(),
        })
    }
    
    // ============================================
    // TRUTH PRESERVATION
    // ============================================
    
    /// Preserve original truth (with automatic deduplication)
    #[instrument(skip(self, content))]
    pub async fn preserve_truth(
        &self,
        content: String,
        user_id: String,
        source_metadata: HashMap<String, serde_json::Value>,
    ) -> Result<Uuid> {
        let content_hash = calculate_content_hash(&content);
        
        // Check for existing truth with same hash (database handles deduplication!)
        if self.config.enable_deduplication {
            let existing: Vec<Record> = self.surrealdb
                .query("SELECT uuid FROM original_truth WHERE content_hash = $hash")
                .bind(("hash", &content_hash))
                .await?
                .take(0)?;
            
            if let Some(record) = existing.first() {
                // Content already exists, return existing UUID
                let uuid_str: String = self.surrealdb
                    .query("SELECT uuid FROM $record")
                    .bind(("record", record))
                    .await?
                    .take(0)?;
                
                debug!("Found existing truth with hash {}", content_hash);
                return Ok(Uuid::parse_str(&uuid_str)?);
            }
        }
        
        // Create new truth record
        let truth_uuid = Uuid::new_v4();
        
        let _: Option<Record> = self.surrealdb
            .create(("original_truth", truth_uuid.to_string()))
            .content(json!({
                "uuid": truth_uuid.to_string(),
                "raw_content": content,
                "content_hash": content_hash,
                "user_id": user_id,
                "created_at": Utc::now(),
                "source_metadata": source_metadata,
            }))
            .await
            .context("Failed to create original truth")?;
        
        info!("✅ Preserved original truth: {}", truth_uuid);
        Ok(truth_uuid)
    }
    
    // ============================================
    // MEMORY MANAGEMENT
    // ============================================
    
    /// Create a memory from a Memory struct (convenience wrapper)
    pub async fn create_memory_from_struct(&self, memory: Memory) -> Result<Uuid> {
        self.create_memory(
            memory.original_uuid,
            memory.content,
            memory.memory_type,
            memory.user_id,
            Some(memory.session_id),
            memory.parent_uuid,
            memory.confidence_score,
            memory.processing_path,
            memory.processing_time_ms,
            memory.metadata,
        ).await
    }
    
    /// Create a new memory linked to original truth
    #[instrument(skip(self, content))]
    pub async fn create_memory(
        &self,
        original_uuid: Uuid,
        content: String,
        memory_type: MemoryType,
        user_id: String,
        session_id: Option<String>,
        parent_uuid: Option<Uuid>,
        confidence_score: f32,
        processing_path: String,
        processing_time_ms: u64,
        metadata: HashMap<String, serde_json::Value>,
    ) -> Result<Uuid> {
        // Check confidence threshold
        if confidence_score < self.config.min_confidence_threshold {
            warn!("Memory confidence {} below threshold {}", 
                  confidence_score, self.config.min_confidence_threshold);
        }
        
        let memory_uuid = Uuid::new_v4();
        let session_id = session_id.unwrap_or_else(|| Uuid::new_v4().to_string());
        
        // Create memory record in SurrealDB
        let _: Option<Record> = self.surrealdb
            .create(("memory", memory_uuid.to_string()))
            .content(json!({
                "uuid": memory_uuid.to_string(),
                "original_uuid": original_uuid.to_string(),
                "content": content,
                "memory_type": memory_type.as_str(),
                "user_id": user_id,
                "session_id": session_id,
                "parent_uuid": parent_uuid.map(|u| u.to_string()),
                "confidence_score": confidence_score,
                "processing_path": processing_path,
                "processing_time_ms": processing_time_ms,
                "metadata": metadata,
                // created_at, last_accessed, access_count auto-set by database
            }))
            .await
            .context("Failed to create memory")?;
        
        // Create relationship to original truth
        self.create_relationship(
            original_uuid,
            memory_uuid,
            RelationshipType::TruthSource,
            1.0,
            HashMap::new(),
        ).await?;
        
        // Create parent-child relationship if parent exists
        if let Some(parent) = parent_uuid {
            self.create_relationship(
                parent,
                memory_uuid,
                RelationshipType::ParentChild,
                1.0,
                HashMap::new(),
            ).await?;
        }
        
        debug!("✅ Created memory {} linked to truth {}", memory_uuid, original_uuid);
        Ok(memory_uuid)
    }
    
    /// Store vector embedding for a memory
    #[instrument(skip(self, embedding))]
    pub async fn store_vector(
        &self,
        uuid: Uuid,
        original_uuid: Uuid,
        embedding: Vec<f32>,
        memory_type: MemoryType,
        user_id: String,
        additional_metadata: HashMap<String, serde_json::Value>,
    ) -> Result<()> {
        // Create metadata with temporal information
        let mut metadata = additional_metadata;
        metadata.insert("uuid".to_string(), json!(uuid.to_string()));
        metadata.insert("original_uuid".to_string(), json!(original_uuid.to_string()));
        metadata.insert("memory_type".to_string(), json!(memory_type.as_str()));
        metadata.insert("user_id".to_string(), json!(user_id));
        
        // Create point with temporal metadata (Qdrant handles indexing!)
        let point = create_temporal_point(
            &uuid.to_string(),
            embedding,
            metadata,
        );
        
        // Store in Qdrant
        self.qdrant
            .upsert_points_blocking(MEMORY_VECTORS_COLLECTION, vec![point], None)
            .await
            .context("Failed to store vector")?;
        
        // Also store metadata reference in SurrealDB
        let _: Option<Record> = self.surrealdb
            .create("vector_metadata")
            .content(json!({
                "uuid": uuid.to_string(),
                "vector_id": uuid.to_string(),
                "collection_name": MEMORY_VECTORS_COLLECTION,
                "created_at": Utc::now(),
            }))
            .await?;
        
        debug!("✅ Stored vector for memory {}", uuid);
        Ok(())
    }
    
    // ============================================
    // RELATIONSHIP MANAGEMENT
    // ============================================
    
    /// Create a relationship between memories
    #[instrument(skip(self))]
    pub async fn create_relationship(
        &self,
        from_uuid: Uuid,
        to_uuid: Uuid,
        relationship_type: RelationshipType,
        strength: f32,
        metadata: HashMap<String, serde_json::Value>,
    ) -> Result<()> {
        // Check if relationship already exists
        let existing: Vec<Record> = self.surrealdb
            .query("SELECT * FROM connects WHERE in.uuid = $from AND out.uuid = $to")
            .bind(("from", from_uuid.to_string()))
            .bind(("to", to_uuid.to_string()))
            .await?
            .take(0)?;
        
        if !existing.is_empty() {
            debug!("Relationship already exists between {} and {}", from_uuid, to_uuid);
            return Ok(());
        }
        
        // Create relationship (SurrealDB handles the graph!)
        let _: Option<Record> = self.surrealdb
            .query("RELATE $from->connects->$to SET relationship_type = $type, strength = $strength, metadata = $metadata")
            .bind(("from", format!("memory:{}", from_uuid)))
            .bind(("to", format!("memory:{}", to_uuid)))
            .bind(("type", relationship_type.as_str()))
            .bind(("strength", strength))
            .bind(("metadata", metadata))
            .await?
            .take(0)?;
        
        debug!("✅ Created {} relationship: {} -> {}", relationship_type.as_str(), from_uuid, to_uuid);
        Ok(())
    }
    
    // ============================================
    // EVOLUTION TRACKING
    // ============================================
    
    /// Track memory evolution
    #[instrument(skip(self))]
    pub async fn track_evolution(
        &self,
        from_uuid: Uuid,
        to_uuid: Uuid,
        evolution_type: EvolutionType,
        change_summary: Option<String>,
    ) -> Result<()> {
        // Get time gap between memories
        let time_gap: f32 = self.surrealdb
            .query("
                LET $old = (SELECT created_at FROM memory WHERE uuid = $from)[0];
                LET $new = (SELECT created_at FROM memory WHERE uuid = $to)[0];
                RETURN time::hour($new.created_at - $old.created_at);
            ")
            .bind(("from", from_uuid.to_string()))
            .bind(("to", to_uuid.to_string()))
            .await?
            .take(0)
            .unwrap_or(0.0);
        
        // Create evolution record
        let _: Option<Record> = self.surrealdb
            .query("RELATE $from->evolves->$to SET evolution_type = $type, time_gap_hours = $gap, change_summary = $summary")
            .bind(("from", format!("memory:{}", from_uuid)))
            .bind(("to", format!("memory:{}", to_uuid)))
            .bind(("type", evolution_type.as_str()))
            .bind(("gap", time_gap))
            .bind(("summary", change_summary))
            .await?
            .take(0)?;
        
        info!("✅ Tracked evolution: {} -> {} ({})", from_uuid, to_uuid, evolution_type);
        Ok(())
    }
    
    // ============================================
    // QUERYING AND RETRIEVAL
    // ============================================
    
    /// Get original truth for a memory
    #[instrument(skip(self))]
    pub async fn get_original_truth(&self, memory_uuid: Uuid) -> Result<OriginalTruth> {
        let result: Option<OriginalTruth> = self.surrealdb
            .query("
                LET $memory = (SELECT original_uuid FROM memory WHERE uuid = $uuid)[0];
                SELECT * FROM original_truth WHERE uuid = $memory.original_uuid;
            ")
            .bind(("uuid", memory_uuid.to_string()))
            .await?
            .take(0)?;
        
        result.ok_or_else(|| UUIDError::TruthNotFound(memory_uuid).into())
    }
    
    /// Get memory by UUID
    #[instrument(skip(self))]
    pub async fn get_memory(&self, uuid: Uuid) -> Result<Memory> {
        let memory: Option<Memory> = self.surrealdb
            .select(("memory", uuid.to_string()))
            .await?;
        
        // Update access tracking if enabled
        if self.config.enable_access_tracking {
            if memory.is_some() {
                let _: Option<Record> = self.surrealdb
                    .query("UPDATE memory SET last_accessed = time::now(), access_count += 1 WHERE uuid = $uuid")
                    .bind(("uuid", uuid.to_string()))
                    .await?
                    .take(0)?;
            }
        }
        
        memory.ok_or_else(|| UUIDError::MemoryNotFound(uuid).into())
    }
    
    /// Temporal search with time-weighted scoring
    #[instrument(skip(self, embedding))]
    pub async fn temporal_search(
        &self,
        embedding: Vec<f32>,
        user_id: &str,
        hours_back: i64,
        limit: usize,
    ) -> Result<Vec<TemporalSearchResult>> {
        // Use Qdrant's temporal search
        let results = temporal_vector_search(
            &self.qdrant,
            MEMORY_VECTORS_COLLECTION,
            embedding,
            Some(user_id),
            hours_back,
            limit as u64,
        ).await?;
        
        // Fetch full memory details from SurrealDB
        let mut temporal_results = Vec::new();
        
        for (uuid_str, vector_score, weighted_score) in results {
            let uuid = Uuid::parse_str(&uuid_str)?;
            
            // Get memory details
            if let Ok(memory) = self.get_memory(uuid).await {
                let age_hours = (Utc::now() - memory.created_at).num_hours() as f32;
                let time_weight = calculate_time_weight(memory.created_at, self.config.time_decay_hours);
                
                temporal_results.push(TemporalSearchResult {
                    uuid,
                    memory_type: memory.memory_type,
                    content: memory.content,
                    vector_score,
                    time_weight,
                    final_score: weighted_score,
                    age_hours,
                    created_at: memory.created_at,
                    metadata: memory.metadata,
                });
            }
        }
        
        Ok(temporal_results)
    }
    
    /// Get related memories through graph traversal
    #[instrument(skip(self))]
    pub async fn get_related_memories(
        &self,
        uuid: Uuid,
        max_depth: usize,
        limit: usize,
    ) -> Result<Vec<Memory>> {
        let query = format!(
            "SELECT ->connects[0..{}]->memory.* FROM memory WHERE uuid = $uuid LIMIT {}",
            max_depth, limit
        );
        
        let related: Vec<Memory> = self.surrealdb
            .query(&query)
            .bind(("uuid", uuid.to_string()))
            .await?
            .take(0)?;
        
        Ok(related)
    }
    
    /// Get memory evolution chain
    #[instrument(skip(self))]
    pub async fn get_evolution_chain(&self, uuid: Uuid) -> Result<MemoryChain> {
        // Find root of evolution chain
        let root_query = "
            LET $current = $uuid;
            LET $chain = [];
            
            WHILE (SELECT <-evolves<-memory FROM memory WHERE uuid = $current)[0] {
                LET $prev = (SELECT <-evolves<-memory FROM memory WHERE uuid = $current)[0];
                LET $current = $prev.uuid;
                LET $chain = array::push($chain, $prev);
            };
            
            RETURN $current;
        ";
        
        let root_uuid: Uuid = self.surrealdb
            .query(root_query)
            .bind(("uuid", uuid.to_string()))
            .await?
            .take(0)
            .unwrap_or(uuid);
        
        // Get all evolutions
        let evolutions: Vec<MemoryEvolution> = self.surrealdb
            .query("SELECT * FROM evolves WHERE in.uuid = $root OR out.uuid = $root ORDER BY evolved_at")
            .bind(("root", root_uuid.to_string()))
            .await?
            .take(0)?;
        
        let evolution_count = evolutions.len();
        let total_time_span = if let (Some(first), Some(last)) = (evolutions.first(), evolutions.last()) {
            (last.evolved_at - first.evolved_at).num_hours() as f32
        } else {
            0.0
        };
        
        Ok(MemoryChain {
            root_uuid,
            current_uuid: uuid,
            evolution_count,
            total_time_span_hours: total_time_span,
            evolutions,
        })
    }
    
    // ============================================
    // PATTERN DISCOVERY
    // ============================================
    
    /// Discover user patterns from historical data
    #[instrument(skip(self))]
    pub async fn discover_user_patterns(&self, user_id: &str) -> Result<UserPatterns> {
        // Analyze peak hours
        let peak_hours: Vec<u32> = self.surrealdb
            .query("
                SELECT time::hour(created_at) as hour, COUNT() as count
                FROM memory
                WHERE user_id = $user_id AND created_at > time::now() - 30d
                GROUP BY hour
                ORDER BY count DESC
                LIMIT 5
            ")
            .bind(("user_id", user_id))
            .await?
            .take(0)?;
        
        // Analyze favorite memory types
        let favorite_types_raw: Vec<String> = self.surrealdb
            .query("
                SELECT memory_type, COUNT() as count
                FROM memory
                WHERE user_id = $user_id AND created_at > time::now() - 30d
                GROUP BY memory_type
                ORDER BY count DESC
                LIMIT 5
            ")
            .bind(("user_id", user_id))
            .await?
            .take(0)?;
        
        let favorite_types: Vec<MemoryType> = favorite_types_raw
            .iter()
            .filter_map(|s| MemoryType::from_str(s))
            .collect();
        
        // Calculate average session length
        let avg_session_length: f32 = self.surrealdb
            .query("
                SELECT AVG(session_duration) as avg_duration FROM (
                    SELECT session_id, 
                           time::minute(MAX(created_at) - MIN(created_at)) as session_duration
                    FROM memory
                    WHERE user_id = $user_id AND created_at > time::now() - 30d
                    GROUP BY session_id
                )
            ")
            .bind(("user_id", user_id))
            .await?
            .take(0)
            .unwrap_or(0.0);
        
        Ok(UserPatterns {
            user_id: user_id.to_string(),
            peak_hours,
            favorite_types,
            common_domains: Vec::new(), // TODO: Implement domain extraction
            query_patterns: Vec::new(),  // TODO: Implement pattern extraction
            avg_session_length,
            discovered_at: Utc::now(),
            confidence: 0.85, // Default confidence
        })
    }
    
    // ============================================
    // AUDIT AND LOGGING
    // ============================================
    
    /// Log processing stage for audit trail
    #[instrument(skip(self))]
    pub async fn log_processing(
        &self,
        uuid: Uuid,
        original_uuid: Uuid,
        memory_uuid: Option<Uuid>,
        stage: &str,
        duration_ms: u64,
        success: bool,
        error_message: Option<String>,
        metadata: HashMap<String, serde_json::Value>,
    ) -> Result<()> {
        if !self.config.enable_audit_logs {
            return Ok(());
        }
        
        let _: Option<Record> = self.surrealdb
            .create("processing_log")
            .content(json!({
                "uuid": uuid.to_string(),
                "original_uuid": original_uuid.to_string(),
                "memory_uuid": memory_uuid.map(|u| u.to_string()),
                "stage": stage,
                "started_at": Utc::now() - chrono::Duration::milliseconds(duration_ms as i64),
                "completed_at": Utc::now(),
                "duration_ms": duration_ms,
                "success": success,
                "error_message": error_message,
                "metadata": metadata,
            }))
            .await?;
        
        debug!("📝 Logged processing: {} - {} ({}ms)", stage, 
               if success { "success" } else { "failed" }, duration_ms);
        Ok(())
    }
    
    /// Get processing history for a memory
    #[instrument(skip(self))]
    pub async fn get_processing_history(&self, uuid: Uuid) -> Result<Vec<ProcessingLog>> {
        let logs: Vec<ProcessingLog> = self.surrealdb
            .query("SELECT * FROM processing_log WHERE original_uuid = $uuid OR memory_uuid = $uuid ORDER BY started_at")
            .bind(("uuid", uuid.to_string()))
            .await?
            .take(0)?;
        
        Ok(logs)
    }
    
    // ============================================
    // STATISTICS
    // ============================================
    
    /// Get statistics for a memory
    #[instrument(skip(self))]
    pub async fn get_memory_stats(&self, uuid: Uuid) -> Result<MemoryStats> {
        let stats: Option<MemoryStats> = self.surrealdb
            .query("
                LET $memory = (SELECT * FROM memory WHERE uuid = $uuid)[0];
                LET $relationships = (SELECT COUNT() as count FROM connects WHERE in.uuid = $uuid OR out.uuid = $uuid)[0];
                LET $evolutions = (SELECT COUNT() as count FROM evolves WHERE in.uuid = $uuid OR out.uuid = $uuid)[0];
                LET $processing = (SELECT SUM(duration_ms) as total, AVG(confidence_score) as avg_conf FROM processing_log WHERE memory_uuid = $uuid);
                
                RETURN {
                    uuid: $memory.uuid,
                    access_count: $memory.access_count,
                    relationship_count: $relationships.count,
                    evolution_count: $evolutions.count,
                    avg_confidence: $processing[0].avg_conf,
                    total_processing_time_ms: $processing[0].total,
                    last_accessed: $memory.last_accessed
                };
            ")
            .bind(("uuid", uuid.to_string()))
            .await?
            .take(0)?;
        
        stats.ok_or_else(|| UUIDError::MemoryNotFound(uuid).into())
    }
    
    /// Get system-wide statistics
    #[instrument(skip(self))]
    pub async fn get_system_stats(&self) -> Result<serde_json::Value> {
        let stats: serde_json::Value = self.surrealdb
            .query("
                LET $total_memories = (SELECT COUNT() as count FROM memory)[0].count;
                LET $total_truths = (SELECT COUNT() as count FROM original_truth)[0].count;
                LET $total_relationships = (SELECT COUNT() as count FROM connects)[0].count;
                LET $total_evolutions = (SELECT COUNT() as count FROM evolves)[0].count;
                LET $active_users = (SELECT COUNT(DISTINCT user_id) as count FROM memory WHERE created_at > time::now() - 24h)[0].count;
                
                RETURN {
                    total_memories: $total_memories,
                    total_truths: $total_truths,
                    total_relationships: $total_relationships,
                    total_evolutions: $total_evolutions,
                    active_users_24h: $active_users,
                    deduplication_ratio: $total_truths / $total_memories
                };
            ")
            .await?
            .take(0)?;
        
        Ok(stats)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    // TODO: Add comprehensive tests for the Enhanced UUID System
    // Tests should cover:
    // - Truth preservation and deduplication
    // - Memory creation and linking
    // - Relationship management
    // - Evolution tracking
    // - Temporal search
    // - Pattern discovery
    // - Audit logging
}