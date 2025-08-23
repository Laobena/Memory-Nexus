//! UUID Generator Block - Foundation of the reference system

use crate::core::{BlockError, BlockResult, PipelineBlock, DeploymentMode, BlockType};
use uuid::Uuid;
use dashmap::DashMap;
use chrono::{DateTime, Utc};
use std::sync::Arc;
use parking_lot::RwLock;
use async_trait::async_trait;
use std::collections::{HashSet, VecDeque};
use blake3::Hasher;
use serde::{Serialize, Deserialize};

/// UUID request input
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UUIDRequest {
    pub session_id: String,
    pub user_id: String,
    pub org_id: String,
    pub request_type: RequestType,
    pub parent_uuid: Option<Uuid>,
    pub metadata: serde_json::Value,
}

/// Request type for tracking
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum RequestType {
    Query,
    Search,
    Storage,
    Fusion,
    Cache,
}

/// UUID context output
#[derive(Debug, Clone)]
pub struct UUIDContext {
    pub uuid: Uuid,
    pub metadata: UUIDMetadata,
    pub registry: Arc<DashMap<Uuid, UUIDMetadata>>,
}

/// UUID configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UUIDConfig {
    pub enable_checksums: bool,
    pub max_registry_size: usize,
    pub session_tracking: bool,
    pub recovery_log_size: usize,
}

impl Default for UUIDConfig {
    fn default() -> Self {
        Self {
            enable_checksums: true,
            max_registry_size: 100_000,
            session_tracking: true,
            recovery_log_size: 10_000,
        }
    }
}

/// UUID metadata for tracking
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct UUIDMetadata {
    pub uuid: Uuid,
    pub timestamp: DateTime<Utc>,
    pub session_id: String,
    pub user_id: String,
    pub organization_id: String,
    pub checksum: [u8; 32],
    pub references: Vec<Uuid>,
    pub block_type: BlockType,
    pub request_type: RequestType,
    pub parent_uuid: Option<Uuid>,
    pub child_uuids: Vec<Uuid>,
}

/// Link type for UUID relationships
#[derive(Debug, Clone, Copy)]
pub enum LinkType {
    Parent,
    Child,
    Reference,
    Derived,
}

/// Recovery operation for audit log
#[derive(Debug, Clone)]
pub struct UUIDRecovery {
    pub uuid: Uuid,
    pub timestamp: DateTime<Utc>,
    pub operation: RecoveryOp,
}

/// Recovery operation type
#[derive(Debug, Clone, Copy)]
pub enum RecoveryOp {
    Created,
    Linked,
    Updated,
    Deleted,
}

/// UUID Generator Block - Foundation of reference system
pub struct UUIDBlock {
    /// Registry of all generated UUIDs
    registry: Arc<DashMap<Uuid, UUIDMetadata>>,
    /// Session tracking
    sessions: Arc<DashMap<String, Vec<Uuid>>>,
    /// Checksum validator
    checksum_cache: Arc<DashMap<Uuid, [u8; 32]>>,
    /// Error recovery
    recovery_log: Arc<RwLock<Vec<UUIDRecovery>>>,
    /// Configuration
    config: UUIDConfig,
    /// Metrics
    metrics: Arc<UUIDMetrics>,
}

/// Metrics for UUID operations
#[derive(Debug, Default)]
struct UUIDMetrics {
    generated: std::sync::atomic::AtomicU64,
    collisions: std::sync::atomic::AtomicU64,
    links_created: std::sync::atomic::AtomicU64,
    checksum_failures: std::sync::atomic::AtomicU64,
}

impl UUIDBlock {
    /// Create new UUID generator block
    pub fn new(config: UUIDConfig) -> Self {
        Self {
            registry: Arc::new(DashMap::with_capacity(config.max_registry_size / 10)),
            sessions: Arc::new(DashMap::new()),
            checksum_cache: Arc::new(DashMap::new()),
            recovery_log: Arc::new(RwLock::new(Vec::with_capacity(config.recovery_log_size))),
            config,
            metrics: Arc::new(UUIDMetrics::default()),
        }
    }

    /// Generate guaranteed unique UUID
    async fn generate_unique_uuid(&self) -> Result<Uuid, BlockError> {
        use std::sync::atomic::Ordering;
        
        let mut attempts = 0;
        loop {
            let uuid = Uuid::new_v4();
            
            // Check uniqueness (UUID v4 has 122 bits of randomness)
            if !self.registry.contains_key(&uuid) {
                self.metrics.generated.fetch_add(1, Ordering::Relaxed);
                return Ok(uuid);
            }
            
            attempts += 1;
            if attempts > 3 {
                self.metrics.collisions.fetch_add(1, Ordering::Relaxed);
                // This should never happen (probability ~10^-38)
                return Err(BlockError::Unknown("UUID collision detected".into()));
            }
        }
    }
    
    /// Calculate checksum for integrity
    fn calculate_checksum(&self, input: &UUIDRequest) -> [u8; 32] {
        let mut hasher = Hasher::new();
        hasher.update(input.session_id.as_bytes());
        hasher.update(input.user_id.as_bytes());
        hasher.update(input.org_id.as_bytes());
        hasher.update(&[input.request_type as u8]);
        
        if let Ok(metadata_bytes) = serde_json::to_vec(&input.metadata) {
            hasher.update(&metadata_bytes);
        }
        
        let hash = hasher.finalize();
        let mut checksum = [0u8; 32];
        checksum.copy_from_slice(hash.as_bytes());
        checksum
    }
    
    /// Register UUID in all tracking systems
    async fn register_uuid(&self, metadata: UUIDMetadata) -> Result<(), BlockError> {
        let uuid = metadata.uuid;
        
        // Store in registry
        self.registry.insert(uuid, metadata.clone());
        
        // Store checksum for validation
        if self.config.enable_checksums {
            self.checksum_cache.insert(uuid, metadata.checksum);
        }
        
        // Log for recovery
        if let Ok(mut log) = self.recovery_log.try_write() {
            log.push(UUIDRecovery {
                uuid,
                timestamp: metadata.timestamp,
                operation: RecoveryOp::Created,
            });
            
            // Keep only last N entries
            if log.len() > self.config.recovery_log_size {
                log.drain(0..self.config.recovery_log_size / 2);
            }
        }
        
        Ok(())
    }
    
    /// Create bidirectional reference
    pub async fn link_uuids(&self, from: Uuid, to: Uuid, link_type: LinkType) -> Result<(), BlockError> {
        use std::sync::atomic::Ordering;
        
        // Update 'from' references
        if let Some(mut from_meta) = self.registry.get_mut(&from) {
            match link_type {
                LinkType::Child => from_meta.child_uuids.push(to),
                _ => from_meta.references.push(to),
            }
        } else {
            return Err(BlockError::Unknown(format!("UUID not found: {}", from)));
        }
        
        // Update 'to' references
        if let Some(mut to_meta) = self.registry.get_mut(&to) {
            match link_type {
                LinkType::Parent => to_meta.parent_uuid = Some(from),
                _ => to_meta.references.push(from),
            }
        }
        
        self.metrics.links_created.fetch_add(1, Ordering::Relaxed);
        
        // Log the link operation
        if let Ok(mut log) = self.recovery_log.try_write() {
            log.push(UUIDRecovery {
                uuid: from,
                timestamp: Utc::now(),
                operation: RecoveryOp::Linked,
            });
        }
        
        Ok(())
    }
    
    /// Get complete reference chain
    pub async fn get_reference_chain(&self, uuid: Uuid) -> Result<Vec<Uuid>, BlockError> {
        let mut chain = Vec::new();
        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();
        
        queue.push_back(uuid);
        visited.insert(uuid);
        
        while let Some(current) = queue.pop_front() {
            chain.push(current);
            
            if let Some(metadata) = self.registry.get(&current) {
                // Add all references
                for reference in &metadata.references {
                    if !visited.contains(reference) {
                        visited.insert(*reference);
                        queue.push_back(*reference);
                    }
                }
                
                // Add children
                for child in &metadata.child_uuids {
                    if !visited.contains(child) {
                        visited.insert(*child);
                        queue.push_back(*child);
                    }
                }
                
                // Add parent
                if let Some(parent) = metadata.parent_uuid {
                    if !visited.contains(&parent) {
                        visited.insert(parent);
                        queue.push_back(parent);
                    }
                }
            }
        }
        
        Ok(chain)
    }
    
    /// Validate UUID checksum
    pub fn validate_checksum(&self, uuid: Uuid, checksum: &[u8; 32]) -> bool {
        if !self.config.enable_checksums {
            return true;
        }
        
        if let Some(stored_checksum) = self.checksum_cache.get(&uuid) {
            stored_checksum.as_ref() == checksum
        } else {
            false
        }
    }
    
    /// Get session UUIDs
    pub fn get_session_uuids(&self, session_id: &str) -> Vec<Uuid> {
        self.sessions
            .get(session_id)
            .map(|uuids| uuids.clone())
            .unwrap_or_default()
    }
    
    /// Clean up old sessions
    pub async fn cleanup_old_sessions(&self, older_than: DateTime<Utc>) -> usize {
        let mut removed = 0;
        
        // Collect sessions to remove
        let sessions_to_remove: Vec<String> = self.sessions
            .iter()
            .filter_map(|entry| {
                let session_id = entry.key().clone();
                let uuids = entry.value();
                
                // Check if all UUIDs in session are old
                let all_old = uuids.iter().all(|uuid| {
                    self.registry
                        .get(uuid)
                        .map(|meta| meta.timestamp < older_than)
                        .unwrap_or(true)
                });
                
                if all_old {
                    Some(session_id)
                } else {
                    None
                }
            })
            .collect();
        
        // Remove old sessions
        for session_id in sessions_to_remove {
            if let Some((_, uuids)) = self.sessions.remove(&session_id) {
                removed += uuids.len();
                
                // Also remove from registry
                for uuid in uuids {
                    self.registry.remove(&uuid);
                    self.checksum_cache.remove(&uuid);
                }
            }
        }
        
        removed
    }
}

#[async_trait]
impl PipelineBlock for UUIDBlock {
    type Input = UUIDRequest;
    type Output = UUIDContext;
    type Config = UUIDConfig;
    
    async fn execute(
        &self,
        input: Self::Input,
        _config: Self::Config,
        _deployment: &DeploymentMode,
    ) -> Result<Self::Output, BlockError> {
        // Generate UUID with retry on collision (extremely rare)
        let uuid = self.generate_unique_uuid().await?;
        
        // Calculate checksum for integrity
        let checksum = if self.config.enable_checksums {
            self.calculate_checksum(&input)
        } else {
            [0u8; 32]
        };
        
        // Create metadata
        let metadata = UUIDMetadata {
            uuid,
            timestamp: Utc::now(),
            session_id: input.session_id.clone(),
            user_id: input.user_id.clone(),
            organization_id: input.org_id.clone(),
            checksum,
            references: Vec::new(),
            block_type: BlockType::Pipeline,
            request_type: input.request_type,
            parent_uuid: input.parent_uuid,
            child_uuids: Vec::new(),
        };
        
        // Register in all systems
        self.register_uuid(metadata.clone()).await?;
        
        // Track in session if enabled
        if self.config.session_tracking {
            self.sessions
                .entry(input.session_id.clone())
                .or_insert_with(Vec::new)
                .push(uuid);
        }
        
        // Link to parent if provided
        if let Some(parent) = input.parent_uuid {
            self.link_uuids(parent, uuid, LinkType::Child).await?;
        }
        
        Ok(UUIDContext {
            uuid,
            metadata,
            registry: Arc::clone(&self.registry),
        })
    }
    
    fn estimated_latency_ms(&self) -> u32 { 1 } // <1ms
    
    fn resource_requirements(&self) -> crate::core::ResourceRequirements {
        crate::core::ResourceRequirements {
            memory_mb: 10,
            cpu_cores: 0.1,
            disk_mb: 0,
            network_bandwidth_mbps: 0.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_uuid_generation() {
        let block = UUIDBlock::new(UUIDConfig::default());
        
        let request = UUIDRequest {
            session_id: "test-session".to_string(),
            user_id: "user-123".to_string(),
            org_id: "org-456".to_string(),
            request_type: RequestType::Query,
            parent_uuid: None,
            metadata: serde_json::json!({"test": true}),
        };
        
        let result = block.execute(
            request.clone(),
            UUIDConfig::default(),
            &DeploymentMode::Standalone
        ).await.unwrap();
        
        assert_ne!(result.uuid, Uuid::nil());
        assert_eq!(result.metadata.session_id, "test-session");
        
        // Test uniqueness
        let result2 = block.execute(
            request,
            UUIDConfig::default(),
            &DeploymentMode::Standalone
        ).await.unwrap();
        
        assert_ne!(result.uuid, result2.uuid);
    }
    
    #[tokio::test]
    async fn test_uuid_linking() {
        let block = UUIDBlock::new(UUIDConfig::default());
        
        // Create parent
        let parent_request = UUIDRequest {
            session_id: "test".to_string(),
            user_id: "user".to_string(),
            org_id: "org".to_string(),
            request_type: RequestType::Query,
            parent_uuid: None,
            metadata: serde_json::json!({}),
        };
        
        let parent = block.execute(
            parent_request,
            UUIDConfig::default(),
            &DeploymentMode::Standalone
        ).await.unwrap();
        
        // Create child
        let child_request = UUIDRequest {
            session_id: "test".to_string(),
            user_id: "user".to_string(),
            org_id: "org".to_string(),
            request_type: RequestType::Search,
            parent_uuid: Some(parent.uuid),
            metadata: serde_json::json!({}),
        };
        
        let child = block.execute(
            child_request,
            UUIDConfig::default(),
            &DeploymentMode::Standalone
        ).await.unwrap();
        
        // Verify linking
        let chain = block.get_reference_chain(parent.uuid).await.unwrap();
        assert!(chain.contains(&parent.uuid));
        assert!(chain.contains(&child.uuid));
    }
}