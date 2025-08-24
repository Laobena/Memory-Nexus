//! UUID Generator Block - Foundation of the reference system

use crate::core::{
    BlockError, BlockResult, PipelineBlock, DeploymentMode, BlockType,
    BlockMetadata, BlockCategory, BlockConfig, BlockInput, BlockOutput,
    PipelineContext, HealthStatus, BlockMetrics, RecoveryStrategy
};
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
    Processing,
    Response,
}

/// UUID metadata for tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
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

/// UUID context returned to pipeline
#[derive(Debug, Clone)]
pub struct UUIDContext {
    pub uuid: Uuid,
    pub metadata: UUIDMetadata,
    pub registry: Arc<DashMap<Uuid, UUIDMetadata>>,
}

/// Link type for UUID relationships
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LinkType {
    Parent,
    Child,
    Reference,
    Derived,
}

/// UUID configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UUIDConfig {
    pub enable_checksums: bool,
    pub session_tracking: bool,
    pub max_retries: u32,
    pub enable_metrics: bool,
}

impl Default for UUIDConfig {
    fn default() -> Self {
        Self {
            enable_checksums: true,
            session_tracking: true,
            max_retries: 3,
            enable_metrics: true,
        }
    }
}

/// UUID generator block
#[derive(Debug, Clone)]
pub struct UUIDBlock {
    metadata: BlockMetadata,
    config: UUIDConfig,
    registry: Arc<DashMap<Uuid, UUIDMetadata>>,
    sessions: Arc<DashMap<String, Vec<Uuid>>>,
    relationships: Arc<DashMap<(Uuid, Uuid), LinkType>>,
    metrics: Arc<RwLock<BlockMetrics>>,
    collision_counter: Arc<parking_lot::Mutex<u32>>,
}

impl UUIDBlock {
    /// Create new UUID block
    pub fn new(config: UUIDConfig) -> Self {
        Self {
            metadata: BlockMetadata {
                id: Uuid::new_v4(),
                name: "UUID Generator".to_string(),
                version: "1.0.0".to_string(),
                category: BlockCategory::Registration,
                deployment_mode: DeploymentMode::Hybrid,
                hot_swappable: true,
                c_abi_compatible: true,
            },
            config,
            registry: Arc::new(DashMap::new()),
            sessions: Arc::new(DashMap::new()),
            relationships: Arc::new(DashMap::new()),
            metrics: Arc::new(RwLock::new(BlockMetrics::default())),
            collision_counter: Arc::new(parking_lot::Mutex::new(0)),
        }
    }
    
    /// Generate unique UUID with collision detection
    async fn generate_unique_uuid(&self) -> BlockResult<Uuid> {
        let mut retries = 0;
        
        loop {
            let uuid = Uuid::new_v4();
            
            // Check for collision (extremely rare)
            if !self.registry.contains_key(&uuid) {
                return Ok(uuid);
            }
            
            // Track collision for metrics
            let mut counter = self.collision_counter.lock();
            *counter += 1;
            
            retries += 1;
            if retries >= self.config.max_retries {
                return Err(BlockError::Internal {
                    message: "UUID collision after max retries".to_string(),
                    recoverable: false,
                });
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
        
        if let Some(parent) = input.parent_uuid {
            hasher.update(parent.as_bytes());
        }
        
        *hasher.finalize().as_bytes()
    }
    
    /// Register UUID in the registry
    async fn register_uuid(&self, metadata: UUIDMetadata) -> BlockResult<()> {
        self.registry.insert(metadata.uuid, metadata.clone());
        
        if self.config.enable_metrics {
            let mut metrics = self.metrics.write();
            metrics.requests_processed += 1;
        }
        
        Ok(())
    }
    
    /// Link two UUIDs with a relationship
    async fn link_uuids(&self, from: Uuid, to: Uuid, link_type: LinkType) -> BlockResult<()> {
        self.relationships.insert((from, to), link_type);
        
        // Update metadata for both UUIDs
        if let Some(mut from_meta) = self.registry.get_mut(&from) {
            if link_type == LinkType::Child {
                from_meta.child_uuids.push(to);
            } else {
                from_meta.references.push(to);
            }
        }
        
        if let Some(mut to_meta) = self.registry.get_mut(&to) {
            if link_type == LinkType::Child {
                to_meta.parent_uuid = Some(from);
            } else {
                to_meta.references.push(from);
            }
        }
        
        Ok(())
    }
    
    /// Get all UUIDs for a session
    pub fn get_session_uuids(&self, session_id: &str) -> Vec<Uuid> {
        self.sessions
            .get(session_id)
            .map(|uuids| uuids.clone())
            .unwrap_or_default()
    }
    
    /// Get metadata for a UUID
    pub fn get_metadata(&self, uuid: &Uuid) -> Option<UUIDMetadata> {
        self.registry.get(uuid).map(|meta| meta.clone())
    }
    
    /// Get all child UUIDs
    pub fn get_children(&self, parent: &Uuid) -> Vec<Uuid> {
        self.registry
            .get(parent)
            .map(|meta| meta.child_uuids.clone())
            .unwrap_or_default()
    }
    
    /// Get parent UUID
    pub fn get_parent(&self, child: &Uuid) -> Option<Uuid> {
        self.registry
            .get(child)
            .and_then(|meta| meta.parent_uuid)
    }
    
    /// Clean up old sessions (garbage collection)
    pub async fn cleanup_old_sessions(&self, older_than: DateTime<Utc>) -> usize {
        let mut removed = 0;
        let mut sessions_to_remove = Vec::new();
        
        // Find sessions to remove
        for session in self.sessions.iter() {
            let session_id = session.key().clone();
            let uuids = session.value().clone();
            
            // Check if all UUIDs in session are old
            let all_old = uuids.iter().all(|uuid| {
                self.registry
                    .get(uuid)
                    .map(|meta| meta.timestamp < older_than)
                    .unwrap_or(true)
            });
            
            if all_old {
                sessions_to_remove.push(session_id);
                removed += uuids.len();
                
                // Remove UUIDs from registry
                for uuid in uuids {
                    self.registry.remove(&uuid);
                }
            }
        }
        
        // Remove sessions
        for session_id in sessions_to_remove {
            self.sessions.remove(&session_id);
        }
        
        removed
    }
}

#[async_trait]
impl PipelineBlock for UUIDBlock {
    fn metadata(&self) -> &BlockMetadata {
        &self.metadata
    }
    
    async fn initialize(&mut self, _config: BlockConfig) -> Result<(), BlockError> {
        // UUID block is always ready
        Ok(())
    }
    
    async fn process(
        &self,
        input: BlockInput,
        context: &mut PipelineContext,
    ) -> BlockResult<BlockOutput> {
        // Extract request from input
        let request = match input {
            BlockInput::Structured(json) => {
                serde_json::from_value::<UUIDRequest>(json)
                    .map_err(|e| BlockError::Validation {
                        field: "input".to_string(),
                        message: format!("Invalid UUID request: {}", e),
                    })?
            }
            _ => {
                // Create default request for other input types
                UUIDRequest {
                    session_id: context.request_id.to_string(),
                    user_id: context.user_id.clone().unwrap_or_else(|| "anonymous".to_string()),
                    org_id: "default".to_string(),
                    request_type: RequestType::Query,
                    parent_uuid: None,
                    metadata: serde_json::json!({}),
                }
            }
        };
        
        // Generate UUID with retry on collision (extremely rare)
        let uuid = self.generate_unique_uuid().await?;
        
        // Calculate checksum for integrity
        let checksum = if self.config.enable_checksums {
            self.calculate_checksum(&request)
        } else {
            [0u8; 32]
        };
        
        // Create metadata
        let metadata = UUIDMetadata {
            uuid,
            timestamp: Utc::now(),
            session_id: request.session_id.clone(),
            user_id: request.user_id.clone(),
            organization_id: request.org_id.clone(),
            checksum,
            references: Vec::new(),
            block_type: BlockType::Pipeline,
            request_type: request.request_type,
            parent_uuid: request.parent_uuid,
            child_uuids: Vec::new(),
        };
        
        // Register in all systems
        self.register_uuid(metadata.clone()).await?;
        
        // Track in session if enabled
        if self.config.session_tracking {
            self.sessions
                .entry(request.session_id.clone())
                .or_insert_with(Vec::new)
                .push(uuid);
        }
        
        // Link to parent if provided
        if let Some(parent) = request.parent_uuid {
            self.link_uuids(parent, uuid, LinkType::Child).await?;
        }
        
        // Store UUID in context metadata
        context.metadata.insert("uuid".to_string(), uuid.to_string());
        context.metadata.insert("uuid_timestamp".to_string(), metadata.timestamp.to_rfc3339());
        
        // Return UUID context as structured output
        Ok(BlockOutput::Structured(serde_json::json!({
            "uuid": uuid,
            "metadata": metadata,
            "success": true
        })))
    }
    
    fn validate_input(&self, _input: &BlockInput) -> Result<(), BlockError> {
        // UUID block accepts any input
        Ok(())
    }
    
    fn recovery_strategy(&self) -> RecoveryStrategy {
        RecoveryStrategy::Retry(Default::default())
    }
    
    async fn health_check(&self) -> Result<HealthStatus, BlockError> {
        let metrics = self.metrics.read();
        let registry_size = self.registry.len();
        let session_count = self.sessions.len();
        
        Ok(HealthStatus::Healthy)
    }
    
    fn metrics(&self) -> BlockMetrics {
        self.metrics.read().clone()
    }
    
    fn clone_box(&self) -> Box<dyn PipelineBlock> {
        Box::new(self.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_uuid_generation() {
        let mut block = UUIDBlock::new(UUIDConfig::default());
        block.initialize(BlockConfig::default()).await.unwrap();
        
        let request = UUIDRequest {
            session_id: "test-session".to_string(),
            user_id: "user-123".to_string(),
            org_id: "org-456".to_string(),
            request_type: RequestType::Query,
            parent_uuid: None,
            metadata: serde_json::json!({"test": true}),
        };
        
        let input = BlockInput::Structured(serde_json::to_value(request).unwrap());
        let mut context = PipelineContext::new(Uuid::new_v4(), DeploymentMode::Standalone);
        
        let result = block.process(input, &mut context).await.unwrap();
        
        match result {
            BlockOutput::Structured(json) => {
                assert!(json["success"].as_bool().unwrap());
                assert!(json["uuid"].as_str().is_some());
            }
            _ => panic!("Expected structured output"),
        }
    }
    
    #[tokio::test]
    async fn test_parent_child_linking() {
        let mut block = UUIDBlock::new(UUIDConfig::default());
        block.initialize(BlockConfig::default()).await.unwrap();
        
        let mut context = PipelineContext::new(Uuid::new_v4(), DeploymentMode::Standalone);
        
        // Create parent
        let parent_request = UUIDRequest {
            session_id: "test-session".to_string(),
            user_id: "user-123".to_string(),
            org_id: "org-456".to_string(),
            request_type: RequestType::Query,
            parent_uuid: None,
            metadata: serde_json::json!({}),
        };
        
        let parent_input = BlockInput::Structured(serde_json::to_value(parent_request).unwrap());
        let parent_result = block.process(parent_input, &mut context).await.unwrap();
        
        let parent_uuid = match parent_result {
            BlockOutput::Structured(json) => {
                Uuid::parse_str(json["uuid"].as_str().unwrap()).unwrap()
            }
            _ => panic!("Expected structured output"),
        };
        
        // Create child
        let child_request = UUIDRequest {
            session_id: "test-session".to_string(),
            user_id: "user-123".to_string(),
            org_id: "org-456".to_string(),
            request_type: RequestType::Processing,
            parent_uuid: Some(parent_uuid),
            metadata: serde_json::json!({}),
        };
        
        let child_input = BlockInput::Structured(serde_json::to_value(child_request).unwrap());
        let child_result = block.process(child_input, &mut context).await.unwrap();
        
        let child_uuid = match child_result {
            BlockOutput::Structured(json) => {
                Uuid::parse_str(json["uuid"].as_str().unwrap()).unwrap()
            }
            _ => panic!("Expected structured output"),
        };
        
        // Verify relationship
        assert_eq!(block.get_parent(&child_uuid), Some(parent_uuid));
        assert!(block.get_children(&parent_uuid).contains(&child_uuid));
    }
}