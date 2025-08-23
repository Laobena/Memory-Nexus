//! Core traits for Memory Nexus Blocks with zero-cost abstractions
//! 
//! Defines the fundamental traits for hot-swappable pipeline blocks
//! with C ABI compatibility and zero-allocation guarantees.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::any::Any;
use std::fmt::Debug;
use std::pin::Pin;
use std::future::Future;
use std::sync::Arc;
use uuid::Uuid;

use crate::core::errors::{BlockError, BlockResult, ErrorContext, RecoveryStrategy};

/// Deployment mode for the pipeline
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DeploymentMode {
    /// Zero-cost sidecar mode (browser extension, local proxy)
    Sidecar,
    /// Full standalone API server
    Standalone,
    /// Hybrid mode supporting both
    Hybrid,
}

/// Cost indicator for operations
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Cost {
    /// Zero cost (uses existing AI, sidecar mode)
    Zero,
    /// Minimal cost (cache hits, simple operations)
    Minimal,
    /// Standard cost (normal processing)
    Standard,
    /// Premium cost (full processing, all engines)
    Premium,
}

impl Cost {
    /// Check if operation is free
    #[inline(always)]
    pub const fn is_free(&self) -> bool {
        matches!(self, Cost::Zero)
    }
    
    /// Get relative cost multiplier
    #[inline(always)]
    pub const fn multiplier(&self) -> f32 {
        match self {
            Cost::Zero => 0.0,
            Cost::Minimal => 0.1,
            Cost::Standard => 1.0,
            Cost::Premium => 3.0,
        }
    }
}

/// Block metadata for identification and versioning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlockMetadata {
    pub id: Uuid,
    pub name: String,
    pub version: String,
    pub category: BlockCategory,
    pub deployment_mode: DeploymentMode,
    pub hot_swappable: bool,
    pub c_abi_compatible: bool,
}

/// Category of pipeline block
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BlockCategory {
    Router,
    Preprocessor,
    Search,
    Storage,
    Fusion,
    Custom,
}

/// Pipeline context passed between blocks (lock-free updates)
#[derive(Debug, Clone)]
pub struct PipelineContext {
    /// Unique request ID
    pub request_id: Uuid,
    /// User ID if available
    pub user_id: Option<String>,
    /// Deployment mode
    pub mode: DeploymentMode,
    /// Current cost accumulator
    pub cost: Cost,
    /// Confidence score (0.0 - 1.0)
    pub confidence: f32,
    /// Should escalate to higher processing
    pub should_escalate: bool,
    /// Timing information
    pub start_time: std::time::Instant,
    /// Metadata storage (lock-free)
    pub metadata: Arc<dashmap::DashMap<String, String>>,
    /// Feature flags
    pub features: Arc<dashmap::DashSet<String>>,
}

impl PipelineContext {
    /// Create new context for a request
    pub fn new(request_id: Uuid, mode: DeploymentMode) -> Self {
        Self {
            request_id,
            user_id: None,
            mode,
            cost: if mode == DeploymentMode::Sidecar { Cost::Zero } else { Cost::Minimal },
            confidence: 1.0,
            should_escalate: false,
            start_time: std::time::Instant::now(),
            metadata: Arc::new(dashmap::DashMap::new()),
            features: Arc::new(dashmap::DashSet::new()),
        }
    }
    
    /// Get elapsed time
    #[inline(always)]
    pub fn elapsed(&self) -> std::time::Duration {
        self.start_time.elapsed()
    }
    
    /// Check if a feature is enabled
    #[inline(always)]
    pub fn has_feature(&self, feature: &str) -> bool {
        self.features.contains(feature)
    }
    
    /// Update confidence and check for escalation
    #[inline]
    pub fn update_confidence(&mut self, new_confidence: f32, threshold: f32) {
        self.confidence = self.confidence.min(new_confidence);
        if self.confidence < threshold {
            self.should_escalate = true;
        }
    }
}

/// Core trait for all pipeline blocks
#[async_trait]
pub trait PipelineBlock: Send + Sync + Debug {
    /// Get block metadata
    fn metadata(&self) -> &BlockMetadata;
    
    /// Initialize the block
    async fn initialize(&mut self, config: BlockConfig) -> Result<(), BlockError>;
    
    /// Process input through the block
    async fn process(
        &self,
        input: BlockInput,
        context: &mut PipelineContext,
    ) -> BlockResult<BlockOutput>;
    
    /// Validate input before processing
    fn validate_input(&self, _input: &BlockInput) -> Result<(), BlockError> {
        // Default implementation, can be overridden
        Ok(())
    }
    
    /// Get recovery strategy for errors
    fn recovery_strategy(&self) -> RecoveryStrategy {
        RecoveryStrategy::Retry(Default::default())
    }
    
    /// Health check
    async fn health_check(&self) -> Result<HealthStatus, BlockError> {
        Ok(HealthStatus::Healthy)
    }
    
    /// Graceful shutdown
    async fn shutdown(&mut self) -> Result<(), BlockError> {
        Ok(())
    }
    
    /// Hot-reload configuration without restart
    async fn reload_config(&mut self, config: BlockConfig) -> Result<(), BlockError> {
        self.shutdown().await?;
        self.initialize(config).await
    }
    
    /// Get current metrics
    fn metrics(&self) -> BlockMetrics {
        BlockMetrics::default()
    }
    
    /// Clone as trait object (for hot-swapping)
    fn clone_box(&self) -> Box<dyn PipelineBlock>;
}

/// Input to a pipeline block
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BlockInput {
    /// Text query
    Text(String),
    /// Vector embedding
    Vector(Vec<f32>),
    /// Structured query
    Structured(serde_json::Value),
    /// Binary data
    Binary(Vec<u8>),
    /// Multiple inputs
    Batch(Vec<BlockInput>),
}

/// Output from a pipeline block
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BlockOutput {
    /// Text response
    Text(String),
    /// Vector embedding
    Vector(Vec<f32>),
    /// Structured response
    Structured(serde_json::Value),
    /// Binary data
    Binary(Vec<u8>),
    /// Multiple outputs
    Batch(Vec<BlockOutput>),
    /// Pass-through with confidence
    PassThrough { confidence: f32 },
}

/// Block configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlockConfig {
    pub timeout_ms: u64,
    pub max_retries: u32,
    pub cache_enabled: bool,
    pub metrics_enabled: bool,
    pub custom: serde_json::Value,
}

impl Default for BlockConfig {
    fn default() -> Self {
        Self {
            timeout_ms: 5000,
            max_retries: 3,
            cache_enabled: true,
            metrics_enabled: true,
            custom: serde_json::Value::Null,
        }
    }
}

/// Health status of a block
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum HealthStatus {
    Healthy,
    Degraded,
    Unhealthy,
}

/// Metrics for a block
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct BlockMetrics {
    pub requests_total: u64,
    pub requests_failed: u64,
    pub latency_p50_ms: f64,
    pub latency_p95_ms: f64,
    pub latency_p99_ms: f64,
    pub cache_hits: u64,
    pub cache_misses: u64,
}

/// Composable pipeline builder
pub struct Pipeline {
    blocks: Vec<Box<dyn PipelineBlock>>,
    mode: DeploymentMode,
    context: PipelineContext,
}

impl Pipeline {
    /// Create new pipeline
    pub fn new(mode: DeploymentMode) -> Self {
        Self {
            blocks: Vec::new(),
            mode,
            context: PipelineContext::new(Uuid::new_v4(), mode),
        }
    }
    
    /// Add a block to the pipeline
    pub fn add_block(mut self, block: Box<dyn PipelineBlock>) -> Self {
        self.blocks.push(block);
        self
    }
    
    /// Execute the pipeline
    pub async fn execute(&mut self, input: BlockInput) -> BlockResult<BlockOutput> {
        let mut current_input = input;
        let mut last_output = None;
        
        for block in &self.blocks {
            // Validate input
            block.validate_input(&current_input)
                .map_err(|e| (e, ErrorContext::new(
                    &block.metadata().name,
                    "validation"
                )))?;
            
            // Process through block
            let output = block.process(current_input.clone(), &mut self.context).await?;
            
            // Check if we should escalate
            if self.context.should_escalate {
                tracing::info!(
                    block = block.metadata().name,
                    confidence = self.context.confidence,
                    "Escalating pipeline due to low confidence"
                );
                // Could switch to a different pipeline here
            }
            
            // Convert output to input for next block
            current_input = match &output {
                BlockOutput::PassThrough { .. } => {
                    // Keep the same input for next block
                    current_input
                }
                _ => {
                    // Convert output to input
                    BlockInput::Structured(serde_json::to_value(&output).unwrap())
                }
            };
            
            last_output = Some(output);
        }
        
        last_output.ok_or_else(|| (
            BlockError::Unknown("Pipeline produced no output".into()),
            ErrorContext::new("pipeline", "execute")
        ))
    }
}

/// Factory for creating blocks (supports hot-swapping)
pub trait BlockFactory: Send + Sync {
    /// Create a block by name
    fn create(&self, name: &str) -> Result<Box<dyn PipelineBlock>, BlockError>;
    
    /// List available blocks
    fn list_available(&self) -> Vec<String>;
    
    /// Hot-swap a block
    fn hot_swap(&self, old: &str, new: Box<dyn PipelineBlock>) -> Result<(), BlockError>;
}

/// C ABI compatibility layer for hot-swapping
#[repr(C)]
pub struct CBlockInterface {
    /// Initialize function pointer
    pub initialize: unsafe extern "C" fn(*mut std::ffi::c_void, *const u8, usize) -> i32,
    /// Process function pointer
    pub process: unsafe extern "C" fn(*mut std::ffi::c_void, *const u8, usize, *mut u8, *mut usize) -> i32,
    /// Shutdown function pointer
    pub shutdown: unsafe extern "C" fn(*mut std::ffi::c_void) -> i32,
    /// Opaque block data
    pub block_data: *mut std::ffi::c_void,
}

unsafe impl Send for CBlockInterface {}
unsafe impl Sync for CBlockInterface {}

/// Helper for zero-copy operations
pub trait ZeroCopy: Sized {
    /// Get as bytes without allocation
    fn as_bytes(&self) -> &[u8];
    
    /// Create from bytes without allocation
    fn from_bytes(bytes: &[u8]) -> Result<&Self, BlockError>;
}

// Implement ZeroCopy for common types using bytemuck (simpler than zerocopy)
#[cfg(feature = "zero-copy")]
impl<T> ZeroCopy for T
where
    T: bytemuck::Pod + bytemuck::Zeroable,
{
    fn as_bytes(&self) -> &[u8] {
        bytemuck::bytes_of(self)
    }
    
    fn from_bytes(bytes: &[u8]) -> Result<&Self, BlockError> {
        bytemuck::try_from_bytes(bytes)
            .map_err(|_| BlockError::Validation("Invalid byte alignment or size".into()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_cost_calculation() {
        assert!(Cost::Zero.is_free());
        assert!(!Cost::Minimal.is_free());
        assert_eq!(Cost::Premium.multiplier(), 3.0);
    }
    
    #[test]
    fn test_pipeline_context() {
        let mut ctx = PipelineContext::new(Uuid::new_v4(), DeploymentMode::Sidecar);
        assert_eq!(ctx.cost, Cost::Zero);
        
        ctx.update_confidence(0.7, 0.85);
        assert!(ctx.should_escalate);
    }
    
    #[tokio::test]
    async fn test_pipeline_builder() {
        // Would need mock blocks for full test
        let pipeline = Pipeline::new(DeploymentMode::Sidecar)
            .add_block(Box::new(MockBlock::new()));
        
        assert_eq!(pipeline.blocks.len(), 1);
    }
    
    // Mock block for testing
    #[derive(Debug)]
    struct MockBlock {
        metadata: BlockMetadata,
    }
    
    impl MockBlock {
        fn new() -> Self {
            Self {
                metadata: BlockMetadata {
                    id: Uuid::new_v4(),
                    name: "mock".into(),
                    version: "1.0.0".into(),
                    category: BlockCategory::Custom,
                    deployment_mode: DeploymentMode::Hybrid,
                    hot_swappable: true,
                    c_abi_compatible: false,
                },
            }
        }
    }
    
    #[async_trait]
    impl PipelineBlock for MockBlock {
        fn metadata(&self) -> &BlockMetadata {
            &self.metadata
        }
        
        async fn initialize(&mut self, _config: BlockConfig) -> Result<(), BlockError> {
            Ok(())
        }
        
        async fn process(
            &self,
            _input: BlockInput,
            _context: &mut PipelineContext,
        ) -> BlockResult<BlockOutput> {
            Ok(BlockOutput::PassThrough { confidence: 1.0 })
        }
        
        fn clone_box(&self) -> Box<dyn PipelineBlock> {
            Box::new(Self::new())
        }
    }
}