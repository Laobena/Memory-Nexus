//! Learning Engine Block - Wraps the existing LearningEngine implementation
//! 
//! Provides adaptive learning and user preference modeling.

use crate::blocks::converters::extract_query;
use crate::core::{
    errors::{BlockError, BlockResult},
    traits::{
        BlockCategory, BlockConfig, BlockInput, BlockMetadata, BlockOutput,
        DeploymentMode, HealthStatus, PipelineBlock, PipelineContext,
    },
};
use async_trait::async_trait;
use memory_nexus::engines::learning::LearningEngine;
use memory_nexus::engines::Engine;
use serde_json::json;
use std::sync::Arc;
use std::time::Instant;
use tracing::{debug, info, instrument};
use uuid::Uuid;

/// Configuration for the Learning Engine Block
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct LearningConfig {
    /// Learning rate for model updates
    pub learning_rate: f32,
    /// Batch size for training
    pub batch_size: usize,
    /// Enable user preference learning
    pub enable_user_preferences: bool,
    /// History window size
    pub history_window: usize,
    /// Target latency in milliseconds
    pub target_latency_ms: u64,
}

impl Default for LearningConfig {
    fn default() -> Self {
        Self {
            learning_rate: 0.01,
            batch_size: 32,
            enable_user_preferences: true,
            history_window: 100,
            target_latency_ms: 10, // 10ms target
        }
    }
}

/// Learning Engine Block that wraps the existing LearningEngine
pub struct LearningEngineBlock {
    /// The actual learning engine implementation
    inner: Arc<LearningEngine>,
    /// Block metadata
    metadata: BlockMetadata,
    /// Configuration
    config: LearningConfig,
}

impl LearningEngineBlock {
    /// Create a new learning engine block
    pub fn new(config: LearningConfig) -> Self {
        Self {
            inner: Arc::new(LearningEngine::new()),
            metadata: BlockMetadata {
                id: Uuid::new_v4(),
                name: "LearningEngine".to_string(),
                version: "1.0.0".to_string(),
                category: BlockCategory::Processing,
                deployment_mode: DeploymentMode::Hybrid,
                hot_swappable: true,
                c_abi_compatible: false,
            },
            config,
        }
    }
    
    /// Create with custom engine instance
    pub fn with_engine(engine: LearningEngine, config: LearningConfig) -> Self {
        Self {
            inner: Arc::new(engine),
            metadata: BlockMetadata {
                id: Uuid::new_v4(),
                name: "LearningEngine".to_string(),
                version: "1.0.0".to_string(),
                category: BlockCategory::Processing,
                deployment_mode: DeploymentMode::Hybrid,
                hot_swappable: true,
                c_abi_compatible: false,
            },
            config,
        }
    }
}

impl Default for LearningEngineBlock {
    fn default() -> Self {
        Self::new(LearningConfig::default())
    }
}

#[async_trait]
impl PipelineBlock for LearningEngineBlock {
    fn metadata(&self) -> &BlockMetadata {
        &self.metadata
    }
    
    async fn initialize(&mut self, _config: BlockConfig) -> Result<(), BlockError> {
        debug!("Initializing LearningEngine block");
        // Initialize the engine
        self.inner.initialize()
            .await
            .map_err(|e| BlockError::Initialization(format!("Failed to initialize learning engine: {}", e)))?;
        info!("LearningEngine initialized with adaptive learning");
        Ok(())
    }
    
    #[instrument(skip(self, input, context))]
    async fn process(
        &self,
        input: BlockInput,
        context: &mut PipelineContext,
    ) -> BlockResult<BlockOutput> {
        let start = Instant::now();
        
        // Extract query or data from input
        let query = extract_query(&input)?;
        
        // Check for user_id in context
        let user_id = context.user_id.clone().unwrap_or_else(|| "anonymous".to_string());
        
        // Process with the learning engine
        let data = query.as_bytes();
        let result = self.inner
            .process(data, memory_nexus::core::types::EngineMode::Adaptive)
            .await
            .map_err(|e| BlockError::Processing(format!("Learning processing failed: {}", e)))?;
        
        // Check if we met our latency target
        let elapsed = start.elapsed();
        if elapsed.as_millis() > self.config.target_latency_ms as u128 {
            debug!(
                "LearningEngine exceeded target latency: {}ms > {}ms",
                elapsed.as_millis(),
                self.config.target_latency_ms
            );
        }
        
        // Get engine metrics
        let metrics = self.inner.get_metrics().await;
        
        // Store metrics in context
        context.metadata.insert(
            "learning_rate".to_string(),
            self.config.learning_rate.to_string(),
        );
        context.metadata.insert(
            "adaptation_score".to_string(),
            metrics.accuracy.to_string(),
        );
        context.metadata.insert(
            "user_id".to_string(),
            user_id.clone(),
        );
        context.metadata.insert(
            "learning_time_ms".to_string(),
            elapsed.as_millis().to_string(),
        );
        
        if self.config.enable_user_preferences {
            context.metadata.insert(
                "user_preferences_enabled".to_string(),
                "true".to_string(),
            );
        }
        
        debug!(
            "LearningEngine adapted for user '{}' in {}ms",
            user_id,
            elapsed.as_millis()
        );
        
        // Convert result to BlockOutput
        Ok(BlockOutput::Structured(json!({
            "result": String::from_utf8_lossy(&result),
            "adaptation_score": metrics.accuracy,
            "user_id": user_id,
            "learning_rate": self.config.learning_rate,
            "processing_time_ms": elapsed.as_millis(),
        })))
    }
    
    fn validate_input(&self, input: &BlockInput) -> Result<(), BlockError> {
        // Ensure we can extract query/data
        extract_query(input)?;
        Ok(())
    }
    
    async fn health_check(&self) -> Result<HealthStatus, BlockError> {
        // Test engine with dummy data
        let test_data = b"health check test for learning";
        
        match self.inner.process(test_data, memory_nexus::core::types::EngineMode::Adaptive).await {
            Ok(_) => {
                let metrics = self.inner.get_metrics().await;
                Ok(HealthStatus::Healthy)
            }
            Err(e) => Ok(HealthStatus::Unhealthy(format!("Engine test failed: {}", e))),
        }
    }
    
    async fn shutdown(&mut self) -> Result<(), BlockError> {
        debug!("Shutting down LearningEngine block");
        Ok(())
    }
    
    fn clone_box(&self) -> Box<dyn PipelineBlock> {
        Box::new(Self {
            inner: Arc::clone(&self.inner),
            metadata: self.metadata.clone(),
            config: self.config.clone(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_learning_wrapper() {
        let mut engine = LearningEngineBlock::new(LearningConfig::default());
        let mut context = PipelineContext::new(Uuid::new_v4(), DeploymentMode::Hybrid);
        
        // Initialize the engine
        engine.initialize(BlockConfig::default()).await.unwrap();
        
        // Set user_id in context
        context.user_id = Some("test_user_123".to_string());
        
        // Test with text input
        let input = BlockInput::Text("Learn from user interactions".to_string());
        let result = engine.process(input, &mut context).await;
        
        assert!(result.is_ok());
        
        // Verify context was updated
        assert!(context.metadata.contains_key("adaptation_score"));
        assert!(context.metadata.contains_key("user_id"));
        assert!(context.metadata.contains_key("learning_time_ms"));
    }
    
    #[tokio::test]
    async fn test_learning_latency() {
        let mut engine = LearningEngineBlock::new(LearningConfig::default());
        let mut context = PipelineContext::new(Uuid::new_v4(), DeploymentMode::Hybrid);
        
        engine.initialize(BlockConfig::default()).await.unwrap();
        
        let input = BlockInput::Text("Quick learning test".to_string());
        let start = Instant::now();
        let _ = engine.process(input, &mut context).await;
        let elapsed = start.elapsed();
        
        // Should complete in under 12ms (allowing margin over 10ms target)
        assert!(elapsed.as_millis() < 12, "LearningEngine took {}ms", elapsed.as_millis());
    }
}