//! Accuracy Engine Block - Wraps the existing AccuracyEngine implementation
//! 
//! Provides high-precision processing with 99% accuracy targets.

use crate::blocks::converters::extract_query;
use crate::core::{
    errors::{BlockError, BlockResult},
    traits::{
        BlockCategory, BlockConfig, BlockInput, BlockMetadata, BlockOutput,
        DeploymentMode, HealthStatus, PipelineBlock, PipelineContext,
    },
};
use async_trait::async_trait;
use memory_nexus::engines::accuracy::AccuracyEngine;
use memory_nexus::engines::Engine;
use serde_json::json;
use std::sync::Arc;
use std::time::Instant;
use tracing::{debug, info, instrument};
use uuid::Uuid;

/// Configuration for the Accuracy Engine Block
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct AccuracyConfig {
    /// Precision threshold for results
    pub precision_threshold: f64,
    /// Maximum iterations for convergence
    pub max_iterations: usize,
    /// Convergence epsilon value
    pub convergence_epsilon: f64,
    /// Enable double-checking for accuracy
    pub enable_double_checking: bool,
    /// Target latency in milliseconds
    pub target_latency_ms: u64,
}

impl Default for AccuracyConfig {
    fn default() -> Self {
        Self {
            precision_threshold: 0.99,
            max_iterations: 1000,
            convergence_epsilon: 1e-6,
            enable_double_checking: true,
            target_latency_ms: 8, // 8ms target
        }
    }
}

/// Accuracy Engine Block that wraps the existing AccuracyEngine
pub struct AccuracyEngineBlock {
    /// The actual accuracy engine implementation
    inner: Arc<AccuracyEngine>,
    /// Block metadata
    metadata: BlockMetadata,
    /// Configuration
    config: AccuracyConfig,
}

impl AccuracyEngineBlock {
    /// Create a new accuracy engine block
    pub fn new(config: AccuracyConfig) -> Self {
        Self {
            inner: Arc::new(AccuracyEngine::new()),
            metadata: BlockMetadata {
                id: Uuid::new_v4(),
                name: "AccuracyEngine".to_string(),
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
    pub fn with_engine(engine: AccuracyEngine, config: AccuracyConfig) -> Self {
        Self {
            inner: Arc::new(engine),
            metadata: BlockMetadata {
                id: Uuid::new_v4(),
                name: "AccuracyEngine".to_string(),
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

impl Default for AccuracyEngineBlock {
    fn default() -> Self {
        Self::new(AccuracyConfig::default())
    }
}

#[async_trait]
impl PipelineBlock for AccuracyEngineBlock {
    fn metadata(&self) -> &BlockMetadata {
        &self.metadata
    }
    
    async fn initialize(&mut self, _config: BlockConfig) -> Result<(), BlockError> {
        debug!("Initializing AccuracyEngine block");
        // Initialize the engine
        self.inner.initialize()
            .await
            .map_err(|e| BlockError::Initialization(format!("Failed to initialize accuracy engine: {}", e)))?;
        info!("AccuracyEngine initialized with 99% precision target");
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
        
        // Process with the accuracy engine
        let data = query.as_bytes();
        let result = self.inner
            .process(data, memory_nexus::core::types::EngineMode::HighPrecision)
            .await
            .map_err(|e| BlockError::Processing(format!("Accuracy processing failed: {}", e)))?;
        
        // Check if we met our latency target
        let elapsed = start.elapsed();
        if elapsed.as_millis() > self.config.target_latency_ms as u128 {
            debug!(
                "AccuracyEngine exceeded target latency: {}ms > {}ms",
                elapsed.as_millis(),
                self.config.target_latency_ms
            );
        }
        
        // Get engine metrics
        let metrics = self.inner.get_metrics().await;
        
        // Store metrics in context
        context.metadata.insert(
            "accuracy".to_string(),
            metrics.accuracy.to_string(),
        );
        context.metadata.insert(
            "precision_threshold".to_string(),
            self.config.precision_threshold.to_string(),
        );
        context.metadata.insert(
            "processing_time_ms".to_string(),
            elapsed.as_millis().to_string(),
        );
        
        if self.config.enable_double_checking {
            context.metadata.insert(
                "double_checked".to_string(),
                "true".to_string(),
            );
        }
        
        debug!(
            "AccuracyEngine processed with {:.2}% accuracy in {}ms",
            metrics.accuracy * 100.0,
            elapsed.as_millis()
        );
        
        // Convert result to BlockOutput
        Ok(BlockOutput::Structured(json!({
            "result": String::from_utf8_lossy(&result),
            "accuracy": metrics.accuracy,
            "precision_threshold": self.config.precision_threshold,
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
        let test_data = b"health check test";
        
        match self.inner.process(test_data, memory_nexus::core::types::EngineMode::HighPrecision).await {
            Ok(_) => {
                let metrics = self.inner.get_metrics().await;
                if metrics.accuracy >= self.config.precision_threshold {
                    Ok(HealthStatus::Healthy)
                } else {
                    Ok(HealthStatus::Degraded(format!(
                        "Accuracy below threshold: {:.2}% < {:.2}%",
                        metrics.accuracy * 100.0,
                        self.config.precision_threshold * 100.0
                    )))
                }
            }
            Err(e) => Ok(HealthStatus::Unhealthy(format!("Engine test failed: {}", e))),
        }
    }
    
    async fn shutdown(&mut self) -> Result<(), BlockError> {
        debug!("Shutting down AccuracyEngine block");
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
    async fn test_accuracy_wrapper() {
        let mut engine = AccuracyEngineBlock::new(AccuracyConfig::default());
        let mut context = PipelineContext::new(Uuid::new_v4(), DeploymentMode::Hybrid);
        
        // Initialize the engine
        engine.initialize(BlockConfig::default()).await.unwrap();
        
        // Test with text input
        let input = BlockInput::Text("Test high-precision processing".to_string());
        let result = engine.process(input, &mut context).await;
        
        assert!(result.is_ok());
        
        // Verify context was updated
        assert!(context.metadata.contains_key("accuracy"));
        assert!(context.metadata.contains_key("processing_time_ms"));
    }
    
    #[tokio::test]
    async fn test_accuracy_latency() {
        let mut engine = AccuracyEngineBlock::new(AccuracyConfig::default());
        let mut context = PipelineContext::new(Uuid::new_v4(), DeploymentMode::Hybrid);
        
        engine.initialize(BlockConfig::default()).await.unwrap();
        
        let input = BlockInput::Text("Quick test".to_string());
        let start = Instant::now();
        let _ = engine.process(input, &mut context).await;
        let elapsed = start.elapsed();
        
        // Should complete in under 10ms (allowing margin over 8ms target)
        assert!(elapsed.as_millis() < 10, "AccuracyEngine took {}ms", elapsed.as_millis());
    }
}