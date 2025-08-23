//! Mining Engine Block - Wraps the existing MiningEngine implementation
//! 
//! Provides pattern discovery and data mining capabilities.

use crate::blocks::converters::extract_query;
use crate::core::{
    errors::{BlockError, BlockResult},
    traits::{
        BlockCategory, BlockConfig, BlockInput, BlockMetadata, BlockOutput,
        DeploymentMode, HealthStatus, PipelineBlock, PipelineContext,
    },
};
use async_trait::async_trait;
use memory_nexus::engines::mining::MiningEngine;
use memory_nexus::engines::Engine;
use serde_json::json;
use std::sync::Arc;
use std::time::Instant;
use tracing::{debug, info, instrument};
use uuid::Uuid;

/// Configuration for the Mining Engine Block
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct MiningConfig {
    /// Pattern discovery threshold
    pub pattern_threshold: f32,
    /// Anomaly detection sensitivity
    pub anomaly_sensitivity: f32,
    /// Mining depth levels
    pub mining_depth: usize,
    /// Enable trend analysis
    pub enable_trend_analysis: bool,
    /// Target latency in milliseconds
    pub target_latency_ms: u64,
}

impl Default for MiningConfig {
    fn default() -> Self {
        Self {
            pattern_threshold: 0.65,
            anomaly_sensitivity: 0.85,
            mining_depth: 3,
            enable_trend_analysis: true,
            target_latency_ms: 15, // 15ms target
        }
    }
}

/// Mining Engine Block that wraps the existing MiningEngine
pub struct MiningEngineBlock {
    /// The actual mining engine implementation
    inner: Arc<MiningEngine>,
    /// Block metadata
    metadata: BlockMetadata,
    /// Configuration
    config: MiningConfig,
}

impl MiningEngineBlock {
    /// Create a new mining engine block
    pub fn new(config: MiningConfig) -> Self {
        Self {
            inner: Arc::new(MiningEngine::new()),
            metadata: BlockMetadata {
                id: Uuid::new_v4(),
                name: "MiningEngine".to_string(),
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
    pub fn with_engine(engine: MiningEngine, config: MiningConfig) -> Self {
        Self {
            inner: Arc::new(engine),
            metadata: BlockMetadata {
                id: Uuid::new_v4(),
                name: "MiningEngine".to_string(),
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

impl Default for MiningEngineBlock {
    fn default() -> Self {
        Self::new(MiningConfig::default())
    }
}

#[async_trait]
impl PipelineBlock for MiningEngineBlock {
    fn metadata(&self) -> &BlockMetadata {
        &self.metadata
    }
    
    async fn initialize(&mut self, _config: BlockConfig) -> Result<(), BlockError> {
        debug!("Initializing MiningEngine block");
        // Initialize the engine
        self.inner.initialize()
            .await
            .map_err(|e| BlockError::Initialization(format!("Failed to initialize mining engine: {}", e)))?;
        info!("MiningEngine initialized with pattern discovery");
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
        
        // Process with the mining engine
        let data = query.as_bytes();
        let result = self.inner
            .process(data, memory_nexus::core::types::EngineMode::Mining)
            .await
            .map_err(|e| BlockError::Processing(format!("Mining processing failed: {}", e)))?;
        
        // Check if we met our latency target
        let elapsed = start.elapsed();
        if elapsed.as_millis() > self.config.target_latency_ms as u128 {
            debug!(
                "MiningEngine exceeded target latency: {}ms > {}ms",
                elapsed.as_millis(),
                self.config.target_latency_ms
            );
        }
        
        // Get engine metrics
        let metrics = self.inner.get_metrics().await;
        
        // Store metrics in context
        context.metadata.insert(
            "pattern_threshold".to_string(),
            self.config.pattern_threshold.to_string(),
        );
        context.metadata.insert(
            "anomaly_sensitivity".to_string(),
            self.config.anomaly_sensitivity.to_string(),
        );
        context.metadata.insert(
            "patterns_discovered".to_string(),
            metrics.throughput.to_string(), // Using throughput as pattern count proxy
        );
        context.metadata.insert(
            "mining_time_ms".to_string(),
            elapsed.as_millis().to_string(),
        );
        
        if self.config.enable_trend_analysis {
            context.metadata.insert(
                "trend_analysis_enabled".to_string(),
                "true".to_string(),
            );
        }
        
        debug!(
            "MiningEngine discovered patterns at depth {} in {}ms",
            self.config.mining_depth,
            elapsed.as_millis()
        );
        
        // Convert result to BlockOutput
        Ok(BlockOutput::Structured(json!({
            "result": String::from_utf8_lossy(&result),
            "pattern_score": metrics.accuracy,
            "mining_depth": self.config.mining_depth,
            "anomaly_sensitivity": self.config.anomaly_sensitivity,
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
        let test_data = b"health check test for mining patterns";
        
        match self.inner.process(test_data, memory_nexus::core::types::EngineMode::Mining).await {
            Ok(_) => {
                let metrics = self.inner.get_metrics().await;
                if metrics.accuracy >= self.config.pattern_threshold as f64 {
                    Ok(HealthStatus::Healthy)
                } else {
                    Ok(HealthStatus::Degraded(format!(
                        "Pattern score below threshold: {:.2}% < {:.2}%",
                        metrics.accuracy * 100.0,
                        self.config.pattern_threshold * 100.0
                    )))
                }
            }
            Err(e) => Ok(HealthStatus::Unhealthy(format!("Engine test failed: {}", e))),
        }
    }
    
    async fn shutdown(&mut self) -> Result<(), BlockError) {
        debug!("Shutting down MiningEngine block");
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
    async fn test_mining_wrapper() {
        let mut engine = MiningEngineBlock::new(MiningConfig::default());
        let mut context = PipelineContext::new(Uuid::new_v4(), DeploymentMode::Hybrid);
        
        // Initialize the engine
        engine.initialize(BlockConfig::default()).await.unwrap();
        
        // Test with text input
        let input = BlockInput::Text("Discover patterns in historical data".to_string());
        let result = engine.process(input, &mut context).await;
        
        assert!(result.is_ok());
        
        // Verify context was updated
        assert!(context.metadata.contains_key("patterns_discovered"));
        assert!(context.metadata.contains_key("anomaly_sensitivity"));
        assert!(context.metadata.contains_key("mining_time_ms"));
    }
    
    #[tokio::test]
    async fn test_mining_latency() {
        let mut engine = MiningEngineBlock::new(MiningConfig::default());
        let mut context = PipelineContext::new(Uuid::new_v4(), DeploymentMode::Hybrid);
        
        engine.initialize(BlockConfig::default()).await.unwrap();
        
        let input = BlockInput::Text("Quick mining test".to_string());
        let start = Instant::now();
        let _ = engine.process(input, &mut context).await;
        let elapsed = start.elapsed();
        
        // Should complete in under 18ms (allowing margin over 15ms target)
        assert!(elapsed.as_millis() < 18, "MiningEngine took {}ms", elapsed.as_millis());
    }
}