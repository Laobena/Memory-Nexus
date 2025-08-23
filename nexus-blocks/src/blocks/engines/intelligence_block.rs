//! Intelligence Engine Block - Wraps the existing IntelligenceEngine implementation
//! 
//! Provides context-aware intelligent processing with cross-domain pattern matching.

use crate::blocks::converters::extract_query;
use crate::core::{
    errors::{BlockError, BlockResult},
    traits::{
        BlockCategory, BlockConfig, BlockInput, BlockMetadata, BlockOutput,
        DeploymentMode, HealthStatus, PipelineBlock, PipelineContext,
    },
};
use async_trait::async_trait;
use memory_nexus::engines::intelligence::IntelligenceEngine;
use memory_nexus::engines::Engine;
use serde_json::json;
use std::sync::Arc;
use std::time::Instant;
use tracing::{debug, info, instrument};
use uuid::Uuid;

/// Configuration for the Intelligence Engine Block
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct IntelligenceConfig {
    /// Context window size for analysis
    pub context_window_size: usize,
    /// Enable cross-domain analysis
    pub enable_cross_domain: bool,
    /// Pattern matching threshold
    pub pattern_threshold: f32,
    /// Maximum inference depth
    pub max_inference_depth: usize,
    /// Target latency in milliseconds
    pub target_latency_ms: u64,
}

impl Default for IntelligenceConfig {
    fn default() -> Self {
        Self {
            context_window_size: 2048,
            enable_cross_domain: true,
            pattern_threshold: 0.75,
            max_inference_depth: 5,
            target_latency_ms: 12, // 12ms target
        }
    }
}

/// Intelligence Engine Block that wraps the existing IntelligenceEngine
pub struct IntelligenceEngineBlock {
    /// The actual intelligence engine implementation
    inner: Arc<IntelligenceEngine>,
    /// Block metadata
    metadata: BlockMetadata,
    /// Configuration
    config: IntelligenceConfig,
}

impl IntelligenceEngineBlock {
    /// Create a new intelligence engine block
    pub fn new(config: IntelligenceConfig) -> Self {
        Self {
            inner: Arc::new(IntelligenceEngine::new()),
            metadata: BlockMetadata {
                id: Uuid::new_v4(),
                name: "IntelligenceEngine".to_string(),
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
    pub fn with_engine(engine: IntelligenceEngine, config: IntelligenceConfig) -> Self {
        Self {
            inner: Arc::new(engine),
            metadata: BlockMetadata {
                id: Uuid::new_v4(),
                name: "IntelligenceEngine".to_string(),
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

impl Default for IntelligenceEngineBlock {
    fn default() -> Self {
        Self::new(IntelligenceConfig::default())
    }
}

#[async_trait]
impl PipelineBlock for IntelligenceEngineBlock {
    fn metadata(&self) -> &BlockMetadata {
        &self.metadata
    }
    
    async fn initialize(&mut self, _config: BlockConfig) -> Result<(), BlockError> {
        debug!("Initializing IntelligenceEngine block");
        // Initialize the engine
        self.inner.initialize()
            .await
            .map_err(|e| BlockError::Initialization(format!("Failed to initialize intelligence engine: {}", e)))?;
        info!("IntelligenceEngine initialized with context-aware processing");
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
        
        // Check for domain hints in context
        let domain = context.metadata.get("domain")
            .map(|s| s.as_str())
            .unwrap_or("general");
        
        // Process with the intelligence engine
        let data = query.as_bytes();
        let result = self.inner
            .process(data, memory_nexus::core::types::EngineMode::Intelligent)
            .await
            .map_err(|e| BlockError::Processing(format!("Intelligence processing failed: {}", e)))?;
        
        // Check if we met our latency target
        let elapsed = start.elapsed();
        if elapsed.as_millis() > self.config.target_latency_ms as u128 {
            debug!(
                "IntelligenceEngine exceeded target latency: {}ms > {}ms",
                elapsed.as_millis(),
                self.config.target_latency_ms
            );
        }
        
        // Get engine metrics
        let metrics = self.inner.get_metrics().await;
        
        // Store metrics in context
        context.metadata.insert(
            "intelligence_score".to_string(),
            metrics.accuracy.to_string(),
        );
        context.metadata.insert(
            "context_window".to_string(),
            self.config.context_window_size.to_string(),
        );
        context.metadata.insert(
            "domain_analysis".to_string(),
            domain.to_string(),
        );
        context.metadata.insert(
            "intelligence_time_ms".to_string(),
            elapsed.as_millis().to_string(),
        );
        
        if self.config.enable_cross_domain {
            context.metadata.insert(
                "cross_domain_enabled".to_string(),
                "true".to_string(),
            );
        }
        
        debug!(
            "IntelligenceEngine processed with domain '{}' in {}ms",
            domain,
            elapsed.as_millis()
        );
        
        // Convert result to BlockOutput
        Ok(BlockOutput::Structured(json!({
            "result": String::from_utf8_lossy(&result),
            "intelligence_score": metrics.accuracy,
            "domain": domain,
            "context_window": self.config.context_window_size,
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
        let test_data = b"health check test for intelligence";
        
        match self.inner.process(test_data, memory_nexus::core::types::EngineMode::Intelligent).await {
            Ok(_) => {
                let metrics = self.inner.get_metrics().await;
                if metrics.accuracy >= self.config.pattern_threshold as f64 {
                    Ok(HealthStatus::Healthy)
                } else {
                    Ok(HealthStatus::Degraded(format!(
                        "Intelligence score below threshold: {:.2}% < {:.2}%",
                        metrics.accuracy * 100.0,
                        self.config.pattern_threshold * 100.0
                    )))
                }
            }
            Err(e) => Ok(HealthStatus::Unhealthy(format!("Engine test failed: {}", e))),
        }
    }
    
    async fn shutdown(&mut self) -> Result<(), BlockError> {
        debug!("Shutting down IntelligenceEngine block");
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
    async fn test_intelligence_wrapper() {
        let mut engine = IntelligenceEngineBlock::new(IntelligenceConfig::default());
        let mut context = PipelineContext::new(Uuid::new_v4(), DeploymentMode::Hybrid);
        
        // Initialize the engine
        engine.initialize(BlockConfig::default()).await.unwrap();
        
        // Set domain in context
        context.metadata.insert("domain".to_string(), "Technical".to_string());
        
        // Test with text input
        let input = BlockInput::Text("Analyze cross-domain patterns in data".to_string());
        let result = engine.process(input, &mut context).await;
        
        assert!(result.is_ok());
        
        // Verify context was updated
        assert!(context.metadata.contains_key("intelligence_score"));
        assert!(context.metadata.contains_key("domain_analysis"));
        assert!(context.metadata.contains_key("intelligence_time_ms"));
    }
    
    #[tokio::test]
    async fn test_intelligence_latency() {
        let mut engine = IntelligenceEngineBlock::new(IntelligenceConfig::default());
        let mut context = PipelineContext::new(Uuid::new_v4(), DeploymentMode::Hybrid);
        
        engine.initialize(BlockConfig::default()).await.unwrap();
        
        let input = BlockInput::Text("Quick intelligence test".to_string());
        let start = Instant::now();
        let _ = engine.process(input, &mut context).await;
        let elapsed = start.elapsed();
        
        // Should complete in under 15ms (allowing margin over 12ms target)
        assert!(elapsed.as_millis() < 15, "IntelligenceEngine took {}ms", elapsed.as_millis());
    }
}