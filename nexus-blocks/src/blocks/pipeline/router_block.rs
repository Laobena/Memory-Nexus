//! Intelligent Router Block - Wraps the existing IntelligentRouter implementation
//! 
//! Provides <0.2ms routing decisions with complexity analysis and cache prediction.

use crate::blocks::converters::{extract_query, routing_to_output, update_context_from_analysis};
use crate::core::{
    errors::{BlockError, BlockResult},
    traits::{
        BlockCategory, BlockConfig, BlockInput, BlockMetadata, BlockOutput,
        DeploymentMode, HealthStatus, PipelineBlock, PipelineContext,
    },
};
use async_trait::async_trait;
use memory_nexus::pipeline::intelligent_router::{IntelligentRouter, QueryAnalysis};
use std::sync::Arc;
use std::time::Instant;
use tracing::{debug, instrument};
use uuid::Uuid;

/// Configuration for the Intelligent Router Block
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct RouterConfig {
    /// Enable cache prediction
    pub enable_cache_prediction: bool,
    /// Enable domain detection
    pub enable_domain_detection: bool,
    /// Confidence threshold for escalation
    pub escalation_threshold: f32,
    /// Target latency in microseconds
    pub target_latency_us: u64,
}

impl Default for RouterConfig {
    fn default() -> Self {
        Self {
            enable_cache_prediction: true,
            enable_domain_detection: true,
            escalation_threshold: 0.85,
            target_latency_us: 200, // 0.2ms target
        }
    }
}

/// Intelligent Router Block that wraps the existing IntelligentRouter
pub struct IntelligentRouterBlock {
    /// The actual router implementation
    inner: Arc<IntelligentRouter>,
    /// Block metadata
    metadata: BlockMetadata,
    /// Configuration
    config: RouterConfig,
}

impl IntelligentRouterBlock {
    /// Create a new router block
    pub fn new(config: RouterConfig) -> Self {
        Self {
            inner: Arc::new(IntelligentRouter::new()),
            metadata: BlockMetadata {
                id: Uuid::new_v4(),
                name: "IntelligentRouter".to_string(),
                version: "1.0.0".to_string(),
                category: BlockCategory::Router,
                deployment_mode: DeploymentMode::Hybrid,
                hot_swappable: true,
                c_abi_compatible: false,
            },
            config,
        }
    }
    
    /// Create with custom router instance
    pub fn with_router(router: IntelligentRouter, config: RouterConfig) -> Self {
        Self {
            inner: Arc::new(router),
            metadata: BlockMetadata {
                id: Uuid::new_v4(),
                name: "IntelligentRouter".to_string(),
                version: "1.0.0".to_string(),
                category: BlockCategory::Router,
                deployment_mode: DeploymentMode::Hybrid,
                hot_swappable: true,
                c_abi_compatible: false,
            },
            config,
        }
    }
}

#[async_trait]
impl PipelineBlock for IntelligentRouterBlock {
    fn metadata(&self) -> &BlockMetadata {
        &self.metadata
    }
    
    async fn initialize(&mut self, _config: BlockConfig) -> Result<(), BlockError> {
        debug!("Initializing IntelligentRouter block");
        // The IntelligentRouter is already initialized
        Ok(())
    }
    
    #[instrument(skip(self, input, context))]
    async fn process(
        &self,
        input: BlockInput,
        context: &mut PipelineContext,
    ) -> BlockResult<BlockOutput> {
        let start = Instant::now();
        
        // Extract query from input
        let query = extract_query(&input)?;
        
        // Analyze query using the real router
        let analysis: QueryAnalysis = self.inner.analyze_query(&query)?;
        
        // Check if we met our latency target
        let elapsed = start.elapsed();
        if elapsed.as_micros() > self.config.target_latency_us as u128 {
            debug!(
                "Router exceeded target latency: {}μs > {}μs",
                elapsed.as_micros(),
                self.config.target_latency_us
            );
        }
        
        // Update context based on analysis
        update_context_from_analysis(
            context,
            &analysis.routing_path,
            analysis.confidence,
        );
        
        // Store routing metadata
        context.metadata.insert(
            "routing_path".to_string(),
            format!("{:?}", analysis.routing_path),
        );
        context.metadata.insert(
            "complexity".to_string(),
            format!("{:?}", analysis.complexity),
        );
        context.metadata.insert(
            "cache_probability".to_string(),
            analysis.cache_probability.to_string(),
        );
        context.metadata.insert(
            "domain".to_string(),
            format!("{:?}", analysis.domain),
        );
        context.metadata.insert(
            "analysis_time_us".to_string(),
            analysis.analysis_time_us.to_string(),
        );
        
        // Convert to BlockOutput
        let output = routing_to_output(
            analysis.routing_path,
            analysis.complexity,
            analysis.cache_probability,
            analysis.confidence,
            analysis.domain,
        );
        
        debug!(
            "Router decision: {:?} (confidence: {:.2}, cache_prob: {:.2})",
            analysis.routing_path,
            analysis.confidence,
            analysis.cache_probability
        );
        
        Ok(output)
    }
    
    fn validate_input(&self, input: &BlockInput) -> Result<(), BlockError> {
        // Ensure we can extract a query
        extract_query(input)?;
        Ok(())
    }
    
    async fn health_check(&self) -> Result<HealthStatus, BlockError> {
        // Test router with a simple query
        let test_query = "test health check".to_string();
        let analysis = self.inner.analyze_query(&test_query)?;
        
        // Check if analysis completed successfully
        if analysis.confidence > 0.0 {
            Ok(HealthStatus::Healthy)
        } else {
            Ok(HealthStatus::Degraded("Low confidence in test analysis".to_string()))
        }
    }
    
    async fn shutdown(&mut self) -> Result<(), BlockError> {
        debug!("Shutting down IntelligentRouter block");
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

// Re-export types from the wrapped implementation
pub use memory_nexus::pipeline::intelligent_router::{
    ComplexityLevel, QueryAnalysis, QueryDomain, QueryFeatures, RoutingPath,
};

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_router_wrapper() {
        let router = IntelligentRouterBlock::new(RouterConfig::default());
        let mut context = PipelineContext::new(Uuid::new_v4(), DeploymentMode::Hybrid);
        
        // Test with text input
        let input = BlockInput::Text("What is machine learning?".to_string());
        let result = router.process(input, &mut context).await;
        
        assert!(result.is_ok());
        
        // Verify context was updated
        assert!(context.metadata.contains_key("routing_path"));
        assert!(context.metadata.contains_key("complexity"));
    }
    
    #[tokio::test]
    async fn test_router_latency() {
        let router = IntelligentRouterBlock::new(RouterConfig::default());
        let mut context = PipelineContext::new(Uuid::new_v4(), DeploymentMode::Hybrid);
        
        let input = BlockInput::Text("Simple query".to_string());
        let start = Instant::now();
        let _ = router.process(input, &mut context).await;
        let elapsed = start.elapsed();
        
        // Should complete in under 1ms (allowing some margin)
        assert!(elapsed.as_millis() < 1, "Router took {}ms", elapsed.as_millis());
    }
}