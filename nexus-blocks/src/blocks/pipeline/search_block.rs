//! Search Orchestrator Block - Wraps the existing SearchOrchestrator implementation
//! 
//! Orchestrates parallel search across 4 engines achieving <25ms total latency.

use crate::blocks::converters::{extract_query, search_results_to_output};
use crate::core::{
    errors::{BlockError, BlockResult},
    traits::{
        BlockCategory, BlockConfig, BlockInput, BlockMetadata, BlockOutput,
        DeploymentMode, HealthStatus, PipelineBlock, PipelineContext,
    },
};
use async_trait::async_trait;
use memory_nexus::pipeline::search_orchestrator::{
    SearchOrchestrator, SearchRequest, SearchResponse, SearchConfig as InnerConfig,
};
use std::sync::Arc;
use std::time::Instant;
use tracing::{debug, info, instrument};
use uuid::Uuid;

/// Configuration for the Search Orchestrator Block
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SearchConfig {
    /// Enable parallel search across engines
    pub parallel_search: bool,
    /// Maximum search timeout in milliseconds
    pub timeout_ms: u64,
    /// Minimum engines required for valid result
    pub min_engines_required: usize,
    /// Enable partial results if some engines fail
    pub allow_partial_results: bool,
    /// Maximum results per engine
    pub max_results_per_engine: usize,
}

impl Default for SearchConfig {
    fn default() -> Self {
        Self {
            parallel_search: true,
            timeout_ms: 25, // 25ms target
            min_engines_required: 2,
            allow_partial_results: true,
            max_results_per_engine: 50,
        }
    }
}

/// Search Orchestrator Block that wraps the existing SearchOrchestrator
pub struct SearchOrchestratorBlock {
    /// The actual search orchestrator implementation
    inner: Arc<SearchOrchestrator>,
    /// Block metadata
    metadata: BlockMetadata,
    /// Configuration
    config: SearchConfig,
}

impl SearchOrchestratorBlock {
    /// Create a new search orchestrator block
    pub fn new() -> Self {
        Self::with_config(SearchConfig::default())
    }
    
    /// Create with custom configuration
    pub fn with_config(config: SearchConfig) -> Self {
        // Convert our config to the inner config
        let inner_config = InnerConfig {
            parallel_execution: config.parallel_search,
            timeout_ms: config.timeout_ms,
            min_engines: config.min_engines_required,
            allow_partial: config.allow_partial_results,
            max_results: config.max_results_per_engine,
        };
        
        Self {
            inner: Arc::new(SearchOrchestrator::new(inner_config)),
            metadata: BlockMetadata {
                id: Uuid::new_v4(),
                name: "SearchOrchestrator".to_string(),
                version: "1.0.0".to_string(),
                category: BlockCategory::Search,
                deployment_mode: DeploymentMode::Hybrid,
                hot_swappable: true,
                c_abi_compatible: false,
            },
            config,
        }
    }
    
    /// Create with custom orchestrator instance
    pub fn with_orchestrator(orchestrator: SearchOrchestrator, config: SearchConfig) -> Self {
        Self {
            inner: Arc::new(orchestrator),
            metadata: BlockMetadata {
                id: Uuid::new_v4(),
                name: "SearchOrchestrator".to_string(),
                version: "1.0.0".to_string(),
                category: BlockCategory::Search,
                deployment_mode: DeploymentMode::Hybrid,
                hot_swappable: true,
                c_abi_compatible: false,
            },
            config,
        }
    }
}

impl Default for SearchOrchestratorBlock {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl PipelineBlock for SearchOrchestratorBlock {
    fn metadata(&self) -> &BlockMetadata {
        &self.metadata
    }
    
    async fn initialize(&mut self, _config: BlockConfig) -> Result<(), BlockError> {
        debug!("Initializing SearchOrchestrator block");
        // Initialize the orchestrator
        self.inner.initialize().await
            .map_err(|e| BlockError::Initialization(format!("Failed to initialize orchestrator: {}", e)))?;
        info!("SearchOrchestrator initialized with {} engines", 4);
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
        
        // Check for routing path in context
        let routing_path = context.metadata.get("routing_path")
            .map(|s| s.as_str())
            .unwrap_or("FullPipeline");
        
        // Create search request
        let request = SearchRequest {
            query: query.clone(),
            user_id: context.user_id.clone(),
            routing_path: routing_path.to_string(),
            max_results: self.config.max_results_per_engine,
        };
        
        // Execute search using the real orchestrator
        let response: SearchResponse = self.inner
            .search(request)
            .await
            .map_err(|e| BlockError::Processing(format!("Search failed: {}", e)))?;
        
        // Check if we met our latency target
        let elapsed = start.elapsed();
        if elapsed.as_millis() > self.config.timeout_ms {
            debug!(
                "Search exceeded target latency: {}ms > {}ms",
                elapsed.as_millis(),
                self.config.timeout_ms
            );
        }
        
        // Store search metadata
        context.metadata.insert(
            "engines_used".to_string(),
            response.engines_used.len().to_string(),
        );
        context.metadata.insert(
            "total_results".to_string(),
            response.results.len().to_string(),
        );
        context.metadata.insert(
            "search_time_ms".to_string(),
            elapsed.as_millis().to_string(),
        );
        
        // Store engine-specific metrics
        for (engine, count) in &response.engine_result_counts {
            context.metadata.insert(
                format!("{}_results", engine),
                count.to_string(),
            );
        }
        
        debug!(
            "Search completed with {} results from {} engines in {}ms",
            response.results.len(),
            response.engines_used.len(),
            elapsed.as_millis()
        );
        
        // Convert to BlockOutput
        Ok(search_results_to_output(response.results))
    }
    
    fn validate_input(&self, input: &BlockInput) -> Result<(), BlockError> {
        // Ensure we can extract a query
        extract_query(input)?;
        Ok(())
    }
    
    async fn health_check(&self) -> Result<HealthStatus, BlockError> {
        // Check health of all engines
        let health = self.inner.check_health().await
            .map_err(|e| BlockError::HealthCheck(format!("Health check failed: {}", e)))?;
        
        if health.healthy_engines >= self.config.min_engines_required {
            Ok(HealthStatus::Healthy)
        } else if health.healthy_engines > 0 {
            Ok(HealthStatus::Degraded(format!(
                "Only {} of {} engines healthy",
                health.healthy_engines, 4
            )))
        } else {
            Ok(HealthStatus::Unhealthy("No healthy engines".to_string()))
        }
    }
    
    async fn shutdown(&mut self) -> Result<(), BlockError> {
        debug!("Shutting down SearchOrchestrator block");
        // Shutdown the orchestrator
        self.inner.shutdown().await
            .map_err(|e| BlockError::Shutdown(format!("Failed to shutdown orchestrator: {}", e)))?;
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
pub use memory_nexus::pipeline::search_orchestrator::{
    SearchRequest, SearchResponse, EngineResult, SearchEngine,
};

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_search_wrapper() {
        let mut orchestrator = SearchOrchestratorBlock::new();
        let mut context = PipelineContext::new(Uuid::new_v4(), DeploymentMode::Hybrid);
        
        // Initialize the orchestrator
        orchestrator.initialize(BlockConfig::default()).await.unwrap();
        
        // Test with text input
        let input = BlockInput::Text("search for machine learning algorithms".to_string());
        let result = orchestrator.process(input, &mut context).await;
        
        assert!(result.is_ok());
        
        // Verify context was updated
        assert!(context.metadata.contains_key("engines_used"));
        assert!(context.metadata.contains_key("total_results"));
        assert!(context.metadata.contains_key("search_time_ms"));
    }
    
    #[tokio::test]
    async fn test_search_latency() {
        let mut orchestrator = SearchOrchestratorBlock::new();
        let mut context = PipelineContext::new(Uuid::new_v4(), DeploymentMode::Hybrid);
        
        orchestrator.initialize(BlockConfig::default()).await.unwrap();
        
        let input = BlockInput::Text("quick search test".to_string());
        let start = Instant::now();
        let _ = orchestrator.process(input, &mut context).await;
        let elapsed = start.elapsed();
        
        // Should complete in under 30ms (allowing margin over 25ms target)
        assert!(elapsed.as_millis() < 30, "Search took {}ms", elapsed.as_millis());
    }
}