//! Fusion Engine Block - Wraps the existing FusionEngine implementation
//! 
//! Provides intelligent result fusion with 6-factor scoring matrix achieving <5ms latency.

use crate::blocks::converters::{fusion_results_to_output, search_results_to_output};
use crate::core::{
    errors::{BlockError, BlockResult},
    traits::{
        BlockCategory, BlockConfig, BlockInput, BlockMetadata, BlockOutput,
        DeploymentMode, HealthStatus, PipelineBlock, PipelineContext,
    },
};
use async_trait::async_trait;
use memory_nexus::pipeline::fusion::{
    FusionEngine, FusionConfig as InnerConfig, ScoringMatrix, FusedResult,
};
use serde_json::Value;
use std::sync::Arc;
use std::time::Instant;
use tracing::{debug, info, instrument};
use uuid::Uuid;

/// Configuration for the Fusion Engine Block
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct FusionConfig {
    /// Target number of results after fusion
    pub target_results: usize,
    /// Minimum quality threshold for results
    pub min_quality_threshold: f32,
    /// Enable MinHash deduplication
    pub enable_deduplication: bool,
    /// Deduplication similarity threshold
    pub dedup_threshold: f32,
    /// Enable cross-validation between sources
    pub enable_cross_validation: bool,
    /// Target latency in milliseconds
    pub target_latency_ms: u64,
    /// Scoring matrix weights
    pub scoring_matrix: ScoringMatrixConfig,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ScoringMatrixConfig {
    pub relevance: f32,
    pub freshness: f32,
    pub diversity: f32,
    pub authority: f32,
    pub coherence: f32,
    pub confidence: f32,
}

impl Default for ScoringMatrixConfig {
    fn default() -> Self {
        Self {
            relevance: 0.35,
            freshness: 0.15,
            diversity: 0.15,
            authority: 0.15,
            coherence: 0.10,
            confidence: 0.10,
        }
    }
}

impl Default for FusionConfig {
    fn default() -> Self {
        Self {
            target_results: 8,
            min_quality_threshold: 0.5,
            enable_deduplication: true,
            dedup_threshold: 0.7,
            enable_cross_validation: true,
            target_latency_ms: 5, // 5ms target
            scoring_matrix: ScoringMatrixConfig::default(),
        }
    }
}

/// Fusion Engine Block that wraps the existing FusionEngine
pub struct ResilientFusionBlock {
    /// The actual fusion engine implementation
    inner: Arc<FusionEngine>,
    /// Block metadata
    metadata: BlockMetadata,
    /// Configuration
    config: FusionConfig,
}

impl ResilientFusionBlock {
    /// Create a new fusion block
    pub fn new(config: FusionConfig) -> Self {
        // Convert our config to the inner config
        let scoring_matrix = ScoringMatrix {
            relevance: config.scoring_matrix.relevance,
            freshness: config.scoring_matrix.freshness,
            diversity: config.scoring_matrix.diversity,
            authority: config.scoring_matrix.authority,
            coherence: config.scoring_matrix.coherence,
            confidence: config.scoring_matrix.confidence,
        };
        
        let inner_config = InnerConfig {
            target_results: config.target_results,
            min_quality: config.min_quality_threshold,
            enable_deduplication: config.enable_deduplication,
            dedup_threshold: config.dedup_threshold,
            enable_cross_validation: config.enable_cross_validation,
            scoring_matrix,
        };
        
        Self {
            inner: Arc::new(FusionEngine::new(inner_config)),
            metadata: BlockMetadata {
                id: Uuid::new_v4(),
                name: "ResilientFusion".to_string(),
                version: "1.0.0".to_string(),
                category: BlockCategory::Fusion,
                deployment_mode: DeploymentMode::Hybrid,
                hot_swappable: true,
                c_abi_compatible: false,
            },
            config,
        }
    }
    
    /// Create with custom fusion engine instance
    pub fn with_engine(engine: FusionEngine, config: FusionConfig) -> Self {
        Self {
            inner: Arc::new(engine),
            metadata: BlockMetadata {
                id: Uuid::new_v4(),
                name: "ResilientFusion".to_string(),
                version: "1.0.0".to_string(),
                category: BlockCategory::Fusion,
                deployment_mode: DeploymentMode::Hybrid,
                hot_swappable: true,
                c_abi_compatible: false,
            },
            config,
        }
    }
}

impl Default for ResilientFusionBlock {
    fn default() -> Self {
        Self::new(FusionConfig::default())
    }
}

#[async_trait]
impl PipelineBlock for ResilientFusionBlock {
    fn metadata(&self) -> &BlockMetadata {
        &self.metadata
    }
    
    async fn initialize(&mut self, _config: BlockConfig) -> Result<(), BlockError> {
        debug!("Initializing FusionEngine block");
        // Initialize the fusion engine
        self.inner.initialize().await
            .map_err(|e| BlockError::Initialization(format!("Failed to initialize fusion engine: {}", e)))?;
        info!("FusionEngine initialized with 6-factor scoring matrix");
        Ok(())
    }
    
    #[instrument(skip(self, input, context))]
    async fn process(
        &self,
        input: BlockInput,
        context: &mut PipelineContext,
    ) -> BlockResult<BlockOutput> {
        let start = Instant::now();
        
        // Extract search results from input
        let results = match &input {
            BlockInput::Structured(value) => {
                // Extract results from structured input
                value.get("results")
                    .and_then(|v| v.as_array())
                    .ok_or_else(|| BlockError::InvalidInput("No results in input".into()))?
            }
            BlockInput::Batch(inputs) => {
                // Collect results from batch
                return Err(BlockError::InvalidInput("Batch fusion not yet implemented".into()));
            }
            _ => {
                return Err(BlockError::InvalidInput("Expected structured input with results".into()));
            }
        };
        
        // Convert JSON results to SearchResults for fusion
        let search_results = results.iter()
            .filter_map(|v| serde_json::from_value(v.clone()).ok())
            .collect::<Vec<memory_nexus::pipeline::SearchResult>>();
        
        // Get embedding if available
        let embedding = match input {
            BlockInput::Structured(ref value) => {
                value.get("embedding")
                    .and_then(|v| v.as_array())
                    .and_then(|arr| {
                        arr.iter()
                            .map(|v| v.as_f64().map(|f| f as f32))
                            .collect::<Option<Vec<f32>>>()
                    })
            }
            _ => None,
        };
        
        // Fuse results using the real fusion engine
        let fused_results: Vec<FusedResult> = self.inner
            .fuse_search_results(search_results, embedding.as_deref())
            .await
            .map_err(|e| BlockError::Processing(format!("Fusion failed: {}", e)))?;
        
        // Check if we met our latency target
        let elapsed = start.elapsed();
        if elapsed.as_millis() > self.config.target_latency_ms {
            debug!(
                "Fusion exceeded target latency: {}ms > {}ms",
                elapsed.as_millis(),
                self.config.target_latency_ms
            );
        }
        
        // Store fusion metadata
        context.metadata.insert(
            "fused_count".to_string(),
            fused_results.len().to_string(),
        );
        context.metadata.insert(
            "fusion_time_ms".to_string(),
            elapsed.as_millis().to_string(),
        );
        
        if self.config.enable_deduplication {
            let stats = self.inner.get_deduplication_stats();
            context.metadata.insert(
                "duplicates_removed".to_string(),
                stats.duplicates_found.to_string(),
            );
        }
        
        if self.config.enable_cross_validation {
            let stats = self.inner.get_cross_validation_stats();
            context.metadata.insert(
                "cross_validated".to_string(),
                stats.cross_validated_count.to_string(),
            );
        }
        
        debug!(
            "Fusion completed: {} results fused to {} in {}ms",
            results.len(),
            fused_results.len(),
            elapsed.as_millis()
        );
        
        // Convert to BlockOutput
        Ok(fusion_results_to_output(fused_results))
    }
    
    fn validate_input(&self, input: &BlockInput) -> Result<(), BlockError> {
        // Ensure we have structured input with results
        match input {
            BlockInput::Structured(value) => {
                if !value.get("results").is_some() {
                    return Err(BlockError::InvalidInput("No results field in input".into()));
                }
                Ok(())
            }
            _ => Err(BlockError::InvalidInput("Expected structured input with results".into()))
        }
    }
    
    async fn health_check(&self) -> Result<HealthStatus, BlockError> {
        // Test fusion with dummy results
        let test_results = vec![
            memory_nexus::pipeline::SearchResult {
                id: Uuid::new_v4(),
                content: "Test result 1".to_string(),
                score: 0.9,
                metadata: Default::default(),
            },
            memory_nexus::pipeline::SearchResult {
                id: Uuid::new_v4(),
                content: "Test result 2".to_string(),
                score: 0.8,
                metadata: Default::default(),
            },
        ];
        
        match self.inner.fuse_search_results(test_results, None).await {
            Ok(fused) if !fused.is_empty() => Ok(HealthStatus::Healthy),
            Ok(_) => Ok(HealthStatus::Degraded("No results from test fusion".to_string())),
            Err(e) => Ok(HealthStatus::Unhealthy(format!("Fusion test failed: {}", e))),
        }
    }
    
    async fn shutdown(&mut self) -> Result<(), BlockError> {
        debug!("Shutting down FusionEngine block");
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
pub use memory_nexus::pipeline::fusion::{
    FusedResult, FusedOutput, FusionItem, FusionMetrics, EngineType,
};

#[cfg(test)]
mod tests {
    use super::*;
    use memory_nexus::pipeline::SearchResult;
    
    #[tokio::test]
    async fn test_fusion_wrapper() {
        let mut fusion = ResilientFusionBlock::new(FusionConfig::default());
        let mut context = PipelineContext::new(Uuid::new_v4(), DeploymentMode::Hybrid);
        
        // Initialize the fusion engine
        fusion.initialize(BlockConfig::default()).await.unwrap();
        
        // Create test results
        let results = vec![
            SearchResult {
                id: Uuid::new_v4(),
                content: "Result 1".to_string(),
                score: 0.9,
                metadata: Default::default(),
            },
            SearchResult {
                id: Uuid::new_v4(),
                content: "Result 2".to_string(),
                score: 0.8,
                metadata: Default::default(),
            },
        ];
        
        let input = BlockInput::Structured(serde_json::json!({
            "results": results
        }));
        
        let result = fusion.process(input, &mut context).await;
        
        assert!(result.is_ok());
        
        // Verify context was updated
        assert!(context.metadata.contains_key("fused_count"));
        assert!(context.metadata.contains_key("fusion_time_ms"));
    }
    
    #[tokio::test]
    async fn test_fusion_latency() {
        let mut fusion = ResilientFusionBlock::new(FusionConfig::default());
        let mut context = PipelineContext::new(Uuid::new_v4(), DeploymentMode::Hybrid);
        
        fusion.initialize(BlockConfig::default()).await.unwrap();
        
        let results = vec![
            SearchResult {
                id: Uuid::new_v4(),
                content: "Test".to_string(),
                score: 0.9,
                metadata: Default::default(),
            },
        ];
        
        let input = BlockInput::Structured(serde_json::json!({
            "results": results
        }));
        
        let start = Instant::now();
        let _ = fusion.process(input, &mut context).await;
        let elapsed = start.elapsed();
        
        // Should complete in under 10ms (allowing margin over 5ms target)
        assert!(elapsed.as_millis() < 10, "Fusion took {}ms", elapsed.as_millis());
    }
}