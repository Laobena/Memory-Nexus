//! Preprocessor Block - Wraps the existing ParallelPreprocessor implementation
//! 
//! Provides parallel text processing with semantic chunking, MinHash deduplication,
//! and entity extraction achieving <10ms latency.

use crate::blocks::converters::{extract_query, chunks_to_output};
use crate::core::{
    errors::{BlockError, BlockResult},
    traits::{
        BlockCategory, BlockConfig, BlockInput, BlockMetadata, BlockOutput,
        DeploymentMode, HealthStatus, PipelineBlock, PipelineContext,
    },
};
use async_trait::async_trait;
use memory_nexus::pipeline::preprocessor_enhanced::{
    ParallelPreprocessor, ChunkingStrategy, ProcessedChunk, PreprocessorConfig as InnerConfig,
};
use std::sync::Arc;
use std::time::Instant;
use tracing::{debug, info, instrument};
use uuid::Uuid;

/// Configuration for the Preprocessor Block
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct PreprocessorConfig {
    /// Chunking strategy to use
    pub chunking_strategy: ChunkingStrategyConfig,
    /// Enable MinHash deduplication
    pub enable_deduplication: bool,
    /// Enable entity extraction
    pub enable_entity_extraction: bool,
    /// Target chunk size in tokens
    pub chunk_size: usize,
    /// Overlap size for sliding window
    pub overlap_size: usize,
    /// Number of parallel workers
    pub num_workers: usize,
    /// Target latency in milliseconds
    pub target_latency_ms: u64,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum ChunkingStrategyConfig {
    Semantic,
    Fixed,
    Sliding,
    Sentence,
    Paragraph,
}

impl Default for PreprocessorConfig {
    fn default() -> Self {
        Self {
            chunking_strategy: ChunkingStrategyConfig::Semantic,
            enable_deduplication: true,
            enable_entity_extraction: true,
            chunk_size: 400,
            overlap_size: 20,
            num_workers: 4,
            target_latency_ms: 10, // 10ms target
        }
    }
}

/// Preprocessor Block that wraps the existing ParallelPreprocessor
pub struct PreprocessorBlock {
    /// The actual preprocessor implementation
    inner: Arc<ParallelPreprocessor>,
    /// Block metadata
    metadata: BlockMetadata,
    /// Configuration
    config: PreprocessorConfig,
}

impl PreprocessorBlock {
    /// Create a new preprocessor block
    pub fn new() -> Self {
        Self::with_config(PreprocessorConfig::default())
    }
    
    /// Create with custom configuration
    pub fn with_config(config: PreprocessorConfig) -> Self {
        // Convert our config to the inner config
        let inner_config = InnerConfig {
            chunk_size: config.chunk_size,
            overlap_size: config.overlap_size,
            enable_deduplication: config.enable_deduplication,
            enable_entity_extraction: config.enable_entity_extraction,
            num_workers: config.num_workers,
            max_chunk_count: 100,
            min_chunk_size: 50,
        };
        
        let strategy = match config.chunking_strategy {
            ChunkingStrategyConfig::Semantic => ChunkingStrategy::Semantic,
            ChunkingStrategyConfig::Fixed => ChunkingStrategy::Fixed(config.chunk_size),
            ChunkingStrategyConfig::Sliding => ChunkingStrategy::Sliding {
                window_size: config.chunk_size,
                step_size: config.chunk_size - config.overlap_size,
            },
            ChunkingStrategyConfig::Sentence => ChunkingStrategy::Sentence,
            ChunkingStrategyConfig::Paragraph => ChunkingStrategy::Paragraph,
        };
        
        Self {
            inner: Arc::new(ParallelPreprocessor::new(inner_config, strategy)),
            metadata: BlockMetadata {
                id: Uuid::new_v4(),
                name: "ParallelPreprocessor".to_string(),
                version: "1.0.0".to_string(),
                category: BlockCategory::Preprocessor,
                deployment_mode: DeploymentMode::Hybrid,
                hot_swappable: true,
                c_abi_compatible: false,
            },
            config,
        }
    }
    
    /// Create with custom preprocessor instance
    pub fn with_preprocessor(preprocessor: ParallelPreprocessor, config: PreprocessorConfig) -> Self {
        Self {
            inner: Arc::new(preprocessor),
            metadata: BlockMetadata {
                id: Uuid::new_v4(),
                name: "ParallelPreprocessor".to_string(),
                version: "1.0.0".to_string(),
                category: BlockCategory::Preprocessor,
                deployment_mode: DeploymentMode::Hybrid,
                hot_swappable: true,
                c_abi_compatible: false,
            },
            config,
        }
    }
}

impl Default for PreprocessorBlock {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl PipelineBlock for PreprocessorBlock {
    fn metadata(&self) -> &BlockMetadata {
        &self.metadata
    }
    
    async fn initialize(&mut self, _config: BlockConfig) -> Result<(), BlockError> {
        debug!("Initializing ParallelPreprocessor block");
        // Initialize the preprocessor
        self.inner.initialize().await
            .map_err(|e| BlockError::Initialization(format!("Failed to initialize preprocessor: {}", e)))?;
        info!("ParallelPreprocessor initialized with {} workers", self.config.num_workers);
        Ok(())
    }
    
    #[instrument(skip(self, input, context))]
    async fn process(
        &self,
        input: BlockInput,
        context: &mut PipelineContext,
    ) -> BlockResult<BlockOutput> {
        let start = Instant::now();
        
        // Extract text from input
        let text = extract_query(&input)?;
        
        // Check if we have a user_id in context
        let user_id = context.user_id.clone();
        
        // Process text using the real preprocessor
        let chunks: Vec<ProcessedChunk> = self.inner
            .process(&text, user_id.as_deref())
            .await
            .map_err(|e| BlockError::Processing(format!("Preprocessing failed: {}", e)))?;
        
        // Check if we met our latency target
        let elapsed = start.elapsed();
        if elapsed.as_millis() > self.config.target_latency_ms {
            debug!(
                "Preprocessor exceeded target latency: {}ms > {}ms",
                elapsed.as_millis(),
                self.config.target_latency_ms
            );
        }
        
        // Store preprocessing metadata
        context.metadata.insert(
            "chunk_count".to_string(),
            chunks.len().to_string(),
        );
        context.metadata.insert(
            "preprocessing_time_ms".to_string(),
            elapsed.as_millis().to_string(),
        );
        
        if self.config.enable_deduplication {
            let dedup_count = self.inner.get_deduplication_stats().duplicates_found;
            context.metadata.insert(
                "duplicates_removed".to_string(),
                dedup_count.to_string(),
            );
        }
        
        if self.config.enable_entity_extraction {
            let entity_count: usize = chunks.iter()
                .filter_map(|c| c.entities.as_ref())
                .map(|e| e.len())
                .sum();
            context.metadata.insert(
                "entities_extracted".to_string(),
                entity_count.to_string(),
            );
        }
        
        debug!(
            "Preprocessed text into {} chunks in {}ms",
            chunks.len(),
            elapsed.as_millis()
        );
        
        // Convert to BlockOutput
        Ok(chunks_to_output(chunks))
    }
    
    fn validate_input(&self, input: &BlockInput) -> Result<(), BlockError> {
        // Ensure we can extract text
        extract_query(input)?;
        Ok(())
    }
    
    async fn health_check(&self) -> Result<HealthStatus, BlockError> {
        // Test preprocessor with a simple text
        let test_text = "This is a health check test. It has multiple sentences.";
        let result = self.inner.process(test_text, None).await;
        
        match result {
            Ok(chunks) if !chunks.is_empty() => Ok(HealthStatus::Healthy),
            Ok(_) => Ok(HealthStatus::Degraded("No chunks produced from test text".to_string())),
            Err(e) => Ok(HealthStatus::Unhealthy(format!("Preprocessing failed: {}", e))),
        }
    }
    
    async fn shutdown(&mut self) -> Result<(), BlockError> {
        debug!("Shutting down ParallelPreprocessor block");
        // Shutdown the preprocessor
        self.inner.shutdown().await
            .map_err(|e| BlockError::Shutdown(format!("Failed to shutdown preprocessor: {}", e)))?;
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
pub use memory_nexus::pipeline::preprocessor_enhanced::{
    ChunkingStrategy, ProcessedChunk, Entity, ChunkMetadata,
};

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_preprocessor_wrapper() {
        let mut preprocessor = PreprocessorBlock::new();
        let mut context = PipelineContext::new(Uuid::new_v4(), DeploymentMode::Hybrid);
        
        // Initialize the preprocessor
        preprocessor.initialize(BlockConfig::default()).await.unwrap();
        
        // Test with text input
        let input = BlockInput::Text(
            "Machine learning is a subset of artificial intelligence. \
             It enables systems to learn from data.".to_string()
        );
        let result = preprocessor.process(input, &mut context).await;
        
        assert!(result.is_ok());
        
        // Verify chunks were created
        if let Ok(BlockOutput::Structured(value)) = result {
            assert!(value.get("chunks").is_some());
            assert!(value.get("count").is_some());
        }
        
        // Verify context was updated
        assert!(context.metadata.contains_key("chunk_count"));
        assert!(context.metadata.contains_key("preprocessing_time_ms"));
    }
    
    #[tokio::test]
    async fn test_preprocessor_latency() {
        let mut preprocessor = PreprocessorBlock::new();
        let mut context = PipelineContext::new(Uuid::new_v4(), DeploymentMode::Hybrid);
        
        preprocessor.initialize(BlockConfig::default()).await.unwrap();
        
        let input = BlockInput::Text("Short text for testing latency.".to_string());
        let start = Instant::now();
        let _ = preprocessor.process(input, &mut context).await;
        let elapsed = start.elapsed();
        
        // Should complete in under 20ms (allowing margin over 10ms target)
        assert!(elapsed.as_millis() < 20, "Preprocessor took {}ms", elapsed.as_millis());
    }
}