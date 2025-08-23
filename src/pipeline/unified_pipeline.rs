//! Unified Adaptive Pipeline - Orchestrates all components for dual-mode operation
//! 
//! Implements the 4-path routing system:
//! - CacheOnly: 2ms target (70% of queries)
//! - SmartRouting: 15ms target (25% of queries)  
//! - FullPipeline: 40ms target (4% of queries)
//! - MaximumIntelligence: 45ms target (1% of queries)

use crate::core::types::{ConstVector, EMBEDDING_DIM};
use crate::database::UnifiedDatabasePool;
use crate::monitoring::MetricsCollector;
use crate::optimizations::memory_pool::{PoolHandle, global_pool};
use anyhow::{Result, Context};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::Instant;
use tracing::{debug, info, warn};
use uuid::Uuid;

use super::{
    IntelligentRouter, QueryAnalysis, RoutingPath, ComplexityLevel,
    ParallelPreprocessor, PreprocessedData, ChunkingStrategy,
    SearchOrchestrator, SearchResult, SearchSource,
    FusionEngine, StorageEngine,
};

/// Unified adaptive pipeline orchestrating all components
pub struct UnifiedPipeline {
    router: Arc<IntelligentRouter>,
    preprocessor: Arc<ParallelPreprocessor>,
    storage: Arc<StorageEngine>,
    search: Arc<SearchOrchestrator>,
    fusion: Arc<FusionEngine>,
    db_pool: Arc<UnifiedDatabasePool>,
    metrics: Arc<MetricsCollector>,
    config: PipelineConfig,
}

/// Pipeline configuration
#[derive(Debug, Clone)]
pub struct PipelineConfig {
    /// Confidence threshold for escalation (default: 0.85)
    pub escalation_threshold: f32,
    /// Enable automatic escalation
    pub auto_escalate: bool,
    /// Maximum escalation attempts
    pub max_escalations: usize,
    /// Cache-only timeout in ms
    pub cache_timeout_ms: u64,
    /// Smart routing timeout in ms
    pub smart_timeout_ms: u64,
    /// Full pipeline timeout in ms
    pub full_timeout_ms: u64,
    /// Maximum intelligence timeout in ms
    pub max_intelligence_timeout_ms: u64,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            escalation_threshold: 0.85,
            auto_escalate: true,
            max_escalations: 2,
            cache_timeout_ms: 2,
            smart_timeout_ms: 15,
            full_timeout_ms: 40,
            max_intelligence_timeout_ms: 45,
        }
    }
}

impl UnifiedPipeline {
    /// Create new unified pipeline
    pub async fn new(
        db_pool: Arc<UnifiedDatabasePool>,
        metrics: Arc<MetricsCollector>,
    ) -> Result<Self> {
        info!("🚀 Initializing Unified Adaptive Pipeline");
        
        // Initialize preprocessor with embedding service
        let preprocessor = Arc::new(ParallelPreprocessor::new());
        preprocessor.initialize().await
            .context("Failed to initialize preprocessor with embedding service")?;
        
        Ok(Self {
            router: Arc::new(IntelligentRouter::new()),
            preprocessor,
            storage: Arc::new(StorageEngine::new()),
            search: Arc::new(SearchOrchestrator::new()),
            fusion: Arc::new(FusionEngine::new()),
            db_pool,
            metrics,
            config: PipelineConfig::default(),
        })
    }
    
    /// Configure the pipeline
    pub fn with_config(mut self, config: PipelineConfig) -> Self {
        self.config = config;
        self
    }

    /// Process query through adaptive pipeline with automatic routing
    pub async fn process(&self, query: String, user_id: Option<String>) -> Result<PipelineResponse> {
        let start = Instant::now();
        let query_id = Uuid::new_v4();
        
        debug!("Processing query {} with adaptive pipeline", query_id);
        
        // Step 1: Analyze query complexity (<0.2ms)
        let analysis = self.router.analyze(&query).await;
        self.metrics.record_routing_decision(&analysis.routing_path);
        
        info!("Query {} routed to {:?} path (complexity: {:?}, cache_prob: {:.2})", 
            query_id, analysis.routing_path, analysis.complexity, analysis.cache_probability);
        
        // Step 2: Execute appropriate path with escalation support
        let mut result = self.execute_path(&analysis, query_id).await?;
        let mut escalation_count = 0;
        let mut current_path = analysis.routing_path.clone();
        
        // Step 3: Handle automatic escalation if needed
        while self.config.auto_escalate 
            && escalation_count < self.config.max_escalations
            && result.confidence < self.config.escalation_threshold 
        {
            if let Some(next_path) = self.escalate_path(&current_path) {
                warn!("Escalating query {} from {:?} to {:?} (confidence: {:.2})", 
                    query_id, current_path, next_path, result.confidence);
                
                self.metrics.record_escalation(&current_path, &next_path);
                
                // Re-process with escalated path
                let escalated_analysis = QueryAnalysis {
                    routing_path: next_path.clone(),
                    ..analysis.clone()
                };
                
                result = self.execute_path(&escalated_analysis, query_id).await?;
                current_path = next_path;
                escalation_count += 1;
            } else {
                break; // No more escalation possible
            }
        }
        
        let latency = start.elapsed();
        self.metrics.record_pipeline_latency(latency);
        
        info!("Query {} completed in {:?} with confidence {:.2} (escalations: {})",
            query_id, latency, result.confidence, escalation_count);
        
        Ok(PipelineResponse {
            query_id,
            results: result.results,
            confidence: result.confidence,
            latency_ms: latency.as_millis() as u64,
            path_taken: current_path,
            escalations: escalation_count,
            metadata: serde_json::json!({
                "complexity": format!("{:?}", analysis.complexity),
                "cache_probability": analysis.cache_probability,
                "domains": analysis.domains,
                "user_id": user_id,
            }),
        })
    }
    
    /// Execute a specific routing path
    async fn execute_path(&self, analysis: &QueryAnalysis, query_id: Uuid) -> Result<ProcessingResult> {
        match &analysis.routing_path {
            RoutingPath::CacheOnly => {
                self.process_cache_only(analysis, query_id).await
            }
            RoutingPath::SmartRouting => {
                self.process_smart_routing(analysis, query_id).await
            }
            RoutingPath::FullPipeline => {
                self.process_full_pipeline(analysis, query_id).await
            }
            RoutingPath::MaximumIntelligence => {
                self.process_maximum_intelligence(analysis, query_id).await
            }
        }
    }

    /// Cache-only path (2ms target) - 70% of queries
    async fn process_cache_only(&self, analysis: &QueryAnalysis, query_id: Uuid) -> Result<ProcessingResult> {
        let start = Instant::now();
        debug!("Query {} using cache-only path", query_id);
        
        // Generate minimal embedding using memory pool
        let embedding = self.generate_minimal_embedding(&analysis.query).await?;
        
        // Search cache only
        let results = tokio::time::timeout(
            std::time::Duration::from_millis(self.config.cache_timeout_ms),
            self.search.search_cache_only(&embedding)
        ).await
            .context("Cache search timeout")?
            .context("Cache search failed")?;
        
        let confidence = self.calculate_result_confidence(&results);
        
        debug!("Cache-only completed in {:?} with {} results", start.elapsed(), results.len());
        
        Ok(ProcessingResult {
            results: self.format_search_results(results),
            confidence,
        })
    }

    /// Smart routing path (15ms target) - 25% of queries
    async fn process_smart_routing(&self, analysis: &QueryAnalysis, query_id: Uuid) -> Result<ProcessingResult> {
        let start = Instant::now();
        debug!("Query {} using smart routing path", query_id);
        
        // Basic preprocessing with semantic chunking
        let preprocessed = tokio::time::timeout(
            std::time::Duration::from_millis(5),
            self.preprocessor.process_basic(&analysis.query)
        ).await
            .context("Preprocessing timeout")?
            .context("Preprocessing failed")?;
        
        // Selective storage (only if novel)
        if analysis.cache_probability < 0.5 {
            let _ = self.storage.store_selective(&preprocessed, query_id).await;
        }
        
        // Search with 2 best engines
        let search_future = self.search.search_selected(
            &preprocessed.embeddings,
            vec![SearchSource::Cache, SearchSource::Accuracy],
        );
        
        let results = tokio::time::timeout(
            std::time::Duration::from_millis(self.config.smart_timeout_ms - 5),
            search_future
        ).await
            .context("Smart search timeout")?
            .context("Smart search failed")?;
        
        // Quick fusion
        let fused = self.fusion.fuse_quick(results);
        let confidence = self.calculate_fused_confidence(&fused);
        
        debug!("Smart routing completed in {:?} with {} results", start.elapsed(), fused.len());
        
        Ok(ProcessingResult {
            results: fused,
            confidence,
        })
    }

    /// Full pipeline path (40ms target) - 4% of queries
    async fn process_full_pipeline(&self, analysis: &QueryAnalysis, query_id: Uuid) -> Result<ProcessingResult> {
        let start = Instant::now();
        debug!("Query {} using full pipeline path", query_id);
        
        // Full preprocessing with all features
        let preprocessed = tokio::time::timeout(
            std::time::Duration::from_millis(10),
            self.preprocessor.process_full(&analysis.query)
        ).await
            .context("Full preprocessing timeout")?
            .context("Full preprocessing failed")?;
        
        // Store in all systems
        let storage_future = self.storage.store_all(&preprocessed, query_id);
        
        // Search all sources in parallel
        let search_future = self.search.search_all(&preprocessed.embeddings);
        
        // Execute storage and search in parallel
        let (storage_result, search_results) = tokio::join!(
            tokio::time::timeout(
                std::time::Duration::from_millis(10),
                storage_future
            ),
            tokio::time::timeout(
                std::time::Duration::from_millis(20),
                search_future
            )
        );
        
        storage_result.context("Storage timeout")?.context("Storage failed")?;
        let results = search_results.context("Search timeout")?.context("Search failed")?;
        
        // Full fusion with cross-validation
        let fused = self.fusion.fuse_with_validation(results);
        let confidence = self.calculate_fused_confidence(&fused);
        
        debug!("Full pipeline completed in {:?} with {} results", start.elapsed(), fused.len());
        
        Ok(ProcessingResult {
            results: fused,
            confidence,
        })
    }

    /// Maximum intelligence path (45ms target) - 1% of queries
    async fn process_maximum_intelligence(&self, analysis: &QueryAnalysis, query_id: Uuid) -> Result<ProcessingResult> {
        let start = Instant::now();
        info!("Query {} using MAXIMUM INTELLIGENCE mode", query_id);
        
        // Everything runs in parallel with maximum resources
        let preprocess_future = self.preprocessor.process_maximum(&analysis.query);
        
        let preprocessed = tokio::time::timeout(
            std::time::Duration::from_millis(10),
            preprocess_future
        ).await
            .context("Maximum preprocessing timeout")?
            .context("Maximum preprocessing failed")?;
        
        // Parallel execution of all operations
        let storage_future = self.storage.store_all_parallel(&preprocessed, query_id);
        let search_future = self.search.search_all_parallel(&preprocessed.embeddings);
        
        let (storage_result, search_results) = tokio::join!(
            tokio::time::timeout(
                std::time::Duration::from_millis(15),
                storage_future
            ),
            tokio::time::timeout(
                std::time::Duration::from_millis(25),
                search_future
            )
        );
        
        storage_result.context("Maximum storage timeout")?.context("Maximum storage failed")?;
        let results = search_results.context("Maximum search timeout")?.context("Maximum search failed")?;
        
        // Maximum fusion with all features
        let fused = self.fusion.fuse_maximum(results);
        let confidence = self.calculate_fused_confidence(&fused);
        
        info!("Maximum intelligence completed in {:?} with {} results at {:.1}% confidence", 
            start.elapsed(), fused.len(), confidence * 100.0);
        
        Ok(ProcessingResult {
            results: fused,
            confidence,
        })
    }
    
    /// Generate minimal embedding for cache-only search
    async fn generate_minimal_embedding(&self, query: &str) -> Result<Vec<f32>> {
        // Use memory pool for zero-allocation
        PoolHandle::with_buffer(EMBEDDING_DIM, |buffer| {
            // TODO: Call actual embedding service
            // For now, generate placeholder embedding
            for i in 0..EMBEDDING_DIM {
                buffer[i] = (i as f32 * 0.1) % 1.0;
            }
            Ok(buffer.to_vec())
        })
    }
    
    /// Calculate confidence from search results
    fn calculate_result_confidence(&self, results: &[SearchResult]) -> f32 {
        if results.is_empty() {
            return 0.0;
        }
        
        // Average of top scores with decay
        let mut total = 0.0;
        let mut weight_sum = 0.0;
        
        for (i, result) in results.iter().take(5).enumerate() {
            let weight = 1.0 / (i as f32 + 1.0);
            total += result.score * weight;
            weight_sum += weight;
        }
        
        if weight_sum > 0.0 {
            total / weight_sum
        } else {
            0.0
        }
    }
    
    /// Calculate confidence from fused results
    fn calculate_fused_confidence(&self, results: &[FusedResult]) -> f32 {
        if results.is_empty() {
            return 0.0;
        }
        
        // Consider both individual confidence and cross-validation
        let avg_confidence: f32 = results.iter().map(|r| r.confidence).sum::<f32>() / results.len() as f32;
        let avg_validation: f32 = results.iter()
            .map(|r| r.metadata.cross_validation_score)
            .sum::<f32>() / results.len() as f32;
        
        // Weighted combination
        avg_confidence * 0.7 + avg_validation * 0.3
    }
    
    /// Format search results into fused results
    fn format_search_results(&self, results: Vec<SearchResult>) -> Vec<FusedResult> {
        results.into_iter().map(|r| FusedResult {
            id: Uuid::new_v4(),
            content: r.content,
            confidence: r.score,
            sources: vec![format!("{:?}", r.source)],
            metadata: FusedMetadata {
                cross_validation_score: 0.0,
                agreement_count: 1,
                temporal_relevance: 0.8,
                domain_match: 0.9,
            },
        }).collect()
    }
    
    /// Escalate to next routing path
    fn escalate_path(&self, current: &RoutingPath) -> Option<RoutingPath> {
        match current {
            RoutingPath::CacheOnly => Some(RoutingPath::SmartRouting),
            RoutingPath::SmartRouting => Some(RoutingPath::FullPipeline),
            RoutingPath::FullPipeline => Some(RoutingPath::MaximumIntelligence),
            RoutingPath::MaximumIntelligence => None, // Already at maximum
        }
    }
}

/// Internal processing result
struct ProcessingResult {
    results: Vec<FusedResult>,
    confidence: f32,
}

/// Fused result after fusion engine processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FusedResult {
    pub id: Uuid,
    pub content: String,
    pub confidence: f32,
    pub sources: Vec<String>,
    pub metadata: FusedMetadata,
}

/// Metadata for fused results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FusedMetadata {
    pub cross_validation_score: f32,
    pub agreement_count: usize,
    pub temporal_relevance: f32,
    pub domain_match: f32,
}

/// Pipeline response sent to clients
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineResponse {
    pub query_id: Uuid,
    pub results: Vec<FusedResult>,
    pub confidence: f32,
    pub latency_ms: u64,
    pub path_taken: RoutingPath,
    pub escalations: usize,
    pub metadata: serde_json::Value,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_pipeline_creation() {
        // This would need mock implementations
        // let pipeline = UnifiedPipeline::new(...).await.unwrap();
        // assert!(pipeline.config.auto_escalate);
    }
    
    #[test]
    fn test_escalation_logic() {
        let pipeline = UnifiedPipeline {
            router: Arc::new(IntelligentRouter::new()),
            preprocessor: Arc::new(ParallelPreprocessor::new()),
            storage: Arc::new(StorageEngine::new()),
            search: Arc::new(SearchOrchestrator::new()),
            fusion: Arc::new(FusionEngine::new()),
            db_pool: Arc::new(unsafe { std::mem::zeroed() }), // Mock
            metrics: Arc::new(MetricsCollector::new()),
            config: PipelineConfig::default(),
        };
        
        assert_eq!(
            pipeline.escalate_path(&RoutingPath::CacheOnly),
            Some(RoutingPath::SmartRouting)
        );
        assert_eq!(
            pipeline.escalate_path(&RoutingPath::MaximumIntelligence),
            None
        );
    }
}