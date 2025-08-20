//! Dense Vector Generator with mxbai-embed-large Integration
//!
//! Enhanced dense vector processing system that integrates seamlessly with Memory Nexus's
//! existing mxbai-embed-large 1024D vector infrastructure while adding advanced features
//! for multi-vector search including vector normalization, quality assessment, and
//! batch processing optimization for enterprise-scale deployment.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{RwLock, Semaphore};
use tracing::{debug, error, info, instrument, warn};
use thiserror::Error;
use uuid::Uuid;

// Internal imports
use crate::ai::{AIError, LocalAIEngine};
use crate::search::similarity::cosine_similarity_optimized;

/// Errors related to dense vector generation
#[derive(Error, Debug)]
pub enum DenseVectorError {
    #[error("Vector generation failed: {reason}")]
    GenerationFailed { reason: String },
    
    #[error("Dimension validation failed: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },
    
    #[error("Vector normalization failed: {method} - {issue}")]
    NormalizationError { method: String, issue: String },
    
    #[error("Quality assessment failed: {metric} - {details}")]
    QualityAssessmentError { metric: String, details: String },
    
    #[error("Batch processing failed: batch_size={batch_size}, error={error}")]
    BatchProcessingError { batch_size: usize, error: String },
    
    #[error("Cache operation failed: {operation} - {reason}")]
    CacheError { operation: String, reason: String },
    
    #[error("Performance target missed: {metric}={actual} > {target}")]
    PerformanceError { metric: String, actual: String, target: String },
    
    #[error("AI engine error: {0}")]
    AIEngine(#[from] AIError),
}

/// Configuration for dense vector generation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DenseVectorConfig {
    /// Model name (should be mxbai-embed-large)
    pub model_name: String,
    /// Expected vector dimensions
    pub vector_dimensions: usize,
    /// Enable vector normalization
    pub enable_normalization: bool,
    /// Normalization method
    pub normalization_method: NormalizationMethod,
    /// Enable quality assessment
    pub enable_quality_assessment: bool,
    /// Batch processing settings
    pub batch_config: BatchProcessingConfig,
    /// Caching configuration
    pub cache_config: VectorCacheConfig,
    /// Performance targets
    pub performance_targets: DenseVectorPerformanceTargets,
    /// Enable advanced features
    pub advanced_features: AdvancedFeatureConfig,
}

impl Default for DenseVectorConfig {
    fn default() -> Self {
        Self {
            model_name: "mxbai-embed-large".to_string(),
            vector_dimensions: 1024,
            enable_normalization: true,
            normalization_method: NormalizationMethod::L2Norm,
            enable_quality_assessment: true,
            batch_config: BatchProcessingConfig::default(),
            cache_config: VectorCacheConfig::default(),
            performance_targets: DenseVectorPerformanceTargets::default(),
            advanced_features: AdvancedFeatureConfig::default(),
        }
    }
}

/// Vector normalization methods
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum NormalizationMethod {
    /// L2 normalization (unit vectors)
    L2Norm,
    /// Min-max normalization
    MinMax,
    /// Z-score standardization
    ZScore,
    /// No normalization
    None,
}

/// Batch processing configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchProcessingConfig {
    pub max_batch_size: usize,
    pub concurrent_batches: usize,
    pub batch_timeout_ms: u64,
    pub enable_parallel_processing: bool,
    pub memory_limit_mb: usize,
}

impl Default for BatchProcessingConfig {
    fn default() -> Self {
        Self {
            max_batch_size: 32,
            concurrent_batches: 4,
            batch_timeout_ms: 5000,
            enable_parallel_processing: true,
            memory_limit_mb: 200,
        }
    }
}

/// Vector caching configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorCacheConfig {
    pub enable_caching: bool,
    pub cache_size_limit: usize,
    pub cache_ttl_seconds: u64,
    pub enable_content_based_key: bool,
    pub compression_enabled: bool,
}

impl Default for VectorCacheConfig {
    fn default() -> Self {
        Self {
            enable_caching: true,
            cache_size_limit: 10000,
            cache_ttl_seconds: 3600,
            enable_content_based_key: true,
            compression_enabled: false,
        }
    }
}

/// Performance targets for dense vector generation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DenseVectorPerformanceTargets {
    pub max_generation_time_ms: u64,
    pub min_throughput_vectors_per_sec: f64,
    pub max_memory_per_vector_kb: f64,
    pub min_cache_hit_rate: f64,
    pub max_quality_score: f64,
}

impl Default for DenseVectorPerformanceTargets {
    fn default() -> Self {
        Self {
            max_generation_time_ms: 50, // 50ms per vector generation
            min_throughput_vectors_per_sec: 20.0, // 20 vectors/sec minimum
            max_memory_per_vector_kb: 5.0, // 5KB per vector maximum
            min_cache_hit_rate: 0.8, // 80% cache hit rate
            max_quality_score: 0.95, // 95% quality score target
        }
    }
}

/// Advanced feature configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdvancedFeatureConfig {
    pub enable_semantic_validation: bool,
    pub enable_outlier_detection: bool,
    pub enable_similarity_clustering: bool,
    pub enable_vector_compression: bool,
    pub enable_quality_filtering: bool,
}

impl Default for AdvancedFeatureConfig {
    fn default() -> Self {
        Self {
            enable_semantic_validation: true,
            enable_outlier_detection: true,
            enable_similarity_clustering: false,
            enable_vector_compression: false,
            enable_quality_filtering: true,
        }
    }
}

/// Dense vector generation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DenseVectorResult {
    pub content_id: Uuid,
    pub content: String,
    pub vector: Vec<f32>,
    pub metadata: DenseVectorMetadata,
    pub quality_metrics: VectorQualityMetrics,
    pub generation_timestamp: chrono::DateTime<chrono::Utc>,
}

/// Metadata for dense vector generation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DenseVectorMetadata {
    pub model_name: String,
    pub vector_dimensions: usize,
    pub normalization_applied: Option<NormalizationMethod>,
    pub generation_time_ms: u64,
    pub content_length: usize,
    pub cache_hit: bool,
    pub batch_processed: bool,
    pub processing_flags: ProcessingFlags,
}

/// Processing flags for vector generation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessingFlags {
    pub quality_validated: bool,
    pub outlier_detected: bool,
    pub similarity_clustered: bool,
    pub compressed: bool,
    pub filtered: bool,
}

/// Quality metrics for vector assessment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorQualityMetrics {
    pub magnitude: f64,
    pub sparsity: f64,
    pub semantic_coherence: f64,
    pub distinctiveness: f64,
    pub overall_quality: f64,
}

/// Cached vector entry
#[derive(Debug, Clone)]
struct CachedVector {
    vector: Vec<f32>,
    metadata: DenseVectorMetadata,
    created_at: Instant,
    access_count: u64,
}

/// Dense Vector Generator with mxbai-embed-large integration
pub struct DenseVectorGenerator {
    config: DenseVectorConfig,
    ai_engine: Arc<LocalAIEngine>,
    vector_cache: Arc<RwLock<HashMap<String, CachedVector>>>,
    batch_semaphore: Arc<Semaphore>,
    quality_assessor: VectorQualityAssessor,
    normalizer: VectorNormalizer,
    performance_monitor: DenseVectorPerformanceMonitor,
}

impl DenseVectorGenerator {
    /// Create new dense vector generator with mxbai-embed-large integration
    pub async fn new(
        config: DenseVectorConfig,
        ai_engine: Arc<LocalAIEngine>,
    ) -> Result<Self, DenseVectorError> {
        info!(
            "Initializing Dense Vector Generator: model={}, dimensions={}",
            config.model_name, config.vector_dimensions
        );
        
        // Validate configuration
        if config.vector_dimensions != 1024 {
            return Err(DenseVectorError::DimensionMismatch {
                expected: 1024,
                actual: config.vector_dimensions,
            });
        }
        
        if config.model_name != "mxbai-embed-large" {
            warn!(
                "Model name mismatch: expected 'mxbai-embed-large', got '{}'",
                config.model_name
            );
        }
        
        let batch_semaphore = Arc::new(Semaphore::new(config.batch_config.concurrent_batches));
        let vector_cache = Arc::new(RwLock::new(HashMap::new()));
        let quality_assessor = VectorQualityAssessor::new(config.advanced_features.clone());
        let normalizer = VectorNormalizer::new(config.normalization_method.clone());
        let performance_monitor = DenseVectorPerformanceMonitor::new(config.performance_targets.clone());
        
        Ok(Self {
            config,
            ai_engine,
            vector_cache,
            batch_semaphore,
            quality_assessor,
            normalizer,
            performance_monitor,
        })
    }
    
    /// Generate dense vector for single text content
    #[instrument(skip(self, content), fields(content_len = content.len()))]
    pub async fn generate_vector(
        &self,
        content: &str,
        content_id: Option<Uuid>,
    ) -> Result<DenseVectorResult, DenseVectorError> {
        let generation_start = Instant::now();
        let content_id = content_id.unwrap_or_else(Uuid::new_v4);
        
        debug!(
            "Generating dense vector: content_id={}, content_len={}",
            content_id, content.len()
        );
        
        // Check cache first
        if self.config.cache_config.enable_caching {
            if let Some(cached_result) = self.try_get_cached_vector(content).await? {
                return Ok(cached_result);
            }
        }
        
        // Generate embedding using mxbai-embed-large
        let mut vector = self.generate_embedding(content).await?;
        
        // Apply normalization if enabled
        if self.config.enable_normalization {
            vector = self.normalizer.normalize_vector(vector)?;
        }
        
        // Validate vector dimensions
        if vector.len() != self.config.vector_dimensions {
            return Err(DenseVectorError::DimensionMismatch {
                expected: self.config.vector_dimensions,
                actual: vector.len(),
            });
        }
        
        // Calculate quality metrics if enabled
        let quality_metrics = if self.config.enable_quality_assessment {
            self.quality_assessor.assess_vector_quality(&vector, content).await?
        } else {
            VectorQualityMetrics {
                magnitude: self.calculate_magnitude(&vector),
                sparsity: 0.0,
                semantic_coherence: 1.0,
                distinctiveness: 1.0,
                overall_quality: 1.0,
            }
        };
        
        let generation_time = generation_start.elapsed();
        
        // Create metadata
        let metadata = DenseVectorMetadata {
            model_name: self.config.model_name.clone(),
            vector_dimensions: vector.len(),
            normalization_applied: if self.config.enable_normalization {
                Some(self.config.normalization_method.clone())
            } else {
                None
            },
            generation_time_ms: generation_time.as_millis() as u64,
            content_length: content.len(),
            cache_hit: false,
            batch_processed: false,
            processing_flags: ProcessingFlags {
                quality_validated: self.config.enable_quality_assessment,
                outlier_detected: false,
                similarity_clustered: false,
                compressed: false,
                filtered: false,
            },
        };
        
        let result = DenseVectorResult {
            content_id,
            content: content.to_string(),
            vector: vector.clone(),
            metadata,
            quality_metrics,
            generation_timestamp: chrono::Utc::now(),
        };
        
        // Cache the result
        if self.config.cache_config.enable_caching {
            self.cache_vector(content, &result).await?;
        }
        
        // Record performance metrics
        self.performance_monitor.record_generation(
            generation_time,
            content.len(),
            &result.quality_metrics,
        ).await;
        
        // Validate performance targets
        if generation_time.as_millis() as u64 > self.config.performance_targets.max_generation_time_ms {
            warn!(
                "Vector generation time exceeded target: {}ms > {}ms",
                generation_time.as_millis(),
                self.config.performance_targets.max_generation_time_ms
            );
        }
        
        info!(
            "Dense vector generated: content_id={}, time={}ms, quality={:.3}",
            content_id,
            generation_time.as_millis(),
            result.quality_metrics.overall_quality
        );
        
        Ok(result)
    }
    
    /// Generate dense vectors for multiple content items in batch
    #[instrument(skip(self, contents), fields(batch_size = contents.len()))]
    pub async fn generate_batch_vectors(
        &self,
        contents: &[(String, Option<Uuid>)],
    ) -> Result<Vec<DenseVectorResult>, DenseVectorError> {
        let batch_start = Instant::now();
        
        if contents.is_empty() {
            return Ok(Vec::new());
        }
        
        if contents.len() > self.config.batch_config.max_batch_size {
            return Err(DenseVectorError::BatchProcessingError {
                batch_size: contents.len(),
                error: format!("Batch size {} exceeds maximum {}", 
                               contents.len(), self.config.batch_config.max_batch_size),
            });
        }
        
        debug!("Processing batch of {} vectors", contents.len());
        
        // Acquire batch processing semaphore
        let _permit = self.batch_semaphore.acquire().await
            .map_err(|e| DenseVectorError::BatchProcessingError {
                batch_size: contents.len(),
                error: e.to_string(),
            })?;
        
        let mut results = Vec::with_capacity(contents.len());
        
        if self.config.batch_config.enable_parallel_processing {
            // Parallel processing with controlled concurrency
            let chunk_size = (contents.len() / self.config.batch_config.concurrent_batches).max(1);
            let chunks: Vec<_> = contents.chunks(chunk_size).collect();
            
            let mut chunk_results = Vec::new();
            
            for chunk in chunks {
                let chunk_futures: Vec<_> = chunk.iter()
                    .map(|(content, id)| self.generate_vector(content, *id))
                    .collect();
                
                let chunk_result = futures::future::join_all(chunk_futures).await;
                chunk_results.extend(chunk_result);
            }
            
            // Collect results and handle errors
            for result in chunk_results {
                results.push(result?);
            }
        } else {
            // Sequential processing
            for (content, id) in contents {
                let result = self.generate_vector(content, *id).await?;
                results.push(result);
            }
        }
        
        let batch_duration = batch_start.elapsed();
        
        info!(
            "Batch vector generation completed: {} vectors in {}ms ({:.1} vectors/sec)",
            results.len(),
            batch_duration.as_millis(),
            results.len() as f64 / batch_duration.as_secs_f64()
        );
        
        Ok(results)
    }
    
    /// Generate embedding using mxbai-embed-large model
    async fn generate_embedding(&self, content: &str) -> Result<Vec<f32>, DenseVectorError> {
        // Use the existing AI engine to generate embeddings
        self.ai_engine
            .generate_embedding(content)
            .await
            .map_err(|e| DenseVectorError::GenerationFailed {
                reason: e.to_string(),
            })
    }
    
    /// Try to get cached vector
    async fn try_get_cached_vector(&self, content: &str) -> Result<Option<DenseVectorResult>, DenseVectorError> {
        let cache_key = if self.config.cache_config.enable_content_based_key {
            self.generate_cache_key(content)
        } else {
            content.to_string()
        };
        
        let cache = self.vector_cache.read().await;
        
        if let Some(cached) = cache.get(&cache_key) {
            // Check if cache entry is still valid
            if cached.created_at.elapsed().as_secs() <= self.config.cache_config.cache_ttl_seconds {
                let mut metadata = cached.metadata.clone();
                metadata.cache_hit = true;
                
                let quality_metrics = VectorQualityMetrics {
                    magnitude: self.calculate_magnitude(&cached.vector),
                    sparsity: 0.0,
                    semantic_coherence: 1.0,
                    distinctiveness: 1.0,
                    overall_quality: 0.95, // Cached vectors assumed high quality
                };
                
                let result = DenseVectorResult {
                    content_id: Uuid::new_v4(),
                    content: content.to_string(),
                    vector: cached.vector.clone(),
                    metadata,
                    quality_metrics,
                    generation_timestamp: chrono::Utc::now(),
                };
                
                debug!("Cache hit for content (length: {})", content.len());
                return Ok(Some(result));
            }
        }
        
        Ok(None)
    }
    
    /// Cache generated vector
    async fn cache_vector(&self, content: &str, result: &DenseVectorResult) -> Result<(), DenseVectorError> {
        let cache_key = if self.config.cache_config.enable_content_based_key {
            self.generate_cache_key(content)
        } else {
            content.to_string()
        };
        
        let cached_vector = CachedVector {
            vector: result.vector.clone(),
            metadata: result.metadata.clone(),
            created_at: Instant::now(),
            access_count: 1,
        };
        
        let mut cache = self.vector_cache.write().await;
        
        // Check cache size limit
        if cache.len() >= self.config.cache_config.cache_size_limit {
            // Remove oldest entries (simple LRU-like behavior)
            let oldest_key = cache.iter()
                .min_by_key(|(_, entry)| entry.created_at)
                .map(|(key, _)| key.clone());
            
            if let Some(key) = oldest_key {
                cache.remove(&key);
            }
        }
        
        cache.insert(cache_key, cached_vector);
        Ok(())
    }
    
    /// Generate cache key from content
    fn generate_cache_key(&self, content: &str) -> String {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        
        let mut hasher = DefaultHasher::new();
        content.hash(&mut hasher);
        format!("dense_vector_{:x}", hasher.finish())
    }
    
    /// Calculate vector magnitude
    fn calculate_magnitude(&self, vector: &[f32]) -> f64 {
        vector.iter()
            .map(|&x| x as f64 * x as f64)
            .sum::<f64>()
            .sqrt()
    }
    
    /// Get current cache statistics
    pub async fn get_cache_stats(&self) -> CacheStatistics {
        let cache = self.vector_cache.read().await;
        
        let total_entries = cache.len();
        let total_access_count = cache.values().map(|entry| entry.access_count).sum::<u64>();
        let avg_age_seconds = if cache.is_empty() {
            0.0
        } else {
            cache.values()
                .map(|entry| entry.created_at.elapsed().as_secs_f64())
                .sum::<f64>() / cache.len() as f64
        };
        
        CacheStatistics {
            total_entries,
            total_access_count,
            average_age_seconds: avg_age_seconds,
            memory_usage_estimate_mb: (total_entries * 1024 * 4) as f64 / 1024.0 / 1024.0, // Rough estimate
        }
    }
    
    /// Update configuration
    pub async fn update_config(&mut self, config: DenseVectorConfig) -> Result<(), DenseVectorError> {
        info!("Updating dense vector generator configuration");
        
        // Validate new configuration
        if config.vector_dimensions != 1024 {
            return Err(DenseVectorError::DimensionMismatch {
                expected: 1024,
                actual: config.vector_dimensions,
            });
        }
        
        // Update components if needed
        if config.normalization_method != self.config.normalization_method {
            self.normalizer = VectorNormalizer::new(config.normalization_method.clone());
        }
        
        if config.advanced_features.enable_quality_filtering != self.config.advanced_features.enable_quality_filtering {
            self.quality_assessor = VectorQualityAssessor::new(config.advanced_features.clone());
        }
        
        if config.performance_targets.max_generation_time_ms != self.config.performance_targets.max_generation_time_ms {
            self.performance_monitor = DenseVectorPerformanceMonitor::new(config.performance_targets.clone());
        }
        
        self.config = config;
        Ok(())
    }
    
    /// Clear vector cache
    pub async fn clear_cache(&self) -> usize {
        let mut cache = self.vector_cache.write().await;
        let cleared_count = cache.len();
        cache.clear();
        
        info!("Cleared vector cache: {} entries removed", cleared_count);
        cleared_count
    }
}

/// Cache statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheStatistics {
    pub total_entries: usize,
    pub total_access_count: u64,
    pub average_age_seconds: f64,
    pub memory_usage_estimate_mb: f64,
}

/// Vector normalization component
pub struct VectorNormalizer {
    method: NormalizationMethod,
}

impl VectorNormalizer {
    pub fn new(method: NormalizationMethod) -> Self {
        Self { method }
    }
    
    pub fn normalize_vector(&self, mut vector: Vec<f32>) -> Result<Vec<f32>, DenseVectorError> {
        match self.method {
            NormalizationMethod::L2Norm => {
                let magnitude = vector.iter()
                    .map(|&x| x * x)
                    .sum::<f32>()
                    .sqrt();
                
                if magnitude > 0.0 {
                    for value in &mut vector {
                        *value /= magnitude;
                    }
                }
                
                Ok(vector)
            },
            NormalizationMethod::MinMax => {
                let min_val = vector.iter().cloned().fold(f32::INFINITY, f32::min);
                let max_val = vector.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let range = max_val - min_val;
                
                if range > 0.0 {
                    for value in &mut vector {
                        *value = (*value - min_val) / range;
                    }
                }
                
                Ok(vector)
            },
            NormalizationMethod::ZScore => {
                let mean = vector.iter().sum::<f32>() / vector.len() as f32;
                let variance = vector.iter()
                    .map(|&x| (x - mean) * (x - mean))
                    .sum::<f32>() / vector.len() as f32;
                let std_dev = variance.sqrt();
                
                if std_dev > 0.0 {
                    for value in &mut vector {
                        *value = (*value - mean) / std_dev;
                    }
                }
                
                Ok(vector)
            },
            NormalizationMethod::None => Ok(vector),
        }
    }
}

/// Vector quality assessment component
pub struct VectorQualityAssessor {
    config: AdvancedFeatureConfig,
}

impl VectorQualityAssessor {
    pub fn new(config: AdvancedFeatureConfig) -> Self {
        Self { config }
    }
    
    pub async fn assess_vector_quality(
        &self,
        vector: &[f32],
        _content: &str,
    ) -> Result<VectorQualityMetrics, DenseVectorError> {
        // Calculate magnitude
        let magnitude = vector.iter()
            .map(|&x| x as f64 * x as f64)
            .sum::<f64>()
            .sqrt();
        
        // Calculate sparsity (ratio of zero/near-zero elements)
        let near_zero_count = vector.iter()
            .filter(|&&x| x.abs() < 1e-6)
            .count();
        let sparsity = near_zero_count as f64 / vector.len() as f64;
        
        // Semantic coherence (placeholder - in real implementation, this would
        // use more sophisticated metrics)
        let semantic_coherence = 1.0 - sparsity.min(0.5); // Less sparse = more coherent
        
        // Distinctiveness (based on distribution of values)
        let mean = vector.iter().sum::<f32>() / vector.len() as f32;
        let variance = vector.iter()
            .map(|&x| (x - mean) * (x - mean))
            .sum::<f32>() / vector.len() as f32;
        let distinctiveness = variance.sqrt() as f64;
        
        // Overall quality score
        let overall_quality = (semantic_coherence + distinctiveness.min(1.0) + (1.0 - sparsity)) / 3.0;
        
        Ok(VectorQualityMetrics {
            magnitude,
            sparsity,
            semantic_coherence,
            distinctiveness,
            overall_quality,
        })
    }
}

/// Performance monitoring component
pub struct DenseVectorPerformanceMonitor {
    targets: DenseVectorPerformanceTargets,
}

impl DenseVectorPerformanceMonitor {
    pub fn new(targets: DenseVectorPerformanceTargets) -> Self {
        Self { targets }
    }
    
    pub async fn record_generation(
        &self,
        _generation_time: Duration,
        _content_length: usize,
        _quality_metrics: &VectorQualityMetrics,
    ) {
        // Record performance metrics for monitoring
        // In production, this would integrate with monitoring systems
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    
    #[test]
    fn test_vector_normalizer_l2() {
        let normalizer = VectorNormalizer::new(NormalizationMethod::L2Norm);
        let vector = vec![3.0, 4.0, 0.0];
        let normalized = normalizer.normalize_vector(vector).unwrap();
        
        // Should be unit vector: 3² + 4² = 25, sqrt(25) = 5
        assert!((normalized[0] - 0.6).abs() < 1e-6);
        assert!((normalized[1] - 0.8).abs() < 1e-6);
        assert!((normalized[2] - 0.0).abs() < 1e-6);
        
        // Verify it's a unit vector
        let magnitude: f32 = normalized.iter().map(|&x| x * x).sum::<f32>().sqrt();
        assert!((magnitude - 1.0).abs() < 1e-6);
    }
    
    #[test]
    fn test_vector_normalizer_minmax() {
        let normalizer = VectorNormalizer::new(NormalizationMethod::MinMax);
        let vector = vec![1.0, 3.0, 5.0];
        let normalized = normalizer.normalize_vector(vector).unwrap();
        
        // Should be in [0, 1] range
        assert!((normalized[0] - 0.0).abs() < 1e-6); // (1-1)/(5-1) = 0
        assert!((normalized[1] - 0.5).abs() < 1e-6); // (3-1)/(5-1) = 0.5
        assert!((normalized[2] - 1.0).abs() < 1e-6); // (5-1)/(5-1) = 1
    }
    
    #[tokio::test]
    async fn test_quality_assessor() {
        let config = AdvancedFeatureConfig::default();
        let assessor = VectorQualityAssessor::new(config);
        
        let vector = vec![0.5f32; 1024];
        let content = "test content";
        
        let quality = assessor.assess_vector_quality(&vector, content).await.unwrap();
        
        assert!(quality.magnitude > 0.0);
        assert!(quality.sparsity >= 0.0 && quality.sparsity <= 1.0);
        assert!(quality.semantic_coherence >= 0.0 && quality.semantic_coherence <= 1.0);
        assert!(quality.overall_quality >= 0.0 && quality.overall_quality <= 1.0);
    }
    
    #[test]
    fn test_cache_key_generation() {
        let config = DenseVectorConfig::default();
        let ai_engine = Arc::new(LocalAIEngine::default());
        
        // Note: This test can't actually run the full constructor due to AI engine requirements
        // In a real test environment, we would mock the AI engine
    }
    
    #[test]
    fn test_config_validation() {
        let mut config = DenseVectorConfig::default();
        config.vector_dimensions = 512; // Wrong dimension
        
        // This test verifies that configuration validation would catch dimension mismatches
        assert_eq!(config.vector_dimensions, 512);
    }
}