//! Multi-Vector Coordinator for Unified Vector Management
//!
//! Coordinates the generation and management of multiple vector types (dense, sparse, token-level)
//! ensuring consistency, quality, and optimal performance across all vector representations.
//! Provides unified interface for batch processing, caching, and quality assurance while
//! maintaining compatibility with Memory Nexus enterprise infrastructure.

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
use crate::ai::LocalAIEngine;
use crate::vectors::dense_vector_generator::{DenseVectorGenerator, DenseVectorResult, DenseVectorError};
use crate::vectors::sparse_vector_generator::{SparseVectorGenerator, SparseVectorResult, SparseVectorError};
use crate::vectors::token_level_matching::{TokenLevelProcessor, TokenVectorResult, TokenLevelError};

/// Errors related to multi-vector coordination
#[derive(Error, Debug)]
pub enum MultiVectorCoordinationError {
    #[error("Dense vector generation failed: {0}")]
    DenseVectorFailed(#[from] DenseVectorError),
    
    #[error("Sparse vector generation failed: {0}")]
    SparseVectorFailed(#[from] SparseVectorError),
    
    #[error("Token-level processing failed: {0}")]
    TokenLevelFailed(#[from] TokenLevelError),
    
    #[error("Batch coordination failed: batch_size={batch_size}, reason={reason}")]
    BatchCoordinationError { batch_size: usize, reason: String },
    
    #[error("Vector consistency check failed: {vector_type} - {issue}")]
    ConsistencyError { vector_type: String, issue: String },
    
    #[error("Quality validation failed: {metric}={value}, threshold={threshold}")]
    QualityValidationError { metric: String, value: f64, threshold: f64 },
    
    #[error("Resource limit exceeded: {resource}={current} > {limit}")]
    ResourceLimitError { resource: String, current: String, limit: String },
    
    #[error("Configuration error: {parameter} - {details}")]
    ConfigurationError { parameter: String, details: String },
}

/// Configuration for multi-vector coordination
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiVectorCoordinatorConfig {
    /// Enable dense vector generation
    pub enable_dense_vectors: bool,
    /// Enable sparse vector generation
    pub enable_sparse_vectors: bool,
    /// Enable token-level processing
    pub enable_token_level: bool,
    /// Batch processing configuration
    pub batch_config: BatchCoordinationConfig,
    /// Quality assurance settings
    pub quality_config: QualityAssuranceConfig,
    /// Caching configuration
    pub cache_config: UnifiedCacheConfig,
    /// Performance targets
    pub performance_targets: CoordinatorPerformanceTargets,
    /// Resource limits
    pub resource_limits: ResourceLimits,
}

impl Default for MultiVectorCoordinatorConfig {
    fn default() -> Self {
        Self {
            enable_dense_vectors: true,
            enable_sparse_vectors: true,
            enable_token_level: true,
            batch_config: BatchCoordinationConfig::default(),
            quality_config: QualityAssuranceConfig::default(),
            cache_config: UnifiedCacheConfig::default(),
            performance_targets: CoordinatorPerformanceTargets::default(),
            resource_limits: ResourceLimits::default(),
        }
    }
}

/// Batch coordination configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchCoordinationConfig {
    pub max_batch_size: usize,
    pub enable_parallel_generation: bool,
    pub max_concurrent_generators: usize,
    pub batch_timeout_ms: u64,
    pub enable_adaptive_batching: bool,
    pub quality_check_interval: usize,
}

impl Default for BatchCoordinationConfig {
    fn default() -> Self {
        Self {
            max_batch_size: 50,
            enable_parallel_generation: true,
            max_concurrent_generators: 8,
            batch_timeout_ms: 30000, // 30 seconds
            enable_adaptive_batching: true,
            quality_check_interval: 10,
        }
    }
}

/// Quality assurance configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityAssuranceConfig {
    pub enable_consistency_checks: bool,
    pub enable_quality_validation: bool,
    pub min_dense_quality: f64,
    pub min_sparse_quality: f64,
    pub min_token_quality: f64,
    pub enable_cross_validation: bool,
    pub quality_improvement_threshold: f64,
}

impl Default for QualityAssuranceConfig {
    fn default() -> Self {
        Self {
            enable_consistency_checks: true,
            enable_quality_validation: true,
            min_dense_quality: 0.8,
            min_sparse_quality: 0.75,
            min_token_quality: 0.7,
            enable_cross_validation: true,
            quality_improvement_threshold: 0.05, // 5% improvement threshold
        }
    }
}

/// Unified cache configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UnifiedCacheConfig {
    pub enable_unified_caching: bool,
    pub cache_size_mb: usize,
    pub cache_ttl_seconds: u64,
    pub enable_cache_compression: bool,
    pub cache_consistency_checks: bool,
}

impl Default for UnifiedCacheConfig {
    fn default() -> Self {
        Self {
            enable_unified_caching: true,
            cache_size_mb: 200,
            cache_ttl_seconds: 1800, // 30 minutes
            enable_cache_compression: true,
            cache_consistency_checks: true,
        }
    }
}

/// Performance targets for coordinator
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoordinatorPerformanceTargets {
    pub max_total_generation_time_ms: u64,
    pub min_throughput_vectors_per_sec: f64,
    pub max_memory_usage_mb: usize,
    pub min_quality_score: f64,
    pub max_error_rate: f64,
}

impl Default for CoordinatorPerformanceTargets {
    fn default() -> Self {
        Self {
            max_total_generation_time_ms: 150, // 150ms total for all vectors
            min_throughput_vectors_per_sec: 15.0,
            max_memory_usage_mb: 300,
            min_quality_score: 0.85,
            max_error_rate: 0.01, // 1% error rate max
        }
    }
}

/// Resource limits
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceLimits {
    pub max_concurrent_operations: usize,
    pub max_memory_per_operation_mb: usize,
    pub max_cpu_cores: usize,
    pub max_disk_usage_mb: usize,
}

impl Default for ResourceLimits {
    fn default() -> Self {
        Self {
            max_concurrent_operations: 20,
            max_memory_per_operation_mb: 50,
            max_cpu_cores: 8,
            max_disk_usage_mb: 1000,
        }
    }
}

/// Unified multi-vector result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UnifiedVectorResult {
    pub content_id: Uuid,
    pub content: String,
    pub dense_vector: Option<DenseVectorResult>,
    pub sparse_vector: Option<SparseVectorResult>,
    pub token_vectors: Option<TokenVectorResult>,
    pub metadata: UnifiedVectorMetadata,
    pub quality_assessment: UnifiedQualityAssessment,
    pub generation_timestamp: chrono::DateTime<chrono::Utc>,
}

/// Metadata for unified vector generation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UnifiedVectorMetadata {
    pub total_generation_time_ms: u64,
    pub vectors_generated: usize,
    pub cache_hits: usize,
    pub quality_checks_passed: bool,
    pub consistency_verified: bool,
    pub resource_usage: ResourceUsageStats,
    pub coordination_flags: CoordinationFlags,
}

/// Resource usage statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceUsageStats {
    pub peak_memory_mb: f64,
    pub cpu_time_ms: u64,
    pub disk_io_mb: f64,
    pub network_io_mb: f64,
}

/// Coordination flags
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoordinationFlags {
    pub batch_processed: bool,
    pub parallel_generated: bool,
    pub quality_validated: bool,
    pub cached: bool,
    pub compressed: bool,
}

/// Quality assessment for unified vectors
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UnifiedQualityAssessment {
    pub overall_quality: f64,
    pub dense_quality: Option<f64>,
    pub sparse_quality: Option<f64>,
    pub token_quality: Option<f64>,
    pub consistency_score: f64,
    pub cross_validation_passed: bool,
    pub quality_improvement_achieved: bool,
}

/// Cached multi-vector entry
#[derive(Debug, Clone)]
struct CachedUnifiedVector {
    result: UnifiedVectorResult,
    created_at: Instant,
    access_count: u64,
    compressed: bool,
}

/// Multi-Vector Coordinator
pub struct MultiVectorCoordinator {
    config: MultiVectorCoordinatorConfig,
    dense_generator: Option<Arc<DenseVectorGenerator>>,
    sparse_generator: Option<Arc<SparseVectorGenerator>>,
    token_processor: Option<Arc<TokenLevelProcessor>>,
    unified_cache: Arc<RwLock<HashMap<String, CachedUnifiedVector>>>,
    coordination_semaphore: Arc<Semaphore>,
    quality_validator: QualityValidator,
    performance_monitor: CoordinatorPerformanceMonitor,
}

impl MultiVectorCoordinator {
    /// Create new multi-vector coordinator
    pub async fn new(
        config: MultiVectorCoordinatorConfig,
        ai_engine: Arc<LocalAIEngine>,
    ) -> Result<Self, MultiVectorCoordinationError> {
        info!(
            "Initializing Multi-Vector Coordinator: dense={}, sparse={}, token={}",
            config.enable_dense_vectors, config.enable_sparse_vectors, config.enable_token_level
        );
        
        // Initialize dense vector generator if enabled
        let dense_generator = if config.enable_dense_vectors {
            let dense_config = crate::vectors::dense_vector_generator::DenseVectorConfig::default();
            let generator = DenseVectorGenerator::new(dense_config, Arc::clone(&ai_engine)).await
                .map_err(MultiVectorCoordinationError::DenseVectorFailed)?;
            Some(Arc::new(generator))
        } else {
            None
        };
        
        // Initialize sparse vector generator if enabled
        let sparse_generator = if config.enable_sparse_vectors {
            let sparse_config = crate::vectors::sparse_vector_generator::SparseVectorConfig::default();
            let generator = SparseVectorGenerator::new(sparse_config).await
                .map_err(MultiVectorCoordinationError::SparseVectorFailed)?;
            Some(Arc::new(generator))
        } else {
            None
        };
        
        // Initialize token-level processor if enabled
        let token_processor = if config.enable_token_level {
            let token_config = crate::vectors::token_level_matching::TokenLevelConfig::default();
            let processor = TokenLevelProcessor::new(token_config, Arc::clone(&ai_engine)).await
                .map_err(MultiVectorCoordinationError::TokenLevelFailed)?;
            Some(Arc::new(processor))
        } else {
            None
        };
        
        let unified_cache = Arc::new(RwLock::new(HashMap::new()));
        let coordination_semaphore = Arc::new(Semaphore::new(config.resource_limits.max_concurrent_operations));
        let quality_validator = QualityValidator::new(config.quality_config.clone());
        let performance_monitor = CoordinatorPerformanceMonitor::new(config.performance_targets.clone());
        
        Ok(Self {
            config,
            dense_generator,
            sparse_generator,
            token_processor,
            unified_cache,
            coordination_semaphore,
            quality_validator,
            performance_monitor,
        })
    }
    
    /// Generate unified multi-vector representation for content
    #[instrument(skip(self, content), fields(content_len = content.len()))]
    pub async fn generate_unified_vectors(
        &self,
        content: &str,
        content_id: Option<Uuid>,
    ) -> Result<UnifiedVectorResult, MultiVectorCoordinationError> {
        let generation_start = Instant::now();
        let content_id = content_id.unwrap_or_else(Uuid::new_v4);
        
        debug!(
            "Generating unified vectors: content_id={}, content_len={}",
            content_id, content.len()
        );
        
        // Acquire coordination semaphore
        let _permit = self.coordination_semaphore.acquire().await
            .map_err(|e| MultiVectorCoordinationError::ResourceLimitError {
                resource: "concurrent_operations".to_string(),
                current: "semaphore_error".to_string(),
                limit: self.config.resource_limits.max_concurrent_operations.to_string(),
            })?;
        
        // Check cache first
        if self.config.cache_config.enable_unified_caching {
            if let Some(cached_result) = self.try_get_cached_result(content).await? {
                return Ok(cached_result);
            }
        }
        
        let mut vectors_generated = 0;
        let mut cache_hits = 0;
        
        // Generate vectors based on configuration
        let dense_result = if self.config.enable_dense_vectors && self.dense_generator.is_some() {
            let generator = self.dense_generator.as_ref().unwrap();
            match generator.generate_vector(content, Some(content_id)).await {
                Ok(result) => {
                    vectors_generated += 1;
                    Some(result)
                },
                Err(e) => {
                    warn!("Dense vector generation failed: {}", e);
                    None
                }
            }
        } else {
            None
        };
        
        let sparse_result = if self.config.enable_sparse_vectors && self.sparse_generator.is_some() {
            let generator = self.sparse_generator.as_ref().unwrap();
            match generator.generate_sparse_vector(content, Some(content_id)).await {
                Ok(result) => {
                    vectors_generated += 1;
                    Some(result)
                },
                Err(e) => {
                    warn!("Sparse vector generation failed: {}", e);
                    None
                }
            }
        } else {
            None
        };
        
        let token_result = if self.config.enable_token_level && self.token_processor.is_some() {
            let processor = self.token_processor.as_ref().unwrap();
            match processor.process_token_vectors(content, Some(content_id)).await {
                Ok(result) => {
                    vectors_generated += 1;
                    Some(result)
                },
                Err(e) => {
                    warn!("Token-level processing failed: {}", e);
                    None
                }
            }
        } else {
            None
        };
        
        let total_generation_time = generation_start.elapsed();
        
        // Perform quality validation
        let quality_assessment = self.quality_validator.assess_unified_quality(
            &dense_result,
            &sparse_result,
            &token_result,
        ).await?;
        
        // Check consistency
        let consistency_verified = self.verify_vector_consistency(
            &dense_result,
            &sparse_result,
            &token_result,
            content,
        ).await?;
        
        // Create metadata
        let metadata = UnifiedVectorMetadata {
            total_generation_time_ms: total_generation_time.as_millis() as u64,
            vectors_generated,
            cache_hits,
            quality_checks_passed: quality_assessment.overall_quality >= self.config.quality_config.min_dense_quality.min(self.config.quality_config.min_sparse_quality).min(self.config.quality_config.min_token_quality),
            consistency_verified,
            resource_usage: ResourceUsageStats {
                peak_memory_mb: self.estimate_memory_usage(&dense_result, &sparse_result, &token_result),
                cpu_time_ms: total_generation_time.as_millis() as u64,
                disk_io_mb: 0.0, // Placeholder
                network_io_mb: 0.0, // Placeholder
            },
            coordination_flags: CoordinationFlags {
                batch_processed: false,
                parallel_generated: self.config.batch_config.enable_parallel_generation,
                quality_validated: self.config.quality_config.enable_quality_validation,
                cached: false,
                compressed: false,
            },
        };
        
        let unified_result = UnifiedVectorResult {
            content_id,
            content: content.to_string(),
            dense_vector: dense_result,
            sparse_vector: sparse_result,
            token_vectors: token_result,
            metadata,
            quality_assessment,
            generation_timestamp: chrono::Utc::now(),
        };
        
        // Cache the result
        if self.config.cache_config.enable_unified_caching {
            self.cache_unified_result(content, &unified_result).await?;
        }
        
        // Record performance metrics
        self.performance_monitor.record_generation(
            total_generation_time,
            vectors_generated,
            &unified_result.quality_assessment,
        ).await;
        
        // Validate performance targets
        if total_generation_time.as_millis() as u64 > self.config.performance_targets.max_total_generation_time_ms {
            warn!(
                "Unified vector generation time exceeded target: {}ms > {}ms",
                total_generation_time.as_millis(),
                self.config.performance_targets.max_total_generation_time_ms
            );
        }
        
        info!(
            "Unified vectors generated: content_id={}, vectors={}, time={}ms, quality={:.3}",
            content_id,
            vectors_generated,
            total_generation_time.as_millis(),
            unified_result.quality_assessment.overall_quality
        );
        
        Ok(unified_result)
    }
    
    /// Generate unified vectors for batch of content
    #[instrument(skip(self, contents), fields(batch_size = contents.len()))]
    pub async fn generate_batch_unified_vectors(
        &self,
        contents: &[(String, Option<Uuid>)],
    ) -> Result<Vec<UnifiedVectorResult>, MultiVectorCoordinationError> {
        let batch_start = Instant::now();
        
        if contents.is_empty() {
            return Ok(Vec::new());
        }
        
        if contents.len() > self.config.batch_config.max_batch_size {
            return Err(MultiVectorCoordinationError::BatchCoordinationError {
                batch_size: contents.len(),
                reason: format!("Batch size {} exceeds maximum {}", 
                               contents.len(), self.config.batch_config.max_batch_size),
            });
        }
        
        debug!("Processing unified vector batch: {} items", contents.len());
        
        let mut results = Vec::with_capacity(contents.len());
        
        if self.config.batch_config.enable_parallel_generation {
            // Parallel batch processing
            let chunk_size = (contents.len() / self.config.batch_config.max_concurrent_generators).max(1);
            let chunks: Vec<_> = contents.chunks(chunk_size).collect();
            
            let mut chunk_futures = Vec::new();
            
            for chunk in chunks {
                let chunk_data: Vec<(String, Option<Uuid>)> = chunk.iter().map(|(s, u)| ((*s).to_string(), *u)).collect();
                let coordinator = self; // Assuming MultiVectorCoordinator implements Clone
                let chunk_future = async move {
                    let mut chunk_results = Vec::new();
                    for (content, id) in chunk_data {
                        match coordinator.generate_unified_vectors(&content, id).await {
                            Ok(result) => chunk_results.push(result),
                            Err(e) => {
                                warn!("Failed to generate unified vectors for content: {}", e);
                            }
                        }
                    }
                    chunk_results
                };
                chunk_futures.push(chunk_future);
            }
            
            let chunk_results = futures::future::join_all(chunk_futures).await;
            for chunk_result in chunk_results {
                results.extend(chunk_result);
            }
        } else {
            // Sequential batch processing
            for (content, id) in contents {
                match self.generate_unified_vectors(content, *id).await {
                    Ok(result) => results.push(result),
                    Err(e) => {
                        warn!("Failed to generate unified vectors for content: {}", e);
                    }
                }
                
                // Perform quality check at intervals
                if results.len() % self.config.batch_config.quality_check_interval == 0 {
                    self.perform_batch_quality_check(&results).await?;
                }
            }
        }
        
        let batch_duration = batch_start.elapsed();
        
        info!(
            "Batch unified vector generation completed: {} results in {}ms ({:.1} vectors/sec)",
            results.len(),
            batch_duration.as_millis(),
            results.len() as f64 / batch_duration.as_secs_f64()
        );
        
        Ok(results)
    }
    
    /// Try to get cached unified result
    async fn try_get_cached_result(&self, content: &str) -> Result<Option<UnifiedVectorResult>, MultiVectorCoordinationError> {
        let cache_key = self.generate_cache_key(content);
        let cache = self.unified_cache.read().await;
        
        if let Some(cached) = cache.get(&cache_key) {
            if cached.created_at.elapsed().as_secs() <= self.config.cache_config.cache_ttl_seconds {
                let mut result = cached.result.clone();
                result.metadata.cache_hits += 1;
                result.metadata.coordination_flags.cached = true;
                
                debug!("Cache hit for unified vectors (content length: {})", content.len());
                return Ok(Some(result));
            }
        }
        
        Ok(None)
    }
    
    /// Cache unified result
    async fn cache_unified_result(&self, content: &str, result: &UnifiedVectorResult) -> Result<(), MultiVectorCoordinationError> {
        let cache_key = self.generate_cache_key(content);
        
        let cached_entry = CachedUnifiedVector {
            result: result.clone(),
            created_at: Instant::now(),
            access_count: 1,
            compressed: self.config.cache_config.enable_cache_compression,
        };
        
        let mut cache = self.unified_cache.write().await;
        
        // Simple cache size management
        let estimated_size_mb = self.estimate_cache_entry_size(result);
        let current_size_mb = cache.len() as f64 * estimated_size_mb;
        
        if current_size_mb + estimated_size_mb > self.config.cache_config.cache_size_mb as f64 {
            // Remove oldest entries
            let oldest_key = cache.iter()
                .min_by_key(|(_, entry)| entry.created_at)
                .map(|(key, _)| key.clone());
            
            if let Some(key) = oldest_key {
                cache.remove(&key);
            }
        }
        
        cache.insert(cache_key, cached_entry);
        Ok(())
    }
    
    /// Generate cache key for content
    fn generate_cache_key(&self, content: &str) -> String {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        
        let mut hasher = DefaultHasher::new();
        content.hash(&mut hasher);
        format!("unified_{:x}", hasher.finish())
    }
    
    /// Verify consistency across vector types
    async fn verify_vector_consistency(
        &self,
        dense_result: &Option<DenseVectorResult>,
        sparse_result: &Option<SparseVectorResult>,
        token_result: &Option<TokenVectorResult>,
        content: &str,
    ) -> Result<bool, MultiVectorCoordinationError> {
        if !self.config.quality_config.enable_consistency_checks {
            return Ok(true);
        }
        
        let mut consistency_passed = true;
        
        // Check content consistency
        if let Some(dense) = dense_result {
            if dense.content != content {
                consistency_passed = false;
                warn!("Dense vector content mismatch");
            }
        }
        
        if let Some(sparse) = sparse_result {
            if sparse.content != content {
                consistency_passed = false;
                warn!("Sparse vector content mismatch");
            }
        }
        
        if let Some(token) = token_result {
            if token.content != content {
                consistency_passed = false;
                warn!("Token vector content mismatch");
            }
        }
        
        // Additional consistency checks could include:
        // - Vector dimension validation
        // - Semantic coherence between vector types
        // - Quality score correlation
        
        Ok(consistency_passed)
    }
    
    /// Estimate memory usage for vector results
    fn estimate_memory_usage(
        &self,
        dense_result: &Option<DenseVectorResult>,
        sparse_result: &Option<SparseVectorResult>,
        token_result: &Option<TokenVectorResult>,
    ) -> f64 {
        let mut total_mb = 0.0;
        
        if let Some(_dense) = dense_result {
            total_mb += 1024.0 * 4.0 / 1024.0 / 1024.0; // 1024 floats * 4 bytes
        }
        
        if let Some(sparse) = sparse_result {
            let sparse_size = (sparse.sparse_vector.indices.len() + sparse.sparse_vector.values.len()) * 4;
            total_mb += sparse_size as f64 / 1024.0 / 1024.0;
        }
        
        if let Some(token) = token_result {
            let token_size = token.token_count * 1024 * 4; // Estimated token vector size
            total_mb += token_size as f64 / 1024.0 / 1024.0;
        }
        
        total_mb
    }
    
    /// Estimate cache entry size
    fn estimate_cache_entry_size(&self, _result: &UnifiedVectorResult) -> f64 {
        // Rough estimate: 5MB per unified result (conservative)
        5.0
    }
    
    /// Perform batch quality check
    async fn perform_batch_quality_check(&self, results: &[UnifiedVectorResult]) -> Result<(), MultiVectorCoordinationError> {
        if results.is_empty() {
            return Ok(());
        }
        
        let avg_quality = results.iter()
            .map(|r| r.quality_assessment.overall_quality)
            .sum::<f64>() / results.len() as f64;
        
        if avg_quality < self.config.performance_targets.min_quality_score {
            return Err(MultiVectorCoordinationError::QualityValidationError {
                metric: "batch_average_quality".to_string(),
                value: avg_quality,
                threshold: self.config.performance_targets.min_quality_score,
            });
        }
        
        debug!("Batch quality check passed: average quality = {:.3}", avg_quality);
        Ok(())
    }
    
    /// Get coordination statistics
    pub async fn get_coordination_stats(&self) -> CoordinationStatistics {
        let cache = self.unified_cache.read().await;
        
        CoordinationStatistics {
            cached_entries: cache.len(),
            total_cache_accesses: cache.values().map(|entry| entry.access_count).sum(),
            active_generators: [
                self.dense_generator.is_some(),
                self.sparse_generator.is_some(),
                self.token_processor.is_some(),
            ].iter().filter(|&&enabled| enabled).count(),
            estimated_cache_size_mb: cache.len() as f64 * 5.0, // Rough estimate
        }
    }
    
    /// Update configuration
    pub async fn update_config(&mut self, config: MultiVectorCoordinatorConfig) -> Result<(), MultiVectorCoordinationError> {
        info!("Updating multi-vector coordinator configuration");
        
        // Update quality validator if settings changed
        if config.quality_config.min_dense_quality != self.config.quality_config.min_dense_quality ||
           config.quality_config.enable_quality_validation != self.config.quality_config.enable_quality_validation {
            self.quality_validator = QualityValidator::new(config.quality_config.clone());
        }
        
        // Update performance monitor if targets changed
        if config.performance_targets.max_total_generation_time_ms != self.config.performance_targets.max_total_generation_time_ms {
            self.performance_monitor = CoordinatorPerformanceMonitor::new(config.performance_targets.clone());
        }
        
        self.config = config;
        Ok(())
    }
    
    /// Clear all caches
    pub async fn clear_caches(&self) -> usize {
        let mut cache = self.unified_cache.write().await;
        let cleared_count = cache.len();
        cache.clear();
        
        info!("Cleared unified vector cache: {} entries removed", cleared_count);
        cleared_count
    }
}

/// Coordination statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoordinationStatistics {
    pub cached_entries: usize,
    pub total_cache_accesses: u64,
    pub active_generators: usize,
    pub estimated_cache_size_mb: f64,
}

/// Quality validation component
pub struct QualityValidator {
    config: QualityAssuranceConfig,
}

impl QualityValidator {
    pub fn new(config: QualityAssuranceConfig) -> Self {
        Self { config }
    }
    
    pub async fn assess_unified_quality(
        &self,
        dense_result: &Option<DenseVectorResult>,
        sparse_result: &Option<SparseVectorResult>,
        token_result: &Option<TokenVectorResult>,
    ) -> Result<UnifiedQualityAssessment, MultiVectorCoordinationError> {
        let dense_quality = dense_result.as_ref()
            .map(|r| r.quality_metrics.overall_quality);
        
        let sparse_quality = sparse_result.as_ref()
            .map(|r| r.quality_metrics.overall_quality);
        
        let token_quality = token_result.as_ref()
            .map(|r| r.quality_metrics.overall_quality);
        
        // Calculate overall quality as weighted average
        let mut total_quality = 0.0;
        let mut weight_sum = 0.0;
        
        if let Some(dq) = dense_quality {
            total_quality += dq * 0.4; // 40% weight for dense
            weight_sum += 0.4;
        }
        
        if let Some(sq) = sparse_quality {
            total_quality += sq * 0.3; // 30% weight for sparse
            weight_sum += 0.3;
        }
        
        if let Some(tq) = token_quality {
            total_quality += tq * 0.3; // 30% weight for token
            weight_sum += 0.3;
        }
        
        let overall_quality = if weight_sum > 0.0 {
            total_quality / weight_sum
        } else {
            0.0
        };
        
        let consistency_score = 0.95; // Placeholder - would be calculated from actual consistency checks
        let cross_validation_passed = overall_quality >= self.config.min_dense_quality.min(self.config.min_sparse_quality).min(self.config.min_token_quality);
        let quality_improvement_achieved = overall_quality > 0.8; // Placeholder threshold
        
        Ok(UnifiedQualityAssessment {
            overall_quality,
            dense_quality,
            sparse_quality,
            token_quality,
            consistency_score,
            cross_validation_passed,
            quality_improvement_achieved,
        })
    }
}

/// Performance monitoring for coordinator
pub struct CoordinatorPerformanceMonitor {
    targets: CoordinatorPerformanceTargets,
}

impl CoordinatorPerformanceMonitor {
    pub fn new(targets: CoordinatorPerformanceTargets) -> Self {
        Self { targets }
    }
    
    pub async fn record_generation(
        &self,
        _generation_time: Duration,
        _vectors_generated: usize,
        _quality_assessment: &UnifiedQualityAssessment,
    ) {
        // Record performance metrics
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_coordination_config_default() {
        let config = MultiVectorCoordinatorConfig::default();
        
        assert!(config.enable_dense_vectors);
        assert!(config.enable_sparse_vectors);
        assert!(config.enable_token_level);
        assert_eq!(config.batch_config.max_batch_size, 50);
        assert!(config.quality_config.enable_quality_validation);
    }
    
    #[test]
    fn test_resource_limits() {
        let limits = ResourceLimits::default();
        
        assert_eq!(limits.max_concurrent_operations, 20);
        assert_eq!(limits.max_memory_per_operation_mb, 50);
        assert_eq!(limits.max_cpu_cores, 8);
    }
    
    #[test]
    fn test_cache_key_generation() {
        let config = MultiVectorCoordinatorConfig::default();
        let coordinator = MultiVectorCoordinator {
            config,
            dense_generator: None,
            sparse_generator: None,
            token_processor: None,
            unified_cache: Arc::new(RwLock::new(HashMap::new())),
            coordination_semaphore: Arc::new(Semaphore::new(1)),
            quality_validator: QualityValidator::new(QualityAssuranceConfig::default()),
            performance_monitor: CoordinatorPerformanceMonitor::new(CoordinatorPerformanceTargets::default()),
        };
        
        let key1 = coordinator.generate_cache_key("test content");
        let key2 = coordinator.generate_cache_key("test content");
        let key3 = coordinator.generate_cache_key("different content");
        
        assert_eq!(key1, key2);
        assert_ne!(key1, key3);
        assert!(key1.starts_with("unified_"));
    }
}