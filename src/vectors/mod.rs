//! Vector Generation and Management Module
//!
//! Comprehensive vector processing system supporting multiple vector types for
//! advanced multi-vector search capabilities. Integrates seamlessly with Memory Nexus's
//! existing search infrastructure while adding support for:
//!
//! - **Dense Vectors**: mxbai-embed-large 1024D semantic embeddings
//! - **Sparse Vectors**: BM25+ keyword-based sparse representations
//! - **Token-level Vectors**: ColBERT-style fine-grained token matching
//! - **Multi-Vector Coordination**: Unified management of all vector types
//!
//! This module provides enterprise-grade vector processing with performance targets
//! of <150ms total generation time, 98.2% accuracy, and support for 1,200+ concurrent users.

pub mod dense_vector_generator;
pub mod sparse_vector_generator;
pub mod token_level_matching;
pub mod multi_vector_coordinator;

// Re-export core vector generation components
pub use dense_vector_generator::{
    DenseVectorGenerator,
    DenseVectorConfig,
    DenseVectorResult,
    DenseVectorError,
    DenseVectorMetadata,
    VectorQualityMetrics,
    NormalizationMethod,
    CacheStatistics,
    BatchProcessingConfig,
    VectorCacheConfig,
    DenseVectorPerformanceTargets,
    AdvancedFeatureConfig,
    ProcessingFlags,
};

pub use sparse_vector_generator::{
    SparseVectorGenerator,
    SparseVectorConfig,
    SparseVectorResult,
    SparseVectorError,
    SparseVectorMetadata,
    SparseVector,
    SparseQualityMetrics,
    BM25Parameters,
    SparseVectorPerformanceTargets,
    VocabularyManager,
    VocabularyStatistics,
};

pub use token_level_matching::{
    TokenLevelProcessor,
    TokenLevelConfig,
    TokenVectorResult,
    TokenLevelError,
    TokenLevelMetadata,
    TokenLevelQualityMetrics,
    TokenSimilarityMatrix,
    AdvancedTokenizer,
    TokenLevelPerformanceTargets,
    Token,
    SimilarityConfig,
    LateInteractionConfig,
    AggregationStrategy,
    InteractionMethod,
};

pub use multi_vector_coordinator::{
    MultiVectorCoordinator,
    MultiVectorCoordinatorConfig,
    UnifiedVectorResult,
    MultiVectorCoordinationError,
    UnifiedVectorMetadata,
    UnifiedQualityAssessment,
    CoordinationStatistics,
    BatchCoordinationConfig,
    QualityAssuranceConfig,
    UnifiedCacheConfig,
    CoordinatorPerformanceTargets,
    ResourceLimits,
    ResourceUsageStats,
    CoordinationFlags,
};

/// Vector processing capabilities
pub struct VectorCapabilities {
    pub supports_dense_vectors: bool,
    pub supports_sparse_vectors: bool,
    pub supports_token_level: bool,
    pub max_vector_dimensions: usize,
    pub max_concurrent_operations: usize,
    pub cache_enabled: bool,
}

impl Default for VectorCapabilities {
    fn default() -> Self {
        Self {
            supports_dense_vectors: true,
            supports_sparse_vectors: true,
            supports_token_level: true,
            max_vector_dimensions: 1024, // mxbai-embed-large dimensions
            max_concurrent_operations: 20,
            cache_enabled: true,
        }
    }
}

/// Create unified vector processing configuration
pub fn create_unified_vector_config() -> MultiVectorCoordinatorConfig {
    MultiVectorCoordinatorConfig {
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

/// Create optimized vector processing configuration for high-performance scenarios
pub fn create_performance_optimized_config() -> MultiVectorCoordinatorConfig {
    MultiVectorCoordinatorConfig {
        enable_dense_vectors: true,
        enable_sparse_vectors: true,
        enable_token_level: true,
        batch_config: BatchCoordinationConfig {
            max_batch_size: 100, // Larger batches for throughput
            enable_parallel_generation: true,
            max_concurrent_generators: 12, // More parallelism
            batch_timeout_ms: 45000, // Longer timeout for complex processing
            enable_adaptive_batching: true,
            quality_check_interval: 20, // Less frequent checks for speed
        },
        quality_config: QualityAssuranceConfig {
            enable_consistency_checks: true,
            enable_quality_validation: true,
            min_dense_quality: 0.85, // Slightly higher quality requirements
            min_sparse_quality: 0.8,
            min_token_quality: 0.75,
            enable_cross_validation: true,
            quality_improvement_threshold: 0.03, // More aggressive improvement
        },
        cache_config: UnifiedCacheConfig {
            enable_unified_caching: true,
            cache_size_mb: 500, // Larger cache for better performance
            cache_ttl_seconds: 3600, // 1 hour cache
            enable_cache_compression: true,
            cache_consistency_checks: true,
        },
        performance_targets: CoordinatorPerformanceTargets {
            max_total_generation_time_ms: 120, // Tighter time constraints
            min_throughput_vectors_per_sec: 25.0, // Higher throughput requirement
            max_memory_usage_mb: 400, // More memory allowance
            min_quality_score: 0.9, // Higher quality target
            max_error_rate: 0.005, // Lower error tolerance
        },
        resource_limits: ResourceLimits {
            max_concurrent_operations: 30, // More concurrent operations
            max_memory_per_operation_mb: 60, // More memory per operation
            max_cpu_cores: 12, // More CPU utilization
            max_disk_usage_mb: 1500, // More disk allowance
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vector_capabilities_default() {
        let capabilities = VectorCapabilities::default();
        
        assert!(capabilities.supports_dense_vectors);
        assert!(capabilities.supports_sparse_vectors);
        assert!(capabilities.supports_token_level);
        assert_eq!(capabilities.max_vector_dimensions, 1024);
        assert!(capabilities.cache_enabled);
    }

    #[test]
    fn test_unified_config_creation() {
        let config = create_unified_vector_config();
        
        assert!(config.enable_dense_vectors);
        assert!(config.enable_sparse_vectors);
        assert!(config.enable_token_level);
        assert_eq!(config.batch_config.max_batch_size, 50);
        assert!(config.quality_config.enable_quality_validation);
    }

    #[test]
    fn test_performance_optimized_config() {
        let config = create_performance_optimized_config();
        
        assert_eq!(config.batch_config.max_batch_size, 100);
        assert_eq!(config.batch_config.max_concurrent_generators, 12);
        assert_eq!(config.cache_config.cache_size_mb, 500);
        assert_eq!(config.performance_targets.min_throughput_vectors_per_sec, 25.0);
        assert_eq!(config.resource_limits.max_concurrent_operations, 30);
    }
}