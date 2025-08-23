//! Core infrastructure for Memory Nexus Blocks
//! 
//! Provides foundational components including traits, error handling,
//! optimized runtime, SIMD operations, and memory management.

pub mod allocator;
pub mod cache;
pub mod errors;
pub mod metrics;
pub mod retry;
pub mod runtime;
pub mod simd;
pub mod traits;
pub mod types;

// Re-export commonly used types
pub use allocator::{
    initialize_allocator,
    report_allocator_stats,
    ArenaAllocator,
    MemoryPool,
    PooledItem,
    AlignedAllocator,
    MemoryTracker,
    MEMORY_TRACKER,
};

pub use cache::{
    CacheConfig,
    TieredCache,
    CacheStats,
    AdaptiveCache,
};

pub use errors::{
    BlockError,
    BlockResult,
    CircuitBreaker,
    CircuitState,
    ErrorContext,
    RecoveryStrategy,
    RetryPolicy,
    RouterError,
    PreprocessorError,
    SearchError,
    StorageError,
    FusionError,
    FallbackTarget,
    PanicHandler,
};

pub use metrics::{
    MetricsConfig,
    MetricsCollector,
    HistogramStats,
    MetricsSnapshot,
    METRICS,
};

pub use retry::{
    RetryExecutor,
    ConditionalRetry,
    Bulkhead,
    HedgedRequest,
    AdaptiveRetry,
};

pub use runtime::{
    create_optimized_runtime,
    RuntimeMetrics,
    RuntimeStats,
    PrioritySpawner,
    BlockingPool,
    LocalExecutor,
    ShutdownCoordinator,
};

pub use simd::{
    CpuFeatures,
    SimdOps,
    BatchOps,
};

pub use traits::{
    PipelineBlock,
    BlockMetadata,
    BlockCategory,
    BlockConfig,
    BlockInput,
    BlockOutput,
    PipelineContext,
    DeploymentMode,
    Cost,
    HealthStatus,
    BlockMetrics,
    Pipeline,
    BlockFactory,
    CBlockInterface,
    ZeroCopy,
};

pub use types::{
    CacheAligned,
    SimdVector,
    BinaryEmbedding,
    Arena,
    MetricsCounter,
    PooledBuffer,
    VectorBatch,
    CACHE_LINE_SIZE,
    SIMD_ALIGN,
    EMBEDDING_DIM_SMALL,
    EMBEDDING_DIM_MEDIUM,
    EMBEDDING_DIM_LARGE,
};

/// Initialize all core systems
pub async fn initialize() -> Result<(), BlockError> {
    // Initialize allocator
    initialize_allocator();
    
    // Install panic handler
    PanicHandler::install();
    
    // Detect CPU features
    let features = CpuFeatures::detect();
    tracing::info!(
        ?features,
        "CPU features detected"
    );
    
    // Report initial memory stats
    report_allocator_stats();
    
    Ok(())
}

/// Global configuration for blocks
pub struct BlocksConfig {
    pub deployment_mode: DeploymentMode,
    pub max_workers: usize,
    pub max_blocking_threads: usize,
    pub enable_metrics: bool,
    pub enable_tracing: bool,
    pub cache_size_mb: usize,
    pub retry_policy: RetryPolicy,
}

impl Default for BlocksConfig {
    fn default() -> Self {
        Self {
            deployment_mode: DeploymentMode::Hybrid,
            max_workers: num_cpus::get() * 2,
            max_blocking_threads: 512,
            enable_metrics: true,
            enable_tracing: true,
            cache_size_mb: 256,
            retry_policy: RetryPolicy::default(),
        }
    }
}

/// Performance statistics collector
pub struct PerfStats {
    pub allocator_stats: AllocatorStats,
    pub runtime_stats: RuntimeStats,
    pub simd_enabled: bool,
    pub cache_hit_rate: f64,
}

pub struct AllocatorStats {
    pub allocated_mb: usize,
    pub peak_mb: usize,
    pub allocations_per_sec: f64,
}

impl PerfStats {
    pub fn collect() -> Self {
        let runtime_handle = tokio::runtime::Handle::current();
        let runtime_metrics = RuntimeMetrics::new(runtime_handle);
        
        Self {
            allocator_stats: AllocatorStats {
                allocated_mb: MEMORY_TRACKER.current_usage() / 1_048_576,
                peak_mb: MEMORY_TRACKER.peak_usage() / 1_048_576,
                allocations_per_sec: 0.0, // Would need time tracking
            },
            runtime_stats: runtime_metrics.collect(),
            simd_enabled: CpuFeatures::detect().avx2 || CpuFeatures::detect().sse42,
            cache_hit_rate: 0.0, // Would come from cache metrics
        }
    }
    
    pub fn report(&self) {
        tracing::info!(
            allocated_mb = self.allocator_stats.allocated_mb,
            peak_mb = self.allocator_stats.peak_mb,
            workers = self.runtime_stats.workers_count,
            active_tasks = self.runtime_stats.active_tasks_count,
            simd = self.simd_enabled,
            cache_hit_rate = format!("{:.2}%", self.cache_hit_rate * 100.0),
            "Performance statistics"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_initialize() {
        initialize().await.unwrap();
        
        // Check that systems are initialized
        let features = CpuFeatures::detect();
        assert!(features.sse42 || features.avx2 || features.neon);
    }
    
    #[test]
    fn test_perf_stats() {
        let stats = PerfStats::collect();
        stats.report();
        
        // Stats should be collected without panic
        assert!(stats.allocator_stats.allocated_mb >= 0);
    }
}