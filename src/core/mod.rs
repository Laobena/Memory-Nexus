pub mod aligned_alloc;
pub mod binary_embeddings;
pub mod config;
pub mod error;
pub mod hash_utils;
pub mod lock_free_cache;
pub mod simd_ops;
pub mod types;

// Aligned allocation exports
pub use aligned_alloc::{
    AlignedVec, CacheLineAllocator, PageAllocator, SimdAllocator,
    alloc_aligned, dealloc_aligned, realloc_aligned, get_allocation_stats,
};

// Binary embeddings exports
pub use binary_embeddings::{
    BinaryEmbedding, BinaryIndex, BinarySearchResult, MultiBitEmbedding,
    get_binary_stats, reset_binary_stats,
};

// Configuration and error exports
pub use config::Config;
pub use error::{NexusError, Result};

// Hash utilities exports
pub use hash_utils::{
    ahash_string, default_hash, xxhash3_64, xxhash3_128,
    generate_cache_key, generate_pipeline_cache_key, generate_embedding_cache_key,
    should_store_content, dedup_hash,
};

// Lock-free cache exports
pub use lock_free_cache::{
    CacheConfig, CacheEntry, CacheStatistics, LockFreeCache,
    LockFreeMPMCQueue, WorkStealingQueue,
};

// SIMD operations exports
pub use simd_ops::{
    CpuFeatures, SimdOps, get_simd_stats, reset_simd_stats,
};

// Core types exports
pub use types::*;

use once_cell::sync::Lazy;
use std::sync::Arc;
use parking_lot::RwLock;

// Global system state
pub static SYSTEM_STATE: Lazy<Arc<RwLock<SystemState>>> = Lazy::new(|| {
    Arc::new(RwLock::new(SystemState::default()))
});

#[derive(Debug, Clone, Default)]
pub struct SystemState {
    pub initialized: bool,
    pub cpu_features: CpuFeatures,
    pub memory_stats: MemoryStats,
}

#[derive(Debug, Clone, Default)]
pub struct CpuFeatures {
    pub has_avx2: bool,
    pub has_avx512: bool,
    pub has_sse: bool,
    pub has_neon: bool,
    pub core_count: usize,
}

#[derive(Debug, Clone, Default)]
pub struct MemoryStats {
    pub total_memory: usize,
    pub available_memory: usize,
    pub cache_size: usize,
}

impl CpuFeatures {
    pub fn detect() -> Self {
        Self {
            has_avx2: cfg!(has_avx2),
            has_avx512: cfg!(has_avx512),
            has_sse: cfg!(has_sse),
            has_neon: cfg!(has_neon),
            core_count: std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(1),
        }
    }
}

pub async fn initialize_system(config: Config) -> Result<()> {
    let mut state = SYSTEM_STATE.write();
    
    if state.initialized {
        return Ok(());
    }
    
    state.cpu_features = CpuFeatures::detect();
    state.initialized = true;
    
    tracing::info!(
        "System initialized with {} cores, AVX2: {}, AVX512: {}, SSE: {}, NEON: {}",
        state.cpu_features.core_count,
        state.cpu_features.has_avx2,
        state.cpu_features.has_avx512,
        state.cpu_features.has_sse,
        state.cpu_features.has_neon
    );
    
    Ok(())
}