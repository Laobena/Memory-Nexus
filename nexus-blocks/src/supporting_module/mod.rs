//! Supporting Modules for Pipeline Blocks
//! 
//! Helper functions, strategies, validators that support the actual blocks.
//! These are NOT blocks themselves - they're implementation details.

pub mod pipeline;
pub mod storage;
// core module removed - UUID is in blocks/registration

// Pipeline support re-exports
pub use pipeline::{
    fusion::{
        deduplication::MinHashDeduplicator,
        fallback_scorer::FallbackScorer,
        partial_handler::PartialResultHandler,
        quality_tracker::QualityTracker,
        scoring::ScoringMatrix,
        selection::TopKSelector,
    },
    preprocessor::{
        checkpoint_manager::CheckpointManager,
        chunking_strategies::ChunkingStrategy,
        utf8_validator::Utf8Validator,
    },
    router::{
        fallback::FallbackRouter,
        simd_analyzer::SimdPatternMatcher,
    },
    search::{
        engine_health::EngineHealth,
        fallback_strategy::FallbackStrategy,
    },
};

// Storage support re-exports
pub use storage::{
    compression::CompressionStrategy,
    recovery::RecoveryManager,
    validation::DataValidator,
    write_coalescer::WriteCoalescer,
};