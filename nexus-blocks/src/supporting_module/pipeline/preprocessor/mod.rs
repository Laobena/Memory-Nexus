//! Zero-copy preprocessor block with <10ms processing and error recovery
//!
//! Provides text chunking, UTF-8 validation, and checkpointing for large documents.

mod zero_copy_preprocessor;
mod chunking_strategies;
mod checkpoint_manager;
mod utf8_validator;

pub use zero_copy_preprocessor::{PreprocessorBlock, PreprocessorConfig, ProcessedChunk};
pub use chunking_strategies::{ChunkingStrategy, ChunkConfig, ChunkBoundary};
pub use checkpoint_manager::{CheckpointManager, Checkpoint, RecoveryState};
pub use utf8_validator::{Utf8Validator, ValidationResult};

use crate::core::{BlockError, BlockResult};
use std::time::Duration;

/// Preprocessing metrics
#[derive(Debug, Clone, Default)]
pub struct PreprocessorMetrics {
    pub documents_processed: u64,
    pub chunks_created: u64,
    pub checkpoints_saved: u64,
    pub recoveries_performed: u64,
    pub utf8_errors_fixed: u64,
    pub average_latency_ms: u64,
    pub bytes_processed: u64,
}