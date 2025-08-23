//! Fusion support utilities (NOT blocks - just helpers)
//!
//! These modules provide support functions for the ResilientFusionBlock

pub mod deduplication;
pub mod scoring;
pub mod selection;
pub mod partial_handler;
pub mod quality_tracker;
pub mod fallback_scorer;

#[cfg(test)]
mod tests;
pub use deduplication::MinHashDeduplicator;
pub use scoring::ScoringMatrix;
pub use selection::TopKSelector;
pub use partial_handler::PartialResultHandler;
pub use quality_tracker::QualityTracker;
pub use fallback_scorer::FallbackScorer;

use crate::core::{BlockError, BlockResult};
use std::sync::Arc;

/// Fusion metrics
#[derive(Debug, Clone, Default)]
pub struct FusionMetrics {
    pub total_fusions: u64,
    pub items_processed: u64,
    pub items_deduplicated: u64,
    pub partial_results: u64,
    pub simd_failures: u64,
    pub average_latency_ms: f64,
    pub quality_degradations: u64,
}

/// Engine types for result sources
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EngineType {
    Accuracy,
    Intelligence,
    Learning,
    Mining,
}

/// Item to be fused
#[derive(Debug, Clone)]
pub struct FusionItem {
    pub id: uuid::Uuid,
    pub content: Vec<u8>,
    pub relevance: f32,
    pub freshness: f32,
    pub diversity: f32,
    pub authority: f32,
    pub coherence: f32,
    pub confidence: f32,
    pub source_engine: EngineType,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}