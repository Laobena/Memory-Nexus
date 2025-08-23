//! Parallel search orchestrator with <25ms latency and failure isolation
//!
//! Executes 4 search engines in parallel with graceful degradation.

mod orchestrator;
mod engine_health;
mod fallback_strategy;

pub use orchestrator::{SearchOrchestratorBlock, SearchConfig, SearchResult};
pub use engine_health::{EngineHealth, HealthMonitor, EngineStatus};
pub use fallback_strategy::{FallbackStrategy, EmergencyFallback};

use crate::core::{BlockError, BlockResult};
use std::time::Duration;

/// Search engine types
#[derive(Debug, Clone, PartialEq)]
pub enum SearchEngine {
    Accuracy,
    Intelligence,
    Learning,
    Mining,
}

/// Search metrics
#[derive(Debug, Clone, Default)]
pub struct SearchMetrics {
    pub total_searches: u64,
    pub successful_searches: u64,
    pub partial_results: u64,
    pub fallback_activations: u64,
    pub average_latency_ms: u64,
    pub engines_healthy: usize,
}