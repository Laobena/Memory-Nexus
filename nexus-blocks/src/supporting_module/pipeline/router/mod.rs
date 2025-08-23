//! High-performance intelligent router block with <2ms latency
//! 
//! Provides lock-free routing with SIMD pattern matching and fallback mechanisms.

mod intelligent_router;
mod simd_analyzer;
mod fallback;

pub use intelligent_router::{IntelligentRouterBlock, RouterConfig, RoutingDecision, CircuitBreaker};
pub use simd_analyzer::{SimdAnalyzer, PatternMatcher};
pub use fallback::{FallbackRouter, RouterHealth};

use crate::core::{BlockError, BlockResult};
use std::time::Duration;

/// Router performance metrics
#[derive(Debug, Clone, Default)]
pub struct RouterMetrics {
    pub total_requests: u64,
    pub cache_hits: u64,
    pub simd_matches: u64,
    pub fallback_activations: u64,
    pub average_latency_us: u64,
    pub p99_latency_us: u64,
}

/// Router health status
#[derive(Debug, Clone, PartialEq)]
pub enum RouterStatus {
    Healthy,
    Degraded(String),
    Failed(String),
}

impl Default for RouterStatus {
    fn default() -> Self {
        RouterStatus::Healthy
    }
}